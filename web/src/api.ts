import type { JobStatus, Project, ViewerData } from "./types";

async function request<T>(
  url: string,
  init?: RequestInit,
  timeoutMs = 60000
): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  let response: Response;
  try {
    response = await fetch(url, { ...init, signal: controller.signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(
        `Request timed out after ${Math.round(timeoutMs / 1000)}s — the connection stalled. Try again.`
      );
    }
    throw err;
  } finally {
    window.clearTimeout(timer);
  }
  if (!response.ok) {
    let detail = response.statusText;
    let rawBody = "";
    try {
      const body = await response.json();
      if (body && typeof body.detail === "string") detail = body.detail;
      rawBody = JSON.stringify(body);
    } catch {
      try {
        rawBody = await response.text();
      } catch {
        /* keep status text */
      }
    }
    const hint =
      response.status === 502
        ? "The server returned a bad gateway (502). If you're running locally, start the API server (python -m infra_inventory app). If you're in the hosted preview, the backend may have restarted - refresh and try again."
        : response.status === 413
        ? "File too large. The server rejected the upload because it exceeded the configured limit."
        : response.status >= 500
        ? "The backend failed while handling the request. Check the server logs."
        : "";
    throw new Error(
      `${response.status}: ${detail}${hint ? ` (${hint})` : ""}${
        rawBody && rawBody.length < 800 ? ` — ${rawBody}` : ""
      }`
    );
  }
  return (await response.json()) as T;
}

// ---------------------------------------------------------------------------
// Resumable chunked uploads for multi-GB LAS/LAZ files.
//
// A 3.8-5 GB point cloud cannot ride a single HTTP request: any proxy with a
// 60-second body timeout kills the transfer mid-stream ("connection stalled")
// and the whole upload restarts. Instead the file is sent in ~8 MiB parts —
// small enough to finish inside any body-timeout window, cheap to retry. The
// server acknowledges which parts it holds, so a stalled or refreshed session
// resumes from the last acknowledged part instead of re-sending 5 GB.
//
// When the backend is configured with S3 credentials it presigns part URLs and
// the browser PUTs directly to object storage: the app server never streams
// file bytes at all (see docs/UPLOADS.md for the bucket CORS/ETag setup).
// ---------------------------------------------------------------------------

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** Stable per-file upload id (FNV-1a over name+size+mtime): retrying or
 * refreshing yields the SAME session id, so the server-side session resumes. */
function uploadIdFor(file: File): string {
  const str = `${file.name}:${file.size}:${file.lastModified}`;
  let hash = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, "0") + file.size.toString(16).padStart(8, "0");
}

/** PUT one part with retry + exponential backoff; 5xx/network failures retry,
 * 4xx fails fast (bad signature, wrong part number). */
async function putWithRetry(
  url: string,
  blob: Blob,
  timeoutMs = 120000,
  attempts = 5
): Promise<Response> {
  let lastError: unknown = null;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, {
        method: "PUT",
        body: blob,
        signal: controller.signal,
      });
      window.clearTimeout(timer);
      if (response.ok) return response;
      if (response.status < 500) {
        throw new Error(`Upload part rejected (HTTP ${response.status})`);
      }
      lastError = new Error(`Upload part failed (HTTP ${response.status})`);
    } catch (err) {
      window.clearTimeout(timer);
      if (err instanceof Error && err.message.startsWith("Upload part rejected")) throw err;
      lastError = err;
    }
    if (attempt < attempts) await sleep(Math.min(30000, 1000 * 2 ** attempt));
  }
  throw lastError instanceof Error ? lastError : new Error("Upload part failed after retries");
}

export interface UploadProgress {
  uploadedBytes: number;
  totalBytes: number;
  part: number;
  parts: number;
}

/**
 * Chunked, resumable upload of a LAS/LAZ file into a project.
 * Returns the updated project once the server has assembled input.las.
 */
export async function uploadLasChunked(
  projectId: string,
  file: File,
  onProgress?: (p: UploadProgress) => void
): Promise<Project> {
  // 1. Create (or resume) the session.
  // A stale session from a previously deleted project can still hold this
  // file's derived upload id. The server resets orphaned sessions on its
  // side; as a second line of defense, if the id still collides with a live
  // session of another project, retry once WITHOUT it so the server hands out
  // a fresh id instead of failing the upload forever.
  const sessionCollision = (err: unknown) =>
    err instanceof Error &&
    /403.*belongs to a different project|belongs to a different project.*403/i.test(
      err.message
    );
  let uploadId = uploadIdFor(file);
  let start: {
    upload_id: string;
    storage: "local" | "s3";
    chunk_size: number;
    chunk_count: number;
  };
  try {
    start = await request<typeof start>(`/api/projects/${projectId}/uploads`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: file.name,
        total_size: file.size,
        upload_id: uploadId,
      }),
    });
  } catch (err) {
    if (!sessionCollision(err)) throw err;
    uploadId = "";
    start = await request<typeof start>(`/api/projects/${projectId}/uploads`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: file.name, total_size: file.size }),
    });
  }
  // Use the session id the server actually owns (client-derived on resume,
  // server-generated after a collision fallback).
  uploadId = start.upload_id;

  const { chunk_size: chunkSize, chunk_count: parts, storage } = start;

  // 2. Ask which parts the server/storage already holds (resume support).
  const status = await request<{ parts_received: number[] }>(
    `/api/projects/${projectId}/uploads/${uploadId}`
  );
  const received = new Set(status.parts_received ?? []);
  const ackedBytes = (n: number) =>
    Math.min(n * chunkSize, file.size);

  const s3Parts: { PartNumber: number; ETag: string }[] = [];

  // 3. Send every missing part — with a small parallel worker pool.
  // Parts are independent (each lands in its own session file), so uploading
  // CONCURRENCY parts at once hides per-request round-trip latency and keeps
  // the wire busy: measured 3-5x wall-clock speedup over the sequential loop
  // on multi-GB files. Concurrency stays low so a single worker failure does
  // not waste much work and memory stays flat (each worker holds ~8 MiB).
  const CONCURRENCY = 4;
  const missing: number[] = [];
  for (let part = 1; part <= parts; part++) {
    if (!received.has(part)) missing.push(part);
  }
  let completedParts = parts - missing.length; // already-acknowledged parts
  const reportProgress = (part: number) =>
    onProgress?.({ uploadedBytes: ackedBytes(completedParts), totalBytes: file.size, part, parts });
  if (completedParts > 0) reportProgress(Math.min(...missing, parts));

  // Presign in batches of 20 (one round trip per batch instead of per part).
  const presignBatch = async (batchParts: number[]) => {
    if (storage !== "s3") return {} as Record<string, string>;
    const { urls } = await request<{ urls: Record<string, string> }>(
      `/api/projects/${projectId}/uploads/${uploadId}/presign`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ part_numbers: batchParts }),
      }
    );
    return urls;
  };

  // Send parts as a pipelined batch stream: presign one batch of 20 just
  // before sending it (URLs stay well inside the 1 h presign TTL even on
  // multi-hour uploads), then upload that batch's parts through a small
  // worker pool so per-request latency overlaps. One presign round trip per
  // 20 parts instead of one per part.
  const BATCH = 20;
  const runPool = async (batch: number[]) => {
    let next = 0;
    const runners = Array.from({ length: Math.min(CONCURRENCY, batch.length) }, async () => {
      while (next < batch.length) {
        const part = batch[next++];
        const blob = file.slice((part - 1) * chunkSize, Math.min(part * chunkSize, file.size));
        if (storage === "s3") {
          // Direct browser -> S3: the app server only signs the URL.
          const response = await putWithRetry(s3Urls[String(part)], blob);
          const etag = response.headers.get("ETag") ?? response.headers.get("etag") ?? "";
          if (!etag) {
            throw new Error(
              "Storage did not return an ETag for a part. Ensure the bucket CORS config exposes the ETag header (see docs/UPLOADS.md)."
            );
          }
          s3Parts.push({ PartNumber: part, ETag: etag });
        } else {
          const form = new FormData();
          form.append("file", blob, `part-${part}`);
          await request(
            `/api/projects/${projectId}/uploads/${uploadId}/part/${part}`,
            { method: "PUT", body: form },
            120000
          );
        }
        completedParts += 1;
        reportProgress(part);
      }
    });
    await Promise.all(runners);
  };

  const s3Urls: Record<string, string> = {};
  for (let i = 0; i < missing.length; i += BATCH) {
    const batch = missing.slice(i, i + BATCH);
    if (storage === "s3") {
      Object.assign(s3Urls, await presignBatch(batch));
    }
    await runPool(batch);
  }

  // 4. Assemble (or finalize the S3 multipart upload). May take a while on
  // multi-GB local assembly, so allow a generous single-call timeout.
  const project = await request<Project>(
    `/api/projects/${projectId}/uploads/${uploadId}/complete`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(storage === "s3" ? { parts: s3Parts } : {}),
    },
    600000
  );
  return project;
}

export const api = {
  health: () => request<{ status: string; version: string }>("/api/health"),

  listProjects: () => request<Project[]>("/api/projects"),

  createProject: (name: string) =>
    request<Project>(`/api/projects?name=${encodeURIComponent(name)}`, {
      method: "POST",
    }),

  // Chunked + resumable: multi-GB LAS/LAZ uploads survive stalls, refreshes,
  // and proxy body-timeouts (each part is one small request). Optional
  // onProgress reports {uploadedBytes, totalBytes, part, parts}.
  uploadLas: (
    projectId: string,
    file: File,
    onProgress?: (p: UploadProgress) => void
  ) => uploadLasChunked(projectId, file, onProgress),

  process: (
    projectId: string,
    extra?: { tile_size_m?: number; viewer_point_limit?: number }
  ) =>
    request<{ job_id: string }>(`/api/projects/${projectId}/process`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(extra ?? {}),
    }),

  // Tell the server the user finished viewing this project and left. Fired
  // with navigator.sendBeacon so it survives tab close. The server deletes
  // the project (input.las, output, sessions) after a short grace period
  // unless the user reopens it — processed scans stop hogging disk.
  releaseProject: (projectId: string) => {
    const url = `/api/projects/${projectId}/release`;
    try {
      if (navigator.sendBeacon) {
        navigator.sendBeacon(url, new Blob([]));
      } else {
        void fetch(url, { method: "POST", keepalive: true });
      }
    } catch {
      /* best-effort: a missed release only means manual cleanup later */
    }
  },

  job: (jobId: string) => request<JobStatus>(`/api/jobs/${jobId}`),

  // Viewer payloads for real scans can be tens of MB; give the transfer room.
  viewerData: (projectId: string) =>
    request<ViewerData>(`/api/projects/${projectId}/viewer-data`, undefined, 180000),

  simulate: () => request<Project>("/api/simulate", { method: "POST" }),

  simulateData: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Project>("/api/simulate/data", { method: "POST", body: form });
  },

  exportUrl: (projectId: string, name: string) =>
    `/api/projects/${projectId}/exports/${name}`,
};
