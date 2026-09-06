import type { JobStatus, Project, ViewerData } from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
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

export const api = {
  health: () => request<{ status: string; version: string }>("/api/health"),

  listProjects: () => request<Project[]>("/api/projects"),

  createProject: (name: string) =>
    request<Project>(`/api/projects?name=${encodeURIComponent(name)}`, {
      method: "POST",
    }),

  uploadLas: (projectId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Project>(`/api/projects/${projectId}/upload`, {
      method: "POST",
      body: form,
    });
  },

  process: (
    projectId: string,
    extra?: { tile_size_m?: number; viewer_point_limit?: number }
  ) =>
    request<{ job_id: string }>(`/api/projects/${projectId}/process`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(extra ?? {}),
    }),

  job: (jobId: string) => request<JobStatus>(`/api/jobs/${jobId}`),

  viewerData: (projectId: string) =>
    request<ViewerData>(`/api/projects/${projectId}/viewer-data`),

  simulate: () => request<Project>("/api/simulate", { method: "POST" }),

  simulateData: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Project>("/api/simulate/data", { method: "POST", body: form });
  },

  exportUrl: (projectId: string, name: string) =>
    `/api/projects/${projectId}/exports/${name}`,
};
