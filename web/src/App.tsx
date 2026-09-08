import { useCallback, useEffect, useState } from "react";
import type { Project } from "./types";
import { api } from "./api";
import Landing from "./screens/Landing";
import Processing from "./screens/Processing";
import Viewer from "./screens/Viewer";

type Screen = "landing" | "processing" | "viewer";

export default function App() {
  const [screen, setScreen] = useState<Screen>("landing");
  const [project, setProject] = useState<Project | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  /** Open a project. If a job is in flight, show the processing screen first. */
  const openProject = useCallback((next: Project, job: string | null) => {
    setProject(next);
    setJobId(job);
    setScreen(job ? "processing" : "viewer");
  }, []);

  const goLanding = useCallback(() => {
    // Leaving the viewer means the user finished inspecting this processed
    // scan: release it so the server frees its disk after a short grace
    // period (unless they reopen it before then).
    setScreen((prev) => {
      if (
        (prev === "viewer" || prev === "processing") &&
        project?.processed
      ) {
        api.releaseProject(project.id);
      }
      return "landing";
    });
    setProject(null);
    setJobId(null);
  }, [project]);

  const handleProcessingDone = useCallback((done: Project) => {
    setProject(done);
    setJobId(null);
    setScreen("viewer");
  }, []);

  // Close/switch away from the tab while viewing a processed project: fire
  // the release beacon so the scan's disk footprint is reclaimed.
  useEffect(() => {
    if (screen !== "viewer" || !project?.processed) return;
    const release = () => api.releaseProject(project.id);
    const onVisibility = () => {
      if (document.visibilityState === "hidden") release();
    };
    window.addEventListener("pagehide", release);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.removeEventListener("pagehide", release);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [screen, project]);

  if (screen === "processing" && project && jobId) {
    return (
      <Processing
        project={project}
        jobId={jobId}
        onDone={handleProcessingDone}
        onBack={goLanding}
      />
    );
  }

  if (screen === "viewer" && project) {
    return <Viewer project={project} onBack={goLanding} />;
  }

  return <Landing onOpen={openProject} />;
}