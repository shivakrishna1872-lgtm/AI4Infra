import { useCallback, useState } from "react";
import type { Project } from "./types";
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
    setScreen("landing");
    setProject(null);
    setJobId(null);
  }, []);

  const handleProcessingDone = useCallback((done: Project) => {
    setProject(done);
    setJobId(null);
    setScreen("viewer");
  }, []);

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