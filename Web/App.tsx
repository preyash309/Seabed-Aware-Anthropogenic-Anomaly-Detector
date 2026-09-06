import { BrowserRouter, Route, Routes } from "react-router-dom";

import { AppShell } from "./components/layout/AppShell";
import { Analysis } from "./pages/Analysis";
import { NewScan } from "./pages/NewScan";
import { Reports } from "./pages/Reports";

function Overview() {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="space-y-3">
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-cyan-400/70">
          Marine Intelligence System
        </p>

        <h1 className="text-3xl font-semibold tracking-tight">
          Mission Overview
        </h1>

        <p className="max-w-2xl text-sm leading-6 text-muted-foreground">
          SAAD is a seabed-aware intelligence platform for detecting,
          evaluating, and prioritizing potential anomalies in side-scan
          sonar imagery.
        </p>
      </div>

      {/* What SAAD does */}
      <div className="rounded-2xl border border-border/70 bg-card/40 p-6">
        <div className="space-y-2">
          <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-cyan-400/70">
            System Purpose
          </p>

          <h2 className="text-lg font-semibold">
            From sonar imagery to actionable evidence
          </h2>

          <p className="max-w-3xl text-sm leading-6 text-muted-foreground">
            SAAD combines multiple analysis stages to identify objects and
            seabed anomalies that may require further investigation. Instead
            of relying on a single detector score, the system combines
            detection confidence, reconstruction behaviour, flow-based
            novelty, and test-time consistency to build an evidence profile
            for each candidate.
          </p>
        </div>
      </div>

      {/* Pipeline */}
      <div className="rounded-2xl border border-border/70 bg-card/40 p-6">
        <div className="space-y-5">
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-cyan-400/70">
              Analysis Pipeline
            </p>

            <h2 className="mt-1 text-lg font-semibold">
              Multi-stage anomaly assessment
            </h2>
          </div>

          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-xl border border-border/50 bg-background/30 p-4">
              <p className="font-mono text-[10px] text-cyan-400">
                01 / DETECT
              </p>

              <p className="mt-2 text-sm font-medium">
                YOLO Detection
              </p>

              <p className="mt-1 text-xs leading-5 text-muted-foreground">
                Locates potential objects and candidate regions within the
                sonar image.
              </p>
            </div>

            <div className="rounded-xl border border-border/50 bg-background/30 p-4">
              <p className="font-mono text-[10px] text-cyan-400">
                02 / RECONSTRUCT
              </p>

              <p className="mt-2 text-sm font-medium">
                VAE Analysis
              </p>

              <p className="mt-1 text-xs leading-5 text-muted-foreground">
                Evaluates how well each candidate conforms to learned
                seabed representations.
              </p>
            </div>

            <div className="rounded-xl border border-border/50 bg-background/30 p-4">
              <p className="font-mono text-[10px] text-cyan-400">
                03 / NOVELTY
              </p>

              <p className="mt-2 text-sm font-medium">
                RealNVP Flow
              </p>

              <p className="mt-1 text-xs leading-5 text-muted-foreground">
                Measures how unusual a candidate is within the learned
                feature distribution.
              </p>
            </div>

            <div className="rounded-xl border border-border/50 bg-background/30 p-4">
              <p className="font-mono text-[10px] text-cyan-400">
                04 / VERIFY
              </p>

              <p className="mt-2 text-sm font-medium">
                TTA Consistency
              </p>

              <p className="mt-1 text-xs leading-5 text-muted-foreground">
                Tests whether the candidate remains stable under image
                transformations.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Evidence + workflow */}
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-2xl border border-border/70 bg-card/40 p-6">
          <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-cyan-400/70">
            Evidence Engine
          </p>

          <h2 className="mt-1 text-lg font-semibold">
            Evidence-driven prioritization
          </h2>

          <p className="mt-3 text-sm leading-6 text-muted-foreground">
            Candidate signals are combined into a priority and uncertainty
            assessment. The resulting evidence profile helps operators focus
            attention on the most relevant sonar anomalies instead of
            reviewing every region equally.
          </p>
        </div>

        <div className="rounded-2xl border border-border/70 bg-card/40 p-6">
          <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-cyan-400/70">
            Operator Workflow
          </p>

          <h2 className="mt-1 text-lg font-semibold">
            Human-in-the-loop review
          </h2>

          <p className="mt-3 text-sm leading-6 text-muted-foreground">
            Analysts can inspect ranked candidates and their supporting
            evidence before making a final decision. Candidates can be
            accepted, rejected, or corrected during the review process.
          </p>
        </div>
      </div>

      {/* CTA */}
      <div className="rounded-2xl border border-cyan-400/10 bg-cyan-400/5 p-5">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-sm font-medium">
              Ready to analyse a survey?
            </p>

            <p className="mt-1 text-xs text-muted-foreground">
              Upload a side-scan sonar image to start the SAAD analysis
              pipeline.
            </p>
          </div>

          <a
            href="/scan"
            className="inline-flex items-center justify-center rounded-lg border border-cyan-400/20 bg-cyan-400/10 px-4 py-2 text-xs font-medium text-cyan-300 transition-colors hover:bg-cyan-400/15"
          >
            Start New Scan →
          </a>
        </div>
      </div>
    </div>
  );
}

function Placeholder({ title }: { title: string }) {
  return (
    <div className="space-y-3">
      <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-cyan-400/70">
        SAAD
      </p>

      <h1 className="text-2xl font-semibold">{title}</h1>

      <p className="text-sm text-muted-foreground">
        This module will be connected in the next development stage.
      </p>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/" element={<Overview />} />

          <Route path="/scan" element={<NewScan />} />

          <Route
            path="/scan/:id"
            element={<Analysis />}
          />

          <Route
            path="/history"
            element={<Placeholder title="Scan History" />}
          />

          <Route
            path="/review"
            element={<Placeholder title="Review Queue" />}
          />

          <Route
            path="/reports"
            element={<Reports />}
          />

          <Route
            path="/analytics"
            element={<Placeholder title="Analytics" />}
          />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;