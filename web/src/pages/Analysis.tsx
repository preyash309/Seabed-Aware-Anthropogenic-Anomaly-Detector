import { useState } from "react";
import {
  ArrowLeft,
  CheckCircle2,
  Clock3,
  Download,
  MapPin,
  Radio,
  RotateCcw,
} from "lucide-react";
import { useLocation, useNavigate } from "react-router-dom";

import { CandidateList } from "../components/evidence/CandidateList";
import { EvidencePanel } from "../components/evidence/EvidencePanel";
import { SonarViewer } from "../components/sonar/SonarViewer";

import type {
  AnalysisResult,
  BoundingBox,
  Candidate,
} from "../types/saad";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

function resolveImageUrl(url: string): string {
  if (!url) return "";

  if (url.startsWith("http://") || url.startsWith("https://")) {
    return url;
  }

  return `${API_BASE_URL}${url.startsWith("/") ? url : `/${url}`}`;
}

/* ============================================================
   Backend → Frontend adapter
   ============================================================ */

function adaptBackendResult(
  backendResult: any,
): AnalysisResult {

  const backendCandidates =
    Array.isArray(backendResult?.candidates)
      ? backendResult.candidates
      : [];

  const candidates: Candidate[] =
    backendCandidates.map(
      (candidate: any, index: number) => {

        return {
          id:
            candidate.id ??
            `candidate-${String(index + 1).padStart(2, "0")}`,

          bbox: {
            x: Number(candidate.bbox?.x ?? 0),
            y: Number(candidate.bbox?.y ?? 0),
            width: Number(candidate.bbox?.width ?? 0),
            height: Number(candidate.bbox?.height ?? 0),
          },

          yoloConfidence:
            Number(candidate.yoloConfidence ?? 0),

          vaeScore:
            candidate.vaeScore === null ||
            candidate.vaeScore === undefined
              ? null
              : Number(candidate.vaeScore),

          flowScore:
            candidate.flowScore === null ||
            candidate.flowScore === undefined
              ? null
              : Number(candidate.flowScore),

          ttaConsistency:
            candidate.ttaConsistency === null ||
            candidate.ttaConsistency === undefined
              ? null
              : Number(candidate.ttaConsistency),

          priority:
            candidate.priority === null ||
            candidate.priority === undefined
              ? null
              : Number(candidate.priority),

          uncertainty:
            candidate.uncertainty === null ||
            candidate.uncertainty === undefined
              ? null
              : Number(candidate.uncertainty),

          priorityLevel:
            candidate.priorityLevel ?? "LOW",

          evidenceProfile:
            candidate.evidenceProfile ??
            "DETECTOR_ONLY",

          recommendation:
            candidate.recommendation ??
            "Candidate requires human review.",

          /*
           * Backend currently returns "PENDING".
           * Frontend Candidate type uses lowercase statuses.
           */
          reviewStatus:
            String(
              candidate.reviewStatus ?? "pending",
            ).toLowerCase() as Candidate["reviewStatus"],
        };
      },
    );

  const backendSummary =
    backendResult?.summary ?? {};

  return {
    id:
      backendResult?.surveyId ??
      "SAAD-UNKNOWN",

    image: {
      url: resolveImageUrl(
          backendResult?.image?.url ?? "",
        ),

      width:
        Number(
          backendResult?.image?.width ?? 0,
        ),

      height:
        Number(
          backendResult?.image?.height ?? 0,
        ),

      filename:
        backendResult?.filename ??
        backendResult?.image?.filename ??
        "sonar_frame",
    },

    summary: {
      detections:
        candidates.length,

      highPriority:
        Number(
          backendSummary.highPriority ?? 0,
        ),

      review:
        Number(
          backendSummary.review ?? 0,
        ),

      /*
       * Stage 1 has no uncertainty model yet.
       * This will be populated when TTA / Evidence Engine
       * is connected.
       */
      uncertain:
        Number(
          backendSummary.uncertain ??
            candidates.filter(
              (candidate) =>
                Number(candidate.uncertainty ?? 0) >= 0.5,
            ).length,
        ),

      lowPriority:
        Number(
          backendSummary.lowPriority ?? 0,
        ),
    },

    candidates,
  };
}


/* ============================================================
   Analysis page
   ============================================================ */

export function Analysis() {

  const location = useLocation();
  const navigate = useNavigate();

  const state = location.state as
    | {
        analysisResult?: any;
      }
    | undefined;


  /*
   * The result now comes directly from FastAPI.
   */
  const result = state?.analysisResult
    ? adaptBackendResult(
        state.analysisResult,
      )
    : null;


  const [selectedId, setSelectedId] =
    useState<string | null>(
      result?.candidates?.[0]?.id ?? null,
    );


  const [candidates, setCandidates] =
    useState<Candidate[]>(
      result?.candidates ?? [],
    );


  const [correctionMode, setCorrectionMode] =
    useState(false);


  const [correctionBox, setCorrectionBox] =
    useState<BoundingBox | null>(null);


  /* ----------------------------------------------------------
     No result
     ---------------------------------------------------------- */

  if (!result) {
    return (
      <div className="flex min-h-[70vh] items-center justify-center">
        <div className="text-center">

          <p className="text-sm font-medium">
            No analysis result found
          </p>

          <p className="mt-2 text-xs text-muted-foreground">
            Start a new sonar scan to view analysis results.
          </p>

          <button
            type="button"
            onClick={() => navigate("/scan")}
            className="mt-5 rounded-lg bg-cyan-400 px-4 py-2 text-xs font-semibold text-slate-950"
          >
            Start new scan
          </button>

        </div>
      </div>
    );
  }


  /* ----------------------------------------------------------
     Selected candidate
     ---------------------------------------------------------- */

  const selectedCandidate =
    candidates.find(
      (candidate) =>
        candidate.id === selectedId,
    ) ?? candidates[0];


  /* ----------------------------------------------------------
     Candidate selection
     ---------------------------------------------------------- */

  function selectCandidate(id: string) {

    if (correctionMode) return;

    setSelectedId(id);
  }


  /* ----------------------------------------------------------
     Correction
     ---------------------------------------------------------- */

  function startCorrection() {

    if (!selectedCandidate) return;

    setCorrectionBox({
      ...selectedCandidate.bbox,
    });

    setCorrectionMode(true);
  }


  function cancelCorrection() {

    setCorrectionMode(false);
    setCorrectionBox(null);
  }


  function saveCorrection() {

    if (
      !selectedCandidate ||
      !correctionBox
    ) {
      return;
    }


    setCandidates(
      (current) =>
        current.map(
          (candidate) =>
            candidate.id ===
            selectedCandidate.id
              ? {
                  ...candidate,

                  bbox: {
                    ...correctionBox,
                  },

                  reviewStatus:
                    "corrected",
                }
              : candidate,
        ),
    );


    setCorrectionMode(false);
    setCorrectionBox(null);
  }


  /* ----------------------------------------------------------
     Accept / Reject
     ---------------------------------------------------------- */

  function updateReviewStatus(
    status: Candidate["reviewStatus"],
  ) {

    if (!selectedCandidate) return;


    setCandidates(
      (current) =>
        current.map(
          (candidate) =>
            candidate.id ===
            selectedCandidate.id
              ? {
                  ...candidate,
                  reviewStatus: status,
                }
              : candidate,
        ),
    );
  }


  function acceptCandidate() {

    updateReviewStatus(
      "accepted",
    );
  }


  function rejectCandidate() {

    updateReviewStatus(
      "rejected",
    );
  }


  /* ----------------------------------------------------------
     Review count
     ---------------------------------------------------------- */

  const reviewedCount =
    candidates.filter(
      (candidate) =>
        candidate.reviewStatus !==
        "pending",
    ).length;


  /* ----------------------------------------------------------
     Render
     ---------------------------------------------------------- */

  return (
    <div className="space-y-6">

      {/* ======================================================
          Header
          ====================================================== */}

      <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">

        <div>

          <button
            type="button"
            onClick={() => navigate("/scan")}
            className="mb-4 flex items-center gap-2 text-xs text-muted-foreground transition-colors hover:text-foreground"
          >
            <ArrowLeft className="h-3.5 w-3.5" />

            Back to new scan
          </button>


          <div className="flex items-center gap-2">

            <Radio className="h-4 w-4 text-cyan-400" />

            <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-400/80">
              Analysis complete
            </span>

          </div>


          <h1 className="mt-2 text-2xl font-semibold tracking-tight">
            Survey analysis
          </h1>


          <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">

            <span className="font-mono">
              {result.id}
            </span>

            <span className="text-border">
              •
            </span>

            <span>
              {result.image.filename}
            </span>

            <span className="text-border">
              •
            </span>

            <span className="flex items-center gap-1">
              <Clock3 className="h-3 w-3" />

              GPU inference
            </span>

          </div>

        </div>


        <div className="flex flex-wrap gap-2">

          <div className="flex items-center gap-2 rounded-lg border border-emerald-400/15 bg-emerald-400/5 px-3 py-2">

            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />

            <span className="font-mono text-[10px] text-emerald-300">
              {reviewedCount}/
              {candidates.length}
              {" "}
              REVIEWED
            </span>

          </div>


          <button
            type="button"
            onClick={() => {
              const reportData = {
                ...result,
                candidates,
                generatedAt: new Date().toISOString(),
              };

              localStorage.setItem(
                "saad-latest-report",
                JSON.stringify(reportData),
              );

              navigate("/reports", {
                state: {
                  reportData,
                },
              });
            }}
            className="flex items-center gap-2 rounded-lg border border-cyan-400/20 bg-cyan-400/5 px-3 py-2 text-xs text-cyan-300 transition-colors hover:bg-cyan-400/10"
          >
            <Download className="h-3.5 w-3.5" />
            Generate Report
          </button>

        </div>

      </div>


      {/* ======================================================
          Summary
          ====================================================== */}

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">

        <SummaryCard
          label="Detections"
          value={result.summary.detections.toString()}
        />

        <SummaryCard
          label="High priority"
          value={result.summary.highPriority.toString()}
          emphasis="high"
        />

        <SummaryCard
          label="Review"
          value={result.summary.review.toString()}
          emphasis="review"
        />

        <SummaryCard
          label="Uncertain"
          value={result.summary.uncertain.toString()}
          emphasis="uncertain"
        />

        <SummaryCard
          label="Low priority"
          value={result.summary.lowPriority.toString()}
        />

      </div>


      {/* ======================================================
          Main viewer
          ====================================================== */}

      <div className="grid min-h-[620px] gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">

        <SonarViewer
          imageUrl={result.image.url}
          candidates={candidates}
          selectedCandidateId={selectedId}
          onCandidateSelect={selectCandidate}
          correctionMode={correctionMode}
          correctionBox={correctionBox}
          onCorrectionChange={setCorrectionBox}
          onSaveCorrection={saveCorrection}
          onCancelCorrection={cancelCorrection}
        />


        <CandidateList
          candidates={candidates}
          selectedCandidateId={selectedId}
          onSelect={selectCandidate}
        />

      </div>


      {/* ======================================================
          Evidence + location
          ====================================================== */}

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_420px]">

        {selectedCandidate ? (

          <EvidencePanel
            candidate={selectedCandidate}
            onAccept={acceptCandidate}
            onReject={rejectCandidate}
            onCorrect={startCorrection}
          />

        ) : (

          <div className="rounded-2xl border border-border/70 bg-card/30 p-5">

            <p className="text-sm font-medium">
              No candidates detected
            </p>

            <p className="mt-1 text-xs text-muted-foreground">
              The detector did not return any candidate anomalies
              for this sonar frame.
            </p>

          </div>

        )}


        <div className="rounded-2xl border border-border/70 bg-card/30 p-5">

          <div className="flex items-center gap-2">

            <MapPin className="h-4 w-4 text-cyan-400" />

            <p className="text-sm font-medium">
              Survey location
            </p>

          </div>


          <p className="mt-1 text-xs text-muted-foreground">
            Geolocation metadata will be connected here.
          </p>


          <div className="mt-5 flex min-h-[220px] items-center justify-center rounded-xl border border-border/50 bg-background/40">

            <div className="text-center">

              <MapPin className="mx-auto h-7 w-7 text-muted-foreground/40" />

              <p className="mt-3 font-mono text-[10px] text-muted-foreground">
                GEOLOCATION UNAVAILABLE
              </p>

              <p className="mt-1 text-[10px] text-muted-foreground/60">
                Awaiting navigation metadata
              </p>

            </div>

          </div>


          <div className="mt-4 grid grid-cols-2 gap-3">

            <Telemetry
              label="Latitude"
              value="—"
            />

            <Telemetry
              label="Longitude"
              value="—"
            />

          </div>

        </div>

      </div>


      {/* ======================================================
          Footer
          ====================================================== */}

      <div className="flex flex-col gap-3 rounded-xl border border-border/60 bg-card/20 p-4 text-xs sm:flex-row sm:items-center sm:justify-between">

        <div className="flex items-center gap-2 text-muted-foreground">

          <RotateCcw className="h-3.5 w-3.5" />

          <span>
            Human decisions are recorded for future model improvement.
          </span>

        </div>


        <span className="font-mono text-[9px] text-muted-foreground/50">
          SAAD / YOLO26s DETECTOR
        </span>

      </div>

    </div>
  );
}


/* ============================================================
   Summary card
   ============================================================ */

function SummaryCard({
  label,
  value,
  emphasis,
}: {
  label: string;
  value: string;
  emphasis?: "high" | "review" | "uncertain";
}) {

  const valueClass =
    emphasis === "high"
      ? "text-red-300"
      : emphasis === "review"
        ? "text-cyan-300"
        : emphasis === "uncertain"
          ? "text-amber-300"
          : "text-foreground";


  return (
    <div className="rounded-xl border border-border/60 bg-card/30 p-4">

      <p className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
        {label}
      </p>

      <p
        className={`mt-2 text-2xl font-semibold ${valueClass}`}
      >
        {value}
      </p>

    </div>
  );
}


/* ============================================================
   Telemetry
   ============================================================ */

function Telemetry({
  label,
  value,
}: {
  label: string;
  value: string;
}) {

  return (
    <div className="rounded-lg border border-border/50 bg-background/30 p-3">

      <p className="font-mono text-[8px] uppercase tracking-wider text-muted-foreground/60">
        {label}
      </p>

      <p className="mt-1 font-mono text-xs">
        {value}
      </p>

    </div>
  );
}