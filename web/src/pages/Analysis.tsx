import { useEffect, useState } from "react";
import {
  ArrowLeft, CheckCircle2, Clock3, Download, MapPin, Radio, RotateCcw,
} from "lucide-react";
import { useNavigate, useParams } from "react-router-dom";

import { CandidateList } from "../components/evidence/CandidateList";
import { EvidencePanel } from "../components/evidence/EvidencePanel";
import { SonarViewer } from "../components/sonar/SonarViewer";
import { adaptScan } from "../lib/adapt";
import { getScan, reviewCandidate } from "../lib/api";
import type { AnalysisResult, BoundingBox } from "../types/saad";

export function Analysis() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [correctionMode, setCorrectionMode] = useState(false);
  const [correctionBox, setCorrectionBox] = useState<BoundingBox | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    let active = true;
    getScan(id)
      .then((detail) => {
        if (!active) return;
        const scan = adaptScan(detail);
        setResult(scan);
        setSelectedId(scan.candidates[0]?.id ?? null);
        setError(null);
      })
      .catch((cause: unknown) => {
        if (active) setError(cause instanceof Error ? cause.message : "Unable to load scan.");
      })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [id]);

  const candidates = result?.candidates ?? [];
  const selectedCandidate = candidates.find((candidate) => candidate.id === selectedId) ?? candidates[0];

  function selectCandidate(candidateId: string) {
    if (!correctionMode) setSelectedId(candidateId);
  }

  function startCorrection() {
    if (!selectedCandidate || saving) return;
    setCorrectionBox({ ...selectedCandidate.bbox });
    setCorrectionMode(true);
  }

  function cancelCorrection() {
    setCorrectionMode(false);
    setCorrectionBox(null);
  }

  async function persistReview(action: "ACCEPT" | "REJECT" | "CORRECT", bbox?: BoundingBox) {
    if (!result || !selectedCandidate || saving) return;
    setSaving(true);
    setError(null);
    try {
      await reviewCandidate(result.id, selectedCandidate.id, action, bbox);
      const updated = adaptScan(await getScan(result.id));
      setResult(updated);
      setCorrectionMode(false);
      setCorrectionBox(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not save review.");
    } finally {
      setSaving(false);
    }
  }

  function saveCorrection() {
    if (correctionBox) void persistReview("CORRECT", correctionBox);
  }
  function acceptCandidate() { void persistReview("ACCEPT"); }
  function rejectCandidate() { void persistReview("REJECT"); }

  if (loading) return <div role="status" className="p-8 text-sm text-muted-foreground">Loading saved scan…</div>;
  if (!result) return (
    <div className="space-y-4 p-8">
      <p className="text-sm text-red-300">{error ?? "Scan unavailable."}</p>
      <button type="button" className="text-xs text-cyan-300" onClick={() => navigate("/scan")}>Start new scan</button>
    </div>
  );

  const reviewedCount = candidates.filter((candidate) => candidate.reviewStatus !== "pending").length;

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

              {result.processingDevice} inference
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
            onClick={() => navigate(`/reports?scanId=${encodeURIComponent(result.id)}`)}
            className="flex items-center gap-2 rounded-lg border border-cyan-400/20 bg-cyan-400/5 px-3 py-2 text-xs text-cyan-300 transition-colors hover:bg-cyan-400/10"
          >
            <Download className="h-3.5 w-3.5" />
            Generate Report
          </button>

        </div>

      </div>

      {result.analysisStatus === "partial" && (
        <p role="status" className="rounded-xl border border-amber-400/30 bg-amber-400/5 p-3 text-xs text-amber-200">
          Partial inference: one or more VAE, flow or TTA scores are unavailable. Review the missing evidence before acting.
        </p>
      )}
      {error && <p role="alert" className="rounded-xl border border-red-400/30 p-3 text-xs text-red-300">{error}</p>}
      {saving && <p role="status" className="text-xs text-muted-foreground">Saving review…</p>}


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
            disabled={saving}
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
