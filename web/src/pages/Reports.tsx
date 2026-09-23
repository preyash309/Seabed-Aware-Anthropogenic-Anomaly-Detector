import { useEffect, useState } from "react";
import {
  ArrowLeft, CheckCircle2, Download, FileJson, FileSpreadsheet, FileText, ShieldAlert,
} from "lucide-react";
import { useNavigate, useSearchParams } from "react-router-dom";

import { adaptScan } from "../lib/adapt";
import { getReport, getScan, listScans, reportUrl } from "../lib/api";
import type { AnalysisResult, Candidate } from "../types/saad";

type ReportData = AnalysisResult & { generatedAt: string };

export function Reports() {
  const navigate = useNavigate();
  const [search] = useSearchParams();
  const scanId = search.get("scanId");
  const [report, setReport] = useState<ReportData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    if (!scanId) {
      listScans()
        .then((scans) => {
          if (active && scans.length) navigate(`/reports?scanId=${encodeURIComponent(scans[0].scanId)}`, { replace: true });
        })
        .catch((cause: unknown) => { if (active) setError(cause instanceof Error ? cause.message : "Cannot load scans."); })
        .finally(() => { if (active) setLoading(false); });
    } else {
      Promise.all([getScan(scanId), getReport(scanId)])
        .then(([detail, serverReport]) => {
          if (active) setReport({ ...adaptScan(detail), generatedAt: serverReport.generatedAt });
        })
        .catch((cause: unknown) => { if (active) setError(cause instanceof Error ? cause.message : "Cannot load report."); })
        .finally(() => { if (active) setLoading(false); });
    }
    return () => { active = false; };
  }, [scanId, navigate]);

  function exportPDF() {
    if (report) window.location.assign(reportUrl(report.id, "pdf"));
  }

  function exportJSON() {
    if (report) window.location.assign(`${reportUrl(report.id, "json")}&download=true`);
  }

  function exportCSV() {
    if (report) window.location.assign(reportUrl(report.id, "csv"));
  }

  if (loading) return <div role="status" className="p-8 text-sm text-muted-foreground">Loading saved report…</div>;

  if (!report) {
    return (
      <div className="flex min-h-[70vh] items-center justify-center">
        <div className="text-center">
          <FileText className="mx-auto h-8 w-8 text-muted-foreground/50" />

          <p className="mt-4 text-sm font-medium">
            No report available
          </p>

          {error && <p role="alert" className="mt-2 text-xs text-red-300">{error}</p>}
          <p className="mt-2 text-xs text-muted-foreground">
            Select a saved scan or complete a new analysis to generate a report.
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

  const highPriority = report.candidates.filter(
    (candidate) => candidate.priorityLevel === "HIGH",
  );

  const generatedAt = report.generatedAt
    ? new Date(report.generatedAt).toLocaleString()
    : new Date().toLocaleString();

  return (
    <div className="space-y-6">
      {/* ============================================================
          Page Header
          ============================================================ */}

      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between print:hidden">
        <div>
          <button
            type="button"
            onClick={() => navigate(-1)}
            className="mb-4 flex items-center gap-2 text-xs text-muted-foreground transition-colors hover:text-foreground"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Back to analysis
          </button>

          <div className="flex items-center gap-2">
            <FileText className="h-4 w-4 text-cyan-400" />

            <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-400/80">
              Reporting
            </span>
          </div>

          <h1 className="mt-2 text-2xl font-semibold tracking-tight">
            Analysis Report
          </h1>

          <p className="mt-2 text-xs text-muted-foreground">
            Structured and human-readable output from the saved SAAD
            sonar analysis.
          </p>
        </div>

        {/* Export actions */}
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={exportJSON}
            className="flex items-center gap-2 rounded-lg border border-border/70 bg-card/40 px-3 py-2.5 text-xs text-muted-foreground transition-colors hover:border-cyan-400/20 hover:text-cyan-300"
          >
            <FileJson className="h-3.5 w-3.5" />
            JSON
          </button>

          <button
            type="button"
            onClick={exportCSV}
            className="flex items-center gap-2 rounded-lg border border-border/70 bg-card/40 px-3 py-2.5 text-xs text-muted-foreground transition-colors hover:border-cyan-400/20 hover:text-cyan-300"
          >
            <FileSpreadsheet className="h-3.5 w-3.5" />
            CSV
          </button>

          <button
            type="button"
            onClick={exportPDF}
            className="flex items-center gap-2 rounded-lg bg-cyan-400 px-4 py-2.5 text-xs font-semibold text-slate-950 transition-colors hover:bg-cyan-300"
          >
            <Download className="h-3.5 w-3.5" />
            PDF
          </button>
        </div>
      </div>

      {/* ============================================================
          Report document
          ============================================================ */}

      <div className="rounded-2xl border border-border/70 bg-card/40 p-6 sm:p-8 print:border-0 print:bg-white print:p-0 print:text-black">

        {/* Header */}
        <div className="border-b border-border/60 pb-6 print:border-gray-300">
          <div className="flex items-start justify-between gap-6">
            <div>
              <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-cyan-400 print:text-cyan-700">
                SAAD / Marine Intelligence System
              </p>

              <h2 className="mt-2 text-2xl font-semibold">
                Sonar Anomaly Analysis Report
              </h2>

              <p className="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground print:text-gray-600">
                Evidence-driven analysis of potential anomalies detected
                in side-scan sonar imagery.
              </p>
            </div>

            <div className="hidden text-right sm:block">
              <p className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
                Survey
              </p>

              <p className="mt-1 font-mono text-xs">
                {report.id}
              </p>
            </div>
          </div>
        </div>

        {/* Survey information */}
        <section className="mt-6">
          <SectionTitle title="Survey Information" />

          <div className="mt-3 grid gap-3 sm:grid-cols-3">
            <InfoCard
              label="Survey ID"
              value={report.id}
            />

            <InfoCard
              label="Source File"
              value={report.image.filename}
            />

            <InfoCard
              label="Generated"
              value={generatedAt}
            />
          </div>
        </section>

        {/* Executive summary */}
        <section className="mt-8">
          <SectionTitle title="Executive Summary" />

          <div className="mt-3 grid gap-3 sm:grid-cols-4">
            <SummaryBox
              label="Detections"
              value={report.summary.detections}
            />

            <SummaryBox
              label="High Priority"
              value={report.summary.highPriority}
            />

            <SummaryBox
              label="Review"
              value={report.summary.review}
            />

            <SummaryBox
              label="Uncertain"
              value={report.summary.uncertain}
            />
          </div>

          <div className="mt-4 rounded-xl border border-border/50 bg-background/30 p-4 print:border-gray-300">
            <p className="text-sm leading-6 text-muted-foreground print:text-gray-700">
              SAAD identified{" "}
              <strong className="text-foreground print:text-black">
                {report.candidates.length}
              </strong>{" "}
              {report.candidates.length === 1 ? "candidate region" : "candidate regions"} within the
              analysed sonar frame. Candidates were ranked using
              detector confidence, reconstruction behaviour, flow-based
              novelty, and test-time consistency to support
              evidence-driven review.
            </p>
          </div>
        </section>

        {/* Priority findings */}
        <section className="mt-8">
          <SectionTitle title="Priority Findings" />

          <div className="mt-3 space-y-3">
            {highPriority.length > 0 ? (
              highPriority.map((candidate) => (
                <div
                  key={candidate.id}
                  className="rounded-xl border border-red-400/15 bg-red-400/5 p-4 print:border-gray-300 print:bg-gray-50"
                >
                  <div className="flex items-start gap-3">
                    <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-red-400 print:text-red-700" />

                    <div className="min-w-0">
                      <p className="text-sm font-medium">
                        Candidate {candidate.id}
                      </p>

                      <p className="mt-1 text-xs leading-5 text-muted-foreground print:text-gray-600">
                        {candidate.evidenceProfile}
                      </p>

                      <p className="mt-2 text-xs leading-5 text-muted-foreground print:text-gray-700">
                        {candidate.recommendation}
                      </p>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="rounded-xl border border-border/50 bg-background/30 p-4 text-xs text-muted-foreground">
                No high-priority candidates were identified.
              </div>
            )}
          </div>
        </section>

        {/* Candidate findings */}
        <section className="mt-8">
          <SectionTitle title="Candidate Findings" />

          <div className="mt-3 overflow-x-auto rounded-xl border border-border/60 print:border-gray-300">
            <table className="w-full min-w-[900px] text-left">
              <thead>
                <tr className="border-b border-border/60 bg-background/40 print:border-gray-300 print:bg-gray-50">
                  <TableHeader>Candidate</TableHeader>
                  <TableHeader>Class</TableHeader>
                  <TableHeader>Priority</TableHeader>
                  <TableHeader>YOLO</TableHeader>
                  <TableHeader>Flow NLL</TableHeader>
                  <TableHeader>TTA</TableHeader>
                  <TableHeader>Bounding Box</TableHeader>
                  <TableHeader>Evidence</TableHeader>
                  <TableHeader>Review</TableHeader>
                </tr>
              </thead>

              <tbody>
                {report.candidates.map((candidate) => (
                  <tr
                    key={candidate.id}
                    className="border-b border-border/40 last:border-0 print:border-gray-200"
                  >
                    <TableCell>
                      <span className="font-mono">
                        {candidate.id}
                      </span>
                    </TableCell>

                    <TableCell>
                      {getClassName(candidate)}
                    </TableCell>

                    <TableCell>
                      <span
                        className={`font-mono text-[10px] font-semibold ${
                          candidate.priorityLevel === "HIGH"
                            ? "text-red-300 print:text-red-700"
                            : candidate.priorityLevel === "UNCERTAIN"
                              ? "text-amber-300 print:text-amber-700"
                              : candidate.priorityLevel === "REVIEW"
                                ? "text-cyan-300 print:text-cyan-700"
                                : "text-muted-foreground print:text-gray-600"
                        }`}
                      >
                        {candidate.priorityLevel}
                      </span>
                    </TableCell>

                    <TableCell>
                      {formatPercent(candidate.yoloConfidence)}
                    </TableCell>

                    <TableCell>
                      {formatFlow(candidate.flowScore)}
                    </TableCell>

                    <TableCell>
                      {formatPercent(candidate.ttaConsistency)}
                    </TableCell>

                    <TableCell>
                      <span className="font-mono text-[9px] text-muted-foreground print:text-gray-600">
                        {formatBBox(candidate.bbox)}
                      </span>
                    </TableCell>

                    <TableCell>
                      <span className="text-[10px] text-muted-foreground print:text-gray-600">
                        {candidate.evidenceProfile}
                      </span>
                    </TableCell>

                    <TableCell>
                      <span className="text-[10px] capitalize">
                        {candidate.reviewStatus}
                      </span>
                    </TableCell>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Recommendations */}
        <section className="mt-8">
          <SectionTitle title="Recommendations" />

          <div className="mt-3 space-y-3">
            {report.candidates.map((candidate) => (
              <div
                key={`${candidate.id}-recommendation`}
                className="rounded-xl border border-border/50 bg-background/30 p-4 print:border-gray-300"
              >
                <div className="flex items-start gap-3">
                  <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-cyan-400 print:text-cyan-700" />

                  <div>
                    <p className="text-xs font-medium">
                      Candidate {candidate.id}
                    </p>

                    <p className="mt-1 text-xs leading-5 text-muted-foreground print:text-gray-700">
                      {candidate.recommendation}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Methodology */}
        <section className="mt-8">
          <SectionTitle title="Analysis Methodology" />

          <div className="mt-3 rounded-xl border border-border/50 bg-background/30 p-4 print:border-gray-300">
            <p className="text-xs leading-6 text-muted-foreground print:text-gray-700">
              The SAAD pipeline combines YOLO-based candidate detection
              with VAE reconstruction analysis, RealNVP flow-based
              novelty assessment, and test-time augmentation consistency.
              These signals are combined by the deployed live API evidence policy to
              prioritize candidates and communicate uncertainty for
              human review.
            </p>
          </div>
        </section>

        {/* Geolocation note */}
        <section className="mt-8">
          <SectionTitle title="Geolocation" />

          <div className="mt-3 rounded-xl border border-border/50 bg-background/30 p-4 print:border-gray-300">
            <p className="text-xs leading-5 text-muted-foreground print:text-gray-700">
              Geolocation metadata is unavailable for this sonar frame.
              The report therefore provides candidate image coordinates
              and bounding dimensions rather than latitude/longitude.
            </p>
          </div>
        </section>

        {/* Footer */}
        <div className="mt-8 flex flex-col gap-2 border-t border-border/60 pt-5 text-[9px] text-muted-foreground sm:flex-row sm:items-center sm:justify-between print:border-gray-300 print:text-gray-500">
          <span>
            SAAD • Marine Intelligence System
          </span>

          <span className="font-mono">
            YOLO26s / VAE / RealNVP / TTA
          </span>
        </div>
      </div>

      <p className="text-center text-[10px] text-muted-foreground print:hidden">
        JSON and CSV exports contain structured candidate-level results.
        PDF output is generated from saved predictions and review decisions.
      </p>
    </div>
  );
}

function formatPercent(
  value: number | null | undefined,
) {
  if (
    value === null ||
    value === undefined ||
    !Number.isFinite(value)
  ) {
    return "—";
  }

  return `${(value * 100).toFixed(0)}%`;
}

function formatFlow(
  value: number | null | undefined,
) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "—";
  }

  return value.toFixed(2);
}

function formatBBox(
  bbox: Candidate["bbox"],
) {
  return `x:${bbox.x.toFixed(1)} y:${bbox.y.toFixed(1)} w:${bbox.width.toFixed(1)} h:${bbox.height.toFixed(1)}`;
}

function getClassName(candidate: Candidate) {
  return candidate.className;
}

/* ============================================================
   UI helpers
   ============================================================ */

function SectionTitle({
  title,
}: {
  title: string;
}) {
  return (
    <div>
      <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-cyan-400/70 print:text-cyan-700">
        SAAD Report
      </p>

      <h3 className="mt-1 text-base font-semibold">
        {title}
      </h3>
    </div>
  );
}

function InfoCard({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-border/50 bg-background/30 p-4 print:border-gray-300">
      <p className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
        {label}
      </p>

      <p className="mt-2 truncate text-xs">
        {value}
      </p>
    </div>
  );
}

function SummaryBox({
  label,
  value,
}: {
  label: string;
  value: number;
}) {
  return (
    <div className="rounded-xl border border-border/50 bg-background/30 p-4 print:border-gray-300">
      <p className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
        {label}
      </p>

      <p className="mt-2 text-2xl font-semibold">
        {value}
      </p>
    </div>
  );
}

function TableHeader({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <th className="px-3 py-3 font-mono text-[8px] uppercase tracking-wider text-muted-foreground">
      {children}
    </th>
  );
}

function TableCell({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <td className="px-3 py-3 text-[10px]">
      {children}
    </td>
  );
}
