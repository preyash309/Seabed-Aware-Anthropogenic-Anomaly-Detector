import {
  AlertTriangle,
  ChevronRight,
  CircleHelp,
  ShieldAlert,
} from "lucide-react";

import type { Candidate } from "../../types/saad";

interface CandidateListProps {
  candidates: Candidate[];
  selectedCandidateId: string | null;
  onSelect: (id: string) => void;
}

function levelStyles(level: Candidate["priorityLevel"]) {
  switch (level) {
    case "HIGH":
      return {
        badge: "border-red-400/20 bg-red-400/10 text-red-300",
        icon: "text-red-400",
      };

    case "UNCERTAIN":
      return {
        badge: "border-amber-400/20 bg-amber-400/10 text-amber-300",
        icon: "text-amber-400",
      };

    case "REVIEW":
      return {
        badge: "border-cyan-400/20 bg-cyan-400/10 text-cyan-300",
        icon: "text-cyan-400",
      };

    default:
      return {
        badge: "border-border bg-muted/30 text-muted-foreground",
        icon: "text-muted-foreground",
      };
  }
}

/**
 * Convert raw RealNVP NLL into the same validation-normalized
 * novelty percentage used by the SAAD evidence system.
 *
 * Calibration:
 *   P5  = 49.3122
 *   P95 = 264.3907
 *
 * Values below P5 -> 0%
 * Values above P95 -> 100%
 */
function flowNoveltyPercent(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) {
    return null;
  }

  const P5 = 49.3122;
  const P95 = 264.3907;

  const normalized =
    (value - P5) / (P95 - P5);

  const clamped = Math.max(
    0,
    Math.min(1, normalized),
  );

  return clamped * 100;
}

export function CandidateList({
  candidates,
  selectedCandidateId,
  onSelect,
}: CandidateListProps) {
  return (
    <div className="flex h-full min-h-0 flex-col rounded-2xl border border-border/70 bg-card/40">
      <div className="border-b border-border/60 p-5">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium">
              Detection queue
            </p>

            <p className="mt-1 text-xs text-muted-foreground">
              Ranked candidates from SAAD
            </p>
          </div>

          <span className="rounded-md border border-border/60 bg-background/50 px-2 py-1 font-mono text-[10px] text-muted-foreground">
            {candidates.length} TOTAL
          </span>
        </div>
      </div>

      <div className="min-h-0 flex-1 space-y-2 overflow-auto p-3">
        {candidates.map((candidate, index) => {
          const selected =
            candidate.id === selectedCandidateId;

          const styles = levelStyles(
            candidate.priorityLevel,
          );

          const Icon =
            candidate.priorityLevel === "HIGH"
              ? ShieldAlert
              : candidate.priorityLevel === "UNCERTAIN"
                ? CircleHelp
                : AlertTriangle;

          const flowPercent = flowNoveltyPercent(
            candidate.flowScore,
          );

          return (
            <button
              key={candidate.id}
              type="button"
              onClick={() => onSelect(candidate.id)}
              className={`w-full rounded-xl border p-3 text-left transition-all ${
                selected
                  ? "border-cyan-400/30 bg-cyan-400/5"
                  : "border-border/50 bg-background/30 hover:border-border hover:bg-background/60"
              }`}
            >
              <div className="flex items-center gap-3">
                <div className={`shrink-0 ${styles.icon}`}>
                  <Icon className="h-4 w-4" />
                </div>

                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px] text-muted-foreground">
                      {String(index + 1).padStart(2, "0")}
                    </span>

                    <span className="text-xs font-medium">
                      Candidate {candidate.id}
                    </span>
                  </div>

                  <p className="mt-1 truncate text-[10px] text-muted-foreground">
                    {candidate.evidenceProfile}
                  </p>
                </div>

                <span
                  className={`rounded-md border px-2 py-1 font-mono text-[9px] font-semibold ${styles.badge}`}
                >
                  {candidate.priorityLevel}
                </span>

                <ChevronRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground/50" />
              </div>

              <div className="mt-3 flex items-center gap-4 border-t border-border/40 pt-2.5">
                <Metric
                  label="YOLO"
                  value={candidate.yoloConfidence}
                />

                <Metric
                  label="FLOW"
                  value={flowPercent}
                />

                <Metric
                  label="TTA"
                  value={candidate.ttaConsistency}
                />
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function Metric({
  label,
  value,
}: {
  label: string;
  value: number | null | undefined;
}) {
  if (value == null || !Number.isFinite(value)) {
    return (
      <div>
        <p className="font-mono text-[8px] uppercase tracking-wider text-muted-foreground/60">
          {label}
        </p>

        <p className="mt-0.5 font-mono text-[10px] text-muted-foreground">
          —
        </p>
      </div>
    );
  }

  return (
    <div>
      <p className="font-mono text-[8px] uppercase tracking-wider text-muted-foreground/60">
        {label}
      </p>

      <p className="mt-0.5 font-mono text-[10px] text-foreground/80">
        {value.toFixed(0)}%
      </p>
    </div>
  );
}