import {
  Check,
  CircleAlert,
  Fingerprint,
  GitBranch,
  ScanSearch,
  ShieldCheck,
  UserRoundCheck,
  X,
} from "lucide-react";

import type { Candidate } from "../../types/saad";

interface EvidencePanelProps {
  candidate: Candidate;
  onAccept: () => void;
  onReject: () => void;
  onCorrect: () => void;
}

export function EvidencePanel({
  candidate,
  onAccept,
  onReject,
  onCorrect,
}: EvidencePanelProps) {
  return (
    <div className="rounded-2xl border border-border/70 bg-card/40">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border/60 p-5">
        <div>
          <div className="flex items-center gap-2">
            <Fingerprint className="h-4 w-4 text-cyan-400" />

            <p className="text-sm font-medium">
              Candidate {candidate.id}
            </p>
          </div>

          <p className="mt-1 text-xs text-muted-foreground">
            Evidence profile and model assessment
          </p>
        </div>

        <span className="rounded-md border border-red-400/20 bg-red-400/10 px-2.5 py-1 font-mono text-[9px] font-semibold text-red-300">
          {candidate.priorityLevel}
        </span>
      </div>

      {/* Recommendation */}
      <div className="border-b border-border/60 p-5">
        <div className="flex items-start gap-3 rounded-xl border border-cyan-400/10 bg-cyan-400/5 p-4">
          <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-cyan-400" />

          <div>
            <p className="text-xs font-medium">
              {candidate.evidenceProfile}
            </p>

            <p className="mt-1 text-[10px] leading-relaxed text-muted-foreground">
              {candidate.recommendation}
            </p>
          </div>
        </div>
      </div>

      {/* Evidence */}
      <div className="p-5">
        <div className="mb-4 flex items-center gap-2">
          <GitBranch className="h-3.5 w-3.5 text-muted-foreground" />

          <p className="text-xs font-medium">Model evidence</p>
        </div>

        <div className="space-y-4">
          <EvidenceBar
            icon={ScanSearch}
            label="YOLO detector"
            description="Anthropogenic object confidence"
            value={candidate.yoloConfidence}
          />

          <EvidenceBar
            icon={CircleAlert}
            label="VAE normality"
            description="Deviation from normal seabed"
            value={candidate.vaeScore}
          />

          <EvidenceBar
            icon={Fingerprint}
            label="RealNVP novelty"
            description="Latent normality likelihood"
            value={candidate.flowScore}
          />

          <EvidenceBar
            icon={UserRoundCheck}
            label="TTA consistency"
            description="Prediction stability under perturbation"
            value={candidate.ttaConsistency}
          />
        </div>
      </div>

      {/* Secondary metrics */}
      <div className="grid grid-cols-2 gap-px border-t border-border/60 bg-border/60">
        <MetricBlock
          label="Priority score"
          value={`${(candidate.priority * 100).toFixed(0)}%`}
        />

        <MetricBlock
          label="Uncertainty"
          value={`${(candidate.uncertainty * 100).toFixed(0)}%`}
        />
      </div>

      {/* HITL controls */}
      <div className="border-t border-border/60 p-5">
        <p className="mb-3 font-mono text-[9px] uppercase tracking-[0.15em] text-muted-foreground/70">
          Human verification
        </p>

        <div className="grid gap-2 sm:grid-cols-3">
          <button
            type="button"
            onClick={onAccept}
            className="flex items-center justify-center gap-2 rounded-lg bg-emerald-400/10 px-3 py-2.5 text-xs font-medium text-emerald-300 transition-colors hover:bg-emerald-400/20"
          >
            <Check className="h-3.5 w-3.5" />
            Accept
          </button>

          <button
            type="button"
            onClick={onReject}
            className="flex items-center justify-center gap-2 rounded-lg bg-red-400/10 px-3 py-2.5 text-xs font-medium text-red-300 transition-colors hover:bg-red-400/20"
          >
            <X className="h-3.5 w-3.5" />
            Reject
          </button>

          <button
            type="button"
            onClick={onCorrect}
            className="flex items-center justify-center gap-2 rounded-lg border border-border/70 px-3 py-2.5 text-xs font-medium text-foreground transition-colors hover:bg-muted"
          >
            <UserRoundCheck className="h-3.5 w-3.5" />
            Correct
          </button>
        </div>
      </div>
    </div>
  );
}

function EvidenceBar({
  icon: Icon,
  label,
  description,
  value,
}: {
  icon: typeof ScanSearch;
  label: string;
  description: string;
  value: number;
}) {
  const percentage = Math.max(0, Math.min(100, value * 100));

  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <Icon className="h-3.5 w-3.5 text-cyan-400/80" />

        <div className="min-w-0 flex-1">
          <p className="text-[11px] font-medium">{label}</p>

          <p className="text-[9px] text-muted-foreground">
            {description}
          </p>
        </div>

        <span className="font-mono text-[10px] text-foreground/80">
          {percentage.toFixed(0)}%
        </span>
      </div>

      <div className="h-1.5 overflow-hidden rounded-full bg-muted/50">
        <div
          className="h-full rounded-full bg-cyan-400/80 transition-all duration-500"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

function MetricBlock({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="bg-background/30 p-4">
      <p className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
        {label}
      </p>

      <p className="mt-1 font-mono text-sm font-semibold">
        {value}
      </p>
    </div>
  );
}