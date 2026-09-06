import {
  BrainCircuit,
  Check,
  Crosshair,
  Loader2,
  ScanSearch,
  Sparkles,
} from "lucide-react";

interface ScanPipelineProps {
  analyzing: boolean;
  complete: boolean;
}

const steps = [
  {
    title: "Preprocessing",
    description: "Normalize acoustic imagery",
    icon: ScanSearch,
  },
  {
    title: "Object detection",
    description: "Locate anthropogenic candidates",
    icon: Crosshair,
  },
  {
    title: "Seabed analysis",
    description: "Evaluate normality deviation",
    icon: BrainCircuit,
  },
  {
    title: "Evidence synthesis",
    description: "Combine model evidence",
    icon: Sparkles,
  },
];

export function ScanPipeline({
  analyzing,
  complete,
}: ScanPipelineProps) {
  return (
    <div className="rounded-2xl border border-border/70 bg-card/40 p-5">
      <div className="mb-5">
        <p className="text-sm font-medium">SAAD analysis pipeline</p>

        <p className="mt-1 text-xs text-muted-foreground">
          Multi-stage acoustic anomaly analysis
        </p>
      </div>

      <div className="space-y-3">
        {steps.map((step, index) => {
          const Icon = step.icon;

          const active = analyzing && index === 2;

          const finished = complete || (analyzing && index < 2);

          return (
            <div
              key={step.title}
              className={`flex items-center gap-3 rounded-xl border p-3 transition-colors ${
                active
                  ? "border-cyan-400/20 bg-cyan-400/5"
                  : "border-border/50 bg-background/30"
              }`}
            >
              <div
                className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${
                  finished
                    ? "bg-emerald-400/10 text-emerald-400"
                    : active
                      ? "bg-cyan-400/10 text-cyan-400"
                      : "bg-muted/40 text-muted-foreground"
                }`}
              >
                {finished ? (
                  <Check className="h-4 w-4" />
                ) : active ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Icon className="h-4 w-4" />
                )}
              </div>

              <div className="min-w-0 flex-1">
                <p className="text-xs font-medium">{step.title}</p>

                <p className="mt-0.5 text-[10px] text-muted-foreground">
                  {step.description}
                </p>
              </div>

              <div className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground/50">
                {finished
                  ? "Complete"
                  : active
                    ? "Running"
                    : "Ready"}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}