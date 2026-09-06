import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowRight,
  CheckCircle2,
  Clock3,
  Database,
  FileImage,
  Gauge,
  ScanLine,
  ShieldCheck,
} from "lucide-react";

import { UploadZone } from "../components/sonar/UploadZone";
import { ScanPipeline } from "../components/sonar/ScanPipeline";

const API_BASE_URL = "http://127.0.0.1:8000";

export function NewScan() {
  const navigate = useNavigate();

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const [analyzing, setAnalyzing] = useState(false);
  const [complete, setComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }

    const reader = new FileReader();

    reader.onload = () => {
      if (typeof reader.result === "string") {
        setPreviewUrl(reader.result);
      }
    };

    reader.readAsDataURL(file);
  }, [file]);

  function handleFileSelected(selectedFile: File) {
    setFile(selectedFile);
    setAnalyzing(false);
    setComplete(false);
    setError(null);
  }

  function removeFile() {
    setFile(null);
    setPreviewUrl(null);
    setAnalyzing(false);
    setComplete(false);
    setError(null);
  }

  async function startAnalysis() {
    if (!file || analyzing) return;

    setAnalyzing(true);
    setComplete(false);
    setError(null);

    try {
      const formData = new FormData();

      formData.append("file", file);

      const response = await fetch(
        `${API_BASE_URL}/api/analyze`,
        {
          method: "POST",
          body: formData,
        },
      );

      if (!response.ok) {
        let message = `Backend returned HTTP ${response.status}`;

        try {
          const errorData = await response.json();

          if (errorData?.detail) {
            message = errorData.detail;
          }
        } catch {
          // Keep the default HTTP error message.
        }

        throw new Error(message);
      }

      const analysisResult = await response.json();

      if (!analysisResult?.success) {
        throw new Error(
          "SAAD backend did not return a successful analysis.",
        );
      }

      setAnalyzing(false);
      setComplete(true);

      /*
       * Pass the REAL backend result directly to the
       * analysis workspace.
       *
       * No mock result is created anymore.
       */
      window.setTimeout(() => {
        navigate(
          `/scan/${analysisResult.surveyId}`,
          {
            state: {
              analysisResult,
            },
          },
        );
      }, 500);
    } catch (err) {
      console.error("SAAD analysis failed:", err);

      setAnalyzing(false);
      setComplete(false);

      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to the SAAD backend.",
      );
    }
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="mb-3 flex items-center gap-2">
          <ScanLine className="h-4 w-4 text-cyan-400" />

          <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-400/80">
            New analysis
          </span>
        </div>

        <h1 className="text-3xl font-semibold tracking-tight">
          Start a sonar inspection
        </h1>

        <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
          Upload side-scan sonar imagery and let SAAD identify and
          prioritize potential anthropogenic anomalies.
        </p>
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="space-y-6">
          <UploadZone
            file={file}
            previewUrl={previewUrl}
            onFileSelected={handleFileSelected}
            onRemove={removeFile}
          />

          {file && (
            <div className="grid gap-3 sm:grid-cols-3">
              <InfoCard
                icon={FileImage}
                label="Input"
                value={file.name}
              />

              <InfoCard
                icon={Database}
                label="Size"
                value={`${(file.size / (1024 * 1024)).toFixed(2)} MB`}
              />

              <InfoCard
                icon={ShieldCheck}
                label="Input status"
                value={analyzing ? "Processing" : "Ready"}
              />
            </div>
          )}

          <div className="flex items-center justify-between rounded-2xl border border-border/70 bg-card/40 p-5">
            <div>
              <p className="text-sm font-medium">
                Ready to inspect the seafloor?
              </p>

              <p className="mt-1 text-xs text-muted-foreground">
                SAAD will process this image through the current
                detection pipeline.
              </p>
            </div>

            <button
              type="button"
              disabled={!file || analyzing}
              onClick={startAnalysis}
              className="group flex items-center gap-2 rounded-lg bg-cyan-400 px-5 py-2.5 text-xs font-semibold text-slate-950 transition-all hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {analyzing ? "Analyzing..." : "Start analysis"}

              {!analyzing && (
                <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
              )}
            </button>
          </div>

          {error && (
            <div className="rounded-2xl border border-red-400/20 bg-red-400/5 p-5">
              <p className="text-sm font-medium text-red-300">
                Analysis failed
              </p>

              <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                {error}
              </p>

              <p className="mt-3 font-mono text-[10px] text-muted-foreground/60">
                Check that the SAAD FastAPI backend is running on
                127.0.0.1:8000.
              </p>
            </div>
          )}

          {complete && (
            <div className="flex items-center gap-4 rounded-2xl border border-emerald-400/20 bg-emerald-400/5 p-5">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-emerald-400/10">
                <CheckCircle2 className="h-5 w-5 text-emerald-400" />
              </div>

              <div className="flex-1">
                <p className="text-sm font-medium text-emerald-300">
                  Analysis complete
                </p>

                <p className="mt-1 text-xs text-muted-foreground">
                  Redirecting to the SAAD analysis workspace...
                </p>
              </div>
            </div>
          )}
        </div>

        <div className="space-y-4">
          <ScanPipeline
            analyzing={analyzing}
            complete={complete}
          />

          <div className="rounded-2xl border border-border/70 bg-card/30 p-5">
            <p className="text-xs font-medium">
              Estimated processing
            </p>

            <div className="mt-4 grid grid-cols-2 gap-3">
              <TelemetryCard
                icon={Clock3}
                label="Runtime"
                value="GPU inference"
              />

              <TelemetryCard
                icon={Gauge}
                label="Hardware"
                value="RTX 4070"
              />
            </div>

            <p className="mt-4 text-[10px] leading-relaxed text-muted-foreground/70">
              Processing time depends on image resolution,
              hardware and the number of detections returned by
              the model.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function InfoCard({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof FileImage;
  label: string;
  value: string;
}) {
  return (
    <div className="flex min-w-0 items-center gap-3 rounded-xl border border-border/60 bg-card/30 p-4">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-muted/40">
        <Icon className="h-4 w-4 text-cyan-400" />
      </div>

      <div className="min-w-0">
        <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
          {label}
        </p>

        <p className="mt-1 truncate text-xs font-medium">
          {value}
        </p>
      </div>
    </div>
  );
}

function TelemetryCard({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Clock3;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-border/50 bg-background/30 p-3">
      <Icon className="h-3.5 w-3.5 text-muted-foreground" />

      <p className="mt-3 text-[10px] text-muted-foreground">
        {label}
      </p>

      <p className="mt-1 font-mono text-xs">{value}</p>
    </div>
  );
}