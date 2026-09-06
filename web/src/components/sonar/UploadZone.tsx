import { useRef } from "react";
import {
  FileImage,
  FolderOpen,
  UploadCloud,
  X,
} from "lucide-react";

interface UploadZoneProps {
  file: File | null;
  previewUrl: string | null;
  onFileSelected: (file: File) => void;
  onRemove: () => void;
}

export function UploadZone({
  file,
  previewUrl,
  onFileSelected,
  onRemove,
}: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;

    const selected = files[0];

    if (!selected.type.startsWith("image/")) {
      return;
    }

    onFileSelected(selected);
  }

  return (
    <div
      className="group relative overflow-hidden rounded-2xl border border-border/70 bg-card/40 transition-colors hover:border-cyan-400/30"
      onDragOver={(event) => {
        event.preventDefault();
        event.currentTarget.classList.add("border-cyan-400/40");
      }}
      onDragLeave={(event) => {
        event.currentTarget.classList.remove("border-cyan-400/40");
      }}
      onDrop={(event) => {
        event.preventDefault();
        event.currentTarget.classList.remove("border-cyan-400/40");
        handleFiles(event.dataTransfer.files);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/jpg,image/webp,image/tiff"
        className="hidden"
        onChange={(event) => handleFiles(event.target.files)}
      />

      {previewUrl && file ? (
        <div className="relative">
          <div className="relative aspect-[16/8] overflow-hidden bg-black">
            <img
              src={previewUrl}
              alt="Selected side-scan sonar"
              className="h-full w-full object-contain"
            />

            <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-black/10" />

            <div className="absolute bottom-0 left-0 right-0 flex items-end justify-between p-5">
              <div>
                <p className="font-mono text-xs text-cyan-300">
                  SSS IMAGE READY
                </p>

                <p className="mt-1 text-sm font-medium text-white">
                  {file.name}
                </p>

                <p className="mt-1 text-xs text-white/60">
                  {(file.size / (1024 * 1024)).toFixed(2)} MB
                </p>
              </div>

              <button
                type="button"
                onClick={onRemove}
                className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/10 bg-black/50 text-white/70 backdrop-blur transition-colors hover:bg-red-500/20 hover:text-red-300"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>

          <div className="flex items-center justify-between border-t border-border/60 px-5 py-3">
            <div className="flex items-center gap-2">
              <FileImage className="h-4 w-4 text-cyan-400" />

              <span className="text-xs text-muted-foreground">
                Image selected for analysis
              </span>
            </div>

            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              className="text-xs font-medium text-cyan-400 transition-colors hover:text-cyan-300"
            >
              Replace image
            </button>
          </div>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="flex min-h-[360px] w-full flex-col items-center justify-center px-6 text-center"
        >
          <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl border border-cyan-400/20 bg-cyan-400/5 transition-transform group-hover:scale-105">
            <UploadCloud className="h-7 w-7 text-cyan-400" />
          </div>

          <p className="text-base font-medium">
            Drop side-scan sonar imagery here
          </p>

          <p className="mt-2 max-w-md text-sm text-muted-foreground">
            Upload a sonar frame to begin automated seabed anomaly analysis.
          </p>

          <div className="mt-6 flex items-center gap-2 rounded-lg border border-border/70 bg-background/50 px-4 py-2.5 text-xs font-medium text-muted-foreground transition-colors group-hover:border-cyan-400/20 group-hover:text-foreground">
            <FolderOpen className="h-4 w-4" />
            Browse files
          </div>

          <p className="mt-5 font-mono text-[10px] uppercase tracking-wider text-muted-foreground/50">
            PNG · JPG · WEBP · TIFF
          </p>
        </button>
      )}
    </div>
  );
}