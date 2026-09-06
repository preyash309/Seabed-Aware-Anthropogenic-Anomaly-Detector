import * as React from "react";

import {
  Check,
  Maximize2,
  Minus,
  Move,
  Plus,
  ScanLine,
  X,
} from "lucide-react";

import type { BoundingBox, Candidate } from "../../types/saad";

interface SonarViewerProps {
  imageUrl: string;
  candidates: Candidate[];
  selectedCandidateId: string | null;
  onCandidateSelect: (id: string) => void;

  correctionMode: boolean;
  correctionBox: BoundingBox | null;
  onCorrectionChange: (box: BoundingBox) => void;
  onSaveCorrection: () => void;
  onCancelCorrection: () => void;
}

type Corner = "nw" | "ne" | "sw" | "se";

interface ImageFrame {
  left: number;
  top: number;
  width: number;
  height: number;
}

export function SonarViewer({
  imageUrl,
  candidates,
  selectedCandidateId,
  onCandidateSelect,
  correctionMode,
  correctionBox,
  onCorrectionChange,
  onSaveCorrection,
  onCancelCorrection,
}: SonarViewerProps) {
  const canvasRef =
    React.useRef<HTMLDivElement | null>(null);

  const imageRef =
    React.useRef<HTMLImageElement | null>(null);

  const [imageFrame, setImageFrame] =
    React.useState<ImageFrame | null>(null);


  /*
   * ==========================================================
   * Calculate the EXACT object-contain image rectangle
   * ==========================================================
   *
   * The viewer has its own dimensions.
   *
   * The sonar image can have a completely different aspect
   * ratio, especially AI4Shipwrecks images.
   *
   * We calculate the dimensions that object-contain would
   * produce and then create an identical image frame.
   *
   * The candidate overlay lives INSIDE this frame.
   * ==========================================================
   */

  const updateImageFrame =
    React.useCallback(() => {
      const canvas = canvasRef.current;
      const image = imageRef.current;

      if (!canvas || !image) return;

      const canvasRect =
        canvas.getBoundingClientRect();

      const canvasWidth =
        canvasRect.width;

      const canvasHeight =
        canvasRect.height;

      const naturalWidth =
        image.naturalWidth;

      const naturalHeight =
        image.naturalHeight;

      if (
        !naturalWidth ||
        !naturalHeight ||
        !canvasWidth ||
        !canvasHeight
      ) {
        return;
      }

      const imageAspect =
        naturalWidth / naturalHeight;

      const canvasAspect =
        canvasWidth / canvasHeight;

      let width: number;
      let height: number;

      /*
       * Same calculation used by object-contain.
       */

      if (imageAspect > canvasAspect) {
        // Image is wider than the canvas.
        width = canvasWidth;
        height = width / imageAspect;
      } else {
        // Image is taller than the canvas.
        height = canvasHeight;
        width = height * imageAspect;
      }

      const left =
        (canvasWidth - width) / 2;

      const top =
        (canvasHeight - height) / 2;

      setImageFrame({
        left,
        top,
        width,
        height,
      });
    }, []);


  /*
   * Update when image loads.
   */

  function handleImageLoad() {
    updateImageFrame();

    requestAnimationFrame(() => {
      updateImageFrame();
    });
  }


  /*
   * Update when viewer dimensions change.
   */

  React.useEffect(() => {
    updateImageFrame();

    const canvas =
      canvasRef.current;

    if (!canvas) return;

    const observer =
      new ResizeObserver(() => {
        updateImageFrame();
      });

    observer.observe(canvas);

    window.addEventListener(
      "resize",
      updateImageFrame,
    );

    return () => {
      observer.disconnect();

      window.removeEventListener(
        "resize",
        updateImageFrame,
      );
    };
  }, [
    updateImageFrame,
    imageUrl,
  ]);


  /*
   * ==========================================================
   * MOVE CORRECTION BOX
   * ==========================================================
   */

  function beginMove(
    event: React.PointerEvent,
  ) {
    if (
      !correctionMode ||
      !correctionBox ||
      !imageFrame
    ) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();

    const startX =
      event.clientX;

    const startY =
      event.clientY;

    const initial = {
      ...correctionBox,
    };

    const target =
      event.currentTarget as HTMLElement;

    target.setPointerCapture?.(
      event.pointerId,
    );


    function move(
      moveEvent: PointerEvent,
    ) {
      const dx =
        ((moveEvent.clientX - startX) /
          imageFrame.width) *
        100;

      const dy =
        ((moveEvent.clientY - startY) /
          imageFrame.height) *
        100;


      const x = clamp(
        initial.x + dx,
        0,
        100 - initial.width,
      );

      const y = clamp(
        initial.y + dy,
        0,
        100 - initial.height,
      );


      onCorrectionChange({
        ...initial,
        x,
        y,
      });
    }


    function end() {
      window.removeEventListener(
        "pointermove",
        move,
      );

      window.removeEventListener(
        "pointerup",
        end,
      );
    }


    window.addEventListener(
      "pointermove",
      move,
    );

    window.addEventListener(
      "pointerup",
      end,
    );
  }


  /*
   * ==========================================================
   * RESIZE CORRECTION BOX
   * ==========================================================
   */

  function beginResize(
    event: React.PointerEvent,
    corner: Corner,
  ) {
    if (
      !correctionMode ||
      !correctionBox ||
      !imageFrame
    ) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();

    const startX =
      event.clientX;

    const startY =
      event.clientY;

    const initial = {
      ...correctionBox,
    };

    const target =
      event.currentTarget as HTMLElement;

    target.setPointerCapture?.(
      event.pointerId,
    );


    function resize(
      moveEvent: PointerEvent,
    ) {
      const dx =
        ((moveEvent.clientX - startX) /
          imageFrame.width) *
        100;

      const dy =
        ((moveEvent.clientY - startY) /
          imageFrame.height) *
        100;


      const next = {
        ...initial,
      };


      /*
       * LEFT
       */

      if (corner.includes("w")) {
        const newX = clamp(
          initial.x + dx,
          0,
          initial.x +
            initial.width -
            2,
        );

        next.x = newX;

        next.width =
          initial.width +
          (initial.x - newX);
      }


      /*
       * RIGHT
       */

      if (corner.includes("e")) {
        next.width = clamp(
          initial.width + dx,
          2,
          100 - initial.x,
        );
      }


      /*
       * TOP
       */

      if (corner.includes("n")) {
        const newY = clamp(
          initial.y + dy,
          0,
          initial.y +
            initial.height -
            2,
        );

        next.y = newY;

        next.height =
          initial.height +
          (initial.y - newY);
      }


      /*
       * BOTTOM
       */

      if (corner.includes("s")) {
        next.height = clamp(
          initial.height + dy,
          2,
          100 - initial.y,
        );
      }


      onCorrectionChange(next);
    }


    function end() {
      window.removeEventListener(
        "pointermove",
        resize,
      );

      window.removeEventListener(
        "pointerup",
        end,
      );
    }


    window.addEventListener(
      "pointermove",
      resize,
    );

    window.addEventListener(
      "pointerup",
      end,
    );
  }


  return (
    <div
      className="relative overflow-hidden rounded-2xl border border-border/70 bg-black"
    >

      {/* ======================================================
          Viewer header
          ====================================================== */}

      <div className="absolute left-0 right-0 top-0 z-30 flex items-center justify-between border-b border-white/10 bg-black/60 px-4 py-3 backdrop-blur-md">

        <div className="flex items-center gap-2">

          <ScanLine className="h-4 w-4 text-cyan-400" />

          <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-white/80">
            Side-scan sonar
          </span>

          {!correctionMode ? (
            <span className="rounded-md border border-emerald-400/20 bg-emerald-400/10 px-2 py-0.5 font-mono text-[9px] text-emerald-300">
              ANALYZED
            </span>
          ) : (
            <span className="rounded-md border border-amber-400/30 bg-amber-400/10 px-2 py-0.5 font-mono text-[9px] text-amber-300">
              CORRECTION MODE
            </span>
          )}

        </div>


        {!correctionMode && (
          <div className="flex items-center gap-1">

            <button
              type="button"
              className="rounded-md p-1.5 text-white/60 hover:bg-white/10 hover:text-white"
            >
              <Minus className="h-3.5 w-3.5" />
            </button>

            <span className="px-2 font-mono text-[9px] text-white/50">
              100%
            </span>

            <button
              type="button"
              className="rounded-md p-1.5 text-white/60 hover:bg-white/10 hover:text-white"
            >
              <Plus className="h-3.5 w-3.5" />
            </button>

            <button
              type="button"
              className="ml-1 rounded-md p-1.5 text-white/60 hover:bg-white/10 hover:text-white"
            >
              <Maximize2 className="h-3.5 w-3.5" />
            </button>

          </div>
        )}

      </div>


      {/* ======================================================
          Correction toolbar
          ====================================================== */}

      {correctionMode && (
        <div className="absolute bottom-4 left-1/2 z-30 flex -translate-x-1/2 items-center gap-2 rounded-xl border border-white/10 bg-black/80 p-2 shadow-2xl backdrop-blur-md">

          <div className="flex items-center gap-2 px-3">

            <Move className="h-3.5 w-3.5 text-amber-300" />

            <span className="font-mono text-[9px] text-white/70">
              DRAG TO POSITION · DRAG CORNERS TO RESIZE
            </span>

          </div>


          <button
            type="button"
            onClick={onCancelCorrection}
            className="flex items-center gap-1.5 rounded-lg border border-white/10 px-3 py-2 text-xs text-white/70 transition-colors hover:bg-white/10 hover:text-white"
          >
            <X className="h-3.5 w-3.5" />
            Cancel
          </button>


          <button
            type="button"
            onClick={onSaveCorrection}
            className="flex items-center gap-1.5 rounded-lg bg-emerald-400 px-3 py-2 text-xs font-semibold text-slate-950 transition-colors hover:bg-emerald-300"
          >
            <Check className="h-3.5 w-3.5" />
            Save correction
          </button>

        </div>
      )}


      {/* ======================================================
          Image canvas
          ====================================================== */}

      <div
        ref={canvasRef}
        className="relative flex aspect-[16/9] w-full items-center justify-center overflow-hidden bg-black"
      >

        {/* ====================================================
            Actual image
            ==================================================== */}

        <img
          ref={imageRef}
          src={imageUrl}
          alt="Analyzed side-scan sonar"
          draggable={false}
          onLoad={handleImageLoad}
          className="absolute inset-0 h-full w-full select-none object-contain"
        />


        {/* ====================================================
            EXACT IMAGE FRAME

            This frame has the exact same dimensions and
            position as object-contain.

            Most importantly, the candidate overlay is INSIDE
            this frame.

            Therefore:

              candidate.x = 54%

            means exactly 54% across the sonar image.
            ==================================================== */}

        {imageFrame && (
          <div
            className="absolute"
            style={{
              left: imageFrame.left,
              top: imageFrame.top,
              width: imageFrame.width,
              height: imageFrame.height,
            }}
          >

            {/* ================================================
                Candidate overlays
                ================================================ */}

            {candidates.map((candidate) => {

              const selected =
                candidate.id ===
                selectedCandidateId;

              const editing =
                correctionMode &&
                candidate.id ===
                selectedCandidateId;

              const box =
                editing &&
                correctionBox
                  ? correctionBox
                  : candidate.bbox;


              return (
                <div
                  key={candidate.id}
                  className="absolute"
                  style={{
                    left: `${box.x}%`,
                    top: `${box.y}%`,
                    width: `${box.width}%`,
                    height: `${box.height}%`,
                    zIndex: editing
                      ? 20
                      : selected
                        ? 15
                        : 10,
                  }}
                >

                  {/* ==========================================
                      Candidate box
                      ========================================== */}

                  <button
                    type="button"
                    aria-label={`Select ${candidate.id}`}
                    onClick={() => {
                      if (!correctionMode) {
                        onCandidateSelect(
                          candidate.id,
                        );
                      }
                    }}
                    onPointerDown={
                      editing
                        ? beginMove
                        : undefined
                    }
                    className={`pointer-events-auto absolute inset-0 rounded-md border-2 transition-all ${
                      editing
                        ? "cursor-move border-amber-300 bg-amber-300/10 shadow-[0_0_30px_rgba(251,191,36,0.35)]"
                        : selected
                          ? "cursor-pointer border-cyan-300 bg-cyan-300/10 shadow-[0_0_25px_rgba(34,211,238,0.35)]"
                          : candidate.priorityLevel ===
                              "HIGH"
                            ? "cursor-pointer border-red-400/80 bg-red-400/5 hover:border-red-300"
                            : candidate.priorityLevel ===
                                "UNCERTAIN"
                              ? "cursor-pointer border-amber-400/80 bg-amber-400/5 hover:border-amber-300"
                              : "cursor-pointer border-cyan-400/70 bg-cyan-400/5 hover:border-cyan-300"
                    }`}
                  />


                  {/* ==========================================
                      Candidate label
                      ========================================== */}

                  <span
                    className={`pointer-events-none absolute -top-6 left-0 rounded-md border px-2 py-1 font-mono text-[9px] font-semibold backdrop-blur-md ${
                      editing
                        ? "border-amber-300/40 bg-amber-400/20 text-amber-200"
                        : selected
                          ? "border-cyan-300/40 bg-cyan-400/20 text-cyan-200"
                          : "border-white/10 bg-black/70 text-white/70"
                    }`}
                  >
                    {candidate.id}
                  </span>


                  {/* ==========================================
                      Resize handles
                      ========================================== */}

                  {editing &&
                    (
                      [
                        "nw",
                        "ne",
                        "sw",
                        "se",
                      ] as Corner[]
                    ).map((corner) => (
                      <button
                        key={corner}
                        type="button"
                        aria-label={`Resize ${corner}`}
                        onPointerDown={(event) =>
                          beginResize(
                            event,
                            corner,
                          )
                        }
                        className={`pointer-events-auto absolute z-30 h-3 w-3 rounded-sm border-2 border-amber-200 bg-amber-400 shadow-[0_0_10px_rgba(251,191,36,0.5)] ${
                          corner === "nw"
                            ? "-left-1.5 -top-1.5 cursor-nwse-resize"
                            : corner === "ne"
                              ? "-right-1.5 -top-1.5 cursor-nesw-resize"
                              : corner === "sw"
                                ? "-bottom-1.5 -left-1.5 cursor-nesw-resize"
                                : "-bottom-1.5 -right-1.5 cursor-nwse-resize"
                        }`}
                      />
                    ))}
                </div>
              );
            })}

          </div>
        )}


        {/* ====================================================
            Frame telemetry
            ==================================================== */}

        <div className="pointer-events-none absolute bottom-3 left-3 rounded-md border border-white/10 bg-black/60 px-3 py-2 backdrop-blur-md">

          <p className="font-mono text-[9px] text-white/50">
            FRAME / SSS-001
          </p>

          <p className="mt-1 font-mono text-[9px] text-cyan-300">
            {candidates.length} CANDIDATES
          </p>

        </div>


        <div className="pointer-events-none absolute bottom-3 right-3 rounded-md border border-white/10 bg-black/60 px-3 py-2 backdrop-blur-md">

          <p className="font-mono text-[9px] text-white/50">
            VIEW / 01
          </p>

          <p className="mt-1 font-mono text-[9px] text-white/80">
            100%
          </p>

        </div>

      </div>

    </div>
  );
}


/* ============================================================
   Clamp helper
   ============================================================ */

function clamp(
  value: number,
  min: number,
  max: number,
): number {
  return Math.min(
    Math.max(value, min),
    max,
  );
}