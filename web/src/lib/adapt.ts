import { resolveImageUrl } from "./api";
import type { AnalysisResult, Candidate, ReviewStatus, ScanDetail } from "../types/saad";

function status(value: string): ReviewStatus {
  switch (value) {
    case "ACCEPTED": return "accepted";
    case "REJECTED": return "rejected";
    case "CORRECTED": return "corrected";
    default: return "pending";
  }
}

export function adaptScan(detail: ScanDetail): AnalysisResult {
  const analysis = detail.analysis;
  const candidates: Candidate[] = analysis.candidates.map((prediction) => {
    const review = detail.reviews[prediction.id];
    const correctedBBox = review?.correctedBBox ?? null;
    return {
      ...prediction,
      modelBBox: prediction.bbox,
      bbox: correctedBBox ?? prediction.bbox,
      correctedBBox,
      reviewStatus: status(review?.status ?? "PENDING"),
    };
  });
  return {
    id: detail.scanId,
    image: {
      url: resolveImageUrl(analysis.image.url),
      width: analysis.image.width,
      height: analysis.image.height,
      filename: analysis.filename,
    },
    summary: {
      detections: analysis.summary.totalCandidates,
      highPriority: analysis.summary.highPriority,
      review: analysis.summary.review,
      uncertain: analysis.summary.uncertain,
      lowPriority: analysis.summary.lowPriority,
    },
    candidates,
    analysisStatus: detail.analysisStatus,
    processingDevice: analysis.processing.device,
  };
}
