export type ReviewStatus = "pending" | "accepted" | "rejected" | "corrected";
export type PriorityLevel = "HIGH" | "MEDIUM" | "REVIEW" | "UNCERTAIN" | "LOW";

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface BackendCandidate {
  id: string;
  bbox: BoundingBox;
  bbox_pixels: { x1: number; y1: number; x2: number; y2: number };
  classId: number;
  className: string;
  yoloConfidence: number;
  vaeScore: number | null;
  flowScore: number | null;
  ttaConsistency: number | null;
  priority: number;
  uncertainty: number;
  priorityLevel: PriorityLevel;
  evidenceProfile: string;
  recommendation: string;
  reviewStatus: string;
  evidence: {
    normalized: { yolo: number; vae: number | null; flow: number | null };
    raw: { yolo: number; vae: number | null; flow: number | null; tta: number | null };
    baseEvidence: number;
    agreement: number;
    disagreement: number;
    corroboration: number;
    ttaStability: number | null;
  };
}

export interface BackendAnalysis {
  success: boolean;
  surveyId: string;
  filename: string;
  image: { url: string; width: number; height: number; format: string | null };
  processing: { device: string; detector: string; imgsz: number; confidenceThreshold: number; tta: boolean; evidenceEngine: boolean };
  summary: { totalCandidates: number; highPriority: number; review: number; uncertain: number; lowPriority: number };
  candidates: BackendCandidate[];
  pipeline: Record<string, boolean>;
}

export interface ReviewState {
  status: "PENDING" | "ACCEPTED" | "REJECTED" | "CORRECTED";
  correctedBBox: BoundingBox | null;
  revision: number;
  updatedAt: string | null;
}

export interface ScanDetail {
  scanId: string;
  createdAt: string;
  analysisStatus: "complete" | "partial";
  analysis: BackendAnalysis;
  reviews: Record<string, ReviewState>;
}

export interface ScanSummary {
  scanId: string;
  createdAt: string;
  filename: string;
  analysisStatus: "complete" | "partial";
  imageUrl: string;
  summary: BackendAnalysis["summary"];
}

export interface ReviewQueueItem {
  scanId: string;
  candidateId: string;
  createdAt: string;
  prediction: BackendCandidate;
}

export interface Candidate extends BackendCandidate {
  bbox: BoundingBox;
  modelBBox: BoundingBox;
  reviewStatus: ReviewStatus;
  correctedBBox: BoundingBox | null;
}

export interface AnalysisResult {
  id: string;
  image: { url: string; width: number; height: number; filename: string };
  summary: { detections: number; highPriority: number; review: number; uncertain: number; lowPriority: number };
  candidates: Candidate[];
  analysisStatus: "complete" | "partial";
  processingDevice: string;
}
