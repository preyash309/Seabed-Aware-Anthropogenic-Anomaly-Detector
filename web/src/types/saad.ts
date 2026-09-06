export type ReviewStatus =
  | "pending"
  | "accepted"
  | "rejected"
  | "corrected";

export type PriorityLevel =
  | "HIGH"
  | "REVIEW"
  | "UNCERTAIN"
  | "LOW";

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Candidate {
  id: string;
  bbox: BoundingBox;

  yoloConfidence: number;
  vaeScore: number;
  flowScore: number;
  ttaConsistency: number;

  priority: number;
  uncertainty: number;

  priorityLevel: PriorityLevel;
  evidenceProfile: string;
  recommendation: string;

  reviewStatus: ReviewStatus;
}

export interface AnalysisResult {
  id: string;

  image: {
    url: string;
    width: number;
    height: number;
    filename: string;
  };

  summary: {
    detections: number;
    highPriority: number;
    review: number;
    uncertain: number;
    lowPriority: number;
  };

  candidates: Candidate[];
}