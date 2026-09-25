import type {
  BackendAnalysis,
  BoundingBox,
  ReviewQueueItem,
  ReviewState,
  ScanDetail,
  ScanSummary,
} from "../types/saad";

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

function path(value: string): string {
  return `${API_BASE_URL}${value.startsWith("/") ? value : `/${value}`}`;
}

export function resolveImageUrl(url: string): string {
  const pathname = new URL(url, API_BASE_URL).pathname;
  return path(pathname);
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path(url), init);
  } catch {
    throw new Error(`Cannot reach the SAAD API at ${API_BASE_URL}.`);
  }
  if (!response.ok) {
    let detail: string | undefined;
    try {
      const body: unknown = await response.json();
      if (typeof body === "object" && body !== null && "detail" in body) {
        detail = String(body.detail);
      }
    } catch { /* Preserve the HTTP status below. */ }
    throw new Error(detail ?? `SAAD API returned HTTP ${response.status}.`);
  }
  return response.json() as Promise<T>;
}

export async function analyzeImage(file: File): Promise<BackendAnalysis> {
  const form = new FormData();
  form.append("file", file);
  const result = await request<BackendAnalysis>("/api/analyze", { method: "POST", body: form });
  if (!result.success || !result.surveyId || !Array.isArray(result.candidates)) {
    throw new Error("The SAAD API returned an invalid analysis response.");
  }
  return result;
}

export const getScan = (id: string) => request<ScanDetail>(`/api/scans/${encodeURIComponent(id)}`);
export const listScans = () => request<ScanSummary[]>("/api/scans");
export const getReviewQueue = () => request<ReviewQueueItem[]>("/api/review-queue");
export interface HealthStatus {
  status: string;
  device: string;
  cuda_available: boolean;
  gpu: string | null;
  modelSetId: string;
  policyId: string;
}
export const getHealth = () => request<HealthStatus>("/api/health");

export function reviewCandidate(
  scanId: string,
  candidateId: string,
  action: "ACCEPT" | "REJECT" | "CORRECT",
  bbox?: BoundingBox,
): Promise<ReviewState> {
  return request<ReviewState>(
    `/api/scans/${encodeURIComponent(scanId)}/candidates/${encodeURIComponent(candidateId)}/review`,
    { method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, ...(bbox ? { bbox } : {}) }) },
  );
}

export function reportUrl(scanId: string, format: "json" | "csv" | "pdf") {
  return path(`/api/scans/${encodeURIComponent(scanId)}/report?format=${format}`);
}

export async function getReport(scanId: string): Promise<ServerReport> {
  return request<ServerReport>(`/api/scans/${encodeURIComponent(scanId)}/report`);
}

export interface ServerReport {
  reportType: string;
  generatedAt: string;
  scanId: string;
  createdAt: string;
  modelSetId: string;
  policyId: string;
  analysisStatus: "complete" | "partial";
  filename: string;
  image: BackendAnalysis["image"];
  summary: BackendAnalysis["summary"];
  candidates: { prediction: BackendAnalysis["candidates"][number]; review: ReviewState }[];
  reviewEvents: { eventId: number; candidateId: string; action: string; createdAt: string }[];
}
