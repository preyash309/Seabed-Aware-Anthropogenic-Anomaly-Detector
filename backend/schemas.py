"""Typed API contracts; the legacy analysis payload remains byte-for-byte shaped."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PercentBox(BaseModel):
    x: float = Field(ge=0, le=100)
    y: float = Field(ge=0, le=100)
    width: float = Field(gt=0, le=100)
    height: float = Field(gt=0, le=100)

    @model_validator(mode="after")
    def inside_image(self) -> "PercentBox":
        if self.x + self.width > 100.000001 or self.y + self.height > 100.000001:
            raise ValueError("Corrected box must stay inside the image")
        return self


class PixelBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class ImageInfo(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    format: str | None
    url: str


class CandidatePrediction(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    bbox: dict[str, float]
    bbox_pixels: PixelBox
    classId: int
    yoloConfidence: float
    vaeScore: float | None
    flowScore: float | None
    ttaConsistency: float | None
    priority: float
    uncertainty: float
    priorityLevel: str
    evidenceProfile: str
    recommendation: str
    reviewStatus: str
    evidence: dict[str, Any]


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    success: bool
    surveyId: str
    filename: str
    image: ImageInfo
    processing: dict[str, Any]
    summary: dict[str, int]
    candidates: list[CandidatePrediction]
    pipeline: dict[str, bool]


class ReviewRequest(BaseModel):
    action: Literal["ACCEPT", "REJECT", "CORRECT"]
    bbox: PercentBox | None = None
    note: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def correction_contract(self) -> "ReviewRequest":
        if self.action == "CORRECT" and self.bbox is None:
            raise ValueError("CORRECT requires bbox")
        if self.action != "CORRECT" and self.bbox is not None:
            raise ValueError("bbox is only valid for CORRECT")
        return self


class ReviewState(BaseModel):
    status: Literal["PENDING", "ACCEPTED", "REJECTED", "CORRECTED"]
    correctedBBox: PercentBox | None = None
    revision: int = 0
    updatedAt: str | None = None


class ScanDetail(BaseModel):
    scanId: str
    createdAt: str
    analysisStatus: Literal["complete", "partial"]
    analysis: AnalysisResponse
    reviews: dict[str, ReviewState]
