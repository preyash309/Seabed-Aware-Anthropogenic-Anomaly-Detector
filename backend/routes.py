"""FastAPI wiring and stable live endpoint contract."""

import hashlib
import json
from pathlib import Path
import sqlite3
from urllib.parse import urlsplit

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from config import Settings
from schemas import ReviewRequest, ReviewState, ScanDetail
from reporting import report_csv, report_data, report_pdf
from saad_inference.registry import load_models
from saad_inference.service import analyze as run_analysis
from store import CandidateNotFound, ScanNotFound, ScanStore

SETTINGS = Settings.from_environment()
MODELS = load_models(SETTINGS)
try:
    STORE = ScanStore(SETTINGS.database_path)
    STORE_ERROR = None
except (OSError, sqlite3.DatabaseError) as exc:
    STORE = None
    STORE_ERROR = str(exc)
UPLOAD_DIR = SETTINGS.upload_dir
YOLO_MODEL_PATH = SETTINGS.artifact_paths["yolo"]
DEVICE = SETTINGS.device
API_BASE_URL = SETTINGS.api_public_url
YOLO_P5 = SETTINGS.calibration["normalization"]["yolo"]["p5"]
YOLO_P95 = SETTINGS.calibration["normalization"]["yolo"]["p95"]
VAE_P5 = SETTINGS.calibration["normalization"]["vae"]["p5"]
VAE_P95 = SETTINGS.calibration["normalization"]["vae"]["p95"]
FLOW_P5 = SETTINGS.calibration["normalization"]["flow"]["p5"]
FLOW_P95 = SETTINGS.calibration["normalization"]["flow"]["p95"]
YOLO_WEIGHT = SETTINGS.calibration["weights"]["yolo"]
VAE_WEIGHT = SETTINGS.calibration["weights"]["vae"]
FLOW_WEIGHT = SETTINGS.calibration["weights"]["flow"]

app = FastAPI(
    title="SAAD Backend",
    version="0.5.0",
    description="SAAD — Seabed-Aware Anthropogenic Anomaly Detector backend",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(SETTINGS.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


@app.get("/api/health")
def health():

    return {

        "status":
            "operational",

        "service":
            "SAAD Backend",

        "version":
            "0.5.0",

        "modelSetId":
            SETTINGS.manifest["model_set_id"],

        "policyId":
            SETTINGS.calibration["policy_id"],

        "device":
            DEVICE,

        "cuda_available":
            torch.cuda.is_available(),

        "cuda_version":
            torch.version.cuda,

        "gpu":
            (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),

        "models": {

            "yolo":
                YOLO_MODEL_PATH.exists(),

            "vae":
                True,

            "realnvp":
                True,

            "tta":
                True,

            "evidenceEngine":
                True,
        },
    }


@app.get("/api/model")
def model_info():

    return {

        "detector": {

            "name":
                "YOLO26s",

            "checkpoint":
                str(
                    YOLO_MODEL_PATH
                ),

            "loaded":
                True,
        },

        "vae": {

            "name":
                "ConvVAE-v1",

            "checkpoint": str(SETTINGS.artifact_paths["vae"]),

            "loaded":
                True,
        },

        "realnvp": {

            "name":
                "RealNVP",

            "checkpoint": str(SETTINGS.artifact_paths["flow"]),

            "latentMean": str(SETTINGS.artifact_paths["latent_mean"]),

            "latentStd": str(SETTINGS.artifact_paths["latent_std"]),

            "latentDim":
                128,

            "layers":
                8,

            "hidden":
                256,

            "loaded":
                True,
        },

        "tta": {

            "enabled":
                True,

            "perturbations": [
                "brightness",
                "contrast",
                "noise",
            ],

            "iouThreshold":
                0.30,
        },

        "evidenceEngine": {

            "enabled":
                True,

            "version":
                SETTINGS.calibration["policy_id"],

            "weights": {

                "yolo":
                    YOLO_WEIGHT,

                "vae":
                    VAE_WEIGHT,

                "flow":
                    FLOW_WEIGHT,
            },
        },

        "pipeline": {

            "yolo":
                True,

            "vae":
                True,

            "realnvp":
                True,

            "tta":
                True,

            "evidenceEngine":
                True,
        },

        "device":
            DEVICE,
    }


async def analyze(file: UploadFile = File(...)):
    return await run_analysis(file, MODELS, SETTINGS)


def available_store() -> ScanStore:
    if STORE is None:
        raise HTTPException(status_code=503, detail=f"Scan storage unavailable: {STORE_ERROR}")
    return STORE


@app.post("/api/analyze")
async def analyze_http(file: UploadFile = File(...)):
    store = available_store()
    result = await run_analysis(file, MODELS, SETTINGS)
    image_name = Path(urlsplit(result["image"]["url"]).path).name
    image_path = SETTINGS.upload_dir / image_name
    try:
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        store.save_scan(result, digest)
    except (OSError, sqlite3.DatabaseError, ValidationError, ValueError) as exc:
        image_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=f"Could not persist scan: {exc}") from exc
    return result


@app.get("/api/ready")
def readiness():
    store = available_store()
    try:
        with store.connection() as db:
            db.execute("SELECT 1").fetchone()
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=503, detail=f"Scan storage unavailable: {exc}") from exc
    return {"status": "ready", "modelSetId": SETTINGS.manifest["model_set_id"],
            "policyId": SETTINGS.calibration["policy_id"], "storage": "ready"}


@app.get("/api/scans")
def list_scans(limit: int = 100):
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
    try:
        return available_store().list_scans(limit)
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=503, detail=f"Scan storage unavailable: {exc}") from exc


@app.get("/api/review-queue")
def review_queue(limit: int = 100):
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
    try:
        return available_store().review_queue(limit)
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=503, detail=f"Scan storage unavailable: {exc}") from exc


@app.get("/api/scans/{scan_id}", response_model=ScanDetail)
def get_scan(scan_id: str):
    try:
        return available_store().get_scan(scan_id)
    except ScanNotFound as exc:
        raise HTTPException(status_code=404, detail="Scan not found") from exc
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=503, detail=f"Scan storage unavailable: {exc}") from exc


@app.put("/api/scans/{scan_id}/candidates/{candidate_id}/review", response_model=ReviewState)
def review_candidate(scan_id: str, candidate_id: str, request: ReviewRequest):
    try:
        return available_store().review(scan_id, candidate_id, request)
    except (ScanNotFound, CandidateNotFound) as exc:
        raise HTTPException(status_code=404, detail="Scan or candidate not found") from exc
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=503, detail=f"Scan storage unavailable: {exc}") from exc


@app.get("/api/scans/{scan_id}/review-events")
def review_events(scan_id: str):
    try:
        return available_store().events(scan_id)
    except ScanNotFound as exc:
        raise HTTPException(status_code=404, detail="Scan not found") from exc
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=503, detail=f"Scan storage unavailable: {exc}") from exc


@app.get("/api/scans/{scan_id}/report")
def scan_report(scan_id: str, format: str = "json", download: bool = False):
    if format not in {"json", "csv", "pdf"}:
        raise HTTPException(status_code=422, detail="format must be json, csv or pdf")
    try:
        store = available_store()
        detail = store.get_scan(scan_id)
        events = store.events(scan_id)
    except ScanNotFound as exc:
        raise HTTPException(status_code=404, detail="Scan not found") from exc
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=503, detail=f"Scan storage unavailable: {exc}") from exc
    data = report_data(
        detail, events, SETTINGS.manifest["model_set_id"], SETTINGS.calibration["policy_id"]
    )
    if format == "csv":
        return Response(
            report_csv(data), media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{scan_id}.csv"'},
        )
    if format == "pdf":
        return Response(
            report_pdf(data), media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{scan_id}.pdf"'},
        )
    if download:
        return Response(
            json.dumps(data), media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{scan_id}.json"'},
        )
    return data


@app.get("/")
def root():

    return {

        "service":
            "SAAD Backend",

        "status":
            "operational",

        "version":
            "0.5.0",

        "pipeline":
            (
                "YOLO26s + VAE-v1 + "
                "RealNVP + TTA + "
                "Evidence Engine"
            ),
    }
