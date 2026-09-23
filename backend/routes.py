"""FastAPI wiring and stable live endpoint contract."""

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import Settings
from saad_inference.registry import load_models
from saad_inference.service import analyze as run_analysis

SETTINGS = Settings.from_environment()
MODELS = load_models(SETTINGS)
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


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    return await run_analysis(file, MODELS, SETTINGS)


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
