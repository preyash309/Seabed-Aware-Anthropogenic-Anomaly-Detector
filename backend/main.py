"""Compatibility entry point: uvicorn main:app and Stage 1 imports."""

from routes import (
    app, analyze, health, model_info, root, SETTINGS, MODELS,
    UPLOAD_DIR, YOLO_MODEL_PATH, DEVICE, API_BASE_URL,
    YOLO_P5, YOLO_P95, VAE_P5, VAE_P95, FLOW_P5, FLOW_P95,
    YOLO_WEIGHT, VAE_WEIGHT, FLOW_WEIGHT,
)
from saad_inference.service import generate_survey_id
from saad_inference.detector import decode_detection, predict as detect
from saad_inference.vae import extract_candidate_patch
from saad_inference.realnvp import score_latent
from saad_inference.tta import compute_tta_consistency
from saad_inference.evidence.live_api_v1 import build_evidence, percentile_score

yolo_model = MODELS.yolo_model
vae_model = MODELS.vae_model
flow_model = MODELS.flow_model
flow_latent_mean = MODELS.flow_latent_mean
flow_latent_std = MODELS.flow_latent_std
