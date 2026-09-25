"""One validated load of the frozen production model set per backend process."""

from dataclasses import dataclass

import torch
from ultralytics import YOLO

from config import Settings
from .vae import load_vae
from .realnvp import load_realnvp


@dataclass(frozen=True)
class ModelRegistry:
    yolo_model: object
    vae_model: object
    flow_model: object
    flow_latent_mean: object
    flow_latent_std: object


def load_models(settings: Settings) -> ModelRegistry:
    """Validate pinned assets and load the original architectures once."""
    settings.validate_assets()
    settings.prepare_runtime_dirs()
    yolo_path = settings.artifact_paths["yolo"]
    if not yolo_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {yolo_path}")

    print("=" * 70)
    print("SAAD BACKEND")
    print("=" * 70)
    print("Device:", settings.device)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Loading YOLO...")
    yolo_model = YOLO(str(yolo_path))
    print("YOLO loaded.")
    print("Loading VAE...")
    vae_model = load_vae(
        checkpoint_path=settings.artifact_paths["vae"], device=settings.device
    )
    print("VAE loaded.")
    print("Loading RealNVP...")
    flow_model, flow_latent_mean, flow_latent_std = load_realnvp(
        flow_checkpoint=settings.artifact_paths["flow"],
        latent_mean_path=settings.artifact_paths["latent_mean"],
        latent_std_path=settings.artifact_paths["latent_std"],
        device=settings.device,
    )
    print("RealNVP loaded.")
    print("TTA: enabled")
    print("Evidence Engine: enabled")
    print("=" * 70)
    return ModelRegistry(
        yolo_model, vae_model, flow_model, flow_latent_mean, flow_latent_std
    )
