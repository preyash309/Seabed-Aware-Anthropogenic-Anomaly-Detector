"""Validated runtime paths and frozen artifact identity for the live API."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import torch
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "config" / "model_manifest.json"
DEFAULT_CALIBRATION = REPO_ROOT / "config" / "live_api_v1.json"
load_dotenv(REPO_ROOT / ".env", override=False)
ARTIFACT_ENV = {
    "yolo": "SAAD_YOLO_WEIGHTS",
    "vae": "SAAD_VAE_WEIGHTS",
    "flow": "SAAD_FLOW_WEIGHTS",
    "latent_mean": "SAAD_LATENT_MEAN",
    "latent_std": "SAAD_LATENT_STD",
}


class ConfigurationError(RuntimeError):
    """The runtime cannot load its declared frozen model set."""


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot read configuration {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ConfigurationError(f"Configuration must be a JSON object: {path}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Settings:
    manifest_path: Path
    calibration_path: Path
    artifact_paths: dict[str, Path]
    dataset_dir: Path | None
    upload_dir: Path
    output_dir: Path
    database_path: Path
    max_upload_bytes: int
    max_image_pixels: int
    api_public_url: str
    cors_origins: tuple[str, ...]
    device: str
    manifest: dict[str, Any]
    calibration: dict[str, Any]

    @classmethod
    def from_environment(cls) -> "Settings":
        env = os.environ
        manifest_path = _path(env.get("SAAD_MODEL_MANIFEST", str(DEFAULT_MANIFEST)))
        manifest = _json(manifest_path)
        if manifest.get("schema_version") != 1:
            raise ConfigurationError("Unsupported model manifest schema_version")
        if manifest.get("default_policy_id") != "saad-live-api-v1":
            raise ConfigurationError("The deployed policy must be saad-live-api-v1")
        specs = manifest.get("artifacts")
        if not isinstance(specs, dict) or set(specs) != set(ARTIFACT_ENV):
            raise ConfigurationError("Manifest must declare exactly the five live artifacts")
        for name, spec in specs.items():
            if not isinstance(spec, dict):
                raise ConfigurationError(f"Invalid {name} artifact specification")
            relative = spec.get("relative_path")
            digest = spec.get("sha256")
            size = spec.get("bytes")
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or not isinstance(size, int)
                or size <= 0
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
            ):
                raise ConfigurationError(f"Invalid {name} artifact path, size or SHA-256")

        calibration_spec = manifest.get("calibration")
        if (
            not isinstance(calibration_spec, dict)
            or calibration_spec.get("policy_id") != "saad-live-api-v1"
            or not isinstance(calibration_spec.get("sha256"), str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", calibration_spec["sha256"]) is None
        ):
            raise ConfigurationError("Manifest must pin the live calibration SHA-256")

        root_value = env.get("SAAD_ARTIFACT_DIR")
        root = _path(root_value) if root_value else None
        paths: dict[str, Path] = {}
        for name, variable in ARTIFACT_ENV.items():
            override = env.get(variable)
            if override:
                paths[name] = _path(override)
            elif root is not None:
                paths[name] = root / specs[name]["relative_path"]
            else:
                raise ConfigurationError(
                    f"Set SAAD_ARTIFACT_DIR or {variable}; no model path is implicit"
                )

        calibration_path = _path(env.get("SAAD_CALIBRATION_FILE", str(DEFAULT_CALIBRATION)))
        calibration = _json(calibration_path)
        if calibration.get("policy_id") != "saad-live-api-v1":
            raise ConfigurationError("Calibration policy must be saad-live-api-v1")
        try:
            normalization = calibration["normalization"]
            weights = calibration["weights"]
            for name in ("yolo", "vae", "flow"):
                low = float(normalization[name]["p5"])
                high = float(normalization[name]["p95"])
                weight = float(weights[name])
                if not (low < high and 0 <= weight <= 1):
                    raise ValueError(name)
            if abs(sum(float(weights[n]) for n in ("yolo", "vae", "flow")) - 1) > 1e-12:
                raise ValueError("weights must sum to one")
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid live calibration: {exc}") from exc

        dataset = env.get("SAAD_DATASET_DIR")
        dataset_dir = _path(dataset) if dataset else None
        if dataset_dir is not None and not dataset_dir.is_dir():
            raise ConfigurationError(f"Dataset directory not found: {dataset_dir}")
        device_value = env.get("SAAD_DEVICE", "auto")
        if device_value == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device_value == "cpu":
            device = "cpu"
        elif device_value.startswith("cuda:") and torch.cuda.is_available():
            try:
                index = int(device_value.split(":", 1)[1])
                if index < 0 or index >= torch.cuda.device_count():
                    raise ValueError()
            except ValueError as exc:
                raise ConfigurationError(f"Unavailable CUDA device: {device_value}") from exc
            device = device_value
        else:
            raise ConfigurationError(f"Unavailable SAAD_DEVICE: {device_value}")

        origins = tuple(x.strip() for x in env.get(
            "SAAD_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",") if x.strip())
        if not origins or any(not x.startswith(("http://", "https://")) for x in origins):
            raise ConfigurationError("SAAD_CORS_ORIGINS must contain HTTP origins")

        try:
            max_upload_bytes = int(env.get("SAAD_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
            if max_upload_bytes <= 0:
                raise ValueError()
        except ValueError as exc:
            raise ConfigurationError("SAAD_MAX_UPLOAD_BYTES must be a positive integer") from exc
        try:
            max_image_pixels = int(env.get("SAAD_MAX_IMAGE_PIXELS", "50000000"))
            if max_image_pixels <= 0:
                raise ValueError()
        except ValueError as exc:
            raise ConfigurationError("SAAD_MAX_IMAGE_PIXELS must be a positive integer") from exc
        upload_dir = _path(env.get("SAAD_UPLOAD_DIR", "var/uploads"))
        output_dir = _path(env.get("SAAD_OUTPUT_DIR", "var/results"))
        database_path = _path(env.get("SAAD_DATABASE_PATH", "var/saad.sqlite3"))
        for runtime_path in (upload_dir, output_dir, database_path):
            if not runtime_path.resolve().is_relative_to(REPO_ROOT.resolve()):
                raise ConfigurationError(
                    f"Writable runtime path must stay inside the clean repository: {runtime_path}"
                )

        return cls(
            manifest_path=manifest_path,
            calibration_path=calibration_path,
            artifact_paths=paths,
            dataset_dir=dataset_dir,
            upload_dir=upload_dir,
            output_dir=output_dir,
            database_path=database_path,
            max_upload_bytes=max_upload_bytes,
            max_image_pixels=max_image_pixels,
            api_public_url=env.get("SAAD_API_PUBLIC_URL", "http://127.0.0.1:8000").rstrip("/"),
            cors_origins=origins,
            device=device,
            manifest=manifest,
            calibration=calibration,
        )

    def validate_assets(self) -> None:
        for name, path in self.artifact_paths.items():
            spec = self.manifest["artifacts"][name]
            if not path.is_file():
                raise ConfigurationError(f"Missing {name} artifact: {path}")
            actual_size = path.stat().st_size
            if actual_size != spec["bytes"]:
                raise ConfigurationError(
                    f"Size mismatch for {name}: {path} ({actual_size} != {spec['bytes']})"
                )
            actual_hash = _sha256(path)
            if actual_hash.lower() != spec["sha256"].lower():
                raise ConfigurationError(f"SHA-256 mismatch for {name}: {path}")
        calibration_spec = self.manifest.get("calibration", {})
        if calibration_spec.get("policy_id") != self.calibration["policy_id"]:
            raise ConfigurationError("Calibration policy and manifest disagree")
        expected_calibration_hash = calibration_spec["sha256"]
        if _sha256(self.calibration_path).lower() != expected_calibration_hash.lower():
            raise ConfigurationError("Calibration SHA-256 mismatch")

    def prepare_runtime_dirs(self) -> None:
        for path in (self.upload_dir, self.output_dir, self.database_path.parent):
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ConfigurationError(f"Cannot create runtime directory {path}: {exc}") from exc
            if not path.is_dir():
                raise ConfigurationError(f"Runtime path is not a directory: {path}")
