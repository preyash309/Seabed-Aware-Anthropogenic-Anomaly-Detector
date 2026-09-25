# ============================================================
# SAAD BACKEND
# FastAPI + YOLO26s + VAE + RealNVP + TTA + Evidence Engine v3
# ============================================================

from pathlib import Path
from typing import Optional
import uuid
import math
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"E:\SIH")

RESULTS_DIR = BASE_DIR / "SIH_Results"

YOLO_MODEL_PATH = (
    RESULTS_DIR
    / "yolo26s_generic_baseline"
    / "weights"
    / "best.pt"
)

VAE_CHECKPOINT_PATH = (
    RESULTS_DIR
    / "vae_normal_seabed"
    / "checkpoints"
    / "best.pt"
)

FLOW_DIR = RESULTS_DIR / "latent_normalizing_flow"

FLOW_CHECKPOINT_PATH = FLOW_DIR / "best_flow.pt"
LATENT_MEAN_PATH = FLOW_DIR / "latent_mean.pt"
LATENT_STD_PATH = FLOW_DIR / "latent_std.pt"

EVIDENCE_DIR = RESULTS_DIR / "saad_evidence_engine_v3"

NORMALIZATION_CSV = (
    EVIDENCE_DIR
    / "normalization_parameters.csv"
)

UPLOAD_DIR = BASE_DIR / "backend" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

DEVICE_STR = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("SAAD BACKEND")
print("=" * 70)
print("Device:", DEVICE)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# EVIDENCE ENGINE V3 CONFIG
# ============================================================

W_YOLO = 0.50
W_VAE = 0.20
W_FLOW = 0.30

LOW_PCT = 5.0
HIGH_PCT = 95.0

WEAK = 0.33
MODERATE = 0.66

CORROBORATION_BONUS = 0.10
TRIPLE_AGREEMENT_BONUS = 0.05
LOW_EVIDENCE_PENALTY = 0.10

DISAGREEMENT_WEIGHT = 0.70
AMBIGUITY_WEIGHT = 0.30


# ============================================================
# HELPERS
# ============================================================

def clean_float(value):
    try:
        value = float(value)

        if math.isfinite(value):
            return value

    except Exception:
        pass

    return None


def clamp01(value):
    if value is None:
        return 0.0

    return float(
        np.clip(
            float(value),
            0.0,
            1.0
        )
    )


def percentile_soft_normalize(
    value,
    low,
    high
):
    """
    Exact runtime equivalent of the v3 soft percentile
    normalization.

    Scores are evidence scales, NOT probabilities.
    """

    if value is None:
        return 0.0

    if high <= low:
        return 0.0

    midpoint = (
        low + high
    ) / 2.0

    scale = (
        high - low
    ) / (
        2.0 * 2.944439
    )

    scale = max(
        scale,
        1e-12
    )

    z = (
        float(value) - midpoint
    ) / scale

    z = np.clip(
        z,
        -40.0,
        40.0
    )

    result = (
        1.0
        / (
            1.0
            + np.exp(-z)
        )
    )

    return float(result)


def evidence_band(score):

    if not math.isfinite(score):
        return "UNKNOWN"

    if score < WEAK:
        return "LOW"

    if score < MODERATE:
        return "MODERATE"

    return "STRONG"


# ============================================================
# LOAD V3 NORMALIZATION
# ============================================================

if not NORMALIZATION_CSV.exists():

    raise FileNotFoundError(
        "SAAD Evidence Engine v3 normalization file "
        "was not found:\n"
        f"{NORMALIZATION_CSV}\n\n"
        "Run the Evidence Engine v3 script first."
    )


normalization_df = pd.read_csv(
    NORMALIZATION_CSV
)

normalization = {}

for _, row in normalization_df.iterrows():

    signal = str(
        row["signal"]
    )

    normalization[signal] = {
        "p_low": float(
            row["low_value"]
        ),
        "p_high": float(
            row["high_value"]
        )
    }


for required_signal in [
    "yolo_raw",
    "vae_raw",
    "flow_raw"
]:

    if required_signal not in normalization:

        raise RuntimeError(
            "Missing normalization parameters for "
            f"{required_signal}"
        )


print()
print("Evidence Engine v3 normalization:")

for signal, params in normalization.items():

    print(
        f"  {signal:10s} "
        f"P5={params['p_low']:.8f} "
        f"P95={params['p_high']:.8f}"
    )


# ============================================================
# REALNVP
# ============================================================
#
# IMPORTANT:
# Your previously reconstructed RealNVP architecture had
# a state_dict naming mismatch.
#
# The working inference module already solved that.
#
# We import it rather than duplicating the architecture here.
# ============================================================

try:

    from realnvp_inference import (
        load_realnvp,
        score_latents,
    )

except ImportError as exc:

    raise RuntimeError(
        "Could not import realnvp_inference.py.\n"
        "Make sure it is located beside main.py:\n"
        "E:\\SIH\\backend\\realnvp_inference.py"
    ) from exc


print()
print("Loading RealNVP...")

FLOW_MODEL, FLOW_MEAN, FLOW_STD = (
    load_realnvp()
)

FLOW_MODEL.to(DEVICE)
FLOW_MODEL.eval()

FLOW_MEAN = FLOW_MEAN.to(
    DEVICE
)

FLOW_STD = FLOW_STD.to(
    DEVICE
)

print("RealNVP loaded.")


# ============================================================
# VAE
# ============================================================

class ConvVAE(nn.Module):

    def __init__(
        self,
        latent_dim=128
    ):

        super().__init__()

        self.latent_dim = (
            latent_dim
        )

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                512,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.fc_mu = nn.Linear(
            512 * 8 * 8,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            512 * 8 * 8,
            latent_dim
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.fc_decode = nn.Linear(
            latent_dim,
            512 * 8 * 8
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                512,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32,
                1,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Sigmoid()
        )


    def encode(
        self,
        x
    ):

        h = self.encoder(x)

        h = h.reshape(
            h.size(0),
            -1
        )

        mu = self.fc_mu(h)

        logvar = self.fc_logvar(h)

        return mu, logvar


    def decode(
        self,
        z
    ):

        h = self.fc_decode(z)

        h = h.reshape(
            -1,
            512,
            8,
            8
        )

        return self.decoder(h)


    def forward(
        self,
        x
    ):

        mu, logvar = self.encode(x)

        # Deterministic inference:
        # use mu instead of sampling.
        z = mu

        reconstruction = self.decode(z)

        return (
            reconstruction,
            mu,
            logvar
        )


print()
print("Loading VAE...")

if not VAE_CHECKPOINT_PATH.exists():

    raise FileNotFoundError(
        "VAE checkpoint not found:\n"
        f"{VAE_CHECKPOINT_PATH}"
    )


VAE_MODEL = ConvVAE(
    latent_dim=128
)


vae_checkpoint = torch.load(
    VAE_CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)


# Support common checkpoint formats.

if isinstance(
    vae_checkpoint,
    dict
):

    if (
        "model_state_dict"
        in vae_checkpoint
    ):

        vae_state = (
            vae_checkpoint[
                "model_state_dict"
            ]
        )

    elif (
        "state_dict"
        in vae_checkpoint
    ):

        vae_state = (
            vae_checkpoint[
                "state_dict"
            ]
        )

    else:

        vae_state = (
            vae_checkpoint
        )

else:

    vae_state = vae_checkpoint


VAE_MODEL.load_state_dict(
    vae_state,
    strict=True
)

VAE_MODEL.to(DEVICE)
VAE_MODEL.eval()

print("VAE loaded.")


# ============================================================
# IMAGE / PATCH PREPROCESSING
# ============================================================

PATCH_SIZE = 256
PATCH_STRIDE = 192


def prepare_grayscale_image(
    image
):

    image = ImageOps.grayscale(
        image
    )

    image = image.convert(
        "L"
    )

    array = np.asarray(
        image,
        dtype=np.float32
    )

    # Same basic [0,1] image representation
    # expected by the VAE.

    array /= 255.0

    array = np.clip(
        array,
        0.0,
        1.0
    )

    return array


def extract_patches(
    array
):

    height, width = (
        array.shape
    )

    patches = []

    locations = []

    # --------------------------------------------------------
    # Standard sliding windows
    # --------------------------------------------------------

    y_positions = list(
        range(
            0,
            max(
                1,
                height - PATCH_SIZE + 1
            ),
            PATCH_STRIDE
        )
    )

    x_positions = list(
        range(
            0,
            max(
                1,
                width - PATCH_SIZE + 1
            ),
            PATCH_STRIDE
        )
    )

    # Ensure last patch touches bottom/right boundary.

    if height > PATCH_SIZE:

        last_y = (
            height - PATCH_SIZE
        )

        if last_y not in y_positions:

            y_positions.append(
                last_y
            )

    else:

        y_positions = [0]


    if width > PATCH_SIZE:

        last_x = (
            width - PATCH_SIZE
        )

        if last_x not in x_positions:

            x_positions.append(
                last_x
            )

    else:

        x_positions = [0]


    for y in y_positions:

        for x in x_positions:

            patch = array[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            # Pad small edge patches.

            if (
                patch.shape[0]
                != PATCH_SIZE
                or
                patch.shape[1]
                != PATCH_SIZE
            ):

                padded = np.zeros(
                    (
                        PATCH_SIZE,
                        PATCH_SIZE
                    ),
                    dtype=np.float32
                )

                padded[
                    :patch.shape[0],
                    :patch.shape[1]
                ] = patch

                patch = padded


            patches.append(
                patch
            )

            locations.append(
                (
                    x,
                    y
                )
            )


    return (
        np.asarray(
            patches,
            dtype=np.float32
        ),
        locations
    )


# ============================================================
# VAE + FLOW SCORING
# ============================================================

@torch.inference_mode()
def score_normality(
    image
):

    array = (
        prepare_grayscale_image(
            image
        )
    )

    patches, locations = (
        extract_patches(
            array
        )
    )

    if len(patches) == 0:

        return {
            "vae_raw": 0.0,
            "flow_raw": 0.0,
            "vae_patch_scores": [],
            "flow_patch_scores": [],
            "patch_locations": []
        }


    batch = torch.from_numpy(
        patches
    ).unsqueeze(1)

    batch = batch.to(
        DEVICE,
        non_blocking=True
    )


    # --------------------------------------------------------
    # VAE
    # --------------------------------------------------------

    reconstruction, mu, _ = (
        VAE_MODEL(
            batch
        )
    )

    mse = torch.mean(
        (
            reconstruction
            - batch
        ) ** 2,
        dim=(1, 2, 3)
    )


    # --------------------------------------------------------
    # RealNVP
    #
    # Standardize deterministic VAE mu using the exact
    # training normal mean/std saved with the flow.
    # --------------------------------------------------------

    standardized_mu = (
        mu - FLOW_MEAN
    ) / torch.clamp(
        FLOW_STD,
        min=1e-8
    )


    # Existing realnvp_inference.py is responsible for
    # calculating the actual flow NLL.

    flow_nll = score_latents(
        FLOW_MODEL,
        standardized_mu
    )


    if torch.is_tensor(
        flow_nll
    ):

        flow_nll = (
            flow_nll.detach()
            .float()
            .cpu()
            .numpy()
        )

    else:

        flow_nll = np.asarray(
            flow_nll,
            dtype=np.float32
        )


    vae_scores = (
        mse.detach()
        .float()
        .cpu()
        .numpy()
    )


    flow_scores = (
        np.asarray(
            flow_nll,
            dtype=np.float32
        )
    )


    # Image-level score = maximum patch anomaly.
    #
    # This matches the image-level evaluation framing used
    # for the normality models.

    vae_raw = float(
        np.max(
            vae_scores
        )
    )

    flow_raw = float(
        np.max(
            flow_scores
        )
    )


    return {
        "vae_raw": vae_raw,
        "flow_raw": flow_raw,

        "vae_patch_scores":
            vae_scores.tolist(),

        "flow_patch_scores":
            flow_scores.tolist(),

        "patch_locations":
            locations
    }


# ============================================================
# YOLO
# ============================================================

print()
print("Loading YOLO...")

if not YOLO_MODEL_PATH.exists():

    raise FileNotFoundError(
        "YOLO checkpoint not found:\n"
        f"{YOLO_MODEL_PATH}"
    )


YOLO_MODEL = YOLO(
    str(YOLO_MODEL_PATH)
)

print("YOLO loaded.")


# ============================================================
# YOLO INFERENCE
# ============================================================

def run_yolo(
    image_path
):

    results = YOLO_MODEL.predict(
        source=str(
            image_path
        ),

        imgsz=640,

        conf=0.05,

        device=DEVICE_STR,

        verbose=False
    )


    if not results:

        return []


    result = results[0]

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):

        return []


    boxes = (
        result.boxes.xyxy
        .detach()
        .cpu()
        .numpy()
    )

    confidences = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    classes = (
        result.boxes.cls
        .detach()
        .cpu()
        .numpy()
    )


    candidates = []


    for i, (
        box,
        confidence,
        class_id
    ) in enumerate(
        zip(
            boxes,
            confidences,
            classes
        )
    ):

        x1, y1, x2, y2 = (
            map(
                float,
                box
            )
        )

        candidates.append(
            {
                "index": i,

                "confidence":
                    float(
                        confidence
                    ),

                "classId":
                    int(
                        class_id
                    ),

                "bbox_pixels": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            }
        )


    # Highest detector confidence first.

    candidates.sort(
        key=lambda x:
            x["confidence"],
        reverse=True
    )

    return candidates


# ============================================================
# TTA
# ============================================================

def bbox_iou(
    box_a,
    box_b
):

    ax1, ay1, ax2, ay2 = (
        box_a
    )

    bx1, by1, bx2, by2 = (
        box_b
    )


    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )


    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )

    intersection = (
        iw * ih
    )


    area_a = max(
        0.0,
        ax2 - ax1
    ) * max(
        0.0,
        ay2 - ay1
    )

    area_b = max(
        0.0,
        bx2 - bx1
    ) * max(
        0.0,
        by2 - by1
    )


    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:

        return 0.0

    return (
        intersection
        / union
    )


def transform_flip_box(
    box,
    width
):

    x1, y1, x2, y2 = (
        box
    )

    return (
        width - x2,
        y1,
        width - x1,
        y2
    )


def run_tta_consistency(
    image_path,
    base_candidates
):

    if not base_candidates:

        return 0.0


    with Image.open(
        image_path
    ) as original:

        image = original.convert(
            "RGB"
        )

        width, height = (
            image.size
        )

        flipped = ImageOps.mirror(
            image
        )


    # Temporary in-memory TTA image.

    tta_path = (
        UPLOAD_DIR
        / (
            f"_tta_"
            f"{uuid.uuid4().hex}.png"
        )
    )

    flipped.save(
        tta_path
    )


    try:

        tta_candidates = (
            run_yolo(
                tta_path
            )
        )

    finally:

        try:
            tta_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass


    if not tta_candidates:

        return 0.0


    consistencies = []


    # Compare each original prediction with
    # the best horizontally flipped prediction.

    for candidate in base_candidates:

        original_box = (
            candidate[
                "bbox_pixels"
            ]
        )

        original_tuple = (
            original_box["x1"],
            original_box["y1"],
            original_box["x2"],
            original_box["y2"]
        )


        best_iou = 0.0


        for tta_candidate in (
            tta_candidates
        ):

            tta_box = (
                tta_candidate[
                    "bbox_pixels"
                ]
            )

            tta_tuple = (
                tta_box["x1"],
                tta_box["y1"],
                tta_box["x2"],
                tta_box["y2"]
            )


            # Convert mirrored prediction back
            # to original coordinate system.

            restored = (
                transform_flip_box(
                    tta_tuple,
                    width
                )
            )


            iou = bbox_iou(
                original_tuple,
                restored
            )

            best_iou = max(
                best_iou,
                iou
            )


        consistencies.append(
            best_iou
        )


    if not consistencies:

        return 0.0


    return float(
        np.mean(
            consistencies
        )
    )


# ============================================================
# EVIDENCE ENGINE V3 RUNTIME
# ============================================================

def build_evidence(
    yolo_raw,
    vae_raw,
    flow_raw,
    tta_consistency
):

    yolo_evidence = (
        percentile_soft_normalize(
            yolo_raw,
            normalization[
                "yolo_raw"
            ]["p_low"],
            normalization[
                "yolo_raw"
            ]["p_high"]
        )
    )


    vae_evidence = (
        percentile_soft_normalize(
            vae_raw,
            normalization[
                "vae_raw"
            ]["p_low"],
            normalization[
                "vae_raw"
            ]["p_high"]
        )
    )


    flow_evidence = (
        percentile_soft_normalize(
            flow_raw,
            normalization[
                "flow_raw"
            ]["p_low"],
            normalization[
                "flow_raw"
            ]["p_high"]
        )
    )


    y = yolo_evidence
    v = vae_evidence
    f = flow_evidence


    # --------------------------------------------------------
    # Base evidence
    # --------------------------------------------------------

    base = (
        W_YOLO * y
        + W_VAE * v
        + W_FLOW * f
    )


    # --------------------------------------------------------
    # Normality corroboration
    # --------------------------------------------------------

    normality_min = min(
        v,
        f
    )

    normality_mean = (
        v + f
    ) / 2.0


    normality_strong = (
        normality_min
        >= MODERATE
    )

    detector_strong = (
        y >= MODERATE
    )

    detector_moderate = (
        y >= WEAK
    )


    detector_plus_normality = (
        detector_strong
        and (
            v >= MODERATE
            or f >= MODERATE
        )
    )


    triple_strong = (
        y >= MODERATE
        and v >= MODERATE
        and f >= MODERATE
    )


    # --------------------------------------------------------
    # Agreement / disagreement
    # --------------------------------------------------------

    evidence_max = max(
        y,
        v,
        f
    )

    evidence_min = min(
        y,
        v,
        f
    )

    disagreement = (
        evidence_max
        - evidence_min
    )

    agreement = (
        1.0
        - disagreement
    )

    agreement = float(
        np.clip(
            agreement,
            0.0,
            1.0
        )
    )


    # --------------------------------------------------------
    # Ambiguity
    # --------------------------------------------------------

    ambiguity = (
        1.0
        - 2.0
        * np.mean(
            [
                abs(y - 0.50),
                abs(v - 0.50),
                abs(f - 0.50)
            ]
        )
    )

    ambiguity = float(
        np.clip(
            ambiguity,
            0.0,
            1.0
        )
    )


    # --------------------------------------------------------
    # Uncertainty
    # --------------------------------------------------------

    uncertainty = (
        DISAGREEMENT_WEIGHT
        * disagreement
        + AMBIGUITY_WEIGHT
        * ambiguity
    )

    uncertainty = float(
        np.clip(
            uncertainty,
            0.0,
            1.0
        )
    )


    # --------------------------------------------------------
    # Priority
    # --------------------------------------------------------

    priority = base


    if detector_plus_normality:

        priority += (
            CORROBORATION_BONUS
        )


    if triple_strong:

        priority += (
            TRIPLE_AGREEMENT_BONUS
        )


    all_weak = (
        y < WEAK
        and v < WEAK
        and f < WEAK
    )


    if all_weak:

        priority -= (
            LOW_EVIDENCE_PENALTY
        )


    priority = float(
        np.clip(
            priority,
            0.0,
            1.0
        )
    )


    # --------------------------------------------------------
    # Evidence profile
    # --------------------------------------------------------

    y_strong = (
        y >= MODERATE
    )

    v_strong = (
        v >= MODERATE
    )

    f_strong = (
        f >= MODERATE
    )

    y_weak = (
        y < WEAK
    )

    v_weak = (
        v < WEAK
    )

    f_weak = (
        f < WEAK
    )


    if (
        y_strong
        and v_strong
        and f_strong
    ):

        profile = (
            "STRONG_MULTI_SIGNAL"
        )

    elif (
        y_strong
        and (
            v_strong
            or f_strong
        )
    ):

        profile = (
            "DETECTOR_PLUS_NORMALITY"
        )

    elif y_strong:

        profile = (
            "DETECTOR_DOMINANT"
        )

    elif (
        v_strong
        and f_strong
    ):

        profile = (
            "NORMALITY_CORROBORATED"
        )

    elif (
        v_strong
        or f_strong
    ):

        profile = (
            "SINGLE_NORMALITY_SIGNAL"
        )

    elif (
        y_weak
        and v_weak
        and f_weak
    ):

        profile = (
            "LOW_EVIDENCE"
        )

    else:

        profile = (
            "AMBIGUOUS"
        )


    # --------------------------------------------------------
    # HITL action
    # --------------------------------------------------------

    if (
        priority >= 0.70
        and uncertainty < 0.35
    ):

        action = (
            "HIGH_PRIORITY_REVIEW"
        )

    elif (
        priority >= 0.70
        and uncertainty >= 0.35
    ):

        action = (
            "HIGH_PRIORITY_UNCERTAIN"
        )

    elif (
        priority >= 0.50
        and uncertainty >= 0.40
    ):

        action = (
            "UNCERTAIN_REVIEW"
        )

    elif priority >= 0.50:

        action = (
            "REVIEW"
        )

    elif (
        priority < 0.50
        and uncertainty >= 0.55
    ):

        action = (
            "NOVELTY_REVIEW"
        )

    elif priority >= 0.30:

        action = (
            "LOW_PRIORITY_REVIEW"
        )

    else:

        action = (
            "DEFER"
        )


    # --------------------------------------------------------
    # Human-readable reason
    # --------------------------------------------------------

    parts = []


    if y >= MODERATE:

        parts.append(
            "strong detector evidence"
        )

    elif y >= WEAK:

        parts.append(
            "moderate detector evidence"
        )

    else:

        parts.append(
            "weak detector evidence"
        )


    if v >= MODERATE:

        parts.append(
            "strong VAE deviation"
        )

    elif v >= WEAK:

        parts.append(
            "moderate VAE deviation"
        )

    else:

        parts.append(
            "low VAE deviation"
        )


    if f >= MODERATE:

        parts.append(
            "strong Flow deviation"
        )

    elif f >= WEAK:

        parts.append(
            "moderate Flow deviation"
        )

    else:

        parts.append(
            "low Flow deviation"
        )


    if agreement >= 0.75:

        parts.append(
            "high cross-model agreement"
        )

    elif agreement >= 0.50:

        parts.append(
            "moderate cross-model agreement"
        )

    else:

        parts.append(
            "high cross-model disagreement"
        )


    if uncertainty >= 0.60:

        parts.append(
            "high uncertainty"
        )

    elif uncertainty >= 0.35:

        parts.append(
            "moderate uncertainty"
        )

    else:

        parts.append(
            "low uncertainty"
        )


    reason = (
        "; ".join(parts)
    )


    return {

        # Raw scores
        "yoloRaw": yolo_raw,
        "vaeRaw": vae_raw,
        "flowRaw": flow_raw,

        # Evidence scales
        "yoloEvidence":
            yolo_evidence,

        "vaeEvidence":
            vae_evidence,

        "flowEvidence":
            flow_evidence,

        # TTA
        "ttaConsistency":
            clamp01(
                tta_consistency
            ),

        # Evidence engine
        "baseEvidence":
            base,

        "normalityEvidence":
            normality_mean,

        "evidenceMin":
            evidence_min,

        "evidenceMax":
            evidence_max,

        "agreement":
            agreement,

        "disagreement":
            disagreement,

        "ambiguity":
            ambiguity,

        "priority":
            priority,

        "uncertainty":
            uncertainty,

        "yoloBand":
            evidence_band(
                yolo_evidence
            ),

        "vaeBand":
            evidence_band(
                vae_evidence
            ),

        "flowBand":
            evidence_band(
                flow_evidence
            ),

        "evidenceProfile":
            profile,

        "recommendedAction":
            action,

        "evidenceReason":
            reason
    }


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="SAAD Backend",
    version="0.2.0"
)


app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


app.mount(
    "/uploads",
    StaticFiles(
        directory=str(
            UPLOAD_DIR
        )
    ),
    name="uploads"
)


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/api/health"
)
def health():

    return {

        "status":
            "operational",

        "device":
            DEVICE_STR,

        "cuda":
            bool(
                torch.cuda.is_available()
            ),

        "gpu":
            (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),

        "models": {

            "yolo":
                True,

            "vae":
                True,

            "realnvp":
                True,

            "tta":
                True,

            "evidenceEngine":
                "v3"
        }
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get(
    "/api/model"
)
def model_info():

    return {

        "detector": {

            "name":
                "YOLO26s",

            "checkpoint":
                str(
                    YOLO_MODEL_PATH
                ),

            "status":
                "loaded"
        },

        "normality": {

            "vae":
                "loaded",

            "realnvp":
                "loaded"
        },

        "tta":
            "enabled",

        "evidenceEngine": {

            "version":
                "v3",

            "weights": {

                "yolo":
                    W_YOLO,

                "vae":
                    W_VAE,

                "flow":
                    W_FLOW
            },

            "normalization":
                "validation_normals_only",

            "scoresAreProbabilities":
                False
        }
    }


# ============================================================
# ANALYZE
# ============================================================

@app.post(
    "/api/analyze"
)
async def analyze(
    file: UploadFile = File(...)
):

    start_time = (
        time.perf_counter()
    )


    # --------------------------------------------------------
    # Validate extension
    # --------------------------------------------------------

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp"
    }


    original_name = (
        file.filename
        or "sonar.png"
    )

    extension = (
        Path(
            original_name
        ).suffix.lower()
    )


    if extension not in (
        allowed_extensions
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format."
            )
        )


    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    survey_id = (
        "SAAD-"
        + uuid.uuid4()
        .hex[:8]
        .upper()
    )


    unique_name = (
        f"{survey_id}"
        f"{extension}"
    )


    output_path = (
        UPLOAD_DIR
        / unique_name
    )


    file_bytes = (
        await file.read()
    )


    with open(
        output_path,
        "wb"
    ) as f:

        f.write(
            file_bytes
        )


    # --------------------------------------------------------
    # Validate / load image
    # --------------------------------------------------------

    try:

        with Image.open(
            output_path
        ) as img:

            image = img.convert(
                "RGB"
            )

            width, height = (
                image.size
            )

    except Exception as exc:

        try:
            output_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass

        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid image: {exc}"
            )
        )


    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    yolo_start = (
        time.perf_counter()
    )

    candidates = run_yolo(
        output_path
    )

    yolo_time = (
        time.perf_counter()
        - yolo_start
    )


    # --------------------------------------------------------
    # Normality models
    # --------------------------------------------------------

    normality_start = (
        time.perf_counter()
    )

    normality = (
        score_normality(
            image
        )
    )

    normality_time = (
        time.perf_counter()
        - normality_start
    )


    # --------------------------------------------------------
    # TTA
    # --------------------------------------------------------

    tta_start = (
        time.perf_counter()
    )

    tta_consistency = (
        run_tta_consistency(
            output_path,
            candidates
        )
    )

    tta_time = (
        time.perf_counter()
        - tta_start
    )


    # --------------------------------------------------------
    # Image-level raw detector score
    # --------------------------------------------------------

    if candidates:

        yolo_raw = max(
            c["confidence"]
            for c in candidates
        )

    else:

        yolo_raw = 0.0


    vae_raw = float(
        normality[
            "vae_raw"
        ]
    )

    flow_raw = float(
        normality[
            "flow_raw"
        ]
    )


    # --------------------------------------------------------
    # Evidence engine
    #
    # IMPORTANT:
    # The same image-level evidence is attached to every
    # candidate for now.
    #
    # This is deliberate because the VAE/Flow evaluation
    # currently operates at image level.
    #
    # Later we can make the normality scores candidate-ROI
    # specific, but the previous experiments showed that
    # ROI scoring was not a reliable object-specific signal.
    # --------------------------------------------------------

    evidence = build_evidence(
        yolo_raw,
        vae_raw,
        flow_raw,
        tta_consistency
    )


    # --------------------------------------------------------
    # Candidate JSON
    # --------------------------------------------------------

    candidate_output = []


    for index, candidate in enumerate(
        candidates
    ):

        box = (
            candidate[
                "bbox_pixels"
            ]
        )


        x1 = box["x1"]
        y1 = box["y1"]
        x2 = box["x2"]
        y2 = box["y2"]


        bbox = {

            "x":
                x1
                / width
                * 100.0,

            "y":
                y1
                / height
                * 100.0,

            "width":
                (
                    x2 - x1
                )
                / width
                * 100.0,

            "height":
                (
                    y2 - y1
                )
                / height
                * 100.0
        }


        # ----------------------------------------------------
        # Recommendation mapping
        # ----------------------------------------------------

        action = (
            evidence[
                "recommendedAction"
            ]
        )


        if action in (
            "HIGH_PRIORITY_REVIEW",
            "HIGH_PRIORITY_UNCERTAIN"
        ):

            priority_level = (
                "HIGH"
            )

        elif action in (
            "REVIEW",
            "UNCERTAIN_REVIEW"
        ):

            priority_level = (
                "MEDIUM"
            )

        elif action == (
            "LOW_PRIORITY_REVIEW"
        ):

            priority_level = (
                "LOW"
            )

        elif action == (
            "NOVELTY_REVIEW"
        ):

            priority_level = (
                "MEDIUM"
            )

        else:

            priority_level = (
                "LOW"
            )


        candidate_output.append({

            "id":
                f"candidate-{index + 1:02d}",

            "bbox":
                bbox,

            "bbox_pixels":
                box,

            "yoloConfidence":
                candidate[
                    "confidence"
                ],

            "vaeScore":
                evidence[
                    "vaeEvidence"
                ],

            "flowScore":
                evidence[
                    "flowEvidence"
                ],

            "ttaConsistency":
                evidence[
                    "ttaConsistency"
                ],

            "priority":
                evidence[
                    "priority"
                ],

            "uncertainty":
                evidence[
                    "uncertainty"
                ],

            "priorityLevel":
                priority_level,

            "evidenceProfile":
                evidence[
                    "evidenceProfile"
                ],

            "recommendedAction":
                action,

            "evidenceReason":
                evidence[
                    "evidenceReason"
                ],

            "evidenceBands": {

                "yolo":
                    evidence[
                        "yoloBand"
                    ],

                "vae":
                    evidence[
                        "vaeBand"
                    ],

                "flow":
                    evidence[
                        "flowBand"
                    ]
            },

            "agreement":
                evidence[
                    "agreement"
                ],

            "disagreement":
                evidence[
                    "disagreement"
                ],

            "ambiguity":
                evidence[
                    "ambiguity"
                ],

            "baseEvidence":
                evidence[
                    "baseEvidence"
                ],

            "normalityEvidence":
                evidence[
                    "normalityEvidence"
                ],

            "reviewStatus":
                "PENDING",

            "classId":
                candidate[
                    "classId"
                ],

            "className":
                "anthropogenic"
        })


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    high_count = sum(
        1
        for c in candidate_output
        if c["priorityLevel"]
        == "HIGH"
    )


    medium_count = sum(
        1
        for c in candidate_output
        if c["priorityLevel"]
        == "MEDIUM"
    )


    low_count = sum(
        1
        for c in candidate_output
        if c["priorityLevel"]
        == "LOW"
    )


    uncertain_count = sum(
        1
        for c in candidate_output
        if (
            c["uncertainty"]
            >= 0.40
        )
    )


    processing_time = (
        time.perf_counter()
        - start_time
    )


    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {

        "status":
            "complete",

        "surveyId":
            survey_id,

        "filename":
            original_name,

        "imageUrl":
            f"/uploads/{unique_name}",

        "image": {

            "width":
                width,

            "height":
                height
        },

        "inference": {

            "device":
                DEVICE_STR,

            "gpu":
                (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),

            "processingSeconds":
                round(
                    processing_time,
                    3
                ),

            "breakdown": {

                "yolo":
                    round(
                        yolo_time,
                        3
                    ),

                "normality":
                    round(
                        normality_time,
                        3
                    ),

                "tta":
                    round(
                        tta_time,
                        3
                    )
            }
        },

        "evidence": {

            "engine":
                "SAAD Evidence Engine v3",

            "scoresAreProbabilities":
                False,

            "normalization":
                "validation_normals_only",

            "raw": {

                "yolo":
                    evidence[
                        "yoloRaw"
                    ],

                "vae":
                    evidence[
                        "vaeRaw"
                    ],

                "flow":
                    evidence[
                        "flowRaw"
                    ]
            },

            "normalized": {

                "yolo":
                    evidence[
                        "yoloEvidence"
                    ],

                "vae":
                    evidence[
                        "vaeEvidence"
                    ],

                "flow":
                    evidence[
                        "flowEvidence"
                    ]
            },

            "priority":
                evidence[
                    "priority"
                ],

            "uncertainty":
                evidence[
                    "uncertainty"
                ],

            "agreement":
                evidence[
                    "agreement"
                ],

            "disagreement":
                evidence[
                    "disagreement"
                ],

            "ambiguity":
                evidence[
                    "ambiguity"
                ],

            "profile":
                evidence[
                    "evidenceProfile"
                ],

            "recommendedAction":
                evidence[
                    "recommendedAction"
                ],

            "reason":
                evidence[
                    "evidenceReason"
                ]
        },

        "summary": {

            "detections":
                len(candidate_output),

            "highPriority":
                high_count,

            "review":
                medium_count,

            "uncertain":
                uncertain_count,

            "lowPriority":
                low_count,

            "total":
                len(candidate_output)
        },

        "candidates":
            candidate_output
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "name":
            "SAAD Backend",

        "status":
            "operational",

        "version":
            "0.2.0"
    }