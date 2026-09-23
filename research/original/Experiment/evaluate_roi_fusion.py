# ================================================================
# SAAD — ROI-ALIGNED SPATIAL FUSION
# ================================================================
#
# Evaluates whether VAE + Flow anomaly evidence is spatially
# aligned with YOLO candidate detections.
#
# IMPORTANT:
# This version reproduces the EXACT RealNVP architecture used
# by the trained SAAD flow checkpoint:
#
#   latent dimension : 128
#   coupling half    : 64
#   hidden dimension : 256
#   output           : 128 (64 scale + 64 translation)
#   layers           : 8
#
# No model is retrained.
#
# ================================================================

import os
import csv
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import torch
import torch.nn as nn

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

from ultralytics import YOLO


# ================================================================
# CONFIG
# ================================================================

SEED = 42

DATA_ROOT = Path(
    r"E:\SIH\Datasets\SAAD_baseline"
)

MANIFEST = (
    DATA_ROOT /
    "manifest.csv"
)

YOLO_WEIGHTS = Path(
    r"E:\SIH\SIH_Results\yolo26s_generic_baseline\weights\best.pt"
)

VAE_WEIGHTS = Path(
    r"E:\SIH\SIH_Results\vae_normal_seabed\checkpoints\best.pt"
)

FLOW_WEIGHTS = Path(
    r"E:\SIH\SIH_Results\latent_normalizing_flow\best_flow.pt"
)

LATENT_MEAN = Path(
    r"E:\SIH\SIH_Results\latent_normalizing_flow\latent_mean.pt"
)

LATENT_STD = Path(
    r"E:\SIH\SIH_Results\latent_normalizing_flow\latent_std.pt"
)

OUT_ROOT = Path(
    r"E:\SIH\SIH_Results\roi_spatial_fusion"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

EXAMPLE_ROOT = (
    OUT_ROOT /
    "examples"
)

EXAMPLE_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# YOLO SETTINGS
# ================================================================

YOLO_CONF = 0.001
YOLO_IOU = 0.50
MAX_DETECTIONS = 300


# ================================================================
# PATCH SETTINGS
# ================================================================

PATCH_SIZE = 256
PATCH_STRIDE = 192

BATCH_SIZE = 64


# ================================================================
# ROI SETTINGS
# ================================================================

ROI_EXPANSION = 0.25

PATCH_OVERLAP_MIN = 0.10

TOP_K_PATCHES = 3


# ================================================================
# VISUAL EXAMPLES
# ================================================================

MAX_TP_EXAMPLES = 25
MAX_FP_EXAMPLES = 25


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 72)
print("SAAD — ROI-ALIGNED SPATIAL FUSION")
print("=" * 72)

print(
    f"Device       : {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU          : "
        f"{torch.cuda.get_device_name(0)}"
    )

print()


# ================================================================
# REPRODUCIBILITY
# ================================================================

def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# ================================================================
# METRIC HELPERS
# ================================================================

def safe_auc(
    y_true,
    scores
):

    y_true = np.asarray(
        y_true
    )

    scores = np.asarray(
        scores
    )

    if len(
        np.unique(y_true)
    ) < 2:

        return float("nan")

    return float(
        roc_auc_score(
            y_true,
            scores
        )
    )


def safe_ap(
    y_true,
    scores
):

    y_true = np.asarray(
        y_true
    )

    scores = np.asarray(
        scores
    )

    if len(
        np.unique(y_true)
    ) < 2:

        return float("nan")

    return float(
        average_precision_score(
            y_true,
            scores
        )
    )


def best_f1(
    y_true,
    scores
):

    y_true = np.asarray(
        y_true
    )

    scores = np.asarray(
        scores
    )

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            scores
        )
    )

    if len(thresholds) == 0:

        return {
            "threshold": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }

    f1 = (
        2.0
        *
        precision[:-1]
        *
        recall[:-1]
        /
        np.maximum(
            precision[:-1]
            +
            recall[:-1],
            1e-12
        )
    )

    idx = int(
        np.nanargmax(f1)
    )

    return {
        "threshold":
            float(thresholds[idx]),

        "precision":
            float(precision[idx]),

        "recall":
            float(recall[idx]),

        "f1":
            float(f1[idx]),
    }


def percentile_normalize(
    x,
    low,
    high
):

    x = float(x)

    if high <= low:

        return 0.0

    return float(
        np.clip(
            (x - low)
            /
            (high - low),
            0.0,
            1.0
        )
    )


# ================================================================
# VAE
# ================================================================

class ConvVAE(nn.Module):

    def __init__(
        self,
        latent_dim=128
    ):

        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                256,
                512,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(512),

            nn.ReLU(
                inplace=True
            ),
        )

        self.fc_mu = nn.Linear(
            512 * 8 * 8,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            512 * 8 * 8,
            latent_dim
        )

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

            nn.ReLU(
                inplace=True
            ),

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(
                inplace=True
            ),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(
                inplace=True
            ),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(
                inplace=True
            ),

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

        logvar = torch.clamp(
            logvar,
            -10.0,
            10.0
        )

        return mu, logvar

    def decode(
        self,
        z
    ):

        h = self.fc_decode(z)

        h = h.view(
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

        # Deterministic evaluation.
        # Use mu rather than random sampling.

        recon = self.decode(mu)

        return (
            recon,
            mu,
            logvar
        )


# ================================================================
# EXACT TRAINED REALNVP
# ================================================================

class AffineCoupling(nn.Module):

    def __init__(
        self,
        dim=128,
        hidden_dim=256,
        mask_type=0,
        scale_clamp=1.5
    ):

        super().__init__()

        self.dim = dim

        self.half_dim = (
            dim // 2
        )

        self.scale_clamp = (
            scale_clamp
        )

        # Alternate which half is transformed.

        mask = torch.zeros(
            dim
        )

        if mask_type == 0:

            mask[
                :self.half_dim
            ] = 1.0

        else:

            mask[
                self.half_dim:
            ] = 1.0

        self.register_buffer(
            "mask",
            mask
        )

        # ========================================================
        # EXACT CHECKPOINT ARCHITECTURE
        #
        # 64 -> 256 -> 256 -> 128
        #
        # This matches:
        #
        # layers.0.net.0.weight : [256, 64]
        # layers.0.net.4.weight : [128, 256]
        #
        # ========================================================

        self.net = nn.Sequential(

            nn.Linear(
                self.half_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                self.half_dim * 2
            )
        )

    def forward(
        self,
        x,
        reverse=False
    ):

        x1 = x[
            :,
            :self.half_dim
        ]

        x2 = x[
            :,
            self.half_dim:
        ]

        # --------------------------------------------------------
        # mask_type 0
        #
        # x1 conditions x2
        # --------------------------------------------------------

        if self.mask[0].item() == 1.0:

            condition = x1

            transform = x2

            params = self.net(
                condition
            )

            s, t = torch.chunk(
                params,
                2,
                dim=1
            )

            s = (
                torch.tanh(s)
                *
                self.scale_clamp
            )

            if not reverse:

                transformed = (
                    transform
                    *
                    torch.exp(s)
                    +
                    t
                )

                log_det = s.sum(
                    dim=1
                )

            else:

                transformed = (
                    transform
                    -
                    t
                ) * torch.exp(-s)

                log_det = -s.sum(
                    dim=1
                )

            z = torch.cat(
                [
                    condition,
                    transformed
                ],
                dim=1
            )

            return (
                z,
                log_det
            )

        # --------------------------------------------------------
        # mask_type 1
        #
        # x2 conditions x1
        # --------------------------------------------------------

        else:

            condition = x2

            transform = x1

            params = self.net(
                condition
            )

            s, t = torch.chunk(
                params,
                2,
                dim=1
            )

            s = (
                torch.tanh(s)
                *
                self.scale_clamp
            )

            if not reverse:

                transformed = (
                    transform
                    *
                    torch.exp(s)
                    +
                    t
                )

                log_det = s.sum(
                    dim=1
                )

            else:

                transformed = (
                    transform
                    -
                    t
                ) * torch.exp(-s)

                log_det = -s.sum(
                    dim=1
                )

            z = torch.cat(
                [
                    transformed,
                    condition
                ],
                dim=1
            )

            return (
                z,
                log_det
            )


class RealNVP(nn.Module):

    def __init__(
        self,
        dim=128,
        hidden_dim=256,
        num_layers=8,
        scale_clamp=1.5
    ):

        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(
            num_layers
        ):

            self.layers.append(
                AffineCoupling(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    mask_type=i % 2,
                    scale_clamp=scale_clamp
                )
            )

    def forward(
        self,
        x,
        reverse=False
    ):

        total_log_det = torch.zeros(
            x.size(0),
            device=x.device,
            dtype=x.dtype
        )

        if not reverse:

            z = x

            for layer in self.layers:

                z, log_det = layer(
                    z,
                    reverse=False
                )

                total_log_det += log_det

            return (
                z,
                total_log_det
            )

        else:

            z = x

            for layer in reversed(
                self.layers
            ):

                z, log_det = layer(
                    z,
                    reverse=True
                )

                total_log_det += log_det

            return (
                z,
                total_log_det
            )

    def log_prob(
        self,
        x
    ):

        z, log_det = self.forward(
            x,
            reverse=False
        )

        log_base = (
            -0.5
            *
            (
                z ** 2
                +
                math.log(
                    2.0 * math.pi
                )
            )
        ).sum(
            dim=1
        )

        return (
            log_base
            +
            log_det
        )

    def nll(
        self,
        x
    ):

        return -self.log_prob(x)


# ================================================================
# LOAD YOLO
# ================================================================

print(
    "Loading YOLO..."
)

yolo = YOLO(
    str(YOLO_WEIGHTS)
)

print(
    "YOLO loaded."
)

print()


# ================================================================
# LOAD VAE
# ================================================================

print(
    "Loading VAE..."
)

vae = ConvVAE(
    latent_dim=128
)

vae_checkpoint = torch.load(
    VAE_WEIGHTS,
    map_location=DEVICE
)

if (
    isinstance(
        vae_checkpoint,
        dict
    )
    and
    "model_state_dict"
    in vae_checkpoint
):

    vae_state = (
        vae_checkpoint[
            "model_state_dict"
        ]
    )

elif (
    isinstance(
        vae_checkpoint,
        dict
    )
    and
    "state_dict"
    in vae_checkpoint
):

    vae_state = (
        vae_checkpoint[
            "state_dict"
        ]
    )

else:

    vae_state = vae_checkpoint

vae.load_state_dict(
    vae_state,
    strict=True
)

vae = vae.to(
    DEVICE
)

vae.eval()

print(
    "VAE checkpoint compatibility: PASSED"
)

print()


# ================================================================
# LOAD FLOW
# ================================================================

print(
    "Loading RealNVP..."
)

flow = RealNVP(
    dim=128,
    hidden_dim=256,
    num_layers=8,
    scale_clamp=1.5
)

flow_checkpoint = torch.load(
    FLOW_WEIGHTS,
    map_location=DEVICE
)

if (
    isinstance(
        flow_checkpoint,
        dict
    )
    and
    "model_state_dict"
    in flow_checkpoint
):

    flow_state = (
        flow_checkpoint[
            "model_state_dict"
        ]
    )

elif (
    isinstance(
        flow_checkpoint,
        dict
    )
    and
    "state_dict"
    in flow_checkpoint
):

    flow_state = (
        flow_checkpoint[
            "state_dict"
        ]
    )

else:

    flow_state = flow_checkpoint


# ------------------------------------------------
# Explicit architecture sanity check
# ------------------------------------------------

expected_shape_1 = (
    256,
    64
)

expected_shape_2 = (
    128,
    256
)

actual_shape_1 = tuple(
    flow_state[
        "layers.0.net.0.weight"
    ].shape
)

actual_shape_2 = tuple(
    flow_state[
        "layers.0.net.4.weight"
    ].shape
)

print(
    "Checkpoint architecture:"
)

print(
    f"  layers.0.net.0.weight : "
    f"{actual_shape_1}"
)

print(
    f"  layers.0.net.4.weight : "
    f"{actual_shape_2}"
)

if actual_shape_1 != expected_shape_1:

    raise RuntimeError(
        "Unexpected RealNVP checkpoint architecture. "
        f"Expected {expected_shape_1}, "
        f"got {actual_shape_1}."
    )

if actual_shape_2 != expected_shape_2:

    raise RuntimeError(
        "Unexpected RealNVP checkpoint architecture. "
        f"Expected {expected_shape_2}, "
        f"got {actual_shape_2}."
    )


flow.load_state_dict(
    flow_state,
    strict=True
)

flow = flow.to(
    DEVICE
)

flow.eval()

flow_parameter_count = sum(
    p.numel()
    for p in flow.parameters()
)

print(
    "RealNVP checkpoint compatibility: PASSED"
)

print(
    f"Flow parameters: "
    f"{flow_parameter_count:,}"
)

if flow_parameter_count != 922624:

    raise RuntimeError(
        "Unexpected flow parameter count. "
        f"Expected 922624, "
        f"got {flow_parameter_count}."
    )

print()


# ================================================================
# LOAD LATENT NORMALIZATION
# ================================================================

latent_mean = torch.load(
    LATENT_MEAN,
    map_location=DEVICE
).float().to(
    DEVICE
)

latent_std = torch.load(
    LATENT_STD,
    map_location=DEVICE
).float().to(
    DEVICE
)

latent_std = torch.clamp(
    latent_std,
    min=1e-6
)

print(
    f"Latent mean shape : "
    f"{tuple(latent_mean.shape)}"
)

print(
    f"Latent std shape  : "
    f"{tuple(latent_std.shape)}"
)

print()


# ================================================================
# LOAD MANIFEST
# ================================================================

print(
    "Loading manifest..."
)

manifest = pd.read_csv(
    MANIFEST
)

required_columns = [
    "dataset",
    "original_path",
    "output_image",
    "output_label",
    "split",
    "group",
    "source_class",
    "saad_class",
    "has_annotation",
    "num_objects",
    "annotation_type"
]

missing = [
    c
    for c in required_columns
    if c not in manifest.columns
]

if missing:

    raise RuntimeError(
        f"Manifest missing columns: {missing}"
    )

test_df = manifest[
    manifest[
        "split"
    ].astype(str).str.lower()
    ==
    "test"
].copy()

print(
    f"Test images in manifest : "
    f"{len(test_df)}"
)

print(
    test_df[
        "dataset"
    ].value_counts()
)

print()


# ================================================================
# VAE NORMAL DATASET
# ================================================================

VAE_ROOT = Path(
    r"E:\SIH\Datasets\SAAD_VAE"
)

VAE_VAL_DIR = (
    VAE_ROOT /
    "val"
)

if not VAE_VAL_DIR.exists():

    raise FileNotFoundError(
        "VAE validation directory not found:\n"
        f"{VAE_VAL_DIR}"
    )


# ================================================================
# PATCH POSITIONS
# ================================================================

def patch_positions(
    width,
    height,
    size=256,
    stride=192
):

    xs = list(
        range(
            0,
            max(
                width - size,
                0
            ) + 1,
            stride
        )
    )

    ys = list(
        range(
            0,
            max(
                height - size,
                0
            ) + 1,
            stride
        )
    )

    if len(xs) == 0:

        xs = [0]

    if len(ys) == 0:

        ys = [0]

    last_x = max(
        width - size,
        0
    )

    last_y = max(
        height - size,
        0
    )

    if xs[-1] != last_x:

        xs.append(
            last_x
        )

    if ys[-1] != last_y:

        ys.append(
            last_y
        )

    return [
        (x, y)
        for y in ys
        for x in xs
    ]


# ================================================================
# PATCH RECT
# ================================================================

def patch_rect(
    x,
    y,
    size=256
):

    return (
        x,
        y,
        x + size,
        y + size
    )


# ================================================================
# INTERSECTION
# ================================================================

def intersection_area(
    a,
    b
):

    ax1, ay1, ax2, ay2 = a

    bx1, by1, bx2, by2 = b

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

    w = max(
        0.0,
        ix2 - ix1
    )

    h = max(
        0.0,
        iy2 - iy1
    )

    return (
        w * h
    )


# ================================================================
# BOX EXPANSION
# ================================================================

def expand_box(
    box,
    width,
    height,
    expansion=0.25
):

    x1, y1, x2, y2 = box

    bw = max(
        1.0,
        x2 - x1
    )

    bh = max(
        1.0,
        y2 - y1
    )

    dx = (
        bw * expansion
    )

    dy = (
        bh * expansion
    )

    x1 = max(
        0.0,
        x1 - dx
    )

    y1 = max(
        0.0,
        y1 - dy
    )

    x2 = min(
        float(width),
        x2 + dx
    )

    y2 = min(
        float(height),
        y2 + dy
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# ================================================================
# SCORE NORMAL VALIDATION PATCHES
# ================================================================

@torch.no_grad()
def collect_normal_validation_scores():

    files = []

    for pattern in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG"
    ]:

        files.extend(
            VAE_VAL_DIR.glob(
                pattern
            )
        )

    files = sorted(
        files
    )

    print(
        f"Normal validation patches : "
        f"{len(files)}"
    )

    vae_scores = []
    flow_scores = []

    for idx, path in enumerate(
        files
    ):

        try:

            img = Image.open(
                path
            ).convert(
                "L"
            )

            arr = np.asarray(
                img,
                dtype=np.float32
            )

            arr /= 255.0

            arr = np.nan_to_num(
                arr,
                nan=0.0,
                posinf=1.0,
                neginf=0.0
            )

            arr = np.clip(
                arr,
                0.0,
                1.0
            )

            tensor = (
                torch.from_numpy(
                    arr
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .to(DEVICE)
            )

            recon, mu, _ = vae(
                tensor
            )

            mse = (
                (recon - tensor) ** 2
            ).flatten(
                1
            ).mean(
                dim=1
            )[0]

            z = (
                mu - latent_mean
            ) / latent_std

            nll = flow.nll(
                z
            )[0]

            vae_scores.append(
                float(
                    mse.item()
                )
            )

            flow_scores.append(
                float(
                    nll.item()
                )
            )

        except Exception as e:

            warnings.warn(
                f"Failed validation patch "
                f"{path}: {e}"
            )

        if (
            idx + 1
        ) % 1000 == 0:

            print(
                f"  processed "
                f"{idx + 1}/"
                f"{len(files)}"
            )

    return (
        np.asarray(
            vae_scores,
            dtype=np.float32
        ),
        np.asarray(
            flow_scores,
            dtype=np.float32
        )
    )


# ================================================================
# BUILD NORMAL VALIDATION DISTRIBUTIONS
# ================================================================

print("=" * 72)
print(
    "BUILDING NORMAL VALIDATION "
    "SCORE DISTRIBUTIONS"
)
print("=" * 72)

normal_val_vae, normal_val_flow = (
    collect_normal_validation_scores()
)

if len(
    normal_val_vae
) < 100:

    raise RuntimeError(
        "Too few normal validation patches."
    )

VAE_P5 = float(
    np.percentile(
        normal_val_vae,
        5
    )
)

VAE_P95 = float(
    np.percentile(
        normal_val_vae,
        95
    )
)

FLOW_P5 = float(
    np.percentile(
        normal_val_flow,
        5
    )
)

FLOW_P95 = float(
    np.percentile(
        normal_val_flow,
        95
    )
)

print()

print(
    f"Normal VAE P5  : "
    f"{VAE_P5:.8f}"
)

print(
    f"Normal VAE P95 : "
    f"{VAE_P95:.8f}"
)

print(
    f"Normal Flow P5  : "
    f"{FLOW_P5:.6f}"
)

print(
    f"Normal Flow P95 : "
    f"{FLOW_P95:.6f}"
)

print()


# ================================================================
# SCORE IMAGE PATCHES
# ================================================================

@torch.no_grad()
def score_patches(
    image,
    positions
):

    patches = []

    valid_positions = []

    width, height = image.size

    for x, y in positions:

        crop = image.crop(
            (
                x,
                y,
                min(
                    x + PATCH_SIZE,
                    width
                ),
                min(
                    y + PATCH_SIZE,
                    height
                )
            )
        )

        if crop.size != (
            PATCH_SIZE,
            PATCH_SIZE
        ):

            padded = Image.new(
                "L",
                (
                    PATCH_SIZE,
                    PATCH_SIZE
                ),
                0
            )

            padded.paste(
                crop,
                (0, 0)
            )

            crop = padded

        arr = np.asarray(
            crop,
            dtype=np.float32
        )

        arr /= 255.0

        arr = np.nan_to_num(
            arr,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        arr = np.clip(
            arr,
            0.0,
            1.0
        )

        patches.append(
            torch.from_numpy(
                arr
            ).unsqueeze(0)
        )

        valid_positions.append(
            (x, y)
        )

    if not patches:

        return []

    patches = torch.stack(
        patches,
        dim=0
    ).to(
        DEVICE,
        non_blocking=True
    )

    results = []

    for start in range(
        0,
        len(patches),
        BATCH_SIZE
    ):

        batch = patches[
            start:
            start + BATCH_SIZE
        ]

        recon, mu, _ = vae(
            batch
        )

        mse = (
            (recon - batch) ** 2
        ).flatten(
            1
        ).mean(
            dim=1
        )

        z = (
            mu - latent_mean
        ) / latent_std

        nll = flow.nll(
            z
        )

        mse_np = (
            mse
            .detach()
            .cpu()
            .numpy()
        )

        nll_np = (
            nll
            .detach()
            .cpu()
            .numpy()
        )

        for i in range(
            len(batch)
        ):

            x, y = (
                valid_positions[
                    start + i
                ]
            )

            results.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "vae": float(
                        mse_np[i]
                    ),
                    "flow": float(
                        nll_np[i]
                    )
                }
            )

    return results


# ================================================================
# LOAD YOLO GT
# ================================================================

def load_yolo_gt(
    label_path,
    width,
    height
):

    boxes = []

    if not label_path.exists():

        return boxes

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:

                continue

            try:

                xc = float(
                    parts[1]
                )

                yc = float(
                    parts[2]
                )

                bw = float(
                    parts[3]
                )

                bh = float(
                    parts[4]
                )

            except Exception:

                continue

            x1 = (
                xc - bw / 2.0
            ) * width

            y1 = (
                yc - bh / 2.0
            ) * height

            x2 = (
                xc + bw / 2.0
            ) * width

            y2 = (
                yc + bh / 2.0
            ) * height

            x1 = np.clip(
                x1,
                0,
                width
            )

            y1 = np.clip(
                y1,
                0,
                height
            )

            x2 = np.clip(
                x2,
                0,
                width
            )

            y2 = np.clip(
                y2,
                0,
                height
            )

            boxes.append(
                (
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2)
                )
            )

    return boxes


# ================================================================
# YOLO PREDICTIONS
# ================================================================

def get_predictions(
    image_path
):

    result = yolo.predict(
        source=str(
            image_path
        ),
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        max_det=MAX_DETECTIONS,
        verbose=False,
        device=(
            0
            if torch.cuda.is_available()
            else "cpu"
        )
    )[0]

    predictions = []

    if result.boxes is None:

        return predictions

    boxes = (
        result.boxes.xyxy
        .detach()
        .cpu()
        .numpy()
    )

    confs = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    for box, conf in zip(
        boxes,
        confs
    ):

        predictions.append(
            {
                "box": tuple(
                    float(v)
                    for v in box
                ),
                "conf": float(
                    conf
                )
            }
        )

    return predictions


# ================================================================
# MATCH PREDICTION TO GT
# ================================================================

def box_iou(
    a,
    b
):

    inter = intersection_area(
        a,
        b
    )

    area_a = (
        max(
            0.0,
            a[2] - a[0]
        )
        *
        max(
            0.0,
            a[3] - a[1]
        )
    )

    area_b = (
        max(
            0.0,
            b[2] - b[0]
        )
        *
        max(
            0.0,
            b[3] - b[1]
        )
    )

    union = (
        area_a
        +
        area_b
        -
        inter
    )

    if union <= 0:

        return 0.0

    return (
        inter /
        union
    )


def classify_prediction(
    pred_box,
    gt_boxes,
    matched_gt
):

    if len(gt_boxes) == 0:

        return (
            0,
            -1,
            0.0
        )

    best_iou = 0.0
    best_idx = -1

    for i, gt in enumerate(
        gt_boxes
    ):

        iou = box_iou(
            pred_box,
            gt
        )

        if iou > best_iou:

            best_iou = iou
            best_idx = i

    if (
        best_iou >= YOLO_IOU
        and
        best_idx not in matched_gt
    ):

        return (
            1,
            best_idx,
            best_iou
        )

    return (
        0,
        best_idx,
        best_iou
    )


# ================================================================
# ROI SCORE AGGREGATION
# ================================================================

def aggregate_roi_scores(
    candidate_box,
    patch_scores,
    image_width,
    image_height
):

    roi = expand_box(
        candidate_box,
        image_width,
        image_height,
        ROI_EXPANSION
    )

    selected = []

    for p in patch_scores:

        patch = patch_rect(
            p["x"],
            p["y"],
            PATCH_SIZE
        )

        overlap = (
            intersection_area(
                roi,
                patch
            )
        )

        overlap_fraction = (
            overlap
            /
            float(
                PATCH_SIZE
                *
                PATCH_SIZE
            )
        )

        if (
            overlap_fraction
            >= PATCH_OVERLAP_MIN
        ):

            q = dict(p)

            q[
                "overlap"
            ] = overlap_fraction

            selected.append(q)

    if not selected:

        return {
            "roi_vae_max": 0.0,
            "roi_vae_mean": 0.0,
            "roi_vae_topk": 0.0,

            "roi_flow_max": 0.0,
            "roi_flow_mean": 0.0,
            "roi_flow_topk": 0.0,

            "roi_vae_norm": 0.0,
            "roi_flow_norm": 0.0,

            "roi_agreement": 0.0,

            "num_roi_patches": 0,

            "roi": roi
        }

    vae_values = np.asarray(
        [
            p["vae"]
            for p in selected
        ],
        dtype=np.float32
    )

    flow_values = np.asarray(
        [
            p["flow"]
            for p in selected
        ],
        dtype=np.float32
    )

    vae_sorted = np.sort(
        vae_values
    )[::-1]

    flow_sorted = np.sort(
        flow_values
    )[::-1]

    k = min(
        TOP_K_PATCHES,
        len(selected)
    )

    vae_topk = float(
        np.mean(
            vae_sorted[:k]
        )
    )

    flow_topk = float(
        np.mean(
            flow_sorted[:k]
        )
    )

    vae_max = float(
        np.max(
            vae_values
        )
    )

    flow_max = float(
        np.max(
            flow_values
        )
    )

    vae_mean = float(
        np.mean(
            vae_values
        )
    )

    flow_mean = float(
        np.mean(
            flow_values
        )
    )

    vae_norm = percentile_normalize(
        vae_topk,
        VAE_P5,
        VAE_P95
    )

    flow_norm = percentile_normalize(
        flow_topk,
        FLOW_P5,
        FLOW_P95
    )

    agreement = math.sqrt(
        max(
            0.0,
            vae_norm
        )
        *
        max(
            0.0,
            flow_norm
        )
    )

    return {
        "roi_vae_max":
            vae_max,

        "roi_vae_mean":
            vae_mean,

        "roi_vae_topk":
            vae_topk,

        "roi_flow_max":
            flow_max,

        "roi_flow_mean":
            flow_mean,

        "roi_flow_topk":
            flow_topk,

        "roi_vae_norm":
            vae_norm,

        "roi_flow_norm":
            flow_norm,

        "roi_agreement":
            agreement,

        "num_roi_patches":
            len(selected),

        "roi":
            roi
    }


# ================================================================
# SAVE VISUAL EXAMPLE
# ================================================================

def save_candidate_example(
    image_path,
    candidate,
    patch_scores,
    label,
    index,
    dataset_name
):

    try:

        img = Image.open(
            image_path
        ).convert(
            "L"
        )

        rgb = img.convert(
            "RGB"
        )

        draw = ImageDraw.Draw(
            rgb
        )

        box = candidate[
            "box"
        ]

        x1, y1, x2, y2 = (
            int(v)
            for v in box
        )

        roi = expand_box(
            box,
            img.width,
            img.height,
            ROI_EXPANSION
        )

        rx1, ry1, rx2, ry2 = (
            int(v)
            for v in roi
        )

        # Red = YOLO candidate

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2
            ],
            outline=(255, 0, 0),
            width=4
        )

        # Yellow = expanded ROI

        draw.rectangle(
            [
                rx1,
                ry1,
                rx2,
                ry2
            ],
            outline=(255, 255, 0),
            width=2
        )

        # Green = anomaly patches contributing
        # to ROI score.

        for p in patch_scores:

            patch = patch_rect(
                p["x"],
                p["y"],
                PATCH_SIZE
            )

            overlap = intersection_area(
                roi,
                patch
            )

            overlap_fraction = (
                overlap
                /
                float(
                    PATCH_SIZE
                    *
                    PATCH_SIZE
                )
            )

            if (
                overlap_fraction
                >= PATCH_OVERLAP_MIN
            ):

                px1, py1, px2, py2 = (
                    int(patch[0]),
                    int(patch[1]),
                    int(patch[2]),
                    int(patch[3])
                )

                draw.rectangle(
                    [
                        px1,
                        py1,
                        px2,
                        py2
                    ],
                    outline=(0, 255, 0),
                    width=2
                )

        safe_dataset = (
            str(dataset_name)
            .replace(
                " ",
                "_"
            )
            .replace(
                "/",
                "_"
            )
            .replace(
                "\\",
                "_"
            )
        )

        filename = (
            f"{label}_"
            f"{safe_dataset}_"
            f"{index:05d}.jpg"
        )

        rgb.save(
            EXAMPLE_ROOT /
            filename,
            quality=92
        )

    except Exception as e:

        warnings.warn(
            f"Could not save example: "
            f"{e}"
        )


# ================================================================
# MAIN EVALUATION
# ================================================================

all_rows = []

tp_example_count = 0
fp_example_count = 0

image_counter = 0

print("=" * 72)
print(
    "RUNNING ROI-ALIGNED EVALUATION"
)
print("=" * 72)
print()

for _, row in test_df.iterrows():

    image_path = Path(
        str(
            row[
                "output_image"
            ]
        )
    )

    if not image_path.exists():

        print(
            "[WARNING] Missing image: "
            f"{image_path}"
        )

        continue

    try:

        image = Image.open(
            image_path
        ).convert(
            "L"
        )

    except Exception as e:

        print(
            "[WARNING] Could not open "
            f"{image_path}: {e}"
        )

        continue

    width, height = image.size

    label_path = Path(
        str(
            row[
                "output_label"
            ]
        )
    )

    gt_boxes = load_yolo_gt(
        label_path,
        width,
        height
    )

    predictions = get_predictions(
        image_path
    )

    # ------------------------------------------------------------
    # Calculate anomaly patches once per image.
    # ------------------------------------------------------------

    positions = patch_positions(
        width,
        height,
        PATCH_SIZE,
        PATCH_STRIDE
    )

    patch_scores = score_patches(
        image,
        positions
    )

    # ------------------------------------------------------------
    # Global anomaly values.
    #
    # Kept only to compare against the old global-max approach.
    # ------------------------------------------------------------

    if patch_scores:

        global_vae = max(
            p["vae"]
            for p in patch_scores
        )

        global_flow = max(
            p["flow"]
            for p in patch_scores
        )

        global_vae_norm = (
            percentile_normalize(
                global_vae,
                VAE_P5,
                VAE_P95
            )
        )

        global_flow_norm = (
            percentile_normalize(
                global_flow,
                FLOW_P5,
                FLOW_P95
            )
        )

    else:

        global_vae = 0.0
        global_flow = 0.0

        global_vae_norm = 0.0
        global_flow_norm = 0.0

    matched_gt = set()

    for pred_idx, pred in enumerate(
        predictions
    ):

        pred_box = pred[
            "box"
        ]

        yolo_conf = pred[
            "conf"
        ]

        is_tp, gt_idx, best_iou = (
            classify_prediction(
                pred_box,
                gt_boxes,
                matched_gt
            )
        )

        if is_tp:

            matched_gt.add(
                gt_idx
            )

        roi_scores = (
            aggregate_roi_scores(
                pred_box,
                patch_scores,
                width,
                height
            )
        )

        roi_vae = (
            roi_scores[
                "roi_vae_norm"
            ]
        )

        roi_flow = (
            roi_scores[
                "roi_flow_norm"
            ]
        )

        roi_agreement = (
            roi_scores[
                "roi_agreement"
            ]
        )

        # --------------------------------------------------------
        # ROI FUSIONS
        # --------------------------------------------------------

        fusion_yolo_vae = (
            yolo_conf
            +
            roi_vae
        ) / 2.0

        fusion_yolo_flow = (
            yolo_conf
            +
            roi_flow
        ) / 2.0

        fusion_vae_flow = (
            roi_vae
            +
            roi_flow
        ) / 2.0

        fusion_yolo_vae_flow = (
            yolo_conf
            +
            roi_vae
            +
            roi_flow
        ) / 3.0

        # Agreement-enhanced diagnostic score.

        fusion_agreement = (
            0.50 * yolo_conf
            +
            0.25 * roi_vae
            +
            0.25 * roi_flow
        )

        # --------------------------------------------------------
        # OLD GLOBAL FUSION
        # --------------------------------------------------------

        fusion_global = (
            yolo_conf
            +
            global_vae_norm
            +
            global_flow_norm
        ) / 3.0

        all_rows.append(
            {
                "image_path":
                    str(image_path),

                "dataset":
                    str(row["dataset"]),

                "group":
                    str(row["group"]),

                "prediction_index":
                    pred_idx,

                "x1":
                    pred_box[0],

                "y1":
                    pred_box[1],

                "x2":
                    pred_box[2],

                "y2":
                    pred_box[3],

                "yolo_conf":
                    yolo_conf,

                "gt_index":
                    gt_idx,

                "best_iou":
                    best_iou,

                "is_tp":
                    int(is_tp),

                "roi_vae_max":
                    roi_scores[
                        "roi_vae_max"
                    ],

                "roi_vae_mean":
                    roi_scores[
                        "roi_vae_mean"
                    ],

                "roi_vae_topk":
                    roi_scores[
                        "roi_vae_topk"
                    ],

                "roi_flow_max":
                    roi_scores[
                        "roi_flow_max"
                    ],

                "roi_flow_mean":
                    roi_scores[
                        "roi_flow_mean"
                    ],

                "roi_flow_topk":
                    roi_scores[
                        "roi_flow_topk"
                    ],

                "roi_vae_norm":
                    roi_vae,

                "roi_flow_norm":
                    roi_flow,

                "roi_agreement":
                    roi_agreement,

                "num_roi_patches":
                    roi_scores[
                        "num_roi_patches"
                    ],

                "global_vae":
                    global_vae,

                "global_flow":
                    global_flow,

                "global_vae_norm":
                    global_vae_norm,

                "global_flow_norm":
                    global_flow_norm,

                "fusion_yolo_vae":
                    fusion_yolo_vae,

                "fusion_yolo_flow":
                    fusion_yolo_flow,

                "fusion_vae_flow":
                    fusion_vae_flow,

                "fusion_yolo_vae_flow":
                    fusion_yolo_vae_flow,

                "fusion_agreement":
                    fusion_agreement,

                "fusion_global":
                    fusion_global,
            }
        )

        # --------------------------------------------------------
        # Save examples
        # --------------------------------------------------------

        if (
            is_tp
            and
            tp_example_count
            <
            MAX_TP_EXAMPLES
        ):

            save_candidate_example(
                image_path,
                pred,
                patch_scores,
                "TP",
                tp_example_count,
                row["dataset"]
            )

            tp_example_count += 1

        elif (
            not is_tp
            and
            fp_example_count
            <
            MAX_FP_EXAMPLES
        ):

            save_candidate_example(
                image_path,
                pred,
                patch_scores,
                "FP",
                fp_example_count,
                row["dataset"]
            )

            fp_example_count += 1

    image_counter += 1

    if image_counter % 25 == 0:

        print(
            f"Processed "
            f"{image_counter}/"
            f"{len(test_df)} images | "
            f"candidates="
            f"{len(all_rows)}"
        )


# ================================================================
# CHECK RESULTS
# ================================================================

if not all_rows:

    raise RuntimeError(
        "No YOLO candidate predictions were produced."
    )

df = pd.DataFrame(
    all_rows
)


# ================================================================
# SAVE CANDIDATE RESULTS
# ================================================================

candidate_csv = (
    OUT_ROOT /
    "roi_candidate_scores.csv"
)

df.to_csv(
    candidate_csv,
    index=False
)

print()
print(
    f"Candidate results saved: "
    f"{candidate_csv}"
)


# ================================================================
# SIGNAL DEFINITIONS
# ================================================================

signals = {

    "YOLO":
        "yolo_conf",

    "VAE_ROI":
        "roi_vae_norm",

    "Flow_ROI":
        "roi_flow_norm",

    "VAE+Flow_ROI":
        "fusion_vae_flow",

    "YOLO+VAE_ROI":
        "fusion_yolo_vae",

    "YOLO+Flow_ROI":
        "fusion_yolo_flow",

    "YOLO+VAE+Flow_ROI":
        "fusion_yolo_vae_flow",

    "YOLO+VAE+Flow+Agreement":
        "fusion_agreement",

    "OLD_GLOBAL_FUSION":
        "fusion_global",
}


# ================================================================
# METRICS
# ================================================================

metric_rows = []

y_true = df[
    "is_tp"
].values

for name, column in signals.items():

    scores = df[
        column
    ].values

    auc = safe_auc(
        y_true,
        scores
    )

    ap = safe_ap(
        y_true,
        scores
    )

    f1 = best_f1(
        y_true,
        scores
    )

    metric_rows.append(
        {
            "signal":
                name,

            "score_column":
                column,

            "ROC_AUC":
                auc,

            "PR_AUC":
                ap,

            "Best_F1":
                f1["f1"],

            "Best_F1_precision":
                f1["precision"],

            "Best_F1_recall":
                f1["recall"],

            "Best_F1_threshold":
                f1["threshold"],
        }
    )


metrics_df = pd.DataFrame(
    metric_rows
)

metrics_csv = (
    OUT_ROOT /
    "roi_fusion_metrics.csv"
)

metrics_df.to_csv(
    metrics_csv,
    index=False
)


# ================================================================
# DATASET-WISE RESULTS
# ================================================================

dataset_rows = []

for dataset, group in df.groupby(
    "dataset"
):

    yt = group[
        "is_tp"
    ].values

    for name, column in signals.items():

        scores = group[
            column
        ].values

        tp_values = group.loc[
            group["is_tp"] == 1,
            column
        ].values

        fp_values = group.loc[
            group["is_tp"] == 0,
            column
        ].values

        dataset_rows.append(
            {
                "dataset":
                    dataset,

                "signal":
                    name,

                "N_predictions":
                    len(group),

                "TP":
                    int(
                        group[
                            "is_tp"
                        ].sum()
                    ),

                "FP":
                    int(
                        (
                            1 -
                            group[
                                "is_tp"
                            ]
                        ).sum()
                    ),

                "ROC_AUC":
                    safe_auc(
                        yt,
                        scores
                    ),

                "PR_AUC":
                    safe_ap(
                        yt,
                        scores
                    ),

                "TP_mean":
                    float(
                        np.mean(
                            tp_values
                        )
                    )
                    if len(tp_values)
                    else float("nan"),

                "TP_median":
                    float(
                        np.median(
                            tp_values
                        )
                    )
                    if len(tp_values)
                    else float("nan"),

                "FP_mean":
                    float(
                        np.mean(
                            fp_values
                        )
                    )
                    if len(fp_values)
                    else float("nan"),

                "FP_median":
                    float(
                        np.median(
                            fp_values
                        )
                    )
                    if len(fp_values)
                    else float("nan"),
            }
        )


dataset_df = pd.DataFrame(
    dataset_rows
)

dataset_csv = (
    OUT_ROOT /
    "dataset_wise_roi_results.csv"
)

dataset_df.to_csv(
    dataset_csv,
    index=False
)


# ================================================================
# ROI VS GLOBAL COMPARISON
# ================================================================

comparison_specs = [

    (
        "VAE",
        "roi_vae_norm",
        "global_vae_norm"
    ),

    (
        "Flow",
        "roi_flow_norm",
        "global_flow_norm"
    ),

    (
        "VAE+Flow",
        "fusion_vae_flow",
        "fusion_global"
    ),

    (
        "YOLO+VAE+Flow",
        "fusion_yolo_vae_flow",
        "fusion_global"
    ),
]

comparison_rows = []

for (
    name,
    roi_column,
    global_column
) in comparison_specs:

    comparison_rows.append(
        {
            "signal":
                name,

            "ROI_ROC_AUC":
                safe_auc(
                    y_true,
                    df[
                        roi_column
                    ].values
                ),

            "GLOBAL_ROC_AUC":
                safe_auc(
                    y_true,
                    df[
                        global_column
                    ].values
                ),

            "ROI_PR_AUC":
                safe_ap(
                    y_true,
                    df[
                        roi_column
                    ].values
                ),

            "GLOBAL_PR_AUC":
                safe_ap(
                    y_true,
                    df[
                        global_column
                    ].values
                ),

            "ROI_TP_MEAN":
                float(
                    df.loc[
                        df["is_tp"] == 1,
                        roi_column
                    ].mean()
                ),

            "ROI_FP_MEAN":
                float(
                    df.loc[
                        df["is_tp"] == 0,
                        roi_column
                    ].mean()
                ),

            "GLOBAL_TP_MEAN":
                float(
                    df.loc[
                        df["is_tp"] == 1,
                        global_column
                    ].mean()
                ),

            "GLOBAL_FP_MEAN":
                float(
                    df.loc[
                        df["is_tp"] == 0,
                        global_column
                    ].mean()
                ),
        }
    )


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_csv = (
    OUT_ROOT /
    "roi_vs_global_comparison.csv"
)

comparison_df.to_csv(
    comparison_csv,
    index=False
)


# ================================================================
# TP / FP DISTRIBUTIONS
# ================================================================

distribution_rows = []

for name, column in signals.items():

    tp = df.loc[
        df["is_tp"] == 1,
        column
    ].values

    fp = df.loc[
        df["is_tp"] == 0,
        column
    ].values

    distribution_rows.append(
        {
            "signal":
                name,

            "TP_N":
                len(tp),

            "FP_N":
                len(fp),

            "TP_mean":
                float(
                    np.mean(tp)
                )
                if len(tp)
                else float("nan"),

            "TP_median":
                float(
                    np.median(tp)
                )
                if len(tp)
                else float("nan"),

            "TP_P90":
                float(
                    np.percentile(
                        tp,
                        90
                    )
                )
                if len(tp)
                else float("nan"),

            "TP_P95":
                float(
                    np.percentile(
                        tp,
                        95
                    )
                )
                if len(tp)
                else float("nan"),

            "FP_mean":
                float(
                    np.mean(fp)
                )
                if len(fp)
                else float("nan"),

            "FP_median":
                float(
                    np.median(fp)
                )
                if len(fp)
                else float("nan"),

            "FP_P90":
                float(
                    np.percentile(
                        fp,
                        90
                    )
                )
                if len(fp)
                else float("nan"),

            "FP_P95":
                float(
                    np.percentile(
                        fp,
                        95
                    )
                )
                if len(fp)
                else float("nan"),
        }
    )


distribution_df = pd.DataFrame(
    distribution_rows
)

distribution_df.to_csv(
    OUT_ROOT /
    "tp_fp_distributions.csv",
    index=False
)


# ================================================================
# PRINT MAIN RESULTS
# ================================================================

print()
print("=" * 72)
print(
    "ROI-ALIGNED SAAD RESULTS"
)
print("=" * 72)

print(
    metrics_df[
        [
            "signal",
            "ROC_AUC",
            "PR_AUC",
            "Best_F1",
            "Best_F1_precision",
            "Best_F1_recall"
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


# ================================================================
# PRINT ROI VS GLOBAL
# ================================================================

print()
print("=" * 72)
print(
    "ROI VS GLOBAL"
)
print("=" * 72)

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


# ================================================================
# PRINT DATASET-WISE
# ================================================================

print()
print("=" * 72)
print(
    "DATASET-WISE ROI RESULTS"
)
print("=" * 72)

for dataset in sorted(
    dataset_df[
        "dataset"
    ].unique()
):

    print()
    print(
        f"--- {dataset} ---"
    )

    sub = dataset_df[
        dataset_df[
            "dataset"
        ]
        ==
        dataset
    ]

    print(
        sub[
            [
                "signal",
                "ROC_AUC",
                "PR_AUC",
                "TP_mean",
                "TP_median",
                "FP_mean",
                "FP_median"
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )


# ================================================================
# BEST SIGNAL
# ================================================================

best_idx = (
    metrics_df[
        "PR_AUC"
    ].idxmax()
)

best_row = (
    metrics_df.iloc[
        best_idx
    ]
)


# ================================================================
# SAVE SUMMARY
# ================================================================

summary_path = (
    OUT_ROOT /
    "roi_fusion_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SAAD ROI-ALIGNED SPATIAL FUSION\n"
    )

    f.write(
        "=" * 72
        +
        "\n\n"
    )

    f.write(
        "Purpose\n"
    )

    f.write(
        "-------\n"
    )

    f.write(
        "Evaluate whether VAE and latent-flow "
        "anomaly evidence is spatially aligned "
        "with YOLO candidate detections.\n\n"
    )

    f.write(
        "This experiment compares the previous "
        "image-level/global anomaly fusion against "
        "candidate-level ROI-aligned fusion.\n\n"
    )

    f.write(
        "Configuration\n"
    )

    f.write(
        "-------------\n"
    )

    f.write(
        f"YOLO confidence : {YOLO_CONF}\n"
    )

    f.write(
        f"YOLO IoU        : {YOLO_IOU}\n"
    )

    f.write(
        f"Patch size      : {PATCH_SIZE}\n"
    )

    f.write(
        f"Patch stride    : {PATCH_STRIDE}\n"
    )

    f.write(
        f"ROI expansion   : {ROI_EXPANSION}\n"
    )

    f.write(
        f"Patch overlap   : {PATCH_OVERLAP_MIN}\n"
    )

    f.write(
        f"Top-K patches   : {TOP_K_PATCHES}\n\n"
    )

    f.write(
        f"Total candidates : "
        f"{len(df)}\n"
    )

    f.write(
        f"TP candidates    : "
        f"{int(df['is_tp'].sum())}\n"
    )

    f.write(
        f"FP candidates    : "
        f"{int((df['is_tp'] == 0).sum())}\n\n"
    )

    f.write(
        "Normal validation normalization\n"
    )

    f.write(
        "--------------------------------\n"
    )

    f.write(
        f"VAE P5  : {VAE_P5:.8f}\n"
    )

    f.write(
        f"VAE P95 : {VAE_P95:.8f}\n"
    )

    f.write(
        f"Flow P5  : {FLOW_P5:.6f}\n"
    )

    f.write(
        f"Flow P95 : {FLOW_P95:.6f}\n\n"
    )

    f.write(
        "Signal comparison\n"
    )

    f.write(
        "-----------------\n"
    )

    f.write(
        metrics_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "ROI versus global comparison\n"
    )

    f.write(
        "-----------------------------\n"
    )

    f.write(
        comparison_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "Best signal by PR-AUC\n"
    )

    f.write(
        "---------------------\n"
    )

    f.write(
        f"Signal  : "
        f"{best_row['signal']}\n"
    )

    f.write(
        f"ROC-AUC : "
        f"{best_row['ROC_AUC']:.6f}\n"
    )

    f.write(
        f"PR-AUC  : "
        f"{best_row['PR_AUC']:.6f}\n"
    )

    f.write(
        f"Best F1 : "
        f"{best_row['Best_F1']:.6f}\n"
    )

    f.write(
        "\n\n"
    )

    f.write(
        "Interpretation\n"
    )

    f.write(
        "--------------\n"
    )

    f.write(
        "A useful result is an improvement in "
        "ROI-aligned anomaly discrimination over "
        "global image-level anomaly pooling.\n\n"
    )

    f.write(
        "A useful fusion result is an improvement "
        "of YOLO+VAE+Flow ROI fusion over YOLO "
        "alone and over the previous global fusion.\n\n"
    )

    f.write(
        "Best-F1 thresholds are diagnostic and "
        "test-set-derived. They must not be "
        "presented as unbiased deployment "
        "thresholds.\n\n"
    )

    f.write(
        "The shadow experiment was intentionally "
        "not included because its image-space "
        "heuristic was quantitatively rejected.\n"
    )


# ================================================================
# FINAL OUTPUT
# ================================================================

print()
print("=" * 72)
print(
    "DONE"
)
print("=" * 72)

print()
print(
    "Output directory:"
)

print(
    f"  {OUT_ROOT}"
)

print()
print(
    "Files:"
)

print(
    "  roi_candidate_scores.csv"
)

print(
    "  roi_fusion_metrics.csv"
)

print(
    "  dataset_wise_roi_results.csv"
)

print(
    "  roi_vs_global_comparison.csv"
)

print(
    "  tp_fp_distributions.csv"
)

print(
    "  roi_fusion_summary.txt"
)

print()
print(
    "Visual examples:"
)

print(
    f"  {EXAMPLE_ROOT}"
)

print()
print(
    "Most important comparison:"
)

print(
    "  OLD_GLOBAL_FUSION"
)

print(
    "        vs"
)

print(
    "  YOLO+VAE+Flow_ROI"
)

print()