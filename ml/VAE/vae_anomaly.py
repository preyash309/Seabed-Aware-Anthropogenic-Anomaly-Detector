import os
import csv
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

VAE_CHECKPOINT = Path(
    r"E:\SIH\SIH_Results\vae_normal_seabed\checkpoints\best.pt"
)

VAE_TEST_DIR = Path(
    r"E:\SIH\Datasets\SAAD_VAE\test"
)

SAAD_ROOT = Path(
    r"E:\SIH\Datasets\SAAD_baseline"
)

SAAD_MANIFEST = SAAD_ROOT / "manifest.csv"

OUTPUT_DIR = Path(
    r"E:\SIH\SIH_Results\vae_anomaly_evaluation"
)

PATCH_SIZE = 256

BATCH_SIZE = 64

NUM_WORKERS = 4

LATENT_DIM = 128

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SEED = 42

# Number of background patches sampled
# from each anthropogenic test image.
BACKGROUND_PATCHES_PER_IMAGE = 3

# Minimum center-to-object distance for background patches.
BACKGROUND_MIN_DISTANCE = 180

# Number of attempts to find a valid background patch.
BACKGROUND_MAX_ATTEMPTS = 50

# Operational threshold:
# Anything above this percentage of normal
# reconstruction errors is considered anomalous.
DEFAULT_THRESHOLD_PERCENTILE = 99.0


# ============================================================
# RANDOM SEEDS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RECON_DIR = OUTPUT_DIR / "reconstructions"

RECON_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PLOT_DIR = OUTPUT_DIR / "plots"

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VAE MODEL
#
# THIS MUST MATCH THE TRAINED MODEL EXACTLY.
#
# The checkpoint contains BatchNorm layers.
# ============================================================

class ConvVAE(nn.Module):

    def __init__(
        self,
        latent_dim=128
    ):

        super().__init__()

        self.latent_dim = latent_dim

        # ====================================================
        # ENCODER
        #
        # 256
        #   ↓
        # 128
        #   ↓
        # 64
        #   ↓
        # 32
        #   ↓
        # 16
        #   ↓
        # 8
        # ====================================================

        self.encoder = nn.Sequential(

            # 256 -> 128
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

            # 128 -> 64
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

            # 64 -> 32
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

            # 32 -> 16
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

            # 16 -> 8
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

        # 512 * 8 * 8 = 32768

        self.fc_mu = nn.Linear(
            512 * 8 * 8,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            512 * 8 * 8,
            latent_dim
        )

        # ====================================================
        # DECODER
        # ====================================================

        self.fc_decode = nn.Linear(
            latent_dim,
            512 * 8 * 8
        )

        self.decoder = nn.Sequential(

            # 8 -> 16
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

            # 16 -> 32
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

            # 32 -> 64
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

            # 64 -> 128
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

            # 128 -> 256
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

        h = h.view(
            h.size(0),
            -1
        )

        mu = self.fc_mu(h)

        logvar = self.fc_logvar(h)

        # Numerical stabilization.
        logvar = torch.clamp(
            logvar,
            min=-10.0,
            max=10.0
        )

        return mu, logvar


    def reparameterize(
        self,
        mu,
        logvar
    ):

        std = torch.exp(
            0.5 * logvar
        )

        eps = torch.randn_like(
            std
        )

        return mu + eps * std


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

        # ====================================================
        # IMPORTANT:
        #
        # During evaluation we use mu directly instead of
        # sampling z.
        #
        # This makes reconstruction error deterministic.
        # ====================================================

        z = mu

        reconstruction = self.decode(
            z
        )

        return (
            reconstruction,
            mu,
            logvar
        )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 70)
    print("LOADING VAE")
    print("=" * 70)

    print("Checkpoint:")
    print(VAE_CHECKPOINT)

    checkpoint = torch.load(
        VAE_CHECKPOINT,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Support common checkpoint formats.
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model = ConvVAE(
        latent_dim=LATENT_DIM
    )

    # STRICT=True is intentional.
    #
    # If this fails, the architecture doesn't match.
    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(
        DEVICE
    )

    model.eval()

    print(
        "Device:",
        DEVICE
    )

    print(
        "Latent dimension:",
        LATENT_DIM
    )

    print(
        "Model loaded successfully."
    )

    return model


# ============================================================
# IMAGE LOADING
# ============================================================

def load_grayscale(
    path
):

    img = Image.open(
        path
    ).convert("L")

    arr = np.asarray(
        img,
        dtype=np.float32
    )

    arr /= 255.0

    return arr


# ============================================================
# PAD IMAGE
# ============================================================

def pad_to_size(
    arr,
    size
):

    h, w = arr.shape

    pad_h = max(
        0,
        size - h
    )

    pad_w = max(
        0,
        size - w
    )

    if (
        pad_h == 0
        and
        pad_w == 0
    ):

        return arr

    top = pad_h // 2

    bottom = pad_h - top

    left = pad_w // 2

    right = pad_w - left

    arr = np.pad(
        arr,
        (
            (top, bottom),
            (left, right)
        ),
        mode="reflect"
    )

    return arr


# ============================================================
# CROP 256x256 PATCH
# ============================================================

def crop_patch(
    arr,
    center_x,
    center_y,
    size=256
):

    arr = pad_to_size(
        arr,
        size
    )

    h, w = arr.shape

    half = size // 2

    cx = int(
        round(center_x)
    )

    cy = int(
        round(center_y)
    )

    x1 = cx - half

    y1 = cy - half

    # Keep crop inside image.
    x1 = max(
        0,
        min(
            x1,
            w - size
        )
    )

    y1 = max(
        0,
        min(
            y1,
            h - size
        )
    )

    x2 = x1 + size

    y2 = y1 + size

    patch = arr[
        y1:y2,
        x1:x2
    ]

    # Safety fallback.
    if patch.shape != (
        size,
        size
    ):

        canvas = np.zeros(
            (
                size,
                size
            ),
            dtype=np.float32
        )

        ph, pw = patch.shape

        canvas[
            :ph,
            :pw
        ] = patch

        patch = canvas

    return patch


# ============================================================
# PATCH -> TENSOR
# ============================================================

def patch_to_tensor(
    patch
):

    return torch.from_numpy(
        patch
    ).float().unsqueeze(0)


# ============================================================
# NORMAL VAE TEST DATASET
# ============================================================

class NormalPatchDataset(
    Dataset
):

    def __init__(
        self,
        root
    ):

        self.paths = sorted(

            list(
                root.rglob(
                    "*.png"
                )
            )

            +

            list(
                root.rglob(
                    "*.jpg"
                )
            )

            +

            list(
                root.rglob(
                    "*.jpeg"
                )
            )
        )

        if len(self.paths) == 0:

            raise RuntimeError(
                f"No images found in {root}"
            )


    def __len__(
        self
    ):

        return len(
            self.paths
        )


    def __getitem__(
        self,
        idx
    ):

        path = self.paths[
            idx
        ]

        arr = load_grayscale(
            path
        )

        # ----------------------------------------------------
        # Normally these are already 256x256.
        # ----------------------------------------------------

        if arr.shape != (
            PATCH_SIZE,
            PATCH_SIZE
        ):

            img = Image.fromarray(
                np.uint8(
                    np.clip(
                        arr,
                        0,
                        1
                    )
                    *
                    255
                )
            )

            img = img.resize(
                (
                    PATCH_SIZE,
                    PATCH_SIZE
                ),
                Image.Resampling.BILINEAR
            )

            arr = np.asarray(
                img,
                dtype=np.float32
            ) / 255.0

        # ====================================================
        # IMPORTANT FIX
        #
        # Return idx, NOT path.
        #
        # score_dataset() uses this index to recover
        # the corresponding path.
        # ====================================================

        return (
            patch_to_tensor(
                arr
            ),
            idx
        )


# ============================================================
# YOLO LABEL READER
# ============================================================

def read_yolo_labels(
    label_path
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

                cls = int(
                    parts[0]
                )

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

            except ValueError:

                continue

            boxes.append(
                (
                    cls,
                    xc,
                    yc,
                    bw,
                    bh
                )
            )

    return boxes


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_path(
    value
):

    if value is None:

        return None

    try:

        if pd.isna(value):

            return None

    except Exception:

        pass

    value = str(
        value
    ).strip()

    if not value:

        return None

    p = Path(
        value
    )

    if p.exists():

        return p

    # Relative to SAAD root.
    p2 = (
        SAAD_ROOT
        /
        value
    )

    if p2.exists():

        return p2

    # Normalize Windows separators.
    value2 = value.replace(
        "\\",
        os.sep
    )

    p3 = Path(
        value2
    )

    if p3.exists():

        return p3

    p4 = (
        SAAD_ROOT
        /
        value2
    )

    if p4.exists():

        return p4

    return None


# ============================================================
# BUILD ANTHROPOGENIC POSITIVE PATCHES
# ============================================================

def build_positive_patches(
    manifest
):

    print()
    print("=" * 70)
    print("BUILDING ANTHROPOGENIC OBJECT PATCHES")
    print("=" * 70)

    df = manifest.copy()

    # --------------------------------------------------------
    # ONLY HELD-OUT TEST IMAGES
    # --------------------------------------------------------

    df = df[
        df["split"]
        .astype(str)
        .str.lower()
        ==
        "test"
    ].copy()

    # --------------------------------------------------------
    # ONLY ANNOTATED IMAGES
    # --------------------------------------------------------

    if "has_annotation" in df.columns:

        annotation_values = (
            df["has_annotation"]
            .astype(str)
            .str.lower()
        )

        df = df[
            annotation_values.isin(
                [
                    "true",
                    "1",
                    "yes"
                ]
            )
        ]

    print(
        "Test images with annotations:",
        len(df)
    )

    records = []

    for _, row in df.iterrows():

        dataset_name = str(
            row.get(
                "dataset",
                "unknown"
            )
        )

        # ----------------------------------------------------
        # Prefer unified output image.
        # ----------------------------------------------------

        image_path = resolve_path(
            row.get(
                "output_image"
            )
        )

        if image_path is None:

            image_path = resolve_path(
                row.get(
                    "original_path"
                )
            )

        label_path = resolve_path(
            row.get(
                "output_label"
            )
        )

        if (
            image_path is None
            or
            label_path is None
        ):

            continue

        try:

            arr = load_grayscale(
                image_path
            )

        except Exception:

            continue

        h, w = arr.shape

        boxes = read_yolo_labels(
            label_path
        )

        for object_id, box in enumerate(
            boxes
        ):

            (
                cls,
                xc,
                yc,
                bw,
                bh
            ) = box

            # ------------------------------------------------
            # YOLO normalized coordinates -> pixels.
            # ------------------------------------------------

            cx = xc * w

            cy = yc * h

            box_w = bw * w

            box_h = bh * h

            patch = crop_patch(
                arr,
                cx,
                cy,
                PATCH_SIZE
            )

            records.append(
                {
                    "patch": patch,
                    "label": 1,
                    "type": "anthropogenic",
                    "dataset": dataset_name,
                    "image": str(
                        image_path
                    ),
                    "object_id": object_id,
                    "cx": cx,
                    "cy": cy,
                    "box_w": box_w,
                    "box_h": box_h,
                }
            )

    print(
        "Anthropogenic patches:",
        len(records)
    )

    dataset_names = sorted(
        set(
            r["dataset"]
            for r in records
        )
    )

    for dataset_name in dataset_names:

        count = sum(
            r["dataset"]
            ==
            dataset_name
            for r in records
        )

        print(
            f"  {dataset_name}: {count}"
        )

    return records


# ============================================================
# BUILD HARD-NEGATIVE BACKGROUND PATCHES
# ============================================================

def build_background_patches(
    manifest
):

    print()
    print("=" * 70)
    print("BUILDING HARD-NEGATIVE BACKGROUND PATCHES")
    print("=" * 70)

    df = manifest.copy()

    df = df[
        df["split"]
        .astype(str)
        .str.lower()
        ==
        "test"
    ].copy()

    if "has_annotation" in df.columns:

        annotation_values = (
            df["has_annotation"]
            .astype(str)
            .str.lower()
        )

        df = df[
            annotation_values.isin(
                [
                    "true",
                    "1",
                    "yes"
                ]
            )
        ]

    records = []

    for _, row in df.iterrows():

        dataset_name = str(
            row.get(
                "dataset",
                "unknown"
            )
        )

        image_path = resolve_path(
            row.get(
                "output_image"
            )
        )

        if image_path is None:

            image_path = resolve_path(
                row.get(
                    "original_path"
                )
            )

        label_path = resolve_path(
            row.get(
                "output_label"
            )
        )

        if (
            image_path is None
            or
            label_path is None
        ):

            continue

        try:

            arr = load_grayscale(
                image_path
            )

        except Exception:

            continue

        h, w = arr.shape

        boxes = read_yolo_labels(
            label_path
        )

        # ----------------------------------------------------
        # Convert normalized boxes to pixel coordinates.
        # ----------------------------------------------------

        pixel_boxes = []

        for box in boxes:

            (
                cls,
                xc,
                yc,
                bw,
                bh
            ) = box

            pixel_boxes.append(
                (
                    xc * w,
                    yc * h,
                    bw * w,
                    bh * h
                )
            )

        accepted = 0

        for attempt in range(
            BACKGROUND_MAX_ATTEMPTS
        ):

            if (
                accepted
                >=
                BACKGROUND_PATCHES_PER_IMAGE
            ):

                break

            # ------------------------------------------------
            # Keep center reasonably away from boundaries.
            # ------------------------------------------------

            margin = PATCH_SIZE // 2

            if w <= PATCH_SIZE:

                cx = w / 2.0

            else:

                cx = random.randint(
                    margin,
                    w - margin
                )

            if h <= PATCH_SIZE:

                cy = h / 2.0

            else:

                cy = random.randint(
                    margin,
                    h - margin
                )

            # ------------------------------------------------
            # Reject points close to objects.
            # ------------------------------------------------

            too_close = False

            for (
                bx,
                by,
                bw,
                bh
            ) in pixel_boxes:

                distance = math.sqrt(
                    (cx - bx) ** 2
                    +
                    (cy - by) ** 2
                )

                exclusion = max(
                    BACKGROUND_MIN_DISTANCE,
                    0.5 * max(
                        bw,
                        bh
                    )
                    +
                    BACKGROUND_MIN_DISTANCE / 2
                )

                if distance < exclusion:

                    too_close = True

                    break

            if too_close:

                continue

            patch = crop_patch(
                arr,
                cx,
                cy,
                PATCH_SIZE
            )

            records.append(
                {
                    "patch": patch,
                    "label": 0,
                    "type": "background",
                    "dataset": dataset_name,
                    "image": str(
                        image_path
                    ),
                    "object_id": -1,
                    "cx": cx,
                    "cy": cy,
                    "box_w": 0,
                    "box_h": 0,
                }
            )

            accepted += 1

    print(
        "Hard-negative patches:",
        len(records)
    )

    dataset_names = sorted(
        set(
            r["dataset"]
            for r in records
        )
    )

    for dataset_name in dataset_names:

        count = sum(
            r["dataset"]
            ==
            dataset_name
            for r in records
        )

        print(
            f"  {dataset_name}: {count}"
        )

    return records


# ============================================================
# RECORD DATASET
# ============================================================

class RecordDataset(
    Dataset
):

    def __init__(
        self,
        records
    ):

        self.records = records


    def __len__(
        self
    ):

        return len(
            self.records
        )


    def __getitem__(
        self,
        idx
    ):

        record = self.records[
            idx
        ]

        return (
            patch_to_tensor(
                record["patch"]
            ),
            idx
        )


# ============================================================
# SCORE DATASET
# ============================================================

@torch.no_grad()
def score_dataset(
    model,
    dataloader,
    records=None,
    normal_paths=False
):

    scores = []

    metadata = []

    for batch in dataloader:

        x, indices = batch

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Safety check.
        # ----------------------------------------------------

        if not torch.isfinite(
            x
        ).all():

            raise RuntimeError(
                "Non-finite input encountered."
            )

        reconstruction, mu, logvar = model(
            x
        )

        # ----------------------------------------------------
        # Safety check.
        # ----------------------------------------------------

        if not torch.isfinite(
            reconstruction
        ).all():

            raise RuntimeError(
                "Non-finite VAE reconstruction encountered."
            )

        # ----------------------------------------------------
        # Per-patch reconstruction MSE.
        # ----------------------------------------------------

        mse = torch.mean(
            (
                x
                -
                reconstruction
            ) ** 2,
            dim=(1, 2, 3)
        )

        mse = (
            mse
            .detach()
            .cpu()
            .numpy()
        )

        scores.extend(
            mse.tolist()
        )

        # ----------------------------------------------------
        # Recover metadata.
        # ----------------------------------------------------

        for i in indices:

            i = int(i)

            if normal_paths:

                metadata.append(
                    str(
                        dataloader
                        .dataset
                        .paths[i]
                    )
                )

            else:

                metadata.append(
                    records[i]
                )

    return (
        np.asarray(
            scores,
            dtype=np.float64
        ),
        metadata
    )


# ============================================================
# PRINT SCORE STATISTICS
# ============================================================

def print_statistics(
    name,
    scores
):

    scores = np.asarray(
        scores
    )

    print()
    print("-" * 70)
    print(name)
    print("-" * 70)

    print(
        "N       :",
        len(scores)
    )

    print(
        "Mean    :",
        f"{scores.mean():.8f}"
    )

    print(
        "Median  :",
        f"{np.median(scores):.8f}"
    )

    print(
        "Std     :",
        f"{scores.std():.8f}"
    )

    print(
        "Min     :",
        f"{scores.min():.8f}"
    )

    print(
        "Max     :",
        f"{scores.max():.8f}"
    )

    print(
        "P90     :",
        f"{np.percentile(scores, 90):.8f}"
    )

    print(
        "P95     :",
        f"{np.percentile(scores, 95):.8f}"
    )

    print(
        "P99     :",
        f"{np.percentile(scores, 99):.8f}"
    )


# ============================================================
# THRESHOLD METRICS
# ============================================================

def threshold_metrics(
    y_true,
    scores,
    threshold
):

    predictions = (
        scores
        >=
        threshold
    ).astype(
        np.int32
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2
        *
        precision
        *
        recall
        /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# BEST F1 THRESHOLD
# ============================================================

def find_best_f1_threshold(
    y_true,
    scores
):

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            scores
        )
    )

    f1 = (
        2
        *
        precision
        *
        recall
        /
        np.maximum(
            precision + recall,
            1e-12
        )
    )

    best_idx = int(
        np.argmax(f1)
    )

    if len(thresholds) == 0:

        return float(
            np.median(scores)
        )

    if best_idx >= len(
        thresholds
    ):

        threshold = thresholds[-1]

    else:

        threshold = thresholds[
            best_idx
        ]

    return float(
        threshold
    )


# ============================================================
# SCORE DISTRIBUTION PLOT
# ============================================================

def plot_score_distribution(
    normal_scores,
    positive_scores,
    background_scores
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        normal_scores,
        bins=80,
        alpha=0.6,
        density=True,
        label="Normal seabed"
    )

    plt.hist(
        positive_scores,
        bins=80,
        alpha=0.6,
        density=True,
        label="Anthropogenic objects"
    )

    if len(
        background_scores
    ) > 0:

        plt.hist(
            background_scores,
            bins=80,
            alpha=0.6,
            density=True,
            label="Hard-negative background"
        )

    plt.xlabel(
        "VAE reconstruction MSE"
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        "VAE Reconstruction Error Distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR /
        "score_distribution.png",
        dpi=200
    )

    plt.close()


# ============================================================
# LOG SCORE DISTRIBUTION
# ============================================================

def plot_log_score_distribution(
    normal_scores,
    positive_scores,
    background_scores
):

    eps = 1e-10

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        np.log10(
            normal_scores
            +
            eps
        ),
        bins=80,
        alpha=0.6,
        density=True,
        label="Normal seabed"
    )

    plt.hist(
        np.log10(
            positive_scores
            +
            eps
        ),
        bins=80,
        alpha=0.6,
        density=True,
        label="Anthropogenic objects"
    )

    if len(
        background_scores
    ) > 0:

        plt.hist(
            np.log10(
                background_scores
                +
                eps
            ),
            bins=80,
            alpha=0.6,
            density=True,
            label="Hard-negative background"
        )

    plt.xlabel(
        "log10(VAE reconstruction MSE)"
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        "Log Reconstruction Error Distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR /
        "score_distribution_log.png",
        dpi=200
    )

    plt.close()


# ============================================================
# ROC + PR CURVES
# ============================================================

def plot_roc_pr(
    y_true,
    scores
):

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_true,
        scores
    )

    roc_auc = roc_auc_score(
        y_true,
        scores
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "VAE Anomaly Detection ROC"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR /
        "roc_curve.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Precision-Recall
    # --------------------------------------------------------

    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            scores
        )
    )

    pr_auc = average_precision_score(
        y_true,
        scores
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        recall,
        precision,
        label=f"PR-AUC = {pr_auc:.4f}"
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "VAE Anomaly Detection Precision-Recall"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR /
        "pr_curve.png",
        dpi=200
    )

    plt.close()

    return (
        roc_auc,
        pr_auc
    )


# ============================================================
# SAVE TOP RECONSTRUCTION EXAMPLES
# ============================================================

@torch.no_grad()
def save_top_reconstructions(
    model,
    records,
    scores,
    name,
    top_k=20
):

    if len(records) == 0:

        return

    order = np.argsort(
        scores
    )[::-1]

    top_indices = order[
        :
        min(
            top_k,
            len(order)
        )
    ]

    for rank, idx in enumerate(
        top_indices
    ):

        record = records[
            idx
        ]

        patch = record[
            "patch"
        ]

        x = (
            torch.from_numpy(
                patch
            )
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(DEVICE)
        )

        reconstruction, _, _ = model(
            x
        )

        reconstruction = (
            reconstruction[
                0,
                0
            ]
            .detach()
            .cpu()
            .numpy()
        )

        error = np.abs(
            patch
            -
            reconstruction
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(12, 4)
        )

        axes[0].imshow(
            patch,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[0].set_title(
            "Input"
        )

        axes[1].imshow(
            reconstruction,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[1].set_title(
            "Reconstruction"
        )

        axes[2].imshow(
            error,
            cmap="hot"
        )

        axes[2].set_title(
            f"Error\nMSE={scores[idx]:.6f}"
        )

        for ax in axes:

            ax.axis(
                "off"
            )

        fig.suptitle(
            f"{name} | rank={rank + 1}"
        )

        fig.tight_layout()

        filename = (
            f"{name.lower()}_"
            f"{rank + 1:03d}_"
            f"score_{scores[idx]:.6f}.png"
        )

        fig.savefig(
            RECON_DIR /
            filename,
            dpi=160
        )

        plt.close(
            fig
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SAAD VAE ANOMALY EVALUATION")
    print("=" * 70)

    print(
        "Device:",
        DEVICE
    )

    # ========================================================
    # CHECK REQUIRED PATHS
    # ========================================================

    if not VAE_CHECKPOINT.exists():

        raise FileNotFoundError(
            "VAE checkpoint not found:\n"
            f"{VAE_CHECKPOINT}"
        )

    if not VAE_TEST_DIR.exists():

        raise FileNotFoundError(
            "VAE test directory not found:\n"
            f"{VAE_TEST_DIR}"
        )

    if not SAAD_MANIFEST.exists():

        raise FileNotFoundError(
            "SAAD manifest not found:\n"
            f"{SAAD_MANIFEST}"
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # NORMAL TEST PATCHES
    # ========================================================

    print()
    print("=" * 70)
    print("NORMAL VAE TEST SET")
    print("=" * 70)

    normal_dataset = NormalPatchDataset(
        VAE_TEST_DIR
    )

    print(
        "Normal test patches:",
        len(normal_dataset)
    )

    normal_loader = DataLoader(
        normal_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    normal_scores, normal_paths = (
        score_dataset(
            model,
            normal_loader,
            normal_paths=True
        )
    )

    print_statistics(
        "NORMAL SEABED",
        normal_scores
    )

    # ========================================================
    # LOAD SAAD MANIFEST
    # ========================================================

    manifest = pd.read_csv(
        SAAD_MANIFEST
    )

    print()
    print(
        "Manifest rows:",
        len(manifest)
    )

    # ========================================================
    # ANTHROPOGENIC OBJECT PATCHES
    # ========================================================

    positive_records = (
        build_positive_patches(
            manifest
        )
    )

    if len(
        positive_records
    ) == 0:

        raise RuntimeError(
            "No anthropogenic patches were created."
        )

    positive_dataset = RecordDataset(
        positive_records
    )

    positive_loader = DataLoader(
        positive_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    positive_scores, positive_meta = (
        score_dataset(
            model,
            positive_loader,
            records=positive_records
        )
    )

    print_statistics(
        "ANTHROPOGENIC OBJECTS",
        positive_scores
    )

    # ========================================================
    # HARD NEGATIVES
    # ========================================================

    background_records = (
        build_background_patches(
            manifest
        )
    )

    if len(
        background_records
    ) > 0:

        background_dataset = (
            RecordDataset(
                background_records
            )
        )

        background_loader = DataLoader(
            background_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available()
        )

        background_scores, background_meta = (
            score_dataset(
                model,
                background_loader,
                records=background_records
            )
        )

        print_statistics(
            "HARD-NEGATIVE BACKGROUND",
            background_scores
        )

    else:

        background_scores = np.array(
            [],
            dtype=np.float64
        )

        background_meta = []

    # ========================================================
    # OVERALL ANOMALY SEPARATION
    # ========================================================

    print()
    print("=" * 70)
    print("OVERALL ANOMALY SEPARATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Normal = 0
    # Anthropogenic = 1
    # --------------------------------------------------------

    y_true = np.concatenate(
        [
            np.zeros(
                len(normal_scores),
                dtype=np.int32
            ),

            np.ones(
                len(positive_scores),
                dtype=np.int32
            )
        ]
    )

    scores = np.concatenate(
        [
            normal_scores,
            positive_scores
        ]
    )

    roc_auc, pr_auc = plot_roc_pr(
        y_true,
        scores
    )

    print(
        f"ROC-AUC : {roc_auc:.6f}"
    )

    print(
        f"PR-AUC  : {pr_auc:.6f}"
    )

    # ========================================================
    # NORMAL 99TH PERCENTILE THRESHOLD
    # ========================================================

    threshold_99 = np.percentile(
        normal_scores,
        DEFAULT_THRESHOLD_PERCENTILE
    )

    metrics_99 = threshold_metrics(
        y_true,
        scores,
        threshold_99
    )

    print()
    print(
        f"Threshold = normal "
        f"{DEFAULT_THRESHOLD_PERCENTILE:.0f}th percentile"
    )

    print(
        f"Threshold: {threshold_99:.8f}"
    )

    print(
        f"Precision: {metrics_99['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics_99['recall']:.4f}"
    )

    print(
        f"F1       : {metrics_99['f1']:.4f}"
    )

    print(
        f"TP={metrics_99['tp']} "
        f"FP={metrics_99['fp']} "
        f"FN={metrics_99['fn']} "
        f"TN={metrics_99['tn']}"
    )

    # ========================================================
    # BEST F1 THRESHOLD
    # ========================================================

    best_threshold = (
        find_best_f1_threshold(
            y_true,
            scores
        )
    )

    best_metrics = threshold_metrics(
        y_true,
        scores,
        best_threshold
    )

    print()
    print(
        "Best-F1 threshold:"
    )

    print(
        f"Threshold: {best_threshold:.8f}"
    )

    print(
        f"Precision: {best_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {best_metrics['recall']:.4f}"
    )

    print(
        f"F1       : {best_metrics['f1']:.4f}"
    )

    # ========================================================
    # HARD NEGATIVE FALSE POSITIVE RATE
    # ========================================================

    if len(
        background_scores
    ) > 0:

        background_flagged = (
            background_scores
            >=
            threshold_99
        )

        background_anomaly_rate = (
            np.mean(
                background_flagged
            )
        )

        print()
        print("=" * 70)
        print("HARD-NEGATIVE FALSE POSITIVE RATE")
        print("=" * 70)

        print(
            "Threshold:",
            f"{threshold_99:.8f}"
        )

        print(
            "Background patches:",
            len(
                background_scores
            )
        )

        print(
            "Background flagged anomalous:",
            int(
                np.sum(
                    background_flagged
                )
            )
        )

        print(
            "False-positive rate:",
            f"{background_anomaly_rate:.4f}"
        )

    # ========================================================
    # DATASET-WISE ANALYSIS
    # ========================================================

    print()
    print("=" * 70)
    print("DATASET-WISE ANTHROPOGENIC SCORES")
    print("=" * 70)

    dataset_rows = []

    dataset_names = sorted(
        set(
            r["dataset"]
            for r in positive_records
        )
    )

    for dataset_name in dataset_names:

        indices = [
            i
            for i, record in enumerate(
                positive_records
            )
            if record["dataset"]
            ==
            dataset_name
        ]

        ds_scores = positive_scores[
            indices
        ]

        mean_score = float(
            np.mean(
                ds_scores
            )
        )

        median_score = float(
            np.median(
                ds_scores
            )
        )

        p95_score = float(
            np.percentile(
                ds_scores,
                95
            )
        )

        detection_rate = float(
            np.mean(
                ds_scores
                >=
                threshold_99
            )
        )

        print()
        print(
            dataset_name
        )

        print(
            "  N:",
            len(
                ds_scores
            )
        )

        print(
            "  Mean:",
            f"{mean_score:.8f}"
        )

        print(
            "  Median:",
            f"{median_score:.8f}"
        )

        print(
            "  P95:",
            f"{p95_score:.8f}"
        )

        print(
            "  >99% normal threshold:",
            f"{detection_rate:.4f}"
        )

        dataset_rows.append(
            {
                "dataset": dataset_name,
                "n": len(ds_scores),
                "mean": mean_score,
                "median": median_score,
                "p95": p95_score,
                "p99_normal_threshold_detection_rate":
                    detection_rate
            }
        )

    pd.DataFrame(
        dataset_rows
    ).to_csv(
        OUTPUT_DIR /
        "dataset_results.csv",
        index=False
    )

    # ========================================================
    # PLOTS
    # ========================================================

    plot_score_distribution(
        normal_scores,
        positive_scores,
        background_scores
    )

    plot_log_score_distribution(
        normal_scores,
        positive_scores,
        background_scores
    )

    # ========================================================
    # HIGH ANOMALY EXAMPLES
    # ========================================================

    print()
    print("=" * 70)
    print("SAVING HIGH-ANOMALY EXAMPLES")
    print("=" * 70)

    save_top_reconstructions(
        model,
        positive_records,
        positive_scores,
        "ANTHROPOGENIC",
        top_k=30
    )

    if len(
        background_records
    ) > 0:

        save_top_reconstructions(
            model,
            background_records,
            background_scores,
            "BACKGROUND",
            top_k=20
        )

    # ========================================================
    # SAVE ALL SCORES
    # ========================================================

    rows = []

    # --------------------------------------------------------
    # Normal
    # --------------------------------------------------------

    for path, score in zip(
        normal_paths,
        normal_scores
    ):

        rows.append(
            {
                "type": "normal",
                "dataset": "VAE_normal_test",
                "image": str(path),
                "object_id": -1,
                "score": float(score),
                "label": 0
            }
        )

    # --------------------------------------------------------
    # Anthropogenic
    # --------------------------------------------------------

    for record, score in zip(
        positive_records,
        positive_scores
    ):

        rows.append(
            {
                "type": "anthropogenic",
                "dataset": record["dataset"],
                "image": record["image"],
                "object_id": record["object_id"],
                "score": float(score),
                "label": 1
            }
        )

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    for record, score in zip(
        background_records,
        background_scores
    ):

        rows.append(
            {
                "type": "background",
                "dataset": record["dataset"],
                "image": record["image"],
                "object_id": -1,
                "score": float(score),
                "label": 0
            }
        )

    score_df = pd.DataFrame(
        rows
    )

    score_df.to_csv(
        OUTPUT_DIR /
        "all_anomaly_scores.csv",
        index=False
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_path = (
        OUTPUT_DIR /
        "evaluation_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SAAD VAE ANOMALY EVALUATION\n"
        )

        f.write(
            "=" * 70
            +
            "\n\n"
        )

        f.write(
            f"Checkpoint: "
            f"{VAE_CHECKPOINT}\n"
        )

        f.write(
            f"Device: {DEVICE}\n\n"
        )

        # ----------------------------------------------------
        # Normal
        # ----------------------------------------------------

        f.write(
            "NORMAL TEST\n"
        )

        f.write(
            f"N = {len(normal_scores)}\n"
        )

        f.write(
            f"Mean = "
            f"{normal_scores.mean():.8f}\n"
        )

        f.write(
            f"Median = "
            f"{np.median(normal_scores):.8f}\n"
        )

        f.write(
            f"P95 = "
            f"{np.percentile(normal_scores,95):.8f}\n"
        )

        f.write(
            f"P99 = "
            f"{np.percentile(normal_scores,99):.8f}\n\n"
        )

        # ----------------------------------------------------
        # Anthropogenic
        # ----------------------------------------------------

        f.write(
            "ANTHROPOGENIC\n"
        )

        f.write(
            f"N = {len(positive_scores)}\n"
        )

        f.write(
            f"Mean = "
            f"{positive_scores.mean():.8f}\n"
        )

        f.write(
            f"Median = "
            f"{np.median(positive_scores):.8f}\n"
        )

        f.write(
            f"P95 = "
            f"{np.percentile(positive_scores,95):.8f}\n"
        )

        f.write(
            f"P99 = "
            f"{np.percentile(positive_scores,99):.8f}\n\n"
        )

        # ----------------------------------------------------
        # Background
        # ----------------------------------------------------

        if len(
            background_scores
        ) > 0:

            f.write(
                "HARD NEGATIVE BACKGROUND\n"
            )

            f.write(
                f"N = "
                f"{len(background_scores)}\n"
            )

            f.write(
                f"Mean = "
                f"{background_scores.mean():.8f}\n"
            )

            f.write(
                f"Median = "
                f"{np.median(background_scores):.8f}\n"
            )

            f.write(
                f"P95 = "
                f"{np.percentile(background_scores,95):.8f}\n"
            )

            f.write(
                f"P99 = "
                f"{np.percentile(background_scores,99):.8f}\n\n"
            )

        # ----------------------------------------------------
        # Overall
        # ----------------------------------------------------

        f.write(
            "OVERALL\n"
        )

        f.write(
            f"ROC-AUC = "
            f"{roc_auc:.8f}\n"
        )

        f.write(
            f"PR-AUC = "
            f"{pr_auc:.8f}\n\n"
        )

        # ----------------------------------------------------
        # 99th percentile threshold
        # ----------------------------------------------------

        f.write(
            "99th percentile normal threshold\n"
        )

        f.write(
            f"Threshold = "
            f"{threshold_99:.8f}\n"
        )

        f.write(
            f"Precision = "
            f"{metrics_99['precision']:.8f}\n"
        )

        f.write(
            f"Recall = "
            f"{metrics_99['recall']:.8f}\n"
        )

        f.write(
            f"F1 = "
            f"{metrics_99['f1']:.8f}\n\n"
        )

        # ----------------------------------------------------
        # Best F1
        # ----------------------------------------------------

        f.write(
            "BEST F1 THRESHOLD\n"
        )

        f.write(
            f"Threshold = "
            f"{best_threshold:.8f}\n"
        )

        f.write(
            f"Precision = "
            f"{best_metrics['precision']:.8f}\n"
        )

        f.write(
            f"Recall = "
            f"{best_metrics['recall']:.8f}\n"
        )

        f.write(
            f"F1 = "
            f"{best_metrics['f1']:.8f}\n"
        )

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print(
        "Results directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Important files:"
    )

    print(
        "  evaluation_summary.txt"
    )

    print(
        "  all_anomaly_scores.csv"
    )

    print(
        "  dataset_results.csv"
    )

    print(
        "  plots/score_distribution.png"
    )

    print(
        "  plots/score_distribution_log.png"
    )

    print(
        "  plots/roc_curve.png"
    )

    print(
        "  plots/pr_curve.png"
    )

    print(
        "  reconstructions/"
    )

    print()
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()