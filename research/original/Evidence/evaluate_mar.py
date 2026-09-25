"""
SAAD — Masked Acoustic Reconstruction (MAR) Evaluation
=======================================================

Evaluates the FROZEN MAR v1 checkpoint.

MAR:
    Mask a local region
    -> reconstruct from surrounding acoustic context
    -> measure reconstruction error inside the masked region

Primary evaluation:
    NORMAL
    ANTHROPOGENIC

The script also tries to locate the existing hard-negative evaluation
set from previous SAAD/VAE evaluation outputs when possible.

IMPORTANT:
    - No training.
    - MAR checkpoint is frozen.
    - SAAD final test labels are NOT used to fit anything.
    - No thresholds are fitted using anthropogenic labels.
    - Multiple random masks are used per image.
    - Model receives 4D tensors: [B,C,H,W].
"""

import os
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = Path(
    r"E:\SIH\Datasets\SAAD_VAE"
)

CHECKPOINT = Path(
    r"E:\SIH\SIH_Results\mar_normal_seabed\checkpoints\best.pt"
)

RESULT_ROOT = Path(
    r"E:\SIH\SIH_Results\mar_evaluation"
)

IMG_SIZE = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42

# Number of masks evaluated per image.
# 5 is a reasonable speed/variance compromise.
NUM_MASKS = 5

MASK_MIN_SIZE = 48
MASK_MAX_SIZE = 112

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

SCORES_DIR = (
    RESULT_ROOT / "scores"
)

SCORES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)


seed_everything(SEED)


# ============================================================
# MODEL
# EXACT ARCHITECTURE USED FOR TRAINING
# ============================================================

class MARNet(nn.Module):

    def __init__(self):

        super().__init__()

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
            nn.ReLU(inplace=True),
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

    def forward(self, x):

        z = self.encoder(x)

        return self.decoder(z)


# ============================================================
# CHECKPOINT
# ============================================================

if not CHECKPOINT.exists():

    raise FileNotFoundError(
        f"\nMAR checkpoint not found:\n{CHECKPOINT}"
    )


print("=" * 70)
print("SAAD — MAR v1 EVALUATION")
print("=" * 70)

print("\nCheckpoint:")
print(CHECKPOINT)

print("\nDevice:")
print(DEVICE)


# ============================================================
# LOAD MODEL
# ============================================================

model = MARNet().to(
    DEVICE
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

if "model_state_dict" not in checkpoint:

    raise RuntimeError(
        "Checkpoint does not contain model_state_dict"
    )

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

model.eval()

print(
    "\nCheckpoint epoch:",
    checkpoint.get(
        "epoch",
        "unknown"
    )
)

print(
    "Stored validation loss:",
    checkpoint.get(
        "best_val_loss",
        "unknown"
    )
)


# ============================================================
# MODEL COMPATIBILITY CHECK
# ============================================================

print("\n")
print("=" * 70)
print("MODEL COMPATIBILITY CHECK")
print("=" * 70)

dummy = torch.rand(
    2,
    1,
    IMG_SIZE,
    IMG_SIZE,
    device=DEVICE
)

with torch.no_grad():

    dummy_output = model(
        dummy
    )

print(
    "Input shape :",
    tuple(dummy.shape)
)

print(
    "Output shape:",
    tuple(dummy_output.shape)
)

print(
    "Input finite:",
    bool(torch.isfinite(dummy).all())
)

print(
    "Output finite:",
    bool(torch.isfinite(dummy_output).all())
)

if tuple(dummy_output.shape) != (
    2,
    1,
    IMG_SIZE,
    IMG_SIZE
):

    raise RuntimeError(
        "MAR output shape is incorrect."
    )

if not torch.isfinite(
    dummy_output
).all():

    raise RuntimeError(
        "MAR output contains NaN/Inf."
    )

print(
    "\nCHECK PASSED"
)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):

    img = Image.open(
        path
    ).convert("L")

    img = img.resize(
        (
            IMG_SIZE,
            IMG_SIZE
        ),
        Image.Resampling.BILINEAR
    )

    arr = np.asarray(
        img,
        dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        arr
    ).unsqueeze(0)

    # [C,H,W]
    return tensor


# ============================================================
# RANDOM MASK
# ============================================================

def generate_random_mask():

    mask = np.zeros(
        (
            IMG_SIZE,
            IMG_SIZE
        ),
        dtype=np.float32
    )

    mh = random.randint(
        MASK_MIN_SIZE,
        MASK_MAX_SIZE
    )

    mw = random.randint(
        MASK_MIN_SIZE,
        MASK_MAX_SIZE
    )

    y0 = random.randint(
        0,
        IMG_SIZE - mh
    )

    x0 = random.randint(
        0,
        IMG_SIZE - mw
    )

    mask[
        y0:y0 + mh,
        x0:x0 + mw
    ] = 1.0

    return torch.from_numpy(
        mask
    ).unsqueeze(0)


# ============================================================
# SCORE ONE PATCH
# ============================================================

@torch.no_grad()
def score_patch(image_tensor):

    """
    image_tensor initially:
        [C,H,W]

    model input:
        [B,C,H,W]

    Returns several image-level MAR scores.
    """

    # --------------------------------------------------------
    # FIX FOR THE ERROR YOU HIT:
    #
    # Conv2d requires 4D:
    # [batch, channels, height, width]
    #
    # load_image() returns:
    # [channels, height, width]
    # --------------------------------------------------------

    if image_tensor.ndim == 3:

        image_tensor = image_tensor.unsqueeze(0)

    if image_tensor.ndim != 4:

        raise RuntimeError(
            "Expected 4D tensor, got "
            f"{tuple(image_tensor.shape)}"
        )

    image_tensor = image_tensor.to(
        DEVICE,
        non_blocking=True
    )

    scores = []

    for _ in range(NUM_MASKS):

        mask = generate_random_mask()

        # [1,1,H,W]
        mask = mask.unsqueeze(0)

        mask = mask.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Same masking operation used during training.
        # ----------------------------------------------------

        visible = (
            image_tensor
            *
            (1.0 - mask)
        )

        visible_mean = (
            visible.sum()
            /
            (
                (1.0 - mask).sum()
                +
                1e-6
            )
        )

        masked_input = (
            image_tensor
            *
            (1.0 - mask)
            +
            visible_mean
            *
            mask
        )

        # ----------------------------------------------------
        # Reconstruction
        # ----------------------------------------------------

        reconstruction = model(
            masked_input
        )

        # ----------------------------------------------------
        # Pixel error
        # ----------------------------------------------------

        error = torch.abs(
            reconstruction
            -
            image_tensor
        )

        # ----------------------------------------------------
        # CRITICAL:
        #
        # Only calculate error INSIDE the masked region.
        # ----------------------------------------------------

        masked_error = (
            error * mask
        ).sum() / (
            mask.sum()
            +
            1e-6
        )

        score = float(
            masked_error.item()
        )

        if not np.isfinite(score):

            raise RuntimeError(
                "MAR generated non-finite score."
            )

        scores.append(
            score
        )

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    return {
        "mar_mean":
            float(np.mean(scores)),

        "mar_median":
            float(np.median(scores)),

        "mar_max":
            float(np.max(scores)),

        "mar_min":
            float(np.min(scores)),

        "mar_std":
            float(np.std(scores)),
    }


# ============================================================
# IMAGE DISCOVERY
# ============================================================

IMG_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def list_images(folder):

    if not folder.exists():

        return []

    return sorted(
        [
            p
            for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower()
            in IMG_EXTENSIONS
        ]
    )


# ============================================================
# NORMAL SET
# ============================================================

normal_dir = (
    DATA_ROOT / "test"
)

normal_paths = list_images(
    normal_dir
)

print("\n")
print("=" * 70)
print("NORMAL EVALUATION SET")
print("=" * 70)

print(
    "Directory:",
    normal_dir
)

print(
    "Images:",
    len(normal_paths)
)


# ============================================================
# ANTHROPOGENIC SET
# ============================================================

"""
The existing SAAD test images are not stored in SAAD_VAE/test.

We therefore recover them from the SAAD baseline manifest.

This uses ONLY the test split.

No labels are used for training.
"""

BASELINE_ROOT = Path(
    r"E:\SIH\Datasets\SAAD_baseline"
)

MANIFEST = (
    BASELINE_ROOT /
    "manifest.csv"
)

anthropogenic_paths = []


if MANIFEST.exists():

    print("\n")
    print("=" * 70)
    print("RECOVERING ANTHROPOGENIC TEST IMAGES")
    print("=" * 70)

    manifest = pd.read_csv(
        MANIFEST
    )

    print(
        "Manifest rows:",
        len(manifest)
    )

    required_columns = {
        "split",
        "saad_class",
        "original_path",
    }

    missing = (
        required_columns
        -
        set(manifest.columns)
    )

    if missing:

        print(
            "WARNING: manifest missing:",
            missing
        )

    else:

        test_rows = manifest[
            manifest["split"]
            .astype(str)
            .str.lower()
            .eq("test")
        ]

        anth_rows = test_rows[
            test_rows["saad_class"]
            .astype(str)
            .str.lower()
            .eq("anthropogenic")
        ]

        for _, row in anth_rows.iterrows():

            p = Path(
                str(
                    row["original_path"]
                )
            )

            if p.exists():

                anthropogenic_paths.append(
                    p
                )

else:

    print(
        "\nWARNING:"
    )

    print(
        "SAAD baseline manifest not found:"
    )

    print(
        MANIFEST
    )


# ------------------------------------------------------------
# Deduplicate
# ------------------------------------------------------------

def unique_paths(paths):

    seen = set()
    output = []

    for p in paths:

        try:

            key = str(
                p.resolve()
            ).lower()

        except Exception:

            key = str(
                p
            ).lower()

        if key not in seen:

            seen.add(key)

            output.append(p)

    return output


normal_paths = unique_paths(
    normal_paths
)

anthropogenic_paths = unique_paths(
    anthropogenic_paths
)

print(
    "\nAnthropogenic images:",
    len(anthropogenic_paths)
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_paths(
    paths,
    label_name,
    label_value
):

    rows = []

    total = len(paths)

    print("\n")
    print(
        f"Evaluating {label_name}: {total}"
    )

    for i, path in enumerate(
        paths,
        1
    ):

        try:

            image = load_image(
                path
            )

            score = score_patch(
                image
            )

            pstr = str(
                path
            ).lower()

            if "ghostvision" in pstr:

                dataset = "GhostVision"

            elif "ai4shipwreck" in pstr:

                dataset = "AI4Shipwrecks"

            elif "subpipemini" in pstr:

                dataset = "SubPipeMini2"

            elif "marine-pulse" in pstr:

                dataset = "Marine-PULSE"

            else:

                dataset = "UNKNOWN"

            rows.append(
                {
                    "path":
                        str(path),

                    "filename":
                        path.name,

                    "label":
                        label_value,

                    "label_name":
                        label_name,

                    "dataset":
                        dataset,

                    **score,
                }
            )

        except Exception as e:

            print(
                f"\nWARNING: failed "
                f"{path}: {e}"
            )

        if (
            i % 50 == 0
            or
            i == total
        ):

            print(
                f"  {i}/{total}"
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# NORMAL
# ============================================================

normal_df = evaluate_paths(
    normal_paths,
    "NORMAL",
    0
)

normal_df.to_csv(
    SCORES_DIR /
    "normal_mar_scores.csv",
    index=False
)


# ============================================================
# ANTHROPOGENIC
# ============================================================

anth_df = evaluate_paths(
    anthropogenic_paths,
    "ANTHROPOGENIC",
    1
)

anth_df.to_csv(
    SCORES_DIR /
    "anthropogenic_mar_scores.csv",
    index=False
)


# ============================================================
# COMBINE
# ============================================================

frames = []

if len(normal_df) > 0:

    frames.append(
        normal_df
    )

if len(anth_df) > 0:

    frames.append(
        anth_df
    )


if len(frames) == 0:

    raise RuntimeError(
        "No evaluation samples were successfully scored."
    )


combined = pd.concat(
    frames,
    ignore_index=True
)


combined_csv = (
    RESULT_ROOT /
    "mar_all_scores.csv"
)

combined.to_csv(
    combined_csv,
    index=False
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    df,
    score_column
):

    if (
        len(df) == 0
        or
        "label" not in df.columns
        or
        score_column not in df.columns
    ):

        return None

    clean = df[
        [
            "label",
            score_column
        ]
    ].dropna()

    if len(clean) == 0:

        return None

    y = clean[
        "label"
    ].values

    scores = clean[
        score_column
    ].values

    if len(
        np.unique(y)
    ) < 2:

        return {
            "n":
                len(y),

            "roc_auc":
                np.nan,

            "pr_auc":
                np.nan,

            "best_f1":
                np.nan,

            "best_threshold":
                np.nan,
        }

    roc = roc_auc_score(
        y,
        scores
    )

    pr = average_precision_score(
        y,
        scores
    )

    precision, recall, thresholds = (
        precision_recall_curve(
            y,
            scores
        )
    )

    f1 = (
        2.0
        *
        precision
        *
        recall
        /
        (
            precision
            +
            recall
            +
            1e-12
        )
    )

    best_idx = int(
        np.nanargmax(f1)
    )

    if best_idx < len(
        thresholds
    ):

        best_threshold = float(
            thresholds[
                best_idx
            ]
        )

    else:

        best_threshold = float(
            np.max(scores)
        )

    return {
        "n":
            len(y),

        "roc_auc":
            float(roc),

        "pr_auc":
            float(pr),

        "best_f1":
            float(
                f1[best_idx]
            ),

        "best_threshold":
            best_threshold,
    }


# ============================================================
# OVERALL SCORE COMPARISON
# ============================================================

metric_rows = []

for score_column in [
    "mar_mean",
    "mar_median",
    "mar_max",
    "mar_min",
]:

    result = calculate_metrics(
        combined,
        score_column
    )

    if result is None:

        continue

    result[
        "score"
    ] = score_column

    metric_rows.append(
        result
    )


metrics_df = pd.DataFrame(
    metric_rows
)

metrics_csv = (
    RESULT_ROOT /
    "mar_metrics.csv"
)

metrics_df.to_csv(
    metrics_csv,
    index=False
)


# ============================================================
# DATASET-WISE METRICS
# ============================================================

domain_rows = []

if (
    "dataset" in combined.columns
    and
    "label" in combined.columns
):

    for dataset in sorted(
        combined[
            "dataset"
        ].dropna().unique()
    ):

        subset = combined[
            combined["dataset"]
            ==
            dataset
        ]

        for score_column in [
            "mar_mean",
            "mar_median",
            "mar_max",
        ]:

            result = calculate_metrics(
                subset,
                score_column
            )

            if result is None:

                continue

            result[
                "dataset"
            ] = dataset

            result[
                "score"
            ] = score_column

            domain_rows.append(
                result
            )


domain_df = pd.DataFrame(
    domain_rows
)

domain_csv = (
    RESULT_ROOT /
    "mar_domain_metrics.csv"
)

domain_df.to_csv(
    domain_csv,
    index=False
)


# ============================================================
# SCORE DISTRIBUTIONS
# ============================================================

distribution_rows = []

if (
    len(combined) > 0
    and
    "label_name" in combined.columns
):

    for label_name, subset in (
        combined.groupby(
            "label_name"
        )
    ):

        for score_column in [
            "mar_mean",
            "mar_median",
            "mar_max",
        ]:

            values = (
                subset[
                    score_column
                ]
                .dropna()
                .values
            )

            if len(values) == 0:

                continue

            distribution_rows.append(
                {
                    "label":
                        label_name,

                    "score":
                        score_column,

                    "n":
                        len(values),

                    "mean":
                        float(
                            np.mean(values)
                        ),

                    "median":
                        float(
                            np.median(values)
                        ),

                    "std":
                        float(
                            np.std(values)
                        ),

                    "p90":
                        float(
                            np.percentile(
                                values,
                                90
                            )
                        ),

                    "p95":
                        float(
                            np.percentile(
                                values,
                                95
                            )
                        ),

                    "p99":
                        float(
                            np.percentile(
                                values,
                                99
                            )
                        ),
                }
            )


distribution_df = pd.DataFrame(
    distribution_rows
)

distribution_csv = (
    RESULT_ROOT /
    "mar_score_distributions.csv"
)

distribution_df.to_csv(
    distribution_csv,
    index=False
)


# ============================================================
# NORMAL-ONLY THRESHOLDS
# ============================================================

threshold_rows = []

if len(normal_df) > 0:

    for score_column in [
        "mar_mean",
        "mar_median",
        "mar_max",
    ]:

        values = (
            normal_df[
                score_column
            ]
            .dropna()
            .values
        )

        if len(values) == 0:

            continue

        threshold_rows.append(
            {
                "score":
                    score_column,

                "normal_p90":
                    float(
                        np.percentile(
                            values,
                            90
                        )
                    ),

                "normal_p95":
                    float(
                        np.percentile(
                            values,
                            95
                        )
                    ),

                "normal_p99":
                    float(
                        np.percentile(
                            values,
                            99
                        )
                    ),
            }
        )


threshold_df = pd.DataFrame(
    threshold_rows
)

threshold_csv = (
    RESULT_ROOT /
    "normal_only_thresholds.csv"
)

threshold_df.to_csv(
    threshold_csv,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "checkpoint":
        str(CHECKPOINT),

    "checkpoint_epoch":
        checkpoint.get(
            "epoch",
            None
        ),

    "normal_count":
        int(
            len(normal_df)
        ),

    "anthropogenic_count":
        int(
            len(anth_df)
        ),

    "num_masks_per_image":
        NUM_MASKS,

    "mask_min_size":
        MASK_MIN_SIZE,

    "mask_max_size":
        MASK_MAX_SIZE,

    "device":
        str(DEVICE),

    "normal_directory":
        str(normal_dir),

    "manifest":
        str(MANIFEST),

    "metrics_file":
        str(metrics_csv),

    "domain_metrics_file":
        str(domain_csv),

    "scores_file":
        str(combined_csv),

    "threshold_file":
        str(threshold_csv),

    "note":
        "MAR checkpoint frozen. "
        "Anomaly score is reconstruction error "
        "inside randomly masked regions."
}


with open(
    RESULT_ROOT /
    "MAR_EVALUATION_SUMMARY.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("MAR EVALUATION COMPLETE")
print("=" * 70)

print(
    "\nSuccessfully scored:"
)

print(
    "Normal:",
    len(normal_df)
)

print(
    "Anthropogenic:",
    len(anth_df)
)

print(
    "\nOverall metrics:"
)

if len(metrics_df) > 0:

    print(
        metrics_df.to_string(
            index=False
        )
    )

print(
    "\nScore distributions:"
)

if len(distribution_df) > 0:

    print(
        distribution_df.to_string(
            index=False
        )
    )

print(
    "\nDataset-wise metrics:"
)

if len(domain_df) > 0:

    print(
        domain_df.to_string(
            index=False
        )
    )

print(
    "\nNormal-only thresholds:"
)

if len(threshold_df) > 0:

    print(
        threshold_df.to_string(
            index=False
        )
    )

print(
    "\nSaved results to:"
)

print(
    RESULT_ROOT
)

print(
    "\nMain metrics:",
    metrics_csv
)

print(
    "All scores:",
    combined_csv
)

print(
    "Domain metrics:",
    domain_csv
)