"""
SAAD - YOLO Test-Time Augmentation Experiment
==============================================

Goal
----
Evaluate whether test-time augmentation improves the frozen YOLO26s
detector on the SAAD validation set.

IMPORTANT
---------
- No training.
- No checkpoint modification.
- The original YOLO checkpoint remains untouched.
- This experiment is intended for VALIDATION first.
- Do NOT use the final 544-image test set for TTA selection.

TTA variants
------------
1. ORIGINAL
2. HFLIP
3. CONTRAST_LOW
4. CONTRAST_HIGH
5. GAMMA_LOW
6. GAMMA_HIGH
7. COMBINED

For each image we measure:
- maximum detection confidence
- number of detections
- prediction consistency
- confidence stability

For detection evaluation:
- image-level ROC-AUC
- image-level PR-AUC
- best-F1 diagnostic
- object-level precision / recall / mAP through YOLO validation

The primary purpose of this script is to determine whether TTA
provides a useful robustness signal before touching the final test.
"""

from pathlib import Path
import random
import json
import math

import numpy as np
import pandas as pd
import cv2
import torch

from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    r"E:\SIH\Datasets\SAAD_baseline"
)

DATA_YAML = DATASET_ROOT / "data.yaml"

YOLO_CHECKPOINT = Path(
    r"E:\SIH\SIH_Results\yolo26s_generic_baseline\weights\best.pt"
)

RESULT_ROOT = Path(
    r"E:\SIH\SIH_Results\yolo_tta_validation"
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# IMPORTANT:
# Start with validation only.
# ------------------------------------------------------------

EVALUATION_SPLIT = "val"

IMAGE_DIR = DATASET_ROOT / "images" / EVALUATION_SPLIT

LABEL_DIR = DATASET_ROOT / "labels" / EVALUATION_SPLIT

IMAGE_SIZE = 640

CONF_SCAN = 0.001

IOU_MATCH = 0.50

DEVICE = 0

# Batch size for inference.
# RTX 4070 Laptop 8GB should comfortably handle this.
BATCH_SIZE = 16

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42


def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# ============================================================
# TTA DEFINITIONS
# ============================================================

def adjust_contrast(image, factor):
    """
    Contrast around the midpoint.
    """

    x = image.astype(np.float32)

    x = (x - 127.5) * factor + 127.5

    x = np.clip(
        x,
        0,
        255
    )

    return x.astype(np.uint8)


def adjust_gamma(image, gamma):
    """
    Gamma correction.

    gamma < 1:
        brighter

    gamma > 1:
        darker
    """

    table = np.array(
        [
            ((i / 255.0) ** gamma) * 255.0
            for i in range(256)
        ]
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    return cv2.LUT(
        image,
        table
    )


def build_tta_images(image):
    """
    Return TTA variants.

    IMPORTANT:
    We deliberately keep the transformations mild.

    Sonar imagery should not be subjected to arbitrary
    natural-image augmentations.
    """

    variants = {}

    # Original
    variants["ORIGINAL"] = image.copy()

    # Horizontal flip
    variants["HFLIP"] = cv2.flip(
        image,
        1
    )

    # Mild contrast
    variants["CONTRAST_LOW"] = adjust_contrast(
        image,
        0.85
    )

    variants["CONTRAST_HIGH"] = adjust_contrast(
        image,
        1.15
    )

    # Mild gamma
    variants["GAMMA_LOW"] = adjust_gamma(
        image,
        0.85
    )

    variants["GAMMA_HIGH"] = adjust_gamma(
        image,
        1.15
    )

    # Combined mild transformation
    combined = adjust_contrast(
        image,
        1.10
    )

    combined = adjust_gamma(
        combined,
        0.90
    )

    variants["COMBINED"] = combined

    return variants


# ============================================================
# IMAGE / LABEL HELPERS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def list_images():

    paths = []

    for p in IMAGE_DIR.iterdir():

        if (
            p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ):
            paths.append(p)

    return sorted(paths)


def read_yolo_labels(label_path):

    if not label_path.exists():
        return []

    rows = []

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

            cls, xc, yc, w, h = map(
                float,
                parts
            )

            rows.append(
                {
                    "class": int(cls),
                    "xc": xc,
                    "yc": yc,
                    "w": w,
                    "h": h,
                }
            )

    return rows


def yolo_to_xyxy(
    annotation,
    width,
    height
):

    xc = annotation["xc"] * width
    yc = annotation["yc"] * height

    w = annotation["w"] * width
    h = annotation["h"] * height

    x1 = xc - w / 2
    y1 = yc - h / 2

    x2 = xc + w / 2
    y2 = yc + h / 2

    return np.array(
        [
            x1,
            y1,
            x2,
            y2
        ],
        dtype=np.float32
    )


def flip_box_xyxy(
    box,
    width
):

    x1, y1, x2, y2 = box

    return np.array(
        [
            width - x2,
            y1,
            width - x1,
            y2
        ],
        dtype=np.float32
    )


# ============================================================
# BOX MATCHING
# ============================================================

def box_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )

    inter = iw * ih

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

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("SAAD YOLO TEST-TIME AUGMENTATION")
print("=" * 70)

print()
print(f"Checkpoint : {YOLO_CHECKPOINT}")
print(f"Split      : {EVALUATION_SPLIT}")
print(f"Image size : {IMAGE_SIZE}")
print(f"Device     : {DEVICE}")
print()

if not YOLO_CHECKPOINT.exists():

    raise FileNotFoundError(
        f"Checkpoint not found:\n{YOLO_CHECKPOINT}"
    )


model = YOLO(
    str(YOLO_CHECKPOINT)
)


# ============================================================
# IMAGE-LEVEL TTA INFERENCE
# ============================================================

def infer_single(
    image,
    variant_name
):

    results = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONF_SCAN,
        iou=0.7,
        device=DEVICE,
        verbose=False,
        max_det=300,
    )

    result = results[0]

    if result.boxes is None:
        return {
            "variant": variant_name,
            "boxes": np.empty(
                (0, 4),
                dtype=np.float32
            ),
            "conf": np.empty(
                (0,),
                dtype=np.float32
            ),
        }

    boxes = (
        result.boxes.xyxy
        .detach()
        .cpu()
        .numpy()
    )

    conf = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    return {
        "variant": variant_name,
        "boxes": boxes,
        "conf": conf,
    }


# ============================================================
# CONSISTENCY
# ============================================================

def calculate_prediction_consistency(
    original,
    transformed_predictions,
    image_width
):
    """
    Compare each TTA prediction to ORIGINAL predictions.

    For HFLIP, boxes are mapped back into original coordinates.

    For intensity/gamma transforms, coordinates are already
    in original image coordinates.
    """

    original_boxes = original["boxes"]

    if len(original_boxes) == 0:
        return 0.0

    matched_scores = []

    for variant_name, pred in transformed_predictions.items():

        if variant_name == "ORIGINAL":
            continue

        boxes = pred["boxes"]
        confs = pred["conf"]

        if variant_name == "HFLIP":

            boxes = np.array(
                [
                    flip_box_xyxy(
                        b,
                        image_width
                    )
                    for b in boxes
                ]
            )

        if len(boxes) == 0:
            matched_scores.append(0.0)
            continue

        variant_match = []

        for obox in original_boxes:

            best_iou = 0.0

            for vbox in boxes:

                iou = box_iou(
                    obox,
                    vbox
                )

                best_iou = max(
                    best_iou,
                    iou
                )

            variant_match.append(
                best_iou
            )

        matched_scores.append(
            np.mean(
                variant_match
            )
        )

    if len(matched_scores) == 0:
        return 1.0

    return float(
        np.mean(
            matched_scores
        )
    )


# ============================================================
# PER-IMAGE EVALUATION
# ============================================================

image_paths = list_images()

print(
    f"Images found: {len(image_paths)}"
)

if len(image_paths) == 0:

    raise RuntimeError(
        f"No images found in:\n{IMAGE_DIR}"
    )


all_rows = []

variant_rows = []


for idx, image_path in enumerate(
    image_paths,
    start=1
):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image is None:

        print(
            f"[WARNING] Could not read: {image_path}"
        )

        continue

    height, width = image.shape[:2]

    label_path = (
        LABEL_DIR
        / f"{image_path.stem}.txt"
    )

    annotations = read_yolo_labels(
        label_path
    )

    image_label = int(
        len(annotations) > 0
    )

    variants = build_tta_images(
        image
    )

    predictions = {}

    for variant_name, variant_image in variants.items():

        pred = infer_single(
            variant_image,
            variant_name
        )

        predictions[variant_name] = pred

        max_conf = (
            float(
                np.max(pred["conf"])
            )
            if len(pred["conf"]) > 0
            else 0.0
        )

        n_detections = len(
            pred["conf"]
        )

        variant_rows.append(
            {
                "image": image_path.name,
                "path": str(image_path),
                "label": image_label,
                "variant": variant_name,
                "max_conf": max_conf,
                "num_detections": n_detections,
            }
        )

    original = predictions["ORIGINAL"]

    original_max_conf = (
        float(
            np.max(
                original["conf"]
            )
        )
        if len(original["conf"]) > 0
        else 0.0
    )

    consistency = calculate_prediction_consistency(
        original,
        predictions,
        width
    )

    max_confs = []

    for variant_name, pred in predictions.items():

        if len(pred["conf"]) > 0:

            max_confs.append(
                float(
                    np.max(
                        pred["conf"]
                    )
                )
            )

    max_confs = np.asarray(
        max_confs,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Aggregation strategies
    # --------------------------------------------------------

    tta_mean = float(
        np.mean(max_confs)
    )

    tta_median = float(
        np.median(max_confs)
    )

    tta_max = float(
        np.max(max_confs)
    )

    tta_min = float(
        np.min(max_confs)
    )

    tta_std = float(
        np.std(max_confs)
    )

    # Conservative score:
    # confidence multiplied by prediction consistency.
    #
    # This is NOT being claimed as a final SAAD score.
    # It is only an experiment.
    tta_consistency_score = (
        tta_mean * consistency
    )

    all_rows.append(
        {
            "image": image_path.name,
            "path": str(image_path),
            "label": image_label,

            "original_conf":
                original_max_conf,

            "tta_mean":
                tta_mean,

            "tta_median":
                tta_median,

            "tta_max":
                tta_max,

            "tta_min":
                tta_min,

            "tta_std":
                tta_std,

            "tta_consistency":
                consistency,

            "tta_consistency_score":
                tta_consistency_score,

            "num_original_detections":
                len(original["conf"]),
        }
    )

    if (
        idx % 25 == 0
        or idx == len(image_paths)
    ):

        print(
            f"[{idx:4d}/{len(image_paths)}] "
            f"{image_path.name}"
        )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    all_rows
)

variant_df = pd.DataFrame(
    variant_rows
)


# ============================================================
# IMAGE-LEVEL METRICS
# ============================================================

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)


def safe_auc(
    y,
    score
):

    if len(np.unique(y)) < 2:
        return np.nan

    return float(
        roc_auc_score(
            y,
            score
        )
    )


def safe_pr_auc(
    y,
    score
):

    if len(np.unique(y)) < 2:
        return np.nan

    return float(
        average_precision_score(
            y,
            score
        )
    )


def best_f1(
    y,
    score
):

    if len(np.unique(y)) < 2:
        return {
            "f1": np.nan,
            "threshold": np.nan,
            "precision": np.nan,
            "recall": np.nan,
        }

    precision, recall, thresholds = (
        precision_recall_curve(
            y,
            score
        )
    )

    if len(thresholds) == 0:

        return {
            "f1": np.nan,
            "threshold": np.nan,
            "precision": np.nan,
            "recall": np.nan,
        }

    f1 = (
        2.0
        * precision[:-1]
        * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    idx = int(
        np.argmax(f1)
    )

    return {
        "f1": float(f1[idx]),
        "threshold": float(thresholds[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
    }


score_columns = [
    "original_conf",
    "tta_mean",
    "tta_median",
    "tta_max",
    "tta_min",
    "tta_consistency_score",
]

metric_rows = []

y = df["label"].values

for score_name in score_columns:

    score = df[
        score_name
    ].values

    best = best_f1(
        y,
        score
    )

    metric_rows.append(
        {
            "score": score_name,
            "N": len(df),
            "normal": int(
                (y == 0).sum()
            ),
            "anthropogenic": int(
                (y == 1).sum()
            ),
            "ROC_AUC": safe_auc(
                y,
                score
            ),
            "PR_AUC": safe_pr_auc(
                y,
                score
            ),
            "BEST_F1": best["f1"],
            "BEST_F1_THRESHOLD": best["threshold"],
            "BEST_F1_PRECISION": best["precision"],
            "BEST_F1_RECALL": best["recall"],
        }
    )


metrics_df = pd.DataFrame(
    metric_rows
)


# ============================================================
# VARIANT SUMMARY
# ============================================================

variant_summary = (
    variant_df
    .groupby("variant")
    .agg(
        N=("image", "count"),
        mean_conf=("max_conf", "mean"),
        median_conf=("max_conf", "median"),
        P90_conf=("max_conf", lambda x: np.percentile(x, 90)),
        P95_conf=("max_conf", lambda x: np.percentile(x, 95)),
        P99_conf=("max_conf", lambda x: np.percentile(x, 99)),
        mean_detections=("num_detections", "mean"),
        median_detections=("num_detections", "median"),
    )
    .reset_index()
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    RESULT_ROOT / "tta_image_scores.csv",
    index=False
)

variant_df.to_csv(
    RESULT_ROOT / "tta_variant_scores.csv",
    index=False
)

metrics_df.to_csv(
    RESULT_ROOT / "tta_image_level_metrics.csv",
    index=False
)

variant_summary.to_csv(
    RESULT_ROOT / "tta_variant_summary.csv",
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("IMAGE-LEVEL TTA RESULTS")
print("=" * 70)

print(
    metrics_df[
        [
            "score",
            "ROC_AUC",
            "PR_AUC",
            "BEST_F1",
            "BEST_F1_THRESHOLD",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


print("\n")
print("=" * 70)
print("TTA VARIANT SUMMARY")
print("=" * 70)

print(
    variant_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# DETECTION-LEVEL YOLO VALIDATION
# ============================================================

print("\n")
print("=" * 70)
print("YOLO DETECTION VALIDATION")
print("=" * 70)

print(
    "Running standard YOLO validation on the SAME validation split."
)

standard_validation = model.val(
    data=str(DATA_YAML),
    split=EVALUATION_SPLIT,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    conf=0.001,
    iou=0.50,
    device=DEVICE,
    workers=0,
    verbose=False,
)

try:

    box_metrics = standard_validation.box

    detection_summary = {
        "precision": float(
            box_metrics.mp
        ),
        "recall": float(
            box_metrics.mr
        ),
        "mAP50": float(
            box_metrics.map50
        ),
        "mAP50_95": float(
            box_metrics.map
        ),
        "mAP75": float(
            box_metrics.map75
        ),
    }

except Exception:

    detection_summary = {
        "precision": None,
        "recall": None,
        "mAP50": None,
        "mAP50_95": None,
        "mAP75": None,
    }


with open(
    RESULT_ROOT / "detection_validation_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        detection_summary,
        f,
        indent=2
    )


print(
    json.dumps(
        detection_summary,
        indent=2
    )
)


# ============================================================
# FINAL TEXT SUMMARY
# ============================================================

best_pr_row = metrics_df.loc[
    metrics_df["PR_AUC"].idxmax()
]

best_f1_row = metrics_df.loc[
    metrics_df["BEST_F1"].idxmax()
]


summary_lines = []

summary_lines.append(
    "SAAD YOLO TTA VALIDATION EXPERIMENT"
)

summary_lines.append(
    "=" * 60
)

summary_lines.append("")

summary_lines.append(
    f"Validation images: {len(df)}"
)

summary_lines.append(
    f"Normal: {(y == 0).sum()}"
)

summary_lines.append(
    f"Anthropogenic: {(y == 1).sum()}"
)

summary_lines.append("")

summary_lines.append(
    "BEST IMAGE-LEVEL PR-AUC"
)

summary_lines.append(
    f"Score: {best_pr_row['score']}"
)

summary_lines.append(
    f"PR-AUC: {best_pr_row['PR_AUC']:.6f}"
)

summary_lines.append(
    f"ROC-AUC: {best_pr_row['ROC_AUC']:.6f}"
)

summary_lines.append("")

summary_lines.append(
    "BEST IMAGE-LEVEL F1"
)

summary_lines.append(
    f"Score: {best_f1_row['score']}"
)

summary_lines.append(
    f"F1: {best_f1_row['BEST_F1']:.6f}"
)

summary_lines.append(
    f"Threshold: {best_f1_row['BEST_F1_THRESHOLD']:.6f}"
)

summary_lines.append("")

summary_lines.append(
    "IMPORTANT"
)

summary_lines.append(
    "These are validation-only results."
)

summary_lines.append(
    "The final 544-image test set must remain untouched."
)

summary_lines.append(
    "No TTA configuration should be selected using final-test labels."
)

summary_lines.append("")

summary_lines.append(
    "Detection-level standard YOLO validation:"
)

for k, v in detection_summary.items():

    summary_lines.append(
        f"{k}: {v}"
    )


with open(
    RESULT_ROOT / "TTA_VALIDATION_SUMMARY.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(summary_lines)
    )


print("\n")
print("=" * 70)
print(
    "TTA VALIDATION COMPLETE"
)
print("=" * 70)

print(
    f"\nResults saved to:\n{RESULT_ROOT}"
)

print("\nFiles:")

for p in sorted(
    RESULT_ROOT.iterdir()
):

    if p.is_file():

        print(
            f"  {p.name}"
        )

print("\nDONE.")