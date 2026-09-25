"""
SAAD Active Learning — 250 Image Retrospective Retraining Experiment
====================================================================

Goal
----
Compare four acquisition strategies at a fixed acquisition budget:

    1. RANDOM
    2. YOLO_UNCERTAINTY
    3. TTA_UNCERTAINTY
    4. SAAD_UNCERTAINTY

Protocol
--------
Training pool:
    E:/SIH/Datasets/SAAD_baseline

Initial labeled seed:
    2000 images, randomly selected with fixed seed

Acquisition pool:
    Remaining training images

Acquisition budget:
    250 images / strategy

Each model is trained on:
    seed + acquired_250

Evaluation:
    SAME fixed validation set for every strategy

IMPORTANT
---------
The acquisition scores are generated using the frozen full-data
baseline YOLO checkpoint. Therefore this is a retrospective
active-learning simulation, NOT a strict online AL protocol.

Final test set is NEVER touched.
"""

import os
import random
import shutil
import math
import json
import csv
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path(r"E:\SIH\Datasets\SAAD_baseline")

RESULT_ROOT = Path(
    r"E:\SIH\SIH_Results\active_learning_retrain_250"
)

CHECKPOINT = Path(
    r"E:\SIH\SIH_Results\yolo26s_generic_baseline\weights\best.pt"
)

SEED_SIZE = 2000
ACQUISITION_BUDGET = 250

IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 30
PATIENCE = 10
WORKERS = 4

DEVICE = 0
AMP = True

GLOBAL_SEED = 42

# acquisition score weights
W_YOLO = 0.40
W_TTA = 0.30
W_CONSISTENCY = 0.30

# TTA transformations
TTA_CONTRAST_LOW = 0.75
TTA_CONTRAST_HIGH = 1.25
TTA_GAMMA_LOW = 0.75
TTA_GAMMA_HIGH = 1.25

# confidence used for inference
CONF = 0.001

# max detections retained per image
MAX_DET = 300


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


seed_everything(GLOBAL_SEED)


# ============================================================
# DIRECTORIES
# ============================================================

RESULT_ROOT.mkdir(parents=True, exist_ok=True)

ACQUISITION_DIR = RESULT_ROOT / "acquisition"
SUBSET_DIR = RESULT_ROOT / "datasets"
TRAINING_DIR = RESULT_ROOT / "training"

ACQUISITION_DIR.mkdir(parents=True, exist_ok=True)
SUBSET_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CHECKS
# ============================================================

if not DATA_ROOT.exists():
    raise FileNotFoundError(DATA_ROOT)

if not CHECKPOINT.exists():
    raise FileNotFoundError(CHECKPOINT)

train_img_dir = DATA_ROOT / "images" / "train"
train_lbl_dir = DATA_ROOT / "labels" / "train"

val_img_dir = DATA_ROOT / "images" / "val"
val_lbl_dir = DATA_ROOT / "labels" / "val"

if not train_img_dir.exists():
    raise FileNotFoundError(train_img_dir)

if not train_lbl_dir.exists():
    raise FileNotFoundError(train_lbl_dir)

if not val_img_dir.exists():
    raise FileNotFoundError(val_img_dir)


# ============================================================
# HELPERS
# ============================================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    return sorted(
        [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ]
    )


def label_path_for_image(img_path):
    return train_lbl_dir / f"{img_path.stem}.txt"


def normalize01(x):
    x = np.asarray(x, dtype=np.float32)

    if len(x) == 0:
        return x

    lo = np.percentile(x, 5)
    hi = np.percentile(x, 95)

    if hi <= lo:
        return np.zeros_like(x)

    return np.clip((x - lo) / (hi - lo), 0, 1)


def yolo_uncertainty(conf):
    """
    High uncertainty when confidence is low.

    No detections -> uncertainty = 1.
    """
    if len(conf) == 0:
        return 1.0

    max_conf = float(np.max(conf))
    return 1.0 - max_conf


def prediction_summary(result):
    """
    Extract image-level detector statistics.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return {
            "max_conf": 0.0,
            "mean_conf": 0.0,
            "num_det": 0,
        }

    conf = result.boxes.conf.detach().cpu().numpy()

    return {
        "max_conf": float(np.max(conf)),
        "mean_conf": float(np.mean(conf)),
        "num_det": int(len(conf)),
    }


# ============================================================
# TTA IMAGE TRANSFORMS
# ============================================================

def apply_contrast(img, factor):
    """
    img is BGR uint8 numpy array.
    """
    arr = img.astype(np.float32)

    mean = arr.mean(axis=(0, 1), keepdims=True)

    arr = (arr - mean) * factor + mean

    return np.clip(arr, 0, 255).astype(np.uint8)


def apply_gamma(img, gamma):
    """
    Gamma correction for uint8 image.
    """
    arr = img.astype(np.float32) / 255.0

    arr = np.power(np.clip(arr, 0, 1), gamma)

    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def horizontal_flip(img):
    return np.ascontiguousarray(img[:, ::-1])


# ============================================================
# RUN SINGLE IMAGE WITH TTA
# ============================================================

def run_tta(model, image_path):
    """
    Run:
        original
        hflip
        contrast low
        contrast high
        gamma low
        gamma high

    Returns image-level confidence statistics.
    """

    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    variants = {
        "original": img,
        "hflip": horizontal_flip(img),
        "contrast_low": apply_contrast(
            img,
            TTA_CONTRAST_LOW
        ),
        "contrast_high": apply_contrast(
            img,
            TTA_CONTRAST_HIGH
        ),
        "gamma_low": apply_gamma(
            img,
            TTA_GAMMA_LOW
        ),
        "gamma_high": apply_gamma(
            img,
            TTA_GAMMA_HIGH
        ),
    }

    scores = []
    num_dets = []

    for name, variant in variants.items():

        result = model.predict(
            source=variant,
            imgsz=IMG_SIZE,
            conf=CONF,
            max_det=MAX_DET,
            device=DEVICE,
            verbose=False
        )[0]

        summary = prediction_summary(result)

        scores.append(summary["max_conf"])
        num_dets.append(summary["num_det"])

    scores = np.asarray(scores, dtype=np.float32)

    # confidence disagreement
    consistency = 1.0 - (
        np.std(scores) /
        (np.mean(scores) + 1e-6)
    )

    consistency = float(
        np.clip(consistency, 0, 1)
    )

    tta_uncertainty = float(
        np.std(scores)
    )

    return {
        "original_conf": float(scores[0]),
        "tta_mean": float(np.mean(scores)),
        "tta_median": float(np.median(scores)),
        "tta_max": float(np.max(scores)),
        "tta_min": float(np.min(scores)),
        "tta_std": float(np.std(scores)),
        "tta_consistency": consistency,
        "tta_uncertainty": tta_uncertainty,
        "mean_detections": float(np.mean(num_dets)),
    }


# ============================================================
# LOAD TRAIN POOL
# ============================================================

print("=" * 70)
print("SAAD ACTIVE LEARNING — RETRAINING EXPERIMENT")
print("=" * 70)

print("\nDATA ROOT:")
print(DATA_ROOT)

print("\nCHECKPOINT:")
print(CHECKPOINT)

train_images = list_images(train_img_dir)

print("\nFULL TRAIN POOL:", len(train_images))

if len(train_images) <= SEED_SIZE:
    raise RuntimeError(
        "Training pool is not larger than seed size."
    )


# ============================================================
# CREATE INITIAL SEED
# ============================================================

rng = random.Random(GLOBAL_SEED)

shuffled = train_images.copy()
rng.shuffle(shuffled)

seed_images = shuffled[:SEED_SIZE]
acquisition_pool = shuffled[SEED_SIZE:]

print("INITIAL SEED:", len(seed_images))
print("ACQUISITION POOL:", len(acquisition_pool))


seed_names = {p.name for p in seed_images}

# save seed list
pd.DataFrame(
    {
        "image": [str(p) for p in seed_images]
    }
).to_csv(
    ACQUISITION_DIR / "initial_seed.csv",
    index=False
)

pd.DataFrame(
    {
        "image": [str(p) for p in acquisition_pool]
    }
).to_csv(
    ACQUISITION_DIR / "acquisition_pool.csv",
    index=False
)


# ============================================================
# LOAD FROZEN BASELINE
# ============================================================

print("\nLoading frozen baseline...")

model = YOLO(str(CHECKPOINT))

print("Loaded:", CHECKPOINT)


# ============================================================
# SCORE ACQUISITION POOL
# ============================================================

print("\n" + "=" * 70)
print("SCORING ACQUISITION POOL")
print("=" * 70)

rows = []

for i, image_path in enumerate(acquisition_pool, 1):

    try:

        result = model.predict(
            source=str(image_path),
            imgsz=IMG_SIZE,
            conf=CONF,
            max_det=MAX_DET,
            device=DEVICE,
            verbose=False
        )[0]

        summary = prediction_summary(result)

        tta = run_tta(model, image_path)

        yolo_unc = 1.0 - summary["max_conf"]

        tta_unc = tta["tta_uncertainty"]

        consistency_unc = 1.0 - tta["tta_consistency"]

        rows.append(
            {
                "image": str(image_path),
                "filename": image_path.name,

                "max_conf":
                    summary["max_conf"],

                "mean_conf":
                    summary["mean_conf"],

                "num_det":
                    summary["num_det"],

                "yolo_uncertainty":
                    yolo_unc,

                "tta_mean":
                    tta["tta_mean"],

                "tta_median":
                    tta["tta_median"],

                "tta_max":
                    tta["tta_max"],

                "tta_min":
                    tta["tta_min"],

                "tta_std":
                    tta["tta_std"],

                "tta_consistency":
                    tta["tta_consistency"],

                "tta_uncertainty":
                    tta_unc,

                "consistency_uncertainty":
                    consistency_unc,

                "mean_detections":
                    tta["mean_detections"],
            }
        )

    except Exception as e:

        print(
            f"\nWARNING: failed {image_path.name}: {e}"
        )

    if i % 100 == 0 or i == len(acquisition_pool):
        print(
            f"Processed {i}/{len(acquisition_pool)}"
        )


scores_df = pd.DataFrame(rows)

scores_df["yolo_unc_norm"] = normalize01(
    scores_df["yolo_uncertainty"].values
)

scores_df["tta_unc_norm"] = normalize01(
    scores_df["tta_uncertainty"].values
)

scores_df["consistency_unc_norm"] = normalize01(
    scores_df["consistency_uncertainty"].values
)

scores_df["saad_uncertainty"] = (
    W_YOLO * scores_df["yolo_unc_norm"]
    +
    W_TTA * scores_df["tta_unc_norm"]
    +
    W_CONSISTENCY *
    scores_df["consistency_unc_norm"]
)


scores_path = (
    ACQUISITION_DIR /
    "acquisition_pool_scores.csv"
)

scores_df.to_csv(
    scores_path,
    index=False
)

print("\nSaved acquisition scores:")
print(scores_path)


# ============================================================
# ACQUISITION STRATEGIES
# ============================================================

def choose_random(df, budget):
    rng = np.random.default_rng(GLOBAL_SEED)

    idx = rng.choice(
        len(df),
        size=budget,
        replace=False
    )

    return df.iloc[idx].copy()


def choose_top(df, column, budget):
    return df.sort_values(
        column,
        ascending=False
    ).head(budget).copy()


strategies = {}

strategies["RANDOM"] = choose_random(
    scores_df,
    ACQUISITION_BUDGET
)

strategies["YOLO_UNCERTAINTY"] = choose_top(
    scores_df,
    "yolo_uncertainty",
    ACQUISITION_BUDGET
)

strategies["TTA_UNCERTAINTY"] = choose_top(
    scores_df,
    "tta_uncertainty",
    ACQUISITION_BUDGET
)

strategies["SAAD_UNCERTAINTY"] = choose_top(
    scores_df,
    "saad_uncertainty",
    ACQUISITION_BUDGET
)


# ============================================================
# SAVE ACQUISITIONS
# ============================================================

for name, df in strategies.items():

    out = (
        ACQUISITION_DIR /
        f"{name}_250.csv"
    )

    df.to_csv(
        out,
        index=False
    )

    print(
        f"{name:22s}: "
        f"{len(df)} images"
    )


# ============================================================
# DATASET BUILDER
# ============================================================

def build_subset_dataset(
    strategy_name,
    acquired_df
):

    out_root = (
        SUBSET_DIR /
        strategy_name
    )

    out_images = (
        out_root /
        "images" /
        "train"
    )

    out_labels = (
        out_root /
        "labels" /
        "train"
    )

    out_images.mkdir(
        parents=True,
        exist_ok=True
    )

    out_labels.mkdir(
        parents=True,
        exist_ok=True
    )

    selected = list(seed_images)

    selected += [
        Path(x)
        for x in acquired_df["image"].tolist()
    ]

    # remove accidental duplicates
    unique = {}

    for p in selected:
        unique[p.name] = p

    selected = list(unique.values())

    print(
        f"\nBuilding {strategy_name}: "
        f"{len(selected)} images"
    )

    copied = 0
    missing_labels = 0

    for src_img in selected:

        src_lbl = train_lbl_dir / f"{src_img.stem}.txt"

        if not src_img.exists():
            print(
                "WARNING missing image:",
                src_img
            )
            continue

        if not src_lbl.exists():
            print(
                "WARNING missing label:",
                src_lbl
            )
            missing_labels += 1
            continue

        dst_img = out_images / src_img.name
        dst_lbl = out_labels / src_lbl.name

        shutil.copy2(
            src_img,
            dst_img
        )

        shutil.copy2(
            src_lbl,
            dst_lbl
        )

        copied += 1

    yaml_path = (
        out_root /
        "data.yaml"
    )

    yaml_text = f"""
path: {out_root.as_posix()}
train: images/train
val: {val_img_dir.as_posix()}
names:
  0: anthropogenic
"""

    yaml_path.write_text(
        yaml_text.strip(),
        encoding="utf-8"
    )

    metadata = {
        "strategy": strategy_name,
        "seed_size": SEED_SIZE,
        "acquisition_budget": ACQUISITION_BUDGET,
        "total_training_images": copied,
        "missing_labels": missing_labels,
        "validation_images": len(
            list_images(val_img_dir)
        ),
    }

    with open(
        out_root / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    return out_root, yaml_path, copied


# ============================================================
# BUILD ALL SUBSETS
# ============================================================

subset_info = {}

for name, df in strategies.items():

    root, yaml_path, count = build_subset_dataset(
        name,
        df
    )

    subset_info[name] = {
        "root": root,
        "yaml": yaml_path,
        "count": count,
    }


# ============================================================
# TRAINING
# ============================================================

training_summary = []


for strategy_name in [
    "RANDOM",
    "YOLO_UNCERTAINTY",
    "TTA_UNCERTAINTY",
    "SAAD_UNCERTAINTY",
]:

    info = subset_info[strategy_name]

    print("\n")
    print("=" * 70)
    print("TRAINING:", strategy_name)
    print("=" * 70)

    run_dir = (
        TRAINING_DIR /
        strategy_name
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Start from generic YOLO26s pretrained weights,
    # NOT from the full-data SAAD best.pt.
    #
    # This prevents the active-learning comparison from
    # inheriting the full training pool.
    # --------------------------------------------------------

    print("\nInitializing YOLO26s...")

    train_model = YOLO("yolo26s.pt")

    print("Starting training...")

    results = train_model.train(
        data=str(info["yaml"]),

        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,

        workers=WORKERS,
        device=DEVICE,

        patience=PATIENCE,

        amp=AMP,

        seed=GLOBAL_SEED,

        project=str(TRAINING_DIR),
        name=strategy_name,

        exist_ok=True,

        pretrained=True,

        verbose=True,
    )

    # --------------------------------------------------------
    # BEST CHECKPOINT
    # --------------------------------------------------------

    best_pt = (
        TRAINING_DIR /
        strategy_name /
        "weights" /
        "best.pt"
    )

    if not best_pt.exists():

        print(
            "WARNING: best.pt not found for",
            strategy_name
        )

        continue

    # --------------------------------------------------------
    # FIXED VALIDATION EVALUATION
    # --------------------------------------------------------

    print(
        "\nEvaluating fixed validation set..."
    )

    eval_model = YOLO(str(best_pt))

    metrics = eval_model.val(
        data=str(info["yaml"]),
        split="val",

        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,

        workers=WORKERS,
        device=DEVICE,

        verbose=True,
    )

    # --------------------------------------------------------
    # EXTRACT METRICS
    # --------------------------------------------------------

    precision = float(
        metrics.box.mp
    )

    recall = float(
        metrics.box.mr
    )

    map50 = float(
        metrics.box.map50
    )

    map5095 = float(
        metrics.box.map
    )

    map75 = float(
        metrics.box.map75
    )

    row = {
        "strategy": strategy_name,

        "seed_size": SEED_SIZE,

        "acquisition_budget":
            ACQUISITION_BUDGET,

        "total_train_images":
            info["count"],

        "precision":
            precision,

        "recall":
            recall,

        "mAP50":
            map50,

        "mAP50_95":
            map5095,

        "mAP75":
            map75,

        "best_checkpoint":
            str(best_pt),
    }

    training_summary.append(row)

    print("\nRESULT:")
    print(json.dumps(row, indent=2))


# ============================================================
# SAVE FINAL COMPARISON
# ============================================================

comparison = pd.DataFrame(
    training_summary
)

comparison_path = (
    RESULT_ROOT /
    "active_learning_250_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# RANKING
# ============================================================

if len(comparison) > 0:

    ranking = comparison.sort_values(
        "mAP50_95",
        ascending=False
    )

    ranking_path = (
        RESULT_ROOT /
        "active_learning_250_ranking.csv"
    )

    ranking.to_csv(
        ranking_path,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("FINAL ACTIVE LEARNING COMPARISON")
    print("=" * 70)

    print(
        ranking[
            [
                "strategy",
                "precision",
                "recall",
                "mAP50",
                "mAP50_95",
                "mAP75",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(comparison_path)
    print(ranking_path)


# ============================================================
# REPORT
# ============================================================

report_path = (
    RESULT_ROOT /
    "EXPERIMENT_REPORT.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SAAD ACTIVE LEARNING — 250 IMAGE EXPERIMENT\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        "Protocol:\n"
    )

    f.write(
        f"Initial labeled seed: {SEED_SIZE}\n"
    )

    f.write(
        f"Acquisition budget: {ACQUISITION_BUDGET}\n"
    )

    f.write(
        "Strategies: RANDOM, YOLO_UNCERTAINTY, "
        "TTA_UNCERTAINTY, SAAD_UNCERTAINTY\n"
    )

    f.write(
        "Evaluation: fixed validation set\n"
    )

    f.write(
        "Final test set: untouched\n\n"
    )

    f.write(
        "IMPORTANT CAVEAT:\n"
    )

    f.write(
        "Acquisition scores were generated using the "
        "frozen full-data baseline checkpoint. "
        "Therefore this is a retrospective AL simulation "
        "rather than a strict online active-learning protocol.\n\n"
    )

    if len(comparison) > 0:

        f.write(
            comparison.to_string(index=False)
        )

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)

print("\nResults:")
print(RESULT_ROOT)

print("\nFinal comparison:")
print(comparison_path)

print("\nFinal report:")
print(report_path)