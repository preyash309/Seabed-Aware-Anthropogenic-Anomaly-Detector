"""
SAAD ACTIVE LEARNING - ACQUISITION EXPERIMENT
=============================================

Purpose
-------
Simulate an active-learning annotation process using the existing
SAAD training data.

The final 544-image TEST set is NEVER used.

We compare:

    1. RANDOM
    2. YOLO_UNCERTAINTY
    3. TTA_UNCERTAINTY
    4. SAAD_UNCERTAINTY

The goal is to determine whether SAAD can select more informative
images for human annotation than random sampling.

IMPORTANT
---------
This is an acquisition experiment.

It does NOT retrain YOLO yet.

Labels are hidden during acquisition and only used afterward
to analyze what each strategy selected.

This avoids contaminating the final test set.

Outputs
-------
E:\\SIH\\SIH_Results\\active_learning_saad\\

    acquisition_pool.csv
    random_selection.csv
    yolo_uncertainty_selection.csv
    tta_uncertainty_selection.csv
    saad_uncertainty_selection.csv

    strategy_summary.csv
    domain_selection_summary.csv
    class_selection_summary.csv

    ACTIVE_LEARNING_SUMMARY.txt
"""

from pathlib import Path
import random
import warnings

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

TRAIN_IMAGE_DIR = (
    DATASET_ROOT / "images" / "train"
)

TRAIN_LABEL_DIR = (
    DATASET_ROOT / "labels" / "train"
)

MANIFEST_PATH = (
    DATASET_ROOT / "manifest.csv"
)

YOLO_CHECKPOINT = Path(
    r"E:\SIH\SIH_Results\yolo26s_generic_baseline\weights\best.pt"
)

OUT_ROOT = Path(
    r"E:\SIH\SIH_Results\active_learning_saad"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

SEED = 42

DEVICE = 0

IMAGE_SIZE = 640

CONF_SCAN = 0.001

# Keep this modest initially.
MAX_POOL_IMAGES = None

# Acquisition budgets.
ACQUISITION_BUDGETS = [
    100,
    250,
    500,
]

# Maximum number of images to process in this experiment.
# None = entire training pool.
MAX_IMAGES_FOR_INFERENCE = None


# ============================================================
# TTA SETTINGS
# ============================================================

def adjust_contrast(
    image,
    factor
):

    x = image.astype(
        np.float32
    )

    x = (
        (x - 127.5)
        * factor
        + 127.5
    )

    return np.clip(
        x,
        0,
        255
    ).astype(
        np.uint8
    )


def adjust_gamma(
    image,
    gamma
):

    table = np.array(
        [
            (
                (i / 255.0) ** gamma
            ) * 255.0
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

    return {
        "ORIGINAL":
            image.copy(),

        "HFLIP":
            cv2.flip(
                image,
                1
            ),

        "CONTRAST_LOW":
            adjust_contrast(
                image,
                0.85
            ),

        "CONTRAST_HIGH":
            adjust_contrast(
                image,
                1.15
            ),

        "GAMMA_LOW":
            adjust_gamma(
                image,
                0.85
            ),

        "GAMMA_HIGH":
            adjust_gamma(
                image,
                1.15
            ),
    }


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(
    seed=42
):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


seed_everything(
    SEED
)


# ============================================================
# IMAGE HELPERS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def list_training_images():

    paths = []

    for p in TRAIN_IMAGE_DIR.iterdir():

        if (
            p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            paths.append(p)

    paths = sorted(
        paths
    )

    if (
        MAX_IMAGES_FOR_INFERENCE
        is not None
    ):

        paths = paths[
            :MAX_IMAGES_FOR_INFERENCE
        ]

    return paths


def read_yolo_labels(
    label_path
):

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
                    "class":
                        int(cls),

                    "xc":
                        xc,

                    "yc":
                        yc,

                    "w":
                        w,

                    "h":
                        h,
                }
            )

    return rows


# ============================================================
# MANIFEST
# ============================================================

def load_manifest():

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    # Restrict to training split.
    if "split" in manifest.columns:

        manifest = manifest[
            manifest["split"].astype(str).str.lower()
            == "train"
        ].copy()

    return manifest


def build_manifest_lookup(
    manifest
):

    lookup = {}

    for _, row in manifest.iterrows():

        output_image = str(
            row.get(
                "output_image",
                ""
            )
        )

        original_path = str(
            row.get(
                "original_path",
                ""
            )
        )

        for key in [
            Path(output_image).name,
            Path(original_path).name,
        ]:

            if key:
                lookup[key] = row

    return lookup


# ============================================================
# YOLO INFERENCE
# ============================================================

print()
print("=" * 70)
print("SAAD ACTIVE LEARNING ACQUISITION EXPERIMENT")
print("=" * 70)
print()

print(
    f"Checkpoint : {YOLO_CHECKPOINT}"
)

print(
    f"Train pool : {TRAIN_IMAGE_DIR}"
)

print(
    f"Device     : {DEVICE}"
)

print(
    f"Seed       : {SEED}"
)

print()


if not YOLO_CHECKPOINT.exists():

    raise FileNotFoundError(
        f"YOLO checkpoint not found:\n"
        f"{YOLO_CHECKPOINT}"
    )


model = YOLO(
    str(YOLO_CHECKPOINT)
)


# ============================================================
# BOX / CONSISTENCY HELPERS
# ============================================================

def box_iou(
    box_a,
    box_b
):

    ax1, ay1, ax2, ay2 = box_a

    bx1, by1, bx2, by2 = box_b

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

    union = (
        area_a
        + area_b
        - inter
    )

    if union <= 0:

        return 0.0

    return (
        inter / union
    )


def flip_box(
    box,
    width
):

    x1, y1, x2, y2 = box

    return np.array(
        [
            width - x2,
            y1,
            width - x1,
            y2,
        ],
        dtype=np.float32
    )


def prediction_consistency(
    original,
    predictions,
    image_width
):

    original_boxes = (
        original["boxes"]
    )

    if len(original_boxes) == 0:

        # No reference detection.
        #
        # We do NOT call this highly consistent.
        # It is treated as uncertain.
        return 0.0

    scores = []

    for name, pred in predictions.items():

        if name == "ORIGINAL":
            continue

        boxes = pred["boxes"]

        if name == "HFLIP":

            if len(boxes) > 0:

                boxes = np.array(
                    [
                        flip_box(
                            b,
                            image_width
                        )
                        for b in boxes
                    ]
                )

        if len(boxes) == 0:

            scores.append(
                0.0
            )

            continue

        image_matches = []

        for original_box in original_boxes:

            best = 0.0

            for box in boxes:

                best = max(
                    best,
                    box_iou(
                        original_box,
                        box
                    )
                )

            image_matches.append(
                best
            )

        if len(image_matches):

            scores.append(
                np.mean(
                    image_matches
                )
            )

    if not scores:

        return 0.0

    return float(
        np.mean(
            scores
        )
    )


# ============================================================
# INFERENCE FUNCTION
# ============================================================

def run_yolo(
    image
):

    result = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONF_SCAN,
        iou=0.7,
        device=DEVICE,
        verbose=False,
        max_det=300,
    )[0]

    if result.boxes is None:

        return {
            "boxes":
                np.empty(
                    (0, 4),
                    dtype=np.float32
                ),

            "conf":
                np.empty(
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
        "boxes": boxes,
        "conf": conf,
    }


# ============================================================
# TRAINING POOL
# ============================================================

image_paths = list_training_images()

print(
    f"Training pool images: "
    f"{len(image_paths)}"
)

if not image_paths:

    raise RuntimeError(
        "No training images found."
    )


manifest = load_manifest()

manifest_lookup = (
    build_manifest_lookup(
        manifest
    )
)


# ============================================================
# ACQUISITION SCORE CALCULATION
# ============================================================

rows = []


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
            f"[WARNING] Failed to read "
            f"{image_path}"
        )

        continue

    h, w = image.shape[:2]

    # --------------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------------

    original = run_yolo(
        image
    )

    original_conf = (
        float(
            np.max(
                original["conf"]
            )
        )
        if len(original["conf"])
        else 0.0
    )

    # --------------------------------------------------------
    # TTA
    # --------------------------------------------------------

    variants = build_tta_images(
        image
    )

    predictions = {
        "ORIGINAL":
            original
    }

    variant_max_conf = []

    for name, variant_image in variants.items():

        if name == "ORIGINAL":
            pred = original

        else:
            pred = run_yolo(
                variant_image
            )

        predictions[name] = pred

        max_conf = (
            float(
                np.max(
                    pred["conf"]
                )
            )
            if len(pred["conf"])
            else 0.0
        )

        variant_max_conf.append(
            max_conf
        )

    variant_max_conf = np.asarray(
        variant_max_conf,
        dtype=np.float32
    )

    tta_mean = float(
        np.mean(
            variant_max_conf
        )
    )

    tta_std = float(
        np.std(
            variant_max_conf
        )
    )

    tta_consistency = (
        prediction_consistency(
            original,
            predictions,
            w
        )
    )

    # --------------------------------------------------------
    # Uncertainty
    # --------------------------------------------------------

    # YOLO uncertainty:
    # confidence close to 0.5 is uncertain.
    #
    # Since our detector is scanned at low conf,
    # use 1 - max confidence as the primary uncertainty.
    yolo_uncertainty = (
        1.0 - original_conf
    )

    # TTA instability:
    #
    # Larger variance = less stable.
    #
    # Normalize approximately to [0,1].
    tta_instability = float(
        np.clip(
            tta_std / 0.25,
            0.0,
            1.0
        )
    )

    # Prediction inconsistency:
    #
    # 1 - consistency.
    consistency_uncertainty = (
        1.0 - tta_consistency
    )

    # Combined TTA uncertainty.
    tta_uncertainty = (
        0.5 * tta_instability
        +
        0.5 * consistency_uncertainty
    )

    # --------------------------------------------------------
    # SAAD uncertainty
    # --------------------------------------------------------

    # We do not yet use VAE/Flow here because the purpose of
    # this first experiment is to establish a clean TTA-driven
    # acquisition baseline.
    #
    # SAAD uncertainty therefore combines:
    #
    #   YOLO uncertainty
    #   TTA instability
    #   TTA prediction inconsistency
    #
    # Equal weighting is intentional and NOT tuned on test.

    saad_uncertainty = (
        0.40 * yolo_uncertainty
        +
        0.30 * tta_instability
        +
        0.30 * consistency_uncertainty
    )

    # --------------------------------------------------------
    # Dataset metadata
    # --------------------------------------------------------

    lookup_row = manifest_lookup.get(
        image_path.name
    )

    if lookup_row is not None:

        dataset = str(
            lookup_row.get(
                "dataset",
                "UNKNOWN"
            )
        )

        group = str(
            lookup_row.get(
                "group",
                "UNKNOWN"
            )
        )

        source_class = str(
            lookup_row.get(
                "source_class",
                ""
            )
        )

    else:

        dataset = "UNKNOWN"
        group = "UNKNOWN"
        source_class = ""

    rows.append(
        {
            "image":
                image_path.name,

            "path":
                str(image_path),

            "dataset":
                dataset,

            "group":
                group,

            "source_class":
                source_class,

            # Raw detector evidence
            "yolo_conf":
                original_conf,

            "tta_mean_conf":
                tta_mean,

            "tta_std":
                tta_std,

            "tta_consistency":
                tta_consistency,

            # Acquisition signals
            "yolo_uncertainty":
                yolo_uncertainty,

            "tta_instability":
                tta_instability,

            "consistency_uncertainty":
                consistency_uncertainty,

            "tta_uncertainty":
                tta_uncertainty,

            "saad_uncertainty":
                saad_uncertainty,

            "num_original_detections":
                len(
                    original["conf"]
                ),
        }
    )

    if (
        idx % 50 == 0
        or idx == len(image_paths)
    ):

        print(
            f"[{idx:5d}/{len(image_paths)}] "
            f"{image_path.name}"
        )


pool_df = pd.DataFrame(
    rows
)


# ============================================================
# OPTIONAL POOL LIMIT
# ============================================================

if (
    MAX_POOL_IMAGES
    is not None
):

    pool_df = pool_df.head(
        MAX_POOL_IMAGES
    ).copy()


# ============================================================
# ADD DETERMINISTIC RANDOM BASELINE
# ============================================================

rng = np.random.default_rng(
    SEED
)

random_order = rng.permutation(
    len(pool_df)
)

pool_df[
    "random_rank"
] = 0

pool_df.loc[
    random_order,
    "random_rank"
] = np.arange(
    1,
    len(pool_df) + 1
)


# ============================================================
# RANKING
# ============================================================

pool_df[
    "yolo_rank"
] = (
    pool_df[
        "yolo_uncertainty"
    ]
    .rank(
        method="first",
        ascending=False
    )
    .astype(int)
)


pool_df[
    "tta_rank"
] = (
    pool_df[
        "tta_uncertainty"
    ]
    .rank(
        method="first",
        ascending=False
    )
    .astype(int)
)


pool_df[
    "saad_rank"
] = (
    pool_df[
        "saad_uncertainty"
    ]
    .rank(
        method="first",
        ascending=False
    )
    .astype(int)
)


# ============================================================
# SAVE FULL POOL
# ============================================================

pool_df.to_csv(
    OUT_ROOT / "acquisition_pool.csv",
    index=False
)


# ============================================================
# SELECTION FUNCTION
# ============================================================

def select_strategy(
    df,
    strategy,
    budget
):

    if strategy == "RANDOM":

        return (
            df.sort_values(
                "random_rank"
            )
            .head(
                budget
            )
            .copy()
        )

    if strategy == "YOLO_UNCERTAINTY":

        return (
            df.sort_values(
                "yolo_uncertainty",
                ascending=False
            )
            .head(
                budget
            )
            .copy()
        )

    if strategy == "TTA_UNCERTAINTY":

        return (
            df.sort_values(
                "tta_uncertainty",
                ascending=False
            )
            .head(
                budget
            )
            .copy()
        )

    if strategy == "SAAD_UNCERTAINTY":

        return (
            df.sort_values(
                "saad_uncertainty",
                ascending=False
            )
            .head(
                budget
            )
            .copy()
        )

    raise ValueError(
        f"Unknown strategy: {strategy}"
    )


# ============================================================
# STRATEGIES
# ============================================================

STRATEGIES = [
    "RANDOM",
    "YOLO_UNCERTAINTY",
    "TTA_UNCERTAINTY",
    "SAAD_UNCERTAINTY",
]


# ============================================================
# ACQUISITION ANALYSIS
# ============================================================

strategy_summary_rows = []

domain_rows = []

class_rows = []


for strategy in STRATEGIES:

    for budget in ACQUISITION_BUDGETS:

        if budget > len(pool_df):
            continue

        selected = select_strategy(
            pool_df,
            strategy,
            budget
        )

        selected = selected.copy()

        selected[
            "strategy"
        ] = strategy

        selected[
            "budget"
        ] = budget

        # ----------------------------------------------------
        # Save each selection
        # ----------------------------------------------------

        output_name = (
            f"{strategy.lower()}_"
            f"{budget}.csv"
        )

        selected.to_csv(
            OUT_ROOT / output_name,
            index=False
        )

        # ----------------------------------------------------
        # Selection statistics
        # ----------------------------------------------------

        strategy_summary_rows.append(
            {
                "strategy":
                    strategy,

                "budget":
                    budget,

                "selected_N":
                    len(selected),

                "mean_yolo_conf":
                    selected[
                        "yolo_conf"
                    ].mean(),

                "mean_tta_consistency":
                    selected[
                        "tta_consistency"
                    ].mean(),

                "mean_yolo_uncertainty":
                    selected[
                        "yolo_uncertainty"
                    ].mean(),

                "mean_tta_uncertainty":
                    selected[
                        "tta_uncertainty"
                    ].mean(),

                "mean_saad_uncertainty":
                    selected[
                        "saad_uncertainty"
                    ].mean(),

                "median_yolo_conf":
                    selected[
                        "yolo_conf"
                    ].median(),

                "median_tta_consistency":
                    selected[
                        "tta_consistency"
                    ].median(),
            }
        )

        # ----------------------------------------------------
        # Domain distribution
        # ----------------------------------------------------

        domain_counts = (
            selected[
                "dataset"
            ]
            .value_counts(
                dropna=False
            )
        )

        for dataset, count in (
            domain_counts.items()
        ):

            domain_rows.append(
                {
                    "strategy":
                        strategy,

                    "budget":
                        budget,

                    "dataset":
                        dataset,

                    "selected_N":
                        int(count),

                    "fraction":
                        float(
                            count
                            / len(selected)
                        ),
                }
            )

        # ----------------------------------------------------
        # If source class is available
        # ----------------------------------------------------

        if (
            "source_class"
            in selected.columns
        ):

            class_counts = (
                selected[
                    "source_class"
                ]
                .replace(
                    "",
                    "UNKNOWN"
                )
                .value_counts(
                    dropna=False
                )
            )

            for cls, count in (
                class_counts.items()
            ):

                class_rows.append(
                    {
                        "strategy":
                            strategy,

                        "budget":
                            budget,

                        "source_class":
                            cls,

                        "selected_N":
                            int(count),

                        "fraction":
                            float(
                                count
                                / len(selected)
                            ),
                    }
                )


strategy_summary = pd.DataFrame(
    strategy_summary_rows
)

domain_summary = pd.DataFrame(
    domain_rows
)

class_summary = pd.DataFrame(
    class_rows
)


# ============================================================
# SAVE SUMMARIES
# ============================================================

strategy_summary.to_csv(
    OUT_ROOT / "strategy_summary.csv",
    index=False
)

domain_summary.to_csv(
    OUT_ROOT / "domain_selection_summary.csv",
    index=False
)

class_summary.to_csv(
    OUT_ROOT / "class_selection_summary.csv",
    index=False
)


# ============================================================
# OVERLAP ANALYSIS
# ============================================================

overlap_rows = []


for budget in ACQUISITION_BUDGETS:

    if budget > len(pool_df):
        continue

    selected_sets = {}

    for strategy in STRATEGIES:

        selected = select_strategy(
            pool_df,
            strategy,
            budget
        )

        selected_sets[
            strategy
        ] = set(
            selected["image"]
        )

    for i, strategy_a in enumerate(
        STRATEGIES
    ):

        for strategy_b in STRATEGIES[
            i + 1:
        ]:

            a = selected_sets[
                strategy_a
            ]

            b = selected_sets[
                strategy_b
            ]

            intersection = (
                len(a & b)
            )

            union = (
                len(a | b)
            )

            jaccard = (
                intersection / union
                if union > 0
                else 0.0
            )

            overlap_rows.append(
                {
                    "budget":
                        budget,

                    "strategy_A":
                        strategy_a,

                    "strategy_B":
                        strategy_b,

                    "intersection":
                        intersection,

                    "union":
                        union,

                    "jaccard":
                        jaccard,
                }
            )


overlap_df = pd.DataFrame(
    overlap_rows
)

overlap_df.to_csv(
    OUT_ROOT / "strategy_overlap.csv",
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("ACTIVE LEARNING STRATEGY SUMMARY")
print("=" * 70)

print(
    strategy_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


print("\n")
print("=" * 70)
print("DOMAIN SELECTION SUMMARY")
print("=" * 70)

print(
    domain_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


print("\n")
print("=" * 70)
print("STRATEGY OVERLAP")
print("=" * 70)

print(
    overlap_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary_lines = []

summary_lines.append(
    "SAAD ACTIVE LEARNING ACQUISITION EXPERIMENT"
)

summary_lines.append(
    "=" * 60
)

summary_lines.append("")

summary_lines.append(
    "Purpose:"
)

summary_lines.append(
    "Compare active-learning acquisition strategies "
    "without touching the final test set."
)

summary_lines.append("")

summary_lines.append(
    f"Training pool size: {len(pool_df)}"
)

summary_lines.append(
    f"Seed: {SEED}"
)

summary_lines.append("")

summary_lines.append(
    "Strategies:"
)

for strategy in STRATEGIES:

    summary_lines.append(
        f"  - {strategy}"
    )

summary_lines.append("")

summary_lines.append(
    "Acquisition budgets:"
)

summary_lines.append(
    ", ".join(
        str(x)
        for x in ACQUISITION_BUDGETS
    )
)

summary_lines.append("")

summary_lines.append(
    "IMPORTANT:"
)

summary_lines.append(
    "This script does not retrain the detector."
)

summary_lines.append(
    "It evaluates which samples each strategy "
    "would send to a human annotator."
)

summary_lines.append(
    "The final 544-image test set is completely "
    "excluded."
)

summary_lines.append("")

summary_lines.append(
    "The next experiment should retrain YOLO using "
    "the selected subsets and compare performance "
    "against random sampling under identical label budgets."
)


with open(
    OUT_ROOT / "ACTIVE_LEARNING_SUMMARY.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            summary_lines
        )
    )


print("\n")
print("=" * 70)
print("ACTIVE LEARNING ACQUISITION COMPLETE")
print("=" * 70)

print()
print(
    f"Results saved to:\n{OUT_ROOT}"
)

print()
print("DONE.")