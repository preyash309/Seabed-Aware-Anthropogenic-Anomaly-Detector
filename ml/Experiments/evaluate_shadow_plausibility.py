import os
import csv
import math
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ultralytics import YOLO
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(r"E:\SIH\Datasets\SAAD_baseline")
MANIFEST_PATH = DATASET_ROOT / "manifest.csv"

YOLO_WEIGHTS = Path(
    r"E:\SIH\SIH_Results\yolo26s_generic_baseline\weights\best.pt"
)

OUTPUT_ROOT = Path(
    r"E:\SIH\SIH_Results\shadow_plausibility"
)

OUTPUT_CSV = OUTPUT_ROOT / "shadow_candidate_scores.csv"
SUMMARY_TXT = OUTPUT_ROOT / "shadow_summary.txt"
PLOTS_DIR = OUTPUT_ROOT / "plots"
EXAMPLES_DIR = OUTPUT_ROOT / "examples"

# Use very low confidence so we don't throw away potentially useful
# candidates before evaluating the physics signal.
YOLO_CONF = 0.001

IOU_MATCH_THRESHOLD = 0.50

IMAGE_SIZE = 640

# Shadow analysis parameters
MIN_SHADOW_LENGTH = 8
MAX_SHADOW_FRACTION = 1.5

# Number of examples to save
N_EXAMPLES = 12

SEED = 42


# ============================================================
# SETUP
# ============================================================

random.seed(SEED)
np.random.seed(SEED)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# UTILITIES
# ============================================================

def iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def load_yolo_labels(label_path):
    """
    YOLO label format:
        class x_center y_center width height

    Coordinates are normalized [0,1].
    """
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls, xc, yc, w, h = map(float, parts)

            boxes.append({
                "class": int(cls),
                "xc": xc,
                "yc": yc,
                "w": w,
                "h": h,
            })

    return boxes


def normalized_box_to_pixel(obj, width, height):
    xc = obj["xc"] * width
    yc = obj["yc"] * height
    bw = obj["w"] * width
    bh = obj["h"] * height

    x1 = xc - bw / 2
    y1 = yc - bh / 2
    x2 = xc + bw / 2
    y2 = yc + bh / 2

    return [
        max(0.0, x1),
        max(0.0, y1),
        min(float(width), x2),
        min(float(height), y2),
    ]


def get_label_path(image_path):
    """
    Converts:
        images/test/foo.jpg

    into:
        labels/test/foo.txt
    """
    image_path = Path(image_path)

    parts = list(image_path.parts)

    try:
        images_index = parts.index("images")
    except ValueError:
        return None

    parts[images_index] = "labels"

    label_path = Path(*parts).with_suffix(".txt")

    return label_path


def match_prediction_to_gt(pred_box, gt_boxes):
    """
    Greedy best-IoU matching.
    Returns:
        matched, best_iou
    """
    best_iou = 0.0

    for gt in gt_boxes:
        current = iou_xyxy(pred_box, gt)
        best_iou = max(best_iou, current)

    return best_iou >= IOU_MATCH_THRESHOLD, best_iou


# ============================================================
# SHADOW ANALYSIS
# ============================================================

def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]

    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(x1 + 1, min(w, int(round(x2))))
    y2 = max(y1 + 1, min(h, int(round(y2))))

    return img[y1:y2, x1:x2]


def local_background_statistics(gray, box):
    """
    Estimate local seabed statistics around a candidate.
    """
    h, w = gray.shape

    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    margin_x = max(5, int(0.5 * bw))
    margin_y = max(5, int(0.5 * bh))

    rx1 = max(0, int(x1 - margin_x))
    ry1 = max(0, int(y1 - margin_y))
    rx2 = min(w, int(x2 + margin_x))
    ry2 = min(h, int(y2 + margin_y))

    region = gray[ry1:ry2, rx1:rx2]

    if region.size == 0:
        return 128.0, 30.0

    # Exclude the candidate itself
    mask = np.ones(region.shape, dtype=bool)

    local_x1 = max(0, int(x1 - rx1))
    local_y1 = max(0, int(y1 - ry1))
    local_x2 = min(region.shape[1], int(x2 - rx1))
    local_y2 = min(region.shape[0], int(y2 - ry1))

    mask[local_y1:local_y2, local_x1:local_x2] = False

    background = region[mask]

    if background.size < 20:
        background = region.flatten()

    median = float(np.median(background))
    mad = float(np.median(np.abs(background - median)))

    return median, max(mad, 1.0)


def analyze_shadow(gray, box):
    """
    Image-only acoustic shadow plausibility.

    Since raw XTF geometry is unavailable, we evaluate the region
    immediately AFTER the candidate along the horizontal image axis.

    IMPORTANT:
    This is a heuristic image-space experiment, not a full physical
    sonar geometry model.
    """

    h, w = gray.shape

    x1, y1, x2, y2 = map(float, box)

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    # --------------------------------------------------------
    # 1. Candidate local background
    # --------------------------------------------------------

    background_median, background_mad = local_background_statistics(
        gray,
        box
    )

    # --------------------------------------------------------
    # 2. Trailing region
    # --------------------------------------------------------

    shadow_length = int(
        max(
            MIN_SHADOW_LENGTH,
            min(
                bw * MAX_SHADOW_FRACTION,
                w * 0.30
            )
        )
    )

    # We evaluate both horizontal directions.
    #
    # Because JPG orientation conventions vary across datasets,
    # we calculate both and retain the stronger one.
    #
    # This avoids falsely assuming "right = shadow".
    # Directionality itself is then captured by the best-vs-worst
    # asymmetry.

    candidates = []

    for direction in [-1, 1]:

        if direction == 1:
            sx1 = int(x2)
            sx2 = min(w, int(x2 + shadow_length))
        else:
            sx1 = max(0, int(x1 - shadow_length))
            sx2 = int(x1)

        sy1 = max(0, int(y1 - 0.15 * bh))
        sy2 = min(h, int(y2 + 0.15 * bh))

        if sx2 <= sx1 or sy2 <= sy1:
            continue

        region = gray[sy1:sy2, sx1:sx2]

        if region.size < 20:
            continue

        # ----------------------------------------------------
        # Darkness score
        # ----------------------------------------------------

        mean_intensity = float(np.mean(region))
        median_intensity = float(np.median(region))

        darkness_difference = (
            background_median - median_intensity
        )

        darkness_score = np.clip(
            darkness_difference /
            max(20.0, 2.0 * background_mad),
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Shadow adjacency score
        # ----------------------------------------------------

        # Examine a narrow strip immediately next to the box.
        adjacency_width = max(3, int(0.10 * bw))

        if direction == 1:
            ax1 = int(x2)
            ax2 = min(w, int(x2 + adjacency_width))
        else:
            ax1 = max(0, int(x1 - adjacency_width))
            ax2 = int(x1)

        ay1 = max(0, int(y1))
        ay2 = min(h, int(y2))

        adjacency = gray[ay1:ay2, ax1:ax2]

        if adjacency.size > 0:
            adjacency_median = float(np.median(adjacency))

            adjacency_difference = (
                background_median - adjacency_median
            )

            adjacency_score = np.clip(
                adjacency_difference /
                max(20.0, 2.0 * background_mad),
                0.0,
                1.0
            )
        else:
            adjacency_score = 0.0

        # ----------------------------------------------------
        # Extent score
        # ----------------------------------------------------

        # For a plausible shadow we want a reasonably large
        # fraction of the trailing region to be darker than
        # local background.

        threshold = (
            background_median -
            max(8.0, 1.5 * background_mad)
        )

        dark_fraction = float(
            np.mean(region < threshold)
        )

        extent_score = np.clip(
            (dark_fraction - 0.15) / 0.60,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Directional continuity
        # ----------------------------------------------------

        # Divide the region into longitudinal sections and see
        # whether darkness persists away from the object.

        section_count = 4

        section_scores = []

        for k in range(section_count):

            start = int(k * region.shape[1] / section_count)
            end = int((k + 1) * region.shape[1] / section_count)

            section = region[:, start:end]

            if section.size == 0:
                continue

            section_dark_fraction = float(
                np.mean(section < threshold)
            )

            section_scores.append(section_dark_fraction)

        if section_scores:

            # We want some persistence into the trailing region.
            later_sections = section_scores[1:]

            if later_sections:
                persistence = float(np.mean(later_sections))
            else:
                persistence = section_scores[0]

            direction_score = np.clip(
                (persistence - 0.10) / 0.60,
                0.0,
                1.0
            )
        else:
            direction_score = 0.0

        # ----------------------------------------------------
        # Combined score
        # ----------------------------------------------------

        total_score = (
            0.25 * darkness_score
            + 0.25 * adjacency_score
            + 0.25 * extent_score
            + 0.25 * direction_score
        )

        candidates.append({
            "direction": direction,
            "darkness_score": darkness_score,
            "adjacency_score": adjacency_score,
            "extent_score": extent_score,
            "direction_score": direction_score,
            "shadow_score": total_score,
            "shadow_length_px": shadow_length,
            "background_median": background_median,
            "background_mad": background_mad,
            "dark_fraction": dark_fraction,
        })

    if not candidates:
        return {
            "shadow_direction": 0,
            "darkness_score": 0.0,
            "adjacency_score": 0.0,
            "extent_score": 0.0,
            "direction_score": 0.0,
            "shadow_score": 0.0,
            "shadow_length_px": 0,
            "background_median": background_median,
            "background_mad": background_mad,
            "dark_fraction": 0.0,
        }

    # Select stronger of left/right.
    best = max(
        candidates,
        key=lambda x: x["shadow_score"]
    )

    return {
        "shadow_direction": best["direction"],
        "darkness_score": best["darkness_score"],
        "adjacency_score": best["adjacency_score"],
        "extent_score": best["extent_score"],
        "direction_score": best["direction_score"],
        "shadow_score": best["shadow_score"],
        "shadow_length_px": best["shadow_length_px"],
        "background_median": best["background_median"],
        "background_mad": best["background_mad"],
        "dark_fraction": best["dark_fraction"],
    }


# ============================================================
# VISUALIZATION
# ============================================================

def save_candidate_visual(
    image,
    box,
    score_info,
    output_path,
    title,
    matched
):
    vis = image.copy()

    x1, y1, x2, y2 = map(int, box)

    # Candidate box
    if matched:
        box_color = (0, 220, 0)
    else:
        box_color = (0, 0, 255)

    cv2.rectangle(
        vis,
        (x1, y1),
        (x2, y2),
        box_color,
        2
    )

    direction = score_info["shadow_direction"]
    shadow_length = score_info["shadow_length_px"]

    if direction == 1:

        sx1 = x2
        sx2 = min(
            vis.shape[1],
            x2 + shadow_length
        )

    else:

        sx1 = max(
            0,
            x1 - shadow_length
        )
        sx2 = x1

    sy1 = max(
        0,
        int(y1 - 0.15 * (y2 - y1))
    )

    sy2 = min(
        vis.shape[0],
        int(y2 + 0.15 * (y2 - y1))
    )

    overlay = vis.copy()

    cv2.rectangle(
        overlay,
        (sx1, sy1),
        (sx2, sy2),
        (255, 0, 255),
        -1
    )

    vis = cv2.addWeighted(
        overlay,
        0.20,
        vis,
        0.80,
        0
    )

    text = (
        f"{title} | "
        f"shadow={score_info['shadow_score']:.3f}"
    )

    cv2.putText(
        vis,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imwrite(
        str(output_path),
        vis
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SAAD ACOUSTIC-SHADOW PLAUSIBILITY EXPERIMENT")
    print("=" * 70)

    print(f"Dataset : {DATASET_ROOT}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"YOLO    : {YOLO_WEIGHTS}")
    print(f"Output  : {OUTPUT_ROOT}")
    print()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(
            f"YOLO checkpoint not found: {YOLO_WEIGHTS}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    # --------------------------------------------------------
    # Test set only
    # --------------------------------------------------------

    test_rows = manifest[
        manifest["split"].astype(str).str.lower() == "test"
    ].copy()

    # We need images with anthropogenic GT so that TP/FP can
    # actually be evaluated against the existing labels.
    test_rows = test_rows[
        test_rows["has_annotation"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
    ].copy()

    print(f"Test images selected: {len(test_rows)}")
    print()

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    print("Loading YOLO model...")

    model = YOLO(
        str(YOLO_WEIGHTS)
    )

    print("YOLO loaded.")
    print()

    rows = []

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for idx, manifest_row in enumerate(
        test_rows.itertuples(index=False),
        start=1
    ):

        image_path = Path(
            getattr(manifest_row, "output_image")
        )

        if not image_path.is_absolute():
            image_path = DATASET_ROOT / image_path

        if not image_path.exists():
            print(
                f"[WARNING] Missing image: {image_path}"
            )
            continue

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            print(
                f"[WARNING] Could not read: {image_path}"
            )
            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        height, width = gray.shape

        # ----------------------------------------------------
        # GT
        # ----------------------------------------------------

        label_path = get_label_path(
            image_path
        )

        gt_raw = load_yolo_labels(
            label_path
        )

        gt_boxes = [
            normalized_box_to_pixel(
                obj,
                width,
                height
            )
            for obj in gt_raw
            if obj["class"] == 0
        ]

        # ----------------------------------------------------
        # YOLO prediction
        # ----------------------------------------------------

        result = model.predict(
            source=str(image_path),
            conf=YOLO_CONF,
            imgsz=IMAGE_SIZE,
            verbose=False,
            device=0
        )[0]

        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for pred_idx, (box, conf) in enumerate(
            zip(boxes, confs)
        ):

            box = box.astype(float).tolist()

            matched, best_iou = match_prediction_to_gt(
                box,
                gt_boxes
            )

            shadow = analyze_shadow(
                gray,
                box
            )

            row = {
                "image": str(image_path),
                "dataset": getattr(
                    manifest_row,
                    "dataset",
                    ""
                ),
                "prediction_id": pred_idx,
                "x1": box[0],
                "y1": box[1],
                "x2": box[2],
                "y2": box[3],
                "yolo_conf": float(conf),
                "matched_gt": int(matched),
                "best_iou": float(best_iou),
                "shadow_direction": shadow[
                    "shadow_direction"
                ],
                "shadow_length_px": shadow[
                    "shadow_length_px"
                ],
                "shadow_darkness_score": shadow[
                    "darkness_score"
                ],
                "shadow_adjacency_score": shadow[
                    "adjacency_score"
                ],
                "shadow_extent_score": shadow[
                    "extent_score"
                ],
                "shadow_direction_score": shadow[
                    "direction_score"
                ],
                "shadow_score": shadow[
                    "shadow_score"
                ],
                "local_background_median": shadow[
                    "background_median"
                ],
                "local_background_mad": shadow[
                    "background_mad"
                ],
                "dark_fraction": shadow[
                    "dark_fraction"
                ],
            }

            rows.append(row)

        if idx % 25 == 0 or idx == len(test_rows):
            print(
                f"Processed {idx}/{len(test_rows)} "
                f"images | candidates={len(rows)}"
            )

    # --------------------------------------------------------
    # Save raw candidate data
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(
            "No YOLO predictions were produced."
        )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print()
    print(
        f"Candidate CSV saved: {OUTPUT_CSV}"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    tp = df[df["matched_gt"] == 1]
    fp = df[df["matched_gt"] == 0]

    print()
    print("=" * 70)
    print("SHADOW SCORE RESULTS")
    print("=" * 70)

    print(
        f"Total predictions : {len(df)}"
    )
    print(
        f"TP predictions    : {len(tp)}"
    )
    print(
        f"FP predictions    : {len(fp)}"
    )

    print()

    if len(tp) > 0:
        print("TRUE POSITIVES")
        print(
            f"Mean   : {tp.shadow_score.mean():.6f}"
        )
        print(
            f"Median : {tp.shadow_score.median():.6f}"
        )
        print(
            f"P90    : {tp.shadow_score.quantile(0.90):.6f}"
        )
        print(
            f"P95    : {tp.shadow_score.quantile(0.95):.6f}"
        )

    print()

    if len(fp) > 0:
        print("FALSE POSITIVES")
        print(
            f"Mean   : {fp.shadow_score.mean():.6f}"
        )
        print(
            f"Median : {fp.shadow_score.median():.6f}"
        )
        print(
            f"P90    : {fp.shadow_score.quantile(0.90):.6f}"
        )
        print(
            f"P95    : {fp.shadow_score.quantile(0.95):.6f}"
        )

    # --------------------------------------------------------
    # ROC / PR
    # --------------------------------------------------------

    y_true = df["matched_gt"].values
    y_score = df["shadow_score"].values

    roc_auc = float("nan")
    pr_auc = float("nan")

    if len(np.unique(y_true)) == 2:

        roc_auc = roc_auc_score(
            y_true,
            y_score
        )

        pr_auc = average_precision_score(
            y_true,
            y_score
        )

        print()
        print(
            f"Shadow ROC-AUC : {roc_auc:.6f}"
        )

        print(
            f"Shadow PR-AUC  : {pr_auc:.6f}"
        )

    # --------------------------------------------------------
    # Threshold table
    # --------------------------------------------------------

    thresholds = np.linspace(
        0.0,
        1.0,
        101
    )

    threshold_rows = []

    for threshold in thresholds:

        predicted = (
            df["shadow_score"].values >= threshold
        )

        tp_count = int(
            np.sum(
                predicted & (y_true == 1)
            )
        )

        fp_count = int(
            np.sum(
                predicted & (y_true == 0)
            )
        )

        fn_count = int(
            np.sum(
                (~predicted) & (y_true == 1)
            )
        )

        tn_count = int(
            np.sum(
                (~predicted) & (y_true == 0)
            )
        )

        precision = (
            tp_count /
            max(1, tp_count + fp_count)
        )

        recall = (
            tp_count /
            max(1, tp_count + fn_count)
        )

        f1 = (
            2 * precision * recall /
            max(1e-12, precision + recall)
        )

        threshold_rows.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp_count,
            "fp": fp_count,
            "fn": fn_count,
            "tn": tn_count,
        })

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    best_row = threshold_df.loc[
        threshold_df["f1"].idxmax()
    ]

    print()
    print("BEST SHADOW-ONLY F1")
    print(
        f"Threshold : {best_row.threshold:.6f}"
    )
    print(
        f"Precision : {best_row.precision:.6f}"
    )
    print(
        f"Recall    : {best_row.recall:.6f}"
    )
    print(
        f"F1        : {best_row.f1:.6f}"
    )

    threshold_df.to_csv(
        OUTPUT_ROOT / "shadow_thresholds.csv",
        index=False
    )

    # --------------------------------------------------------
    # Plot score distributions
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    if len(tp) > 0:
        plt.hist(
            tp["shadow_score"],
            bins=50,
            alpha=0.60,
            density=True,
            label="YOLO True Positives"
        )

    if len(fp) > 0:
        plt.hist(
            fp["shadow_score"],
            bins=50,
            alpha=0.60,
            density=True,
            label="YOLO False Positives"
        )

    plt.xlabel("Acoustic-shadow plausibility score")
    plt.ylabel("Density")
    plt.title("Shadow Plausibility: TP vs FP")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "tp_vs_fp_shadow_distribution.png",
        dpi=180
    )

    plt.close()

    # --------------------------------------------------------
    # Dataset-wise statistics
    # --------------------------------------------------------

    dataset_rows = []

    for dataset_name, group in df.groupby("dataset"):

        group_tp = group[
            group["matched_gt"] == 1
        ]

        group_fp = group[
            group["matched_gt"] == 0
        ]

        dataset_rows.append({
            "dataset": dataset_name,
            "predictions": len(group),
            "tp_predictions": len(group_tp),
            "fp_predictions": len(group_fp),
            "tp_mean_shadow": (
                group_tp["shadow_score"].mean()
                if len(group_tp) else np.nan
            ),
            "tp_median_shadow": (
                group_tp["shadow_score"].median()
                if len(group_tp) else np.nan
            ),
            "fp_mean_shadow": (
                group_fp["shadow_score"].mean()
                if len(group_fp) else np.nan
            ),
            "fp_median_shadow": (
                group_fp["shadow_score"].median()
                if len(group_fp) else np.nan
            ),
        })

    dataset_df = pd.DataFrame(
        dataset_rows
    )

    dataset_df.to_csv(
        OUTPUT_ROOT / "dataset_wise_shadow_scores.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save strongest examples
    # --------------------------------------------------------

    def save_examples(subset, prefix):

        if len(subset) == 0:
            return

        strongest = subset.nlargest(
            N_EXAMPLES,
            "shadow_score"
        )

        for rank, (_, row) in enumerate(
            strongest.iterrows(),
            start=1
        ):

            image_path = Path(
                row["image"]
            )

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                continue

            box = [
                row["x1"],
                row["y1"],
                row["x2"],
                row["y2"],
            ]

            score_info = {
                "shadow_direction":
                    row["shadow_direction"],
                "shadow_length_px":
                    row["shadow_length_px"],
                "shadow_score":
                    row["shadow_score"],
            }

            filename = (
                f"{prefix}_{rank:02d}_"
                f"score_{row['shadow_score']:.3f}.jpg"
            )

            save_candidate_visual(
                image,
                box,
                score_info,
                EXAMPLES_DIR / filename,
                prefix,
                row["matched_gt"] == 1
            )

    save_examples(
        tp,
        "TP"
    )

    save_examples(
        fp,
        "FP"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    with open(
        SUMMARY_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SAAD ACOUSTIC-SHADOW PLAUSIBILITY EXPERIMENT\n"
        )
        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "IMPORTANT:\n"
        )

        f.write(
            "This is an image-only shadow plausibility experiment.\n"
        )

        f.write(
            "Raw XTF sonar geometry was not available, so this\n"
        )

        f.write(
            "experiment does NOT claim absolute target height,\n"
        )

        f.write(
            "slant-range geometry, or full sonar-equation validation.\n\n"
        )

        f.write(
            f"YOLO confidence threshold: {YOLO_CONF}\n"
        )

        f.write(
            f"IoU match threshold: {IOU_MATCH_THRESHOLD}\n\n"
        )

        f.write(
            f"Total predictions: {len(df)}\n"
        )

        f.write(
            f"TP predictions: {len(tp)}\n"
        )

        f.write(
            f"FP predictions: {len(fp)}\n\n"
        )

        if len(tp):

            f.write("TRUE POSITIVES\n")
            f.write(
                f"Mean: {tp.shadow_score.mean():.6f}\n"
            )
            f.write(
                f"Median: {tp.shadow_score.median():.6f}\n"
            )
            f.write(
                f"P90: {tp.shadow_score.quantile(.90):.6f}\n"
            )
            f.write(
                f"P95: {tp.shadow_score.quantile(.95):.6f}\n\n"
            )

        if len(fp):

            f.write("FALSE POSITIVES\n")
            f.write(
                f"Mean: {fp.shadow_score.mean():.6f}\n"
            )
            f.write(
                f"Median: {fp.shadow_score.median():.6f}\n"
            )
            f.write(
                f"P90: {fp.shadow_score.quantile(.90):.6f}\n"
            )
            f.write(
                f"P95: {fp.shadow_score.quantile(.95):.6f}\n\n"
            )

        f.write(
            f"Shadow ROC-AUC: {roc_auc:.6f}\n"
        )

        f.write(
            f"Shadow PR-AUC: {pr_auc:.6f}\n\n"
        )

        f.write(
            "BEST SHADOW-ONLY F1\n"
        )

        f.write(
            f"Threshold: {best_row.threshold:.6f}\n"
        )

        f.write(
            f"Precision: {best_row.precision:.6f}\n"
        )

        f.write(
            f"Recall: {best_row.recall:.6f}\n"
        )

        f.write(
            f"F1: {best_row.f1:.6f}\n"

        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Candidate scores : {OUTPUT_CSV}"
    )

    print(
        f"Summary          : {SUMMARY_TXT}"
    )

    print(
        f"Examples         : {EXAMPLES_DIR}"
    )

    print(
        f"Plots            : {PLOTS_DIR}"
    )


if __name__ == "__main__":
    main()