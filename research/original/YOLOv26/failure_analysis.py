from pathlib import Path
import json
import math
import shutil
import pandas as pd
import numpy as np
from PIL import Image


# ============================================================
# SAAD BASELINE — FAILURE ANALYSIS
# ============================================================
#
# Purpose:
#   Diagnose WHY the generic YOLO26s baseline fails across
#   GhostVision, AI4Shipwrecks and SubPipeMini2.
#
# This script DOES NOT:
#   - retrain the model
#   - modify best.pt
#   - modify the dataset
#   - run inference
#
# It analyzes:
#   - prediction confidence
#   - prediction/GT IoU
#   - recall vs confidence threshold
#   - localization quality
#   - object scale
#   - oversized predictions
#   - boundary-touching predictions
#   - prediction density
#   - missing prediction images
#
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

MANIFEST_PATH = Path(
    r"E:\SIH\Datasets\SAAD_baseline\manifest.csv"
)

EVAL_ROOT = Path(
    r"E:\SIH\SIH_Results\dataset_wise_eval"
)

OUTPUT_DIR = EVAL_ROOT / "failure_analysis"

DATASETS = [
    "GhostVision",
    "AI4Shipwrecks",
    "SubPipeMini2",
]

# Prediction confidence thresholds to investigate
CONF_THRESHOLDS = [
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
]

# IoU thresholds
IOU_THRESHOLDS = [
    0.30,
    0.50,
    0.75,
]

# A prediction whose area exceeds this fraction of the
# entire image is suspiciously large.
OVERSIZED_AREA_FRACTION = 0.25

# A box touching any image boundary is counted.
BOUNDARY_EPSILON = 2.0


# ============================================================
# 2. PATH HELPERS
# ============================================================

def find_prediction_file(dataset):
    """
    Find predictions.json for a dataset.

    Expected layout:
        dataset_wise_eval/
            GhostVision_evaluation/
                predictions.json

    etc.
    """

    candidates = [
        EVAL_ROOT / f"{dataset}_evaluation" / "predictions.json",
        EVAL_ROOT / dataset / "predictions.json",
        EVAL_ROOT / f"{dataset}_evaluation" / "predictions" / "predictions.json",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Last resort: recursive search
    matches = list(EVAL_ROOT.rglob("predictions.json"))

    dataset_lower = dataset.lower()

    for path in matches:
        if dataset_lower in str(path).lower():
            return path

    return None


# ============================================================
# 3. BOX UTILITIES
# ============================================================

def xywh_to_xyxy(box):
    """
    Convert:
        x, y, width, height

    to:
        x1, y1, x2, y2
    """

    x, y, w, h = map(float, box)

    return [
        x,
        y,
        x + w,
        y + h,
    ]


def box_area(box):
    x1, y1, x2, y2 = box

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)

    return w * h


def intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)

    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)

    return w * h


def iou(a, b):
    inter = intersection_area(a, b)

    if inter <= 0:
        return 0.0

    union = box_area(a) + box_area(b) - inter

    if union <= 0:
        return 0.0

    return inter / union


# ============================================================
# 4. MATCHING
# ============================================================

def greedy_match(gt_boxes, predictions, iou_threshold):
    """
    Greedy one-to-one matching.

    predictions:
        list of dicts containing:
            box
            score

    Returns:
        tp
        fp
        fn
        matched_ious
    """

    if len(gt_boxes) == 0:
        return 0, len(predictions), 0, []

    if len(predictions) == 0:
        return 0, 0, len(gt_boxes), []

    # Sort predictions by confidence
    predictions = sorted(
        predictions,
        key=lambda x: x["score"],
        reverse=True
    )

    matched_gt = set()
    matched_ious = []

    tp = 0

    for pred in predictions:

        best_iou = 0.0
        best_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes):

            if gt_idx in matched_gt:
                continue

            current_iou = iou(
                pred["box"],
                gt_box
            )

            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_idx = gt_idx

        if (
            best_gt_idx is not None
            and best_iou >= iou_threshold
        ):
            matched_gt.add(best_gt_idx)
            tp += 1
            matched_ious.append(best_iou)

    fp = len(predictions) - tp
    fn = len(gt_boxes) - tp

    return tp, fp, fn, matched_ious


# ============================================================
# 5. BEST IOU PER GT
# ============================================================

def best_iou_per_gt(gt_boxes, predictions):
    """
    For every GT object, find the highest IoU prediction.

    This is especially useful for distinguishing:

        confidence problem
        from
        localization problem
    """

    results = []

    for gt_idx, gt_box in enumerate(gt_boxes):

        best = 0.0
        best_score = 0.0
        best_pred_idx = None

        for pred_idx, pred in enumerate(predictions):

            current_iou = iou(
                pred["box"],
                gt_box
            )

            if current_iou > best:
                best = current_iou
                best_score = pred["score"]
                best_pred_idx = pred_idx

        results.append({
            "gt_index": gt_idx,
            "best_iou": best,
            "best_prediction_score": best_score,
            "best_prediction_index": best_pred_idx,
        })

    return results


# ============================================================
# 6. LOAD MANIFEST
# ============================================================

print("=" * 80)
print("SAAD FAILURE ANALYSIS")
print("=" * 80)

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Manifest not found:\n{MANIFEST_PATH}"
    )

manifest = pd.read_csv(MANIFEST_PATH)

required_columns = [
    "dataset",
    "output_image",
    "output_label",
    "split",
]

missing_columns = [
    c for c in required_columns
    if c not in manifest.columns
]

if missing_columns:
    raise RuntimeError(
        f"Manifest is missing columns: {missing_columns}"
    )


# Only held-out test data
manifest_test = manifest[
    manifest["split"].astype(str).str.lower() == "test"
].copy()

manifest_test = manifest_test[
    manifest_test["dataset"].isin(DATASETS)
].copy()

print("\nTest-set manifest counts:")

for dataset in DATASETS:
    subset = manifest_test[
        manifest_test["dataset"] == dataset
    ]

    print(
        f"  {dataset:<20} {len(subset):>5} images"
    )


# ============================================================
# 7. CREATE OUTPUT
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 8. MAIN ANALYSIS
# ============================================================

all_image_rows = []
all_gt_rows = []
summary_rows = []

for dataset in DATASETS:

    print("\n")
    print("=" * 80)
    print(f"DATASET: {dataset}")
    print("=" * 80)

    dataset_manifest = manifest_test[
        manifest_test["dataset"] == dataset
    ].copy()

    prediction_path = find_prediction_file(dataset)

    if prediction_path is None:
        print(
            f"ERROR: predictions.json not found for {dataset}"
        )
        continue

    print(
        f"Predictions: {prediction_path}"
    )

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    with open(
        prediction_path,
        "r",
        encoding="utf-8"
    ) as f:
        predictions_raw = json.load(f)

    if not isinstance(predictions_raw, list):
        raise RuntimeError(
            f"{dataset}: predictions.json is not a list."
        )

    print(
        f"Raw predictions: {len(predictions_raw):,}"
    )

    # --------------------------------------------------------
    # Index predictions by filename
    # --------------------------------------------------------

    prediction_by_filename = {}

    for pred in predictions_raw:

        filename = pred.get("file_name")

        if filename is None:

            image_id = pred.get("image_id")

            if image_id is not None:
                filename = str(image_id)

        if filename is None:
            continue

        filename = Path(str(filename)).name

        bbox = pred.get("bbox")
        score = pred.get("score")

        if bbox is None or score is None:
            continue

        try:
            score = float(score)
            box = xywh_to_xyxy(bbox)
        except Exception:
            continue

        prediction_by_filename.setdefault(
            filename,
            []
        ).append({
            "box": box,
            "score": score,
            "raw_bbox": bbox,
        })

    # --------------------------------------------------------
    # Check prediction image coverage
    # --------------------------------------------------------

    manifest_filenames = set(
        Path(str(x)).name
        for x in dataset_manifest["output_image"]
    )

    prediction_filenames = set(
        prediction_by_filename.keys()
    )

    missing_prediction_images = (
        manifest_filenames -
        prediction_filenames
    )

    extra_prediction_images = (
        prediction_filenames -
        manifest_filenames
    )

    print(
        f"Manifest test images:       {len(manifest_filenames)}"
    )

    print(
        f"Images with predictions:     {len(prediction_filenames & manifest_filenames)}"
    )

    print(
        f"Images with NO predictions:  {len(missing_prediction_images)}"
    )

    if missing_prediction_images:

        print("\nMissing prediction image(s):")

        for filename in sorted(
            missing_prediction_images
        ):
            print(
                f"  {filename}"
            )

    if extra_prediction_images:

        print(
            f"\nExtra prediction filenames: "
            f"{len(extra_prediction_images)}"
        )

    # --------------------------------------------------------
    # Per-image analysis
    # --------------------------------------------------------

    dataset_gt_count = 0

    for _, row in dataset_manifest.iterrows():

        image_path = Path(
            str(row["output_image"])
        )

        label_path = Path(
            str(row["output_label"])
        )

        filename = image_path.name

        # ----------------------------------------------------
        # Image dimensions
        # ----------------------------------------------------

        if image_path.exists():

            try:
                with Image.open(image_path) as im:
                    image_width, image_height = im.size

            except Exception:
                image_width = 640
                image_height = 640

        else:
            image_width = 640
            image_height = 640

        image_area = (
            image_width *
            image_height
        )

        # ----------------------------------------------------
        # Load YOLO GT labels
        # ----------------------------------------------------

        gt_boxes = []

        if label_path.exists():

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

                    try:

                        cls = int(float(parts[0]))
                        cx = float(parts[1])
                        cy = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])

                    except Exception:
                        continue

                    # YOLO normalized -> pixels

                    cx *= image_width
                    cy *= image_height
                    w *= image_width
                    h *= image_height

                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    gt_boxes.append([
                        x1,
                        y1,
                        x2,
                        y2,
                    ])

        dataset_gt_count += len(gt_boxes)

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = prediction_by_filename.get(
            filename,
            []
        )

        # ----------------------------------------------------
        # Confidence statistics
        # ----------------------------------------------------

        scores = [
            p["score"]
            for p in predictions
        ]

        max_score = (
            max(scores)
            if scores
            else 0.0
        )

        mean_score = (
            float(np.mean(scores))
            if scores
            else 0.0
        )

        # ----------------------------------------------------
        # Boundary / oversized statistics
        # ----------------------------------------------------

        oversized_count = 0
        boundary_count = 0

        for pred in predictions:

            x1, y1, x2, y2 = pred["box"]

            area = box_area(
                pred["box"]
            )

            area_fraction = (
                area / image_area
                if image_area > 0
                else 0
            )

            if (
                area_fraction >=
                OVERSIZED_AREA_FRACTION
            ):
                oversized_count += 1

            touches_boundary = (
                x1 <= BOUNDARY_EPSILON
                or y1 <= BOUNDARY_EPSILON
                or x2 >= image_width - BOUNDARY_EPSILON
                or y2 >= image_height - BOUNDARY_EPSILON
            )

            if touches_boundary:
                boundary_count += 1

        # ----------------------------------------------------
        # Per-image row
        # ----------------------------------------------------

        image_result = {
            "dataset": dataset,
            "filename": filename,
            "image_width": image_width,
            "image_height": image_height,
            "gt_count": len(gt_boxes),
            "prediction_count": len(predictions),
            "max_score": max_score,
            "mean_prediction_score": mean_score,
            "oversized_prediction_count": oversized_count,
            "boundary_touching_prediction_count": boundary_count,
        }

        # ----------------------------------------------------
        # Threshold analysis
        # ----------------------------------------------------

        for conf in CONF_THRESHOLDS:

            filtered_predictions = [
                p
                for p in predictions
                if p["score"] >= conf
            ]

            image_result[
                f"pred_count_ge_{conf:.3f}"
            ] = len(filtered_predictions)

            for iou_threshold in IOU_THRESHOLDS:

                tp, fp, fn, matched = greedy_match(
                    gt_boxes,
                    filtered_predictions,
                    iou_threshold
                )

                precision = (
                    tp / (tp + fp)
                    if (tp + fp) > 0
                    else 0.0
                )

                recall = (
                    tp / len(gt_boxes)
                    if len(gt_boxes) > 0
                    else (
                        1.0
                        if len(filtered_predictions) == 0
                        else 0.0
                    )
                )

                image_result[
                    f"tp_c{conf:.3f}_i{iou_threshold:.2f}"
                ] = tp

                image_result[
                    f"fp_c{conf:.3f}_i{iou_threshold:.2f}"
                ] = fp

                image_result[
                    f"fn_c{conf:.3f}_i{iou_threshold:.2f}"
                ] = fn

                image_result[
                    f"precision_c{conf:.3f}_i{iou_threshold:.2f}"
                ] = precision

                image_result[
                    f"recall_c{conf:.3f}_i{iou_threshold:.2f}"
                ] = recall

        # ----------------------------------------------------
        # Best IoU for each GT
        # ----------------------------------------------------

        best_gt_results = best_iou_per_gt(
            gt_boxes,
            predictions
        )

        if best_gt_results:

            best_ious = [
                x["best_iou"]
                for x in best_gt_results
            ]

            image_result["mean_best_gt_iou"] = float(
                np.mean(best_ious)
            )

            image_result["max_gt_iou"] = float(
                np.max(best_ious)
            )

            image_result["gt_best_iou_ge_0.30"] = sum(
                x >= 0.30
                for x in best_ious
            )

            image_result["gt_best_iou_ge_0.50"] = sum(
                x >= 0.50
                for x in best_ious
            )

            image_result["gt_best_iou_ge_0.75"] = sum(
                x >= 0.75
                for x in best_ious
            )

        else:

            image_result["mean_best_gt_iou"] = 0.0
            image_result["max_gt_iou"] = 0.0
            image_result["gt_best_iou_ge_0.30"] = 0
            image_result["gt_best_iou_ge_0.50"] = 0
            image_result["gt_best_iou_ge_0.75"] = 0

        all_image_rows.append(
            image_result
        )

        # ----------------------------------------------------
        # Per-GT rows
        # ----------------------------------------------------

        for gt_info in best_gt_results:

            gt_idx = gt_info["gt_index"]
            gt_box = gt_boxes[gt_idx]

            gt_area = box_area(
                gt_box
            )

            gt_area_fraction = (
                gt_area / image_area
                if image_area > 0
                else 0
            )

            best_prediction_index = (
                gt_info["best_prediction_index"]
            )

            best_prediction_score = (
                gt_info["best_prediction_score"]
            )

            # Determine object size category.
            #
            # These thresholds are deliberately relative to
            # image area because the datasets have different
            # native resolutions.
            if gt_area_fraction < 0.01:
                size_category = "small"
            elif gt_area_fraction < 0.05:
                size_category = "medium"
            else:
                size_category = "large"

            gt_result = {
                "dataset": dataset,
                "filename": filename,
                "gt_index": gt_idx,
                "gt_x1": gt_box[0],
                "gt_y1": gt_box[1],
                "gt_x2": gt_box[2],
                "gt_y2": gt_box[3],
                "gt_area": gt_area,
                "gt_area_fraction": gt_area_fraction,
                "size_category": size_category,
                "best_iou_all_predictions": gt_info["best_iou"],
                "best_prediction_score": best_prediction_score,
                "best_prediction_index": best_prediction_index,
            }

            # For each confidence threshold:
            # what is the best IoU achievable if we only
            # consider predictions above that confidence?
            for conf in CONF_THRESHOLDS:

                filtered_predictions = [
                    p
                    for p in predictions
                    if p["score"] >= conf
                ]

                if filtered_predictions:

                    best_iou_threshold = max(
                        iou(
                            gt_box,
                            p["box"]
                        )
                        for p in filtered_predictions
                    )

                else:

                    best_iou_threshold = 0.0

                gt_result[
                    f"best_iou_c{conf:.3f}"
                ] = best_iou_threshold

            all_gt_rows.append(
                gt_result
            )

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    dataset_image_rows = [
        x
        for x in all_image_rows
        if x["dataset"] == dataset
    ]

    dataset_gt_rows = [
        x
        for x in all_gt_rows
        if x["dataset"] == dataset
    ]

    total_images = len(dataset_image_rows)
    total_gt = len(dataset_gt_rows)
    total_predictions = sum(
        x["prediction_count"]
        for x in dataset_image_rows
    )

    print("\nBasic statistics:")

    print(
        f"  Test images:       {total_images}"
    )

    print(
        f"  GT objects:        {total_gt}"
    )

    print(
        f"  Predictions:       {total_predictions:,}"
    )

    # --------------------------------------------------------
    # Dataset threshold metrics
    # --------------------------------------------------------

    summary = {
        "dataset": dataset,
        "test_images": total_images,
        "gt_objects": total_gt,
        "total_predictions": total_predictions,
        "prediction_images": len(
            prediction_filenames & manifest_filenames
        ),
        "missing_prediction_images": len(
            missing_prediction_images
        ),
    }

    print("\nConfidence threshold analysis:")

    print(
        f"{'CONF':>8} "
        f"{'P@.50':>10} "
        f"{'R@.50':>10} "
        f"{'TP':>8} "
        f"{'FP':>8} "
        f"{'FN':>8}"
    )

    for conf in CONF_THRESHOLDS:

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for image_row in dataset_image_rows:

            total_tp += image_row[
                f"tp_c{conf:.3f}_i0.50"
            ]

            total_fp += image_row[
                f"fp_c{conf:.3f}_i0.50"
            ]

            total_fn += image_row[
                f"fn_c{conf:.3f}_i0.50"
            ]

        precision = (
            total_tp /
            (total_tp + total_fp)
            if total_tp + total_fp > 0
            else 0
        )

        recall = (
            total_tp / total_gt
            if total_gt > 0
            else 0
        )

        print(
            f"{conf:>8.3f} "
            f"{precision:>10.4f} "
            f"{recall:>10.4f} "
            f"{total_tp:>8} "
            f"{total_fp:>8} "
            f"{total_fn:>8}"
        )

        summary[
            f"precision_conf_{conf:.3f}"
        ] = precision

        summary[
            f"recall_conf_{conf:.3f}"
        ] = recall

        summary[
            f"tp_conf_{conf:.3f}"
        ] = total_tp

        summary[
            f"fp_conf_{conf:.3f}"
        ] = total_fp

        summary[
            f"fn_conf_{conf:.3f}"
        ] = total_fn

    # --------------------------------------------------------
    # Localization analysis
    # --------------------------------------------------------

    if dataset_gt_rows:

        best_ious = [
            x["best_iou_all_predictions"]
            for x in dataset_gt_rows
        ]

        best_scores = [
            x["best_prediction_score"]
            for x in dataset_gt_rows
        ]

        summary["mean_best_gt_iou"] = float(
            np.mean(best_ious)
        )

        summary["median_best_gt_iou"] = float(
            np.median(best_ious)
        )

        summary["gt_recall_iou_030"] = (
            sum(x >= 0.30 for x in best_ious)
            / total_gt
        )

        summary["gt_recall_iou_050"] = (
            sum(x >= 0.50 for x in best_ious)
            / total_gt
        )

        summary["gt_recall_iou_075"] = (
            sum(x >= 0.75 for x in best_ious)
            / total_gt
        )

        summary["gt_best_prediction_score_mean"] = float(
            np.mean(best_scores)
        )

        summary["gt_best_prediction_score_median"] = float(
            np.median(best_scores)
        )

        # ----------------------------------------------------
        # How many GT objects have a good localization but
        # weak confidence?
        #
        # This is a key diagnostic.
        # ----------------------------------------------------

        weak_conf_good_localization = sum(
            (
                x["best_iou_all_predictions"] >= 0.50
                and
                x["best_prediction_score"] < 0.10
            )
            for x in dataset_gt_rows
        )

        good_conf_bad_localization = sum(
            (
                x["best_prediction_score"] >= 0.10
                and
                x["best_iou_all_predictions"] < 0.30
            )
            for x in dataset_gt_rows
        )

        good_conf_good_localization = sum(
            (
                x["best_prediction_score"] >= 0.10
                and
                x["best_iou_all_predictions"] >= 0.50
            )
            for x in dataset_gt_rows
        )

        no_reasonable_localization = sum(
            x["best_iou_all_predictions"] < 0.30
            for x in dataset_gt_rows
        )

        summary[
            "weak_conf_good_localization_count"
        ] = weak_conf_good_localization

        summary[
            "good_conf_bad_localization_count"
        ] = good_conf_bad_localization

        summary[
            "good_conf_good_localization_count"
        ] = good_conf_good_localization

        summary[
            "no_reasonable_localization_count"
        ] = no_reasonable_localization

    else:

        summary["mean_best_gt_iou"] = 0
        summary["median_best_gt_iou"] = 0
        summary["gt_recall_iou_030"] = 0
        summary["gt_recall_iou_050"] = 0
        summary["gt_recall_iou_075"] = 0
        summary["gt_best_prediction_score_mean"] = 0
        summary["gt_best_prediction_score_median"] = 0
        summary["weak_conf_good_localization_count"] = 0
        summary["good_conf_bad_localization_count"] = 0
        summary["good_conf_good_localization_count"] = 0
        summary["no_reasonable_localization_count"] = 0

    # --------------------------------------------------------
    # Prediction morphology
    # --------------------------------------------------------

    total_oversized = sum(
        x["oversized_prediction_count"]
        for x in dataset_image_rows
    )

    total_boundary = sum(
        x["boundary_touching_prediction_count"]
        for x in dataset_image_rows
    )

    summary[
        "oversized_predictions"
    ] = total_oversized

    summary[
        "boundary_touching_predictions"
    ] = total_boundary

    summary[
        "oversized_prediction_fraction"
    ] = (
        total_oversized / total_predictions
        if total_predictions > 0
        else 0
    )

    summary[
        "boundary_prediction_fraction"
    ] = (
        total_boundary / total_predictions
        if total_predictions > 0
        else 0
    )

    # --------------------------------------------------------
    # Object size analysis
    # --------------------------------------------------------

    for size in [
        "small",
        "medium",
        "large",
    ]:

        size_rows = [
            x
            for x in dataset_gt_rows
            if x["size_category"] == size
        ]

        if not size_rows:
            summary[
                f"{size}_gt_count"
            ] = 0

            summary[
                f"{size}_recall_iou_050"
            ] = 0

            continue

        summary[
            f"{size}_gt_count"
        ] = len(size_rows)

        summary[
            f"{size}_recall_iou_050"
        ] = (
            sum(
                x["best_iou_all_predictions"] >= 0.50
                for x in size_rows
            )
            / len(size_rows)
        )

    summary_rows.append(
        summary
    )


# ============================================================
# 9. SAVE CSV FILES
# ============================================================

image_df = pd.DataFrame(
    all_image_rows
)

gt_df = pd.DataFrame(
    all_gt_rows
)

summary_df = pd.DataFrame(
    summary_rows
)

image_csv = (
    OUTPUT_DIR /
    "per_image_failure_analysis.csv"
)

gt_csv = (
    OUTPUT_DIR /
    "per_ground_truth_failure_analysis.csv"
)

summary_csv = (
    OUTPUT_DIR /
    "failure_analysis_summary.csv"
)

image_df.to_csv(
    image_csv,
    index=False
)

gt_df.to_csv(
    gt_csv,
    index=False
)

summary_df.to_csv(
    summary_csv,
    index=False
)


# ============================================================
# 10. SAVE MISSING PREDICTION IMAGES
# ============================================================

missing_rows = []

for dataset in DATASETS:

    dataset_manifest = manifest_test[
        manifest_test["dataset"] == dataset
    ]

    prediction_path = find_prediction_file(
        dataset
    )

    if prediction_path is None:
        continue

    with open(
        prediction_path,
        "r",
        encoding="utf-8"
    ) as f:
        preds = json.load(f)

    pred_names = set()

    for pred in preds:

        filename = pred.get(
            "file_name"
        )

        if filename is None:
            image_id = pred.get(
                "image_id"
            )

            if image_id is not None:
                filename = str(
                    image_id
                )

        if filename:
            pred_names.add(
                Path(
                    str(filename)
                ).name
            )

    for _, row in dataset_manifest.iterrows():

        filename = Path(
            str(row["output_image"])
        ).name

        if filename not in pred_names:

            missing_rows.append({
                "dataset": dataset,
                "filename": filename,
                "output_image": row[
                    "output_image"
                ],
                "output_label": row[
                    "output_label"
                ],
            })


missing_df = pd.DataFrame(
    missing_rows
)

missing_csv = (
    OUTPUT_DIR /
    "missing_prediction_images.csv"
)

missing_df.to_csv(
    missing_csv,
    index=False
)


# ============================================================
# 11. TOP FAILURE CASES
# ============================================================

if not gt_df.empty:

    # Worst localized GT objects
    worst_gt = gt_df.sort_values(
        [
            "best_iou_all_predictions",
            "best_prediction_score",
        ],
        ascending=[
            True,
            True,
        ]
    ).head(100)

    worst_gt.to_csv(
        OUTPUT_DIR /
        "worst_localized_ground_truths.csv",
        index=False
    )

    # GT objects where model has good localization
    # but weak confidence.
    confidence_collapse = gt_df[
        (
            gt_df[
                "best_iou_all_predictions"
            ] >= 0.50
        )
        &
        (
            gt_df[
                "best_prediction_score"
            ] < 0.10
        )
    ].sort_values(
        "best_prediction_score"
    )

    confidence_collapse.to_csv(
        OUTPUT_DIR /
        "confidence_collapse_cases.csv",
        index=False
    )

    # High-confidence but poorly localized
    localization_collapse = gt_df[
        (
            gt_df[
                "best_prediction_score"
            ] >= 0.10
        )
        &
        (
            gt_df[
                "best_iou_all_predictions"
            ] < 0.30
        )
    ].sort_values(
        "best_prediction_score",
        ascending=False
    )

    localization_collapse.to_csv(
        OUTPUT_DIR /
        "localization_collapse_cases.csv",
        index=False
    )


# ============================================================
# 12. SAVE JSON SUMMARY
# ============================================================

summary_json = (
    OUTPUT_DIR /
    "failure_analysis_summary.json"
)

with open(
    summary_json,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary_rows,
        f,
        indent=2
    )


# ============================================================
# 13. HUMAN-READABLE FINAL REPORT
# ============================================================

report_path = (
    OUTPUT_DIR /
    "FAILURE_ANALYSIS_REPORT.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SAAD BASELINE FAILURE ANALYSIS\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    for row in summary_rows:

        dataset = row["dataset"]

        f.write(
            f"{dataset}\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            f"Test images: {row['test_images']}\n"
        )

        f.write(
            f"GT objects: {row['gt_objects']}\n"
        )

        f.write(
            f"Predictions: {row['total_predictions']}\n"
        )

        f.write(
            f"Images with predictions: "
            f"{row['prediction_images']}\n"
        )

        f.write(
            f"Missing prediction images: "
            f"{row['missing_prediction_images']}\n\n"
        )

        f.write(
            "Confidence threshold performance "
            "(IoU >= 0.50)\n"
        )

        f.write(
            "conf,precision,recall,tp,fp,fn\n"
        )

        for conf in CONF_THRESHOLDS:

            f.write(
                f"{conf:.3f},"
                f"{row[f'precision_conf_{conf:.3f}']:.4f},"
                f"{row[f'recall_conf_{conf:.3f}']:.4f},"
                f"{row[f'tp_conf_{conf:.3f}']},"
                f"{row[f'fp_conf_{conf:.3f}']},"
                f"{row[f'fn_conf_{conf:.3f}']}\n"
            )

        f.write("\n")

        f.write(
            "Localization\n"
        )

        f.write(
            f"Mean best IoU: "
            f"{row['mean_best_gt_iou']:.4f}\n"
        )

        f.write(
            f"Median best IoU: "
            f"{row['median_best_gt_iou']:.4f}\n"
        )

        f.write(
            f"GT localization recall @ IoU 0.30: "
            f"{row['gt_recall_iou_030']:.4f}\n"
        )

        f.write(
            f"GT localization recall @ IoU 0.50: "
            f"{row['gt_recall_iou_050']:.4f}\n"
        )

        f.write(
            f"GT localization recall @ IoU 0.75: "
            f"{row['gt_recall_iou_075']:.4f}\n"
        )

        f.write("\n")

        f.write(
            "Failure categories\n"
        )

        f.write(
            f"Good localization + weak confidence: "
            f"{row['weak_conf_good_localization_count']}\n"
        )

        f.write(
            f"Good confidence + bad localization: "
            f"{row['good_conf_bad_localization_count']}\n"
        )

        f.write(
            f"Good confidence + good localization: "
            f"{row['good_conf_good_localization_count']}\n"
        )

        f.write(
            f"No reasonable localization (IoU < 0.30): "
            f"{row['no_reasonable_localization_count']}\n"
        )

        f.write("\n")

        f.write(
            "Prediction morphology\n"
        )

        f.write(
            f"Oversized predictions: "
            f"{row['oversized_predictions']}\n"
        )

        f.write(
            f"Oversized fraction: "
            f"{row['oversized_prediction_fraction']:.4f}\n"
        )

        f.write(
            f"Boundary-touching predictions: "
            f"{row['boundary_touching_predictions']}\n"
        )

        f.write(
            f"Boundary prediction fraction: "
            f"{row['boundary_prediction_fraction']:.4f}\n"
        )

        f.write("\n")

        f.write(
            "Object scale\n"
        )

        for size in [
            "small",
            "medium",
            "large",
        ]:

            f.write(
                f"{size}: "
                f"{row[f'{size}_gt_count']} GT, "
                f"recall@IoU0.50="
                f"{row[f'{size}_recall_iou_050']:.4f}\n"
            )

        f.write("\n\n")


# ============================================================
# 14. PRINT FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("FAILURE ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"\nOutput directory:\n{OUTPUT_DIR}"
)

print(
    "\nFiles created:"
)

print(
    f"  {image_csv.name}"
)

print(
    f"  {gt_csv.name}"
)

print(
    f"  {summary_csv.name}"
)

print(
    f"  {missing_csv.name}"
)

print(
    "  worst_localized_ground_truths.csv"
)

print(
    "  confidence_collapse_cases.csv"
)

print(
    "  localization_collapse_cases.csv"
)

print(
    f"  {summary_json.name}"
)

print(
    f"  {report_path.name}"
)


# ============================================================
# 15. CONCISE CONCLUSIONS
# ============================================================

print("\n")
print("=" * 80)
print("QUICK DIAGNOSTIC")
print("=" * 80)

for row in summary_rows:

    print(
        f"\n{row['dataset']}"
    )

    print(
        f"  Recall @ conf=0.001, IoU=.50: "
        f"{row['recall_conf_0.001']:.3f}"
    )

    print(
        f"  Recall @ conf=0.10,  IoU=.50: "
        f"{row['recall_conf_0.100']:.3f}"
    )

    print(
        f"  Mean best GT IoU: "
        f"{row['mean_best_gt_iou']:.3f}"
    )

    print(
        f"  Good localization + weak confidence: "
        f"{row['weak_conf_good_localization_count']}"
    )

    print(
        f"  Good confidence + bad localization: "
        f"{row['good_conf_bad_localization_count']}"
    )

    print(
        f"  No reasonable localization: "
        f"{row['no_reasonable_localization_count']}"
    )

    print(
        f"  Oversized prediction fraction: "
        f"{row['oversized_prediction_fraction']:.3f}"
    )

    print(
        f"  Boundary prediction fraction: "
        f"{row['boundary_prediction_fraction']:.3f}"
    )

print("\n")
print("Do NOT retrain yet.")
print(
    "Use these diagnostics to decide whether the next intervention "
    "should target confidence, localization, scale, or domain shift."
)

print("\nDone.")