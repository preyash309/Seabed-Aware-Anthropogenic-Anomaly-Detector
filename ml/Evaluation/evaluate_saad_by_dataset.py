"""
SAAD — DATASET-WISE EVALUATION
================================

Evaluates the frozen SAAD YOLO26s baseline separately on:

    1. GhostVision
    2. AI4Shipwrecks
    3. SubPipeMini2

IMPORTANT:
- The SAAD manifest is the authoritative source for image/label paths.
- Only rows with split == "test" are evaluated.
- The model is NOT retrained.
- Each dataset is evaluated independently.
- Temporary evaluation datasets are created using hardlinks when possible.
- Ultralytics requires train/val/test keys in the temporary YAML, even
  though only the test split is actually evaluated.
"""

from pathlib import Path
import json
import shutil
import hashlib
import sys
import traceback

import pandas as pd
import torch
from ultralytics import YOLO


# ======================================================================
# CONFIGURATION
# ======================================================================

MODEL_PATH = Path(
    r"E:\SIH\SIH_Results\yolo26s_generic_baseline\weights\best.pt"
)

MANIFEST_PATH = Path(
    r"E:\SIH\Datasets\SAAD_baseline\manifest.csv"
)

OUTPUT_DIR = Path(
    r"E:\SIH\SIH_Results\dataset_wise_eval"
)

TARGET_DATASETS = [
    "GhostVision",
    "AI4Shipwrecks",
    "SubPipeMini2",
]

# Inference / validation settings.
IMGSZ = 640
BATCH = 16
WORKERS = 4
DEVICE = 0

# Very low confidence is intentional for mAP evaluation.
# Ultralytics evaluates the full confidence/precision-recall curve.
CONF = 0.001

# NMS IoU.
IOU = 0.7

# Deterministic seed.
SEED = 42


# ======================================================================
# PRINT HELPERS
# ======================================================================

def print_header(title: str):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_subheader(title: str):
    print()
    print(title)
    print("-" * 50)


# ======================================================================
# ENVIRONMENT CHECK
# ======================================================================

def check_environment():
    print_header("ENVIRONMENT")

    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")

        try:
            vram_gb = (
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3)
            )
            print(f"VRAM    : {vram_gb:.2f} GB")
        except Exception:
            print("VRAM    : unavailable")
    else:
        print("GPU     : CPU ONLY")

    print()
    print("Model:")
    print(MODEL_PATH)

    print()
    print("Manifest:")
    print(MANIFEST_PATH)

    print()
    print("Output:")
    print(OUTPUT_DIR)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model does not exist:\n{MODEL_PATH}"
        )

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest does not exist:\n{MANIFEST_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "This evaluation is configured for GPU execution."
        )


# ======================================================================
# MANIFEST LOADING
# ======================================================================

def load_manifest() -> pd.DataFrame:
    print_header("LOADING MANIFEST")

    df = pd.read_csv(MANIFEST_PATH)

    print(f"Total manifest rows: {len(df)}")

    required_columns = [
        "dataset",
        "output_image",
        "output_label",
        "split",
    ]

    missing_columns = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Manifest is missing required columns:\n"
            + "\n".join(missing_columns)
        )

    print()
    print("Dataset counts:")
    print(df["dataset"].value_counts())

    print()
    print("Split counts:")
    print(df["split"].value_counts())

    test_df = df[df["split"].astype(str).str.lower() == "test"].copy()

    print()
    print(f"Total TEST rows in manifest: {len(test_df)}")

    if len(test_df) == 0:
        raise RuntimeError("No test rows found in manifest.")

    return df


# ======================================================================
# FROZEN MODEL LOADING
# ======================================================================

def load_model() -> YOLO:
    print_header("LOADING FROZEN MODEL")

    model = YOLO(str(MODEL_PATH))

    print("Model loaded:")
    print(MODEL_PATH)

    return model


# ======================================================================
# PATH NORMALIZATION
# ======================================================================

def normalize_path(value) -> Path:
    """
    Convert manifest path into a Path object.

    The manifest contains Windows absolute paths such as:

        E:\\SIH\\Datasets\\SAAD_baseline\\images\\test\\...

    We intentionally use the paths exactly as supplied rather than
    reconstructing them from filenames.
    """

    if pd.isna(value):
        return Path("")

    return Path(str(value))


# ======================================================================
# FILE HASH
# ======================================================================

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ======================================================================
# HARDLINK / COPY
# ======================================================================

def link_or_copy(src: Path, dst: Path):
    """
    Try hardlink first.

    This avoids duplicating hundreds of MB of image data.

    Falls back to copy2 if hardlinks are unavailable.
    """

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        dst.unlink()

    try:
        os_link = getattr(__import__("os"), "link")
        os_link(src, dst)

    except Exception:
        shutil.copy2(src, dst)


# ======================================================================
# CREATE TEMPORARY DATASET
# ======================================================================

def create_eval_dataset(
    dataset_name: str,
    dataset_df: pd.DataFrame,
) -> Path:
    """
    Creates:

        OUTPUT_DIR/
            _eval_<dataset>/
                images/
                labels/
                dataset.yaml

    IMPORTANT:
    Ultralytics requires train and val keys even when we only call
    model.val(..., split='test').

    Therefore the YAML contains:

        train: images
        val: images
        test: images

    but model.val() explicitly evaluates split='test'.
    """

    safe_name = dataset_name.lower().replace(" ", "_")

    eval_root = OUTPUT_DIR / f"_eval_{safe_name}"

    # Start fresh.
    if eval_root.exists():
        shutil.rmtree(eval_root)

    images_dir = eval_root / "images"
    labels_dir = eval_root / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    copied_labels = 0

    print()
    print("Temporary evaluation dataset:")
    print(eval_root)

    # --------------------------------------------------------------
    # Copy/link files referenced EXACTLY by the manifest.
    # --------------------------------------------------------------

    for idx, row in dataset_df.iterrows():

        src_image = normalize_path(row["output_image"])
        src_label = normalize_path(row["output_label"])

        if not src_image.exists():
            raise FileNotFoundError(
                f"Manifest image does not exist:\n{src_image}"
            )

        if not src_label.exists():
            raise FileNotFoundError(
                f"Manifest label does not exist:\n{src_label}"
            )

        image_dst = images_dir / src_image.name
        label_dst = labels_dir / src_label.name

        link_or_copy(src_image, image_dst)
        link_or_copy(src_label, label_dst)

        copied_images += 1
        copied_labels += 1

    # --------------------------------------------------------------
    # Integrity checks.
    # --------------------------------------------------------------

    actual_images = list(images_dir.iterdir())
    actual_labels = list(labels_dir.iterdir())

    # Only count actual image/label extensions.
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    label_extensions = {
        ".txt",
    }

    actual_image_files = [
        p for p in actual_images
        if p.suffix.lower() in image_extensions
    ]

    actual_label_files = [
        p for p in actual_labels
        if p.suffix.lower() in label_extensions
    ]

    if len(actual_image_files) != copied_images:
        raise RuntimeError(
            f"Image count mismatch for {dataset_name}: "
            f"expected {copied_images}, "
            f"found {len(actual_image_files)}"
        )

    if len(actual_label_files) != copied_labels:
        raise RuntimeError(
            f"Label count mismatch for {dataset_name}: "
            f"expected {copied_labels}, "
            f"found {len(actual_label_files)}"
        )

    print(f"Images: {len(actual_image_files)}")

    # --------------------------------------------------------------
    # IMPORTANT YAML FIX
    # --------------------------------------------------------------
    #
    # Ultralytics check_det_dataset() requires train and val keys.
    # We therefore define all three.
    #
    # model.val(..., split='test') guarantees only test is evaluated.
    # --------------------------------------------------------------

    yaml_path = eval_root / "dataset.yaml"

    yaml_text = (
        f"path: {eval_root.as_posix()}\n"
        f"train: images\n"
        f"val: images\n"
        f"test: images\n"
        f"names:\n"
        f"  0: anthropogenic\n"
    )

    yaml_path.write_text(
        yaml_text,
        encoding="utf-8",
    )

    print()
    print("Dataset YAML:")
    print(yaml_text)

    return yaml_path


# ======================================================================
# DATASET INTEGRITY VERIFICATION
# ======================================================================

def verify_test_datasets(
    manifest: pd.DataFrame,
):
    print_header("VERIFYING TEST DATASETS")

    test_df = manifest[
        manifest["split"].astype(str).str.lower() == "test"
    ].copy()

    for dataset_name in TARGET_DATASETS:

        print()
        print(dataset_name)
        print("-" * 50)

        dataset_df = test_df[
            test_df["dataset"].astype(str) == dataset_name
        ].copy()

        if len(dataset_df) == 0:
            raise RuntimeError(
                f"No TEST rows found for {dataset_name}"
            )

        missing_images = []
        missing_labels = []

        for _, row in dataset_df.iterrows():

            image_path = normalize_path(row["output_image"])
            label_path = normalize_path(row["output_label"])

            if not image_path.exists():
                missing_images.append(str(image_path))

            if not label_path.exists():
                missing_labels.append(str(label_path))

        print(f"Manifest TEST rows : {len(dataset_df)}")
        print(f"Missing images     : {len(missing_images)}")
        print(f"Missing labels     : {len(missing_labels)}")

        if missing_images:
            print()
            print("First missing images:")

            for p in missing_images[:10]:
                print(p)

        if missing_labels:
            print()
            print("First missing labels:")

            for p in missing_labels[:10]:
                print(p)

        if missing_images or missing_labels:
            raise RuntimeError(
                f"Integrity failure for {dataset_name}"
            )

    print()
    print("=" * 72)
    print("ALL DATASET INTEGRITY CHECKS PASSED")
    print("=" * 72)


# ======================================================================
# METRIC EXTRACTION
# ======================================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def extract_metrics(metrics):
    """
    Extract standard Ultralytics detection metrics.

    Different Ultralytics versions expose slightly different objects,
    so this function is deliberately defensive.
    """

    result = {}

    # --------------------------------------------------------------
    # Detection metrics
    # --------------------------------------------------------------

    try:
        result["precision"] = safe_float(
            metrics.box.mp
        )
    except Exception:
        result["precision"] = None

    try:
        result["recall"] = safe_float(
            metrics.box.mr
        )
    except Exception:
        result["recall"] = None

    try:
        result["mAP50"] = safe_float(
            metrics.box.map50
        )
    except Exception:
        result["mAP50"] = None

    try:
        result["mAP50-95"] = safe_float(
            metrics.box.map
        )
    except Exception:
        result["mAP50-95"] = None

    try:
        result["mAP75"] = safe_float(
            metrics.box.map75
        )
    except Exception:
        result["mAP75"] = None

    # --------------------------------------------------------------
    # Per-class AP if available.
    # --------------------------------------------------------------

    try:
        result["per_class_AP50"] = [
            safe_float(x)
            for x in metrics.box.ap50
        ]
    except Exception:
        result["per_class_AP50"] = None

    try:
        result["per_class_AP50-95"] = [
            safe_float(x)
            for x in metrics.box.ap
        ]
    except Exception:
        result["per_class_AP50-95"] = None

    # --------------------------------------------------------------
    # Confusion matrix / speed when available.
    # --------------------------------------------------------------

    try:
        result["speed_ms_preprocess"] = safe_float(
            metrics.speed.get("preprocess", None)
        )
    except Exception:
        result["speed_ms_preprocess"] = None

    try:
        result["speed_ms_inference"] = safe_float(
            metrics.speed.get("inference", None)
        )
    except Exception:
        result["speed_ms_inference"] = None

    try:
        result["speed_ms_postprocess"] = safe_float(
            metrics.speed.get("postprocess", None)
        )
    except Exception:
        result["speed_ms_postprocess"] = None

    return result


# ======================================================================
# DATASET EVALUATION
# ======================================================================

def evaluate_dataset(
    model: YOLO,
    dataset_name: str,
    dataset_df: pd.DataFrame,
):
    print_header(f"DATASET: {dataset_name}")

    yaml_path = create_eval_dataset(
        dataset_name=dataset_name,
        dataset_df=dataset_df,
    )

    print()
    print("Running validation...")

    metrics = model.val(
        data=str(yaml_path),
        split="test",

        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,

        conf=CONF,
        iou=IOU,

        plots=True,
        save_json=True,

        verbose=True,
        seed=SEED,

        project=str(OUTPUT_DIR),
        name=f"{dataset_name}_evaluation",
        exist_ok=True,
    )

    extracted = extract_metrics(metrics)

    print()
    print("-" * 50)
    print(f"RESULTS — {dataset_name}")
    print("-" * 50)

    for key, value in extracted.items():
        print(f"{key:25s}: {value}")

    return extracted


# ======================================================================
# SAVE RESULTS
# ======================================================================

def save_results(results):
    print_header("SAVING RESULTS")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # CSV
    # --------------------------------------------------------------

    rows = []

    for dataset_name, metrics in results.items():

        row = {
            "dataset": dataset_name,
        }

        row.update(metrics)

        rows.append(row)

    results_df = pd.DataFrame(rows)

    csv_path = OUTPUT_DIR / "dataset_wise_results.csv"

    results_df.to_csv(
        csv_path,
        index=False,
    )

    print()
    print("CSV:")
    print(csv_path)

    # --------------------------------------------------------------
    # JSON
    # --------------------------------------------------------------

    json_path = OUTPUT_DIR / "dataset_wise_results.json"

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("JSON:")
    print(json_path)

    # --------------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------------

    summary_path = OUTPUT_DIR / "dataset_wise_summary.txt"

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "SAAD — DATASET-WISE EVALUATION\n"
        )

        f.write(
            "=" * 72 + "\n\n"
        )

        f.write(
            f"Model: {MODEL_PATH}\n"
        )

        f.write(
            f"Manifest: {MANIFEST_PATH}\n\n"
        )

        for dataset_name, metrics in results.items():

            f.write(
                f"{dataset_name}\n"
            )

            f.write(
                "-" * 50 + "\n"
            )

            for key, value in metrics.items():

                f.write(
                    f"{key:25s}: {value}\n"
                )

            f.write("\n")

    print()
    print("Summary:")
    print(summary_path)


# ======================================================================
# FINAL SUMMARY
# ======================================================================

def print_final_summary(results):
    print_header("FINAL DATASET-WISE SUMMARY")

    rows = []

    for dataset_name, metrics in results.items():

        rows.append({
            "Dataset": dataset_name,
            "Precision": metrics.get("precision"),
            "Recall": metrics.get("recall"),
            "mAP50": metrics.get("mAP50"),
            "mAP50-95": metrics.get("mAP50-95"),
            "mAP75": metrics.get("mAP75"),
        })

    df = pd.DataFrame(rows)

    print()
    print(df.to_string(index=False))

    print()
    print("=" * 72)
    print("EVALUATION COMPLETE")
    print("=" * 72)

    print()
    print("Results saved to:")
    print(OUTPUT_DIR)


# ======================================================================
# MAIN
# ======================================================================

def main():

    try:

        # ----------------------------------------------------------
        # Environment
        # ----------------------------------------------------------

        check_environment()

        # ----------------------------------------------------------
        # Manifest
        # ----------------------------------------------------------

        manifest = load_manifest()

        # ----------------------------------------------------------
        # Verify exact manifest paths BEFORE evaluation.
        # ----------------------------------------------------------

        verify_test_datasets(manifest)

        # ----------------------------------------------------------
        # Frozen model
        # ----------------------------------------------------------

        model = load_model()

        # ----------------------------------------------------------
        # Test subset
        # ----------------------------------------------------------

        test_df = manifest[
            manifest["split"].astype(str).str.lower() == "test"
        ].copy()

        # ----------------------------------------------------------
        # Evaluate datasets independently.
        # ----------------------------------------------------------

        results = {}

        for dataset_name in TARGET_DATASETS:

            dataset_df = test_df[
                test_df["dataset"].astype(str) == dataset_name
            ].copy()

            results[dataset_name] = evaluate_dataset(
                model=model,
                dataset_name=dataset_name,
                dataset_df=dataset_df,
            )

        # ----------------------------------------------------------
        # Save
        # ----------------------------------------------------------

        save_results(results)

        # ----------------------------------------------------------
        # Final display
        # ----------------------------------------------------------

        print_final_summary(results)

    except KeyboardInterrupt:

        print()
        print("=" * 72)
        print("EVALUATION INTERRUPTED BY USER")
        print("=" * 72)

        sys.exit(1)

    except Exception as e:

        print()
        print("=" * 72)
        print("EVALUATION FAILED")
        print("=" * 72)

        print()
        print(type(e).__name__)
        print(e)

        print()
        print("Traceback:")
        traceback.print_exc()

        sys.exit(1)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()