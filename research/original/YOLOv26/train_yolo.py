"""
SAAD — YOLO26 GENERIC PRETRAINED BASELINE
==========================================

Purpose
-------
Train the first conventional detector baseline on the frozen SAAD dataset.

Model:
    YOLO26s pretrained checkpoint

Dataset:
    E:\SIH\Datasets\SAAD_baseline\data.yaml

Task:
    1 class:
        0 = anthropogenic

Hardware target:
    RTX 4070 8 GB laptop GPU

This is intentionally the SIMPLE BASELINE.
Do not add SAAD anomaly logic, tiling, domain adaptation,
or custom losses here. We need a clean reference experiment first.

Outputs:
    E:\SIH\SIH_Results\yolo26s_generic_baseline\

The script:
    - checks CUDA/GPU
    - checks the SAAD dataset YAML
    - loads pretrained YOLO26s
    - trains
    - validates on SAAD val
    - evaluates on SAAD test
    - prints the key metrics
    - saves the best checkpoint path

Windows multiprocessing is protected by if __name__ == "__main__".
"""

from pathlib import Path
import sys
import time


# ============================================================
# CONFIGURATION
# ============================================================

DATA_YAML = Path(
    r"E:\SIH\Datasets\SAAD_baseline\data.yaml"
)

MODEL_NAME = "yolo26s.pt"

PROJECT_DIR = Path(
    r"E:\SIH\SIH_Results"
)

RUN_NAME = "yolo26s_generic_baseline"


# ------------------------------------------------------------
# RTX 4070 8 GB conservative starting configuration
# ------------------------------------------------------------

EPOCHS = 80
IMGSZ = 640
BATCH = 8
WORKERS = 4
DEVICE = 0

# Stop if validation does not improve for this many epochs.
PATIENCE = 20

# Use mixed precision on CUDA.
AMP = True

# Keep checkpointing enabled.
SAVE = True

# Save validation plots/predictions.
PLOTS = True

# Reproducibility.
SEED = 42


# ============================================================
# ENVIRONMENT CHECK
# ============================================================

def check_environment():
    print("=" * 72)
    print("SAAD YOLO26 GENERIC BASELINE")
    print("=" * 72)

    print()
    print("Dataset YAML:")
    print(DATA_YAML)

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"\nSAAD dataset YAML not found:\n{DATA_YAML}\n"
            "Make sure rebuild_saad_v4.py completed successfully."
        )

    print()
    print("Model:")
    print(MODEL_NAME)

    print()
    print("Training configuration:")
    print(f"  epochs  = {EPOCHS}")
    print(f"  imgsz   = {IMGSZ}")
    print(f"  batch   = {BATCH}")
    print(f"  workers = {WORKERS}")
    print(f"  device  = {DEVICE}")
    print(f"  amp     = {AMP}")
    print(f"  seed    = {SEED}")

    try:
        import torch
    except ImportError:
        raise RuntimeError(
            "PyTorch is not installed in this environment."
        )

    print()
    print("PyTorch:")
    print(f"  version = {torch.__version__}")
    print(f"  CUDA    = {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "\nCUDA GPU was not detected.\n"
            "For this experiment, do not accidentally train on CPU.\n"
            "Check your PyTorch/CUDA installation first."
        )

    gpu_name = torch.cuda.get_device_name(DEVICE)
    gpu_memory_gb = (
        torch.cuda.get_device_properties(DEVICE).total_memory
        / (1024 ** 3)
    )

    print(f"  GPU     = {gpu_name}")
    print(f"  VRAM    = {gpu_memory_gb:.2f} GB")

    # The expected laptop GPU is an RTX 4070 with ~8 GB VRAM.
    # We do not hard-fail on another GPU, but report it.
    print()

    return torch


# ============================================================
# DATASET YAML CHECK
# ============================================================

def check_dataset_yaml():
    """
    Basic textual sanity check.

    We intentionally do not rewrite data.yaml here.
    The frozen dataset should remain untouched.
    """

    text = DATA_YAML.read_text(
        encoding="utf-8"
    )

    print("=" * 72)
    print("DATASET CHECK")
    print("=" * 72)

    print()
    print(text)

    if "nc: 1" not in text:
        print(
            "\nWARNING: could not find 'nc: 1' literally in data.yaml."
        )

    if "anthropogenic" not in text.lower():
        print(
            "\nWARNING: could not find 'anthropogenic' in data.yaml."
        )

    # Check expected directories.
    dataset_root = DATA_YAML.parent

    expected_dirs = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "images" / "test",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
        dataset_root / "labels" / "test",
    ]

    print()
    print("Expected dataset directories:")

    for d in expected_dirs:
        exists = d.exists()
        print(
            f"  {'OK ' if exists else 'BAD'} {d}"
        )

        if not exists:
            raise FileNotFoundError(
                f"Required dataset directory missing:\n{d}"
            )

    print()
    print("✓ Dataset structure found.")


# ============================================================
# TRAIN
# ============================================================

def train_model():
    from ultralytics import YOLO

    print()
    print("=" * 72)
    print("LOADING YOLO26s")
    print("=" * 72)

    print()
    print(
        "Loading pretrained YOLO26s checkpoint..."
    )

    model = YOLO(MODEL_NAME)

    print()
    print(
        "Starting fine-tuning on SAAD."
    )

    print(
        "This is the generic pretrained baseline."
    )

    print()

    start = time.time()

    results = model.train(
        data=str(DATA_YAML),

        # ----------------------------------------------------
        # Core training
        # ----------------------------------------------------
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,

        # ----------------------------------------------------
        # Runtime
        # ----------------------------------------------------
        workers=WORKERS,
        amp=AMP,

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------
        seed=SEED,

        # ----------------------------------------------------
        # Validation/checkpointing
        # ----------------------------------------------------
        val=True,
        save=SAVE,
        plots=PLOTS,
        patience=PATIENCE,

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,

        # ----------------------------------------------------
        # Preserve original rectangular sonar geometry as
        # much as Ultralytics' preprocessing permits.
        #
        # We DO NOT enable aggressive custom augmentation yet.
        # This keeps the first baseline easier to interpret.
        # ----------------------------------------------------
        rect=False,

        # ----------------------------------------------------
        # Standard pretrained transfer learning.
        # Do not freeze the backbone for this baseline.
        # ----------------------------------------------------
        pretrained=True,
    )

    elapsed = time.time() - start

    print()
    print("=" * 72)
    print("TRAINING FINISHED")
    print("=" * 72)

    print(
        f"\nTraining time: {elapsed / 3600:.2f} hours"
    )

    print()
    print(
        "Training results object:"
    )
    print(results)

    return model


# ============================================================
# VALIDATION
# ============================================================

def evaluate_validation(model):
    print()
    print("=" * 72)
    print("FINAL VALIDATION — SAAD VAL")
    print("=" * 72)

    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        plots=True,
    )

    print()
    print("Validation metrics:")

    try:
        print(
            f"  mAP50-95 : {metrics.box.map:.4f}"
        )
        print(
            f"  mAP50    : {metrics.box.map50:.4f}"
        )
        print(
            f"  mAP75    : {metrics.box.map75:.4f}"
        )

        if hasattr(metrics.box, "mp"):
            print(
                f"  precision: {metrics.box.mp:.4f}"
            )

        if hasattr(metrics.box, "mr"):
            print(
                f"  recall   : {metrics.box.mr:.4f}"
            )

    except Exception as e:
        print(
            f"Could not extract all metrics automatically: {e}"
        )

    return metrics


# ============================================================
# TEST
# ============================================================

def evaluate_test(model):
    print()
    print("=" * 72)
    print("FINAL TEST — SAAD TEST")
    print("=" * 72)

    print()
    print(
        "IMPORTANT: this is the held-out test set."
    )

    print(
        "Do not use these results to tune hyperparameters."
    )

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        plots=True,
    )

    print()
    print("Test metrics:")

    try:
        print(
            f"  mAP50-95 : {metrics.box.map:.4f}"
        )
        print(
            f"  mAP50    : {metrics.box.map50:.4f}"
        )
        print(
            f"  mAP75    : {metrics.box.map75:.4f}"
        )

        if hasattr(metrics.box, "mp"):
            print(
                f"  precision: {metrics.box.mp:.4f}"
            )

        if hasattr(metrics.box, "mr"):
            print(
                f"  recall   : {metrics.box.mr:.4f}"
            )

    except Exception as e:
        print(
            f"Could not extract all metrics automatically: {e}"
        )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    torch = check_environment()

    check_dataset_yaml()

    # --------------------------------------------------------
    # Clear CUDA cache before starting.
    # --------------------------------------------------------

    torch.cuda.empty_cache()

    model = train_model()

    # --------------------------------------------------------
    # The training run itself already performs validation.
    # These explicit evaluations make the final reported
    # numbers unambiguous.
    # --------------------------------------------------------

    val_metrics = evaluate_validation(model)

    test_metrics = evaluate_test(model)

    # --------------------------------------------------------
    # Best checkpoint
    # --------------------------------------------------------

    best_path = (
        PROJECT_DIR
        / RUN_NAME
        / "weights"
        / "best.pt"
    )

    last_path = (
        PROJECT_DIR
        / RUN_NAME
        / "weights"
        / "last.pt"
    )

    print()
    print("=" * 72)
    print("SAAD YOLO26 BASELINE COMPLETE")
    print("=" * 72)

    print()
    print("Best checkpoint:")
    print(best_path)

    print()
    print("Last checkpoint:")
    print(last_path)

    if best_path.exists():
        print(
            "\n✓ best.pt exists."
        )
    else:
        print(
            "\nWARNING: best.pt was not found at expected path."
        )

    print()
    print(
        "NEXT STEP:"
    )
    print(
        "Do NOT modify this run."
    )
    print(
        "Use its metrics as the generic YOLO26 reference baseline."
    )
    print(
        "Next we will evaluate performance separately on "
        "GhostVision, AI4Shipwrecks, and SubPipeMini2."
    )


if __name__ == "__main__":
    main()
