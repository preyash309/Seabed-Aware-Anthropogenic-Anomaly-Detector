"""
SAAD — Masked Acoustic Reconstruction (MAR) v1
================================================

Purpose
-------
Learn what NORMAL seabed should look like from surrounding acoustic
context.

Training:
    NORMAL seabed patches only

Input:
    256x256 grayscale sonar patch with a region masked

Target:
    Original unmasked patch

Inference:
    Mask a region -> reconstruct it -> compare reconstruction with
    actual pixels in the masked region.

Anomaly score:
    Mean absolute reconstruction error inside the mask.

Dataset:
    E:/SIH/Datasets/SAAD_VAE

IMPORTANT
---------
This script does NOT touch SAAD_baseline.

This is MAR v1: intentionally small and stable so it can be trained
quickly on an RTX 4070 Laptop GPU.
"""

import os
import random
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = Path(
    r"E:\SIH\Datasets\SAAD_VAE"
)

RESULT_ROOT = Path(
    r"E:\SIH\SIH_Results\mar_normal_seabed"
)

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"

IMG_SIZE = 256

BATCH_SIZE = 64
EPOCHS = 60
PATIENCE = 10

LR = 2e-4
WEIGHT_DECAY = 1e-5

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

AMP = DEVICE.type == "cuda"

SEED = 42

# ------------------------------------------------------------
# Mask configuration
# ------------------------------------------------------------

MASK_MIN_SIZE = 48
MASK_MAX_SIZE = 112

# Multiple masks per patch are possible
MIN_MASKS = 1
MAX_MASKS = 2

# Reconstruction loss weighting
MASK_LOSS_WEIGHT = 1.0
CONTEXT_LOSS_WEIGHT = 0.10

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

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_DIR = RESULT_ROOT / "checkpoints"
RECON_DIR = RESULT_ROOT / "reconstructions"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RECON_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHECKS
# ============================================================

if not DATA_ROOT.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_ROOT}"
    )

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Train directory not found: {TRAIN_DIR}"
    )

if not VAL_DIR.exists():
    raise FileNotFoundError(
        f"Val directory not found: {VAL_DIR}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found: {TEST_DIR}"
    )


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

    return sorted(
        [
            p
            for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMG_EXTENSIONS
        ]
    )


train_images = list_images(TRAIN_DIR)
val_images = list_images(VAL_DIR)
test_images = list_images(TEST_DIR)

print("=" * 70)
print("SAAD — MASKED ACOUSTIC RECONSTRUCTION")
print("=" * 70)

print("\nDataset:")
print(DATA_ROOT)

print("\nTrain patches:", len(train_images))
print("Val patches  :", len(val_images))
print("Test patches :", len(test_images))

print("\nDevice:", DEVICE)

if DEVICE.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# MASK GENERATION
# ============================================================

def generate_mask(
    h=IMG_SIZE,
    w=IMG_SIZE
):
    """
    Generate one or two random rectangular masks.

    Mask value:
        1 = masked region
        0 = visible region
    """

    mask = np.zeros(
        (h, w),
        dtype=np.float32
    )

    num_masks = random.randint(
        MIN_MASKS,
        MAX_MASKS
    )

    for _ in range(num_masks):

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
            max(0, h - mh)
        )

        x0 = random.randint(
            0,
            max(0, w - mw)
        )

        mask[
            y0:y0 + mh,
            x0:x0 + mw
        ] = 1.0

    return mask


# ============================================================
# DATASET
# ============================================================

class MARDataset(Dataset):

    def __init__(
        self,
        image_paths,
        training=True
    ):

        self.image_paths = image_paths
        self.training = training

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):

        path = self.image_paths[idx]

        img = Image.open(path).convert("L")

        img = img.resize(
            (IMG_SIZE, IMG_SIZE),
            Image.Resampling.BILINEAR
        )

        arr = np.asarray(
            img,
            dtype=np.float32
        ) / 255.0

        original = torch.from_numpy(
            arr
        ).unsqueeze(0)

        if self.training:
            mask = generate_mask()
        else:
            # deterministic center mask for validation
            mask = np.zeros(
                (IMG_SIZE, IMG_SIZE),
                dtype=np.float32
            )

            size = 96

            y0 = (
                IMG_SIZE - size
            ) // 2

            x0 = (
                IMG_SIZE - size
            ) // 2

            mask[
                y0:y0 + size,
                x0:x0 + size
            ] = 1.0

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        # ----------------------------------------------------
        # Masked image
        #
        # We use local mean rather than pure black.
        # This avoids introducing an artificial "black square"
        # that the network could trivially detect.
        # ----------------------------------------------------

        visible_mean = (
            original * (1.0 - mask)
        ).sum() / (
            (1.0 - mask).sum() + 1e-6
        )

        masked = (
            original * (1.0 - mask)
            +
            visible_mean * mask
        )

        return (
            masked,
            original,
            mask,
            str(path)
        )


# ============================================================
# MODEL
# ============================================================

class MARNet(nn.Module):

    """
    Lightweight convolutional masked reconstruction network.

    Encoder:
        256 -> 128 -> 64 -> 32 -> 16

    Decoder:
        16 -> 32 -> 64 -> 128 -> 256
    """

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

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

        out = self.decoder(z)

        return out


# ============================================================
# LOSS
# ============================================================

def mar_loss(
    reconstruction,
    target,
    mask
):
    """
    Primary loss:
        reconstruction error INSIDE masked region.

    Small contextual loss:
        reconstruction error outside masked region.

    The masked region receives much larger weight.
    """

    abs_error = torch.abs(
        reconstruction - target
    )

    masked_error = (
        abs_error * mask
    ).sum() / (
        mask.sum() + 1e-6
    )

    context_error = (
        abs_error * (1.0 - mask)
    ).sum() / (
        (1.0 - mask).sum() + 1e-6
    )

    total = (
        MASK_LOSS_WEIGHT * masked_error
        +
        CONTEXT_LOSS_WEIGHT * context_error
    )

    return (
        total,
        masked_error,
        context_error
    )


# ============================================================
# DATALOADERS
# ============================================================

train_dataset = MARDataset(
    train_images,
    training=True
)

val_dataset = MARDataset(
    val_images,
    training=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True
    if NUM_WORKERS > 0 else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True
    if NUM_WORKERS > 0 else False
)


# ============================================================
# MODEL / OPTIMIZER
# ============================================================

model = MARNet().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=4,
    min_lr=1e-6
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=AMP
)


# ============================================================
# PARAMETER COUNT
# ============================================================

num_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "\nMAR parameters:",
    f"{num_params:,}"
)


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    total_loss = 0.0
    total_mask = 0.0
    total_context = 0.0

    count = 0

    for masked, target, mask, _ in val_loader:

        masked = masked.to(
            DEVICE,
            non_blocking=True
        )

        target = target.to(
            DEVICE,
            non_blocking=True
        )

        mask = mask.to(
            DEVICE,
            non_blocking=True
        )

        reconstruction = model(
            masked
        )

        loss, mask_loss, context_loss = mar_loss(
            reconstruction,
            target,
            mask
        )

        bs = masked.size(0)

        total_loss += (
            loss.item() * bs
        )

        total_mask += (
            mask_loss.item() * bs
        )

        total_context += (
            context_loss.item() * bs
        )

        count += bs

    return (
        total_loss / count,
        total_mask / count,
        total_context / count
    )


# ============================================================
# RECONSTRUCTION PREVIEW
# ============================================================

@torch.no_grad()
def save_preview(epoch):

    model.eval()

    masked, target, mask, paths = next(
        iter(val_loader)
    )

    masked = masked.to(DEVICE)

    reconstruction = model(
        masked
    )

    masked = masked.cpu()
    target = target.cpu()
    reconstruction = reconstruction.cpu()
    mask = mask.cpu()

    n = min(4, len(masked))

    # --------------------------------------------------------
    # Create horizontal visual:
    #
    # original | masked | reconstruction | error
    # --------------------------------------------------------

    rows = []

    for i in range(n):

        orig = target[i, 0].numpy()
        msk = masked[i, 0].numpy()
        rec = reconstruction[i, 0].numpy()

        error = np.abs(
            orig - rec
        )

        error = np.clip(
            error * 5.0,
            0,
            1
        )

        row = np.concatenate(
            [
                orig,
                msk,
                rec,
                error
            ],
            axis=1
        )

        rows.append(row)

    canvas = np.concatenate(
        rows,
        axis=0
    )

    canvas = (
        np.clip(canvas, 0, 1)
        * 255
    ).astype(np.uint8)

    out = Image.fromarray(
        canvas
    )

    out.save(
        RECON_DIR /
        f"epoch_{epoch:03d}.png"
    )


# ============================================================
# TRAINING
# ============================================================

history = []

best_val = float("inf")
best_epoch = -1
epochs_without_improvement = 0

print("\n")
print("=" * 70)
print("STARTING MAR TRAINING")
print("=" * 70)


for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    running_mask = 0.0
    running_context = 0.0

    count = 0

    for masked, target, mask, _ in train_loader:

        masked = masked.to(
            DEVICE,
            non_blocking=True
        )

        target = target.to(
            DEVICE,
            non_blocking=True
        )

        mask = mask.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=AMP
        ):

            reconstruction = model(
                masked
            )

            loss, mask_loss, context_loss = mar_loss(
                reconstruction,
                target,
                mask
            )

        if not torch.isfinite(loss):

            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}"
            )

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        bs = masked.size(0)

        running_loss += (
            loss.item() * bs
        )

        running_mask += (
            mask_loss.item() * bs
        )

        running_context += (
            context_loss.item() * bs
        )

        count += bs

    train_loss = (
        running_loss / count
    )

    train_mask = (
        running_mask / count
    )

    train_context = (
        running_context / count
    )

    val_loss, val_mask, val_context = (
        validate()
    )

    scheduler.step(
        val_loss
    )

    lr_now = optimizer.param_groups[0]["lr"]

    row = {
        "epoch": epoch,

        "train_loss":
            train_loss,

        "train_mask_loss":
            train_mask,

        "train_context_loss":
            train_context,

        "val_loss":
            val_loss,

        "val_mask_loss":
            val_mask,

        "val_context_loss":
            val_context,

        "lr":
            lr_now,
    }

    history.append(row)

    print(
        f"Epoch {epoch:03d} | "
        f"train {train_loss:.6f} | "
        f"val {val_loss:.6f} | "
        f"mask {val_mask:.6f} | "
        f"context {val_context:.6f} | "
        f"lr {lr_now:.2e}"
    )

    # --------------------------------------------------------
    # Save preview
    # --------------------------------------------------------

    if (
        epoch == 1
        or epoch % 5 == 0
    ):
        save_preview(epoch)

    # --------------------------------------------------------
    # Best checkpoint
    # --------------------------------------------------------

    if val_loss < best_val:

        best_val = val_loss
        best_epoch = epoch
        epochs_without_improvement = 0

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val,
            "config": {
                "img_size": IMG_SIZE,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "mask_min_size": MASK_MIN_SIZE,
                "mask_max_size": MASK_MAX_SIZE,
                "mask_loss_weight": MASK_LOSS_WEIGHT,
                "context_loss_weight":
                    CONTEXT_LOSS_WEIGHT,
            }
        }

        torch.save(
            checkpoint,
            CHECKPOINT_DIR /
            "best.pt"
        )

        print(
            "  -> NEW BEST CHECKPOINT"
        )

    else:

        epochs_without_improvement += 1

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            f"\nEarly stopping after "
            f"{epoch} epochs."
        )

        break


# ============================================================
# SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    RESULT_ROOT /
    "training_history.csv",
    index=False
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

best_checkpoint_path = (
    CHECKPOINT_DIR /
    "best.pt"
)

checkpoint = torch.load(
    best_checkpoint_path,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# FINAL VALIDATION
# ============================================================

final_val = validate()

print("\n")
print("=" * 70)
print("FINAL MAR VALIDATION")
print("=" * 70)

print(
    "Best epoch:",
    checkpoint["epoch"]
)

print(
    "Best validation loss:",
    checkpoint["best_val_loss"]
)

print(
    "Final validation total:",
    final_val[0]
)

print(
    "Final validation masked:",
    final_val[1]
)

print(
    "Final validation context:",
    final_val[2]
)


# ============================================================
# MODEL COMPATIBILITY TEST
# ============================================================

print("\n")
print("=" * 70)
print("MAR COMPATIBILITY CHECK")
print("=" * 70)

dummy = torch.rand(
    2,
    1,
    IMG_SIZE,
    IMG_SIZE,
    device=DEVICE
)

with torch.no_grad():

    dummy_out = model(
        dummy
    )

print(
    "Input shape :",
    tuple(dummy.shape)
)

print(
    "Output shape:",
    tuple(dummy_out.shape)
)

print(
    "Input finite:",
    bool(torch.isfinite(dummy).all())
)

print(
    "Output finite:",
    bool(torch.isfinite(dummy_out).all())
)

if (
    tuple(dummy.shape)
    !=
    tuple(dummy_out.shape)
):

    raise RuntimeError(
        "MAR output shape mismatch"
    )

if not torch.isfinite(
    dummy_out
).all():

    raise RuntimeError(
        "MAR output contains NaN/Inf"
    )

print(
    "\nCHECK PASSED"
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "dataset": str(DATA_ROOT),

    "train_patches":
        len(train_images),

    "val_patches":
        len(val_images),

    "test_patches":
        len(test_images),

    "img_size":
        IMG_SIZE,

    "batch_size":
        BATCH_SIZE,

    "epochs_requested":
        EPOCHS,

    "epochs_completed":
        len(history),

    "best_epoch":
        checkpoint["epoch"],

    "best_val_loss":
        float(
            checkpoint["best_val_loss"]
        ),

    "final_val_total_loss":
        float(final_val[0]),

    "final_val_mask_loss":
        float(final_val[1]),

    "final_val_context_loss":
        float(final_val[2]),

    "parameters":
        int(num_params),

    "checkpoint":
        str(best_checkpoint_path),

    "device":
        str(DEVICE),

    "note":
        "MAR trained using normal seabed patches only. "
        "Anomaly score is reconstruction error in masked region."
}


with open(
    RESULT_ROOT /
    "mar_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


with open(
    RESULT_ROOT /
    "MAR_TRAINING_SUMMARY.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SAAD — MASKED ACOUSTIC RECONSTRUCTION v1\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    for key, value in summary.items():

        f.write(
            f"{key}: {value}\n"
        )


print("\n")
print("=" * 70)
print("MAR TRAINING COMPLETE")
print("=" * 70)

print(
    "\nResults:",
    RESULT_ROOT
)

print(
    "\nBest checkpoint:",
    best_checkpoint_path
)

print(
    "\nTraining history:",
    RESULT_ROOT /
    "training_history.csv"
)

print(
    "\nReconstructions:",
    RECON_DIR
)