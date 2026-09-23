import os
import csv
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = Path(r"E:\SIH\Datasets\SAAD_VAE")

# IMPORTANT:
# VAE v1 is untouched.
OUTPUT_ROOT = Path(
    r"E:\SIH\SIH_Results\vae_normal_seabed_v2_ssim"
)

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR   = DATA_ROOT / "val"
TEST_DIR  = DATA_ROOT / "test"

CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
RECON_DIR = OUTPUT_ROOT / "reconstructions"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RECON_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42


# ============================================================
# IMAGE / MODEL
# ============================================================

IMAGE_SIZE = 256
LATENT_DIM = 128


# ============================================================
# TRAINING
# ============================================================

BATCH_SIZE = 64
NUM_WORKERS = 4

EPOCHS = 100
PATIENCE = 15

LR = 2e-4
WEIGHT_DECAY = 1e-5


# ============================================================
# KL
# ============================================================

BETA_MAX = 1e-5
KL_WARMUP_EPOCHS = 15


# ============================================================
# RECONSTRUCTION
#
# Final objective:
#
#   0.8 * MSE
# + 0.2 * (1 - SSIM)
# + beta * KL
#
# But SSIM is gradually introduced during the first
# 10 epochs for numerical/training stability.
# ============================================================

MSE_WEIGHT_FINAL = 0.8
SSIM_WEIGHT_FINAL = 0.2

SSIM_WARMUP_EPOCHS = 10


# ============================================================
# SSIM
# ============================================================

SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5


# ============================================================
# SAFETY
# ============================================================

GRAD_CLIP = 5.0

USE_AMP = True


# ============================================================
# VALID IMAGE EXTENSIONS
# ============================================================

VALID_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# SEED EVERYTHING
# ============================================================

def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SAAD VAE V2 — STABLE MSE + SSIM + KL")
print("=" * 70)

print(
    f"Device       : {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU          : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA         : "
        f"{torch.version.cuda}"
    )

print(
    f"Dataset      : {DATA_ROOT}"
)

print(
    f"Output       : {OUTPUT_ROOT}"
)

print(
    f"Batch size   : {BATCH_SIZE}"
)

print(
    f"Learning rate: {LR}"
)

print(
    f"Final MSE wt  : {MSE_WEIGHT_FINAL}"
)

print(
    f"Final SSIM wt : {SSIM_WEIGHT_FINAL}"
)

print(
    f"SSIM warmup   : "
    f"{SSIM_WARMUP_EPOCHS} epochs"
)

print(
    f"Beta max      : {BETA_MAX}"
)

print(
    f"KL warmup     : "
    f"{KL_WARMUP_EPOCHS} epochs"
)

print("=" * 70)


# ============================================================
# DATASET
# ============================================================

class NormalPatchDataset(Dataset):

    def __init__(self, root):

        self.root = Path(root)

        self.files = sorted(
            [
                p
                for p in self.root.rglob("*")
                if (
                    p.is_file()
                    and
                    p.suffix.lower()
                    in VALID_EXTENSIONS
                )
            ]
        )

        if len(self.files) == 0:

            raise RuntimeError(
                f"No images found in:\n{self.root}"
            )

        print(
            f"Loaded {len(self.files):,} patches "
            f"from {self.root}"
        )

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        path = self.files[idx]

        try:

            img = Image.open(
                path
            ).convert("L")

            if img.size != (
                IMAGE_SIZE,
                IMAGE_SIZE
            ):

                img = img.resize(
                    (
                        IMAGE_SIZE,
                        IMAGE_SIZE
                    ),
                    Image.Resampling.BILINEAR
                )

            arr = np.asarray(
                img,
                dtype=np.float32
            ) / 255.0

            # ------------------------------------------------
            # Numerical safety
            # ------------------------------------------------

            if not np.isfinite(arr).all():

                raise ValueError(
                    f"Non-finite values in {path}"
                )

            tensor = torch.from_numpy(
                arr
            ).unsqueeze(0)

            # IMPORTANT:
            # Return integer index as v1 evaluator expects.
            return tensor, idx

        except Exception as e:

            raise RuntimeError(
                f"Failed to load {path}: {e}"
            )


# ============================================================
# GAUSSIAN WINDOW FOR SSIM
# ============================================================

def create_gaussian_window(
    window_size,
    sigma,
    channels,
    device,
    dtype
):

    coords = torch.arange(
        window_size,
        device=device,
        dtype=dtype
    )

    coords -= window_size // 2

    gaussian = torch.exp(
        -(coords ** 2)
        /
        (2 * sigma ** 2)
    )

    gaussian /= gaussian.sum()

    window_1d = gaussian.unsqueeze(1)

    window_2d = (
        window_1d
        @
        window_1d.t()
    )

    window_2d = (
        window_2d
        .unsqueeze(0)
        .unsqueeze(0)
    )

    window = window_2d.expand(
        channels,
        1,
        window_size,
        window_size
    ).contiguous()

    return window


# ============================================================
# DIFFERENTIABLE SSIM
# ============================================================

def ssim(
    x,
    y,
    window_size=11,
    sigma=1.5,
    data_range=1.0
):

    channels = x.size(1)

    window = create_gaussian_window(
        window_size,
        sigma,
        channels,
        x.device,
        x.dtype
    )

    padding = window_size // 2

    # --------------------------------------------------------
    # Local means
    # --------------------------------------------------------

    mu_x = F.conv2d(
        x,
        window,
        padding=padding,
        groups=channels
    )

    mu_y = F.conv2d(
        y,
        window,
        padding=padding,
        groups=channels
    )

    # --------------------------------------------------------
    # Squares
    # --------------------------------------------------------

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    # --------------------------------------------------------
    # Local variances / covariance
    # --------------------------------------------------------

    sigma_x_sq = (
        F.conv2d(
            x * x,
            window,
            padding=padding,
            groups=channels
        )
        - mu_x_sq
    )

    sigma_y_sq = (
        F.conv2d(
            y * y,
            window,
            padding=padding,
            groups=channels
        )
        - mu_y_sq
    )

    sigma_xy = (
        F.conv2d(
            x * y,
            window,
            padding=padding,
            groups=channels
        )
        - mu_xy
    )

    # --------------------------------------------------------
    # SSIM constants
    # --------------------------------------------------------

    C1 = (
        0.01 * data_range
    ) ** 2

    C2 = (
        0.03 * data_range
    ) ** 2

    # --------------------------------------------------------
    # SSIM map
    # --------------------------------------------------------

    numerator = (
        (2.0 * mu_xy + C1)
        *
        (2.0 * sigma_xy + C2)
    )

    denominator = (
        (mu_x_sq + mu_y_sq + C1)
        *
        (sigma_x_sq + sigma_y_sq + C2)
    )

    ssim_map = (
        numerator
        /
        (
            denominator
            + 1e-12
        )
    )

    # Numerical protection.
    ssim_map = torch.clamp(
        ssim_map,
        -1.0,
        1.0
    )

    return ssim_map.mean()


# ============================================================
# VAE
#
# THIS ARCHITECTURE MATCHES YOUR STABLE V1 ARCHITECTURE.
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
#
# latent = 128
# ============================================================

class ConvVAE(nn.Module):

    def __init__(
        self,
        latent_dim=128
    ):

        super().__init__()

        # ----------------------------------------------------
        # ENCODER
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

            nn.ReLU(
                inplace=True
            ),


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

        # ----------------------------------------------------
        # LATENT
        # ----------------------------------------------------

        self.fc_mu = nn.Linear(
            512 * 8 * 8,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            512 * 8 * 8,
            latent_dim
        )

        # ----------------------------------------------------
        # DECODER INPUT
        # ----------------------------------------------------

        self.decoder_input = nn.Linear(
            latent_dim,
            512 * 8 * 8
        )

        # ----------------------------------------------------
        # DECODER
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

            nn.ReLU(
                inplace=True
            ),


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


            nn.ConvTranspose2d(
                32,
                1,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Sigmoid()
        )


    # ========================================================
    # ENCODE
    # ========================================================

    def encode(self, x):

        h = self.encoder(x)

        h = h.flatten(
            start_dim=1
        )

        mu = self.fc_mu(h)

        logvar = self.fc_logvar(h)

        # ----------------------------------------------------
        # STRONG NUMERICAL PROTECTION
        # ----------------------------------------------------

        mu = torch.clamp(
            mu,
            min=-20.0,
            max=20.0
        )

        logvar = torch.clamp(
            logvar,
            min=-8.0,
            max=8.0
        )

        return mu, logvar


    # ========================================================
    # REPARAMETERIZATION
    # ========================================================

    def reparameterize(
        self,
        mu,
        logvar
    ):

        # ----------------------------------------------------
        # Extra protection before exp()
        # ----------------------------------------------------

        safe_logvar = torch.clamp(
            logvar,
            min=-8.0,
            max=8.0
        )

        std = torch.exp(
            0.5 * safe_logvar
        )

        eps = torch.randn_like(
            std
        )

        z = (
            mu
            +
            eps * std
        )

        # Final latent safety guard.
        z = torch.clamp(
            z,
            min=-30.0,
            max=30.0
        )

        return z


    # ========================================================
    # DECODE
    # ========================================================

    def decode(self, z):

        h = self.decoder_input(z)

        h = h.view(
            -1,
            512,
            8,
            8
        )

        return self.decoder(h)


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(
            mu,
            logvar
        )

        recon = self.decode(z)

        return (
            recon,
            mu,
            logvar
        )


# ============================================================
# KL DIVERGENCE
# ============================================================

def kl_divergence(
    mu,
    logvar
):

    # --------------------------------------------------------
    # Protect KL calculation.
    # --------------------------------------------------------

    logvar = torch.clamp(
        logvar,
        min=-8.0,
        max=8.0
    )

    kl = -0.5 * (
        1.0
        +
        logvar
        -
        mu.pow(2)
        -
        logvar.exp()
    )

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    kl = torch.clamp(
        kl,
        min=0.0,
        max=1e6
    )

    # Mean over batch,
    # sum over latent dimensions.
    kl = kl.sum(
        dim=1
    ).mean()

    return kl


# ============================================================
# KL BETA WARMUP
# ============================================================

def get_beta(epoch):

    if KL_WARMUP_EPOCHS <= 0:

        return BETA_MAX

    progress = min(
        1.0,
        (epoch + 1)
        /
        KL_WARMUP_EPOCHS
    )

    return (
        BETA_MAX
        *
        progress
    )


# ============================================================
# SSIM WARMUP
#
# Epoch 1:
#   SSIM = 0.00
#
# Epoch 5:
#   SSIM = 0.08
#
# Epoch 10:
#   SSIM = 0.20
#
# Epoch 11+:
#   SSIM = 0.20
# ============================================================

def get_ssim_weight(epoch):

    if SSIM_WARMUP_EPOCHS <= 0:

        return SSIM_WEIGHT_FINAL

    progress = min(
        1.0,
        (epoch + 1)
        /
        SSIM_WARMUP_EPOCHS
    )

    return (
        SSIM_WEIGHT_FINAL
        *
        progress
    )


# ============================================================
# COMPUTE LOSS
# ============================================================

def compute_loss(
    x,
    recon,
    mu,
    logvar,
    beta,
    ssim_weight
):

    # --------------------------------------------------------
    # MSE
    # --------------------------------------------------------

    mse = F.mse_loss(
        recon,
        x,
        reduction="mean"
    )

    # --------------------------------------------------------
    # SSIM
    # --------------------------------------------------------

    ssim_value = ssim(
        x,
        recon,
        window_size=SSIM_WINDOW_SIZE,
        sigma=SSIM_SIGMA,
        data_range=1.0
    )

    ssim_loss = (
        1.0
        -
        ssim_value
    )

    # --------------------------------------------------------
    # Dynamic MSE / SSIM weights
    #
    # They always sum to 1.
    # --------------------------------------------------------

    mse_weight = (
        1.0
        -
        ssim_weight
    )

    reconstruction_loss = (
        mse_weight * mse
        +
        ssim_weight * ssim_loss
    )

    # --------------------------------------------------------
    # KL
    # --------------------------------------------------------

    kl = kl_divergence(
        mu,
        logvar
    )

    # --------------------------------------------------------
    # TOTAL
    # --------------------------------------------------------

    total = (
        reconstruction_loss
        +
        beta * kl
    )

    return (
        total,
        mse,
        ssim_loss,
        kl,
        ssim_value
    )


# ============================================================
# RUN ONE EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    optimizer=None,
    scaler=None,
    beta=0.0,
    ssim_weight=0.0
):

    training = (
        optimizer is not None
    )

    if training:

        model.train()

    else:

        model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_ssim_loss = 0.0
    total_kl = 0.0
    total_ssim = 0.0

    num_samples = 0

    # --------------------------------------------------------
    # Batches
    # --------------------------------------------------------

    for batch_idx, (
        x,
        _
    ) in enumerate(loader):

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Input safety
        # ----------------------------------------------------

        if not torch.isfinite(
            x
        ).all():

            raise RuntimeError(
                "Non-finite input detected."
            )

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        with torch.set_grad_enabled(
            training
        ):

            if (
                USE_AMP
                and
                DEVICE.type == "cuda"
            ):

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):

                    (
                        recon,
                        mu,
                        logvar
                    ) = model(x)

                    (
                        loss,
                        mse,
                        ssim_loss,
                        kl,
                        ssim_value
                    ) = compute_loss(
                        x,
                        recon,
                        mu,
                        logvar,
                        beta,
                        ssim_weight
                    )

            else:

                (
                    recon,
                    mu,
                    logvar
                ) = model(x)

                (
                    loss,
                    mse,
                    ssim_loss,
                    kl,
                    ssim_value
                ) = compute_loss(
                    x,
                    recon,
                    mu,
                    logvar,
                    beta,
                    ssim_weight
                )

            # ------------------------------------------------
            # Output safety
            # ------------------------------------------------

            if not torch.isfinite(
                recon
            ).all():

                raise RuntimeError(
                    "Non-finite reconstruction detected."
                )

            if not torch.isfinite(
                mu
            ).all():

                raise RuntimeError(
                    "Non-finite mu detected."
                )

            if not torch.isfinite(
                logvar
            ).all():

                raise RuntimeError(
                    "Non-finite logvar detected."
                )

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "Non-finite loss detected: "
                    f"{loss.item()}"
                )

            # ------------------------------------------------
            # BACKWARD
            # ------------------------------------------------

            if training:

                if scaler is not None:

                    scaler.scale(
                        loss
                    ).backward()

                    scaler.unscale_(
                        optimizer
                    )

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        GRAD_CLIP
                    )

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                else:

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        GRAD_CLIP
                    )

                    optimizer.step()

        # ----------------------------------------------------
        # Accumulate
        # ----------------------------------------------------

        bs = x.size(0)

        total_loss += (
            loss.item()
            *
            bs
        )

        total_mse += (
            mse.item()
            *
            bs
        )

        total_ssim_loss += (
            ssim_loss.item()
            *
            bs
        )

        total_kl += (
            kl.item()
            *
            bs
        )

        total_ssim += (
            ssim_value.item()
            *
            bs
        )

        num_samples += bs

    # --------------------------------------------------------
    # Return averages
    # --------------------------------------------------------

    return {

        "loss":
            total_loss
            /
            num_samples,

        "mse":
            total_mse
            /
            num_samples,

        "ssim_loss":
            total_ssim_loss
            /
            num_samples,

        "kl":
            total_kl
            /
            num_samples,

        "ssim":
            total_ssim
            /
            num_samples,
    }


# ============================================================
# SAVE RECONSTRUCTION PREVIEW
# ============================================================

@torch.no_grad()
def save_reconstruction_preview(
    model,
    loader,
    epoch,
    max_images=8
):

    model.eval()

    try:

        x, _ = next(
            iter(loader)
        )

    except StopIteration:

        return

    x = x[
        :max_images
    ].to(DEVICE)

    recon, _, _ = model(x)

    x = x.cpu()
    recon = recon.cpu()

    n = x.size(0)

    rows = []

    for i in range(n):

        original = (
            x[i, 0]
            .numpy()
        )

        reconstructed = (
            recon[i, 0]
            .numpy()
        )

        separator = np.ones(
            (
                IMAGE_SIZE,
                4
            ),
            dtype=np.float32
        )

        row = np.concatenate(
            [
                original,
                separator,
                reconstructed
            ],
            axis=1
        )

        rows.append(row)

    grid = np.concatenate(
        rows,
        axis=0
    )

    grid = np.clip(
        grid * 255.0,
        0,
        255
    ).astype(
        np.uint8
    )

    out_path = (
        RECON_DIR
        /
        f"epoch_{epoch:03d}.png"
    )

    Image.fromarray(
        grid,
        mode="L"
    ).save(
        out_path
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    metrics
):

    checkpoint = {

        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "metrics":
            metrics,

        "config": {

            "image_size":
                IMAGE_SIZE,

            "latent_dim":
                LATENT_DIM,

            "mse_weight_final":
                MSE_WEIGHT_FINAL,

            "ssim_weight_final":
                SSIM_WEIGHT_FINAL,

            "ssim_warmup_epochs":
                SSIM_WARMUP_EPOCHS,

            "beta_max":
                BETA_MAX,

            "kl_warmup_epochs":
                KL_WARMUP_EPOCHS,

            "lr":
                LR,

            "batch_size":
                BATCH_SIZE,

            "seed":
                SEED,
        }
    }

    if scaler is not None:

        checkpoint[
            "scaler_state_dict"
        ] = scaler.state_dict()

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    # ========================================================
    # DATASETS
    # ========================================================

    print("\nLoading datasets...")

    train_dataset = NormalPatchDataset(
        TRAIN_DIR
    )

    val_dataset = NormalPatchDataset(
        VAL_DIR
    )

    test_dataset = NormalPatchDataset(
        TEST_DIR
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        drop_last=False
    )

    # ========================================================
    # MODEL
    # ========================================================

    print("\nCreating model...")

    model = ConvVAE(
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters   : "
        f"{total_params:,}"
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # AMP
    # ========================================================

    scaler = None

    if (
        USE_AMP
        and
        DEVICE.type == "cuda"
    ):

        scaler = torch.amp.GradScaler(
            "cuda"
        )

    # ========================================================
    # TRAINING STATE
    # ========================================================

    best_val_loss = float(
        "inf"
    )

    best_epoch = -1

    epochs_without_improvement = 0

    history = []

    # ========================================================
    # TRAIN
    # ========================================================

    print("\nStarting training...")
    print("=" * 70)

    for epoch in range(EPOCHS):

        epoch_start = time.time()

        # ----------------------------------------------------
        # Current weights
        # ----------------------------------------------------

        beta = get_beta(
            epoch
        )

        ssim_weight = get_ssim_weight(
            epoch
        )

        mse_weight = (
            1.0
            -
            ssim_weight
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scaler=scaler,
            beta=beta,
            ssim_weight=ssim_weight
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
            scaler=None,
            beta=beta,
            ssim_weight=ssim_weight
        )

        elapsed = (
            time.time()
            -
            epoch_start
        )

        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        row = {

            "epoch":
                epoch + 1,

            "beta":
                beta,

            "mse_weight":
                mse_weight,

            "ssim_weight":
                ssim_weight,

            "train_loss":
                train_metrics["loss"],

            "train_mse":
                train_metrics["mse"],

            "train_ssim_loss":
                train_metrics["ssim_loss"],

            "train_kl":
                train_metrics["kl"],

            "train_ssim":
                train_metrics["ssim"],

            "val_loss":
                val_metrics["loss"],

            "val_mse":
                val_metrics["mse"],

            "val_ssim_loss":
                val_metrics["ssim_loss"],

            "val_kl":
                val_metrics["kl"],

            "val_ssim":
                val_metrics["ssim"],
        }

        history.append(row)

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"Epoch {epoch + 1:03d} | "
            f"beta={beta:.2e} | "
            f"MSEw={mse_weight:.3f} "
            f"SSIMw={ssim_weight:.3f} | "
            f"Train "
            f"loss={train_metrics['loss']:.6f} "
            f"mse={train_metrics['mse']:.6f} "
            f"ssimL={train_metrics['ssim_loss']:.6f} "
            f"kl={train_metrics['kl']:.4f} "
            f"SSIM={train_metrics['ssim']:.4f} | "
            f"Val "
            f"loss={val_metrics['loss']:.6f} "
            f"mse={val_metrics['mse']:.6f} "
            f"ssimL={val_metrics['ssim_loss']:.6f} "
            f"kl={val_metrics['kl']:.4f} "
            f"SSIM={val_metrics['ssim']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # ----------------------------------------------------
        # Preview
        # ----------------------------------------------------

        if (
            epoch == 0
            or
            (epoch + 1) % 5 == 0
        ):

            save_reconstruction_preview(
                model,
                val_loader,
                epoch + 1
            )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if (
            val_metrics["loss"]
            <
            best_val_loss
        ):

            best_val_loss = (
                val_metrics["loss"]
            )

            best_epoch = (
                epoch + 1
            )

            epochs_without_improvement = 0

            save_checkpoint(
                CHECKPOINT_DIR
                /
                "best.pt",
                model,
                optimizer,
                scaler,
                epoch + 1,
                row
            )

            print(
                f"  -> New best model "
                f"(val loss="
                f"{best_val_loss:.6f})"
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # LAST MODEL
        # ----------------------------------------------------

        save_checkpoint(
            CHECKPOINT_DIR
            /
            "last.pt",
            model,
            optimizer,
            scaler,
            epoch + 1,
            row
        )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                f"\nEarly stopping after "
                f"{epoch + 1} epochs."
            )

            break

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_path = (
        OUTPUT_ROOT
        /
        "training_history.csv"
    )

    if len(history) > 0:

        fieldnames = list(
            history[0].keys()
        )

        with open(
            history_path,
            "w",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                history
            )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    best_path = (
        CHECKPOINT_DIR
        /
        "best.pt"
    )

    checkpoint = torch.load(
        best_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print("\n" + "=" * 70)
    print("BEST MODEL LOADED")
    print("=" * 70)

    print(
        f"Best epoch    : "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best val loss : "
        f"{checkpoint['metrics']['val_loss']:.8f}"
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print(
        "\nRunning final normal test..."
    )

    test_metrics = run_epoch(
        model,
        test_loader,
        optimizer=None,
        scaler=None,
        beta=BETA_MAX,
        ssim_weight=SSIM_WEIGHT_FINAL
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Test total loss : "
        f"{test_metrics['loss']:.8f}"
    )

    print(
        f"Test MSE        : "
        f"{test_metrics['mse']:.8f}"
    )

    print(
        f"Test SSIM loss  : "
        f"{test_metrics['ssim_loss']:.8f}"
    )

    print(
        f"Test SSIM       : "
        f"{test_metrics['ssim']:.8f}"
    )

    print(
        f"Test KL         : "
        f"{test_metrics['kl']:.8f}"
    )

    # ========================================================
    # SAVE TEST METRICS
    # ========================================================

    metrics_path = (
        OUTPUT_ROOT
        /
        "test_metrics.txt"
    )

    with open(
        metrics_path,
        "w"
    ) as f:

        f.write(
            "SAAD VAE V2 — Stable MSE + SSIM + KL\n"
        )

        f.write(
            "====================================\n"
        )

        f.write(
            f"Best epoch: "
            f"{checkpoint['epoch']}\n"
        )

        f.write(
            f"Best val loss: "
            f"{checkpoint['metrics']['val_loss']:.10f}\n"
        )

        f.write(
            f"Test total loss: "
            f"{test_metrics['loss']:.10f}\n"
        )

        f.write(
            f"Test MSE: "
            f"{test_metrics['mse']:.10f}\n"
        )

        f.write(
            f"Test SSIM loss: "
            f"{test_metrics['ssim_loss']:.10f}\n"
        )

        f.write(
            f"Test SSIM: "
            f"{test_metrics['ssim']:.10f}\n"
        )

        f.write(
            f"Test KL: "
            f"{test_metrics['kl']:.10f}\n"
        )

        f.write(
            f"Final MSE weight: "
            f"{MSE_WEIGHT_FINAL}\n"
        )

        f.write(
            f"Final SSIM weight: "
            f"{SSIM_WEIGHT_FINAL}\n"
        )

        f.write(
            f"SSIM warmup epochs: "
            f"{SSIM_WARMUP_EPOCHS}\n"
        )

        f.write(
            f"Beta max: "
            f"{BETA_MAX}\n"
        )

        f.write(
            f"KL warmup epochs: "
            f"{KL_WARMUP_EPOCHS}\n"
        )

        f.write(
            f"Learning rate: "
            f"{LR}\n"
        )

        f.write(
            f"Batch size: "
            f"{BATCH_SIZE}\n"
        )

    # ========================================================
    # FINAL RECONSTRUCTION
    # ========================================================

    save_reconstruction_preview(
        model,
        test_loader,
        999
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    total_time = (
        time.time()
        -
        start_time
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best checkpoint:\n"
        f"  {best_path}"
    )

    print(
        f"\nTraining history:\n"
        f"  {history_path}"
    )

    print(
        f"\nTest metrics:\n"
        f"  {metrics_path}"
    )

    print(
        f"\nTotal time: "
        f"{total_time / 3600:.2f} hours"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()