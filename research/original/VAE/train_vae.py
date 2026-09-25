# ============================================================
# SAAD — VAE NORMAL-SEABED ANOMALY MODEL
# ============================================================
#
# Purpose:
#   Learn the acoustic appearance of NORMAL seabed patches.
#
# Dataset:
#   E:\SIH\Datasets\SAAD_VAE
#
# Input:
#   256 x 256 grayscale SSS patches
#
# Output:
#   E:\SIH\SIH_Results\vae_normal_seabed
#
# ============================================================

from pathlib import Path
import csv
import random
import time

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path(
    r"E:\SIH\Datasets\SAAD_VAE"
)

OUTPUT_ROOT = Path(
    r"E:\SIH\SIH_Results\vae_normal_seabed"
)

CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
RECON_DIR = OUTPUT_ROOT / "reconstructions"

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"


# ------------------------------------------------------------
# Image
# ------------------------------------------------------------

IMAGE_SIZE = 256


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

LATENT_DIM = 128


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

EPOCHS = 100

BATCH_SIZE = 64

NUM_WORKERS = 4

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 1e-5


# ------------------------------------------------------------
# VAE KL configuration
# ------------------------------------------------------------
#
# We deliberately start with a small KL weight and gradually
# increase it.
#
# This prevents the latent variance from becoming unstable
# before the reconstruction network has learned useful
# structure.
# ------------------------------------------------------------

BETA = 1e-5

KL_WARMUP_EPOCHS = 15


# ------------------------------------------------------------
# Numerical stability
# ------------------------------------------------------------

LOGVAR_MIN = -10.0
LOGVAR_MAX = 10.0

GRAD_CLIP_NORM = 5.0


# ------------------------------------------------------------
# Early stopping
# ------------------------------------------------------------

PATIENCE = 15


# ------------------------------------------------------------
# Reconstruction previews
# ------------------------------------------------------------

SAVE_RECON_EVERY = 5

NUM_RECON_IMAGES = 8


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 42


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# ============================================================
# DATASET
# ============================================================

class NormalSeabedDataset(Dataset):

    def __init__(self, root):

        self.root = Path(root)

        if not self.root.exists():

            raise FileNotFoundError(
                f"Dataset directory does not exist:\n"
                f"{self.root}"
            )

        self.files = sorted(
            list(self.root.glob("*.png"))
            + list(self.root.glob("*.jpg"))
            + list(self.root.glob("*.jpeg"))
        )

        if len(self.files) == 0:

            raise RuntimeError(
                f"No image files found in:\n"
                f"{self.root}"
            )

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        path = self.files[idx]

        try:

            image = Image.open(
                path
            ).convert("L")

            image = image.resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                ),
                Image.Resampling.BILINEAR
            )

            image = np.asarray(
                image,
                dtype=np.float32
            ) / 255.0

            image = torch.from_numpy(
                image
            ).unsqueeze(0)

            # ------------------------------------------------
            # Explicit finite-value check
            # ------------------------------------------------

            if not torch.isfinite(image).all():

                raise RuntimeError(
                    "Non-finite values found."
                )

            # ------------------------------------------------
            # Explicit range check
            # ------------------------------------------------

            if image.min() < 0.0 or image.max() > 1.0:

                raise RuntimeError(
                    "Image values outside [0, 1]."
                )

            return image

        except Exception as e:

            raise RuntimeError(
                f"Failed to load image:\n"
                f"{path}\n"
                f"{e}"
            )


# ============================================================
# VAE MODEL
# ============================================================

class ConvVAE(nn.Module):

    def __init__(self, latent_dim=128):

        super().__init__()

        self.latent_dim = latent_dim

        # ====================================================
        # ENCODER
        # ====================================================
        #
        # Input:
        #   1 × 256 × 256
        #
        # After conv layers:
        #
        #   32  × 128 × 128
        #   64  ×  64 ×  64
        #   128 ×  32 ×  32
        #   256 ×  16 ×  16
        #   512 ×   8 ×   8
        #
        # ====================================================

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


        self.feature_dim = (
            512 * 8 * 8
        )


        # ----------------------------------------------------
        # Latent distribution
        # ----------------------------------------------------

        self.fc_mu = nn.Linear(
            self.feature_dim,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            self.feature_dim,
            latent_dim
        )


        # ====================================================
        # DECODER
        # ====================================================

        self.fc_decode = nn.Linear(
            latent_dim,
            self.feature_dim
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


    # ========================================================
    # ENCODE
    # ========================================================

    def encode(self, x):

        x = self.encoder(x)

        x = x.view(
            x.size(0),
            -1
        )

        mu = self.fc_mu(x)

        logvar = self.fc_logvar(x)

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Prevent exp(logvar) from overflowing.
        # ----------------------------------------------------

        logvar = torch.clamp(
            logvar,
            min=LOGVAR_MIN,
            max=LOGVAR_MAX
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

        # logvar is already clamped.

        std = torch.exp(
            0.5 * logvar
        )

        eps = torch.randn_like(
            std
        )

        return (
            mu
            + std * eps
        )


    # ========================================================
    # DECODE
    # ========================================================

    def decode(self, z):

        x = self.fc_decode(z)

        x = x.view(
            -1,
            512,
            8,
            8
        )

        x = self.decoder(x)

        return x


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(
            mu,
            logvar
        )

        reconstruction = self.decode(z)

        return (
            reconstruction,
            mu,
            logvar
        )


# ============================================================
# LOSS
# ============================================================

def vae_loss(
    reconstruction,
    target,
    mu,
    logvar,
    beta
):

    # --------------------------------------------------------
    # Reconstruction loss
    # --------------------------------------------------------

    reconstruction_loss = F.mse_loss(
        reconstruction,
        target,
        reduction="mean"
    )

    # --------------------------------------------------------
    # KL divergence
    # --------------------------------------------------------

    kl_loss = -0.5 * torch.mean(
        1
        + logvar
        - mu.pow(2)
        - torch.exp(logvar)
    )

    total = (
        reconstruction_loss
        + beta * kl_loss
    )

    return (
        total,
        reconstruction_loss,
        kl_loss
    )


# ============================================================
# KL SCHEDULE
# ============================================================

def get_beta(epoch):

    if KL_WARMUP_EPOCHS <= 0:

        return BETA

    progress = min(
        1.0,
        epoch / KL_WARMUP_EPOCHS
    )

    return BETA * progress


# ============================================================
# RECONSTRUCTION PREVIEW
# ============================================================

@torch.no_grad()
def save_reconstructions(
    model,
    loader,
    epoch,
    output_dir
):

    model.eval()

    batch = next(iter(loader))

    batch = batch[
        :NUM_RECON_IMAGES
    ]

    batch = batch.to(
        DEVICE
    )

    reconstruction, _, _ = model(
        batch
    )

    batch = batch.cpu()

    reconstruction = (
        reconstruction.cpu()
    )

    n = batch.shape[0]

    gap = 4

    sheet_width = (
        IMAGE_SIZE * 2
        + gap
    )

    sheet_height = (
        n * IMAGE_SIZE
        + (n - 1) * gap
    )

    sheet = Image.new(
        "L",
        (
            sheet_width,
            sheet_height
        )
    )

    for i in range(n):

        original = (
            batch[i, 0].numpy()
            * 255.0
        ).clip(
            0,
            255
        ).astype(
            np.uint8
        )

        recon = (
            reconstruction[i, 0].numpy()
            * 255.0
        ).clip(
            0,
            255
        ).astype(
            np.uint8
        )

        original = Image.fromarray(
            original,
            mode="L"
        )

        recon = Image.fromarray(
            recon,
            mode="L"
        )

        y = i * (
            IMAGE_SIZE + gap
        )

        sheet.paste(
            original,
            (0, y)
        )

        sheet.paste(
            recon,
            (
                IMAGE_SIZE + gap,
                y
            )
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    path = (
        output_dir
        / f"epoch_{epoch:03d}.png"
    )

    sheet.save(path)

    print(
        f"  Saved reconstruction preview: "
        f"{path}"
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    beta,
    path
):

    torch.save(
        {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "train_loss":
                train_loss,

            "val_loss":
                val_loss,

            "beta":
                beta,

            "latent_dim":
                LATENT_DIM,

            "image_size":
                IMAGE_SIZE,

            "logvar_min":
                LOGVAR_MIN,

            "logvar_max":
                LOGVAR_MAX,
        },
        path
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    beta
):

    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    num_samples = 0

    for batch_idx, batch in enumerate(
        loader
    ):

        batch = batch.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Input sanity check
        # ----------------------------------------------------

        if not torch.isfinite(batch).all():

            raise RuntimeError(
                f"Non-finite input detected "
                f"at batch {batch_idx}"
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # AMP
        # ----------------------------------------------------

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(
                DEVICE.type == "cuda"
            )
        ):

            (
                reconstruction,
                mu,
                logvar
            ) = model(batch)

            (
                loss,
                recon_loss,
                kl_loss
            ) = vae_loss(
                reconstruction,
                batch,
                mu,
                logvar,
                beta
            )

        # ----------------------------------------------------
        # Output sanity checks
        # ----------------------------------------------------

        if not torch.isfinite(loss):

            raise RuntimeError(
                "\n"
                "NON-FINITE LOSS DETECTED\n"
                f"Batch: {batch_idx}\n"
                f"Loss: {loss.item()}\n"
                f"Recon: {recon_loss.item()}\n"
                f"KL: {kl_loss.item()}\n"
                f"Beta: {beta}\n"
                f"mu min/max: "
                f"{mu.min().item()} / "
                f"{mu.max().item()}\n"
                f"logvar min/max: "
                f"{logvar.min().item()} / "
                f"{logvar.max().item()}"
            )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        if DEVICE.type == "cuda":

            scaler.scale(
                loss
            ).backward()

            # Unscale before clipping.
            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP_NORM
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP_NORM
            )

            optimizer.step()

        batch_size = batch.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_recon += (
            recon_loss.item()
            * batch_size
        )

        total_kl += (
            kl_loss.item()
            * batch_size
        )

        num_samples += batch_size

    return (
        total_loss / num_samples,
        total_recon / num_samples,
        total_kl / num_samples
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    beta
):

    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    num_samples = 0

    for batch in loader:

        batch = batch.to(
            DEVICE,
            non_blocking=True
        )

        (
            reconstruction,
            mu,
            logvar
        ) = model(batch)

        (
            loss,
            recon_loss,
            kl_loss
        ) = vae_loss(
            reconstruction,
            batch,
            mu,
            logvar,
            beta
        )

        if not torch.isfinite(loss):

            raise RuntimeError(
                "Non-finite validation loss detected."
            )

        batch_size = batch.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_recon += (
            recon_loss.item()
            * batch_size
        )

        total_kl += (
            kl_loss.item()
            * batch_size
        )

        num_samples += batch_size

    return (
        total_loss / num_samples,
        total_recon / num_samples,
        total_kl / num_samples
    )


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything(SEED)

    print("=" * 70)
    print("SAAD — VAE NORMAL-SEABED ANOMALY MODEL")
    print("=" * 70)

    print()
    print(f"Device       : {DEVICE}")
    print(f"Dataset      : {DATA_ROOT}")
    print(f"Output       : {OUTPUT_ROOT}")
    print(f"Image size   : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Latent dim   : {LATENT_DIM}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Epochs       : {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Final beta   : {BETA}")
    print(f"KL warmup    : {KL_WARMUP_EPOCHS} epochs")
    print(f"Logvar range : [{LOGVAR_MIN}, {LOGVAR_MAX}]")
    print()

    if DEVICE.type == "cuda":

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA         : "
            f"{torch.version.cuda}"
        )

        print()

    # ========================================================
    # DIRECTORIES
    # ========================================================

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RECON_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # DATASETS
    # ========================================================

    print("Loading datasets...")

    train_dataset = NormalSeabedDataset(
        TRAIN_DIR
    )

    val_dataset = NormalSeabedDataset(
        VAL_DIR
    )

    test_dataset = NormalSeabedDataset(
        TEST_DIR
    )

    print(
        f"Train patches : "
        f"{len(train_dataset)}"
    )

    print(
        f"Val patches   : "
        f"{len(val_dataset)}"
    )

    print(
        f"Test patches  : "
        f"{len(test_dataset)}"
    )

    print()

    # ========================================================
    # DATALOADERS
    # ========================================================

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": (
            DEVICE.type == "cuda"
        ),
    }

    if NUM_WORKERS > 0:

        loader_kwargs[
            "persistent_workers"
        ] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = ConvVAE(
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    print()

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # AMP
    # ========================================================

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            DEVICE.type == "cuda"
        )
    )

    # ========================================================
    # TRAINING HISTORY
    # ========================================================

    history_path = (
        OUTPUT_ROOT
        / "training_history.csv"
    )

    with open(
        history_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "beta",
            "train_total",
            "train_reconstruction",
            "train_kl",
            "val_total",
            "val_reconstruction",
            "val_kl",
            "epoch_seconds",
        ])

    # ========================================================
    # TRAINING
    # ========================================================

    best_val_loss = float("inf")

    epochs_without_improvement = 0

    best_checkpoint = (
        CHECKPOINT_DIR
        / "best.pt"
    )

    last_checkpoint = (
        CHECKPOINT_DIR
        / "last.pt"
    )

    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print()

    training_start = time.time()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        epoch_start = time.time()

        # ----------------------------------------------------
        # KL warm-up
        # ----------------------------------------------------

        current_beta = get_beta(
            epoch
        )

        (
            train_loss,
            train_recon,
            train_kl
        ) = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            current_beta
        )

        (
            val_loss,
            val_recon,
            val_kl
        ) = validate(
            model,
            val_loader,
            current_beta
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        print(
            f"Epoch "
            f"{epoch:03d}/{EPOCHS} | "
            f"time {epoch_time / 60:.1f} min"
        )

        print(
            f"  beta        : "
            f"{current_beta:.8f}"
        )

        print(
            f"  train total : "
            f"{train_loss:.6f}"
        )

        print(
            f"  train recon : "
            f"{train_recon:.6f}"
        )

        print(
            f"  train KL    : "
            f"{train_kl:.6f}"
        )

        print(
            f"  val total   : "
            f"{val_loss:.6f}"
        )

        print(
            f"  val recon   : "
            f"{val_recon:.6f}"
        )

        print(
            f"  val KL      : "
            f"{val_kl:.6f}"
        )

        # ----------------------------------------------------
        # Log
        # ----------------------------------------------------

        with open(
            history_path,
            "a",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch,
                current_beta,
                train_loss,
                train_recon,
                train_kl,
                val_loss,
                val_recon,
                val_kl,
                epoch_time,
            ])

        # ----------------------------------------------------
        # Save last checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            model,
            optimizer,
            epoch,
            train_loss,
            val_loss,
            current_beta,
            last_checkpoint
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            epochs_without_improvement = 0

            save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                current_beta,
                best_checkpoint
            )

            print(
                "  *** NEW BEST MODEL ***"
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Reconstruction preview
        # ----------------------------------------------------

        if (
            epoch == 1
            or epoch % SAVE_RECON_EVERY == 0
        ):

            save_reconstructions(
                model,
                val_loader,
                epoch,
                RECON_DIR
            )

        print()

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "Early stopping triggered."
            )

            print(
                f"No validation improvement "
                f"for {PATIENCE} epochs."
            )

            break

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    total_training_time = (
        time.time()
        - training_start
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Total training time: "
        f"{total_training_time / 3600:.2f} hours"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        "Best checkpoint:"
    )

    print(
        f"  {best_checkpoint}"
    )

    print()

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    checkpoint = torch.load(
        best_checkpoint,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # ========================================================
    # NORMAL TEST SET
    # ========================================================

    final_beta = checkpoint.get(
        "beta",
        BETA
    )

    (
        test_loss,
        test_recon,
        test_kl
    ) = validate(
        model,
        test_loader,
        final_beta
    )

    print("=" * 70)
    print("NORMAL TEST SET")
    print("=" * 70)

    print(
        f"Total loss          : "
        f"{test_loss:.6f}"
    )

    print(
        f"Reconstruction loss : "
        f"{test_recon:.6f}"
    )

    print(
        f"KL loss             : "
        f"{test_kl:.6f}"
    )

    print()

    # ========================================================
    # SAVE TEST METRICS
    # ========================================================

    metrics_path = (
        OUTPUT_ROOT
        / "test_metrics.txt"
    )

    with open(
        metrics_path,
        "w"
    ) as f:

        f.write(
            "SAAD VAE — NORMAL TEST SET\n"
        )

        f.write(
            "==========================\n\n"
        )

        f.write(
            f"Total loss: "
            f"{test_loss:.8f}\n"
        )

        f.write(
            f"Reconstruction loss: "
            f"{test_recon:.8f}\n"
        )

        f.write(
            f"KL loss: "
            f"{test_kl:.8f}\n"
        )

        f.write(
            f"Beta: "
            f"{final_beta:.8f}\n"
        )

    # ========================================================
    # FINAL TEST RECONSTRUCTION
    # ========================================================

    save_reconstructions(
        model,
        test_loader,
        999,
        RECON_DIR
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        f"Training history:"
    )

    print(
        f"  {history_path}"
    )

    print()

    print(
        f"Test metrics:"
    )

    print(
        f"  {metrics_path}"
    )

    print()

    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()