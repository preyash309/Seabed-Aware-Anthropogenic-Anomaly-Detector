# ============================================================
# SAAD — LATENT NORMALIZING FLOW
# ============================================================
#
# Purpose:
#   Learn the probability distribution of NORMAL seabed
#   in the frozen VAE latent space.
#
# Pipeline:
#
#   Normal sonar patch
#        ↓
#   Frozen VAE encoder
#        ↓
#   deterministic latent μ (128-D)
#        ↓
#   train-set standardization
#        ↓
#   RealNVP normalizing flow
#        ↓
#   negative log-likelihood (NLL)
#        ↓
#   latent normality / anomaly score
#
# IMPORTANT:
#   - VAE is FROZEN.
#   - Only VAE μ is used.
#   - Flow is trained in float32 for numerical stability.
#   - Standardization is learned ONLY from the normal train set.
#
# ============================================================

import os
import random
import math
import json
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

VAE_CHECKPOINT = r"E:\SIH\SIH_Results\vae_normal_seabed\checkpoints\best.pt"

DATA_ROOT = r"E:\SIH\Datasets\SAAD_VAE"

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR   = os.path.join(DATA_ROOT, "val")
TEST_DIR  = os.path.join(DATA_ROOT, "test")

OUTPUT_DIR = r"E:\SIH\SIH_Results\latent_normalizing_flow"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# VAE
# ------------------------------------------------------------

IMAGE_SIZE = 256
LATENT_DIM = 128


# ------------------------------------------------------------
# Flow
# ------------------------------------------------------------

FLOW_LAYERS = 8
FLOW_HIDDEN = 256

# Prevent exp(scale) from becoming numerically extreme.
SCALE_CLAMP = 1.5

FLOW_BATCH_SIZE = 256

FLOW_EPOCHS = 150
FLOW_PATIENCE = 20

FLOW_LR = 1e-3
FLOW_WEIGHT_DECAY = 1e-5

GRAD_CLIP = 5.0

NUM_WORKERS = 0


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

    # Reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================

class NormalPatchDataset(Dataset):

    def __init__(self, root):

        self.root = root

        valid_ext = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        }

        self.files = []

        for dirpath, _, filenames in os.walk(root):

            for filename in filenames:

                ext = os.path.splitext(filename)[1].lower()

                if ext in valid_ext:

                    self.files.append(
                        os.path.join(dirpath, filename)
                    )

        self.files.sort()

        if len(self.files) == 0:
            raise RuntimeError(
                f"No image files found in:\n{root}"
            )

        print(
            f"Loaded {len(self.files):,} patches from {root}"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        path = self.files[idx]

        from PIL import Image

        image = Image.open(path).convert("L")

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        image = np.asarray(
            image,
            dtype=np.float32
        )

        # Convert 0-255 → 0-1.
        image /= 255.0

        image = torch.from_numpy(image)

        image = image.unsqueeze(0)

        return image


# ============================================================
# VAE — EXACT V1 ARCHITECTURE
# ============================================================
#
# IMPORTANT:
# This architecture intentionally uses:
#
#     fc_mu
#     fc_logvar
#     fc_decode
#
# The checkpoint contains:
#
#     fc_decode.weight
#     fc_decode.bias
#
# NOT:
#
#     decoder_input.weight
#     decoder_input.bias
#
# ============================================================

class ConvVAE(nn.Module):

    def __init__(self, latent_dim=128):

        super().__init__()

        self.latent_dim = latent_dim

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

        # 256 → 128 → 64 → 32 → 16 → 8
        #
        # Final feature map:
        # 512 × 8 × 8

        self.fc_mu = nn.Linear(
            512 * 8 * 8,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            512 * 8 * 8,
            latent_dim
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Exact checkpoint parameter name.
        # ----------------------------------------------------

        self.fc_decode = nn.Linear(
            latent_dim,
            512 * 8 * 8
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

            nn.Sigmoid(),
        )

    def encode(self, x):

        h = self.encoder(x)

        h = h.view(
            h.size(0),
            -1
        )

        mu = self.fc_mu(h)

        logvar = self.fc_logvar(h)

        logvar = torch.clamp(
            logvar,
            -10.0,
            10.0
        )

        return mu, logvar

    def decode(self, z):

        h = self.fc_decode(z)

        h = h.view(
            -1,
            512,
            8,
            8
        )

        return self.decoder(h)

    def forward(self, x):

        mu, logvar = self.encode(x)

        std = torch.exp(
            0.5 * logvar
        )

        z = (
            mu
            + std * torch.randn_like(std)
        )

        reconstruction = self.decode(z)

        return reconstruction, mu, logvar


# ============================================================
# LOAD FROZEN VAE
# ============================================================

def load_vae():

    print()
    print("Loading frozen VAE v1...")

    model = ConvVAE(
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    checkpoint = torch.load(
        VAE_CHECKPOINT,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Handle common checkpoint formats.
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            # Sometimes checkpoint itself is state_dict.
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # Remove possible DataParallel prefix.
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[len("module."):]

        cleaned_state_dict[key] = value

    # --------------------------------------------------------
    # STRICT LOAD.
    #
    # We WANT this to fail if the architecture is wrong.
    # --------------------------------------------------------

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    print()
    print("VAE checkpoint compatibility check")
    print("----------------------------------")
    print("Missing keys   :", missing)
    print("Unexpected keys:", unexpected)

    if missing or unexpected:

        raise RuntimeError(
            "\nVAE architecture does NOT exactly match "
            "the checkpoint.\n"
            f"Missing keys: {missing}\n"
            f"Unexpected keys: {unexpected}\n"
        )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    print()
    print("VAE loaded successfully.")
    print("VAE parameters frozen.")

    return model


# ============================================================
# LATENT EXTRACTION
# ============================================================

@torch.no_grad()
def extract_latents(
    vae,
    dataset,
    batch_size=256
):

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    all_mu = []

    print()
    print(
        f"Extracting {len(dataset):,} latent vectors..."
    )

    for batch_idx, images in enumerate(loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Use deterministic μ.
        #
        # Do NOT sample z.
        # ----------------------------------------------------

        mu, _ = vae.encode(images)

        mu = mu.float()

        if not torch.isfinite(mu).all():

            raise RuntimeError(
                "Non-finite latent representation detected."
            )

        all_mu.append(
            mu.cpu()
        )

        if (
            batch_idx % 20 == 0
            or batch_idx == len(loader) - 1
        ):

            processed = min(
                (batch_idx + 1) * batch_size,
                len(dataset)
            )

            print(
                f"  {processed:,}/{len(dataset):,}"
            )

    latents = torch.cat(
        all_mu,
        dim=0
    )

    print(
        "Latent shape:",
        tuple(latents.shape)
    )

    return latents


# ============================================================
# STANDARDIZATION
# ============================================================

def fit_standardizer(train_latents):

    mean = train_latents.mean(
        dim=0
    )

    std = train_latents.std(
        dim=0,
        unbiased=False
    )

    # Avoid division by zero.
    std = torch.clamp(
        std,
        min=1e-6
    )

    return mean, std


def standardize(
    latents,
    mean,
    std
):

    return (
        latents - mean
    ) / std


# ============================================================
# REALNVP COUPLING LAYER
# ============================================================

class AffineCoupling(nn.Module):

    def __init__(
        self,
        dim,
        mask,
        hidden_dim=256,
        scale_clamp=1.5
    ):

        super().__init__()

        self.dim = dim
        self.scale_clamp = scale_clamp

        mask = mask.float()

        self.register_buffer(
            "mask",
            mask
        )

        mask_bool = mask.bool()

        self.condition_indices = torch.where(
            mask_bool
        )[0]

        self.transform_indices = torch.where(
            ~mask_bool
        )[0]

        condition_dim = len(
            self.condition_indices
        )

        transform_dim = len(
            self.transform_indices
        )

        self.condition_dim = condition_dim
        self.transform_dim = transform_dim

        self.net = nn.Sequential(

            nn.Linear(
                condition_dim,
                hidden_dim
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                hidden_dim,
                transform_dim * 2
            ),
        )

        # Start as approximately identity.
        final_layer = self.net[-1]

        nn.init.zeros_(
            final_layer.weight
        )

        nn.init.zeros_(
            final_layer.bias
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Extract conditioning dimensions.
        # ----------------------------------------------------

        x_condition = x[
            :,
            self.condition_indices
        ]

        st = self.net(
            x_condition
        )

        s, t = torch.chunk(
            st,
            2,
            dim=1
        )

        # ----------------------------------------------------
        # Bound scale for numerical stability.
        # ----------------------------------------------------

        s = torch.tanh(s)

        s = (
            s
            * self.scale_clamp
        )

        # ----------------------------------------------------
        # Transform ONLY the other 64 dimensions.
        # ----------------------------------------------------

        x_transform = x[
            :,
            self.transform_indices
        ]

        y_transform = (
            x_transform
            * torch.exp(s)
            + t
        )

        # ----------------------------------------------------
        # Reconstruct complete 128-D vector.
        # ----------------------------------------------------

        y = x.clone()

        y[
            :,
            self.transform_indices
        ] = y_transform

        log_det = s.sum(
            dim=1
        )

        return y, log_det

    def inverse(self, y):

        # ----------------------------------------------------
        # Conditioning dimensions are unchanged.
        # ----------------------------------------------------

        y_condition = y[
            :,
            self.condition_indices
        ]

        st = self.net(
            y_condition
        )

        s, t = torch.chunk(
            st,
            2,
            dim=1
        )

        s = torch.tanh(s)

        s = (
            s
            * self.scale_clamp
        )

        # ----------------------------------------------------
        # Invert transformed dimensions.
        # ----------------------------------------------------

        y_transform = y[
            :,
            self.transform_indices
        ]

        x_transform = (
            y_transform - t
        ) * torch.exp(-s)

        # ----------------------------------------------------
        # Reconstruct complete vector.
        # ----------------------------------------------------

        x = y.clone()

        x[
            :,
            self.transform_indices
        ] = x_transform

        log_det = (
            -s
        ).sum(
            dim=1
        )

        return x, log_det


# ============================================================
# REALNVP
# ============================================================

class RealNVP(nn.Module):

    def __init__(
        self,
        dim=128,
        num_layers=8,
        hidden_dim=256,
        scale_clamp=1.5
    ):

        super().__init__()

        layers = []

        for i in range(num_layers):

            # Alternating binary mask.
            #
            # Layer 0:
            #   first half condition
            #
            # Layer 1:
            #   second half condition
            #

            if i % 2 == 0:

                mask = torch.cat(
                    [
                        torch.ones(
                            dim // 2
                        ),
                        torch.zeros(
                            dim - dim // 2
                        ),
                    ]
                )

            else:

                mask = torch.cat(
                    [
                        torch.zeros(
                            dim // 2
                        ),
                        torch.ones(
                            dim - dim // 2
                        ),
                    ]
                )

            layers.append(
                AffineCoupling(
                    dim=dim,
                    mask=mask,
                    hidden_dim=hidden_dim,
                    scale_clamp=scale_clamp
                )
            )

        self.layers = nn.ModuleList(
            layers
        )

    def forward(self, x):

        z = x

        total_log_det = torch.zeros(
            x.size(0),
            device=x.device,
            dtype=x.dtype
        )

        for layer in self.layers:

            z, log_det = layer(z)

            total_log_det += log_det

        return z, total_log_det

    def inverse(self, z):

        x = z

        total_log_det = torch.zeros(
            z.size(0),
            device=z.device,
            dtype=z.dtype
        )

        for layer in reversed(
            self.layers
        ):

            x, log_det = layer.inverse(x)

            total_log_det += log_det

        return x, total_log_det


# ============================================================
# STANDARD NORMAL LOG PROBABILITY
# ============================================================

def standard_normal_log_prob(z):

    return (
        -0.5
        * (
            z.pow(2)
            + math.log(
                2.0 * math.pi
            )
        )
    ).sum(
        dim=1
    )


# ============================================================
# FLOW LOG PROBABILITY
# ============================================================

def flow_log_prob(
    flow,
    x
):

    z, log_det = flow(x)

    base_log_prob = (
        standard_normal_log_prob(z)
    )

    return (
        base_log_prob
        + log_det
    )


# ============================================================
# TRAINING
# ============================================================

def train_flow(
    flow,
    train_latents,
    val_latents
):

    train_dataset = torch.utils.data.TensorDataset(
        train_latents
    )

    val_dataset = torch.utils.data.TensorDataset(
        val_latents
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=FLOW_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=FLOW_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    optimizer = torch.optim.AdamW(
        flow.parameters(),
        lr=FLOW_LR,
        weight_decay=FLOW_WEIGHT_DECAY
    )

    best_val_loss = float("inf")

    best_state = None

    patience_counter = 0

    history = []

    print()
    print("=" * 70)
    print("TRAINING REALNVP")
    print("=" * 70)

    for epoch in range(
        1,
        FLOW_EPOCHS + 1
    ):

        epoch_start = time.time()

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        flow.train()

        train_losses = []

        for (x,) in train_loader:

            x = x.to(
                DEVICE,
                non_blocking=True
            )

            # Flow deliberately stays FP32.
            x = x.float()

            optimizer.zero_grad(
                set_to_none=True
            )

            log_prob = flow_log_prob(
                flow,
                x
            )

            if not torch.isfinite(
                log_prob
            ).all():

                raise RuntimeError(
                    "Non-finite flow log probability "
                    "during training."
                )

            loss = -log_prob.mean()

            if not torch.isfinite(loss):

                raise RuntimeError(
                    "Non-finite flow loss."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                flow.parameters(),
                GRAD_CLIP
            )

            optimizer.step()

            train_losses.append(
                loss.item()
            )

        train_loss = float(
            np.mean(train_losses)
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        flow.eval()

        val_losses = []

        with torch.no_grad():

            for (x,) in val_loader:

                x = x.to(
                    DEVICE,
                    non_blocking=True
                )

                x = x.float()

                log_prob = flow_log_prob(
                    flow,
                    x
                )

                loss = -log_prob.mean()

                val_losses.append(
                    loss.item()
                )

        val_loss = float(
            np.mean(val_losses)
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        history.append(
            {
                "epoch": epoch,
                "train_nll": train_loss,
                "val_nll": val_loss,
                "time_sec": epoch_time,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train NLL={train_loss:.5f} | "
            f"Val NLL={val_loss:.5f} | "
            f"{epoch_time:.1f}s"
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in flow.state_dict().items()
            }

            patience_counter = 0

            print(
                f"  -> New best "
                f"validation NLL: "
                f"{best_val_loss:.5f}"
            )

        else:

            patience_counter += 1

        if patience_counter >= FLOW_PATIENCE:

            print()
            print(
                f"Early stopping after "
                f"{epoch} epochs."
            )

            break

    # --------------------------------------------------------
    # RESTORE BEST MODEL
    # --------------------------------------------------------

    if best_state is None:

        raise RuntimeError(
            "No valid flow checkpoint was produced."
        )

    flow.load_state_dict(
        best_state
    )

    flow.to(DEVICE)

    flow.eval()

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = os.path.join(
        OUTPUT_DIR,
        "training_history.csv"
    )

    pd.DataFrame(history).to_csv(
        history_path,
        index=False
    )

    return flow, best_val_loss


# ============================================================
# SCORE DATASET
# ============================================================

@torch.no_grad()
def score_latents(
    flow,
    latents
):

    dataset = torch.utils.data.TensorDataset(
        latents
    )

    loader = DataLoader(
        dataset,
        batch_size=FLOW_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    scores = []

    flow.eval()

    for (x,) in loader:

        x = x.to(
            DEVICE
        ).float()

        log_prob = flow_log_prob(
            flow,
            x
        )

        # Higher NLL = less normal.
        nll = -log_prob

        if not torch.isfinite(nll).all():

            raise RuntimeError(
                "Non-finite anomaly score detected."
            )

        scores.append(
            nll.cpu()
        )

    return torch.cat(
        scores
    )


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def describe_scores(
    name,
    scores
):

    values = scores.numpy()

    print()
    print(name)
    print("-" * 50)

    print(
        f"N       : {len(values):,}"
    )

    print(
        f"Mean    : {np.mean(values):.6f}"
    )

    print(
        f"Median  : {np.median(values):.6f}"
    )

    print(
        f"Std     : {np.std(values):.6f}"
    )

    print(
        f"Min     : {np.min(values):.6f}"
    )

    print(
        f"Max     : {np.max(values):.6f}"
    )

    print(
        f"P90     : {np.percentile(values, 90):.6f}"
    )

    print(
        f"P95     : {np.percentile(values, 95):.6f}"
    )

    print(
        f"P99     : {np.percentile(values, 99):.6f}"
    )


# ============================================================
# FLOW SANITY CHECK
# ============================================================

@torch.no_grad()
def flow_sanity_check(
    flow,
    latents
):

    print()
    print("=" * 70)
    print("FLOW SANITY CHECK")
    print("=" * 70)

    x = latents[
        : min(32, len(latents))
    ].to(DEVICE).float()

    z, log_det_forward = flow(x)

    reconstructed, log_det_inverse = (
        flow.inverse(z)
    )

    reconstruction_error = (
        reconstructed - x
    ).abs().max().item()

    logdet_error = (
        log_det_forward
        + log_det_inverse
    ).abs().max().item()

    print(
        f"Max inverse reconstruction error : "
        f"{reconstruction_error:.10e}"
    )

    print(
        f"Max log-det cancellation error   : "
        f"{logdet_error:.10e}"
    )

    if reconstruction_error > 1e-4:

        raise RuntimeError(
            "RealNVP inverse sanity check failed."
        )

    if logdet_error > 1e-4:

        raise RuntimeError(
            "RealNVP log-det sanity check failed."
        )

    print("Flow inverse check: PASSED")


# ============================================================
# SAVE LATENT DATA
# ============================================================

def save_tensor(
    tensor,
    filename
):

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    torch.save(
        tensor,
        path
    )

    return path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SAAD — LATENT NORMALIZING FLOW")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()
    print("VAE checkpoint:")
    print(
        f"  {VAE_CHECKPOINT}"
    )

    print()
    print("Flow output:")
    print(
        f"  {OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = NormalPatchDataset(
        TRAIN_DIR
    )

    val_dataset = NormalPatchDataset(
        VAL_DIR
    )

    test_dataset = NormalPatchDataset(
        TEST_DIR
    )

    # --------------------------------------------------------
    # Load frozen VAE
    # --------------------------------------------------------

    vae = load_vae()

    # --------------------------------------------------------
    # Extract deterministic μ
    # --------------------------------------------------------

    train_mu = extract_latents(
        vae,
        train_dataset
    )

    val_mu = extract_latents(
        vae,
        val_dataset
    )

    test_mu = extract_latents(
        vae,
        test_dataset
    )

    # --------------------------------------------------------
    # Save raw latent representations
    # --------------------------------------------------------

    save_tensor(
        train_mu,
        "train_mu.pt"
    )

    save_tensor(
        val_mu,
        "val_mu.pt"
    )

    save_tensor(
        test_mu,
        "test_mu.pt"
    )

    # --------------------------------------------------------
    # Fit standardizer ONLY on train set
    # --------------------------------------------------------

    print()
    print(
        "Fitting latent standardization "
        "on normal training set..."
    )

    latent_mean, latent_std = (
        fit_standardizer(train_mu)
    )

    save_tensor(
        latent_mean,
        "latent_mean.pt"
    )

    save_tensor(
        latent_std,
        "latent_std.pt"
    )

    # --------------------------------------------------------
    # Standardize
    # --------------------------------------------------------

    train_z_input = standardize(
        train_mu,
        latent_mean,
        latent_std
    )

    val_z_input = standardize(
        val_mu,
        latent_mean,
        latent_std
    )

    test_z_input = standardize(
        test_mu,
        latent_mean,
        latent_std
    )

    print()
    print(
        "Standardized train latent statistics:"
    )

    print(
        f"Mean: "
        f"{train_z_input.mean().item():.6f}"
    )

    print(
        f"Std : "
        f"{train_z_input.std().item():.6f}"
    )

    # --------------------------------------------------------
    # Build flow
    # --------------------------------------------------------

    print()
    print(
        "Building RealNVP..."
    )

    flow = RealNVP(
        dim=LATENT_DIM,
        num_layers=FLOW_LAYERS,
        hidden_dim=FLOW_HIDDEN,
        scale_clamp=SCALE_CLAMP
    ).to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in flow.parameters()
    )

    print(
        f"Flow parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Initial sanity check
    # --------------------------------------------------------

    flow_sanity_check(
        flow,
        train_z_input
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    flow, best_val_nll = train_flow(
        flow,
        train_z_input,
        val_z_input
    )

    # --------------------------------------------------------
    # Final flow sanity check
    # --------------------------------------------------------

    flow_sanity_check(
        flow,
        train_z_input
    )

    # --------------------------------------------------------
    # Score normal datasets
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "SCORING NORMAL SEABED DATA"
    )

    print(
        "=" * 70
    )

    train_scores = score_latents(
        flow,
        train_z_input
    )

    val_scores = score_latents(
        flow,
        val_z_input
    )

    test_scores = score_latents(
        flow,
        test_z_input
    )

    describe_scores(
        "NORMAL TRAIN",
        train_scores
    )

    describe_scores(
        "NORMAL VALIDATION",
        val_scores
    )

    describe_scores(
        "NORMAL TEST",
        test_scores
    )

    # --------------------------------------------------------
    # Save scores
    # --------------------------------------------------------

    torch.save(
        train_scores,
        os.path.join(
            OUTPUT_DIR,
            "train_scores.pt"
        )
    )

    torch.save(
        val_scores,
        os.path.join(
            OUTPUT_DIR,
            "val_scores.pt"
        )
    )

    torch.save(
        test_scores,
        os.path.join(
            OUTPUT_DIR,
            "test_scores.pt"
        )
    )

    # --------------------------------------------------------
    # Save flow
    # --------------------------------------------------------

    flow_checkpoint = os.path.join(
        OUTPUT_DIR,
        "best_flow.pt"
    )

    torch.save(
        {
            "model_state_dict":
                flow.state_dict(),

            "latent_dim":
                LATENT_DIM,

            "num_layers":
                FLOW_LAYERS,

            "hidden_dim":
                FLOW_HIDDEN,

            "scale_clamp":
                SCALE_CLAMP,

            "best_val_nll":
                best_val_nll,

            "seed":
                SEED,
        },
        flow_checkpoint
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = os.path.join(
        OUTPUT_DIR,
        "flow_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SAAD — LATENT NORMALIZING FLOW\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            f"Device: {DEVICE}\n"
        )

        if torch.cuda.is_available():

            f.write(
                "GPU: "
                + torch.cuda.get_device_name(0)
                + "\n"
            )

        f.write("\n")

        f.write(
            "VAE checkpoint:\n"
        )

        f.write(
            VAE_CHECKPOINT
            + "\n\n"
        )

        f.write(
            "Dataset:\n"
        )

        f.write(
            f"Train patches: "
            f"{len(train_dataset):,}\n"
        )

        f.write(
            f"Val patches: "
            f"{len(val_dataset):,}\n"
        )

        f.write(
            f"Test patches: "
            f"{len(test_dataset):,}\n\n"
        )

        f.write(
            "Flow configuration:\n"
        )

        f.write(
            f"Latent dimension: "
            f"{LATENT_DIM}\n"
        )

        f.write(
            f"Coupling layers: "
            f"{FLOW_LAYERS}\n"
        )

        f.write(
            f"Hidden dimension: "
            f"{FLOW_HIDDEN}\n"
        )

        f.write(
            f"Scale clamp: "
            f"{SCALE_CLAMP}\n"
        )

        f.write(
            f"Best validation NLL: "
            f"{best_val_nll:.8f}\n\n"
        )

        for name, scores in [
            ("NORMAL TRAIN", train_scores),
            ("NORMAL VALIDATION", val_scores),
            ("NORMAL TEST", test_scores),
        ]:

            values = scores.numpy()

            f.write(
                name + "\n"
            )

            f.write(
                "-" * 50
                + "\n"
            )

            f.write(
                f"N       : "
                f"{len(values):,}\n"
            )

            f.write(
                f"Mean    : "
                f"{np.mean(values):.8f}\n"
            )

            f.write(
                f"Median  : "
                f"{np.median(values):.8f}\n"
            )

            f.write(
                f"Std     : "
                f"{np.std(values):.8f}\n"
            )

            f.write(
                f"Min     : "
                f"{np.min(values):.8f}\n"
            )

            f.write(
                f"Max     : "
                f"{np.max(values):.8f}\n"
            )

            f.write(
                f"P90     : "
                f"{np.percentile(values, 90):.8f}\n"
            )

            f.write(
                f"P95     : "
                f"{np.percentile(values, 95):.8f}\n"
            )

            f.write(
                f"P99     : "
                f"{np.percentile(values, 99):.8f}\n\n"
            )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FLOW TRAINING COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Best validation NLL: "
        f"{best_val_nll:.6f}"
    )

    print()
    print("Saved:")

    print(
        f"  {flow_checkpoint}"
    )

    print(
        f"  {summary_path}"
    )

    print(
        f"  {os.path.join(OUTPUT_DIR, 'training_history.csv')}"
    )

    print(
        f"  {os.path.join(OUTPUT_DIR, 'latent_mean.pt')}"
    )

    print(
        f"  {os.path.join(OUTPUT_DIR, 'latent_std.pt')}"
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Evaluate this trained flow on:"
    )

    print(
        "  1. Normal seabed"
    )

    print(
        "  2. GhostVision anthropogenic"
    )

    print(
        "  3. AI4Shipwrecks anthropogenic"
    )

    print(
        "  4. SubPipeMini2 anthropogenic"
    )

    print(
        "  5. Hard-negative background"
    )

    print()
    print(
        "Higher flow NLL = less likely under the "
        "learned normal-seabed distribution."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    seed_everything(SEED)

    main()