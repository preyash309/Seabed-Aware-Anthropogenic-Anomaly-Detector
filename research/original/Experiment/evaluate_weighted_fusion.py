# ============================================================
# SAAD — VALIDATION-WEIGHTED GLOBAL FUSION
# ============================================================
#
# Purpose:
#   Select YOLO / VAE / Flow fusion weights using ONLY validation
#   data, then evaluate the frozen system on the untouched test set.
#
# IMPORTANT:
#   - Do NOT optimize weights on the test set.
#   - This uses GLOBAL image-level anomaly scores, not the failed
#     ROI-aligned formulation.
#   - YOLO, VAE and Flow models are frozen.
#
# Signals:
#   YOLO  = max YOLO confidence over image
#   VAE   = max VAE reconstruction error over 256x256 patches
#   Flow  = max RealNVP NLL over 256x256 patches
#
# Fusion:
#   S = w_y * YOLO + w_v * VAE + w_f * Flow
#   where w_y + w_v + w_f = 1
#
# Weight selection:
#   maximize validation PR-AUC
#
# Output:
#   E:\SIH\SIH_Results\validation_weighted_fusion
# ============================================================

import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

SEED = 42

DATA_ROOT = Path(r"E:\SIH\Datasets\SAAD_baseline")
MANIFEST = DATA_ROOT / "manifest.csv"

YOLO_WEIGHTS = Path(
    r"E:\SIH\SIH_Results\yolo26s_generic_baseline\weights\best.pt"
)

VAE_WEIGHTS = Path(
    r"E:\SIH\SIH_Results\vae_normal_seabed\checkpoints\best.pt"
)

FLOW_WEIGHTS = Path(
    r"E:\SIH\SIH_Results\latent_normalizing_flow\best_flow.pt"
)

LATENT_MEAN = Path(
    r"E:\SIH\SIH_Results\latent_normalizing_flow\latent_mean.pt"
)

LATENT_STD = Path(
    r"E:\SIH\SIH_Results\latent_normalizing_flow\latent_std.pt"
)

VAE_NORMAL_ROOT = Path(
    r"E:\SIH\Datasets\SAAD_VAE"
)

OUTPUT_DIR = Path(
    r"E:\SIH\SIH_Results\validation_weighted_fusion"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

YOLO_CONF = 0.001

PATCH_SIZE = 256
PATCH_STRIDE = 192

# Grid resolution.
#
# 0.05 means:
#   0.00, 0.05, 0.10, ... 1.00
#
# Number of combinations = 231.
WEIGHT_STEP = 0.05

BATCH_SIZE = 64

NUM_WORKERS = 4


# ============================================================
# REPRODUCIBILITY
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

print("=" * 70)
print("SAAD — VALIDATION-WEIGHTED GLOBAL FUSION")
print("=" * 70)

print("Device :", DEVICE)
print("Output :", OUTPUT_DIR)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    MANIFEST,
    YOLO_WEIGHTS,
    VAE_WEIGHTS,
    FLOW_WEIGHTS,
    LATENT_MEAN,
    LATENT_STD,
]

for p in required_files:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file:\n{p}")

print("\nAll required model/data files found.")


# ============================================================
# LOAD MANIFEST
# ============================================================

manifest = pd.read_csv(MANIFEST)

print("\nManifest columns:")
print(list(manifest.columns))

required_cols = [
    "output_image",
    "split",
    "saad_class",
]

for c in required_cols:
    if c not in manifest.columns:
        raise RuntimeError(
            f"Manifest does not contain required column: {c}"
        )


def normalize_class(x):

    s = str(x).strip().lower()

    if "anthropogenic" in s:
        return "anthropogenic"

    if "normal" in s:
        return "normal"

    return s


manifest["saad_class_norm"] = manifest["saad_class"].apply(
    normalize_class
)

print("\nSAAD classes:")
print(manifest["saad_class_norm"].value_counts(dropna=False))


# ============================================================
# RESOLVE IMAGE PATH
# ============================================================

def resolve_image_path(x):

    p = Path(str(x))

    if p.exists():
        return p

    # If manifest contains a relative path
    p2 = DATA_ROOT / str(x)

    if p2.exists():
        return p2

    # Try output image location
    p3 = DATA_ROOT / "images" / "val" / p.name

    if p3.exists():
        return p3

    p4 = DATA_ROOT / "images" / "test" / p.name

    if p4.exists():
        return p4

    p5 = DATA_ROOT / "images" / "train" / p.name

    if p5.exists():
        return p5

    return None


# ============================================================
# SPLIT DATA
# ============================================================

val_df = manifest[
    manifest["split"].astype(str).str.lower() == "val"
].copy()

test_df = manifest[
    manifest["split"].astype(str).str.lower() == "test"
].copy()

print("\nSplit sizes:")
print("Validation:", len(val_df))
print("Test      :", len(test_df))


# ============================================================
# IMPORTANT:
# Keep only one row per output image.
#
# The manifest can contain multiple rows for some source-level
# bookkeeping. Fusion is image-level.
# ============================================================

val_df = val_df.drop_duplicates(
    subset=["output_image"]
).reset_index(drop=True)

test_df = test_df.drop_duplicates(
    subset=["output_image"]
).reset_index(drop=True)

print("\nUnique images:")
print("Validation:", len(val_df))
print("Test      :", len(test_df))


# ============================================================
# NORMAL / ANTHROPOGENIC SPLITS
# ============================================================

val_normal = val_df[
    val_df["saad_class_norm"] == "normal"
].copy()

val_anthro = val_df[
    val_df["saad_class_norm"] == "anthropogenic"
].copy()

test_normal = test_df[
    test_df["saad_class_norm"] == "normal"
].copy()

test_anthro = test_df[
    test_df["saad_class_norm"] == "anthropogenic"
].copy()

print("\nValidation:")
print("  Normal         :", len(val_normal))
print("  Anthropogenic  :", len(val_anthro))

print("\nTest:")
print("  Normal         :", len(test_normal))
print("  Anthropogenic  :", len(test_anthro))


if len(val_anthro) == 0:
    raise RuntimeError(
        "No anthropogenic validation images found."
    )

if len(test_anthro) == 0:
    raise RuntimeError(
        "No anthropogenic test images found."
    )


# ============================================================
# LOAD YOLO
# ============================================================

print("\n" + "=" * 70)
print("LOADING YOLO")
print("=" * 70)

from ultralytics import YOLO

yolo = YOLO(str(YOLO_WEIGHTS))

print("YOLO loaded.")


# ============================================================
# VAE ARCHITECTURE
# EXACT ARCHITECTURE USED FOR V1
# ============================================================

class ConvVAE(nn.Module):

    def __init__(self, latent_dim=128):

        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128, 256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256, 512,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.fc_mu = nn.Linear(
            512 * 8 * 8,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            512 * 8 * 8,
            latent_dim
        )

        self.fc_decode = nn.Linear(
            latent_dim,
            512 * 8 * 8
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                512, 256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32, 1,
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
            z.size(0),
            512,
            8,
            8
        )

        return self.decoder(h)

    def forward(self, x):

        mu, logvar = self.encode(x)

        std = torch.exp(
            0.5 * torch.clamp(
                logvar,
                -10.0,
                10.0
            )
        )

        z = mu + std * torch.randn_like(std)

        recon = self.decode(z)

        return recon, mu, logvar


print("\nLoading VAE...")

vae = ConvVAE(latent_dim=128).to(DEVICE)

vae_ckpt = torch.load(
    VAE_WEIGHTS,
    map_location=DEVICE
)

if "model_state_dict" in vae_ckpt:
    vae.load_state_dict(
        vae_ckpt["model_state_dict"]
    )
elif "state_dict" in vae_ckpt:
    vae.load_state_dict(
        vae_ckpt["state_dict"]
    )
else:
    vae.load_state_dict(vae_ckpt)

vae.eval()

print("VAE loaded.")


# ============================================================
# REALNVP ARCHITECTURE
#
# EXACT ARCHITECTURE OF THE TRAINED CHECKPOINT
#
# Checkpoint keys:
#
#   layers.0.mask
#   layers.0.net.0.weight
#   layers.0.net.2.weight
#   layers.0.net.4.weight
#
# Coupling network:
#
#   64 -> 256 -> 256 -> 128
#
# Last 128 outputs:
#   first 64  = scale
#   last 64   = translation
#
# 8 coupling layers
# scale clamp = 1.5
# ============================================================

class AffineCoupling(nn.Module):

    def __init__(
        self,
        dim=128,
        hidden=256,
        mask_first=True,
        scale_clamp=1.5
    ):

        super().__init__()

        self.dim = dim
        self.scale_clamp = scale_clamp

        # IMPORTANT:
        # The original trained model stores this as:
        #
        # layers.X.mask
        #
        # so it MUST be registered as a buffer.
        if mask_first:

            mask = torch.cat([
                torch.ones(dim // 2),
                torch.zeros(dim // 2)
            ])

        else:

            mask = torch.cat([
                torch.zeros(dim // 2),
                torch.ones(dim // 2)
            ])

        self.register_buffer(
            "mask",
            mask
        )

        # IMPORTANT:
        # This must be directly called `self.net`.
        #
        # NOT:
        # self.net = CouplingNet()
        #
        # because that would produce:
        #
        # layers.X.net.net.0.weight
        #
        # instead of the checkpoint's:
        #
        # layers.X.net.0.weight
        self.net = nn.Sequential(

            nn.Linear(
                dim // 2,
                hidden
            ),

            nn.ReLU(),

            nn.Linear(
                hidden,
                hidden
            ),

            nn.ReLU(),

            nn.Linear(
                hidden,
                dim
            ),
        )

    def forward(
        self,
        x,
        reverse=False
    ):

        # ----------------------------------------------------
        # Split according to mask
        # ----------------------------------------------------

        x_masked = x * self.mask

        x_pass = x_masked

        # The unmasked half is what the network receives.
        x_cond = x_pass[
            :,
            self.mask.bool()
        ]

        # ----------------------------------------------------
        # Predict scale + translation
        # ----------------------------------------------------

        st = self.net(
            x_cond
        )

        s = st[:, :self.dim // 2]

        t = st[:, self.dim // 2:]

        s = self.scale_clamp * torch.tanh(s)

        # ----------------------------------------------------
        # Extract transform half
        # ----------------------------------------------------

        transform_mask = ~self.mask.bool()

        x_transform = x[
            :,
            transform_mask
        ]

        # ----------------------------------------------------
        # Forward transformation
        # ----------------------------------------------------

        if not reverse:

            y_transform = (
                x_transform
                * torch.exp(s)
                + t
            )

            logdet = s.sum(
                dim=1
            )

        # ----------------------------------------------------
        # Inverse transformation
        # ----------------------------------------------------

        else:

            y_transform = (
                x_transform
                - t
            ) * torch.exp(-s)

            logdet = -s.sum(
                dim=1
            )

        # ----------------------------------------------------
        # Reconstruct complete vector
        # ----------------------------------------------------

        y = x.clone()

        y[
            :,
            transform_mask
        ] = y_transform

        return y, logdet


class RealNVP(nn.Module):

    def __init__(
        self,
        dim=128,
        hidden=256,
        n_layers=8,
        scale_clamp=1.5
    ):

        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(n_layers):

            self.layers.append(
                AffineCoupling(
                    dim=dim,
                    hidden=hidden,
                    mask_first=(i % 2 == 0),
                    scale_clamp=scale_clamp
                )
            )

    def forward(self, x):

        z = x

        total_logdet = torch.zeros(
            x.size(0),
            device=x.device,
            dtype=x.dtype
        )

        for layer in self.layers:

            z, logdet = layer(
                z,
                reverse=False
            )

            total_logdet += logdet

        return z, total_logdet

    def inverse(self, z):

        x = z

        for layer in reversed(
            self.layers
        ):

            x, _ = layer(
                x,
                reverse=True
            )

        return x


# ============================================================
# LOAD REALNVP
# ============================================================

print("\nLoading RealNVP...")

flow = RealNVP(
    dim=128,
    hidden=256,
    n_layers=8,
    scale_clamp=1.5
).to(DEVICE)

flow_ckpt = torch.load(
    FLOW_WEIGHTS,
    map_location=DEVICE
)

if "model_state_dict" in flow_ckpt:

    flow_state = (
        flow_ckpt[
            "model_state_dict"
        ]
    )

elif "state_dict" in flow_ckpt:

    flow_state = (
        flow_ckpt[
            "state_dict"
        ]
    )

else:

    flow_state = flow_ckpt


# ------------------------------------------------------------
# Print checkpoint keys BEFORE loading.
# This makes future architecture mismatches immediately obvious.
# ------------------------------------------------------------

print("\nCheckpoint sample keys:")

for key in list(
    flow_state.keys()
)[:12]:

    print(
        " ",
        key
    )


# ------------------------------------------------------------
# Exact compatibility check
# ------------------------------------------------------------

model_keys = set(
    flow.state_dict().keys()
)

checkpoint_keys = set(
    flow_state.keys()
)

missing = sorted(
    model_keys
    - checkpoint_keys
)

unexpected = sorted(
    checkpoint_keys
    - model_keys
)

if missing or unexpected:

    print("\nFLOW CHECKPOINT MISMATCH")

    if missing:

        print("\nMissing keys:")

        for key in missing[:30]:

            print(
                " ",
                key
            )

    if unexpected:

        print("\nUnexpected keys:")

        for key in unexpected[:30]:

            print(
                " ",
                key
            )

    raise RuntimeError(
        "\nThe RealNVP architecture still does not "
        "match the trained checkpoint."
    )


flow.load_state_dict(
    flow_state,
    strict=True
)

flow.eval()


# ============================================================
# LOAD LATENT STANDARDIZATION
# ============================================================

latent_mean = torch.load(
    LATENT_MEAN,
    map_location=DEVICE
).float()

latent_std = torch.load(
    LATENT_STD,
    map_location=DEVICE
).float()

latent_mean = latent_mean.view(
    1,
    -1
)

latent_std = latent_std.view(
    1,
    -1
)

latent_std = torch.clamp(
    latent_std,
    min=1e-6
)


# ============================================================
# FLOW SANITY CHECK
# ============================================================

print("\nRunning RealNVP sanity check...")

with torch.no_grad():

    test_z = torch.randn(
        4,
        128,
        device=DEVICE
    )

    transformed, forward_ld = flow(
        test_z
    )

    reconstructed = flow.inverse(
        transformed
    )

    reconstruction_error = (
        torch.max(
            torch.abs(
                reconstructed
                - test_z
            )
        )
        .item()
    )

print(
    "Max inverse reconstruction error:",
    reconstruction_error
)

if reconstruction_error > 1e-4:

    raise RuntimeError(
        "RealNVP inverse sanity check failed."
    )

print(
    "RealNVP compatibility check PASSED."
)

print(
    "Flow loaded."
)


# ============================================================
# IMAGE HELPERS
# ============================================================

def load_gray(path):

    img = Image.open(path).convert("L")

    arr = np.asarray(
        img,
        dtype=np.float32
    ) / 255.0

    return arr


def make_tensor(arr):

    return torch.from_numpy(
        arr
    ).unsqueeze(0).unsqueeze(0).float()


def extract_patches(arr):

    h, w = arr.shape

    patches = []
    positions = []

    # If smaller than patch size, resize.
    if h < PATCH_SIZE or w < PATCH_SIZE:

        img = Image.fromarray(
            np.clip(
                arr * 255.0,
                0,
                255
            ).astype(np.uint8)
        )

        img = img.resize(
            (PATCH_SIZE, PATCH_SIZE),
            Image.Resampling.BILINEAR
        )

        p = np.asarray(
            img,
            dtype=np.float32
        ) / 255.0

        return [
            p
        ], [
            (0, 0)
        ]

    ys = list(
        range(
            0,
            h - PATCH_SIZE + 1,
            PATCH_STRIDE
        )
    )

    xs = list(
        range(
            0,
            w - PATCH_SIZE + 1,
            PATCH_STRIDE
        )
    )

    # Always cover the right/bottom edges.
    last_y = h - PATCH_SIZE
    last_x = w - PATCH_SIZE

    if ys[-1] != last_y:
        ys.append(last_y)

    if xs[-1] != last_x:
        xs.append(last_x)

    for y in ys:

        for x in xs:

            patch = arr[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            patches.append(patch)
            positions.append((x, y))

    return patches, positions


# ============================================================
# VAE PATCH SCORES
# ============================================================

@torch.no_grad()
def vae_patch_scores(patches):

    if len(patches) == 0:
        return np.array([], dtype=np.float32)

    arr = np.stack(patches)

    x = torch.from_numpy(
        arr
    ).unsqueeze(1).float()

    scores = []

    for start in range(
        0,
        len(x),
        BATCH_SIZE
    ):

        xb = x[
            start:start + BATCH_SIZE
        ].to(DEVICE)

        recon, _, _ = vae(xb)

        mse = F.mse_loss(
            recon,
            xb,
            reduction="none"
        )

        mse = mse.mean(
            dim=(1, 2, 3)
        )

        scores.append(
            mse.detach()
            .cpu()
            .numpy()
        )

    return np.concatenate(scores)


# ============================================================
# FLOW PATCH SCORES
# ============================================================

@torch.no_grad()
def flow_patch_scores(patches):

    if len(patches) == 0:
        return np.array([], dtype=np.float32)

    arr = np.stack(patches)

    x = torch.from_numpy(
        arr
    ).unsqueeze(1).float()

    nll_scores = []

    for start in range(
        0,
        len(x),
        BATCH_SIZE
    ):

        xb = x[
            start:start + BATCH_SIZE
        ].to(DEVICE)

        mu, _ = vae.encode(xb)

        z = (
            mu - latent_mean
        ) / latent_std

        z_flow, logdet = flow(z)

        log_base = (
            -0.5
            * (
                z_flow ** 2
                + np.log(2.0 * np.pi)
            )
        ).sum(dim=1)

        log_prob = (
            log_base
            + logdet
        )

        nll = -log_prob

        nll_scores.append(
            nll.detach()
            .float()
            .cpu()
            .numpy()
        )

    return np.concatenate(nll_scores)


# ============================================================
# YOLO IMAGE SCORE
# ============================================================

def yolo_image_score(path):

    try:

        result = yolo.predict(
            source=str(path),
            conf=YOLO_CONF,
            verbose=False,
            device=0 if DEVICE.type == "cuda" else "cpu",
        )[0]

        if result.boxes is None:
            return 0.0

        if len(result.boxes) == 0:
            return 0.0

        confs = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        if len(confs) == 0:
            return 0.0

        return float(
            np.max(confs)
        )

    except Exception as e:

        print(
            f"YOLO ERROR: {path}\n{e}"
        )

        return 0.0


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def score_image(path):

    arr = load_gray(path)

    patches, positions = extract_patches(
        arr
    )

    vae_scores = vae_patch_scores(
        patches
    )

    flow_scores = flow_patch_scores(
        patches
    )

    yolo_score = yolo_image_score(
        path
    )

    if len(vae_scores) == 0:
        vae_score = 0.0
    else:
        vae_score = float(
            np.max(vae_scores)
        )

    if len(flow_scores) == 0:
        flow_score = 0.0
    else:
        flow_score = float(
            np.max(flow_scores)
        )

    return {
        "yolo_raw": yolo_score,
        "vae_raw": vae_score,
        "flow_raw": flow_score,
    }


# ============================================================
# SCORE DATAFRAME
# ============================================================

def score_dataframe(df, name):

    rows = []

    print("\n" + "=" * 70)
    print(f"SCORING {name}")
    print("=" * 70)

    total = len(df)

    for i, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        path = resolve_image_path(
            row["output_image"]
        )

        if path is None:

            print(
                f"\nWARNING: image not found: "
                f"{row['output_image']}"
            )

            continue

        scores = score_image(
            path
        )

        rows.append({

            "output_image":
                str(row["output_image"]),

            "path":
                str(path),

            "dataset":
                str(row.get(
                    "dataset",
                    "unknown"
                )),

            "group":
                str(row.get(
                    "group",
                    "unknown"
                )),

            "label":
                1 if row["saad_class_norm"]
                == "anthropogenic"
                else 0,

            "yolo_raw":
                scores["yolo_raw"],

            "vae_raw":
                scores["vae_raw"],

            "flow_raw":
                scores["flow_raw"],
        })

        if (
            i % 25 == 0
            or i == total
        ):

            print(
                f"\rProcessed "
                f"{i}/{total}",
                end=""
            )

    print()

    result = pd.DataFrame(rows)

    if len(result) == 0:
        raise RuntimeError(
            f"No images successfully scored for {name}"
        )

    return result


# ============================================================
# SCORE VALIDATION
# ============================================================

val_scores = score_dataframe(
    val_df,
    "VALIDATION"
)

val_scores.to_csv(
    OUTPUT_DIR / "validation_raw_scores.csv",
    index=False
)

# ============================================================
# SCORE TEST
# ============================================================

test_scores = score_dataframe(
    test_df,
    "TEST"
)

test_scores.to_csv(
    OUTPUT_DIR / "test_raw_scores.csv",
    index=False
)


# ============================================================
# NORMAL VALIDATION SCORES
#
# These are used ONLY to calibrate the scale of each signal.
#
# This is deliberately separated from the anthropogenic
# validation images used for weight selection.
# ============================================================

normal_val_scores = val_scores[
    val_scores["label"] == 0
].copy()

if len(normal_val_scores) < 20:
    raise RuntimeError(
        "Too few normal validation images for calibration."
    )


# ============================================================
# ROBUST NORMALIZATION
#
# Each signal is mapped using:
#
#   (x - P5) / (P95 - P5)
#
# then clipped to [0,1].
#
# This avoids a few extreme normal images dominating the scale.
# ============================================================

normalization_stats = {}

for signal in [
    "yolo_raw",
    "vae_raw",
    "flow_raw",
]:

    p5 = float(
        np.percentile(
            normal_val_scores[signal],
            5
        )
    )

    p95 = float(
        np.percentile(
            normal_val_scores[signal],
            95
        )
    )

    if p95 <= p5:
        p95 = p5 + 1e-8

    normalization_stats[
        signal
    ] = {
        "p5": p5,
        "p95": p95,
    }


print("\n" + "=" * 70)
print("NORMAL-VALIDATION CALIBRATION")
print("=" * 70)

for signal, stats in normalization_stats.items():

    print(
        f"{signal:10s} "
        f"P5={stats['p5']:.8f} "
        f"P95={stats['p95']:.8f}"
    )


# ============================================================
# APPLY NORMALIZATION
# ============================================================

def normalize_signal(
    series,
    p5,
    p95
):

    x = (
        series.astype(np.float64)
        - p5
    ) / (
        p95 - p5
    )

    return np.clip(
        x,
        0.0,
        1.0
    )


for df in [
    val_scores,
    test_scores,
]:

    for signal in [
        "yolo",
        "vae",
        "flow",
    ]:

        raw = signal + "_raw"

        p5 = normalization_stats[
            raw
        ]["p5"]

        p95 = normalization_stats[
            raw
        ]["p95"]

        df[
            signal
        ] = normalize_signal(
            df[raw],
            p5,
            p95
        )


# ============================================================
# WEIGHT GRID
# ============================================================

def generate_weights(step=0.05):

    weights = []

    n = int(
        round(
            1.0 / step
        )
    )

    for iy in range(n + 1):

        wy = iy * step

        for iv in range(n + 1):

            wv = iv * step

            wf = 1.0 - wy - wv

            if wf < -1e-9:
                continue

            wf = max(
                0.0,
                wf
            )

            weights.append(
                (
                    round(wy, 6),
                    round(wv, 6),
                    round(wf, 6),
                )
            )

    return weights


weight_grid = generate_weights(
    WEIGHT_STEP
)

print("\n" + "=" * 70)
print("WEIGHT SEARCH")
print("=" * 70)

print(
    "Weight combinations:",
    len(weight_grid)
)

print(
    "Objective: validation PR-AUC"
)


# ============================================================
# VALIDATION WEIGHT SEARCH
#
# IMPORTANT:
# We optimize using the anthropogenic-vs-normal validation
# labels only.
#
# Test data is untouched.
# ============================================================

val_y = val_scores[
    "label"
].to_numpy(
    dtype=np.int32
)

val_yolo = val_scores[
    "yolo"
].to_numpy(
    dtype=np.float64
)

val_vae = val_scores[
    "vae"
].to_numpy(
    dtype=np.float64
)

val_flow = val_scores[
    "flow"
].to_numpy(
    dtype=np.float64
)

weight_results = []

for wy, wv, wf in weight_grid:

    fused = (
        wy * val_yolo
        + wv * val_vae
        + wf * val_flow
    )

    roc = roc_auc_score(
        val_y,
        fused
    )

    pr = average_precision_score(
        val_y,
        fused
    )

    precision, recall, thresholds = (
        precision_recall_curve(
            val_y,
            fused
        )
    )

    f1_values = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
            + 1e-12
        )
    )

    best_idx = int(
        np.argmax(f1_values)
    )

    best_f1 = float(
        f1_values[best_idx]
    )

    weight_results.append({

        "w_yolo": wy,
        "w_vae": wv,
        "w_flow": wf,

        "val_roc_auc": roc,
        "val_pr_auc": pr,

        "val_best_f1": best_f1,
    })


weight_results = pd.DataFrame(
    weight_results
)

weight_results = weight_results.sort_values(
    [
        "val_pr_auc",
        "val_roc_auc",
    ],
    ascending=False
).reset_index(
    drop=True
)

weight_results.to_csv(
    OUTPUT_DIR / "weight_search_results.csv",
    index=False
)


# ============================================================
# BEST WEIGHTS
# ============================================================

best = weight_results.iloc[0]

BEST_W_YOLO = float(
    best["w_yolo"]
)

BEST_W_VAE = float(
    best["w_vae"]
)

BEST_W_FLOW = float(
    best["w_flow"]
)

print("\nBEST VALIDATION WEIGHTS")
print(
    f"YOLO : {BEST_W_YOLO:.2f}"
)

print(
    f"VAE  : {BEST_W_VAE:.2f}"
)

print(
    f"Flow : {BEST_W_FLOW:.2f}"
)

print(
    f"Validation ROC-AUC : "
    f"{best['val_roc_auc']:.6f}"
)

print(
    f"Validation PR-AUC  : "
    f"{best['val_pr_auc']:.6f}"
)

print(
    f"Validation Best F1 : "
    f"{best['val_best_f1']:.6f}"
)


# ============================================================
# FUSION FUNCTION
# ============================================================

def fuse(
    df,
    wy,
    wv,
    wf
):

    return (
        wy * df["yolo"].to_numpy(
            dtype=np.float64
        )
        +
        wv * df["vae"].to_numpy(
            dtype=np.float64
        )
        +
        wf * df["flow"].to_numpy(
            dtype=np.float64
        )
    )


# ============================================================
# FIND VALIDATION THRESHOLD
#
# We choose threshold using validation F1.
#
# This threshold is NOT allowed to see test labels.
# ============================================================

val_fused = fuse(
    val_scores,
    BEST_W_YOLO,
    BEST_W_VAE,
    BEST_W_FLOW
)

precision, recall, thresholds = (
    precision_recall_curve(
        val_y,
        val_fused
    )
)

f1_values = (
    2.0
    * precision
    * recall
    / (
        precision
        + recall
        + 1e-12
    )
)

best_val_f1_idx = int(
    np.argmax(f1_values)
)

if (
    best_val_f1_idx
    < len(thresholds)
):

    validation_threshold = float(
        thresholds[
            best_val_f1_idx
        ]
    )

else:

    validation_threshold = float(
        np.median(val_fused)
    )


print("\nValidation-selected threshold:")
print(
    f"{validation_threshold:.8f}"
)


# ============================================================
# NORMAL-ONLY THRESHOLD
#
# Also calculate a conservative threshold from normal validation
# distribution. This is useful for deployment-style discussion.
# ============================================================

normal_val_fused = val_fused[
    val_scores["label"].to_numpy()
    == 0
]

normal_p95_threshold = float(
    np.percentile(
        normal_val_fused,
        95
    )
)

normal_p99_threshold = float(
    np.percentile(
        normal_val_fused,
        99
    )
)

print("\nNormal-validation thresholds:")
print(
    f"P95 : {normal_p95_threshold:.8f}"
)

print(
    f"P99 : {normal_p99_threshold:.8f}"
)


# ============================================================
# TEST EVALUATION
# ============================================================

test_y = test_scores[
    "label"
].to_numpy(
    dtype=np.int32
)

test_fused = fuse(
    test_scores,
    BEST_W_YOLO,
    BEST_W_VAE,
    BEST_W_FLOW
)

test_scores[
    "fusion_weighted"
] = test_fused

test_scores[
    "fusion_equal"
] = (
    1.0 / 3.0
    * (
        test_scores["yolo"]
        + test_scores["vae"]
        + test_scores["flow"]
    )
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def evaluate_scores(
    y,
    scores,
    threshold=None
):

    roc = roc_auc_score(
        y,
        scores
    )

    pr = average_precision_score(
        y,
        scores
    )

    precision_curve, recall_curve, thresholds = (
        precision_recall_curve(
            y,
            scores
        )
    )

    f1_curve = (
        2.0
        * precision_curve
        * recall_curve
        / (
            precision_curve
            + recall_curve
            + 1e-12
        )
    )

    idx = int(
        np.argmax(f1_curve)
    )

    best_f1 = float(
        f1_curve[idx]
    )

    best_precision = float(
        precision_curve[idx]
    )

    best_recall = float(
        recall_curve[idx]
    )

    if (
        idx < len(thresholds)
    ):

        best_threshold = float(
            thresholds[idx]
        )

    else:

        best_threshold = float(
            np.median(scores)
        )

    result = {

        "roc_auc":
            roc,

        "pr_auc":
            pr,

        "best_f1":
            best_f1,

        "best_f1_precision":
            best_precision,

        "best_f1_recall":
            best_recall,

        "best_f1_threshold":
            best_threshold,
    }

    if threshold is not None:

        pred = (
            scores
            >= threshold
        )

        result[
            "threshold_precision"
        ] = precision_score(
            y,
            pred,
            zero_division=0
        )

        result[
            "threshold_recall"
        ] = recall_score(
            y,
            pred,
            zero_division=0
        )

        result[
            "threshold_f1"
        ] = f1_score(
            y,
            pred,
            zero_division=0
        )

        result[
            "threshold_tp"
        ] = int(
            np.sum(
                (pred == 1)
                & (y == 1)
            )
        )

        result[
            "threshold_fp"
        ] = int(
            np.sum(
                (pred == 1)
                & (y == 0)
            )
        )

        result[
            "threshold_fn"
        ] = int(
            np.sum(
                (pred == 0)
                & (y == 1)
            )
        )

        result[
            "threshold_tn"
        ] = int(
            np.sum(
                (pred == 0)
                & (y == 0)
            )
        )

    return result


# ============================================================
# EVALUATE WEIGHTED + EQUAL
# ============================================================

weighted_metrics = evaluate_scores(
    test_y,
    test_fused,
    threshold=validation_threshold
)

equal_scores = test_scores[
    "fusion_equal"
].to_numpy(
    dtype=np.float64
)

equal_metrics = evaluate_scores(
    test_y,
    equal_scores
)


# ============================================================
# ALSO TEST NORMAL-P95/P99 THRESHOLDS
# ============================================================

p95_metrics = evaluate_scores(
    test_y,
    test_fused,
    threshold=normal_p95_threshold
)

p99_metrics = evaluate_scores(
    test_y,
    test_fused,
    threshold=normal_p99_threshold
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print("\nVALIDATION-WEIGHTED YOLO + VAE + FLOW")

print(
    f"ROC-AUC : "
    f"{weighted_metrics['roc_auc']:.6f}"
)

print(
    f"PR-AUC  : "
    f"{weighted_metrics['pr_auc']:.6f}"
)

print(
    f"Best F1 : "
    f"{weighted_metrics['best_f1']:.6f}"
)

print(
    f"Best-F1 Precision : "
    f"{weighted_metrics['best_f1_precision']:.6f}"
)

print(
    f"Best-F1 Recall    : "
    f"{weighted_metrics['best_f1_recall']:.6f}"
)

print(
    f"Validation-selected threshold "
    f"Precision : "
    f"{weighted_metrics['threshold_precision']:.6f}"
)

print(
    f"Validation-selected threshold "
    f"Recall    : "
    f"{weighted_metrics['threshold_recall']:.6f}"
)

print(
    f"Validation-selected threshold "
    f"F1        : "
    f"{weighted_metrics['threshold_f1']:.6f}"
)

print(
    f"TP={weighted_metrics['threshold_tp']} "
    f"FP={weighted_metrics['threshold_fp']} "
    f"FN={weighted_metrics['threshold_fn']} "
    f"TN={weighted_metrics['threshold_tn']}"
)


print("\n" + "-" * 70)

print("EQUAL-WEIGHT YOLO + VAE + FLOW")

print(
    f"ROC-AUC : "
    f"{equal_metrics['roc_auc']:.6f}"
)

print(
    f"PR-AUC  : "
    f"{equal_metrics['pr_auc']:.6f}"
)

print(
    f"Best F1 : "
    f"{equal_metrics['best_f1']:.6f}"
)


print("\n" + "-" * 70)

print("NORMAL-VALIDATION P95 THRESHOLD")

print(
    f"Threshold : "
    f"{normal_p95_threshold:.8f}"
)

print(
    f"Precision : "
    f"{p95_metrics['threshold_precision']:.6f}"
)

print(
    f"Recall    : "
    f"{p95_metrics['threshold_recall']:.6f}"
)

print(
    f"F1        : "
    f"{p95_metrics['threshold_f1']:.6f}"
)

print(
    f"TP={p95_metrics['threshold_tp']} "
    f"FP={p95_metrics['threshold_fp']} "
    f"FN={p95_metrics['threshold_fn']} "
    f"TN={p95_metrics['threshold_tn']}"
)


print("\n" + "-" * 70)

print("NORMAL-VALIDATION P99 THRESHOLD")

print(
    f"Threshold : "
    f"{normal_p99_threshold:.8f}"
)

print(
    f"Precision : "
    f"{p99_metrics['threshold_precision']:.6f}"
)

print(
    f"Recall    : "
    f"{p99_metrics['threshold_recall']:.6f}"
)

print(
    f"F1        : "
    f"{p99_metrics['threshold_f1']:.6f}"
)

print(
    f"TP={p99_metrics['threshold_tp']} "
    f"FP={p99_metrics['threshold_fp']} "
    f"FN={p99_metrics['threshold_fn']} "
    f"TN={p99_metrics['threshold_tn']}"
)


# ============================================================
# DATASET-WISE TEST RESULTS
# ============================================================

dataset_rows = []

for dataset_name, group in test_scores.groupby(
    "dataset"
):

    y = group[
        "label"
    ].to_numpy(
        dtype=np.int32
    )

    scores = group[
        "fusion_weighted"
    ].to_numpy(
        dtype=np.float64
    )

    if (
        len(np.unique(y))
        < 2
    ):

        roc = np.nan
        pr = np.nan

    else:

        roc = roc_auc_score(
            y,
            scores
        )

        pr = average_precision_score(
            y,
            scores
        )

    pred = (
        scores
        >= validation_threshold
    )

    dataset_rows.append({

        "dataset":
            dataset_name,

        "N":
            len(group),

        "positives":
            int(np.sum(y == 1)),

        "negatives":
            int(np.sum(y == 0)),

        "ROC_AUC":
            roc,

        "PR_AUC":
            pr,

        "threshold_precision":
            precision_score(
                y,
                pred,
                zero_division=0
            ),

        "threshold_recall":
            recall_score(
                y,
                pred,
                zero_division=0
            ),

        "threshold_f1":
            f1_score(
                y,
                pred,
                zero_division=0
            ),
    })


dataset_results = pd.DataFrame(
    dataset_rows
)

dataset_results.to_csv(
    OUTPUT_DIR / "dataset_wise_weighted_results.csv",
    index=False
)

print("\n" + "=" * 70)
print("DATASET-WISE WEIGHTED RESULTS")
print("=" * 70)

print(
    dataset_results.to_string(
        index=False
    )
)


# ============================================================
# SAVE COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame([

    {
        "signal":
            "YOLO + VAE + Flow (equal)",

        "ROC_AUC":
            equal_metrics["roc_auc"],

        "PR_AUC":
            equal_metrics["pr_auc"],

        "Best_F1":
            equal_metrics["best_f1"],

        "Best_F1_precision":
            equal_metrics[
                "best_f1_precision"
            ],

        "Best_F1_recall":
            equal_metrics[
                "best_f1_recall"
            ],
    },

    {
        "signal":
            "YOLO + VAE + Flow (validation-weighted)",

        "ROC_AUC":
            weighted_metrics["roc_auc"],

        "PR_AUC":
            weighted_metrics["pr_auc"],

        "Best_F1":
            weighted_metrics["best_f1"],

        "Best_F1_precision":
            weighted_metrics[
                "best_f1_precision"
            ],

        "Best_F1_recall":
            weighted_metrics[
                "best_f1_recall"
            ],
    },

])

comparison.to_csv(
    OUTPUT_DIR / "fusion_comparison.csv",
    index=False
)


# ============================================================
# SAVE PER-IMAGE RESULTS
# ============================================================

test_scores.to_csv(
    OUTPUT_DIR / "per_image_weighted_scores.csv",
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = (
    OUTPUT_DIR
    / "weighted_fusion_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SAAD — VALIDATION-WEIGHTED GLOBAL FUSION\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )

    f.write(
        "IMPORTANT METHODOLOGY\n"
    )

    f.write(
        "Weights were selected using validation data only.\n"
    )

    f.write(
        "The test set was not used for weight selection.\n"
    )

    f.write(
        "YOLO, VAE and Flow models remained frozen.\n"
    )

    f.write(
        "Fusion is image-level/global, not ROI-aligned.\n\n"
    )

    f.write(
        "SELECTED WEIGHTS\n"
    )

    f.write(
        f"YOLO = {BEST_W_YOLO:.4f}\n"
    )

    f.write(
        f"VAE  = {BEST_W_VAE:.4f}\n"
    )

    f.write(
        f"Flow = {BEST_W_FLOW:.4f}\n\n"
    )

    f.write(
        "VALIDATION PERFORMANCE\n"
    )

    f.write(
        f"ROC-AUC = "
        f"{best['val_roc_auc']:.6f}\n"
    )

    f.write(
        f"PR-AUC  = "
        f"{best['val_pr_auc']:.6f}\n"
    )

    f.write(
        f"Best F1 = "
        f"{best['val_best_f1']:.6f}\n\n"
    )

    f.write(
        "TEST PERFORMANCE — VALIDATION-WEIGHTED\n"
    )

    for k, v in weighted_metrics.items():

        f.write(
            f"{k}: {v}\n"
        )

    f.write(
        "\nTEST PERFORMANCE — EQUAL WEIGHT\n"
    )

    for k, v in equal_metrics.items():

        f.write(
            f"{k}: {v}\n"
        )

    f.write(
        "\nNORMAL VALIDATION THRESHOLDS\n"
    )

    f.write(
        f"P95 = {normal_p95_threshold:.8f}\n"
    )

    f.write(
        f"P99 = {normal_p99_threshold:.8f}\n"
    )

    f.write(
        "\nFILES\n"
    )

    f.write(
        "validation_raw_scores.csv\n"
    )

    f.write(
        "test_raw_scores.csv\n"
    )

    f.write(
        "weight_search_results.csv\n"
    )

    f.write(
        "fusion_comparison.csv\n"
    )

    f.write(
        "dataset_wise_weighted_results.csv\n"
    )

    f.write(
        "per_image_weighted_scores.csv\n"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    "\nBest weights:"
)

print(
    f"  YOLO = {BEST_W_YOLO:.2f}"
)

print(
    f"  VAE  = {BEST_W_VAE:.2f}"
)

print(
    f"  Flow = {BEST_W_FLOW:.2f}"
)

print(
    "\nTest weighted fusion:"
)

print(
    f"  ROC-AUC = "
    f"{weighted_metrics['roc_auc']:.6f}"
)

print(
    f"  PR-AUC  = "
    f"{weighted_metrics['pr_auc']:.6f}"
)

print(
    f"  Best F1 = "
    f"{weighted_metrics['best_f1']:.6f}"
)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR
)