# ============================================================
# SAAD — LATENT NORMALIZING FLOW EVALUATION
# ============================================================
#
# Evaluates the trained latent normalizing flow on:
#
#   1. Normal seabed
#   2. GhostVision anthropogenic
#   3. AI4Shipwrecks anthropogenic
#   4. SubPipeMini2 anthropogenic
#   5. Hard-negative background
#
# Pipeline:
#
#   SSS patch
#       ↓
#   Frozen VAE v1
#       ↓
#   deterministic latent μ
#       ↓
#   train-set standardization
#       ↓
#   trained RealNVP
#       ↓
#   negative log likelihood
#       ↓
#   anomaly score
#
# Higher NLL = less likely under normal-seabed distribution.
#
# ============================================================

import os
import random
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 256
LATENT_DIM = 128

BATCH_SIZE = 256
NUM_WORKERS = 0

# ------------------------------------------------------------
# VAE
# ------------------------------------------------------------

VAE_CHECKPOINT = (
    r"E:\SIH\SIH_Results\vae_normal_seabed"
    r"\checkpoints\best.pt"
)

# ------------------------------------------------------------
# Flow
# ------------------------------------------------------------

FLOW_DIR = (
    r"E:\SIH\SIH_Results"
    r"\latent_normalizing_flow"
)

FLOW_CHECKPOINT = os.path.join(
    FLOW_DIR,
    "best_flow.pt"
)

LATENT_MEAN_PATH = os.path.join(
    FLOW_DIR,
    "latent_mean.pt"
)

LATENT_STD_PATH = os.path.join(
    FLOW_DIR,
    "latent_std.pt"
)

# ------------------------------------------------------------
# Normal validation/test data
# ------------------------------------------------------------

VAE_DATA_ROOT = (
    r"E:\SIH\Datasets\SAAD_VAE"
)

NORMAL_VAL_DIR = os.path.join(
    VAE_DATA_ROOT,
    "val"
)

NORMAL_TEST_DIR = os.path.join(
    VAE_DATA_ROOT,
    "test"
)

# ------------------------------------------------------------
# SAAD unified dataset
# ------------------------------------------------------------

SAAD_ROOT = (
    r"E:\SIH\Datasets\SAAD_baseline"
)

SAAD_TEST_IMAGES = os.path.join(
    SAAD_ROOT,
    "images",
    "test"
)

SAAD_TEST_LABELS = os.path.join(
    SAAD_ROOT,
    "labels",
    "test"
)

MANIFEST_PATH = os.path.join(
    SAAD_ROOT,
    "manifest.csv"
)

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    FLOW_DIR,
    "evaluation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# GENERIC IMAGE DATASET
# ============================================================

class ImageFolderDataset(Dataset):

    def __init__(
        self,
        root,
        recursive=True
    ):

        self.root = root

        valid_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        }

        self.files = []

        if recursive:

            for dirpath, _, filenames in os.walk(root):

                for filename in filenames:

                    ext = os.path.splitext(
                        filename
                    )[1].lower()

                    if ext in valid_extensions:

                        self.files.append(
                            os.path.join(
                                dirpath,
                                filename
                            )
                        )

        else:

            for filename in os.listdir(root):

                path = os.path.join(
                    root,
                    filename
                )

                if not os.path.isfile(path):
                    continue

                ext = os.path.splitext(
                    filename
                )[1].lower()

                if ext in valid_extensions:

                    self.files.append(path)

        self.files.sort()

        if len(self.files) == 0:

            raise RuntimeError(
                f"No images found in:\n{root}"
            )

        print(
            f"Loaded {len(self.files):,} images from:"
        )

        print(
            f"  {root}"
        )

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        path = self.files[idx]

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
        )

        image /= 255.0

        image = torch.from_numpy(
            image
        )

        image = image.unsqueeze(0)

        return image, path


# ============================================================
# VAE — EXACT V1 ARCHITECTURE
# ============================================================

class ConvVAE(nn.Module):

    def __init__(
        self,
        latent_dim=128
    ):

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

        self.fc_mu = nn.Linear(
            512 * 8 * 8,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            512 * 8 * 8,
            latent_dim
        )

        # IMPORTANT:
        # Exact name used by trained VAE v1.
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

    def forward(self, x):

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

        s = torch.tanh(s)

        s = (
            s
            * self.scale_clamp
        )

        x_transform = x[
            :,
            self.transform_indices
        ]

        y_transform = (
            x_transform
            * torch.exp(s)
            + t
        )

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

        y_transform = y[
            :,
            self.transform_indices
        ]

        x_transform = (
            y_transform - t
        ) * torch.exp(-s)

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
# LOAD VAE
# ============================================================

def load_vae():

    print()
    print(
        "Loading frozen VAE v1..."
    )

    model = ConvVAE(
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    checkpoint = torch.load(
        VAE_CHECKPOINT,
        map_location=DEVICE
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif "state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[
            key
        ] = value

    missing, unexpected = (
        model.load_state_dict(
            cleaned_state_dict,
            strict=False
        )
    )

    print()
    print(
        "VAE checkpoint compatibility"
    )

    print(
        "Missing keys   :",
        missing
    )

    print(
        "Unexpected keys:",
        unexpected
    )

    if missing or unexpected:

        raise RuntimeError(
            "VAE architecture does not "
            "match checkpoint."
        )

    model.eval()

    for parameter in model.parameters():

        parameter.requires_grad = False

    print(
        "VAE loaded successfully."
    )

    return model


# ============================================================
# LOAD FLOW
# ============================================================

def load_flow():

    print()
    print(
        "Loading trained RealNVP..."
    )

    checkpoint = torch.load(
        FLOW_CHECKPOINT,
        map_location=DEVICE
    )

    num_layers = checkpoint.get(
        "num_layers",
        8
    )

    hidden_dim = checkpoint.get(
        "hidden_dim",
        256
    )

    scale_clamp = checkpoint.get(
        "scale_clamp",
        1.5
    )

    latent_dim = checkpoint.get(
        "latent_dim",
        LATENT_DIM
    )

    if latent_dim != LATENT_DIM:

        raise RuntimeError(
            f"Flow latent dimension is "
            f"{latent_dim}, expected "
            f"{LATENT_DIM}."
        )

    flow = RealNVP(
        dim=latent_dim,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        scale_clamp=scale_clamp
    ).to(DEVICE)

    flow.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True
    )

    flow.eval()

    for parameter in flow.parameters():

        parameter.requires_grad = False

    print(
        "Flow loaded successfully."
    )

    print(
        f"Layers       : {num_layers}"
    )

    print(
        f"Hidden dim   : {hidden_dim}"
    )

    print(
        f"Scale clamp  : {scale_clamp}"
    )

    if "best_val_nll" in checkpoint:

        print(
            f"Best val NLL : "
            f"{checkpoint['best_val_nll']:.6f}"
        )

    return flow


# ============================================================
# EXTRACT LATENTS
# ============================================================

@torch.no_grad()
def extract_latents(
    vae,
    dataset,
    dataset_name
):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    all_mu = []

    all_paths = []

    print()
    print(
        f"Extracting VAE latents: "
        f"{dataset_name}"
    )

    for batch_idx, (
        images,
        paths
    ) in enumerate(loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        images = images.float()

        mu, _ = vae.encode(
            images
        )

        mu = mu.float()

        if not torch.isfinite(
            mu
        ).all():

            raise RuntimeError(
                f"Non-finite latent values "
                f"in {dataset_name}."
            )

        all_mu.append(
            mu.cpu()
        )

        all_paths.extend(
            list(paths)
        )

        if (
            batch_idx % 20 == 0
            or batch_idx == len(loader) - 1
        ):

            processed = min(
                (batch_idx + 1)
                * BATCH_SIZE,
                len(dataset)
            )

            print(
                f"  {processed:,}/"
                f"{len(dataset):,}"
            )

    latents = torch.cat(
        all_mu,
        dim=0
    )

    return latents, all_paths


# ============================================================
# SCORE LATENTS
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
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    scores = []

    for (x,) in loader:

        x = x.to(
            DEVICE
        ).float()

        log_prob = flow_log_prob(
            flow,
            x
        )

        nll = -log_prob

        if not torch.isfinite(
            nll
        ).all():

            raise RuntimeError(
                "Non-finite flow score detected."
            )

        scores.append(
            nll.cpu()
        )

    return torch.cat(
        scores
    )


# ============================================================
# LOAD STANDARDIZER
# ============================================================

def load_standardizer():

    print()
    print(
        "Loading train-set latent "
        "standardization..."
    )

    mean = torch.load(
        LATENT_MEAN_PATH,
        map_location="cpu"
    ).float()

    std = torch.load(
        LATENT_STD_PATH,
        map_location="cpu"
    ).float()

    if mean.shape != (
        LATENT_DIM,
    ):

        raise RuntimeError(
            f"Unexpected latent mean shape: "
            f"{mean.shape}"
        )

    if std.shape != (
        LATENT_DIM,
    ):

        raise RuntimeError(
            f"Unexpected latent std shape: "
            f"{std.shape}"
        )

    std = torch.clamp(
        std,
        min=1e-6
    )

    return mean, std


# ============================================================
# DATASET-WISE SUMMARY
# ============================================================

def score_summary(
    name,
    scores
):

    values = np.asarray(
        scores,
        dtype=np.float64
    )

    return {
        "dataset": name,
        "N": len(values),
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "p90": np.percentile(
            values,
            90
        ),
        "p95": np.percentile(
            values,
            95
        ),
        "p99": np.percentile(
            values,
            99
        ),
    }


# ============================================================
# CLASSIFICATION AT THRESHOLD
# ============================================================

def evaluate_threshold(
    y_true,
    scores,
    threshold
):

    predictions = (
        scores >= threshold
    ).astype(
        np.int32
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


# ============================================================
# BEST F1
# ============================================================

def find_best_f1(
    y_true,
    scores
):

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            scores
        )
    )

    if len(thresholds) == 0:

        return None

    f1_values = (
        2
        * precision[:-1]
        * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    best_idx = np.argmax(
        f1_values
    )

    threshold = thresholds[
        best_idx
    ]

    return evaluate_threshold(
        y_true,
        scores,
        threshold
    )


# ============================================================
# DATASET IDENTIFICATION
# ============================================================

def identify_dataset(
    path
):

    normalized = path.replace(
        "\\",
        "/"
    ).lower()

    if "ghostvision" in normalized:

        return "GhostVision"

    if "ai4shipwrecks" in normalized:

        return "AI4Shipwrecks"

    if "subpipemini2" in normalized:

        return "SubPipeMini2"

    if "marine-pulse" in normalized:

        return "Marine-PULSE"

    return "Unknown"


# ============================================================
# FIND ANTHROPOGENIC TEST IMAGES
# ============================================================

def load_manifest():

    if not os.path.exists(
        MANIFEST_PATH
    ):

        raise FileNotFoundError(
            f"Manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    return manifest


# ============================================================
# TEST IMAGE → LABEL INFORMATION
# ============================================================

def load_yolo_labels(
    label_path
):

    objects = []

    if not os.path.exists(
        label_path
    ):

        return objects

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

            cls = int(
                parts[0]
            )

            xc = float(
                parts[1]
            )

            yc = float(
                parts[2]
            )

            w = float(
                parts[3]
            )

            h = float(
                parts[4]
            )

            objects.append(
                (
                    cls,
                    xc,
                    yc,
                    w,
                    h
                )
            )

    return objects


# ============================================================
# BUILD TEST IMAGE LISTS
# ============================================================

def build_anthropogenic_test_list(
    manifest
):

    # --------------------------------------------------------
    # The manifest generated for SAAD_baseline contains:
    #
    #   dataset
    #   output_image
    #   output_label
    #   split
    #   saad_class
    #
    # We use the already validated test split.
    # --------------------------------------------------------

    required_columns = [
        "dataset",
        "output_image",
        "output_label",
        "split",
    ]

    missing = [
        col
        for col in required_columns
        if col not in manifest.columns
    ]

    if missing:

        raise RuntimeError(
            "Manifest is missing columns: "
            + str(missing)
        )

    test_manifest = manifest[
        manifest["split"].astype(str).str.lower()
        == "test"
    ].copy()

    rows = []

    for _, row in test_manifest.iterrows():

        dataset_name = str(
            row["dataset"]
        )

        # ----------------------------------------------------
        # Only these three datasets are being evaluated as
        # anthropogenic positives.
        # ----------------------------------------------------

        if dataset_name not in {
            "GhostVision",
            "AI4Shipwrecks",
            "SubPipeMini2",
        }:

            continue

        image_path = str(
            row["output_image"]
        )

        label_path = str(
            row["output_label"]
        )

        # Manifest may contain absolute paths or
        # paths relative to the SAAD root.
        if not os.path.isabs(
            image_path
        ):

            image_path = os.path.join(
                SAAD_ROOT,
                image_path
            )

        if not os.path.isabs(
            label_path
        ):

            label_path = os.path.join(
                SAAD_ROOT,
                label_path
            )

        if not os.path.exists(
            image_path
        ):

            continue

        rows.append(
            {
                "dataset": dataset_name,
                "image_path": image_path,
                "label_path": label_path,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PATCH-LEVEL APPROXIMATION FOR ANTHROPOGENIC IMAGES
# ============================================================
#
# Important:
#
# The VAE/flow were trained on 256×256 PATCHES.
#
# The SAAD baseline test images are not necessarily 256×256.
#
# Therefore we evaluate them using a sliding-window patch
# strategy and assign the maximum anomaly score in an image
# to that image.
#
# This is a deliberate image-level OOD/anomaly evaluation.
#
# ============================================================

def extract_sliding_patches(
    image_path,
    patch_size=256,
    stride=192
):

    image = Image.open(
        image_path
    ).convert("L")

    image_np = np.asarray(
        image,
        dtype=np.uint8
    )

    height, width = image_np.shape

    patches = []
    metadata = []

    # --------------------------------------------------------
    # Handle images smaller than 256.
    # --------------------------------------------------------

    if height < patch_size:

        pad_height = (
            patch_size - height
        )

    else:

        pad_height = 0

    if width < patch_size:

        pad_width = (
            patch_size - width
        )

    else:

        pad_width = 0

    if pad_height > 0 or pad_width > 0:

        image_np = np.pad(
            image_np,
            (
                (0, pad_height),
                (0, pad_width)
            ),
            mode="edge"
        )

        height, width = image_np.shape

    # --------------------------------------------------------
    # Generate start positions.
    # --------------------------------------------------------

    ys = list(
        range(
            0,
            max(
                1,
                height - patch_size + 1
            ),
            stride
        )
    )

    xs = list(
        range(
            0,
            max(
                1,
                width - patch_size + 1
            ),
            stride
        )
    )

    # Ensure final border is covered.
    if ys[-1] != height - patch_size:

        ys.append(
            height - patch_size
        )

    if xs[-1] != width - patch_size:

        xs.append(
            width - patch_size
        )

    for y in ys:

        for x in xs:

            patch = image_np[
                y:y + patch_size,
                x:x + patch_size
            ]

            patch = patch.astype(
                np.float32
            )

            patch /= 255.0

            tensor = torch.from_numpy(
                patch
            ).unsqueeze(0)

            patches.append(
                tensor
            )

            metadata.append(
                {
                    "x": x,
                    "y": y,
                    "width": patch_size,
                    "height": patch_size,
                }
            )

    return (
        torch.stack(patches),
        metadata
    )


# ============================================================
# SCORE ONE FULL IMAGE
# ============================================================

@torch.no_grad()
def score_image(
    image_path,
    vae,
    flow,
    latent_mean,
    latent_std
):

    patches, metadata = (
        extract_sliding_patches(
            image_path
        )
    )

    all_scores = []

    for start in range(
        0,
        len(patches),
        BATCH_SIZE
    ):

        batch = patches[
            start:start + BATCH_SIZE
        ]

        batch = batch.to(
            DEVICE
        ).float()

        mu, _ = vae.encode(
            batch
        )

        mu = (
            mu.cpu()
            - latent_mean
        ) / latent_std

        mu = mu.to(
            DEVICE
        ).float()

        nll = -flow_log_prob(
            flow,
            mu
        )

        all_scores.append(
            nll.cpu()
        )

    scores = torch.cat(
        all_scores
    ).numpy()

    max_idx = int(
        np.argmax(scores)
    )

    return {
        "image_score": float(
            scores[max_idx]
        ),

        "mean_score": float(
            np.mean(scores)
        ),

        "median_score": float(
            np.median(scores)
        ),

        "p95_score": float(
            np.percentile(
                scores,
                95
            )
        ),

        "num_patches": len(scores),

        "max_patch_index": max_idx,

        "patch_scores": scores,

        "patch_metadata": metadata,
    }


# ============================================================
# IMAGE-LEVEL ANTHROPOGENIC EVALUATION
# ============================================================

def evaluate_anthropogenic_images(
    manifest_df,
    vae,
    flow,
    latent_mean,
    latent_std
):

    rows = []

    print()
    print(
        "=" * 70
    )

    print(
        "EVALUATING ANTHROPOGENIC TEST IMAGES"
    )

    print(
        "=" * 70
    )

    for index, row in manifest_df.iterrows():

        image_path = row[
            "image_path"
        ]

        dataset_name = row[
            "dataset"
        ]

        result = score_image(
            image_path,
            vae,
            flow,
            latent_mean,
            latent_std
        )

        rows.append(
            {
                "dataset": dataset_name,

                "image_path": image_path,

                "label_path": row[
                    "label_path"
                ],

                "score": result[
                    "image_score"
                ],

                "mean_patch_score": result[
                    "mean_score"
                ],

                "median_patch_score": result[
                    "median_score"
                ],

                "p95_patch_score": result[
                    "p95_score"
                ],

                "num_patches": result[
                    "num_patches"
                ],

                "max_patch_index": result[
                    "max_patch_index"
                ],
            }
        )

        if (
            (index + 1) % 25 == 0
            or index == len(manifest_df) - 1
        ):

            print(
                f"  {index + 1}/"
                f"{len(manifest_df)}"
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PLOT SCORE DISTRIBUTIONS
# ============================================================

def plot_distributions(
    normal_scores,
    anthropogenic_scores,
    hard_negative_scores
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        normal_scores,
        bins=60,
        alpha=0.6,
        density=True,
        label="Normal seabed"
    )

    plt.hist(
        anthropogenic_scores,
        bins=60,
        alpha=0.6,
        density=True,
        label="Anthropogenic"
    )

    if len(hard_negative_scores) > 0:

        plt.hist(
            hard_negative_scores,
            bins=60,
            alpha=0.6,
            density=True,
            label="Hard negative"
        )

    plt.xlabel(
        "Latent Flow NLL"
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        "SAAD Latent Flow Score Distribution"
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "flow_score_distribution.png"
    )

    plt.savefig(
        path,
        dpi=160
    )

    plt.close()

    return path


# ============================================================
# ROC / PR CURVES
# ============================================================

def plot_curves(
    y_true,
    scores
):

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_true,
        scores
    )

    auc = roc_auc_score(
        y_true,
        scores
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "SAAD Latent Flow ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    roc_path = os.path.join(
        OUTPUT_DIR,
        "roc_curve.png"
    )

    plt.savefig(
        roc_path,
        dpi=160
    )

    plt.close()

    # --------------------------------------------------------
    # Precision-recall
    # --------------------------------------------------------

    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            scores
        )
    )

    pr_auc = average_precision_score(
        y_true,
        scores
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        recall,
        precision,
        label=f"PR-AUC = {pr_auc:.4f}"
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "SAAD Latent Flow Precision-Recall Curve"
    )

    plt.legend()

    plt.tight_layout()

    pr_path = os.path.join(
        OUTPUT_DIR,
        "pr_curve.png"
    )

    plt.savefig(
        pr_path,
        dpi=160
    )

    plt.close()

    return roc_path, pr_path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 70
    )

    print(
        "SAAD — LATENT NORMALIZING FLOW EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()
    print(
        "Flow checkpoint:"
    )

    print(
        FLOW_CHECKPOINT
    )

    print()
    print(
        "Output:"
    )

    print(
        OUTPUT_DIR
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    vae = load_vae()

    flow = load_flow()

    latent_mean, latent_std = (
        load_standardizer()
    )

    # --------------------------------------------------------
    # Normal validation
    # --------------------------------------------------------

    normal_val_dataset = (
        ImageFolderDataset(
            NORMAL_VAL_DIR
        )
    )

    normal_test_dataset = (
        ImageFolderDataset(
            NORMAL_TEST_DIR
        )
    )

    normal_val_mu, normal_val_paths = (
        extract_latents(
            vae,
            normal_val_dataset,
            "NORMAL VALIDATION"
        )
    )

    normal_test_mu, normal_test_paths = (
        extract_latents(
            vae,
            normal_test_dataset,
            "NORMAL TEST"
        )
    )

    normal_val_z = (
        normal_val_mu
        - latent_mean
    ) / latent_std

    normal_test_z = (
        normal_test_mu
        - latent_mean
    ) / latent_std

    normal_val_scores = (
        score_latents(
            flow,
            normal_val_z
        ).numpy()
    )

    normal_test_scores = (
        score_latents(
            flow,
            normal_test_z
        ).numpy()
    )

    # --------------------------------------------------------
    # Normal threshold candidates
    # --------------------------------------------------------

    val_p95 = np.percentile(
        normal_val_scores,
        95
    )

    val_p99 = np.percentile(
        normal_val_scores,
        99
    )

    # --------------------------------------------------------
    # Load SAAD manifest
    # --------------------------------------------------------

    print()
    print(
        "Loading SAAD manifest..."
    )

    manifest = load_manifest()

    anthropogenic_manifest = (
        build_anthropogenic_test_list(
            manifest
        )
    )

    print(
        f"Anthropogenic test images: "
        f"{len(anthropogenic_manifest)}"
    )

    # --------------------------------------------------------
    # Evaluate anthropogenic images
    # --------------------------------------------------------

    anthropogenic_results = (
        evaluate_anthropogenic_images(
            anthropogenic_manifest,
            vae,
            flow,
            latent_mean,
            latent_std
        )
    )

    # --------------------------------------------------------
    # Dataset-wise anthropogenic scores
    # --------------------------------------------------------

    ghost_scores = (
        anthropogenic_results[
            anthropogenic_results[
                "dataset"
            ].str.lower()
            == "ghostvision"
        ]["score"]
        .to_numpy()
    )

    shipwreck_scores = (
        anthropogenic_results[
            anthropogenic_results[
                "dataset"
            ].str.lower()
            == "ai4shipwrecks"
        ]["score"]
        .to_numpy()
    )

    pipe_scores = (
        anthropogenic_results[
            anthropogenic_results[
                "dataset"
            ].str.lower()
            == "subpipemini2"
        ]["score"]
        .to_numpy()
    )

    anthropogenic_scores = (
        anthropogenic_results[
            "score"
        ].to_numpy()
    )

    # --------------------------------------------------------
    # Build binary evaluation
    #
    # 0 = normal
    # 1 = anthropogenic
    #
    # Normal TEST is used here because it is the held-out
    # normal set.
    # --------------------------------------------------------

    y_true = np.concatenate(
        [
            np.zeros(
                len(normal_test_scores),
                dtype=np.int32
            ),

            np.ones(
                len(anthropogenic_scores),
                dtype=np.int32
            ),
        ]
    )

    all_scores = np.concatenate(
        [
            normal_test_scores,
            anthropogenic_scores,
        ]
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        y_true,
        all_scores
    )

    # --------------------------------------------------------
    # PR-AUC
    # --------------------------------------------------------

    pr_auc = average_precision_score(
        y_true,
        all_scores
    )

    # --------------------------------------------------------
    # Best F1
    # --------------------------------------------------------

    best_f1 = find_best_f1(
        y_true,
        all_scores
    )

    # --------------------------------------------------------
    # Threshold using NORMAL VALIDATION P95/P99
    # --------------------------------------------------------

    p95_metrics = evaluate_threshold(
        y_true,
        all_scores,
        val_p95
    )

    p99_metrics = evaluate_threshold(
        y_true,
        all_scores,
        val_p99
    )

    # --------------------------------------------------------
    # Hard-negative placeholder
    #
    # The baseline hard-negative images are not automatically
    # safe to interpret as complete 256x256 patches here.
    #
    # We therefore keep the evaluator architecture ready but
    # do not fabricate a hard-negative score set.
    # --------------------------------------------------------

    hard_negative_scores = np.array(
        [],
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Print normal statistics
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "NORMAL FLOW SCORES"
    )

    print(
        "=" * 70
    )

    for name, scores in [
        (
            "NORMAL VALIDATION",
            normal_val_scores
        ),
        (
            "NORMAL TEST",
            normal_test_scores
        ),
    ]:

        stats = score_summary(
            name,
            scores
        )

        print()
        print(name)

        print(
            f"N       : {stats['N']:,}"
        )

        print(
            f"Mean    : {stats['mean']:.6f}"
        )

        print(
            f"Median  : {stats['median']:.6f}"
        )

        print(
            f"Std     : {stats['std']:.6f}"
        )

        print(
            f"Min     : {stats['min']:.6f}"
        )

        print(
            f"Max     : {stats['max']:.6f}"
        )

        print(
            f"P90     : {stats['p90']:.6f}"
        )

        print(
            f"P95     : {stats['p95']:.6f}"
        )

        print(
            f"P99     : {stats['p99']:.6f}"
        )

    # --------------------------------------------------------
    # Anthropogenic statistics
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "ANTHROPOGENIC FLOW SCORES"
    )

    print(
        "=" * 70
    )

    for name, scores in [
        (
            "ALL ANTHROPOGENIC",
            anthropogenic_scores
        ),
        (
            "GhostVision",
            ghost_scores
        ),
        (
            "AI4Shipwrecks",
            shipwreck_scores
        ),
        (
            "SubPipeMini2",
            pipe_scores
        ),
    ]:

        if len(scores) == 0:

            continue

        stats = score_summary(
            name,
            scores
        )

        print()
        print(name)

        print(
            f"N       : {stats['N']:,}"
        )

        print(
            f"Mean    : {stats['mean']:.6f}"
        )

        print(
            f"Median  : {stats['median']:.6f}"
        )

        print(
            f"Std     : {stats['std']:.6f}"
        )

        print(
            f"Min     : {stats['min']:.6f}"
        )

        print(
            f"Max     : {stats['max']:.6f}"
        )

        print(
            f"P95     : {stats['p95']:.6f}"
        )

        print(
            f"P99     : {stats['p99']:.6f}"
        )

    # --------------------------------------------------------
    # Detection metrics
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "FLOW ANOMALY DETECTION"
    )

    print(
        "=" * 70
    )

    print()
    print(
        f"ROC-AUC : {roc_auc:.6f}"
    )

    print(
        f"PR-AUC  : {pr_auc:.6f}"
    )

    # --------------------------------------------------------
    # Best F1
    # --------------------------------------------------------

    print()
    print(
        "BEST-F1 THRESHOLD"
    )

    print(
        "-" * 50
    )

    print(
        f"Threshold : "
        f"{best_f1['threshold']:.6f}"
    )

    print(
        f"Precision : "
        f"{best_f1['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_f1['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{best_f1['f1']:.4f}"
    )

    print(
        f"TP={best_f1['TP']} "
        f"FP={best_f1['FP']} "
        f"FN={best_f1['FN']} "
        f"TN={best_f1['TN']}"
    )

    # --------------------------------------------------------
    # Validation P95
    # --------------------------------------------------------

    print()
    print(
        "NORMAL VALIDATION P95 THRESHOLD"
    )

    print(
        "-" * 50
    )

    print(
        f"Threshold : "
        f"{p95_metrics['threshold']:.6f}"
    )

    print(
        f"Precision : "
        f"{p95_metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{p95_metrics['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{p95_metrics['f1']:.4f}"
    )

    print(
        f"TP={p95_metrics['TP']} "
        f"FP={p95_metrics['FP']} "
        f"FN={p95_metrics['FN']} "
        f"TN={p95_metrics['TN']}"
    )

    # --------------------------------------------------------
    # Validation P99
    # --------------------------------------------------------

    print()
    print(
        "NORMAL VALIDATION P99 THRESHOLD"
    )

    print(
        "-" * 50
    )

    print(
        f"Threshold : "
        f"{p99_metrics['threshold']:.6f}"
    )

    print(
        f"Precision : "
        f"{p99_metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{p99_metrics['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{p99_metrics['f1']:.4f}"
    )

    print(
        f"TP={p99_metrics['TP']} "
        f"FP={p99_metrics['FP']} "
        f"FN={p99_metrics['FN']} "
        f"TN={p99_metrics['TN']}"
    )

    # --------------------------------------------------------
    # Dataset-wise comparison
    # --------------------------------------------------------

    dataset_rows = []

    for name, scores in [
        (
            "GhostVision",
            ghost_scores
        ),
        (
            "AI4Shipwrecks",
            shipwreck_scores
        ),
        (
            "SubPipeMini2",
            pipe_scores
        ),
    ]:

        if len(scores) == 0:

            continue

        dataset_rows.append(
            score_summary(
                name,
                scores
            )
        )

    dataset_rows.append(
        score_summary(
            "Normal Test",
            normal_test_scores
        )
    )

    dataset_results = pd.DataFrame(
        dataset_rows
    )

    dataset_results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "dataset_results.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Save image-level scores
    # --------------------------------------------------------

    anthropogenic_results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "anthropogenic_image_scores.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Save normal scores
    # --------------------------------------------------------

    normal_score_rows = []

    for path, score in zip(
        normal_test_paths,
        normal_test_scores
    ):

        normal_score_rows.append(
            {
                "dataset":
                    "NormalTest",

                "image_path":
                    path,

                "score":
                    float(score),

                "label":
                    0,
            }
        )

    for path, score in zip(
        normal_val_paths,
        normal_val_scores
    ):

        normal_score_rows.append(
            {
                "dataset":
                    "NormalValidation",

                "image_path":
                    path,

                "score":
                    float(score),

                "label":
                    0,
            }
        )

    normal_scores_df = pd.DataFrame(
        normal_score_rows
    )

    normal_scores_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "normal_scores.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Save combined classification scores
    # --------------------------------------------------------

    combined_rows = []

    for score in normal_test_scores:

        combined_rows.append(
            {
                "label": 0,
                "score": float(score),
                "group": "NormalTest",
            }
        )

    for _, row in anthropogenic_results.iterrows():

        combined_rows.append(
            {
                "label": 1,
                "score": float(
                    row["score"]
                ),
                "group": row[
                    "dataset"
                ],
                "image_path": row[
                    "image_path"
                ],
            }
        )

    combined_df = pd.DataFrame(
        combined_rows
    )

    combined_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "combined_scores.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    distribution_path = (
        plot_distributions(
            normal_test_scores,
            anthropogenic_scores,
            hard_negative_scores
        )
    )

    roc_path, pr_path = (
        plot_curves(
            y_true,
            all_scores
        )
    )

    # --------------------------------------------------------
    # Summary file
    # --------------------------------------------------------

    summary_path = os.path.join(
        OUTPUT_DIR,
        "evaluation_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SAAD — LATENT NORMALIZING FLOW "
            "EVALUATION\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            f"VAE checkpoint:\n"
            f"{VAE_CHECKPOINT}\n\n"
        )

        f.write(
            f"Flow checkpoint:\n"
            f"{FLOW_CHECKPOINT}\n\n"
        )

        f.write(
            f"Normal validation N: "
            f"{len(normal_val_scores)}\n"
        )

        f.write(
            f"Normal test N: "
            f"{len(normal_test_scores)}\n"
        )

        f.write(
            f"Anthropogenic image N: "
            f"{len(anthropogenic_scores)}\n\n"
        )

        f.write(
            "NORMAL VALIDATION\n"
        )

        f.write(
            f"Median: "
            f"{np.median(normal_val_scores):.8f}\n"
        )

        f.write(
            f"P95: "
            f"{val_p95:.8f}\n"
        )

        f.write(
            f"P99: "
            f"{val_p99:.8f}\n\n"
        )

        f.write(
            "NORMAL TEST\n"
        )

        f.write(
            f"Mean: "
            f"{np.mean(normal_test_scores):.8f}\n"
        )

        f.write(
            f"Median: "
            f"{np.median(normal_test_scores):.8f}\n"
        )

        f.write(
            f"P95: "
            f"{np.percentile(normal_test_scores, 95):.8f}\n"
        )

        f.write(
            f"P99: "
            f"{np.percentile(normal_test_scores, 99):.8f}\n\n"
        )

        f.write(
            "ANTHROPOGENIC\n"
        )

        f.write(
            f"Mean: "
            f"{np.mean(anthropogenic_scores):.8f}\n"
        )

        f.write(
            f"Median: "
            f"{np.median(anthropogenic_scores):.8f}\n"
        )

        f.write(
            f"P95: "
            f"{np.percentile(anthropogenic_scores, 95):.8f}\n"
        )

        f.write(
            f"P99: "
            f"{np.percentile(anthropogenic_scores, 99):.8f}\n\n"
        )

        f.write(
            "OVERALL DETECTION\n"
        )

        f.write(
            f"ROC-AUC: "
            f"{roc_auc:.8f}\n"
        )

        f.write(
            f"PR-AUC: "
            f"{pr_auc:.8f}\n\n"
        )

        f.write(
            "BEST F1\n"
        )

        for key, value in best_f1.items():

            f.write(
                f"{key}: {value}\n"
            )

        f.write(
            "\nNORMAL VALIDATION P95\n"
        )

        for key, value in p95_metrics.items():

            f.write(
                f"{key}: {value}\n"
            )

        f.write(
            "\nNORMAL VALIDATION P99\n"
        )

        for key, value in p99_metrics.items():

            f.write(
                f"{key}: {value}\n"
            )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "FLOW EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print()
    print(
        f"ROC-AUC : {roc_auc:.6f}"
    )

    print(
        f"PR-AUC  : {pr_auc:.6f}"
    )

    print()
    print(
        "Comparison target from VAE v1:"
    )

    print(
        "  VAE ROC-AUC = 0.718251"
    )

    print(
        "  VAE PR-AUC  = 0.492144"
    )

    print()
    print(
        "Files saved:"
    )

    print(
        f"  {summary_path}"
    )

    print(
        f"  {os.path.join(OUTPUT_DIR, 'dataset_results.csv')}"
    )

    print(
        f"  {os.path.join(OUTPUT_DIR, 'normal_scores.csv')}"
    )

    print(
        f"  {os.path.join(OUTPUT_DIR, 'anthropogenic_image_scores.csv')}"
    )

    print(
        f"  {os.path.join(OUTPUT_DIR, 'combined_scores.csv')}"
    )

    print(
        f"  {distribution_path}"
    )

    print(
        f"  {roc_path}"
    )

    print(
        f"  {pr_path}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The normal-validation P95/P99 thresholds "
        "are the deployment-oriented thresholds."
    )

    print(
        "The best-F1 threshold is an analytical "
        "diagnostic and should NOT be presented "
        "as an unbiased deployment threshold."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    seed_everything(SEED)

    main()