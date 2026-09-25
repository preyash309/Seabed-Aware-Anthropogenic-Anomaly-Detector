# ============================================================
# SAAD — EVALUATE SCORE FUSION
# ============================================================
#
# Seabed-Aware Anthropogenic Anomaly Detector
#
# Evaluates:
#
#   1. YOLO alone
#   2. VAE reconstruction anomaly alone
#   3. VAE + latent RealNVP flow
#   4. YOLO + Flow
#   5. YOLO + VAE
#   6. YOLO + VAE + Flow
#
# IMPORTANT:
#
# This script does NOT train any model.
#
# Frozen components:
#
#   YOLO26s
#   VAE v1
#   RealNVP
#
# The goal is to determine whether the three independent
# signals provide complementary information.
#
# ============================================================

import os
import random
import math
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from PIL import Image

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
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
# YOLO
# ------------------------------------------------------------

YOLO_CHECKPOINT = (
    r"E:\SIH\SIH_Results"
    r"\yolo26s_generic_baseline"
    r"\weights\best.pt"
)

# ------------------------------------------------------------
# VAE
# ------------------------------------------------------------

VAE_CHECKPOINT = (
    r"E:\SIH\SIH_Results"
    r"\vae_normal_seabed"
    r"\checkpoints\best.pt"
)

# ------------------------------------------------------------
# FLOW
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
# VAE normal data
# ------------------------------------------------------------

VAE_DATA_ROOT = (
    r"E:\SIH\Datasets\SAAD_VAE"
)

NORMAL_TRAIN_DIR = os.path.join(
    VAE_DATA_ROOT,
    "train"
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
# SAAD baseline
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

OUTPUT_DIR = (
    r"E:\SIH\SIH_Results"
    r"\saad_fusion_evaluation"
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
# IMAGE DATASET
# ============================================================

class ImageDataset(Dataset):

    def __init__(self, root):

        self.files = []

        extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        }

        for dirpath, _, filenames in os.walk(root):

            for filename in filenames:

                ext = os.path.splitext(
                    filename
                )[1].lower()

                if ext in extensions:

                    self.files.append(
                        os.path.join(
                            dirpath,
                            filename
                        )
                    )

        self.files.sort()

        if len(self.files) == 0:

            raise RuntimeError(
                f"No images found:\n{root}"
            )

        print(
            f"Loaded {len(self.files):,} images:"
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

        tensor = torch.from_numpy(
            image
        ).unsqueeze(0)

        return tensor, path


# ============================================================
# VAE V1
# ============================================================

class ConvVAE(nn.Module):

    def __init__(
        self,
        latent_dim=128
    ):

        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1, 32,
                4, 2, 1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32, 64,
                4, 2, 1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64, 128,
                4, 2, 1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128, 256,
                4, 2, 1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256, 512,
                4, 2, 1
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

        # EXACT VAE V1 checkpoint name.
        self.fc_decode = nn.Linear(
            latent_dim,
            512 * 8 * 8
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                512, 256,
                4, 2, 1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                256, 128,
                4, 2, 1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128, 64,
                4, 2, 1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64, 32,
                4, 2, 1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32, 1,
                4, 2, 1
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

        recon = self.decode(z)

        return recon, mu, logvar


# ============================================================
# REALNVP COUPLING
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

        condition = x[
            :,
            self.condition_indices
        ]

        st = self.net(
            condition
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


# ============================================================
# FLOW LOG PROBABILITY
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


def flow_log_prob(
    flow,
    x
):

    z, log_det = flow(x)

    return (
        standard_normal_log_prob(z)
        + log_det
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def get_state_dict(checkpoint):

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            return checkpoint[
                "model_state_dict"
            ]

        if "state_dict" in checkpoint:

            return checkpoint[
                "state_dict"
            ]

    return checkpoint


# ============================================================
# LOAD VAE
# ============================================================

def load_vae():

    print()
    print(
        "Loading frozen VAE..."
    )

    vae = ConvVAE(
        LATENT_DIM
    ).to(DEVICE)

    checkpoint = torch.load(
        VAE_CHECKPOINT,
        map_location=DEVICE
    )

    state_dict = get_state_dict(
        checkpoint
    )

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned[key] = value

    missing, unexpected = (
        vae.load_state_dict(
            cleaned,
            strict=False
        )
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
            "VAE checkpoint mismatch."
        )

    vae.eval()

    for p in vae.parameters():

        p.requires_grad = False

    return vae


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

    flow = RealNVP(
        dim=checkpoint.get(
            "latent_dim",
            LATENT_DIM
        ),
        num_layers=checkpoint.get(
            "num_layers",
            8
        ),
        hidden_dim=checkpoint.get(
            "hidden_dim",
            256
        ),
        scale_clamp=checkpoint.get(
            "scale_clamp",
            1.5
        )
    ).to(DEVICE)

    flow.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True
    )

    flow.eval()

    for p in flow.parameters():

        p.requires_grad = False

    print(
        "Flow loaded successfully."
    )

    return flow


# ============================================================
# YOLO LOADING
# ============================================================

def load_yolo():

    print()
    print(
        "Loading frozen YOLO26s..."
    )

    try:

        from ultralytics import YOLO

    except ImportError:

        raise RuntimeError(
            "Ultralytics is not installed."
        )

    model = YOLO(
        YOLO_CHECKPOINT
    )

    print(
        "YOLO loaded successfully."
    )

    return model


# ============================================================
# LOAD LATENT STANDARDIZATION
# ============================================================

def load_standardizer():

    mean = torch.load(
        LATENT_MEAN_PATH,
        map_location="cpu"
    ).float()

    std = torch.load(
        LATENT_STD_PATH,
        map_location="cpu"
    ).float()

    std = torch.clamp(
        std,
        min=1e-6
    )

    return mean, std


# ============================================================
# VAE RECONSTRUCTION SCORE
# ============================================================

@torch.no_grad()
def vae_score_batch(
    vae,
    images
):

    mu, _ = vae.encode(
        images
    )

    reconstruction = vae.decode(
        mu
    )

    # Mean squared reconstruction error.
    mse = (
        reconstruction
        - images
    ).pow(2)

    mse = mse.mean(
        dim=(1, 2, 3)
    )

    return mse


# ============================================================
# FLOW SCORE
# ============================================================

@torch.no_grad()
def flow_score_batch(
    vae,
    flow,
    images,
    latent_mean,
    latent_std
):

    mu, _ = vae.encode(
        images
    )

    mu = mu.float()

    mu = (
        mu.cpu()
        - latent_mean
    ) / latent_std

    mu = mu.to(
        DEVICE
    )

    log_prob = flow_log_prob(
        flow,
        mu
    )

    return -log_prob


# ============================================================
# SLIDING PATCH EXTRACTION
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

    if height < patch_size:

        image_np = np.pad(
            image_np,
            (
                (
                    0,
                    patch_size - height
                ),
                (0, 0)
            ),
            mode="edge"
        )

    if width < patch_size:

        image_np = np.pad(
            image_np,
            (
                (0, 0),
                (
                    0,
                    patch_size - width
                )
            ),
            mode="edge"
        )

    height, width = image_np.shape

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

    final_y = (
        height - patch_size
    )

    final_x = (
        width - patch_size
    )

    if ys[-1] != final_y:

        ys.append(
            final_y
        )

    if xs[-1] != final_x:

        xs.append(
            final_x
        )

    patches = []

    locations = []

    for y in ys:

        for x in xs:

            patch = image_np[
                y:y + patch_size,
                x:x + patch_size
            ]

            patch = (
                patch.astype(
                    np.float32
                )
                / 255.0
            )

            tensor = torch.from_numpy(
                patch
            ).unsqueeze(0)

            patches.append(
                tensor
            )

            locations.append(
                (
                    x,
                    y
                )
            )

    return (
        torch.stack(patches),
        locations
    )


# ============================================================
# ANTHROPOGENIC TEST IMAGES
# ============================================================

def load_anthropogenic_images():

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    required = {
        "dataset",
        "output_image",
        "output_label",
        "split",
    }

    missing = required - set(
        manifest.columns
    )

    if missing:

        raise RuntimeError(
            f"Manifest missing columns: "
            f"{missing}"
        )

    test = manifest[
        manifest[
            "split"
        ].astype(str).str.lower()
        == "test"
    ].copy()

    allowed = {
        "GhostVision",
        "AI4Shipwrecks",
        "SubPipeMini2",
    }

    rows = []

    for _, row in test.iterrows():

        dataset = str(
            row["dataset"]
        )

        if dataset not in allowed:

            continue

        image_path = str(
            row["output_image"]
        )

        label_path = str(
            row["output_label"]
        )

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
                "dataset": dataset,
                "image_path": image_path,
                "label_path": label_path,
            }
        )

    result = pd.DataFrame(
        rows
    )

    print()
    print(
        f"Anthropogenic test images: "
        f"{len(result)}"
    )

    return result


# ============================================================
# YOLO IMAGE SCORE
# ============================================================
#
# Image-level YOLO score:
#
#   maximum confidence among predictions
#
# This is intentionally simple for the first fusion test.
#
# ============================================================

def calculate_yolo_scores(
    yolo,
    image_paths
):

    scores = []

    print()
    print(
        "Calculating YOLO image scores..."
    )

    for i, path in enumerate(
        image_paths
    ):

        results = yolo.predict(
            source=path,
            conf=0.001,
            iou=0.5,
            verbose=False,
            device=0 if torch.cuda.is_available()
            else "cpu",
        )

        result = results[0]

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):

            score = 0.0

        else:

            confidence = (
                result.boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

            score = float(
                np.max(confidence)
            )

        scores.append(
            score
        )

        if (
            (i + 1) % 25 == 0
            or i == len(image_paths) - 1
        ):

            print(
                f"  {i + 1}/"
                f"{len(image_paths)}"
            )

    return np.asarray(
        scores,
        dtype=np.float64
    )


# ============================================================
# ANOMALY IMAGE SCORE
# ============================================================

@torch.no_grad()
def calculate_anomaly_scores(
    vae,
    flow,
    latent_mean,
    latent_std,
    image_paths
):

    vae_scores = []
    flow_scores = []

    print()
    print(
        "Calculating VAE + Flow image scores..."
    )

    for i, path in enumerate(
        image_paths
    ):

        patches, _ = (
            extract_sliding_patches(
                path
            )
        )

        patch_vae_scores = []
        patch_flow_scores = []

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

            mse = vae_score_batch(
                vae,
                batch
            )

            nll = flow_score_batch(
                vae,
                flow,
                batch,
                latent_mean,
                latent_std
            )

            patch_vae_scores.append(
                mse.cpu()
            )

            patch_flow_scores.append(
                nll.cpu()
            )

        patch_vae_scores = torch.cat(
            patch_vae_scores
        ).numpy()

        patch_flow_scores = torch.cat(
            patch_flow_scores
        ).numpy()

        # Maximum patch anomaly.
        vae_scores.append(
            float(
                np.max(
                    patch_vae_scores
                )
            )
        )

        flow_scores.append(
            float(
                np.max(
                    patch_flow_scores
                )
            )
        )

        if (
            (i + 1) % 25 == 0
            or i == len(image_paths) - 1
        ):

            print(
                f"  {i + 1}/"
                f"{len(image_paths)}"
            )

    return (
        np.asarray(
            vae_scores,
            dtype=np.float64
        ),

        np.asarray(
            flow_scores,
            dtype=np.float64
        )
    )


# ============================================================
# NORMAL IMAGE SCORES
# ============================================================

@torch.no_grad()
def calculate_normal_anomaly_scores(
    vae,
    flow,
    latent_mean,
    latent_std,
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

    vae_scores = []
    flow_scores = []

    print()
    print(
        f"Calculating anomaly scores: "
        f"{dataset_name}"
    )

    for batch_idx, (
        images,
        paths
    ) in enumerate(loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        ).float()

        mse = vae_score_batch(
            vae,
            images
        )

        nll = flow_score_batch(
            vae,
            flow,
            images,
            latent_mean,
            latent_std
        )

        vae_scores.append(
            mse.cpu()
        )

        flow_scores.append(
            nll.cpu()
        )

    return (
        torch.cat(
            vae_scores
        ).numpy(),

        torch.cat(
            flow_scores
        ).numpy()
    )


# ============================================================
# NORMALIZATION
# ============================================================
#
# Robust percentile normalization.
#
# We use NORMAL VALIDATION statistics only.
#
# Score below P5:
#     0
#
# Score above P95:
#     1
#
# This avoids one extreme normal sample dominating the scale.
#
# ============================================================

class PercentileNormalizer:

    def __init__(
        self,
        low,
        high
    ):

        self.low = float(low)
        self.high = float(high)

        if self.high <= self.low:

            self.high = (
                self.low + 1e-6
            )

    def transform(
        self,
        scores
    ):

        scores = np.asarray(
            scores,
            dtype=np.float64
        )

        normalized = (
            scores - self.low
        ) / (
            self.high - self.low
        )

        return np.clip(
            normalized,
            0.0,
            1.0
        )


# ============================================================
# METRICS
# ============================================================

def best_f1(
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

        return {
            "threshold": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "TN": 0,
        }

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

    idx = int(
        np.argmax(
            f1_values
        )
    )

    threshold = float(
        thresholds[idx]
    )

    prediction = (
        scores >= threshold
    ).astype(
        np.int32
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            prediction,
            labels=[0, 1]
        ).ravel()
    )

    return {
        "threshold": threshold,

        "precision": precision_score(
            y_true,
            prediction,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            prediction,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            prediction,
            zero_division=0
        ),

        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


def threshold_metrics(
    y_true,
    scores,
    threshold
):

    prediction = (
        scores >= threshold
    ).astype(
        np.int32
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            prediction,
            labels=[0, 1]
        ).ravel()
    )

    return {
        "threshold": threshold,

        "precision": precision_score(
            y_true,
            prediction,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            prediction,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            prediction,
            zero_division=0
        ),

        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


# ============================================================
# EVALUATE ONE SIGNAL
# ============================================================

def evaluate_signal(
    name,
    y_true,
    scores
):

    roc = roc_auc_score(
        y_true,
        scores
    )

    pr = average_precision_score(
        y_true,
        scores
    )

    f1 = best_f1(
        y_true,
        scores
    )

    return {
        "signal": name,

        "ROC_AUC": roc,

        "PR_AUC": pr,

        "Best_F1": f1["f1"],

        "Best_F1_threshold": f1[
            "threshold"
        ],

        "Best_F1_precision": f1[
            "precision"
        ],

        "Best_F1_recall": f1[
            "recall"
        ],
    }


# ============================================================
# PLOT COMPARISON
# ============================================================

def plot_metric_comparison(
    results
):

    df = pd.DataFrame(
        results
    )

    x = np.arange(
        len(df)
    )

    width = 0.35

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        x - width / 2,
        df["ROC_AUC"],
        width,
        label="ROC-AUC"
    )

    plt.bar(
        x + width / 2,
        df["PR_AUC"],
        width,
        label="PR-AUC"
    )

    plt.xticks(
        x,
        df["signal"],
        rotation=30,
        ha="right"
    )

    plt.ylim(
        0,
        1
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "SAAD Signal Fusion Comparison"
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "fusion_metric_comparison.png"
    )

    plt.savefig(
        path,
        dpi=160
    )

    plt.close()

    return path


# ============================================================
# ROC COMPARISON
# ============================================================

def plot_roc_comparison(
    y_true,
    signal_dict
):

    plt.figure(
        figsize=(8, 7)
    )

    for name, scores in signal_dict.items():

        fpr, tpr, _ = roc_curve(
            y_true,
            scores
        )

        auc = roc_auc_score(
            y_true,
            scores
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{name} ({auc:.3f})"
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
        "SAAD ROC Comparison"
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "fusion_roc_comparison.png"
    )

    plt.savefig(
        path,
        dpi=160
    )

    plt.close()

    return path


# ============================================================
# PR COMPARISON
# ============================================================

def plot_pr_comparison(
    y_true,
    signal_dict
):

    plt.figure(
        figsize=(8, 7)
    )

    for name, scores in signal_dict.items():

        precision, recall, _ = (
            precision_recall_curve(
                y_true,
                scores
            )
        )

        auc = average_precision_score(
            y_true,
            scores
        )

        plt.plot(
            recall,
            precision,
            label=f"{name} ({auc:.3f})"
        )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "SAAD Precision-Recall Comparison"
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "fusion_pr_comparison.png"
    )

    plt.savefig(
        path,
        dpi=160
    )

    plt.close()

    return path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 70
    )

    print(
        "SAAD — SCORE FUSION EVALUATION"
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

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    vae = load_vae()

    flow = load_flow()

    yolo = load_yolo()

    latent_mean, latent_std = (
        load_standardizer()
    )

    # --------------------------------------------------------
    # Load normal datasets
    # --------------------------------------------------------

    train_dataset = ImageDataset(
        NORMAL_TRAIN_DIR
    )

    val_dataset = ImageDataset(
        NORMAL_VAL_DIR
    )

    test_dataset = ImageDataset(
        NORMAL_TEST_DIR
    )

    # --------------------------------------------------------
    # NORMAL VALIDATION
    #
    # Used ONLY to establish score normalization.
    # --------------------------------------------------------

    val_vae, val_flow = (
        calculate_normal_anomaly_scores(
            vae,
            flow,
            latent_mean,
            latent_std,
            val_dataset,
            "NORMAL VALIDATION"
        )
    )

    # --------------------------------------------------------
    # Fit normalizers on VALIDATION ONLY
    # --------------------------------------------------------

    vae_normalizer = PercentileNormalizer(
        np.percentile(
            val_vae,
            5
        ),
        np.percentile(
            val_vae,
            95
        )
    )

    flow_normalizer = PercentileNormalizer(
        np.percentile(
            val_flow,
            5
        ),
        np.percentile(
            val_flow,
            95
        )
    )

    # --------------------------------------------------------
    # NORMAL TEST
    # --------------------------------------------------------

    test_vae, test_flow = (
        calculate_normal_anomaly_scores(
            vae,
            flow,
            latent_mean,
            latent_std,
            test_dataset,
            "NORMAL TEST"
        )
    )

    # --------------------------------------------------------
    # Anthropogenic test images
    # --------------------------------------------------------

    anthropogenic = (
        load_anthropogenic_images()
    )

    anthropogenic_paths = (
        anthropogenic[
            "image_path"
        ].tolist()
    )

    anthropogenic_dataset_names = (
        anthropogenic[
            "dataset"
        ].tolist()
    )

    # --------------------------------------------------------
    # YOLO scores
    # --------------------------------------------------------

    anth_yolo = calculate_yolo_scores(
        yolo,
        anthropogenic_paths
    )

    # --------------------------------------------------------
    # VAE + flow image-level scores
    # --------------------------------------------------------

    anth_vae, anth_flow = (
        calculate_anomaly_scores(
            vae,
            flow,
            latent_mean,
            latent_std,
            anthropogenic_paths
        )
    )

    # --------------------------------------------------------
    # NORMAL YOLO scores
    #
    # Required for fair normalization of the YOLO branch.
    # --------------------------------------------------------

    normal_test_paths = (
        test_dataset.files
    )

    normal_val_paths = (
        val_dataset.files
    )

    normal_val_yolo = (
        calculate_yolo_scores(
            yolo,
            normal_val_paths
        )
    )

    normal_test_yolo = (
        calculate_yolo_scores(
            yolo,
            normal_test_paths
        )
    )

    yolo_normalizer = PercentileNormalizer(
        np.percentile(
            normal_val_yolo,
            5
        ),
        np.percentile(
            normal_val_yolo,
            95
        )
    )

    # --------------------------------------------------------
    # Normalize all three signals.
    #
    # IMPORTANT:
    # normalizers were fitted on NORMAL VALIDATION only.
    # --------------------------------------------------------

    normal_yolo_norm = (
        yolo_normalizer.transform(
            normal_test_yolo
        )
    )

    anth_yolo_norm = (
        yolo_normalizer.transform(
            anth_yolo
        )
    )

    normal_vae_norm = (
        vae_normalizer.transform(
            test_vae
        )
    )

    anth_vae_norm = (
        vae_normalizer.transform(
            anth_vae
        )
    )

    normal_flow_norm = (
        flow_normalizer.transform(
            test_flow
        )
    )

    anth_flow_norm = (
        flow_normalizer.transform(
            anth_flow
        )
    )

    # --------------------------------------------------------
    # Binary evaluation
    #
    # 0 = normal
    # 1 = anthropogenic
    # --------------------------------------------------------

    y_true = np.concatenate(
        [
            np.zeros(
                len(normal_test_yolo),
                dtype=np.int32
            ),

            np.ones(
                len(anth_yolo),
                dtype=np.int32
            ),
        ]
    )

    # --------------------------------------------------------
    # Individual signals
    # --------------------------------------------------------

    normal_yolo_all = normal_yolo_norm
    anth_yolo_all = anth_yolo_norm

    normal_vae_all = normal_vae_norm
    anth_vae_all = anth_vae_norm

    normal_flow_all = normal_flow_norm
    anth_flow_all = anth_flow_norm

    # --------------------------------------------------------
    # Fusion signals
    #
    # Equal-weight fusion.
    #
    # This is intentionally NOT optimized on the test set.
    # --------------------------------------------------------

    normal_yolo_vae = (
        0.5 * normal_yolo_norm
        + 0.5 * normal_vae_norm
    )

    anth_yolo_vae = (
        0.5 * anth_yolo_norm
        + 0.5 * anth_vae_norm
    )

    normal_yolo_flow = (
        0.5 * normal_yolo_norm
        + 0.5 * normal_flow_norm
    )

    anth_yolo_flow = (
        0.5 * anth_yolo_norm
        + 0.5 * anth_flow_norm
    )

    normal_vae_flow = (
        0.5 * normal_vae_norm
        + 0.5 * normal_flow_norm
    )

    anth_vae_flow = (
        0.5 * anth_vae_norm
        + 0.5 * anth_flow_norm
    )

    normal_all = (
        (
            normal_yolo_norm
            + normal_vae_norm
            + normal_flow_norm
        )
        / 3.0
    )

    anth_all = (
        (
            anth_yolo_norm
            + anth_vae_norm
            + anth_flow_norm
        )
        / 3.0
    )

    # --------------------------------------------------------
    # Combine into complete score arrays
    # --------------------------------------------------------

    signal_dict = {

        "YOLO":
            np.concatenate(
                [
                    normal_yolo_all,
                    anth_yolo_all
                ]
            ),

        "VAE":
            np.concatenate(
                [
                    normal_vae_all,
                    anth_vae_all
                ]
            ),

        "Flow":
            np.concatenate(
                [
                    normal_flow_all,
                    anth_flow_all
                ]
            ),

        "YOLO + VAE":
            np.concatenate(
                [
                    normal_yolo_vae,
                    anth_yolo_vae
                ]
            ),

        "YOLO + Flow":
            np.concatenate(
                [
                    normal_yolo_flow,
                    anth_yolo_flow
                ]
            ),

        "VAE + Flow":
            np.concatenate(
                [
                    normal_vae_flow,
                    anth_vae_flow
                ]
            ),

        "YOLO + VAE + Flow":
            np.concatenate(
                [
                    normal_all,
                    anth_all
                ]
            ),
    }

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = []

    for name, scores in signal_dict.items():

        results.append(
            evaluate_signal(
                name,
                y_true,
                scores
            )
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "SAAD SIGNAL COMPARISON"
    )

    print(
        "=" * 70
    )

    print()

    print(
        results_df[
            [
                "signal",
                "ROC_AUC",
                "PR_AUC",
                "Best_F1",
                "Best_F1_precision",
                "Best_F1_recall",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # Save result table
    # --------------------------------------------------------

    results_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "fusion_results.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Save per-image scores
    # --------------------------------------------------------

    score_rows = []

    # Normal
    for i, path in enumerate(
        normal_test_paths
    ):

        score_rows.append(
            {
                "dataset":
                    "NormalTest",

                "image_path":
                    path,

                "label":
                    0,

                "yolo":
                    float(
                        normal_yolo_norm[i]
                    ),

                "vae":
                    float(
                        normal_vae_norm[i]
                    ),

                "flow":
                    float(
                        normal_flow_norm[i]
                    ),

                "yolo_vae":
                    float(
                        normal_yolo_vae[i]
                    ),

                "yolo_flow":
                    float(
                        normal_yolo_flow[i]
                    ),

                "vae_flow":
                    float(
                        normal_vae_flow[i]
                    ),

                "yolo_vae_flow":
                    float(
                        normal_all[i]
                    ),
            }
        )

    # Anthropogenic
    for i, path in enumerate(
        anthropogenic_paths
    ):

        score_rows.append(
            {
                "dataset":
                    anthropogenic_dataset_names[i],

                "image_path":
                    path,

                "label":
                    1,

                "yolo":
                    float(
                        anth_yolo_norm[i]
                    ),

                "vae":
                    float(
                        anth_vae_norm[i]
                    ),

                "flow":
                    float(
                        anth_flow_norm[i]
                    ),

                "yolo_vae":
                    float(
                        anth_yolo_vae[i]
                    ),

                "yolo_flow":
                    float(
                        anth_yolo_flow[i]
                    ),

                "vae_flow":
                    float(
                        anth_vae_flow[i]
                    ),

                "yolo_vae_flow":
                    float(
                        anth_all[i]
                    ),
            }
        )

    scores_df = pd.DataFrame(
        score_rows
    )

    scores_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "per_image_fusion_scores.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Dataset-wise results
    # --------------------------------------------------------

    dataset_rows = []

    for dataset_name in [
        "GhostVision",
        "AI4Shipwrecks",
        "SubPipeMini2",
    ]:

        mask = (
            scores_df["dataset"]
            == dataset_name
        )

        subset = scores_df[
            mask
        ]

        if len(subset) == 0:

            continue

        # Only anthropogenic examples.
        for signal in [
            "yolo",
            "vae",
            "flow",
            "yolo_vae",
            "yolo_flow",
            "vae_flow",
            "yolo_vae_flow",
        ]:

            values = subset[
                signal
            ].to_numpy()

            dataset_rows.append(
                {
                    "dataset":
                        dataset_name,

                    "signal":
                        signal,

                    "N":
                        len(values),

                    "mean":
                        np.mean(values),

                    "median":
                        np.median(values),

                    "p95":
                        np.percentile(
                            values,
                            95
                        ),
                }
            )

    dataset_df = pd.DataFrame(
        dataset_rows
    )

    dataset_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "dataset_wise_scores.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Threshold analysis
    #
    # Normal validation P95 threshold in normalized
    # score space = approximately 0.95.
    #
    # Since the three branches were individually normalized,
    # a fused score >= 0.95 represents a high anomaly score.
    # --------------------------------------------------------

    threshold_rows = []

    for name, scores in signal_dict.items():

        metrics = threshold_metrics(
            y_true,
            scores,
            0.95
        )

        threshold_rows.append(
            {
                "signal":
                    name,

                "threshold":
                    0.95,

                "precision":
                    metrics[
                        "precision"
                    ],

                "recall":
                    metrics[
                        "recall"
                    ],

                "f1":
                    metrics[
                        "f1"
                    ],

                "TP":
                    metrics[
                        "TP"
                    ],

                "FP":
                    metrics[
                        "FP"
                    ],

                "FN":
                    metrics[
                        "FN"
                    ],

                "TN":
                    metrics[
                        "TN"
                    ],
            }
        )

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    threshold_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "threshold_results.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    metric_plot = (
        plot_metric_comparison(
            results
        )
    )

    roc_plot = (
        plot_roc_comparison(
            y_true,
            signal_dict
        )
    )

    pr_plot = (
        plot_pr_comparison(
            y_true,
            signal_dict
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_path = os.path.join(
        OUTPUT_DIR,
        "fusion_summary.txt"
    )

    best_row = results_df.iloc[
        results_df["PR_AUC"].argmax()
    ]

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SAAD — SCORE FUSION EVALUATION\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            "MODELS\n"
        )

        f.write(
            f"YOLO: {YOLO_CHECKPOINT}\n"
        )

        f.write(
            f"VAE: {VAE_CHECKPOINT}\n"
        )

        f.write(
            f"Flow: {FLOW_CHECKPOINT}\n\n"
        )

        f.write(
            "EVALUATION\n"
        )

        f.write(
            f"Normal test images: "
            f"{len(normal_test_paths)}\n"
        )

        f.write(
            f"Anthropogenic images: "
            f"{len(anthropogenic_paths)}\n\n"
        )

        f.write(
            "IMPORTANT:\n"
        )

        f.write(
            "Fusion weights are fixed equal weights.\n"
        )

        f.write(
            "No fusion weights were optimized on "
            "the test set.\n"
        )

        f.write(
            "Score normalization uses normal "
            "validation data only.\n\n"
        )

        f.write(
            "RESULTS\n"
        )

        f.write(
            results_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            "BEST PR-AUC SIGNAL\n"
        )

        f.write(
            f"Signal: "
            f"{best_row['signal']}\n"
        )

        f.write(
            f"ROC-AUC: "
            f"{best_row['ROC_AUC']:.6f}\n"
        )

        f.write(
            f"PR-AUC: "
            f"{best_row['PR_AUC']:.6f}\n"
        )

        f.write(
            f"Best F1: "
            f"{best_row['Best_F1']:.6f}\n"
        )

        f.write(
            "\n\n"
        )

        f.write(
            "OUTPUT FILES\n"
        )

        f.write(
            f"{metric_plot}\n"
        )

        f.write(
            f"{roc_plot}\n"
        )

        f.write(
            f"{pr_plot}\n"
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "SAAD FUSION EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print()
    print(
        "Best signal by PR-AUC:"
    )

    print(
        f"  {best_row['signal']}"
    )

    print(
        f"  ROC-AUC = "
        f"{best_row['ROC_AUC']:.6f}"
    )

    print(
        f"  PR-AUC  = "
        f"{best_row['PR_AUC']:.6f}"
    )

    print()
    print(
        "Results:"
    )

    print(
        f"  {os.path.join(OUTPUT_DIR, 'fusion_results.csv')}"
    )

    print(
        f"  {os.path.join(OUTPUT_DIR, 'per_image_fusion_scores.csv')}"
    )

    print(
        f"  {os.path.join(OUTPUT_DIR, 'dataset_wise_scores.csv')}"
    )

    print(
        f"  {os.path.join(OUTPUT_DIR, 'threshold_results.csv')}"
    )

    print(
        f"  {summary_path}"
    )

    print()
    print(
        "Plots:"
    )

    print(
        f"  {metric_plot}"
    )

    print(
        f"  {roc_plot}"
    )

    print(
        f"  {pr_plot}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    seed_everything(
        SEED
    )

    main()