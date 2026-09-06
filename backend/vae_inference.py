from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIGURATION
# ============================================================

VAE_CHECKPOINT = Path(
    r"E:\SIH\SIH_Results\vae_normal_seabed\checkpoints\best.pt"
)

DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

PATCH_SIZE = 256


# ============================================================
# VAE-v1
# ============================================================

class ConvVAE(nn.Module):

    def __init__(
        self,
        latent_dim: int = 128,
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
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                512,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.fc_mu = nn.Linear(
            512 * 8 * 8,
            latent_dim,
        )

        self.fc_logvar = nn.Linear(
            512 * 8 * 8,
            latent_dim,
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.fc_decode = nn.Linear(
            latent_dim,
            512 * 8 * 8,
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                512,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32,
                1,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.Sigmoid(),
        )


    def encode(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        h = self.encoder(x)

        h = h.view(
            h.size(0),
            -1,
        )

        mu = self.fc_mu(h)

        logvar = self.fc_logvar(h)

        logvar = torch.clamp(
            logvar,
            -10,
            10,
        )

        return mu, logvar


    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:

        std = torch.exp(
            0.5 * logvar
        )

        eps = torch.randn_like(std)

        return mu + eps * std


    def decode(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:

        h = self.fc_decode(z)

        h = h.view(
            -1,
            512,
            8,
            8,
        )

        return self.decoder(h)


    def forward(
        self,
        x: torch.Tensor,
    ):

        mu, logvar = self.encode(x)

        # ----------------------------------------------------
        # IMPORTANT
        #
        # For inference we use deterministic mu rather than
        # random sampling. This makes anomaly scores stable.
        # ----------------------------------------------------

        reconstruction = self.decode(mu)

        return reconstruction, mu, logvar


# ============================================================
# LOAD MODEL
# ============================================================

def _extract_state_dict(checkpoint):

    if isinstance(
        checkpoint,
        dict,
    ):

        possible_keys = [
            "model_state_dict",
            "state_dict",
            "model",
        ]

        for key in possible_keys:

            value = checkpoint.get(key)

            if isinstance(
                value,
                dict,
            ):
                return value

    if isinstance(
        checkpoint,
        dict,
    ):
        return checkpoint

    raise RuntimeError(
        "Unable to locate VAE state_dict in checkpoint."
    )


def load_vae():

    if not VAE_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"VAE checkpoint not found: "
            f"{VAE_CHECKPOINT}"
        )

    model = ConvVAE(
        latent_dim=128,
    )

    checkpoint = torch.load(
        VAE_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    state_dict = _extract_state_dict(
        checkpoint
    )

    # --------------------------------------------------------
    # Handle possible DataParallel checkpoints
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[len("module."):]

        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False,
    )

    if missing:
        raise RuntimeError(
            "VAE checkpoint is incompatible with the "
            "reconstructed VAE-v1 architecture.\n"
            f"Missing keys: {missing[:10]}"
        )

    if unexpected:
        print(
            "WARNING: unexpected VAE checkpoint keys:",
            unexpected[:10],
        )

    model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def image_to_grayscale_array(
    image: Image.Image,
) -> np.ndarray:

    gray = image.convert("L")

    array = np.asarray(
        gray,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Same basic [0,1] representation used by the VAE.
    # --------------------------------------------------------

    array /= 255.0

    return array


# ============================================================
# CANDIDATE PATCH EXTRACTION
# ============================================================

def extract_candidate_patch(
    image: Image.Image,
    bbox_pixels: Dict[str, float],
) -> Image.Image:

    gray = image.convert("L")

    width, height = gray.size

    x1 = float(
        bbox_pixels["x1"]
    )

    y1 = float(
        bbox_pixels["y1"]
    )

    x2 = float(
        bbox_pixels["x2"]
    )

    y2 = float(
        bbox_pixels["y2"]
    )

    x1 = max(
        0.0,
        min(x1, width - 1),
    )

    y1 = max(
        0.0,
        min(y1, height - 1),
    )

    x2 = max(
        x1 + 1,
        min(x2, width),
    )

    y2 = max(
        y1 + 1,
        min(y2, height),
    )

    box_width = x2 - x1
    box_height = y2 - y1

    # --------------------------------------------------------
    # Give the object some local acoustic context.
    #
    # We deliberately keep this modest because previous
    # experiments showed that excessive ROI context makes the
    # VAE respond to surrounding seabed heterogeneity.
    # --------------------------------------------------------

    context = 0.20

    pad_x = box_width * context
    pad_y = box_height * context

    x1 -= pad_x
    y1 -= pad_y
    x2 += pad_x
    y2 += pad_y

    # --------------------------------------------------------
    # Make crop square.
    # --------------------------------------------------------

    crop_width = x2 - x1
    crop_height = y2 - y1

    side = max(
        crop_width,
        crop_height,
    )

    center_x = (
        x1 + x2
    ) / 2.0

    center_y = (
        y1 + y2
    ) / 2.0

    x1 = center_x - side / 2.0
    x2 = center_x + side / 2.0

    y1 = center_y - side / 2.0
    y2 = center_y + side / 2.0

    # --------------------------------------------------------
    # Clamp square to image.
    # --------------------------------------------------------

    if x1 < 0:
        x2 -= x1
        x1 = 0

    if y1 < 0:
        y2 -= y1
        y1 = 0

    if x2 > width:
        x1 -= x2 - width
        x2 = width

    if y2 > height:
        y1 -= y2 - height
        y2 = height

    x1 = max(
        0,
        int(round(x1)),
    )

    y1 = max(
        0,
        int(round(y1)),
    )

    x2 = min(
        width,
        int(round(x2)),
    )

    y2 = min(
        height,
        int(round(y2)),
    )

    crop = gray.crop(
        (
            x1,
            y1,
            x2,
            y2,
        )
    )

    crop = crop.resize(
        (
            PATCH_SIZE,
            PATCH_SIZE,
        ),
        Image.Resampling.BILINEAR,
    )

    return crop


# ============================================================
# VAE SCORE
# ============================================================

@torch.inference_mode()
def score_patch(
    model: ConvVAE,
    patch: Image.Image,
) -> float:

    array = np.asarray(
        patch,
        dtype=np.float32,
    )

    array /= 255.0

    tensor = torch.from_numpy(
        array
    )

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(
        DEVICE,
        non_blocking=True,
    )

    reconstruction, _, _ = model(
        tensor
    )

    mse = F.mse_loss(
        reconstruction,
        tensor,
        reduction="mean",
    )

    return float(
        mse.detach()
        .cpu()
        .item()
    )


# ============================================================
# PUBLIC SCORING FUNCTION
# ============================================================

def score_candidate(
    model: ConvVAE,
    image: Image.Image,
    bbox_pixels: Dict[str, float],
) -> float:

    patch = extract_candidate_patch(
        image,
        bbox_pixels,
    )

    return score_patch(
        model,
        patch,
    )