from typing import Dict

import numpy as np
from PIL import Image

PATCH_SIZE = 256


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
