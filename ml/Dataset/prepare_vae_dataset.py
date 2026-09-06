from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import hashlib
import random
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

SAAD_ROOT = Path(r"E:\SIH\Datasets\SAAD_baseline")
OUTPUT_ROOT = Path(r"E:\SIH\Datasets\SAAD_VAE")

MANIFEST_PATH = SAAD_ROOT / "manifest.csv"

# Patch configuration
PATCH_SIZE = 256
STRIDE = 192          # 25% overlap

# Minimum fraction of valid pixels required in a patch
MIN_VALID_FRACTION = 0.95

# Reproducibility
SEED = 42

# Maximum number of patches per source image.
# Set to None for unlimited.
MAX_PATCHES_PER_IMAGE = None

# ============================================================
# SOURCE SELECTION
# ============================================================

# Sources that are considered NORMAL for the first VAE experiment.
#
# GhostVision:
#   normal/background images
#
# Marine-PULSE:
#   seabed surface
#
# We deliberately do NOT use arbitrary unlabeled imagery.
# ============================================================


def is_normal_row(row):
    dataset = str(row.get("dataset", "")).strip().lower()
    source_class = str(row.get("source_class", "")).strip().lower()
    saad_class = str(row.get("saad_class", "")).strip().lower()

    # --------------------------------------------------------
    # GhostVision normal/background
    # --------------------------------------------------------
    if dataset == "ghostvision":
        normal_terms = {
            "normal",
            "background",
            "empty",
            "negative",
            "none",
            "no object",
            "no-object",
        }

        if (
            source_class in normal_terms
            or saad_class in normal_terms
            or (
                str(row.get("has_annotation", "")).lower() in
                {"false", "0", "no"}
                and int(float(row.get("num_objects", 0) or 0)) == 0
            )
        ):
            return True

    # --------------------------------------------------------
    # Marine-PULSE seabed surface
    # --------------------------------------------------------
    if dataset == "marine-pulse":
        if source_class in {
            "seabed surface",
            "seabed",
            "normal seabed",
            "normal",
            "background",
        }:
            return True

    return False


# ============================================================
# UTILITIES
# ============================================================

def safe_name(text):
    """Make a filesystem-safe string."""
    text = str(text)
    chars = []

    for c in text:
        if c.isalnum() or c in ("-", "_"):
            chars.append(c)
        else:
            chars.append("_")

    return "".join(chars)


def image_hash(path):
    """SHA256 hash of an image file."""
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def load_grayscale(path):
    """
    Load image as grayscale float32 array in [0, 1].
    """
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def generate_patch_positions(height, width, patch_size, stride):
    """
    Generate patch coordinates while ensuring the image borders
    are covered.
    """

    ys = list(range(0, max(height - patch_size + 1, 1), stride))
    xs = list(range(0, max(width - patch_size + 1, 1), stride))

    # Ensure final row/column reaches the image boundary
    if height > patch_size:
        final_y = height - patch_size
        if final_y not in ys:
            ys.append(final_y)

    if width > patch_size:
        final_x = width - patch_size
        if final_x not in xs:
            xs.append(final_x)

    return ys, xs


def save_patch(arr, output_path):
    """
    Save normalized float patch as 8-bit grayscale PNG.
    """

    arr = np.clip(arr, 0.0, 1.0)
    arr_uint8 = (arr * 255.0).round().astype(np.uint8)

    img = Image.fromarray(arr_uint8, mode="L")
    img.save(output_path)


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 70)
    print("SAAD VAE DATASET PREPARATION")
    print("=" * 70)

    print(f"Input root : {SAAD_ROOT}")
    print(f"Output     : {OUTPUT_ROOT}")
    print(f"Patch size : {PATCH_SIZE}")
    print(f"Stride     : {STRIDE}")
    print()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    # --------------------------------------------------------
    # Clean/create output
    # --------------------------------------------------------

    if OUTPUT_ROOT.exists():
        print("WARNING: Output directory already exists.")
        print("It will be removed and rebuilt.")

        shutil.rmtree(OUTPUT_ROOT)

    train_dir = OUTPUT_ROOT / "train"
    val_dir = OUTPUT_ROOT / "val"
    test_dir = OUTPUT_ROOT / "test"

    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    # --------------------------------------------------------
    # Read manifest
    # --------------------------------------------------------

    df = pd.read_csv(MANIFEST_PATH)

    print(f"Manifest rows: {len(df)}")

    # --------------------------------------------------------
    # Select normal images
    # --------------------------------------------------------

    normal_mask = df.apply(is_normal_row, axis=1)
    normal_df = df[normal_mask].copy()

    print(f"Normal candidate rows: {len(normal_df)}")

    if len(normal_df) == 0:
        raise RuntimeError(
            "No normal images were found.\n"
            "Inspect the manifest's dataset/source_class/saad_class values."
        )

    # --------------------------------------------------------
    # Resolve image paths
    # --------------------------------------------------------

    resolved_rows = []

    for _, row in normal_df.iterrows():

        image_path = str(row["output_image"])

        path = Path(image_path)

        if not path.is_absolute():
            path = SAAD_ROOT / path

        if not path.exists():
            print(f"[WARNING] Missing image: {path}")
            continue

        row = row.copy()
        row["_resolved_path"] = str(path)

        resolved_rows.append(row)

    normal_df = pd.DataFrame(resolved_rows)

    print(f"Existing normal images: {len(normal_df)}")

    if len(normal_df) == 0:
        raise RuntimeError("No usable normal images found.")

    # --------------------------------------------------------
    # Remove exact duplicate source images
    # --------------------------------------------------------

    print()
    print("Checking exact image duplicates...")

    hashes = {}
    keep_rows = []

    duplicate_count = 0

    for _, row in normal_df.iterrows():

        path = Path(row["_resolved_path"])

        try:
            h = image_hash(path)
        except Exception as e:
            print(f"[WARNING] Could not hash {path}: {e}")
            continue

        if h in hashes:
            duplicate_count += 1
            continue

        hashes[h] = path
        row["_image_hash"] = h
        keep_rows.append(row)

    normal_df = pd.DataFrame(keep_rows)

    print(f"Exact duplicate images removed: {duplicate_count}")
    print(f"Unique normal images: {len(normal_df)}")

    # --------------------------------------------------------
    # IMAGE-LEVEL SPLIT
    #
    # VERY IMPORTANT:
    # We split source images BEFORE generating patches.
    # --------------------------------------------------------

    images = list(range(len(normal_df)))
    random.shuffle(images)

    n = len(images)

    n_train = int(n * 0.80)
    n_val = int(n * 0.10)

    train_indices = set(images[:n_train])
    val_indices = set(images[n_train:n_train + n_val])
    test_indices = set(images[n_train + n_val:])

    print()
    print("Image-level split:")
    print(f"  train images: {len(train_indices)}")
    print(f"  val images  : {len(val_indices)}")
    print(f"  test images : {len(test_indices)}")

    # --------------------------------------------------------
    # PATCH EXTRACTION
    # --------------------------------------------------------

    output_records = []

    patch_counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    failed_images = 0

    for idx, (_, row) in enumerate(normal_df.iterrows()):

        if idx in train_indices:
            split = "train"
            split_dir = train_dir

        elif idx in val_indices:
            split = "val"
            split_dir = val_dir

        else:
            split = "test"
            split_dir = test_dir

        image_path = Path(row["_resolved_path"])

        print(
            f"[{idx + 1}/{len(normal_df)}] "
            f"{row['dataset']} | "
            f"{image_path.name} | "
            f"{split}"
        )

        try:
            arr = load_grayscale(image_path)

        except Exception as e:
            print(f"    FAILED: {e}")
            failed_images += 1
            continue

        height, width = arr.shape

        if height < PATCH_SIZE or width < PATCH_SIZE:
            print(
                f"    SKIPPED: image too small "
                f"({width}x{height})"
            )
            continue

        ys, xs = generate_patch_positions(
            height,
            width,
            PATCH_SIZE,
            STRIDE,
        )

        candidates = []

        for y in ys:
            for x in xs:

                patch = arr[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ]

                if patch.shape != (PATCH_SIZE, PATCH_SIZE):
                    continue

                # ------------------------------------------------
                # Validity check
                # ------------------------------------------------

                finite_fraction = np.isfinite(patch).mean()

                if finite_fraction < MIN_VALID_FRACTION:
                    continue

                patch = np.nan_to_num(
                    patch,
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                )

                candidates.append((x, y, patch))

        # Optional patch limit
        if (
            MAX_PATCHES_PER_IMAGE is not None
            and len(candidates) > MAX_PATCHES_PER_IMAGE
        ):
            random.shuffle(candidates)
            candidates = candidates[:MAX_PATCHES_PER_IMAGE]

        # --------------------------------------------------------
        # Save patches
        # --------------------------------------------------------

        image_stem = safe_name(image_path.stem)

        source_id = (
            f"{row['dataset']}_"
            f"{image_stem}_"
            f"{row['_image_hash'][:8]}"
        )

        for patch_id, (x, y, patch) in enumerate(candidates):

            filename = (
                f"{safe_name(source_id)}"
                f"_x{x}_y{y}_p{patch_id:04d}.png"
            )

            output_path = split_dir / filename

            save_patch(patch, output_path)

            output_records.append({
                "patch_path": str(
                    output_path.relative_to(OUTPUT_ROOT)
                ),
                "split": split,
                "dataset": row["dataset"],
                "source_image": str(image_path),
                "source_image_hash": row["_image_hash"],
                "source_class": row["source_class"],
                "x": x,
                "y": y,
                "patch_size": PATCH_SIZE,
                "original_width": width,
                "original_height": height,
            })

            patch_counts[split] += 1

    # --------------------------------------------------------
    # Write manifest
    # --------------------------------------------------------

    patch_manifest = pd.DataFrame(output_records)

    manifest_out = OUTPUT_ROOT / "manifest.csv"

    patch_manifest.to_csv(
        manifest_out,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VAE DATASET COMPLETE")
    print("=" * 70)

    print()
    print("Images:")
    print(f"  total normal images : {len(normal_df)}")
    print(f"  failed images       : {failed_images}")

    print()
    print("Patches:")
    print(f"  train: {patch_counts['train']}")
    print(f"  val  : {patch_counts['val']}")
    print(f"  test : {patch_counts['test']}")
    print(f"  total: {sum(patch_counts.values())}")

    print()
    print("Dataset:")
    print(f"  {OUTPUT_ROOT}")

    print()
    print("Manifest:")
    print(f"  {manifest_out}")

    print()
    print("IMPORTANT:")
    print("  Splitting was performed at SOURCE IMAGE level.")
    print("  Patches from one source image cannot cross train/val/test.")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()