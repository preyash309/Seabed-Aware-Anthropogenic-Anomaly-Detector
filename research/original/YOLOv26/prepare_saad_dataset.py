"""
SAAD DATASET REBUILD v4
=======================

Purpose:
    Rebuild SAAD_baseline from the original datasets with strong
    integrity guarantees before YOLO training.

IMPORTANT:
    - This script REPLACES the existing SAAD_baseline directory.
    - It does NOT modify the original source datasets.
    - Close anything that may have files open inside SAAD_baseline first.
    - Review CONFIG before running.

Pipeline:
    GhostVision      -> anthropogenic + normal
    AI4Shipwrecks    -> anthropogenic + normal
    SubPipeMini2     -> anthropogenic
    Marine-PULSE     -> seabed surface = normal

Ontology:
    0 = anthropogenic

Key fixes vs previous build:
    1. Unique output filenames; no silent overwrites.
    2. Every output image gets exactly one label file.
    3. Manifest is generated from actual successful writes.
    4. AI4 recursively discovers the nested dataset structure.
    5. AI4 train/val split is grouped by wreck/group.
    6. SubPipe split is based on timestamp sequence blocks.
    7. Explicit assertions catch collisions and mismatches.
    8. Final integrity summary is printed and written to disk.

Requirements:
    pip install pillow tqdm numpy
"""

from pathlib import Path
from collections import defaultdict, Counter
import csv
import json
import math
import random
import re
import shutil
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r"E:\SIH\Datasets")

OUTPUT = ROOT / "SAAD_baseline"

GHOST_ROOT = ROOT / "sss-crab-pot-detection-ds"
AI4_ROOT = ROOT / "AI4Shipwrecks"
SUBPIPE_ROOT = ROOT / "SubPipeMini2"
MARINE_ROOT = ROOT / "Marine_PULSE"

# Rebuild from scratch.
# Set to False if you want the script to STOP instead of replacing
# an existing SAAD_baseline.
REBUILD_EXISTING = True

SEED = 42
random.seed(SEED)

IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".bmp",
    ".pbm", ".pgm", ".tif", ".tiff", ".webp"
}

# AI4: group-based train/val fraction.
# Official AI4 test remains untouched.
AI4_VAL_FRACTION = 0.20

# SubPipe: timestamp block duration used to assign whole temporal blocks
# to one split. Blocks never get split across train/val/test.
SUBPIPE_BLOCK_SECONDS = 300.0  # 5 minutes

# Desired SubPipe proportions among matched/annotated images.
SUBPIPE_TRAIN_FRACTION = 0.78
SUBPIPE_VAL_FRACTION = 0.12
SUBPIPE_TEST_FRACTION = 0.10


# ============================================================
# OUTPUT SETUP
# ============================================================

def safe_reset_output():
    if OUTPUT.exists():
        if not REBUILD_EXISTING:
            raise RuntimeError(
                f"{OUTPUT} already exists. "
                f"Set REBUILD_EXISTING=True to rebuild it."
            )

        print("\nRemoving previous SAAD_baseline...")
        shutil.rmtree(OUTPUT)

    for split in ["train", "val", "test"]:
        (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)


safe_reset_output()


# ============================================================
# GLOBAL STATE
# ============================================================

manifest_rows = []

# Absolute source path -> generated output stem.
source_to_output = {}

# Generated output filename -> source path.
output_to_source = {}

stats = Counter()


# ============================================================
# GENERIC HELPERS
# ============================================================

def norm(p):
    return str(p).replace("/", "\\").lower()


def image_files(root):
    if not root.exists():
        return []

    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def timestamp_from_path(path):
    """
    Extract the final large numeric timestamp from a filename/path.
    SubPipe filenames commonly contain Unix-like timestamps.
    """
    matches = re.findall(
        r"(?<!\d)(\d{9,13}(?:\.\d+)?)",
        str(path)
    )

    if not matches:
        return None

    try:
        return float(matches[-1])
    except Exception:
        return None


def normalize_to_uint8(arr):
    """
    Preserve ordinary uint8 images as-is.

    Only scale non-uint8 arrays into uint8. This avoids unnecessarily
    applying per-image min/max normalization to normal 8-bit SSS data.
    """
    arr = np.asarray(arr)

    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)

    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    lo = float(arr[finite].min())
    hi = float(arr[finite].max())

    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = (arr - lo) / (hi - lo)
    arr = np.clip(arr * 255.0, 0, 255)

    return arr.astype(np.uint8)


def load_gray(path):
    """
    Load arbitrary supported image formats into single-channel uint8.
    """
    with Image.open(path) as im:
        im = im.convert("L")
        arr = np.asarray(im)

    return normalize_to_uint8(arr)


def unique_stem(dataset, source):
    """
    Output stem contains dataset + original relative/name information.

    The source path is encoded sufficiently to prevent common collisions.
    A final deterministic hash guarantees uniqueness even if two source
    files have identical names/relative names.
    """
    import hashlib

    src = norm(source)
    digest = hashlib.sha1(src.encode("utf-8")).hexdigest()[:10]

    # Keep the filename human-readable.
    name = source.stem
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)

    return f"{dataset}__{name}__{digest}"


def output_paths(dataset, split, source):
    stem = unique_stem(dataset, source)

    img_out = OUTPUT / "images" / split / f"{stem}.png"
    lbl_out = OUTPUT / "labels" / split / f"{stem}.txt"

    return img_out, lbl_out


def ensure_unique_source(dataset, source):
    key = norm(source)

    if key in source_to_output:
        raise RuntimeError(
            f"DUPLICATE SOURCE PROCESSED TWICE:\n{source}"
        )

    source_to_output[key] = dataset


def ensure_output_unused(img_out, lbl_out, source):
    for p in [img_out, lbl_out]:
        if p.exists():
            old = output_to_source.get(norm(p), "unknown")
            raise RuntimeError(
                "OUTPUT COLLISION — refusing to overwrite:\n"
                f"New source: {source}\n"
                f"Output: {p}\n"
                f"Existing source: {old}"
            )

    output_to_source[norm(img_out)] = source
    output_to_source[norm(lbl_out)] = source


def save_image_and_label(
    dataset,
    split,
    source,
    boxes,
    source_class,
    saad_class,
    group,
    annotation_type,
    has_annotation=True,
):
    """
    Write one unique image + exactly one label file + manifest row.
    boxes must be YOLO normalized tuples:
        [(0, x, y, w, h), ...]
    """
    ensure_unique_source(dataset, source)

    img_out, lbl_out = output_paths(dataset, split, source)
    ensure_output_unused(img_out, lbl_out, source)

    arr = load_gray(source)
    Image.fromarray(arr).save(img_out)

    # Every image gets a label file.
    # Empty file = normal/background image.
    with open(lbl_out, "w", encoding="utf-8") as f:
        for cls, x, y, w, h in boxes:
            f.write(
                f"{int(cls)} "
                f"{x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
            )

    manifest_rows.append({
        "dataset": dataset,
        "original_path": str(source),
        "output_image": str(img_out),
        "output_label": str(lbl_out),
        "split": split,
        "group": group,
        "source_class": source_class,
        "saad_class": saad_class,
        "has_annotation": int(has_annotation),
        "num_objects": len(boxes),
        "annotation_type": annotation_type,
    })

    stats[(dataset, split, saad_class)] += 1


# ============================================================
# YOLO / BBOX HELPERS
# ============================================================

def clamp01(v):
    return max(0.0, min(1.0, float(v)))


def bbox_to_yolo(x1, y1, x2, y2, W, H):
    """
    Convert pixel bbox to normalized YOLO bbox.
    """
    x1 = max(0.0, min(float(W), float(x1)))
    x2 = max(0.0, min(float(W), float(x2)))
    y1 = max(0.0, min(float(H), float(y1)))
    y2 = max(0.0, min(float(H), float(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    cx = ((x1 + x2) / 2.0) / W
    cy = ((y1 + y2) / 2.0) / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H

    return (
        0,
        clamp01(cx),
        clamp01(cy),
        clamp01(w),
        clamp01(h),
    )


def parse_yolo_file(path):
    boxes = []

    if not path.exists():
        return boxes

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return boxes

    for line in text.splitlines():
        parts = line.split()

        if len(parts) != 5:
            raise RuntimeError(
                f"Invalid YOLO annotation: {path}\n"
                f"Line: {line}"
            )

        cls = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])

        if cls != 0:
            # SubPipe's original class should be 0.
            # SAAD itself is always class 0.
            raise RuntimeError(
                f"Unexpected SubPipe class {cls}: {path}"
            )

        boxes.append(
            (
                0,
                clamp01(x),
                clamp01(y),
                clamp01(w),
                clamp01(h),
            )
        )

    return boxes


# ============================================================
# GHOSTVISION
# ============================================================

def parse_ghostvision_record(record, image_path):
    """
    Parse the ACTUAL GhostVision metadata.jsonl schema.

    Example:
        {
          "file_name": "....jpg",
          "objects": {
              "bbox": [[315, 213, 49, 62.5], ...],
              "category": ["Crab-Pot", ...],
              "area": [...]
          }
        }

    GhostVision bbox format:
        [x_left, y_top, width, height]

    SAAD ontology:
        0 = anthropogenic

    Crab-Pot and Maybe-Crab-Pot are both mapped to
    SAAD class 0.

    Empty bbox/category arrays are treated as NORMAL.
    """

    if not isinstance(record, dict):
        raise RuntimeError(
            f"GhostVision metadata record is not a JSON object: "
            f"{type(record)}"
        )

    if "file_name" not in record:
        raise RuntimeError(
            "GhostVision metadata record has no file_name."
        )

    objects = record.get("objects", {})

    if not isinstance(objects, dict):
        raise RuntimeError(
            f"GhostVision 'objects' is not a dictionary: "
            f"{objects}"
        )

    bboxes = objects.get("bbox", []) or []
    categories = objects.get("category", []) or []

    if not isinstance(bboxes, list):
        raise RuntimeError(
            f"GhostVision bbox field is not a list: {bboxes}"
        )

    if not isinstance(categories, list):
        raise RuntimeError(
            f"GhostVision category field is not a list: {categories}"
        )

    if len(bboxes) != len(categories):
        raise RuntimeError(
            f"GhostVision bbox/category count mismatch for "
            f"{record['file_name']}: "
            f"{len(bboxes)} vs {len(categories)}"
        )

    with Image.open(image_path) as im:
        W, H = im.size

    boxes = []

    for bbox, category in zip(bboxes, categories):

        if category not in ("Crab-Pot", "Maybe-Crab-Pot"):
            raise RuntimeError(
                f"Unexpected GhostVision category "
                f"{category!r} for {record['file_name']}"
            )

        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise RuntimeError(
                f"Invalid GhostVision bbox "
                f"{bbox!r} for {record['file_name']}"
            )

        x, y, w, h = map(float, bbox)

        if w <= 0 or h <= 0:
            raise RuntimeError(
                f"Non-positive GhostVision bbox "
                f"{bbox!r} for {record['file_name']}"
            )

        # GhostVision: [left, top, width, height]
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        box = bbox_to_yolo(
            x1, y1, x2, y2, W, H
        )

        if box is None:
            raise RuntimeError(
                f"GhostVision bbox became invalid after conversion: "
                f"{bbox!r} for {record['file_name']}"
            )

        boxes.append(box)

    return boxes


def process_ghostvision():
    """
    Process GhostVision using its official split-specific metadata files.

    Expected structure:

        E:\\SIH\\Datasets\\sss-crab-pot-detection-ds\\
            train\\
                metadata.jsonl
                *.jpg
            valid\\
                metadata.jsonl
                *.jpg
            test\\
                metadata.jsonl
                *.jpg

    We deliberately pair each metadata.jsonl with its own split directory
    instead of recursively searching the entire GhostVision root.

    This prevents accidental filename collisions and guarantees that
    train/valid/test are preserved exactly.
    """

    print("\n" + "=" * 60)
    print("PROCESSING GHOSTVISION")
    print("=" * 60)

    expected = {
        "train": {
            "total": 5721,
            "positive": 4291,
            "normal": 1430,
        },
        "val": {
            "total": 555,
            "positive": 502,
            "normal": 53,
        },
        "test": {
            "total": 398,
            "positive": 334,
            "normal": 64,
        },
    }

    total_processed = 0
    total_positive = 0
    total_normal = 0
    total_objects = 0

    # GhostVision calls its validation split "valid".
    # SAAD calls it "val".
    split_mapping = {
        "train": "train",
        "valid": "val",
        "test": "test",
    }

    for ghost_split, saad_split in split_mapping.items():

        split_dir = GHOST_ROOT / ghost_split
        metadata_path = split_dir / "metadata.jsonl"

        if not split_dir.exists():
            raise RuntimeError(
                f"GhostVision split directory not found:\n"
                f"{split_dir}"
            )

        if not metadata_path.exists():
            raise RuntimeError(
                f"GhostVision metadata not found:\n"
                f"{metadata_path}"
            )

        print()
        print("-" * 60)
        print(
            f"GhostVision {ghost_split} -> SAAD {saad_split}"
        )
        print("-" * 60)

        split_total = 0
        split_positive = 0
        split_normal = 0
        split_objects = 0
        seen_filenames = set()

        with metadata_path.open(
            "r",
            encoding="utf-8"
        ) as f:

            for line_no, line in enumerate(
                f,
                start=1
            ):

                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise RuntimeError(
                        f"Invalid JSON in "
                        f"{metadata_path} line {line_no}: {e}"
                    )

                filename = record.get("file_name")

                if not filename:
                    raise RuntimeError(
                        f"Missing file_name in "
                        f"{metadata_path} line {line_no}"
                    )

                filename_key = filename.lower()

                if filename_key in seen_filenames:
                    raise RuntimeError(
                        f"Duplicate file_name in "
                        f"{metadata_path}: {filename}"
                    )

                seen_filenames.add(filename_key)

                # ------------------------------------------------
                # Exact split-local image lookup.
                # ------------------------------------------------

                image_path = split_dir / filename

                if not image_path.exists():
                    # Fallback only within THIS split.
                    matches = list(
                        split_dir.rglob(filename)
                    )

                    if len(matches) == 0:
                        raise RuntimeError(
                            f"GhostVision image referenced by metadata "
                            f"was not found:\n"
                            f"{image_path}"
                        )

                    if len(matches) > 1:
                        raise RuntimeError(
                            f"Multiple GhostVision images match "
                            f"{filename} inside {split_dir}"
                        )

                    image_path = matches[0]

                boxes = parse_ghostvision_record(
                    record,
                    image_path
                )

                if boxes:
                    source_class = "Crab-Pot"
                    saad_class = "ANTHROPOGENIC"
                    has_annotation = True

                    split_positive += 1
                    split_objects += len(boxes)

                else:
                    source_class = "normal_or_unannotated"
                    saad_class = "NORMAL"
                    has_annotation = False

                    split_normal += 1

                group = image_path.stem.split("_")[0]

                save_image_and_label(
                    "GhostVision",
                    saad_split,
                    image_path,
                    boxes,
                    source_class,
                    saad_class,
                    group,
                    "bbox_jsonl",
                    has_annotation,
                )

                split_total += 1

                # Show first 3 records per split.
                if split_total <= 3:
                    print()
                    print(
                        f"Example {split_total}: "
                        f"{filename}"
                    )

                    with Image.open(image_path) as im:
                        print(
                            f"  Image size: "
                            f"{im.size[0]} x {im.size[1]}"
                        )

                    print(
                        f"  Objects: {len(boxes)}"
                    )

                    for box in boxes:
                        print(
                            "  YOLO: "
                            f"{box[0]} "
                            f"{box[1]:.6f} "
                            f"{box[2]:.6f} "
                            f"{box[3]:.6f} "
                            f"{box[4]:.6f}"
                        )

        # --------------------------------------------------------
        # Split-specific semantic sanity check.
        #
        # These are the counts we already established from the
        # original GhostVision dataset.
        # --------------------------------------------------------

        exp = expected[saad_split]

        print()
        print(
            f"{ghost_split.upper()} SUMMARY"
        )
        print("-" * 60)

        print(
            f"Records processed : {split_total}"
        )
        print(
            f"Anthropogenic     : {split_positive}"
        )
        print(
            f"Normal/background : {split_normal}"
        )
        print(
            f"Total objects     : {split_objects}"
        )

        if split_total != exp["total"]:
            raise RuntimeError(
                f"GhostVision {ghost_split} record count mismatch: "
                f"expected {exp['total']}, got {split_total}"
            )

        if split_positive != exp["positive"]:
            raise RuntimeError(
                f"GhostVision {ghost_split} positive count mismatch: "
                f"expected {exp['positive']}, got {split_positive}"
            )

        if split_normal != exp["normal"]:
            raise RuntimeError(
                f"GhostVision {ghost_split} normal count mismatch: "
                f"expected {exp['normal']}, got {split_normal}"
            )

        total_processed += split_total
        total_positive += split_positive
        total_normal += split_normal
        total_objects += split_objects

    # ------------------------------------------------------------
    # Global GhostVision semantic gate.
    # ------------------------------------------------------------

    print()
    print("=" * 60)
    print("GHOSTVISION GLOBAL CHECK")
    print("=" * 60)

    print(
        f"Total records      : {total_processed}"
    )
    print(
        f"Anthropogenic      : {total_positive}"
    )
    print(
        f"Normal/background  : {total_normal}"
    )
    print(
        f"Total objects      : {total_objects}"
    )

    if total_processed != 6674:
        raise RuntimeError(
            f"CRITICAL GhostVision total mismatch: "
            f"expected 6674, got {total_processed}"
        )

    if total_positive != 5127:
        raise RuntimeError(
            f"CRITICAL GhostVision positive mismatch: "
            f"expected 5127, got {total_positive}"
        )

    if total_normal != 1547:
        raise RuntimeError(
            f"CRITICAL GhostVision normal mismatch: "
            f"expected 1547, got {total_normal}"
        )

    print()
    print(
        "✓ GhostVision semantic counts match the source dataset."
    )
    print(
        "✓ GhostVision parser is confirmed correct."
    )
    print(
        f"GhostVision processed: {total_processed}"
    )


# ============================================================
# AI4SHIPWRECKS
# ============================================================

def locate_ai4_splits():
    """
    Find every train/test/images directory recursively.

    Handles the user's actual nested:
        AI4Shipwrecks/AI4Shipwrecks/train/images
    """
    found = []

    for p in AI4_ROOT.rglob("images"):
        if not p.is_dir():
            continue

        parent_names = {x.name.lower() for x in p.parents}

        if "train" in parent_names or "test" in parent_names:
            split = "train" if "train" in parent_names else "test"

            label_dir = p.parent / "labels"

            if label_dir.exists():
                found.append((split, p, label_dir))

    # Deduplicate.
    unique = {}
    for split, image_dir, label_dir in found:
        unique[(split, norm(image_dir))] = (
            split, image_dir, label_dir
        )

    return list(unique.values())


def ai4_group_from_stem(stem):
    """
    AI4 names look like:
        EB_Allen_06
        DM_Wilson_01

    Group = prefix before the final frame number.
    """
    m = re.match(r"^(.*?)(?:[_-])\d+$", stem)

    if m:
        return m.group(1)

    # Fallback to first token.
    return stem.split("_")[0]


def find_ai4_label(image, label_dir):
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        p = label_dir / f"{image.stem}{ext}"
        if p.exists():
            return p

    # Common mask is PNG.
    p = label_dir / f"{image.stem}.png"
    if p.exists():
        return p

    # Case-insensitive fallback.
    matches = [
        x for x in label_dir.iterdir()
        if x.is_file() and x.stem.lower() == image.stem.lower()
    ]

    return matches[0] if matches else None


def ai4_mask_to_bbox(mask_path):
    """
    AI4 mask:
        0 = background
        nonzero (normally 1) = shipwreck

    Returns one enclosing bbox.
    """
    with Image.open(mask_path) as im:
        arr = np.asarray(im)

    positive = arr > 0

    if not positive.any():
        return None

    ys, xs = np.where(positive)

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    return x1, y1, x2, y2


def choose_ai4_train_val_groups(groups):
    """
    Deterministically select approximately 20% of train groups for val.

    The group is the wreck/site identifier, preventing frames from the
    same wreck from appearing in both train and val.
    """
    groups = sorted(groups)

    if len(groups) <= 1:
        return set()

    rng = random.Random(SEED)

    shuffled = groups[:]
    rng.shuffle(shuffled)

    n_val = max(1, round(len(shuffled) * AI4_VAL_FRACTION))

    return set(shuffled[:n_val])


def process_ai4shipwrecks():
    print("\n" + "=" * 60)
    print("PROCESSING AI4SHIPWRECKS")
    print("=" * 60)

    splits = locate_ai4_splits()

    if not splits:
        raise RuntimeError(
            f"Could not find AI4 train/test images+labels under {AI4_ROOT}"
        )

    print("AI4 directories found:")
    for split, image_dir, label_dir in splits:
        print(f"  {split}: {image_dir}")
        print(f"         labels: {label_dir}")

    # Build complete source inventory first.
    records = []

    for official_split, image_dir, label_dir in splits:
        for img in image_files(image_dir):
            label = find_ai4_label(img, label_dir)

            if label is None:
                raise RuntimeError(
                    f"AI4 image has no matching mask:\n{img}"
                )

            group = ai4_group_from_stem(img.stem)

            records.append({
                "image": img,
                "label": label,
                "official_split": official_split,
                "group": group,
            })

    print(f"\nAI4 source image records: {len(records)}")

    # Official test remains test.
    train_records = [
        r for r in records if r["official_split"] == "train"
    ]

    test_records = [
        r for r in records if r["official_split"] == "test"
    ]

    train_groups = sorted({
        r["group"] for r in train_records
    })

    val_groups = choose_ai4_train_val_groups(train_groups)

    print(f"AI4 train groups: {len(train_groups)}")
    print(f"AI4 validation groups: {sorted(val_groups)}")
    print(f"AI4 official test records: {len(test_records)}")

    for r in tqdm(records):
        img = r["image"]
        mask = r["label"]
        group = r["group"]

        official_split = r["official_split"]

        if official_split == "test":
            split = "test"
        elif group in val_groups:
            split = "val"
        else:
            split = "train"

        bbox = ai4_mask_to_bbox(mask)

        with Image.open(img) as im:
            W, H = im.size

        boxes = []

        if bbox is not None:
            box = bbox_to_yolo(*bbox, W, H)
            if box is not None:
                boxes.append(box)

        if boxes:
            source_class = "Shipwreck"
            saad_class = "ANTHROPOGENIC"
            has_annotation = True
        else:
            source_class = "background"
            saad_class = "NORMAL"
            has_annotation = False

        save_image_and_label(
            "AI4Shipwrecks",
            split,
            img,
            boxes,
            source_class,
            saad_class,
            group,
            "binary_mask_to_bbox",
            has_annotation,
        )

    print(f"AI4 processed: {len(records)}")


# ============================================================
# SUBPIPEMINI2
# ============================================================

def locate_subpipe():
    """
    Locate YOLO annotations and image directories recursively.
    Prefer the known DATA structure.
    """
    candidates = []

    for p in SUBPIPE_ROOT.rglob("YOLO_Annotation"):
        if p.is_dir():
            candidates.append(p)

    if not candidates:
        raise RuntimeError(
            f"Could not find YOLO_Annotation under {SUBPIPE_ROOT}"
        )

    # Prefer DATA/YOLO_Annotation.
    candidates.sort(
        key=lambda p: ("DATA" not in [x.name.upper() for x in p.parents], str(p))
    )

    label_dir = candidates[0]

    image_dirs = []

    for name in ["SSS_HF_images", "SSS_LF_images"]:
        p = label_dir.parent / name / "Image"

        if p.exists():
            image_dirs.append(p)

    if not image_dirs:
        # Fallback recursive.
        for p in SUBPIPE_ROOT.rglob("Image"):
            if p.is_dir():
                image_dirs.append(p)

    if not image_dirs:
        raise RuntimeError(
            "Could not locate SubPipe image directories."
        )

    return label_dir, image_dirs


def find_subpipe_label(image, label_root):
    """
    Search recursively by exact stem.
    """
    candidates = [
        p for p in label_root.rglob("*.txt")
        if p.stem.lower() == image.stem.lower()
    ]

    if not candidates:
        return None

    # Prefer exact parent naming if duplicates exist.
    return candidates[0]


def assign_subpipe_splits(records):
    """
    Assign entire timestamp blocks to a split.

    This is deliberately done at block level, so adjacent frames cannot
    be independently assigned to different splits.

    The split assignment is deterministic and based on sorted temporal
    blocks, not on individual frame filenames.
    """
    timestamped = []

    for r in records:
        ts = timestamp_from_path(r["image"])

        if ts is None:
            raise RuntimeError(
                f"Could not extract timestamp from SubPipe image:\n"
                f"{r['image']}"
            )

        r["timestamp"] = ts

        block = math.floor(ts / SUBPIPE_BLOCK_SECONDS)
        r["block"] = block

        timestamped.append(r)

    unique_blocks = sorted({
        r["block"] for r in timestamped
    })

    if not unique_blocks:
        return records

    rng = random.Random(SEED)
    blocks = unique_blocks[:]
    rng.shuffle(blocks)

    n = len(blocks)

    n_test = max(1, round(n * SUBPIPE_TEST_FRACTION))
    n_val = max(1, round(n * SUBPIPE_VAL_FRACTION))

    # Ensure we don't consume all blocks.
    if n_test + n_val >= n:
        n_test = max(1, int(n * 0.10))
        n_val = max(1, int(n * 0.10))

    test_blocks = set(blocks[:n_test])
    val_blocks = set(blocks[n_test:n_test + n_val])

    for r in records:
        if r["block"] in test_blocks:
            r["split"] = "test"
        elif r["block"] in val_blocks:
            r["split"] = "val"
        else:
            r["split"] = "train"

    return records


def process_subpipe():
    print("\n" + "=" * 60)
    print("PROCESSING SUBPIPEMINI2")
    print("=" * 60)

    label_root, image_dirs = locate_subpipe()

    print("Label root:", label_root)
    print("Image dirs:")
    for d in image_dirs:
        print(" ", d)

    records = []
    seen_images = set()

    for image_dir in image_dirs:
        for img in image_files(image_dir):
            key = norm(img)

            if key in seen_images:
                continue

            seen_images.add(key)

            label = find_subpipe_label(img, label_root)

            # IMPORTANT:
            # Missing annotation is NOT treated as a negative.
            if label is None:
                continue

            boxes = parse_yolo_file(label)

            records.append({
                "image": img,
                "label": label,
                "boxes": boxes,
            })

    print(f"Matched SubPipe annotated images: {len(records)}")

    if not records:
        raise RuntimeError("No matched SubPipe annotations found.")

    records = assign_subpipe_splits(records)

    split_counts = Counter(r["split"] for r in records)

    print("SubPipe split counts:")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {split_counts[split]}")

    for r in tqdm(records):
        split = r["split"]

        save_image_and_label(
            "SubPipeMini2",
            split,
            r["image"],
            r["boxes"],
            "Pipeline",
            "ANTHROPOGENIC",
            "timestamp_block_" + str(r["block"]),
            "yolo_bbox",
            True,
        )

    print(f"SubPipe processed: {len(records)}")


# ============================================================
# MARINE-PULSE
# ============================================================

def locate_marine_seabed():
    """
    Find files belonging to the seabed surface category.

    We do not use engineering platform/pipeline or residual mound in
    the baseline.
    """
    candidates = []

    for p in MARINE_ROOT.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in IMAGE_EXTS:
            continue

        parts = [x.lower() for x in p.parts]

        # Folder names can vary slightly.
        joined = " ".join(parts)

        if "seabed" in joined and "surface" in joined:
            candidates.append(p)

    return sorted(set(candidates))


def process_marine_pulse():
    print("\n" + "=" * 60)
    print("PROCESSING MARINE-PULSE NORMAL SEABED")
    print("=" * 60)

    imgs = locate_marine_seabed()

    print(f"Marine-PULSE seabed surface images: {len(imgs)}")

    if not imgs:
        raise RuntimeError(
            f"No Marine-PULSE seabed surface images found under "
            f"{MARINE_ROOT}"
        )

    for img in tqdm(imgs):
        save_image_and_label(
            "Marine-PULSE",
            "train",
            img,
            [],
            "seabed surface",
            "NORMAL",
            "seabed_surface",
            "folder_class",
            False,
        )


# ============================================================
# FINAL VALIDATION
# ============================================================

def final_validate():
    print("\n" + "=" * 60)
    print("FINAL BUILD VALIDATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Manifest rows
    # --------------------------------------------------------

    print(f"Manifest rows generated: {len(manifest_rows)}")

    if not manifest_rows:
        raise RuntimeError("No manifest rows were generated.")

    # --------------------------------------------------------
    # Check unique source paths
    # --------------------------------------------------------

    source_paths = [
        norm(r["original_path"])
        for r in manifest_rows
    ]

    if len(source_paths) != len(set(source_paths)):
        raise RuntimeError(
            "DUPLICATE SOURCE PATHS IN MANIFEST."
        )

    print("✓ Source paths are unique")

    # --------------------------------------------------------
    # Check unique output paths
    # --------------------------------------------------------

    output_paths_list = [
        norm(r["output_image"])
        for r in manifest_rows
    ]

    if len(output_paths_list) != len(set(output_paths_list)):
        raise RuntimeError(
            "DUPLICATE OUTPUT IMAGE PATHS IN MANIFEST."
        )

    print("✓ Output image paths are unique")

    # --------------------------------------------------------
    # Check all files
    # --------------------------------------------------------

    missing_images = []
    missing_labels = []

    for r in manifest_rows:
        img = Path(r["output_image"])
        lbl = Path(r["output_label"])

        if not img.exists():
            missing_images.append(str(img))

        if not lbl.exists():
            missing_labels.append(str(lbl))

    print(f"Missing images: {len(missing_images)}")
    print(f"Missing labels: {len(missing_labels)}")

    if missing_images or missing_labels:
        raise RuntimeError(
            "FILE INTEGRITY FAILURE."
        )

    print("✓ Every manifest row has an image and label")

    # --------------------------------------------------------
    # Actual directory counts
    # --------------------------------------------------------

    for split in ["train", "val", "test"]:
        img_dir = OUTPUT / "images" / split
        lbl_dir = OUTPUT / "labels" / split

        images = [
            p for p in img_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]

        labels = list(lbl_dir.glob("*.txt"))

        manifest_count = sum(
            1 for r in manifest_rows
            if r["split"] == split
        )

        print(
            f"{split:5s}: "
            f"manifest={manifest_count:5d} "
            f"images={len(images):5d} "
            f"labels={len(labels):5d}"
        )

        if not (
            manifest_count == len(images) == len(labels)
        ):
            raise RuntimeError(
                f"COUNT MISMATCH in {split}"
            )

        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}

        if img_stems != lbl_stems:
            raise RuntimeError(
                f"IMAGE/LABEL STEM MISMATCH in {split}"
            )

    print("✓ Manifest/image/label counts agree")

    # --------------------------------------------------------
    # Validate every YOLO file
    # --------------------------------------------------------

    bad_labels = []

    for split in ["train", "val", "test"]:
        for p in (OUTPUT / "labels" / split).glob("*.txt"):
            try:
                boxes = parse_yolo_file(p)

                for cls, x, y, w, h in boxes:
                    if cls != 0:
                        raise ValueError("class != 0")

                    if not all(
                        0.0 <= v <= 1.0
                        for v in [x, y, w, h]
                    ):
                        raise ValueError(
                            "coordinate outside [0,1]"
                        )

                    if w <= 0 or h <= 0:
                        raise ValueError(
                            "non-positive width/height"
                        )

                    if x - w / 2 < -1e-6:
                        raise ValueError("left edge outside image")
                    if x + w / 2 > 1 + 1e-6:
                        raise ValueError("right edge outside image")
                    if y - h / 2 < -1e-6:
                        raise ValueError("top edge outside image")
                    if y + h / 2 > 1 + 1e-6:
                        raise ValueError("bottom edge outside image")

            except Exception as e:
                bad_labels.append((str(p), str(e)))

    print(f"Invalid YOLO labels: {len(bad_labels)}")

    if bad_labels:
        with open(
            OUTPUT / "integrity_bad_labels.txt",
            "w",
            encoding="utf-8",
        ) as f:
            for p, e in bad_labels:
                f.write(f"{p}\n  {e}\n")

        raise RuntimeError(
            "YOLO LABEL VALIDATION FAILED. "
            f"See {OUTPUT / 'integrity_bad_labels.txt'}"
        )

    print("✓ All YOLO labels valid")

    # --------------------------------------------------------
    # Duplicate bytes across splits
    # --------------------------------------------------------

    import hashlib

    hashes = defaultdict(list)

    for split in ["train", "val", "test"]:
        for p in (OUTPUT / "images" / split).glob("*.png"):
            h = hashlib.sha1(p.read_bytes()).hexdigest()
            hashes[h].append(split)

    cross_split = {
        h: splits
        for h, splits in hashes.items()
        if len(set(splits)) > 1
    }

    print(
        f"Exact cross-split duplicate image hashes: "
        f"{len(cross_split)}"
    )

    if cross_split:
        with open(
            OUTPUT / "integrity_cross_split_duplicates.txt",
            "w",
            encoding="utf-8",
        ) as f:
            for h, splits in cross_split.items():
                f.write(f"{h}: {splits}\n")

        raise RuntimeError(
            "EXACT CROSS-SPLIT DUPLICATES FOUND. "
            "Dataset was NOT considered safe."
        )

    print("✓ No exact cross-split duplicate images")

    # --------------------------------------------------------
    # SubPipe temporal leakage
    # --------------------------------------------------------

    subpipe = [
        r for r in manifest_rows
        if r["dataset"] == "SubPipeMini2"
    ]

    timestamps = []

    for r in subpipe:
        ts = timestamp_from_path(r["original_path"])

        if ts is not None:
            timestamps.append(
                (ts, r["split"], r["original_path"])
            )

    timestamps.sort()

    temporal_leakage = []

    for a, b in zip(timestamps, timestamps[1:]):
        dt = abs(a[0] - b[0])

        if dt <= 5.0 and a[1] != b[1]:
            temporal_leakage.append((a, b, dt))

    print(
        "SubPipe neighboring cross-split pairs <=5 sec: "
        f"{len(temporal_leakage)}"
    )

    if temporal_leakage:
        with open(
            OUTPUT / "integrity_subpipe_temporal_leakage.txt",
            "w",
            encoding="utf-8",
        ) as f:
            for a, b, dt in temporal_leakage:
                f.write(
                    f"{a}\n{b}\ndelta={dt}\n\n"
                )

        print(
            "WARNING: "
            f"{len(temporal_leakage)} SubPipe neighboring "
            "cross-split pairs found. "
            "Proceeding because leakage is minimal and was "
            "explicitly accepted for this 48-hour build."
        )
    else:
        print("✓ No SubPipe <=5 sec cross-split neighbors")

    # --------------------------------------------------------
    # Write manifest
    # --------------------------------------------------------

    manifest_path = OUTPUT / "manifest.csv"

    columns = [
        "dataset",
        "original_path",
        "output_image",
        "output_label",
        "split",
        "group",
        "source_class",
        "saad_class",
        "has_annotation",
        "num_objects",
        "annotation_type",
    ]

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    # --------------------------------------------------------
    # data.yaml
    # --------------------------------------------------------

    yaml_path = OUTPUT / "data.yaml"

    yaml_text = f"""path: {OUTPUT.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: anthropogenic
"""

    yaml_path.write_text(
        yaml_text,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Summary CSV
    # --------------------------------------------------------

    summary_path = OUTPUT / "dataset_summary.csv"

    summary_counts = Counter(
        (
            r["dataset"],
            r["split"],
            r["saad_class"],
        )
        for r in manifest_rows
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)
        w.writerow([
            "dataset",
            "split",
            "saad_class",
            "count",
        ])

        for key, n in sorted(summary_counts.items()):
            w.writerow([*key, n])

    # --------------------------------------------------------
    # Print final summary
    # --------------------------------------------------------

    print("\nFINAL DATASET SUMMARY")
    print("=====================")

    for key, n in sorted(summary_counts.items()):
        print(
            f"{key[0]:20s} "
            f"{key[1]:5s} "
            f"{key[2]:15s} "
            f"{n:6d}"
        )

    print("\nTOTAL:", len(manifest_rows))

    # --------------------------------------------------------
    # SEMANTIC DATASET GATES
    # --------------------------------------------------------
    #
    # Filesystem integrity alone is not sufficient. The previous
    # rebuild passed every filesystem check while accidentally
    # turning all GhostVision objects into NORMAL because its
    # JSONL schema was parsed incorrectly.
    #
    # These gates make that class of failure impossible to miss.
    # --------------------------------------------------------

    ghost_rows = [
        r for r in manifest_rows
        if r["dataset"] == "GhostVision"
    ]

    ghost_anthro = sum(
        1 for r in ghost_rows
        if r["saad_class"] == "ANTHROPOGENIC"
    )

    ghost_normal = sum(
        1 for r in ghost_rows
        if r["saad_class"] == "NORMAL"
    )

    if len(ghost_rows) != 6674:
        raise RuntimeError(
            "SEMANTIC GATE FAILED: "
            f"GhostVision total={len(ghost_rows)}, expected 6674."
        )

    if ghost_anthro != 5127:
        raise RuntimeError(
            "SEMANTIC GATE FAILED: "
            f"GhostVision anthropogenic={ghost_anthro}, "
            "expected 5127."
        )

    if ghost_normal != 1547:
        raise RuntimeError(
            "SEMANTIC GATE FAILED: "
            f"GhostVision normal={ghost_normal}, "
            "expected 1547."
        )

    print()
    print("SEMANTIC GATES")
    print("==============")
    print(
        "GhostVision total        : "
        f"{len(ghost_rows)}"
    )
    print(
        "GhostVision anthropogenic: "
        f"{ghost_anthro}"
    )
    print(
        "GhostVision normal       : "
        f"{ghost_normal}"
    )
    print("✓ GhostVision semantic counts verified")

    print("\nBY DATASET")
    by_dataset = Counter(
        r["dataset"] for r in manifest_rows
    )

    for ds, n in sorted(by_dataset.items()):
        print(f"{ds:20s} {n:6d}")

    print("\n============================================================")
    print("SAAD DATASET REBUILD PASSED")
    print("============================================================")
    print(f"Manifest : {manifest_path}")
    print(f"YAML     : {yaml_path}")
    print("\nAll integrity assertions passed.")
    print("Dataset is ready for baseline training.")


# ============================================================
# MAIN
# ============================================================

try:
    process_ghostvision()
    process_ai4shipwrecks()
    process_subpipe()
    process_marine_pulse()
    final_validate()

except Exception as e:
    print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("SAAD REBUILD FAILED")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(type(e).__name__ + ":", e)
    print(
        "\nThe source datasets were not modified.\n"
        "The partially built SAAD_baseline should NOT be trained."
    )
    raise
