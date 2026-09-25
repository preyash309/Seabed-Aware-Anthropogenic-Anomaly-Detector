"""
SAAD-v1 Dataset Audit
=====================
Read-only audit for the current SIH sonar datasets.

Expected root:
    E:\SIH\Datasets

The script DOES NOT modify, rename, convert, resize, or copy the original datasets.
It only reads them and writes an "audit" folder next to the script.

Audits:
    - GhostVision JSONL + images
    - AI4Shipwrecks images + binary masks
    - SubPipeMini2 PBM/images + YOLO annotations
    - Marine-PULSE category folders
    - SeabedObjects-KLSG (generic recursive image audit)

Outputs:
    audit/
      dataset_summary.csv
      annotation_errors.csv
      image_records.csv
      class_distribution.png
      resolution_distribution.png
      visual_samples/
          <dataset>_samples.png
"""

from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import math
import random
import re
import sys

try:
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.pyplot as plt
except ImportError as e:
    print("\nMissing dependency:", e)
    print("Install with:")
    print("    pip install numpy pandas pillow matplotlib")
    sys.exit(1)


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(r"E:\SIH\Datasets")

# Change this if you want more/less examples.
SAMPLES_PER_DATASET = 12

# Recursive image extensions.
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
    ".pbm", ".pgm", ".ppm", ".webp"
}

YOLO_EXTS = {".txt"}

SEED = 42
random.seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def find_dataset_dir(keyword: str):
    """Find the first top-level directory whose name contains keyword."""
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {DATASET_ROOT}")

    matches = [
        p for p in DATASET_ROOT.iterdir()
        if p.is_dir() and keyword.lower() in p.name.lower()
    ]

    if not matches:
        return None

    # Prefer exact-ish names when possible.
    matches.sort(key=lambda p: (len(p.name), p.name.lower()))
    return matches[0]


def find_all_images(root: Path):
    if root is None or not root.exists():
        return []
    return [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def image_info(path: Path):
    try:
        with Image.open(path) as im:
            return {
                "width": im.width,
                "height": im.height,
                "mode": im.mode,
                "format": im.format,
                "ok": True,
                "error": ""
            }
    except Exception as e:
        return {
            "width": None,
            "height": None,
            "mode": None,
            "format": None,
            "ok": False,
            "error": str(e)
        }


def stem_key(path: Path):
    """
    Robust-ish matching key.
    For normal files, use stem.
    """
    return path.stem.lower()


def make_record(dataset, path, source_class="", saad_class="",
                annotation_type="", annotation_path="",
                has_object=None, num_objects=None,
                annotation_valid=True, error=""):
    info = image_info(path)

    return {
        "dataset": dataset,
        "image_path": str(path),
        "image_relative_path": safe_rel(path, DATASET_ROOT),
        "annotation_path": str(annotation_path) if annotation_path else "",
        "source_class": source_class,
        "saad_class": saad_class,
        "annotation_type": annotation_type,
        "width": info["width"],
        "height": info["height"],
        "mode": info["mode"],
        "format": info["format"],
        "has_object": has_object,
        "num_objects": num_objects,
        "image_read_ok": info["ok"],
        "annotation_valid": annotation_valid,
        "error": error or info["error"],
    }


def read_jsonl(path: Path):
    rows = []
    errors = []

    try:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append((line_no, json.loads(line)))
                except Exception as e:
                    errors.append((line_no, str(e)))
    except Exception as e:
        errors.append((0, str(e)))

    return rows, errors


def valid_yolo_line(parts):
    if len(parts) < 5:
        return False
    try:
        cls = int(float(parts[0]))
        vals = [float(x) for x in parts[1:5]]
        if cls < 0:
            return False
        return all(0 <= x <= 1 for x in vals) and vals[2] > 0 and vals[3] > 0
    except Exception:
        return False


def read_yolo(path: Path):
    boxes = []
    errors = []

    try:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if not valid_yolo_line(parts):
                    errors.append(f"line {line_no}: invalid YOLO line: {line}")
                    continue

                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
                boxes.append((cls, x, y, w, h))
    except Exception as e:
        errors.append(str(e))

    return boxes, errors


def yolo_to_pixels(box, width, height):
    cls, xc, yc, bw, bh = box

    x1 = int(round((xc - bw / 2) * width))
    y1 = int(round((yc - bh / 2) * height))
    x2 = int(round((xc + bw / 2) * width))
    y2 = int(round((yc + bh / 2) * height))

    return cls, x1, y1, x2, y2


def mask_stats(path: Path):
    """
    AI4Shipwrecks masks are expected to contain 0 and 1.
    Some viewers render 1 as nearly black, so we explicitly inspect
    numerical values instead of visual appearance.
    """
    try:
        arr = np.array(Image.open(path))
        unique, counts = np.unique(arr, return_counts=True)
        stats = dict(zip(unique.tolist(), counts.tolist()))

        positive = int(np.count_nonzero(arr == 1))
        # Also detect 255 in case a mask was exported that way.
        positive_255 = int(np.count_nonzero(arr == 255))

        return {
            "ok": True,
            "unique_values": stats,
            "positive_1": positive,
            "positive_255": positive_255,
            "positive_any_nonzero": int(np.count_nonzero(arr)),
            "shape": arr.shape,
            "error": ""
        }
    except Exception as e:
        return {
            "ok": False,
            "unique_values": {},
            "positive_1": 0,
            "positive_255": 0,
            "positive_any_nonzero": 0,
            "shape": None,
            "error": str(e)
        }


# ============================================================
# GHOSTVISION
# ============================================================

def audit_ghostvision(root: Path):
    records = []
    errors = []
    visual_items = []

    if root is None:
        return records, errors, visual_items

    jsonls = list(root.rglob("*.jsonl"))

    if not jsonls:
        errors.append({
            "dataset": "GhostVision",
            "type": "missing_metadata",
            "path": str(root),
            "message": "No JSONL metadata file found."
        })
        return records, errors, visual_items

    image_by_name = {}
    for p in find_all_images(root):
        image_by_name[p.name] = p
        image_by_name[p.name.lower()] = p

    for jsonl in jsonls:
        rows, json_errors = read_jsonl(jsonl)

        for line_no, err in json_errors:
            errors.append({
                "dataset": "GhostVision",
                "type": "jsonl_error",
                "path": str(jsonl),
                "message": f"line {line_no}: {err}"
            })

        for line_no, row in rows:
            fname = row.get("file_name", "")
            obj = row.get("objects", {}) or {}

            bboxes = obj.get("bbox", []) or []
            cats = obj.get("category", []) or []

            # Match by basename first.
            img = image_by_name.get(fname)
            if img is None:
                img = image_by_name.get(Path(fname).name)
            if img is None:
                img = image_by_name.get(Path(fname).name.lower())

            if img is None:
                errors.append({
                    "dataset": "GhostVision",
                    "type": "missing_image",
                    "path": str(jsonl),
                    "message": f"line {line_no}: image not found: {fname}"
                })
                continue

            info = image_info(img)
            valid = True
            messages = []

            if len(bboxes) != len(cats):
                valid = False
                messages.append(
                    f"bbox/category count mismatch ({len(bboxes)} vs {len(cats)})"
                )

            for i, bbox in enumerate(bboxes):
                try:
                    x, y, w, h = map(float, bbox)
                    if info["width"] and info["height"]:
                        if x < 0 or y < 0 or w <= 0 or h <= 0:
                            valid = False
                            messages.append(f"invalid bbox {i}: {bbox}")
                        if x + w > info["width"] + 1 or y + h > info["height"] + 1:
                            valid = False
                            messages.append(f"bbox outside image {i}: {bbox}")
                except Exception:
                    valid = False
                    messages.append(f"invalid bbox {i}: {bbox}")

            source_classes = ",".join(map(str, cats)) if cats else "none"
            saad = "anthropogenic" if bboxes else "background"

            records.append(make_record(
                "GhostVision", img,
                source_class=source_classes,
                saad_class=saad,
                annotation_type="bbox_jsonl",
                annotation_path=jsonl,
                has_object=bool(bboxes),
                num_objects=len(bboxes),
                annotation_valid=valid,
                error="; ".join(messages)
            ))

            if len(visual_items) < SAMPLES_PER_DATASET and img:
                visual_items.append({
                    "image": img,
                    "type": "ghostvision",
                    "boxes": bboxes,
                    "labels": cats
                })

    return records, errors, visual_items


# ============================================================
# AI4SHIPWRECKS
# ============================================================

def audit_ai4shipwrecks(root: Path):
    records = []
    errors = []
    visual_items = []

    if root is None:
        return records, errors, visual_items

    # Expected:
    # train/images, train/labels
    # test/images, test/labels
    image_dirs = [
        p for p in root.rglob("images")
        if p.is_dir()
    ]

    for img_dir in image_dirs:
        split = img_dir.parent.name.lower()
        label_dir = img_dir.parent / "labels"

        if not label_dir.exists():
            errors.append({
                "dataset": "AI4Shipwrecks",
                "type": "missing_label_dir",
                "path": str(img_dir),
                "message": f"Expected {label_dir}"
            })
            continue

        label_by_name = {p.name.lower(): p for p in find_all_images(label_dir)}

        for img in find_all_images(img_dir):
            label = label_by_name.get(img.name.lower())

            if label is None:
                errors.append({
                    "dataset": "AI4Shipwrecks",
                    "type": "missing_label",
                    "path": str(img),
                    "message": f"No matching label for {img.name}"
                })
                continue

            im_info = image_info(img)
            lab_info = image_info(label)

            valid = True
            messages = []

            if (
                im_info["width"] != lab_info["width"]
                or im_info["height"] != lab_info["height"]
            ):
                valid = False
                messages.append(
                    f"image/mask size mismatch: "
                    f"{im_info['width']}x{im_info['height']} vs "
                    f"{lab_info['width']}x{lab_info['height']}"
                )

            stats = mask_stats(label)

            # AI4Shipwrecks uses 0 / 1.
            positive = stats["positive_1"] > 0
            unique_vals = stats["unique_values"]

            unexpected = [
                v for v in unique_vals
                if v not in (0, 1)
            ]

            if unexpected:
                messages.append(
                    f"unexpected mask values: {unexpected[:20]}"
                )

            records.append(make_record(
                "AI4Shipwrecks",
                img,
                source_class="Shipwreck" if positive else "none",
                saad_class="anthropogenic" if positive else "background",
                annotation_type="binary_mask",
                annotation_path=label,
                has_object=positive,
                num_objects=1 if positive else 0,
                annotation_valid=valid,
                error="; ".join(messages)
            ))

            if len(visual_items) < SAMPLES_PER_DATASET:
                visual_items.append({
                    "image": img,
                    "type": "ai4shipwrecks",
                    "mask": label
                })

    return records, errors, visual_items


# ============================================================
# SUBPIPE
# ============================================================

def audit_subpipe(root: Path):
    records = []
    errors = []
    visual_items = []

    if root is None:
        return records, errors, visual_items

    image_dirs = [
        p for p in root.rglob("Image")
        if p.is_dir()
    ]

    yolo_dirs = [
        p for p in root.rglob("YOLO_Annotation")
        if p.is_dir()
    ]

    # Build annotation lookup by stem.
    yolo_by_stem = {}
    for ydir in yolo_dirs:
        for txt in ydir.rglob("*.txt"):
            yolo_by_stem[txt.stem.lower()] = txt

    if not yolo_by_stem:
        errors.append({
            "dataset": "SubPipeMini2",
            "type": "missing_yolo",
            "path": str(root),
            "message": "No YOLO .txt annotations found."
        })

    for img_dir in image_dirs:
        for img in find_all_images(img_dir):
            txt = yolo_by_stem.get(img.stem.lower())

            if txt is None:
                errors.append({
                    "dataset": "SubPipeMini2",
                    "type": "missing_annotation",
                    "path": str(img),
                    "message": f"No YOLO annotation found for {img.name}"
                })
                continue

            boxes, yolo_errors = read_yolo(txt)
            valid = len(yolo_errors) == 0
            messages = list(yolo_errors)

            info = image_info(img)

            for box in boxes:
                _, x1, y1, x2, y2 = yolo_to_pixels(
                    box, info["width"], info["height"]
                )
                if x1 < 0 or y1 < 0 or x2 > info["width"] or y2 > info["height"]:
                    valid = False
                    messages.append(
                        f"bbox outside image after conversion: {box}"
                    )

            records.append(make_record(
                "SubPipeMini2",
                img,
                source_class="Pipeline" if boxes else "none",
                saad_class="anthropogenic" if boxes else "background",
                annotation_type="yolo_bbox",
                annotation_path=txt,
                has_object=bool(boxes),
                num_objects=len(boxes),
                annotation_valid=valid,
                error="; ".join(messages)
            ))

            if len(visual_items) < SAMPLES_PER_DATASET:
                visual_items.append({
                    "image": img,
                    "type": "subpipe",
                    "boxes": boxes
                })

    return records, errors, visual_items


# ============================================================
# MARINE-PULSE
# ============================================================

def audit_marine_pulse(root: Path):
    records = []
    errors = []
    visual_items = []

    if root is None:
        return records, errors, visual_items

    # Marine-PULSE is organized by category folders.
    category_dirs = [
        p for p in root.rglob("*")
        if p.is_dir() and p.name.lower() in {
            "engineering platform",
            "pipeline or cable",
            "seabed surface",
            "underwater residual mound"
        }
    ]

    # If the exact names are not found, fall back to immediate subdirectories
    # under train/valid/test.
    if not category_dirs:
        for split in ["train", "valid", "val", "test"]:
            d = root / split
            if d.exists():
                category_dirs.extend([p for p in d.iterdir() if p.is_dir()])

    seen = set()

    for cdir in category_dirs:
        category = cdir.name
        for img in find_all_images(cdir):
            if img in seen:
                continue
            seen.add(img)

            low = category.lower()
            if "seabed" in low:
                saad = "background"
            elif "residual" in low:
                saad = "ambiguous"
            elif "pipeline" in low or "engineering" in low:
                saad = "anthropogenic"
            else:
                saad = "unknown"

            records.append(make_record(
                "Marine-PULSE",
                img,
                source_class=category,
                saad_class=saad,
                annotation_type="folder_class",
                annotation_path=cdir,
                has_object=(saad == "anthropogenic"),
                num_objects=None,
                annotation_valid=True,
                error=""
            ))

            if len(visual_items) < SAMPLES_PER_DATASET:
                visual_items.append({
                    "image": img,
                    "type": "plain",
                    "label": category
                })

    if not records:
        errors.append({
            "dataset": "Marine-PULSE",
            "type": "no_images_found",
            "path": str(root),
            "message": "No category images found."
        })

    return records, errors, visual_items


# ============================================================
# KLSG / GENERIC
# ============================================================

def audit_klsg(root: Path):
    records = []
    errors = []
    visual_items = []

    if root is None:
        return records, errors, visual_items

    images = find_all_images(root)

    for img in images:
        # Try to infer a source class from parent folder.
        parent = img.parent.name

        records.append(make_record(
            "SeabedObjects-KLSG",
            img,
            source_class=parent,
            saad_class="unassigned",
            annotation_type="unknown",
            annotation_path="",
            has_object=None,
            num_objects=None,
            annotation_valid=True,
            error=""
        ))

    # Random samples instead of first files.
    if images:
        sample = random.sample(images, min(SAMPLES_PER_DATASET, len(images)))
        for img in sample:
            visual_items.append({
                "image": img,
                "type": "plain",
                "label": img.parent.name
            })

    if not records:
        errors.append({
            "dataset": "SeabedObjects-KLSG",
            "type": "no_images_found",
            "path": str(root),
            "message": "No images found."
        })

    return records, errors, visual_items


# ============================================================
# VISUALIZATION
# ============================================================

def normalize_for_display(arr):
    arr = np.asarray(arr)

    if arr.ndim == 3:
        arr = arr[..., 0]

    arr = arr.astype(np.float32)

    lo = np.percentile(arr, 1)
    hi = np.percentile(arr, 99)

    if hi <= lo:
        lo = arr.min()
        hi = arr.max()

    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)

    arr = (arr - lo) / (hi - lo)
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)


def load_display_image(path: Path, max_side=700):
    with Image.open(path) as im:
        im = im.convert("L")
        arr = np.array(im)

    arr = normalize_for_display(arr)
    out = Image.fromarray(arr, mode="L")

    scale = min(1.0, max_side / max(out.width, out.height))
    if scale < 1:
        out = out.resize(
            (max(1, int(out.width * scale)),
             max(1, int(out.height * scale))),
            Image.Resampling.LANCZOS
        )

    return out


def draw_box_overlay(im, box, label=""):
    draw = ImageDraw.Draw(im)
    cls, x1, y1, x2, y2 = box

    draw.rectangle([x1, y1, x2, y2], outline=255, width=3)

    if label:
        # Simple readable label background.
        try:
            bbox = draw.textbbox((0, 0), label)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = 60, 15

        draw.rectangle([x1, max(0, y1 - th - 4), x1 + tw + 6, y1], fill=255)
        draw.text((x1 + 3, max(0, y1 - th - 2)), label, fill=0)


def create_visual_grid(items, output_path: Path, title: str):
    if not items:
        return

    cols = 3
    rows = math.ceil(len(items) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, item in zip(axes, items):
        path = item["image"]

        try:
            with Image.open(path) as raw:
                raw = raw.convert("L")
                raw_arr = np.array(raw)

            disp = normalize_for_display(raw_arr)
            ax.imshow(disp, cmap="gray", aspect="auto")

            typ = item["type"]

            if typ == "ghostvision":
                for i, bbox in enumerate(item["boxes"]):
                    x, y, w, h = map(float, bbox)
                    rect = plt.Rectangle(
                        (x, y), w, h,
                        fill=False,
                        linewidth=2
                    )
                    ax.add_patch(rect)
                    if i < len(item["labels"]):
                        ax.text(
                            x, max(0, y - 4),
                            str(item["labels"][i]),
                            fontsize=8,
                            bbox=dict(facecolor="white", alpha=0.7)
                        )

            elif typ == "subpipe":
                h, w = raw_arr.shape[:2]
                for box in item["boxes"]:
                    cls, x1, y1, x2, y2 = yolo_to_pixels(box, w, h)
                    rect = plt.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        fill=False,
                        linewidth=2
                    )
                    ax.add_patch(rect)
                    ax.text(
                        x1, max(0, y1 - 4),
                        f"Pipeline (class {cls})",
                        fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.7)
                    )

            elif typ == "ai4shipwrecks":
                with Image.open(item["mask"]) as m:
                    mask = np.array(m)
                # Display positive mask as transparent red-ish overlay.
                # This is visualization only; originals are untouched.
                positive = mask == 1
                if positive.any():
                    overlay = np.zeros((*positive.shape, 4), dtype=np.float32)
                    overlay[..., 0] = 1.0
                    overlay[..., 3] = positive.astype(np.float32) * 0.45
                    ax.imshow(overlay, aspect="auto")

            elif typ == "plain":
                pass

            ax.set_title(
                f"{item.get('label', '')}\n{path.name}",
                fontsize=9
            )

        except Exception as e:
            ax.text(
                0.5, 0.5,
                f"Could not display\n{path.name}\n{e}",
                ha="center",
                va="center"
            )

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# PLOTS
# ============================================================

def make_plots(df: pd.DataFrame, outdir: Path):
    if df.empty:
        return

    # Class distribution.
    dist = (
        df.groupby(["dataset", "saad_class"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )

    if not dist.empty:
        ax = dist.plot(kind="bar", figsize=(12, 6))
        ax.set_title("SAAD class distribution inferred from audit")
        ax.set_ylabel("Images")
        ax.set_xlabel("Dataset")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / "class_distribution.png", dpi=150)
        plt.close()

    # Resolution distribution.
    res = (
        df.dropna(subset=["width", "height"])
        .copy()
    )

    if not res.empty:
        res["resolution"] = (
            res["width"].astype(int).astype(str)
            + "x"
            + res["height"].astype(int).astype(str)
        )

        top_res = res["resolution"].value_counts().head(20)

        ax = top_res.plot(kind="bar", figsize=(12, 6))
        ax.set_title("Top image resolutions")
        ax.set_ylabel("Images")
        ax.set_xlabel("Resolution")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / "resolution_distribution.png", dpi=150)
        plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("SAAD-v1 DATASET AUDIT")
    print("=" * 70)
    print(f"Dataset root: {DATASET_ROOT}")

    if not DATASET_ROOT.exists():
        print("\nERROR: Dataset root does not exist.")
        print("Change DATASET_ROOT near the top of this script.")
        sys.exit(1)

    outdir = Path(__file__).resolve().parent / "audit"
    sample_dir = outdir / "visual_samples"
    outdir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    all_errors = []

    # Locate current datasets.
    roots = {
        "AI4Shipwrecks": find_dataset_dir("AI4Shipwrecks"),
        "Marine-PULSE": find_dataset_dir("Marine_PULSE"),
        "SeabedObjects-KLSG": find_dataset_dir("SeabedObjects"),
        "GhostVision": find_dataset_dir("sss-crab-pot"),
        "SubPipeMini2": find_dataset_dir("SubPipeMini2"),
    }

    print("\nLocated datasets:")
    for name, root in roots.items():
        print(f"  {name:22s}: {root if root else 'NOT FOUND'}")

    # Audit each dataset.
    audit_functions = [
        ("GhostVision", audit_ghostvision),
        ("AI4Shipwrecks", audit_ai4shipwrecks),
        ("SubPipeMini2", audit_subpipe),
        ("Marine-PULSE", audit_marine_pulse),
        ("SeabedObjects-KLSG", audit_klsg),
    ]

    for name, fn in audit_functions:
        root = roots[name]

        if root is None:
            all_errors.append({
                "dataset": name,
                "type": "dataset_not_found",
                "path": "",
                "message": "Could not locate dataset under DATASET_ROOT."
            })
            continue

        print(f"\nAuditing {name} ...")

        records, errors, visual_items = fn(root)

        all_records.extend(records)
        all_errors.extend(errors)

        print(f"  images/records: {len(records):,}")
        print(f"  errors:        {len(errors):,}")

        # For KLSG, samples are already random.
        # For others, randomize the collected samples.
        random.shuffle(visual_items)
        visual_items = visual_items[:SAMPLES_PER_DATASET]

        if visual_items:
            create_visual_grid(
                visual_items,
                sample_dir / f"{name.replace('-', '_')}_samples.png",
                f"{name} — audit samples"
            )

    # Save records.
    df = pd.DataFrame(all_records)

    if not df.empty:
        df.to_csv(
            outdir / "image_records.csv",
            index=False,
            encoding="utf-8-sig"
        )

    err_df = pd.DataFrame(all_errors)
    if err_df.empty:
        err_df = pd.DataFrame(
            columns=["dataset", "type", "path", "message"]
        )

    err_df.to_csv(
        outdir / "annotation_errors.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Summary.
    summary_rows = []

    if not df.empty:
        for dataset, g in df.groupby("dataset"):
            summary_rows.append({
                "dataset": dataset,
                "images_records": len(g),
                "read_ok": int(g["image_read_ok"].fillna(False).sum()),
                "annotation_valid": int(g["annotation_valid"].fillna(False).sum()),
                "annotation_errors": int((~g["annotation_valid"].fillna(False)).sum()),
                "positive_anthropogenic": int(
                    (g["saad_class"] == "anthropogenic").sum()
                ),
                "background": int(
                    (g["saad_class"] == "background").sum()
                ),
                "ambiguous": int(
                    (g["saad_class"] == "ambiguous").sum()
                ),
                "unassigned": int(
                    (g["saad_class"] == "unassigned").sum()
                ),
                "min_width": g["width"].min(),
                "max_width": g["width"].max(),
                "min_height": g["height"].min(),
                "max_height": g["height"].max(),
                "annotation_types": "; ".join(
                    sorted(
                        str(x) for x in g["annotation_type"].dropna().unique()
                    )
                ),
            })

    # Add missing datasets to summary.
    known = {r["dataset"] for r in summary_rows}
    for name in roots:
        if name not in known:
            summary_rows.append({
                "dataset": name,
                "images_records": 0,
                "read_ok": 0,
                "annotation_valid": 0,
                "annotation_errors": 0,
                "positive_anthropogenic": 0,
                "background": 0,
                "ambiguous": 0,
                "unassigned": 0,
                "min_width": None,
                "max_width": None,
                "min_height": None,
                "max_height": None,
                "annotation_types": "",
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        outdir / "dataset_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    make_plots(df, outdir)

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print(f"\nOutput folder:")
    print(f"  {outdir}")

    print("\nFiles:")
    for p in sorted(outdir.iterdir()):
        if p.is_file():
            print(f"  {p.name}")

    print("\nVisual samples:")
    for p in sorted(sample_dir.glob("*.png")):
        print(f"  {p.name}")

    print("\nDataset summary:")
    if not summary_df.empty:
        print(
            summary_df[
                [
                    "dataset",
                    "images_records",
                    "positive_anthropogenic",
                    "background",
                    "ambiguous",
                    "unassigned",
                    "annotation_errors"
                ]
            ].to_string(index=False)
        )

    print("\nIMPORTANT:")
    print("This audit does NOT include Seafloor Sediments yet.")
    print("That is intentional. We will decide the normal-seabed sampling")
    print("strategy after inspecting these results.")
    print("\nNo original dataset files were modified.")


if __name__ == "__main__":
    main()
