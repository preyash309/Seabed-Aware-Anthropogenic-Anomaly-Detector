from pathlib import Path
import json
import sys
from PIL import Image

from safety import configured_path, dry_run, require_new_output, SAFE_OUTPUT_ROOT


# ============================================================
# GHOSTVISION -> SAAD CONVERTER / VALIDATOR
# ============================================================
#
# GhostVision structure:
#
# E:\SIH\Datasets\sss-crab-pot-detection-ds\
#     train\
#         *.jpg
#         metadata.jsonl
#     valid\
#         *.jpg
#         metadata.jsonl
#     test\
#         *.jpg
#         metadata.jsonl
#
#
# GhostVision metadata:
#
# {
#   "file_name": "...jpg",
#   "objects": {
#       "bbox": [[x, y, width, height], ...],
#       "category": ["Crab-Pot", ...],
#       "area": [...]
#   }
# }
#
#
# SAAD ontology:
#
#   0 = anthropogenic
#
# Crab-Pot and Maybe-Crab-Pot are both mapped to:
#
#   0 = anthropogenic
#
# Empty bbox = NORMAL / background
#
# ============================================================


# ------------------------------------------------------------
# CHANGE ONLY THIS IF YOUR DATASET IS SOMEWHERE ELSE
# ------------------------------------------------------------

GHOSTVISION_ROOT = configured_path("SAAD_GHOSTVISION_SOURCE", Path(r"E:\SIH\Datasets\sss-crab-pot-detection-ds"))


# Where the repaired YOLO labels will be written.
# This DOES NOT modify your original GhostVision dataset.
OUTPUT_ROOT = configured_path("SAAD_RESEARCH_OUTPUT_DIR", SAFE_OUTPUT_ROOT / "GhostVision_fixed")


SPLITS = ["train", "valid", "test"]


# ============================================================
# HELPERS
# ============================================================

def find_image(split_dir, filename):
    """
    Find an image inside the appropriate GhostVision split.

    First try the expected direct path.
    Then recursively search within ONLY that split.
    """

    direct = split_dir / filename

    if direct.exists():
        return direct

    matches = list(split_dir.rglob(filename))

    if len(matches) == 0:
        return None

    if len(matches) > 1:
        print(
            f"WARNING: multiple matches for {filename}"
        )
        print("Using:", matches[0])

    return matches[0]


def convert_bbox_to_yolo(bbox, img_w, img_h):
    """
    GhostVision:
        [x, y, width, height]

    YOLO:
        x_center y_center width height

    All normalized to [0,1].
    """

    if not isinstance(bbox, (list, tuple)):
        raise ValueError(
            f"bbox is not a list/tuple: {bbox}"
        )

    if len(bbox) != 4:
        raise ValueError(
            f"bbox does not contain 4 values: {bbox}"
        )

    x, y, w, h = map(float, bbox)

    if w <= 0 or h <= 0:
        raise ValueError(
            f"Non-positive bbox: {bbox}"
        )

    # GhostVision uses top-left x/y + width/height
    xc = x + w / 2.0
    yc = y + h / 2.0

    # Normalize
    xc /= img_w
    yc /= img_h
    w /= img_w
    h /= img_h

    # --------------------------------------------------------
    # We allow tiny numerical tolerance, but the actual
    # bounding box must stay inside the image.
    # --------------------------------------------------------

    if not (
        0.0 <= xc <= 1.0
        and 0.0 <= yc <= 1.0
        and 0.0 < w <= 1.0
        and 0.0 < h <= 1.0
    ):
        raise ValueError(
            f"Invalid normalized bbox:\n"
            f"original = {bbox}\n"
            f"image = {img_w} x {img_h}\n"
            f"normalized = {xc}, {yc}, {w}, {h}"
        )

    return (
        f"0 "
        f"{xc:.6f} "
        f"{yc:.6f} "
        f"{w:.6f} "
        f"{h:.6f}"
    )


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split):
    split_dir = GHOSTVISION_ROOT / split

    # --------------------------------------------------------
    # GhostVision calls it "valid", not "val"
    # --------------------------------------------------------

    metadata_path = split_dir / "metadata.jsonl"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found:\n{metadata_path}"
        )

    output_image_dir = OUTPUT_ROOT / "images" / split
    output_label_dir = OUTPUT_ROOT / "labels" / split

    output_image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_label_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    total = 0
    positive = 0
    normal = 0
    total_objects = 0
    missing_images = 0
    invalid_records = 0

    print()
    print("=" * 70)
    print(f"PROCESSING GHOSTVISION: {split}")
    print("=" * 70)

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

            total += 1

            try:
                record = json.loads(line)

                filename = record["file_name"]

                image_path = find_image(
                    split_dir,
                    filename
                )

                if image_path is None:
                    print(
                        f"WARNING line {line_no}: "
                        f"image not found: {filename}"
                    )
                    missing_images += 1
                    continue

                objects = record.get(
                    "objects",
                    {}
                )

                bboxes = objects.get(
                    "bbox",
                    []
                ) or []

                categories = objects.get(
                    "category",
                    []
                ) or []

                # ------------------------------------------------
                # Sanity check metadata arrays
                # ------------------------------------------------

                if len(bboxes) != len(categories):
                    raise ValueError(
                        f"bbox/category count mismatch: "
                        f"{len(bboxes)} vs "
                        f"{len(categories)}"
                    )

                # ------------------------------------------------
                # Get actual image dimensions
                # ------------------------------------------------

                with Image.open(image_path) as img:
                    img_w, img_h = img.size

                if img_w <= 0 or img_h <= 0:
                    raise ValueError(
                        f"Invalid image dimensions: "
                        f"{img_w}x{img_h}"
                    )

                # ------------------------------------------------
                # Generate YOLO labels
                # ------------------------------------------------

                yolo_lines = []

                for bbox, category in zip(
                    bboxes,
                    categories
                ):

                    # --------------------------------------------
                    # Current SAAD ontology:
                    #
                    # Crab-Pot
                    # Maybe-Crab-Pot
                    #
                    # -> anthropogenic = class 0
                    # --------------------------------------------

                    if category not in (
                        "Crab-Pot",
                        "Maybe-Crab-Pot"
                    ):
                        print(
                            f"WARNING line {line_no}: "
                            f"unknown category "
                            f"{category}"
                        )

                    yolo_line = convert_bbox_to_yolo(
                        bbox,
                        img_w,
                        img_h
                    )

                    yolo_lines.append(
                        yolo_line
                    )

                # ------------------------------------------------
                # Output filename
                #
                # We preserve the original filename.
                # Since each split has its own directory,
                # collisions across splits are impossible here.
                # ------------------------------------------------

                output_image_path = (
                    output_image_dir /
                    image_path.name
                )

                output_label_path = (
                    output_label_dir /
                    f"{image_path.stem}.txt"
                )

                # Copy image
                #
                # We use bytes so PIL doesn't recompress
                # the original image.
                # ------------------------------------------------

                output_image_path.write_bytes(
                    image_path.read_bytes()
                )

                # Write YOLO label
                output_label_path.write_text(
                    "\n".join(yolo_lines),
                    encoding="utf-8"
                )

                # ------------------------------------------------
                # Statistics
                # ------------------------------------------------

                if len(yolo_lines) > 0:
                    positive += 1
                    total_objects += len(yolo_lines)
                else:
                    normal += 1

                # ------------------------------------------------
                # Print first few records for sanity
                # ------------------------------------------------

                if total <= 5:

                    print()
                    print(
                        f"Example {total}: "
                        f"{filename}"
                    )

                    print(
                        f"  Image size: "
                        f"{img_w} x {img_h}"
                    )

                    print(
                        f"  Objects: "
                        f"{len(yolo_lines)}"
                    )

                    for line_out in yolo_lines:
                        print(
                            f"  YOLO: {line_out}"
                        )

            except Exception as e:

                invalid_records += 1

                print()
                print(
                    f"ERROR line {line_no}: {e}"
                )

    # ------------------------------------------------------------
    # Split summary
    # ------------------------------------------------------------

    print()
    print("-" * 70)
    print(f"{split.upper()} SUMMARY")
    print("-" * 70)

    print(
        f"Records processed : {total}"
    )

    print(
        f"Anthropogenic     : {positive}"
    )

    print(
        f"Normal/background : {normal}"
    )

    print(
        f"Total objects     : {total_objects}"
    )

    print(
        f"Missing images    : {missing_images}"
    )

    print(
        f"Invalid records   : {invalid_records}"
    )

    return {
        "split": split,
        "total": total,
        "positive": positive,
        "normal": normal,
        "objects": total_objects,
        "missing": missing_images,
        "invalid": invalid_records,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if "--dry-run" in sys.argv:
        dry_run(GHOSTVISION_ROOT, OUTPUT_ROOT)
        sys.exit(0)
    require_new_output(GHOSTVISION_ROOT, OUTPUT_ROOT)

    print()
    print("=" * 70)
    print("GHOSTVISION -> SAAD REPAIR")
    print("=" * 70)

    print()
    print("Source:")
    print(GHOSTVISION_ROOT)

    print()
    print("Output:")
    print(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    for split in SPLITS:

        result = process_split(split)

        results.append(result)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("FINAL GHOSTVISION SUMMARY")
    print("=" * 70)

    grand_total = 0
    grand_positive = 0
    grand_normal = 0
    grand_objects = 0
    grand_missing = 0
    grand_invalid = 0

    for r in results:

        print(
            f"{r['split']:>5} | "
            f"total={r['total']:>5} | "
            f"anthro={r['positive']:>5} | "
            f"normal={r['normal']:>5} | "
            f"objects={r['objects']:>6}"
        )

        grand_total += r["total"]
        grand_positive += r["positive"]
        grand_normal += r["normal"]
        grand_objects += r["objects"]
        grand_missing += r["missing"]
        grand_invalid += r["invalid"]

    print("-" * 70)

    print(
        f"TOTAL | "
        f"{grand_total:>5} | "
        f"{grand_positive:>5} | "
        f"{grand_normal:>5} | "
        f"{grand_objects:>6}"
    )

    print()
    print(
        f"Missing images : {grand_missing}"
    )

    print(
        f"Invalid records: {grand_invalid}"
    )

    # ========================================================
    # HARD SAFETY CHECKS
    # ========================================================

    if grand_positive == 0:

        raise RuntimeError(
            "\n"
            "CRITICAL: ZERO ANTHROPOGENIC IMAGES FOUND.\n"
            "GhostVision parser is still incorrect.\n"
            "DO NOT TRAIN."
        )

    if grand_missing > 0:

        raise RuntimeError(
            f"\n"
            f"CRITICAL: {grand_missing} images "
            f"could not be matched."
        )

    if grand_invalid > 0:

        raise RuntimeError(
            f"\n"
            f"CRITICAL: {grand_invalid} "
            f"metadata records were invalid."
        )

    # --------------------------------------------------------
    # Check expected rough magnitude
    # --------------------------------------------------------

    if grand_positive < 4000:

        print()
        print(
            "WARNING:"
        )
        print(
            "GhostVision has fewer than 4000 "
            "anthropogenic images."
        )
        print(
            "This is substantially below the "
            "expected rough count (~5127)."
        )
        print(
            "Inspect before rebuilding SAAD."
        )

    else:

        print()
        print(
            "✓ GhostVision contains a substantial "
            "number of anthropogenic examples."
        )

    print()
    print("=" * 70)
    print("GHOSTVISION REPAIR PASSED")
    print("=" * 70)

    print()
    print(
        "YOLO dataset written to:"
    )
    print(OUTPUT_ROOT)

    print()
    print(
        "DO NOT delete the original GhostVision dataset."
    )
