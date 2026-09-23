"""
SAAD DATASET INTEGRITY PASS v2
==============================

READ-ONLY diagnostic checker for:
    E:\SIH\Datasets\SAAD_baseline

This version is tailored to the actual SAAD manifest columns:
    dataset
    original_path
    output_image
    split
    group
    source_class
    saad_class
    has_annotation
    num_objects
    annotation_type

It does NOT modify SAAD_baseline or the original datasets.
"""

from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import random
import re

from PIL import Image, ImageDraw

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r"E:\SIH\Datasets")
SAAD = ROOT / "SAAD_baseline"
MANIFEST = SAAD / "manifest.csv"

AI4_ROOT = ROOT / "AI4Shipwrecks"

REPORT_DIR = SAAD / "integrity_pass_v2"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
VISUAL_SAMPLES_PER_SPLIT = 15
random.seed(SEED)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".pbm", ".tif", ".tiff", ".webp"}


# ============================================================
# HELPERS
# ============================================================

def norm_path(x):
    return str(x).strip().replace("/", "\\").lower()


def resolve_manifest_path(value, base=SAAD):
    """Resolve an absolute path or a path relative to SAAD."""
    if not value:
        return None

    p = Path(str(value).strip())

    if p.is_absolute():
        return p

    return base / p


def sha1_file(path, chunk=1024 * 1024):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def parse_yolo(label_path):
    """
    Returns:
        boxes: list[(class, x, y, w, h)]
        errors: list[str]
    """
    errors = []
    boxes = []

    try:
        text = label_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        return [], [f"read_error: {e}"]

    if not text:
        return [], []

    for line_no, line in enumerate(text.splitlines(), 1):
        parts = line.split()

        if len(parts) != 5:
            errors.append(
                f"line {line_no}: expected 5 fields, got {len(parts)}"
            )
            continue

        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except Exception:
            errors.append(f"line {line_no}: non-numeric value")
            continue

        if cls != 0:
            errors.append(
                f"line {line_no}: class={cls}, expected class 0"
            )

        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
            errors.append(
                f"line {line_no}: normalized coordinate outside [0,1]"
            )

        if w <= 0 or h <= 0:
            errors.append(
                f"line {line_no}: width/height <= 0"
            )

        if x - w / 2 < -1e-5 or x + w / 2 > 1 + 1e-5:
            errors.append(
                f"line {line_no}: x box extends outside image"
            )

        if y - h / 2 < -1e-5 or y + h / 2 > 1 + 1e-5:
            errors.append(
                f"line {line_no}: y box extends outside image"
            )

        boxes.append((cls, x, y, w, h))

    return boxes, errors


def make_visual(path, label_path, out_path, title):
    img = Image.open(path).convert("RGB")
    W, H = img.size

    boxes, errors = parse_yolo(label_path)

    # Avoid gigantic visual sheets for extreme SSS aspect ratios.
    img.thumbnail((1100, 700))

    canvas = Image.new("RGB", (img.width, img.height + 45), "black")
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)

    sx = img.width / W
    sy = img.height / H

    for cls, x, y, w, h in boxes:
        x1 = int((x - w / 2) * W * sx)
        y1 = int((y - h / 2) * H * sy)
        x2 = int((x + w / 2) * W * sx)
        y2 = int((y + h / 2) * H * sy)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1 + 3, y1 + 3), "anthropogenic", fill="yellow")

    draw.text((8, img.height + 8), title, fill="white")
    canvas.save(out_path, quality=92)

    return errors


# ============================================================
# LOAD MANIFEST
# ============================================================

print("\nSAAD DATASET INTEGRITY PASS v2")
print("==============================")
print(f"SAAD      : {SAAD}")
print(f"MANIFEST  : {MANIFEST}")
print(f"REPORTS   : {REPORT_DIR}")

if not MANIFEST.exists():
    raise FileNotFoundError(MANIFEST)

with open(MANIFEST, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fields = reader.fieldnames or []

required = [
    "dataset",
    "original_path",
    "output_image",
    "split",
    "group",
    "source_class",
    "saad_class",
    "has_annotation",
    "num_objects",
    "annotation_type",
]

print("\nMANIFEST COLUMNS")
print("----------------")
for x in fields:
    print(" ", x)

missing_required = [x for x in required if x not in fields]
if missing_required:
    print("\nWARNING — missing expected columns:")
    for x in missing_required:
        print(" ", x)

print(f"\nManifest rows: {len(rows)}")


# ============================================================
# 1. MANIFEST SUMMARY
# ============================================================

counts = Counter(
    (
        r.get("dataset", ""),
        r.get("split", ""),
        r.get("saad_class", ""),
    )
    for r in rows
)

print("\nMANIFEST SUMMARY")
print("----------------")
for (ds, sp, cl), n in sorted(counts.items()):
    print(f"{ds:20s} {sp:5s} {cl:15s} {n:6d}")


# ============================================================
# 2. OUTPUT FILE + YOLO LABEL CHECK
# ============================================================

missing_images = []
missing_labels = []
invalid_labels = []

for r in rows:
    img_path = resolve_manifest_path(r.get("output_image", ""))

    if img_path is None or not img_path.exists():
        missing_images.append(str(img_path))
        continue

    split = r.get("split", "").strip()
    label_path = SAAD / "labels" / split / f"{img_path.stem}.txt"

    if not label_path.exists():
        missing_labels.append(str(label_path))
        continue

    _, errs = parse_yolo(label_path)

    if errs:
        invalid_labels.append(
            (str(label_path), errs)
        )

print("\nMANIFEST FILE CHECK")
print("-------------------")
print(f"Missing images       : {len(missing_images)}")
print(f"Missing labels       : {len(missing_labels)}")
print(f"Invalid YOLO labels  : {len(invalid_labels)}")

with open(REPORT_DIR / "invalid_yolo_labels.txt", "w", encoding="utf-8") as f:
    for path, errs in invalid_labels:
        f.write(path + "\n")
        for e in errs:
            f.write("  " + e + "\n")


# ============================================================
# 3. DIRECTORY PAIRING
# ============================================================

print("\nDIRECTORY COUNTS")
print("----------------")

directory_pair_problems = []

for split in ["train", "val", "test"]:
    img_dir = SAAD / "images" / split
    lbl_dir = SAAD / "labels" / split

    images = [
        p for p in img_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ] if img_dir.exists() else []

    labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

    img_stems = {p.stem for p in images}
    lbl_stems = {p.stem for p in labels}

    missing = sorted(img_stems - lbl_stems)
    orphan = sorted(lbl_stems - img_stems)

    print(f"{split:5s}: images={len(images):5d} "
          f"labels={len(labels):5d} "
          f"missing_labels={len(missing):4d} "
          f"orphan_labels={len(orphan):4d}")

    if missing:
        directory_pair_problems.append((split, "missing_labels", missing))

    if orphan:
        directory_pair_problems.append((split, "orphan_labels", orphan))


# ============================================================
# 4. ACTUAL MANIFEST vs DIRECTORY COUNT
# ============================================================

print("\nMANIFEST ↔ DIRECTORY COUNT")
print("--------------------------")

for split in ["train", "val", "test"]:
    manifest_n = sum(
        1 for r in rows if r.get("split", "").strip() == split
    )

    img_dir = SAAD / "images" / split
    actual_n = sum(
        1 for p in img_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ) if img_dir.exists() else 0

    print(
        f"{split:5s}: manifest={manifest_n:5d} "
        f"actual_images={actual_n:5d} "
        f"delta={actual_n - manifest_n:+d}"
    )


# ============================================================
# 5. CROSS-SPLIT DUPLICATE CHECK
# ============================================================

print("\nCROSS-SPLIT DUPLICATE CHECK")
print("---------------------------")

hash_map = defaultdict(list)

for split in ["train", "val", "test"]:
    img_dir = SAAD / "images" / split

    if not img_dir.exists():
        continue

    for p in img_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue

        try:
            h = sha1_file(p)
            hash_map[h].append((split, str(p)))
        except Exception as e:
            print("Hash error:", p, e)

duplicate_groups = []

for h, entries in hash_map.items():
    splits = {x[0] for x in entries}

    if len(splits) > 1:
        duplicate_groups.append((h, entries))

print(f"Unique hashes              : {len(hash_map)}")
print(f"Cross-split duplicate groups: {len(duplicate_groups)}")

with open(
    REPORT_DIR / "cross_split_duplicates.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:
    w = csv.writer(f)
    w.writerow(["sha1", "split", "output_path"])

    for h, entries in duplicate_groups:
        for split, path in entries:
            w.writerow([h, split, path])


# ============================================================
# 6. MAP OUTPUT IMAGE BACK TO MANIFEST
# ============================================================

manifest_by_output = {}

for r in rows:
    p = resolve_manifest_path(r.get("output_image", ""))
    if p is not None:
        manifest_by_output[norm_path(p)] = r


# ============================================================
# 7. AI4 EXACT SOURCE COVERAGE
# ============================================================

print("\nAI4SHIPWRECKS SOURCE COVERAGE")
print("------------------------------")

ai4_source_images = []

if AI4_ROOT.exists():
    for p in AI4_ROOT.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in IMAGE_EXTS:
            continue

        parts = [x.lower() for x in p.parts]

        # Restrict to the dataset's train/test image directories.
        if "images" in parts and ("train" in parts or "test" in parts):
            ai4_source_images.append(p)

print(f"AI4 source images found: {len(ai4_source_images)}")

ai4_rows = [
    r for r in rows
    if r.get("dataset", "").strip().lower() == "ai4shipwrecks"
]

print(f"AI4 manifest rows     : {len(ai4_rows)}")

source_set = {norm_path(p) for p in ai4_source_images}

manifest_originals = set()

for r in ai4_rows:
    p = resolve_manifest_path(r.get("original_path", ""), base=ROOT)
    if p is not None:
        manifest_originals.add(norm_path(p))

print(f"AI4 original paths in manifest: {len(manifest_originals)}")

missing_from_manifest = sorted(source_set - manifest_originals)
not_found_in_source = sorted(manifest_originals - source_set)

print(f"AI4 source NOT in manifest    : {len(missing_from_manifest)}")
print(f"Manifest AI4 path not in source: {len(not_found_in_source)}")

with open(
    REPORT_DIR / "ai4_missing_from_manifest.txt",
    "w",
    encoding="utf-8",
) as f:
    for p in missing_from_manifest:
        f.write(p + "\n")

with open(
    REPORT_DIR / "ai4_manifest_paths_not_in_source.txt",
    "w",
    encoding="utf-8",
) as f:
    for p in not_found_in_source:
        f.write(p + "\n")


# ============================================================
# 8. AI4 GROUP / SPLIT REPORT
# ============================================================

print("\nAI4 GROUP / SPLIT DISTRIBUTION")
print("------------------------------")

ai4_groups = Counter()

for r in ai4_rows:
    ai4_groups[
        (
            r.get("split", ""),
            r.get("group", ""),
            r.get("saad_class", ""),
        )
    ] += 1

for (sp, group, cl), n in sorted(ai4_groups.items()):
    print(f"{sp:5s} | {group:30s} | {cl:15s} | {n:4d}")


# ============================================================
# 9. SUBPIPE SEQUENCE CHECK FROM original_path
# ============================================================

print("\nSUBPIPE SEQUENCE / TIMESTAMP CHECK")
print("----------------------------------")

subpipe_rows = [
    r for r in rows
    if r.get("dataset", "").strip().lower() == "subpipemini2"
]

timestamp_rows = []

for r in subpipe_rows:
    src = r.get("original_path", "")
    split = r.get("split", "")

    # Typical SubPipe filenames contain Unix-like timestamps.
    matches = re.findall(r"(?<!\d)(\d{9,13}(?:\.\d+)?)", src)

    if matches:
        try:
            ts = float(matches[-1])
            timestamp_rows.append((ts, split, src))
        except ValueError:
            pass

print(f"SubPipe manifest rows       : {len(subpipe_rows)}")
print(f"Rows with detected timestamp: {len(timestamp_rows)}")

sequence_leakage = []

timestamp_rows.sort()

for a, b in zip(timestamp_rows, timestamp_rows[1:]):
    dt = abs(a[0] - b[0])

    # Adjacent frames within 5 seconds assigned to different splits
    # are a strong warning sign for temporal leakage.
    if dt <= 5.0 and a[1] != b[1]:
        sequence_leakage.append((a, b, dt))

print(
    "Cross-split neighboring pairs <=5 sec: "
    f"{len(sequence_leakage)}"
)

with open(
    REPORT_DIR / "subpipe_sequence_leakage.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:
    w = csv.writer(f)
    w.writerow([
        "time_a", "split_a", "source_a",
        "time_b", "split_b", "source_b",
        "delta_seconds",
    ])

    for a, b, dt in sequence_leakage:
        w.writerow([
            a[0], a[1], a[2],
            b[0], b[1], b[2],
            dt,
        ])


# ============================================================
# 10. VISUAL CHECK — STRATIFIED
# ============================================================

print("\nCREATING VISUAL CHECKS")
print("----------------------")

candidates = defaultdict(list)

for r in rows:
    img_path = resolve_manifest_path(r.get("output_image", ""))

    if img_path is None or not img_path.exists():
        continue

    split = r.get("split", "")
    dataset = r.get("dataset", "")
    label_path = SAAD / "labels" / split / f"{img_path.stem}.txt"

    if not label_path.exists():
        continue

    candidates[(split, dataset)].append((img_path, label_path, r))


for split in ["train", "val", "test"]:
    selected = []

    datasets = sorted({
        ds for (sp, ds) in candidates if sp == split
    })

    for ds in datasets:
        pool = candidates[(split, ds)]
        random.shuffle(pool)

        # Up to 5 per dataset.
        selected.extend(pool[:5])

    random.shuffle(selected)
    selected = selected[:VISUAL_SAMPLES_PER_SPLIT]

    if not selected:
        continue

    out_dir = REPORT_DIR / f"visual_{split}"
    out_dir.mkdir(exist_ok=True)

    print(f"\n{split}: {len(selected)} samples")

    for i, (img_path, label_path, r) in enumerate(selected, 1):
        out = out_dir / f"{i:02d}_{r.get('dataset','unknown')}_{img_path.stem}.jpg"

        try:
            make_visual(
                img_path,
                label_path,
                out,
                f"{r.get('dataset','')} | {split} | "
                f"{r.get('saad_class','')}"
            )
        except Exception as e:
            print("Visual error:", img_path, e)


# ============================================================
# 11. FINAL REPORT
# ============================================================

report = REPORT_DIR / "integrity_report.txt"

with open(report, "w", encoding="utf-8") as f:
    f.write("SAAD DATASET INTEGRITY REPORT v2\n")
    f.write("================================\n\n")

    f.write(f"Manifest rows: {len(rows)}\n\n")

    f.write("Manifest summary:\n")
    for (ds, sp, cl), n in sorted(counts.items()):
        f.write(f"  {ds} | {sp} | {cl} | {n}\n")

    f.write("\nFile checks:\n")
    f.write(f"  Missing images: {len(missing_images)}\n")
    f.write(f"  Missing labels: {len(missing_labels)}\n")
    f.write(f"  Invalid YOLO labels: {len(invalid_labels)}\n")

    f.write("\nDuplicate check:\n")
    f.write(f"  Unique hashes: {len(hash_map)}\n")
    f.write(f"  Cross-split duplicate groups: {len(duplicate_groups)}\n")

    f.write("\nAI4:\n")
    f.write(f"  Source images: {len(ai4_source_images)}\n")
    f.write(f"  Manifest rows: {len(ai4_rows)}\n")
    f.write(f"  Source not in manifest: {len(missing_from_manifest)}\n")
    f.write(f"  Manifest path not in source: {len(not_found_in_source)}\n")

    f.write("\nSubPipe:\n")
    f.write(f"  Rows with timestamps: {len(timestamp_rows)}\n")
    f.write(
        "  Cross-split neighboring timestamp pairs <=5 sec: "
        f"{len(sequence_leakage)}\n"
    )

print("\nREPORT")
print("------")
print(report)

print("\n============================================================")
print("INTEGRITY PASS v2 COMPLETE")
print("============================================================")
print("Dataset was NOT modified.")
print(f"Open: {REPORT_DIR}")
