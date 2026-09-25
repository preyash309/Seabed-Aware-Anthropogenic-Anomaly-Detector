"""Read-only preflight and opt-in fixed-image evaluation for offline research."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from config import Settings  # noqa: E402


def verify() -> None:
    settings = Settings.from_environment()
    settings.validate_assets()
    records = json.loads((ROOT / "audit" / "fixed_image_stage_capture.json").read_text())
    api = json.loads((ROOT / "audit" / "baseline_capture.json").read_text())
    mirror = os.environ.get("SAAD_GOLDEN_IMAGE_DIR")
    for record in [*records, api]:
        source = Path(record.get("source", record.get("image")))
        image = Path(mirror) / source.name if mirror else source
        if not image.is_file():
            raise FileNotFoundError(f"Golden image missing: {image}")
        expected = record.get("source_sha256", record.get("sha256"))
        actual = hashlib.sha256(image.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Golden image hash mismatch: {image}")
    print(f"Validated {len(settings.artifact_paths)} frozen tensors, live policy {settings.calibration['policy_id']} and {len(records)} fixed domain images; no outputs written.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run the existing fixed-image numerical test after preflight")
    args = parser.parse_args()
    verify()
    if args.run:
        completed = subprocess.run(
            [sys.executable, "-B", "-m", "unittest", "discover", "-s", str(ROOT / "tests"),
             "-p", "test_stage1_golden.py", "-v"], cwd=ROOT, check=False,
        )
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
