"""Guards for historical dataset builders in the clean repository."""

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_OUTPUT_ROOT = (REPO_ROOT / "var" / "research_datasets").resolve()


def configured_path(variable: str, default: Path) -> Path:
    return Path(os.environ.get(variable, str(default))).resolve()


def dry_run(source: Path, output: Path) -> None:
    print(f"Source (read only): {source}")
    print(f"Output (not created): {output}")
    print("This is a dry run; no images, labels or manifests were written.")


def require_new_output(source: Path, output: Path) -> None:
    if os.environ.get("SAAD_ALLOW_DATASET_BUILD") != "YES":
        raise RuntimeError("Dataset build disabled. Set SAAD_ALLOW_DATASET_BUILD=YES explicitly.")
    if SAFE_OUTPUT_ROOT not in output.parents:
        raise RuntimeError(f"Output must be a new child of {SAFE_OUTPUT_ROOT}: {output}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing dataset: {output}")
    if not source.is_dir():
        raise FileNotFoundError(f"Missing source dataset directory: {source}")
