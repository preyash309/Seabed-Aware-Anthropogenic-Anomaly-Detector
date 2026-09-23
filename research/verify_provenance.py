"""Check archived research source bytes against the Stage 5 ledger."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, help="Also check the read-only original ML source root")
    args = parser.parse_args()
    entries = json.loads((ROOT / "research" / "source_manifest.json").read_text(encoding="utf-8-sig"))
    if len(entries) != 31:
        raise ValueError(f"Expected 31 archived source records, found {len(entries)}")
    for entry in entries:
        archive = ROOT / entry["archive"]
        if not archive.is_file() or archive.stat().st_size != entry["bytes"] or digest(archive) != entry["sha256"]:
            raise ValueError(f"Archived source changed: {archive}")
        if args.original_root:
            original = args.original_root / entry["original"]
            if not original.is_file() or original.stat().st_size != entry["bytes"] or digest(original) != entry["sha256"]:
                raise ValueError(f"Original source changed: {original}")
    print(f"Verified {len(entries)} archived research source hashes" + (" against read-only originals." if args.original_root else "."))


if __name__ == "__main__":
    main()
