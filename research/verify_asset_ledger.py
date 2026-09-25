"""Stream-hash explicitly inventoried external assets without walking datasets."""

import hashlib
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
ROW = re.compile(r"`(E:\\SIH[^`]+)`\s*\|\s*(\d+)\s*\|\s*`([0-9a-f]{64})`")


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    text = (ROOT / "ASSET_MANIFEST.md").read_text(encoding="utf-8")
    records = [(Path(path), int(size), expected) for path, size, expected in ROW.findall(text)]
    if len(records) != 202:
        raise ValueError(f"Expected 202 manifest rows, found {len(records)}")
    checked = set()
    total_bytes = 0
    for path, size, expected in records:
        if path in checked:
            continue
        checked.add(path)
        if not path.is_file() or path.stat().st_size != size or sha256(path) != expected:
            raise ValueError(f"External asset missing or changed: {path}")
        total_bytes += size
    print(f"Verified {len(records)} ledger rows, {len(checked)} unique original files, {total_bytes} bytes by SHA-256; no dataset image traversal.")


if __name__ == "__main__":
    main()
