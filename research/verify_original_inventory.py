"""Compare read-only original source file sizes with the pre-migration audit."""

import argparse
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
ROW = re.compile(r"^\| `([^`]+)` \| (\d+) \|")


def section_rows(text: str, start: str, end: str) -> list[tuple[str, int]]:
    section = text.split(start, 1)[1].split(end, 1)[0]
    return [(match.group(1), int(match.group(2))) for line in section.splitlines()
            if (match := ROW.match(line))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-root", type=Path, default=Path(r"E:\SIH"))
    parser.add_argument("--web-root", type=Path, default=Path(r"E:\SIH Web\saad"))
    args = parser.parse_args()
    inventory = (ROOT / "SOURCE_INVENTORY.md").read_text(encoding="utf-8")
    groups = [
        (args.ml_root, section_rows(inventory, "## Original ML and backend", "## Original web"), 37),
        (args.web_root, section_rows(inventory, "## Original web", "## Clean repository"), 40),
    ]
    for base, rows, expected_count in groups:
        if len(rows) != expected_count:
            raise ValueError(f"Audit row count changed for {base}: {len(rows)}")
        for relative, expected_size in rows:
            file = base / relative
            if not file.is_file() or file.stat().st_size != expected_size:
                raise ValueError(f"Original source missing or size changed: {file}")
    print("Original inventory sizes matched: 37 ML/backend and 40 web files (including .env.local size only).")


if __name__ == "__main__":
    main()
