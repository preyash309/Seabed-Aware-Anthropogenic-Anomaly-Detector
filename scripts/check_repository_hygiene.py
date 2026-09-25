"""Fail CI if the proposed tree or reachable Git history contains publishable hazards.

Only Git-tracked bytes are scanned. External SAAD assets are never read or uploaded.
"""

from collections import Counter
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MAX_BLOB_BYTES = 1_048_576
BLOCKED_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".pkl", ".joblib",
    ".npy", ".npz", ".h5", ".hdf5", ".parquet", ".feather",
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pbm",
    ".db", ".sqlite", ".sqlite3", ".pem", ".key",
}
BLOCKED_PARTS = {"node_modules", "__pycache__", ".venv", "venv", "var", "uploads", "outputs", "runs", "dist", "build"}
SECRET_PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "AWS key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "OpenAI key": re.compile(rb"sk-[A-Za-z0-9]{32,}"),
    "Slack token": re.compile(rb"xox[baprs]-[A-Za-z0-9-]{20,}"),
    "credential assignment": re.compile(
        rb"(?i)(?:password|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*['\"][^'\"\r\n]{8,}"
    ),
}


def git(*args: str, input_data: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=ROOT, input=input_data, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=True,
    ).stdout


def path_error(path: str, historical: bool = False) -> str | None:
    p = PurePosixPath(path)
    parts = set(p.parts)
    if p.suffix.lower() in BLOCKED_SUFFIXES or p.name.endswith((".db-journal", ".sqlite-wal", ".sqlite-shm")):
        return "binary/data/runtime extension"
    if parts & BLOCKED_PARTS:
        return "generated or runtime directory"
    if p.parts[0] in {"data", "dataset", "Datasets"}:
        return "dataset directory"
    if p.parts[0] == "datasets" and path != "datasets/.gitignore":
        return "dataset directory"
    if p.name == ".env" or p.name.startswith(".env."):
        if p.name == ".env.example":
            return None
        if historical and p.name == ".env.local":
            return None  # Exact contents are checked below.
        return "local environment file"
    return None


def scan_secrets(data: bytes) -> list[str]:
    return [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(data)]


def main() -> int:
    findings: list[str] = []
    tracked = [x.decode() for x in git("ls-files", "-z").split(b"\0") if x]
    for path in tracked:
        problem = path_error(path)
        if problem:
            findings.append(f"tracked {path}: {problem}")
        size = (ROOT / path).stat().st_size
        if size > MAX_BLOB_BYTES:
            findings.append(f"tracked {path}: {size} bytes exceeds {MAX_BLOB_BYTES}")
    required_ignores = (
        ".env", "web/.env.local", "var/saad.sqlite3", "var/uploads/scan.png",
        "uploads/scan.png", "outputs/report.pdf", "datasets/sample.png",
        "weights.pt", "tensor.npy", "web/node_modules/pkg/index.js", "web/dist/app.js",
    )
    for path in required_ignores:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            findings.append(f".gitignore does not cover {path}")

    objects = [line.split(b" ", 1) for line in git("rev-list", "--objects", "--all").splitlines()]
    ids = [item[0] for item in objects]
    metadata = git("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
                   input_data=b"\n".join(ids) + b"\n")
    blob_sizes = {}
    for line in metadata.splitlines():
        sha, kind, size = line.split()
        if kind == b"blob":
            blob_sizes[sha] = int(size)
    names_by_sha: dict[bytes, set[str]] = {}
    for item in objects:
        if item[0] in blob_sizes:
            names_by_sha.setdefault(item[0], set()).add(item[1].decode() if len(item) > 1 else "<unnamed>")
    for sha, size in blob_sizes.items():
        paths = names_by_sha.get(sha, {"<unnamed>"})
        label = ", ".join(sorted(paths))
        if size > MAX_BLOB_BYTES:
            findings.append(f"history {label}: {size} bytes exceeds {MAX_BLOB_BYTES}")
            continue
        for path in paths:
            problem = path_error(path, historical=True)
            if problem:
                findings.append(f"history {path}: {problem}")
        data = git("cat-file", "blob", sha.decode())
        if any(PurePosixPath(path).name == ".env.local" for path in paths) and data != b"VITE_API_BASE_URL=http://127.0.0.1:8000":
            findings.append("history .env.local: unexpected contents")
        for name in scan_secrets(data):
            findings.append(f"history {label}: {name} signature")

    if findings:
        print("Repository hygiene FAILED:")
        for finding in sorted(set(findings)):
            print(f"  {finding}")
        return 1
    print(f"Repository hygiene PASS: {len(tracked)} tracked files, {len(blob_sizes)} reachable historical blobs; required ignores present; no blocked data, runtime files or secret signatures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
