# Setup and local operation

Use this guide to install and operate the verified loopback application. For setting definitions see [Configuration](../config/README.md); for browser/API behavior see [Web](../web/README.md) and [Backend](../backend/README.md).

## Verified prerequisites

| Requirement | Audited value / boundary |
| --- | --- |
| Platform | Windows; one RTX 4070 Laptop GPU machine with 8 GB VRAM. |
| Python | 3.11 (observed 3.11.0). |
| PyTorch | Torch 2.11.0+cu128 and torchvision 0.26.0+cu128, official CUDA 12.8 wheels. |
| Frontend tools | Node 24/npm 11 (observed 24.20.0/11.19.0). |
| External inputs | Authorized copies of five frozen tensors; datasets are not required for ordinary inference. |
| Verification inputs | Three original hashed image files, only for fixture evaluation. |

Other observed dependencies include Ultralytics 8.4.140, FastAPI 0.141.1, NumPy 2.4.6, Pillow 12.3.0 and reportlab 5.0.1. [The Windows lock](../backend/requirements-original-windows-cu128.lock.txt) captures the audited environment before reportlab was added. [Direct requirements](../backend/requirements.txt) support hosted unit tests but are not the exact numerical-parity recipe.

A fresh isolated wheel installation passed on this machine. Another machine/OS, CPU inference parity and another GPU remain unverified. GPU-driver installation is outside the tested setup commands.

## Supply external assets

Use the directory layout in [config/README.md](../config/README.md#asset-identities-and-placement). Obtain authorized original tensors from the SAAD owner/source holder; no public download is supplied. Do not commit copies. Startup checks exact byte sizes and SHA-256, not filenames alone.

The repository supplies the live calibration, pinned by the manifest. Preserve its LF bytes using the existing Git attribute. Offline research calibration tables are separate external inputs.

Copy root `.env.example` to ignored `.env` and edit `SAAD_ARTIFACT_DIR` to your external root. Individual tensor overrides are supported, but must match the same frozen identities. Runtime upload/output/database paths must resolve inside the clean checkout; default ignored `var/` holds local state. Relative runtime paths resolve from repository root.

## Windows installation

Run PowerShell from the repository root and create a **new** environment in this checkout. Never use or alter the original project's environment.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r backend\requirements-original-windows-cu128.lock.txt
.\.venv\Scripts\python.exe -m pip install reportlab==5.0.1
.\.venv\Scripts\python.exe -m pip check
Copy-Item .env.example .env
# Edit SAAD_ARTIFACT_DIR in .env to the authorized external asset root.
```

Expected dependency validation: `No broken requirements found.` No activation is required because commands name the environment's Python explicitly.

If pip repeatedly transfers the large CUDA Torch wheel, stage that exact official wheel in ignored `var/wheels`, install it there first, then run the locked install. The fresh publication installation used this path:

```powershell
New-Item -ItemType Directory -Path var\wheels -Force | Out-Null
.\.venv\Scripts\python.exe -m pip download --no-deps --index-url https://download.pytorch.org/whl/cu128 --dest var\wheels 'torch==2.11.0+cu128'
.\.venv\Scripts\python.exe -m pip install --no-deps '.\var\wheels\torch-2.11.0+cu128-cp311-cp311-win_amd64.whl'
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r backend\requirements-original-windows-cu128.lock.txt
.\.venv\Scripts\python.exe -m pip install reportlab==5.0.1
.\.venv\Scripts\python.exe -m pip check
```

Do not substitute another Torch build and claim golden parity. Do not repeat environment creation over an existing installation as a repair step.

## Preflight and backend launch

From the root in the backend session:

```powershell
New-Item -ItemType Directory -Path var\ultralytics -Force | Out-Null
$env:YOLO_AUTOINSTALL='False'
$env:YOLO_CONFIG_DIR=(Resolve-Path var\ultralytics).Path
.\.venv\Scripts\python.exe -B research/verify_fixtures.py
.\.venv\Scripts\python.exe -B -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Fixture preflight checks five tensors, live calibration and three image hashes. For ordinary startup without test images, omit preflight; startup still validates model assets/calibration. `SAAD_GOLDEN_IMAGE_DIR` can select an authorized mirror of the three recorded image basenames.

`SAAD_DEVICE=auto` selects CUDA 0 if available; explicitly set `cuda:0` before a parity run so CPU fallback cannot be mistaken for audited GPU execution. Keep `YOLO_AUTOINSTALL=False` and local `YOLO_CONFIG_DIR` set in each backend/test shell.

After startup, `GET /api/ready` checks storage; `/api/health` and `/api/model` describe the loaded models. Interactive API docs are at `http://127.0.0.1:8000/docs`. Models are loaded before any route becomes available.

## Browser launch and production preview

From a second PowerShell session at the root:

```powershell
cd web
npm ci
npm run dev -- --host 127.0.0.1
```

Open the displayed Vite URL. Default API base is `http://127.0.0.1:8000`; default CORS allows localhost and 127.0.0.1 at port 5173. If Vite selects another port, configure that origin explicitly or free the intended port.

To change the browser API base, copy `web/.env.example` to ignored `web/.env.local`, set `VITE_API_BASE_URL`, and restart/rebuild the client. For production preview, allow `http://127.0.0.1:4173` in `SAAD_CORS_ORIGINS` before backend startup, then from `web/`:

```powershell
npm run typecheck
npm run lint
npm run build
npm run preview -- --host 127.0.0.1 --port 4173
```

This is local build verification, not a public deployment recipe.

## First-use check

Upload an authorized PNG/JPEG/WebP/TIFF image through New Scan. Confirm navigation to a saved scan, select a candidate, then save accept/reject or a corrected box. Refresh the direct scan page and inspect history/queue. Export the scan as JSON, CSV or PDF.

An empty candidate list is a valid inference outcome, not a review failure. A partial scan means an anomaly or TTA signal failed; check backend logs before interpreting evidence. [Backend](../backend/README.md) defines responses and errors.

## Tests, local state and troubleshooting

[Tests](../tests/README.md) provides exact 13-test asset-free and 25-test full commands. Stop the API before full regression on the audited 8 GB device; concurrent model processes previously caused CUDA/host-memory failures.

Uploads and `var/saad.sqlite3` are ignored runtime state. Preserve both when backing up; reports are generated from saved state on demand. There is no automatic deletion/retention or authenticated backup feature.

| Symptom | Check |
| --- | --- |
| Startup missing/hash mismatch | Confirm external root, manifest-relative placement and exact source hashes; do not swap research weights. |
| Calibration SHA mismatch | Check committed LF bytes and editor/Git line endings. |
| Browser fetch/CORS failure | Verify API loopback address, build-time API base and actual browser origin. |
| Upload 400/413 | Check supported suffix, decoded image validity and server byte/pixel limits. |
| Readiness/storage 503 | Check local runtime paths, SQLite permissions and logs; preserve state before repair. |
| Partial analysis | Inspect VAE/flow/TTA logs; HTTP success alone does not prove full inference. |
| Native/CUDA test failure | Stop other model processes and record the failed run; it is not a parity result. |

The API has no authentication. Keep both services bound to loopback; [Limitations](LIMITATIONS.md) defines the supported security and deployment boundary.
