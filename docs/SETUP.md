# Setup and local operation

SAAD has been verified on one Windows RTX 4070 Laptop GPU machine for local loopback use. Use Python 3.11 and Node 24/npm 11. The observed parity stack includes Torch 2.11.0+cu128, torchvision 0.26.0+cu128, Ultralytics 8.4.140, FastAPI 0.141.1, NumPy 2.4.6, Pillow 12.3.0 and reportlab 5.0.1. `backend/requirements-original-windows-cu128.lock.txt` records the audited Windows environment before reportlab was added; `backend/requirements.txt` lists direct dependencies and is not the numerical-parity installation recipe. A fresh isolated wheel installation passed on the audited machine; a second machine has not been verified.

## External inputs

Obtain the original five model/normalization tensors from an authorized copy of the SAAD results. The repository supplies no checkpoint or dataset download and grants no redistribution rights. Put the tensors outside Git in the relative layout of [the model manifest](../config/model_manifest.json). The manifest pins exact byte sizes and SHA-256 hashes; startup rejects missing or mismatched tensors. The committed `config/live_api_v1.json` is the live calibration and has a manifest-pinned hash. A Git attribute preserves its LF line endings on Windows so checkout conversion cannot change that raw hash. The original three sonar test images are external too; their hashes are in the fixtures under `audit/`.

From the repository root, copy `.env.example` to ignored `.env` and set `SAAD_ARTIFACT_DIR` to the external root. On the audited machine `E:/SIH/SIH_Results` is read-only. Optional `SAAD_YOLO_WEIGHTS`, `SAAD_VAE_WEIGHTS`, `SAAD_FLOW_WEIGHTS`, `SAAD_LATENT_MEAN`, `SAAD_LATENT_STD`, `SAAD_MODEL_MANIFEST` and `SAAD_CALIBRATION_FILE` select exact matching assets. `SAAD_DATASET_DIR` is an optional read-only prepared dataset. Relative paths resolve from the repository root. Upload, output and SQLite paths (`SAAD_UPLOAD_DIR`, `SAAD_OUTPUT_DIR`, `SAAD_DATABASE_PATH`) must remain inside the clean checkout and default to ignored `var/`. The `.env.example` also lists upload byte/pixel limits, API URL, CORS origins and device choice.

## Windows installation

In PowerShell at the repository root, create a **new** environment in the clean checkout. Do not use or modify an original project's environment.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r backend\requirements-original-windows-cu128.lock.txt
.\.venv\Scripts\python.exe -m pip install reportlab==5.0.1
.\.venv\Scripts\python.exe -m pip check
Copy-Item .env.example .env
# Edit SAAD_ARTIFACT_DIR in .env to the authorized external asset root.
```

If pip repeatedly transfers the large CUDA Torch wheel, stage that exact official wheel in ignored `var/wheels`, install it there first, then run the locked install. The publication audit used this path for its fresh installation:

```powershell
New-Item -ItemType Directory -Path var\wheels -Force | Out-Null
.\.venv\Scripts\python.exe -m pip download --no-deps --index-url https://download.pytorch.org/whl/cu128 --dest var\wheels 'torch==2.11.0+cu128'
.\.venv\Scripts\python.exe -m pip install --no-deps '.\var\wheels\torch-2.11.0+cu128-cp311-cp311-win_amd64.whl'
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r backend\requirements-original-windows-cu128.lock.txt
.\.venv\Scripts\python.exe -m pip install reportlab==5.0.1
.\.venv\Scripts\python.exe -m pip check
```

Do not substitute a different Torch build and claim golden parity.

## Start backend and browser

In the backend PowerShell session at the repository root:

```powershell
New-Item -ItemType Directory -Path var\ultralytics -Force | Out-Null
$env:YOLO_AUTOINSTALL='False'
$env:YOLO_CONFIG_DIR=(Resolve-Path var\ultralytics).Path
.\.venv\Scripts\python.exe -B research/verify_fixtures.py
.\.venv\Scripts\python.exe -B -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Preflight checks all five tensors, live calibration and the three fixed external image hashes. For normal API startup without those three test images, omit the `verify_fixtures.py` command; startup still checks the five tensors and calibration. `SAAD_GOLDEN_IMAGE_DIR` can point the preflight and golden suite to an authorized mirror of the three fixed basenames.

In a second PowerShell session:

```powershell
cd web
npm ci
npm run dev -- --host 127.0.0.1
```

The frontend defaults to `http://127.0.0.1:8000` through `VITE_API_BASE_URL`. Copy `web/.env.example` to ignored `web/.env.local` to change it. Include the browser origin in `SAAD_CORS_ORIGINS` before API startup. The default allows the Vite development origin on port 5173. For a local production preview, run `npm run typecheck`, `npm run lint`, `npm run build`, then `npm run preview -- --host 127.0.0.1 --port 4173`; allow `http://127.0.0.1:4173` in CORS first. Rebuild after changing `VITE_API_BASE_URL`.

`GET /api/ready` checks storage after model load; `POST /api/analyze` accepts image upload; scan/review/report routes are described in [Architecture](ARCHITECTURE.md). Runtime uploads, reports and `var/saad.sqlite3` are ignored by Git. Back up local state before replacing a machine. Keep both services on loopback: the API has no authentication or multi-user authorization.

## Troubleshooting

A missing or mismatched model fails startup before inference. Check the manifest path and hash; do not substitute a similarly named research checkpoint. Invalid uploads and storage failures return HTTP errors and remove attempted uploads. If the browser cannot connect, verify the loopback API, `VITE_API_BASE_URL` and CORS origins. Stop the model-loaded server before running the full GPU golden suite on the audited 8 GB device; concurrent execution caused CUDA/host-memory failures during migration. See [Evaluation](EVALUATION.md) for test commands and [Limitations](LIMITATIONS.md) for the verified boundary.
