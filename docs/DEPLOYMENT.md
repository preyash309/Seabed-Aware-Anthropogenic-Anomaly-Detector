# Local Windows operation and installation

SAAD is verified as a **loopback** application on the audited RTX 4070 Windows machine. Network exposure and multi-user operation require authentication, authorization, upload access controls and a deployment review. The production score policy is pinned to `saad-live-api-v1`; the archived offline Evidence Engine v3 is not a deployment option.

## External inputs and directories

Keep the five tensors named in `config/model_manifest.json` outside Git. Copy `.env.example` to `.env` in the repository root and replace `SAAD_ARTIFACT_DIR` with the directory containing those relative asset paths. On the audited machine this is `E:/SIH/SIH_Results`, used read-only. The startup validator checks byte sizes, SHA-256 hashes and live calibration identity before loading models. `SAAD_UPLOAD_DIR`, `SAAD_OUTPUT_DIR` and `SAAD_DATABASE_PATH` must resolve inside the clean repository; their defaults are under ignored `var/`. Never use the original source, dataset or results tree as a writable runtime path.

Use Python 3.11 and Node 24. The observed parity stack is Torch `2.11.0+cu128`, torchvision `0.26.0+cu128`, Ultralytics `8.4.140`, FastAPI `0.141.1`, NumPy `2.4.6`, Pillow `12.3.0` and reportlab `5.0.1`. The original Windows dependency snapshot is in `backend/requirements-original-windows-cu128.lock.txt`; it predates the added reportlab dependency. Downloading the exact 2.8 GB CUDA Torch wheel for a fresh environment failed during this migration, so these installation commands are a reproduction recipe, **not a fresh-install PASS claim**:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r backend\requirements-original-windows-cu128.lock.txt
.\.venv\Scripts\python.exe -m pip install reportlab==5.0.1
.\.venv\Scripts\python.exe -m pip check
Copy-Item .env.example .env
# Edit SAAD_ARTIFACT_DIR in .env to your external frozen-asset root.
```

The audited machine used a newly created `.venv` whose installed package files were copied read-only from the original environment after the Torch wheel download failed. Runtime imports resolve from the clean repository, and `pip check` passes. CPU and a second GPU have not passed numerical parity tests.

## Preflight and start

In a PowerShell session at the repository root, set Ultralytics to avoid package installation and settings outside the clean repository. Create its settings directory under `var/`. The backend reads `.env`; shell variables override it.

```powershell
New-Item -ItemType Directory -Path var\ultralytics -Force | Out-Null
$env:YOLO_AUTOINSTALL='False'
$env:YOLO_CONFIG_DIR=(Resolve-Path var\ultralytics).Path
.\.venv\Scripts\python.exe -B research/verify_fixtures.py
.\.venv\Scripts\python.exe -B -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

For browser development, from `web/` run `npm ci` and `npm run dev -- --host 127.0.0.1`. To serve the production build locally, run `npm run typecheck`, `npm run lint`, `npm run build`, then `npm run preview -- --host 127.0.0.1 --port 4173`. Copy `web/.env.example` to ignored `web/.env.local` only when changing `VITE_API_BASE_URL`; this is compiled into a Vite build, so rebuild after changing it. Include the served browser origin in `SAAD_CORS_ORIGINS` **before** starting the API; the default covers port 5173, and a preview on port 4173 needs `http://127.0.0.1:4173`. Keep API and browser on loopback.

`GET /api/health` reports loaded device/policy metadata. `GET /api/ready` checks database access after model loading. `POST /api/analyze` returns the original response shape and saves a scan. `GET /api/scans/{id}` includes immutable predictions and separate review state. Reviews and audit events persist in `var/saad.sqlite3` by default. `GET /api/scans/{id}/report` returns JSON; `format=csv`, `format=pdf` and `format=json&download=true` provide attachments. Files and database under `var/` are runtime data, ignored by Git; back them up according to local operating policy before replacing a machine.

## Verification and troubleshooting

Stop a model-loaded API before running the GPU golden suite on the audited 8 GB GPU. Concurrent server and regression execution caused CUDA and host memory errors in one attempt; the isolated rerun passed with all numerical differences zero.

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_stage*.py' -q
.\.venv\Scripts\python.exe -B research/verify_provenance.py --original-root E:/SIH
.\.venv\Scripts\python.exe -B research/verify_original_inventory.py
.\.venv\Scripts\python.exe -B research/verify_asset_ledger.py
```

If asset validation fails, check the configured path and SHA-256 manifest before starting the server. Do not replace a tensor with a similarly named research checkpoint. If an upload fails, check the HTTP detail and byte/pixel limits; invalid requests and failed persistence remove attempted uploads. If the browser cannot connect, check the API loopback address, `VITE_API_BASE_URL` and `SAAD_CORS_ORIGINS`. No production deployment or external ingress was performed in this release validation.
