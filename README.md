# SAAD

**SAAD (Seabed-Aware Anthropogenic Anomaly Detector)** helps a reviewer inspect possible objects and anomalies in side-scan sonar images. It combines a frozen YOLO26s detector, ConvVAE reconstruction, RealNVP latent novelty and three test-time augmentations into a candidate evidence profile. A human can accept, reject or correct each candidate; scans, decisions and reports persist locally.

The application has been verified for **local loopback use** on one Windows RTX 4070 Laptop GPU machine. The repository contains no model checkpoints or sonar datasets. It is not an authenticated network service, and its evidence scores are review priorities rather than probabilities. See [Limitations](docs/LIMITATIONS.md).

## How it works

```text
Sonar image -> YOLO26s candidates -> grayscale context crops
            -> ConvVAE reconstruction + RealNVP latent NLL
            -> brightness / contrast / seeded-noise TTA
            -> saad-live-api-v1 evidence and review priority
            -> saved scan -> human review -> JSON / CSV / PDF report
```

The backend preserves the original `/api/analyze` response. SQLite stores immutable model predictions separately from corrected boxes, current review decisions and an append-only event trail. The React client supports upload, candidate inspection, saved history, a pending review queue, direct scan navigation and reports. [Architecture](docs/ARCHITECTURE.md) describes modules, routes and the policy formulas.

**Policy distinction:** `saad-live-api-v1` is the deployed default. The archived offline Evidence Engine v3 shares some normalization values but has different priority, uncertainty and action rules and does not use TTA. It is research provenance, not a selectable live policy. No model algorithm or evidence policy was changed during repository cleanup.

## Quick start on the audited Windows setup

Prerequisites are Python 3.11, Node 24/npm 11, the official CUDA 12.8 PyTorch wheel stack and authorized copies of the five original frozen tensors. A new isolated Python environment was installed and passed `pip check` on the audited machine. The precise package snapshot is `backend/requirements-original-windows-cu128.lock.txt`; install `reportlab==5.0.1` in addition for PDF exports.

From the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r backend\requirements-original-windows-cu128.lock.txt
.\.venv\Scripts\python.exe -m pip install reportlab==5.0.1
Copy-Item .env.example .env
# Set SAAD_ARTIFACT_DIR in .env to an authorized external asset root.
New-Item -ItemType Directory -Path var\ultralytics -Force | Out-Null
$env:YOLO_AUTOINSTALL='False'
$env:YOLO_CONFIG_DIR=(Resolve-Path var\ultralytics).Path
.\.venv\Scripts\python.exe -B -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

In another PowerShell session:

```powershell
cd web
npm ci
npm run dev -- --host 127.0.0.1
```

Open the local Vite URL. The backend is bound to `127.0.0.1:8000`; the browser defaults to that API address. [Setup](docs/SETUP.md) covers the complete preflight, CORS, production preview and troubleshooting steps. Do not expose the API outside loopback.

## External assets and configuration

`config/model_manifest.json` pins the five original tensors by relative path, byte size and SHA-256; startup validates them and the `saad-live-api-v1` calibration before loading. Supply authorized assets outside Git and set `SAAD_ARTIFACT_DIR` in ignored `.env`, or use individual exact-matching overrides. The committed `config/live_api_v1.json` contains the live calibration. Optional dataset and runtime settings are listed in `.env.example`; uploads, outputs and SQLite default to ignored `var/`. The three fixed sonar images used by the golden suite are also external. [ASSET_MANIFEST.md](ASSET_MANIFEST.md) records source hashes and dataset caveats. There is no bundled checkpoint, dataset, screenshot or public asset download.

## Verification

On the audited Windows RTX 4070 machine, the fresh-install backend suite passed **25/25** with every reported golden maximum numerical difference **0.0**: seven GhostVision API candidates and three fixed domain stage cases with 14/7/6 candidates. Real loopback HTTP and browser runs exercised uploads, persistent reviews/corrections, history, queue and JSON/CSV/PDF downloads. These are migration and integration checks, not accuracy benchmarks. The original captured values and tolerances remain in [BASELINE_VERIFICATION.md](BASELINE_VERIFICATION.md); methodology and scope are in [Evaluation](docs/EVALUATION.md).

GitHub Actions runs **13 asset-independent** backend tests, `pip check`, frontend `npm ci`/typecheck/lint/build and repository hygiene. The other **12** backend tests need external tensors, fixed images or model-loaded routes and run locally. [CI](docs/CI.md) lists the exact split. With the authorized external assets and a stopped API, run the full local suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -B research/verify_fixtures.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_stage*.py' -q
```

## Repository map

| Path | Purpose |
| --- | --- |
| `backend/`, `config/` | FastAPI, frozen inference, model/calibration validation, SQLite and reports. |
| `web/` | React/Vite browser client. |
| `tests/`, `audit/` | Unit/API tests and small numeric/hash regression fixtures; no image bytes. |
| `ml/` | Historical offline research selection and guarded dataset builder dry runs. |
| `research/original/`, `research/source_manifest.json` | 31 original source scripts and SHA-256 provenance. |
| `research/verify_*.py`, `scripts/` | Asset, source and repository hygiene checks. |
| `docs/` | Permanent architecture, setup, evaluation, CI, research and limitations guides. |

The dated [source inventory](SOURCE_INVENTORY.md), [external asset ledger](ASSET_MANIFEST.md) and [original baseline](BASELINE_VERIFICATION.md) remain at the repository root because verification tools and the model manifest refer to them. [Research](docs/RESEARCH.md) explains the 31-script archive, supported offline evaluation, guarded builders and historical workflows.

## Provenance and acknowledgements

This migration preserves the original local SAAD model set and research source bytes; their SHA-256 identities are recorded in the manifests. The fixed examples come from AI4Shipwrecks, GhostVision and SubPipeMini2 source domains, supplied externally. Dataset/model redistribution rights and independent code ownership have not been established, and this repository has no `LICENSE` file. Consult the original rights holders and source terms before reuse. See [Research](docs/RESEARCH.md) and [Limitations](docs/LIMITATIONS.md) for the precise boundaries.
