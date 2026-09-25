# SAAD

SAAD combines the original YOLO26s detector, ConvVAE, RealNVP latent flow, three-variant TTA and the deployed live API evidence policy. Stages 1–6 have been validated for local loopback use: frozen external assets are checked before loading, modular inference preserves the golden fixtures, the API stores scans and human reviews durably, the React application uses those saved records, and original research sources are archived with hash provenance. A fresh isolated Python wheel installation passed on the audited Windows machine during publication review. Installation on another machine and external deployment remain unverified; see [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).

## Application and architecture

The React/Vite client uploads sonar images to FastAPI. The backend validates external frozen assets at startup, detects candidates, computes VAE/flow/TTA evidence, applies `saad-live-api-v1`, and returns the original `/api/analyze` response. SQLite stores scans, immutable model predictions, separate human reviews, and review events. The browser supports candidate inspection, accept/reject/correct, saved history and review queue, and JSON/CSV/PDF reports. `ARCHITECTURE.md` documents the module dependency flow; `research/STATUS.md` identifies historical experiments that are outside the production import path.

Only local loopback operation on the audited Windows RTX 4070 machine has passed end-to-end validation. The API has no authentication. Do not expose it on a network interface.

## Windows setup

Prerequisites: Windows, Python 3.11, Node 24/npm 11, a compatible NVIDIA CUDA device for the audited GPU path, adequate disk space for the large CUDA wheels, and access to the frozen external assets. The observed stack uses Torch 2.11.0+cu128 and Ultralytics 8.4.140. `backend/requirements-original-windows-cu128.lock.txt` records the original Python environment; `reportlab==5.0.1` is additionally required for PDF reports. `backend/requirements.txt` lists direct requirements but is not the parity installation recipe. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for operating details and [PUBLICATION_REVIEW.md](PUBLICATION_REVIEW.md) for the current fresh-install result.

1. Create a new virtual environment in this clean repository and install the pinned Windows/CUDA stack:

    py -3.11 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r backend\requirements-original-windows-cu128.lock.txt
    .\.venv\Scripts\python.exe -m pip install reportlab==5.0.1
    .\.venv\Scripts\python.exe -m pip check

   The clean install passed on the audited machine using the exact CUDA wheel and pins. Pip twice requested the 2.75 GB Torch wheel during the first attempt; the completed official wheel was installed locally into the new environment, then the lockfile installation completed. `PUBLICATION_REVIEW.md` records the commands and test result. Do not substitute a different Torch build and claim golden parity.
2. Supply the five original tensors and `live_api_v1.json` calibration from an authorized copy of the original SAAD results, or another authorized source with the **same hashes** in `config/model_manifest.json`. No checkpoint, calibration download, dataset bundle, or redistribution license is provided by this repository. Place them outside Git in the manifest's relative directory layout. The three original test images are likewise external; use an authorized prepared dataset for golden tests. Copy `.env.example` to a local `.env` and set `SAAD_ARTIFACT_DIR` to that external asset root. On the audited machine, `E:/SIH/SIH_Results` is the read-only source.

    Copy-Item .env.example .env
3. From the repository root, create the ignored Ultralytics settings directory, disable automatic package installation, then run the preflight and API:

    New-Item -ItemType Directory -Path var\ultralytics -Force
    $env:YOLO_AUTOINSTALL='False'
    $env:YOLO_CONFIG_DIR=(Resolve-Path var\ultralytics).Path
    .\.venv\Scripts\python.exe -B research/verify_fixtures.py
    .\.venv\Scripts\python.exe -B -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000

The API uses var/uploads, var/results and var/saad.sqlite3 under the clean repository by default. These paths are ignored by Git and writable runtime paths are validated to remain inside the clean repository. Startup fails before model loading if a checkpoint is missing or fails size/SHA-256 validation. Set `YOLO_AUTOINSTALL=False` and `YOLO_CONFIG_DIR` to a directory under `var/` before starting the API; create that directory first. This prevents Ultralytics from installing packages or writing settings outside the clean repository.

To run the frontend locally:

    cd web
    npm ci
    npm run dev

The frontend uses `VITE_API_BASE_URL` (default `http://127.0.0.1:8000`) for all API calls. Copy `web/.env.example` to ignored `web/.env.local` if the backend runs at a different origin. The API CORS origins must include the frontend origin. Reviews and corrected boxes are saved in SQLite, so scan and report URLs work after reload and direct navigation. JSON, CSV and PDF report attachments come from the backend. Keep the API on loopback until authentication and access control are designed.

## Configuration

backend/config.py loads local .env without overriding process environment variables. Relative paths resolve from the repository root. SAAD_ARTIFACT_DIR supplies the external root. SAAD_YOLO_WEIGHTS, SAAD_VAE_WEIGHTS, SAAD_FLOW_WEIGHTS, SAAD_LATENT_MEAN and SAAD_LATENT_STD can override individual paths only if their bytes match the selected manifest. SAAD_MODEL_MANIFEST selects a model manifest; SAAD_CALIBRATION_FILE selects a live-policy calibration with a matching manifest hash. SAAD_DATASET_DIR is an optional read-only prepared dataset. SAAD_DEVICE is auto, cpu or an available cuda:N. SAAD_UPLOAD_DIR, SAAD_OUTPUT_DIR, SAAD_DATABASE_PATH, SAAD_MAX_UPLOAD_BYTES, SAAD_MAX_IMAGE_PIXELS, SAAD_API_PUBLIC_URL and SAAD_CORS_ORIGINS configure runtime storage, upload bounds and API origins. `VITE_API_BASE_URL` configures the React API client.

config/model_manifest.json identifies the five runtime tensors by size and SHA-256. config/live_api_v1.json contains the exact deployed percentile and weight values. The original loaders, crop, detector, VAE, flow, TTA and evidence calculations remain intact. `backend/main.py:app` remains the FastAPI entry point. `backend/routes.py` owns route wiring; `backend/saad_inference/registry.py` loads the model set once at startup; `service.py` coordinates inference; `response.py` preserves candidate sorting, summary and response fields. See [ARCHITECTURE.md](ARCHITECTURE.md) for the dependency map.

## Distinct evidence policies

| Policy | Status | Shared inputs | Distinguishing behavior |
| --- | --- | --- | --- |
| saad-live-api-v1 | Deployed default and golden baseline | Validation-normal P5/P95 and 0.50/0.20/0.30 YOLO/VAE/flow weights | Priority is 0.80 base + 0.10 corroboration + 0.10 TTA; live disagreement/TTA/agreement uncertainty and live action/profile rules. |
| saad-offline-evidence-v3 | Research only; not selectable for deployment | Same normal-reference values and primary weights | Different priority bonuses and penalty, different ambiguity/disagreement uncertainty and action/profile rules; no TTA. |

The offline v3 configuration remains externally at E:/SIH/SIH_Results/saad_evidence_engine_v3/engine_config.json and its builder at E:/SIH/Evidence/build_saad_evidence_engine_v3.py. The final offline evaluation also explored YOLO_FLOW_MEAN. Neither replaces the deployed API. Scores are review-priority scales, not calibrated probabilities. BASELINE_VERIFICATION.md records the numerical baseline.

## Regression checks

After configuring the external artifact root, run:

    python -B -m unittest discover -s tests -p "test_stage*.py" -v

The golden test checks the original API candidate response on the fixed GhostVision image and all detector/crop/VAE/flow/TTA/evidence fields on one image from each prepared test domain. Image SHA-256 is checked before inference. `SAAD_GOLDEN_IMAGE_DIR` can point to the same three basenames on another machine. These images are not bundled. Stop the model-loaded API before running the GPU suite on the audited 8 GB device. Temporary request uploads stay in configured clean-repository storage and are removed after their hashes are verified.

SOURCE_INVENTORY.md, ASSET_MANIFEST.md, MIGRATION_MAP.md and REFACTOR_PLAN.md contain the audit. STAGE1_VERIFICATION.md through STAGE6_VERIFICATION.md record test results and limits. Research status and reproducible commands are in [research/STATUS.md](research/STATUS.md); the 31-script source hash ledger is `research/source_manifest.json`. The local API exposes `GET /api/ready`, `/api/scans`, `/api/review-queue`, scan detail, review-event and JSON/CSV/PDF report retrieval, and `PUT /api/scans/{scan_id}/candidates/{candidate_id}/review` with `ACCEPT`, `REJECT` or `CORRECT`. The original `POST /api/analyze` response is unchanged and persists its result. Run `npm run typecheck`, `npm run lint` and `npm run build` in `web/` for frontend checks.

## Release limits

The public repository contains no model weights, calibration file, dataset image or screenshot. The 31-script research archive preserves source bytes and original path literals for provenance; those scripts are not a portable production workflow. CPU/other GPU numerical parity, historical training reruns, multi-user coordination and external deployment are unverified. Dataset quality caveats and the distinct offline Evidence Engine v3 policy are documented in `ASSET_MANIFEST.md` and `research/STATUS.md`. See `RELEASE_CHECKLIST.md` and `PUBLICATION_REVIEW.md` before opening a pull request.
