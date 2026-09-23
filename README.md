# SAAD

SAAD combines the original YOLO26s detector, ConvVAE, RealNVP latent flow, three-variant TTA and the deployed live API evidence policy. Stages 1–5 are complete: frozen external assets are validated before loading, modular inference preserves the golden fixtures, the API stores scans and human reviews durably, the React application uses those saved records, and the original research sources are archived with hash provenance.

## Windows setup

Use Python 3.11. The exact original Windows/CUDA environment snapshot is in backend/requirements-original-windows-cu128.lock.txt. It records Torch 2.11.0+cu128 and Ultralytics 8.4.140. Availability of those exact wheels on a fresh machine is UNVERIFIED. backend/requirements.txt lists direct backend requirements.

1. Create a virtual environment in this clean repository and install the direct backend requirements:

    py -3.11 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt

   For GPU parity, install a CUDA-enabled Torch build matching the original environment before running inference. The exact wheel acquisition and a fresh install have not yet been tested. Check the locked snapshot before changing any package version.
2. Copy .env.example to a local .env. Set SAAD_ARTIFACT_DIR to an external directory containing the five original files at the relative paths in config/model_manifest.json. On this machine, E:/SIH/SIH_Results is the audited read-only source.

    Copy-Item .env.example .env
3. From the repository root, run the preflight and API:

    python -B -c "import sys; sys.path.insert(0, 'backend'); from config import Settings; s=Settings.from_environment(); s.validate_assets(); print(s.manifest['model_set_id'], s.device)"
    python -B -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000

The API uses var/uploads, var/results and var/saad.sqlite3 under the clean repository by default. These paths are ignored by Git and writable runtime paths are validated to remain inside the clean repository. Startup fails before model loading if a checkpoint is missing or fails size/SHA-256 validation. Set `YOLO_AUTOINSTALL=False` and `YOLO_CONFIG_DIR` to a directory under `var/` before starting the API; create that directory first. This prevents Ultralytics from installing packages or writing settings outside the clean repository.

To run the frontend locally:

    cd web
    npm ci
    npm run dev

The frontend uses `VITE_API_BASE_URL` (default `http://127.0.0.1:8000`) for all API calls. Set it in `web/.env.local` if the backend runs at a different origin. The API CORS origins must include the frontend origin. Reviews and corrected boxes are saved in SQLite, so scan and report URLs work after reload and direct navigation. JSON, CSV and PDF report attachments come from the backend. Keep the API on loopback until authentication and access control are designed.

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

The golden test checks the original API candidate response on the fixed GhostVision image and all detector/crop/VAE/flow/TTA/evidence fields on one image from each prepared test domain. Image SHA-256 is checked before inference. SAAD_GOLDEN_IMAGE_DIR can point to the same three basenames on another machine. Temporary request uploads stay in configured clean-repository storage and are removed after their hashes are verified.

SOURCE_INVENTORY.md, ASSET_MANIFEST.md, MIGRATION_MAP.md and REFACTOR_PLAN.md contain the audit. STAGE1_VERIFICATION.md through STAGE5_VERIFICATION.md record test results and limits. Research status and reproducible commands are in [research/STATUS.md](research/STATUS.md); the 31-script source hash ledger is `research/source_manifest.json`. The local API exposes `GET /api/ready`, `/api/scans`, `/api/review-queue`, scan detail, review-event and JSON/CSV/PDF report retrieval, and `PUT /api/scans/{scan_id}/candidates/{candidate_id}/review` with `ACCEPT`, `REJECT` or `CORRECT`. The original `POST /api/analyze` response is unchanged and persists its result. Run `npm run typecheck`, `npm run lint` and `npm run build` in `web/` for frontend checks.
