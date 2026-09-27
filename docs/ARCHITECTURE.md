# Architecture

SAAD is a local side-scan sonar review application. The browser submits an image to FastAPI; the backend loads the original frozen model set once per process, scores detector candidates, applies the deployed evidence policy, and stores the result for human review. The original `POST /api/analyze` response remains the compatibility contract.

```text
web/src/pages + components
  -> web/src/lib/api.ts (VITE_API_BASE_URL)
  -> backend/main.py:app -> routes.py
       -> config.py + config/model_manifest.json + external frozen tensors
       -> saad_inference/registry.py (one model set per process)
       -> saad_inference/service.py
            -> detector.py (YOLO26s)
            -> preprocessing.py (context crop, grayscale, 256x256 resize)
            -> vae.py (ConvVAE reconstruction MSE and encoder mu)
            -> realnvp.py (NLL of standardized mu)
            -> tta.py (brightness, contrast, seeded noise)
            -> evidence/live_api_v1.py (saad-live-api-v1)
            -> response.py (priority sort, IDs and original API fields)
       -> store.py (SQLite scans, predictions, reviews, events)
       -> reporting.py (JSON, CSV, PDF from saved state)
```

The model registry validates five tensor sizes and SHA-256 hashes plus the pinned live calibration before loading. During extraction, the frozen ConvVAE checkpoint matched 71 state entries and RealNVP matched 56 by exact key and shape; latent mean and standard deviation each had 128 elements. No checkpoint was rewritten. `main.py` remains the Uvicorn entry point and compatibility export; `vae_inference.py`, `realnvp_inference.py` and `tta_inference.py` remain import shims for older callers. `routes.py` owns upload validation, the static upload mount and FastAPI route wiring. A process that imports the routes needs the configured external assets, including in asset-dependent tests.

## Inference and policy boundary

YOLO26s predicts at `imgsz=640` and `conf=0.05`. The candidate crop uses the original context, coordinate rounding, Pillow grayscale conversion and bilinear 256×256 interpolation. The ConvVAE uses deterministic encoder `mu` and reconstruction MSE; RealNVP scores standardized `mu` as negative log likelihood. Three TTA variants use brightness ×1.15, contrast ×0.85 and Gaussian noise with seed 42 and sigma 5. Same-class matches use IoU 0.30 and a 0.70 IoU / 0.30 confidence-agreement score. The full fixed-image regression checks these details against [the captured baseline](../BASELINE_VERIFICATION.md).

Only `saad-live-api-v1` is deployed. It clips raw scores to normal-reference P5/P95 ranges and combines YOLO/VAE/flow at 0.50/0.20/0.30. Its priority is 0.80 base evidence + 0.10 corroboration + 0.10 TTA. Its uncertainty is 0.50 disagreement + 0.30 (1−TTA) + 0.20 (1−agreement). These are review-priority scales, not probabilities. The offline Evidence Engine v3 shares normalization inputs but has different priority bonuses/penalties, uncertainty and action/profile rules and no TTA. Its historical source is preserved in `research/original/Evidence/build_saad_evidence_engine_v3.py`; it is not imported or selectable by the live service. See [Research](RESEARCH.md).

The service preserves original candidate ordering/fallback behavior before `response.py` sorts by priority and renumbers IDs. A failed per-candidate anomaly/TTA score can leave partial inference; scan retrieval exposes that state. The golden suite asserts available scores so a superficially successful response cannot count as parity.

## API, persistence and browser

`POST /api/analyze` returns the original candidate response and saves a scan. `GET /api/health`, `/api/model` and `/api/ready` expose model and storage status. Scan list/detail, pending review queue, review-event and JSON/CSV/PDF report routes are in `routes.py`. `PUT /api/scans/{scan_id}/candidates/{candidate_id}/review` accepts `ACCEPT`, `REJECT` or `CORRECT` with a bounded percentage box. The SQLite store retains immutable model predictions separately from current reviews and append-only review events. Corrected boxes change the display and reports, never the saved model box. Pending reviews are ordered by saved model priority.

`web/src/lib/api.ts` is the typed transport client; `adapt.ts` combines saved predictions and review state for display. The React pages support image upload, candidate inspection, review, history, queue and reports. Direct scan URLs and refresh read SQLite through the API. The sonar viewer places percentage-coordinate boxes against the rendered `object-contain` image rectangle. The browser is verified only against a loopback API; see [Limitations](LIMITATIONS.md).
