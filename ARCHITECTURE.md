# Stage 2 backend dependency map and movement plan

This map was written **before** the Stage 2 code moves. The preserved entry point is `backend/main.py:app`; `POST /api/analyze` and its response fields remain compatible. The golden fixtures in `audit/` are the numerical gate after each move.

## Current dependencies

```text
backend/main.py
  -> backend/config.py (Settings, manifest and SHA-256 checks)
  -> Ultralytics YOLO (checkpoint and predict)
  -> backend/vae_inference.py (ConvVAE, loader, crop, MSE)
  -> backend/realnvp_inference.py (RealNVP, loader, latent NLL)
  -> backend/tta_inference.py (perturb, detect, match, consistency)
  -> inline percentile_score/build_evidence (deployed policy)
  -> inline upload, orchestration, response and FastAPI endpoints
```

The original `E:/SIH/Evidence/build_saad_evidence_engine_v3.py` and its offline result configuration are **not** imported by the API. The production policy is `saad-live-api-v1` and uses TTA. Offline v3 has different priority and uncertainty rules and no TTA.

## Proposed dependency flow

```text
main.py -> routes.py -> service.py -> detector.py
                  |              -> preprocessing.py
                  |              -> vae.py -> preprocessing.py
                  |              -> realnvp.py
                  |              -> tta.py -> detector.py
                  |              -> evidence/live_api_v1.py
                  |              -> response.py
                  -> registry.py -> config.py + external validated assets
```

`registry.py` owns one startup model set. `service.py` owns request inference and original error/fallback semantics. `response.py` owns the final response dictionary, candidate ordering and summary. `routes.py` owns FastAPI endpoints and the upload mount. `main.py` remains a compatibility entry point and reexports the symbols used by the Stage 1 golden test. Only the live policy enters the service.

## File movement and verification gates

| Step | Current source | Destination | Compatibility / gate |
| --- | --- | --- | --- |
| 2 | `vae_inference.py` crop helpers; `main.py` detector call and box decoding | `saad_inference/preprocessing.py`, `saad_inference/detector.py` | Keep Pillow operations, crop context/rounding, detector `source` and kwargs, box clamp and order; run all Stage 1 tests before commit. |
| 3 | `vae_inference.py` architecture/loader/scoring; `realnvp_inference.py` architecture/loader/scoring | `saad_inference/vae.py`, `saad_inference/realnvp.py`; old files become import shims | Check saved state keys/shapes and fixed outputs first; preserve `weights_only=False`, load strictness, `mu`, MSE and flow normalization; run checkpoint and golden tests before commit. |
| 4 | `tta_inference.py`; `main.py` percentile and `build_evidence` | `saad_inference/tta.py`, `saad_inference/evidence/live_api_v1.py`; old TTA file becomes shim | Preserve seeded variants, same-class IoU matching, missing/non-finite scores, threshold ties and all live formulas; add boundary tests and run golden suite. |
| 5 | `main.py` model initialization, `analyze`, response assembly and routes | `saad_inference/registry.py`, `service.py`, `response.py`, `backend/routes.py`; `main.py` compatibility exports | One model load, original API error/fallback behavior and response keys, golden suite and route tests. |

Each extraction uses the Stage 1 manifest and calibration unchanged. Research scripts, the React app and persistence are outside Stage 2. No numerical tolerance will be widened to make a move pass.

## Implemented Stage 2 backend

The plan above is now implemented. `backend/main.py:app` still serves the original routes through `backend/routes.py`; imports used by the Stage 1 fixture remain available from `main`. Route startup creates one `Settings` and one `ModelRegistry`. `load_models` validates sizes and SHA-256 values before loading the frozen YOLO, ConvVAE and RealNVP checkpoint set. The route hands this registry to `service.analyze` for each upload. No model is loaded per request.

`service.py` retains the original upload validation, image opening, detector call, candidate loop, anomaly-score fallback and TTA/evidence sequence. `response.py` owns the original priority sort, candidate renumbering, summary and response dictionary. `preprocessing.py` retains context crop, grayscale and interpolation; `detector.py` retains YOLO arguments and coordinate clamping. `vae.py`, `realnvp.py` and `tta.py` contain the exact runtime definitions. The old `vae_inference.py`, `realnvp_inference.py` and `tta_inference.py` paths remain compatibility import shims.

`evidence/live_api_v1.py` is the only evidence policy imported by the service. It implements `saad-live-api-v1`, with TTA and the original priority/uncertainty/profile rules. The offline Evidence Engine v3 is still at the read-only original research location and has distinct bonus, penalty, uncertainty and action rules. It is not exposed as a production option. The common P5/P95 values and weights do not imply policy equivalence.

The response is an untyped dictionary in Stage 2 to preserve its exact keys and null/fallback behavior. Typed request/response schemas and durable scan/review state belong to Stage 3. The model registry still loads at module import to preserve the `uvicorn main:app` startup contract; this is one load per process, so multiple worker processes each load their own model set.

## Stage 3 API and persistence

`backend/schemas.py` validates legacy analysis output and types scan detail, review actions and corrected percentage boxes. `backend/routes.py` keeps the original analysis response, then writes it through `backend/store.py`. The store has separate SQLite tables for scans, immutable candidate predictions, current review state and append-only review events. Retrieval joins reviews onto a copy of the original analysis; corrections never overwrite model boxes or scores. The review queue selects pending candidates from those tables. `GET /api/ready` checks the database after model startup.

The upload path is bounded by configurable byte and pixel limits. Runtime upload, output and database paths are confined to the clean repository. A successful analysis retains its image; invalid uploads, failed detector inference and failed persistence clean up the attempted image. The API still loads the frozen model registry once per process and uses only `saad-live-api-v1` for scoring. Stage 4 adds a typed frontend client and server-backed report exports.

## Stage 4 browser client and reports

`web/src/lib/api.ts` is the typed transport boundary. It takes `VITE_API_BASE_URL`, uploads an image, retrieves saved scans and the pending queue, submits reviews, reads health and obtains report attachments. `web/src/lib/adapt.ts` turns immutable backend predictions plus separate review state into display candidates. A corrected box affects the displayed overlay and report while the saved model box remains intact. `Analysis.tsx` loads by scan ID on every direct visit and refresh; `NewScan.tsx` navigates to the saved ID. `History.tsx`, `ReviewQueue.tsx` and `Reports.tsx` read server state. The existing viewer calculates the `object-contain` image rectangle before drawing percentage-coordinate boxes, including tall and wide sonar images.

`backend/reporting.py` builds JSON, CSV and PDF exports from the saved scan, current review state and append-only events; the report includes the model-set and live-policy IDs. The JSON API response remains available without an attachment; `download=true` adds an attachment header. CSV carries both model and corrected boxes in separate columns and quotes spreadsheet formula prefixes. The browser uses these server attachments for all three export buttons. No frontend local storage controls review truth. The service remains a local loopback application without authentication or multi-user review coordination.
