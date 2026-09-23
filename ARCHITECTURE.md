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
