# Stage 3 verification — API and durable reviews

Date: 2026-09-23. Branch: `refactor/saad-production`. Stage 3 adds typed contracts, SQLite persistence and HTTP review routes while retaining the original `POST /api/analyze` response keys, values and ranking.

## Preflight and environment

- The branch and working tree were clean before Stage 3. The five frozen artifact sizes/SHA-256 values and `saad-live-api-v1` calibration were validated with `Settings.validate_assets()` before implementation.
- An isolated Python 3.11 environment was created at the ignored `D:/SAAD/SAAD/.venv`. The exact CUDA wheel download for Torch 2.11.0+cu128 was interrupted at 1.9/2.8 GB. As a safe local fallback, installed package files were copied **read-only** from `E:/SIH/new/Lib/site-packages` into `.venv/Lib/site-packages`. Runtime imports resolve from the clean repository `.venv`, not the original environment. A fresh wheel installation remains **UNVERIFIED**.
- Isolated runtime versions: Python 3.11.0, Torch 2.11.0+cu128, torchvision 0.26.0+cu128, Ultralytics 8.4.140, FastAPI 0.141.1, NumPy 2.4.6 and Pillow 12.3.0; CUDA device is the RTX 4070 Laptop GPU. `YOLO_AUTOINSTALL=False` was set. Runtime uploads, database, test logs and Ultralytics settings were directed to ignored paths under the clean repository. Node 24.20.0 and npm 11.19.0 were found; frontend package installation is a Stage 4 gate.
- Before changes, all 20 existing tests passed from `.venv` with maximum absolute golden difference 0.0 on every compared numeric field.

## Implementation and API contract

`backend/schemas.py` defines typed analysis, candidate, scan, review and corrected percentage-box contracts. The legacy analysis response is validated for storage but returned unchanged. `backend/store.py` stores the immutable full analysis and each prediction in separate SQLite rows. Review state and append-only review events live in separate tables. Review updates never change original boxes, scores or evidence. Scan IDs remain original `SAAD-…` survey IDs; candidate IDs remain stable within their scan.

New endpoints: `GET /api/ready`, `GET /api/scans`, `GET /api/scans/{scan_id}`, `GET /api/review-queue`, `PUT /api/scans/{scan_id}/candidates/{candidate_id}/review`, and `GET /api/scans/{scan_id}/review-events`. The existing health, model and analysis endpoints remain. Review actions are `ACCEPT`, `REJECT` and `CORRECT`, with correction boxes expressed as bounded image percentages. Each change increments a revision and appends an audit event. Partial inference is recorded when a VAE, flow or TTA score is missing and surfaced on scan retrieval; the original analysis fields and fallback values remain intact.

Writable runtime paths are confined to the clean repository. `SAAD_DATABASE_PATH`, `SAAD_MAX_UPLOAD_BYTES` (default 25 MiB) and `SAAD_MAX_IMAGE_PIXELS` (default 50 million) are configurable. Over-limit uploads return HTTP 413; malformed/unsupported uploads retain clear 400 errors. Failed image decoding, detector inference and database persistence remove the attempted upload. Database outages return 503, unknown scans/candidates return 404, and invalid review boxes return 422. Missing or mismatched model assets still fail startup before model load.

## Tests and real HTTP

```powershell
$env:YOLO_AUTOINSTALL='False'
$env:YOLO_CONFIG_DIR='D:\SAAD\SAAD\var\ultralytics'
$env:SAAD_ARTIFACT_DIR='E:\SIH\SIH_Results'
$env:SAAD_UPLOAD_DIR='D:\SAAD\SAAD\var\uploads'
$env:SAAD_OUTPUT_DIR='D:\SAAD\SAAD\var\results'
& 'D:\SAAD\SAAD\.venv\Scripts\python.exe' -B -m unittest discover -s 'D:\SAAD\SAAD\tests' -p 'test_stage*.py' -q
```

Final result: **22 tests passed, 0 failures, 0 skips**. The GhostVision API case produced seven candidates and the three domain fixtures produced 14/7/6. All compared numerical maxima were **0.0**; no fixture or tolerance changed. Stage 3 tests exercise HTTP upload through the ASGI client, image serving, scan list/detail, accept/reject/correct, audit revisions, review queue, reopen-from-disk persistence, partial status, upload byte/pixel limits, invalid reviews and database unavailability/failure cleanup.

A separate loopback Uvicorn process received a real multipart `POST /api/analyze` for the fixed GhostVision image: HTTP 200, seven candidates. `GET /api/scans/{id}` returned the saved scan, and the image URL returned HTTP 200 with 308,133 bytes. An HTTP accept returned revision 1. After terminating and restarting Uvicorn, the same scan and accepted review remained available with one audit event. Additional HTTP correction and rejection succeeded; corrected x was 10% while the immutable prediction x remained 18.022369146347%.

The first expanded suite exposed a Windows file lock when rejecting an oversized image by pixel count. The image is now closed before deleting the temporary upload. The final 22-test suite passed after that fix. No original source, checkpoint, dataset, result or original Python environment was changed during Stage 3.

## Limits carried forward

The exact Torch wheel could not be fetched reliably, so a clean pip installation is unverified. A live browser workflow, report exports, external deployment, CPU and second-GPU parity are not Stage 3 claims. The current API has no authentication and is intended for local loopback use; production network exposure needs an authentication and access-control decision. Stage 4 connects the original React application to these durable endpoints.
