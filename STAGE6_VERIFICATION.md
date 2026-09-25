# Stage 6 verification — end-to-end local release

Date: 2026-09-24. Branch: `refactor/saad-production`. The frozen original ML/backend, frontend, datasets, checkpoints and results remained read-only. The previously documented Stage 2 `pi-heif` installation into the original `E:/SIH/new` environment was not repeated or reversed. No push, merge, model training, dataset rebuild or external deployment occurred.

## Runtime and assets

The isolated clean-repository environment reports Python 3.11.0, Torch 2.11.0+cu128, CUDA 12.8, Ultralytics 8.4.140, FastAPI 0.141.1, NumPy 2.4.6, Pillow 12.3.0 and reportlab 5.0.1. Inference used `cuda:0`, NVIDIA GeForce RTX 4070 Laptop GPU. Node is 24.20.0 and npm is 11.19.0. `python -m pip check` returned “No broken requirements found.” `YOLO_AUTOINSTALL=False` and a settings path under the ignored clean-repository `var/` were used. All runtime imports came from `D:/SAAD/SAAD/.venv`.

`research/verify_fixtures.py` validated five frozen tensor byte sizes/SHA-256 values, the `saad-live-api-v1` calibration and three fixed image hashes. `research/verify_asset_ledger.py` stream-hashed **202 rows, 197 unique original asset/result files, 1,090,035,579 bytes** against `ASSET_MANIFEST.md` without traversing dataset images. `research/verify_provenance.py --original-root E:/SIH` verified all 31 original research source hashes against the archive. `research/verify_original_inventory.py` confirmed the pre-migration sizes of 37 original ML/backend and 40 original web source/config files, reading `.env.local` size only. Size checks cannot prove byte identity for original web/backend files that had no hash baseline.

## Numerical and automated gates

With the local model-loaded API stopped to avoid GPU contention:

```powershell
$env:YOLO_AUTOINSTALL='False'
$env:YOLO_CONFIG_DIR='D:\SAAD\SAAD\var\ultralytics'
$env:SAAD_ARTIFACT_DIR='E:\SIH\SIH_Results'
$env:SAAD_UPLOAD_DIR='D:\SAAD\SAAD\var\uploads'
$env:SAAD_OUTPUT_DIR='D:\SAAD\SAAD\var\results'
& 'D:\SAAD\SAAD\.venv\Scripts\python.exe' -B -m unittest discover -s tests -p 'test_stage*.py' -q
```

Final result: **25 tests passed, 0 failures, 0 skips**. The original GhostVision API fixture returned seven candidates; the three-domain stage fixtures returned 14/7/6. Every printed maximum absolute golden difference was **0.0** for detector coordinates/confidence, VAE MSE, RealNVP NLL, TTA, normalized evidence, priority and uncertainty. No fixture or tolerance was changed. Configuration, missing/corrupt assets, checkpoint compatibility, live-policy boundaries, upload/error modes, persistence, review audit, reports, global queue ordering and builder safety were included.

From `web/`, `npm run typecheck`, `npm run lint` and `npm run build` each passed. The final build transformed 1,858 modules. `npm ci` had installed 488 packages with zero reported vulnerabilities in Stage 4. The production bundle was served by Vite preview on loopback for browser validation.

## Real API and browser

The in-app browser submitted the audited GhostVision and SubPipeMini2 images through the development client and the audited AI4Shipwrecks fixed image through the **production build**. These were real HTTP multipart requests to local Uvicorn, returning 7, 6 and 14 candidates respectively. The production AI4 scan ID was `SAAD-D392F945`. In that built client, candidate 01 was accepted, 02 rejected, and 03 moved/resized and corrected. Reload retained `3 / 14 REVIEWED`. Its report displayed the three statuses and corrected percentage box; history linked to the scan/report, and the review queue listed pending candidates. The Stage 4 browser had observed JSON, CSV and PDF download events. Tall and wide sonar overlay geometry was independently measured against object-contain within 0.005 CSS pixels in Stage 4. No browser console errors were observed in those tests.

After a backend restart, read-only HTTP retrieval showed scan `SAAD-D392F945` with 14 candidates, the three saved review statuses and three audit events. Candidate 03's immutable model x remained `17.4079559467457%`; the corrected x was `24.1192925761151%` in both review state and report. `/api/health` reported `saad-live-api-v1`, `/api/ready` was ready, and the global pending queue began with the highest saved priority (`0.948991066356471`, an older scan), confirming the release queue-order fix. Actual HTTP invalid-file upload returned 400, nonexistent scan 404, and invalid report format 422. Automated tests also cover byte/pixel limits and storage failures.

An explicit missing-asset preflight with `SAAD_ARTIFACT_DIR=D:/SAAD/SAAD/var/missing-assets` failed before model load and named the absent YOLO checkpoint. Report JSON and CSV content were checked against saved reviews and immutable predictions; the PDF response signature and browser download were checked, but its rendered text was not independently parsed.

## Repository and limitations

The final tracked-file scan found no checkpoint, raw dataset, image, generated result, SQLite database, credential or local `.env`; no tracked file exceeded 1 MiB. `web/node_modules`, `web/dist`, `.venv`, `var/` and `.env.local` were ignored. The code search found no original-machine path in production `backend/` or `web/src/`; `E:/SIH` references remain in provenance, examples, tests and historical research scripts by design. The source archive is separated from production imports. `git diff --check` passed.

A **fresh from-wheel Python installation remains UNVERIFIED**. The exact 2.8 GB Torch CUDA wheel download failed during Stage 3, so installed package files were copied read-only into a new clean-repository `.venv`. The application and tests pass in that isolated environment, and `pip check` passes, but a new machine installation has not been demonstrated. Reproduction commands and required versions are in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). CPU, another GPU, external ingress, authentication and multi-user review coordination are **UNVERIFIED**. Do not expose the API publicly without a security design. Historical training/evaluation and dataset rebuilds were not run; dataset integrity caveats are in `ASSET_MANIFEST.md` and `research/STATUS.md`.

An intermediate Stage 5 golden attempt while Uvicorn and Vite were also running hit CUDA unknown/memory errors and failed parity. Both servers were stopped; the isolated golden rerun and subsequent 25-test suite passed with all numerical maxima zero. This resource constraint is documented in the local deployment guide. The exact original environment before the Stage 2 `pi-heif` incident cannot be retested without altering the read-only original environment again.
