# Evaluation and verification

SAAD's checked-in JSON fixtures are numerical **regression references**, not a detector accuracy benchmark. The fixed images and five model tensors are external. The [original direct-call baseline](../BASELINE_VERIFICATION.md) records two identical runs of a GhostVision image, full seven-candidate response fields, the three-domain stage capture, and same-environment tolerances. The original source and dataset audit ledgers remain in [SOURCE_INVENTORY.md](../SOURCE_INVENTORY.md) and [ASSET_MANIFEST.md](../ASSET_MANIFEST.md); the model manifest points to the baseline and asset ledger.

## Fixed-image method

`audit/baseline_capture.json` holds the original GhostVision API response for a 640×640 image (7 candidates after priority sorting). `audit/fixed_image_stage_capture.json` holds detector-order stage outputs for fixed AI4Shipwrecks (1728×2476, 14 candidates), GhostVision (640×640, 7) and SubPipeMini2 (2500×500, 6) images. The fixtures include image SHA-256 identities; the test checks those bytes before inference. No sonar image is committed.

The golden tests compare candidate count, class, ordering, boxes, YOLO confidence, VAE reconstruction MSE, RealNVP NLL, TTA consistency, normalized evidence, profile/action, priority and uncertainty. Original within-process repeatability had maximum absolute difference **0.0** for the measured fields on the audited GPU. The same-environment gates allow ≤1 pixel box difference, ≤1e-5 YOLO confidence, ≤1e-6 VAE MSE, ≤1e-2 Flow NLL and ≤1e-3 for TTA, normalized evidence, priority and uncertainty. Those margins are engineering tolerances around the observed zero repeat difference; they do not establish cross-device parity. Do not change fixture bytes or tolerances to make a run pass.

The final fresh-install run on the audited Windows RTX 4070 Laptop GPU passed all **25** backend tests with 0 failures/skips. Every reported maximum golden numerical difference was **0.0**. The observed environment was Python 3.11.0, Torch 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA 12.8, Ultralytics 8.4.140, FastAPI 0.141.1, NumPy 2.4.6, Pillow 12.3.0 and reportlab 5.0.1. `pip check` passed. This was a new isolated wheel installation in the clean repository; another machine, CPU numerical inference and a second GPU remain unverified. The original `E:/SIH/new` environment was not used for that final fresh-install result.

## Commands

For GitHub-hosted, asset-independent checks (13 of the 25 existing tests), use:

```powershell
.\.venv\Scripts\python.exe -B tests/run_ci_backend.py
cd web
npm ci
npm run typecheck
npm run lint
npm run build
```

[CI](CI.md) lists each selected test and why the other 12 need external assets or model-loaded imports. Hosted CI uses CPU Torch and does **not** claim the golden result.

For the authorized local GPU suite, set `SAAD_ARTIFACT_DIR` in `.env` and set `YOLO_AUTOINSTALL=False` plus `YOLO_CONFIG_DIR` to an ignored directory under `var/`. Supply the three fixed images at their original paths or point `SAAD_GOLDEN_IMAGE_DIR` to an authorized mirror. Stop any model-loaded API first, then from the repository root run:

```powershell
New-Item -ItemType Directory -Path var\ultralytics -Force | Out-Null
$env:YOLO_AUTOINSTALL='False'
$env:YOLO_CONFIG_DIR=(Resolve-Path var\ultralytics).Path
.\.venv\Scripts\python.exe -B research/verify_fixtures.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_stage*.py' -q
```

`research/verify_fixtures.py --run` runs the two focused golden tests after preflight. `research/verify_asset_ledger.py` hashes the **202 manifest rows / 197 unique external files** (1,090,035,579 bytes) without traversing dataset images. `research/verify_provenance.py --original-root E:/SIH` verifies all 31 archived scripts by SHA-256. `research/verify_original_inventory.py` checks historical sizes for 37 ML/backend and 40 web original source/config files; these size-only records cannot prove byte identity for non-archived files. All those original locations are read-only.

## Application checks and interpretation

The audited local browser sent real multipart uploads for GhostVision, SubPipeMini2 and AI4Shipwrecks, yielding 7, 6 and 14 candidates. The AI4 scan used the built Vite client. Accept, reject and corrected percentage boxes persisted after reload and backend restart; the original model box remained unchanged, three review events were saved, history/direct navigation and pending queue worked, and JSON/CSV/PDF attachments downloaded. Automated API tests covered upload failures, storage errors, reports and queue ordering. PDF signature and browser download were checked; rendered PDF text was not independently extracted.

These counts measure pipeline repeatability and application integration, not precision, recall or real-world debris identification accuracy. The AI4 fixture has an empty label file, GhostVision has two class-0 labels, and SubPipeMini2 has one; candidate count is not ground-truth object count. The prepared detector training split has 45 more label files than image files, the source audit found 685 missing annotations, and the SubPipeMini2 temporal-neighbor split report found adjacent frames across train/validation/test. These caveats constrain any offline performance claim; see [the asset ledger](../ASSET_MANIFEST.md) and [Limitations](LIMITATIONS.md).
