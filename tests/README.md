# Test guide

Tests use Python's `unittest`. There are 25 `test_stage*.py` methods: 13 asset-independent hosted checks and 12 local-only tests needing frozen tensors, fixed images or model-loaded route imports. The exact per-file split is maintained in [CI](../docs/CI.md#exact-backend-test-split).

## Asset-independent suite

From the repository root in the installed backend environment:

```powershell
.\.venv\Scripts\python.exe -B tests/run_ci_backend.py
```

[run_ci_backend.py](run_ci_backend.py) explicitly selects:

- Three configuration cases: missing asset declaration, offline-policy rejection and missing-asset preflight failure.
- Three extracted-module cases: detector coordinates, synthetic grayscale crops and seeded augmentation.
- Four live-policy boundary cases: missing/non-finite raw scores, threshold ties and priority/uncertainty calculations.
- Three dataset safety cases: opt-in, existing output and forbidden output boundaries.

Expected: 13 tests, no skips and `OK`; the selector fails on count mismatch. It sets CPU and an artifact-root declaration solely for policy import. No weights or sonar images are opened for inference. Missing-asset validation is tested; real five-asset hash validation is local-only.

## Full local suite

Use the audited CUDA installation from [Setup](../docs/SETUP.md), the five manifest-matching tensors and three external fixture images. Set `SAAD_ARTIFACT_DIR` in `.env`; optionally set `SAAD_GOLDEN_IMAGE_DIR` to an authorized mirror of the fixture basenames. Stop model-loaded API processes first.

```powershell
New-Item -ItemType Directory -Path var\ultralytics -Force | Out-Null
$env:SAAD_DEVICE='cuda:0'
$env:YOLO_AUTOINSTALL='False'
$env:YOLO_CONFIG_DIR=(Resolve-Path var\ultralytics).Path
.\.venv\Scripts\python.exe -B research/verify_fixtures.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_stage*.py' -q
```

Expected: preflight passes, 25 tests run, no skips and `OK`. The 12 additional tests cover real asset mismatch detection, VAE/flow state compatibility and latent tensors, two golden cases, four orchestration cases and two persisted API cases. Route imports load the models even in tests that later mock a handler.

A focused numerical run is `research/verify_fixtures.py --run`, which preflights then runs the two golden methods. It does not replace the full suite.

## Golden references

[audit/baseline_capture.json](../audit/baseline_capture.json) contains the seven-candidate GhostVision API reference. [audit/fixed_image_stage_capture.json](../audit/fixed_image_stage_capture.json) contains three detector-order domain cases with 14/7/6 candidates. Image hashes are checked before scoring; images are not committed.

[Evaluation](../docs/EVALUATION.md) defines the exact tolerances and distinction between API-normalized and stage fields. Candidate counts, class/order and categorical outputs must match. Raw/model/priority differences must remain inside their original gates. Tests print `STAGE2_GOLDEN_MAX_ABSOLUTE_DELTAS`; previously measured values were all 0.0.

A missing-asset import failure, native process crash or skipped golden test is not numerical parity. A numerical mismatch requires diagnosis before continuing; do not change fixtures/tolerances to pass.

## Other gates and scope

From `web/`: `npm ci`, `npm run typecheck`, `npm run lint`, `npm run build`. Repository hygiene from root: `python -B scripts/check_repository_hygiene.py`.

Hosted CI performs those gates plus the 13 tests and dependency validation. It does not run GPU regression, real HTTP inference or a browser workflow. Historical loopback browser evidence and fresh-install GPU results are documented in [Evaluation](../docs/EVALUATION.md); another machine or CPU numerical parity remains unverified.
