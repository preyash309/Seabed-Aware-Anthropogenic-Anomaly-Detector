# GitHub Actions CI boundary

`.github/workflows/ci.yml` runs on pull requests to `main` and pushes to `main`. It has three independent jobs and read-only repository permissions. No GitHub secret, private image, dataset or frozen checkpoint is required or uploaded.

| Job | Runner and command | Claim |
| --- | --- | --- |
| Backend | Ubuntu 24.04, Python 3.11, PyTorch/torchvision 2.11.0/0.26.0 CPU wheels, `backend/requirements.txt`, `pip check`, `python -B tests/run_ci_backend.py` | 13 existing asset-independent tests pass. No model inference or GPU parity is tested. |
| Frontend | Ubuntu 24.04, Node 24, `npm ci`, `npm run typecheck`, `npm run lint`, `npm run build` in `web/` | The committed client dependency lock, types, lint rules and production compilation pass. |
| Repository hygiene | Ubuntu 24.04, `python3 scripts/check_repository_hygiene.py` | Tracked files and all reachable Git blobs are checked for secret signatures, model/data/runtime file types and files above 1 MiB. `.gitignore` coverage is asserted. A single historical 39-byte `.env.local` is allowed only if it exactly equals the audited loopback URL. |

## Exact backend test split

The 25 existing `test_stage*.py` methods divide as follows. The CI selector lists each runnable class or method explicitly and fails if its count changes unexpectedly.

| Existing file | Total | Runs in hosted CI | Local asset/GPU only | Reason |
| --- | ---: | ---: | ---: | --- |
| `test_stage1_config.py` | 4 | 3 | 1 | The hash-mismatch test requires the real five-asset set; the other tests check missing paths, policy rejection and startup preflight without opening a model. |
| `test_stage1_golden.py` | 2 | 0 | 2 | Imports the model-loaded API and uses three external fixed images plus the GPU baseline. |
| `test_stage2_checkpoints.py` | 3 | 0 | 3 | Validates original VAE/RealNVP weights and latent tensors. |
| `test_stage2_modules.py` | 3 | 3 | 0 | Synthetic tensors and images exercise detector clamp, grayscale crop and seeded TTA. |
| `test_stage2_orchestration.py` | 4 | 0 | 4 | The module imports `main`/`routes`, which load the frozen model set at import time, even where an individual test mocks request handling. |
| `test_stage2_policy.py` | 4 | 4 | 0 | Exercises live `saad-live-api-v1` boundary behavior with committed calibration. |
| `test_stage3_api.py` | 2 | 0 | 2 | Imports model-loaded routes and submits a fixed external sonar image. |
| `test_stage5_safety.py` | 3 | 3 | 0 | Exercises dataset builder opt-in and original-output guards without creating a dataset. |
| **Total** | **25** | **13** | **12** | |

The hosted backend job uses `SAAD_ARTIFACT_DIR` pointing at the checkout only to satisfy the live policy module's path declaration. It never calls `load_models` or asset hash validation against a real checkpoint. This is an asset-free CPU unit/configuration slice, not proof that the application can start without assets. `test_stage1_config.py::test_hash_mismatch_is_rejected` is excluded explicitly; its validation remains in the local suite.

## Local numerical regression

The already audited Windows RTX 4070 suite uses the official CUDA 12.8 wheel, authorized five external tensors, live calibration and the three hashed fixed sonar images. Configure `SAAD_ARTIFACT_DIR`, `YOLO_AUTOINSTALL=False` and a clean-repository `YOLO_CONFIG_DIR`, then run:

```powershell
.\var\publication-venv\Scripts\python.exe -B research/verify_fixtures.py
.\var\publication-venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_stage*.py' -q
```

The full local suite previously passed 25/25 with zero maximum reported golden deltas on the audited machine. GitHub CI does not rerun, store or attest to that GPU result. Keep all external assets outside Git and Actions. The hosted CPU wheel is chosen for test imports; no CPU numerical parity has been established.
