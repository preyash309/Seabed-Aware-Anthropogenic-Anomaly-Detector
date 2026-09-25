# Stage 2 verification — modular frozen backend

Date: 2026-09-23. Branch: `refactor/saad-production`. Scope: Stage 2 only. The original ML and web source, datasets, and model/checkpoint files were not edited. All repository changes are in `D:/SAAD/SAAD`.

## Verified extraction and commits

| Commit | Change | Gate before commit |
| --- | --- | --- |
| `3fb18b5` | Exact crop/preprocessing and YOLO prediction/decoding modules; old crop exports retained | 6/6 Stage 1 tests passed |
| `f943797` | Canonical ConvVAE and RealNVP modules; old paths became import shims | 9/9 tests passed, including checkpoint key/shape checks |
| `f89a961` | Canonical TTA and `saad-live-api-v1` evidence module | 16/16 tests passed, including missing/non-finite/tie and priority boundaries |
| `ba24301` | One startup model registry, inference service, response serializer, FastAPI routes and `main.py` compatibility exports | 20/20 tests passed, including original upload errors and route/registry boundaries |

`backend/main.py:app` remains the entry point. `POST /api/analyze`, `GET /api/health`, `GET /api/model`, `GET /` and the `/uploads` mount remain registered. The response serializer retains the original candidate priority sort, ID renumbering, summary and output keys. Model initialization and manifest/SHA-256 validation occur once per process before request handling. The route passes the loaded registry to the service; the service never reloads models.

The saved ConvVAE checkpoint has 71 state entries and RealNVP has 56; both match the canonical runtime classes by exact key and shape, with zero missing, unexpected or shape-mismatched entries. Latent mean/std each contain 128 elements. No checkpoint was rewritten.

## Golden regression

Run in PowerShell with the read-only original artifact directory and clean-repository runtime outputs:

```powershell
$env:YOLO_AUTOINSTALL='False'
$env:SAAD_ARTIFACT_DIR='E:\SIH\SIH_Results'
$env:SAAD_UPLOAD_DIR='D:\SAAD\SAAD\var\uploads'
$env:SAAD_OUTPUT_DIR='D:\SAAD\SAAD\var\results'
& 'E:\SIH\new\Scripts\python.exe' -B -m unittest discover -s 'D:\SAAD\SAAD\tests' -p 'test_stage*.py' -v
```

Final result: **20 tests passed, 0 failures, 0 skips** on Python 3.11.0, Torch 2.11.0+cu128, Ultralytics 8.4.140 and NVIDIA GeForce RTX 4070 Laptop GPU (`cuda:0`). Startup validated the five external assets and pinned live calibration. The GhostVision API upload produced 7 candidates; the AI4Shipwrecks, GhostVision and SubPipeMini2 stage fixtures produced 14, 7 and 6 respectively. Fixture source SHA-256 values were checked before inference. No tolerance was relaxed.

The golden test now reports the maximum absolute difference for each compared numeric field. On the final run, every reported maximum was **0.0**: API YOLO confidence, VAE MSE, flow NLL, TTA, priority, uncertainty, pixel box coordinates and normalized evidence; stage detector boxes/confidence, VAE MSE, flow NLL, TTA, base evidence, priority and uncertainty. Candidate counts, class IDs, profiles, priority levels and recommendations also matched. This is equality against the captured fixture values on this machine, not a cross-device bitwise guarantee. The original audited tolerances remain in `tests/test_stage1_golden.py`.

New module tests cover crop shape/grayscale, detector clamping/class, seeded TTA and IoU boundary, live-policy missing/non-finite values, threshold ties and priority/uncertainty, checkpoint identity, route/registry sharing, response ordering and original invalid-upload HTTP errors. The invalid-image test mocks Pillow's rejection to exercise the same cleanup/error branch without probing optional decoders.

## Live policy and research policy

The deployed default is still `saad-live-api-v1` in `backend/saad_inference/evidence/live_api_v1.py`. It uses TTA in priority and uncertainty and retains the live action/profile thresholds. The offline Evidence Engine v3 remains research code in the original read-only project, with different bonus/penalty, uncertainty and action/profile rules. Both share normal-reference P5/P95 values and primary weights. The service never imports or selects offline v3. The manifest and calibration reject an offline policy ID for production startup.

## Incident and remaining limits

During an intermediate invalid-image test, Ultralytics automatically installed `pi-heif 1.4.0` into the original `E:/SIH/new` Python environment. This was unintended: the test command did not request package installation. No original source file, dataset or checkpoint was edited, but the original environment was changed contrary to the read-only workspace rule. The test was changed to mock the invalid-image decoder branch, and subsequent verification set `YOLO_AUTOINSTALL=False`. The package was not uninstalled because that would modify the original directory again. The final test run passed after this environment change; parity on the exact pre-incident environment is therefore unverified.

A fresh clean-environment install, CPU or another GPU, a live HTTP listener, browser upload/review, durable persistence and reports are **UNVERIFIED**. The backend retains the original per-candidate anomaly/TTA fallback and upload lifecycle to preserve behavior. Typed API schemas, durable scan/review state and HTTP integration belong to Stage 3; frontend and research cleanup remain later stages. No push, merge or Stage 3 work was performed.
