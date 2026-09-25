# Stage 1 verification — portable paths and frozen assets

Date: 2026-09-23. Scope: Stage 1 only. Original directories remained read-only. No checkpoint, data, preprocessing algorithm, evidence formula, frontend flow or API candidate field was changed.

## Changed files

| File | Stage 1 change |
| --- | --- |
| `backend/config.py` | Central environment/path configuration, manifest and calibration validation, hash/size checks, device selection and runtime directory setup. |
| `config/model_manifest.json` | Five original tensor paths, exact sizes and SHA-256 hashes; pinned live calibration identity; distinct research policy ID. |
| `config/live_api_v1.json` | The exact deployed percentile and primary-weight values, with versioned live policy identity. |
| `backend/main.py` | Reads the validated paths, device, URL, CORS, calibration and policy ID; retains original inference and response assembly. |
| `backend/vae_inference.py`, `backend/realnvp_inference.py` | Load the same original architectures and checkpoint bytes from configured paths/devices. |
| `backend/requirements.txt`, `backend/requirements-original-windows-cu128.lock.txt` | Added dotenv support and recorded exact original Python environment versions. |
| `.env.example`, `.gitignore`, `README.md` | Operator configuration, safe asset exclusion and setup/policy instructions. |
| `tests/test_stage1_config.py`, `tests/test_stage1_golden.py` | Fail-closed configuration tests and API/three-domain numerical regression checks. |

## Configuration behavior

- The application reads process environment first, then local `.env` values for unset keys. `SAAD_ARTIFACT_DIR` or all five explicit asset paths are required; no checkpoint path silently defaults to an original machine directory.
- Each model/normalization tensor is checked against the model manifest's byte count and SHA-256 **before** model loading. The calibration file is checked against its pinned hash and `saad-live-api-v1` ID. Missing or mismatched assets stop startup with a configuration error.
- Relative application paths resolve from the clean repository. Upload and output directories default to ignored `var/` locations. The optional dataset root is validated as a directory and is not needed for inference. GPU selection remains `cuda:0` on the audited machine under `SAAD_DEVICE=auto`.
- The default runtime policy is `saad-live-api-v1`. `saad-offline-evidence-v3` is research only and fails the runtime policy check if selected as the deployed policy. The policies share normalization values and primary weights but have different priority, uncertainty, TTA, action and profile behavior. No offline v3 algorithm was substituted.

## Checks performed

| Check | Result |
| --- | --- |
| SHA-256 and size validation of all five frozen runtime tensors | **PASS** against `E:\SIH\SIH_Results` read-only assets. |
| Configuration tests: missing root, missing asset, altered hash, offline policy selection | **PASS**, 4 tests. |
| Original API coroutine on fixed GhostVision image | **PASS**, 7 candidates; original sorting, class, pixel boxes, raw/normalized scores, profiles, recommendation, priority and uncertainty within the audit tolerances. |
| Original stage functions on AI4Shipwrecks, GhostVision and SubPipeMini2 fixtures | **PASS**, 14/7/6 candidates; all available candidate-level detector, VAE, flow, TTA and evidence fields within audit tolerances. |
| Combined Stage 1 suite on Python 3.11.0 / Torch 2.11.0+cu128 / Ultralytics 8.4.140 / RTX 4070 Laptop GPU | **PASS**, 6 tests, no failures or skips. |
| Original source/checkpoint/data write | **NONE**. Test upload copy was confined to the clean repository and removed after SHA-256 verification. |

Run command used (with external asset and clean upload/output environment variables set):

    E:\SIH\new\Scripts\python.exe -B -m unittest discover -s D:\SAAD\SAAD\tests -p test_stage1_*.py -v

The tests use `audit/baseline_capture.json` and `audit/fixed_image_stage_capture.json` with the three read-only source images. The limits in [BASELINE_VERIFICATION.md](BASELINE_VERIFICATION.md) are same-environment engineering margins around the original's observed zero repeat difference. The suite reports parity within those limits; it does not claim bit-for-bit equality across devices.

## Unresolved and deliberately deferred

- A fresh installation from `requirements.txt` or the locked original environment snapshot is **UNVERIFIED**; exact CUDA wheel acquisition may need a machine-specific index. CPU and second-GPU numerical behavior are **UNVERIFIED**.
- An actual HTTP listener, browser upload, direct navigation, durable review, corrected boxes and reports are **UNVERIFIED** or not implemented. These belong to Stages 3–6.
- The live API still loads models at import, swallows per-candidate anomaly/TTA failures and has the original upload and error lifecycle. Stage 1 preserves that behavior for numerical parity; failure-mode restructuring belongs to Stage 2 and API work belongs to Stage 3.
- The offline Evidence Engine v3 remains a separate research policy. Its metrics are not claims about the live API.

## Proposed Stage 2, pending approval

Verify checkpoint state dictionaries and output equivalence, then extract canonical YOLO, crop/VAE, flow, TTA and live-evidence modules in small changes. Run the full golden suite after each substantive extraction. Add focused failure-mode tests without changing the deployed evidence policy or numerical output.
