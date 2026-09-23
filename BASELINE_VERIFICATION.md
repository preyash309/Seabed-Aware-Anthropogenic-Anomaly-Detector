# Baseline verification

Audit date: 2026-09-23. Original code and artifacts under `E:\SIH` were used without edits. The original `main.analyze` coroutine was invoked with a real `UploadFile`; its upload directory global was redirected at runtime to the clean repository for this verification. Generated upload copies were removed after verifying their content hashes. No network listener or browser flow was exercised.

## Environment and fixed contract

- Python: 3.11.0, from `E:\SIH\new\Scripts\python.exe` with `-B` (no bytecode writes).
- Device: `cuda:0`, NVIDIA GeForce RTX 4070 Laptop GPU. Torch 2.11.0+cu128, Ultralytics 8.4.140, FastAPI 0.141.1, NumPy 2.4.6, Pillow 12.3.0.
- Live loaders: YOLO26s `yolo26s_generic_baseline/weights/best.pt`; ConvVAE `vae_normal_seabed/checkpoints/best.pt`; RealNVP `latent_normalizing_flow/best_flow.pt` plus `latent_mean.pt` and `latent_std.pt`. Exact paths, sizes and SHA-256 values are in [ASSET_MANIFEST.md](ASSET_MANIFEST.md).
- Detector: original `YOLO.predict(source=<saved image path>, imgsz=640, conf=0.05, device=cuda:0)`; model class map observed as `{0: anthropogenic}`.
- Candidate patch: grayscale Pillow conversion, clamped YOLO box with context/square crop, bilinear resize to 256×256, float32 / 255. The VAE uses deterministic encoder `mu` and mean squared reconstruction error; RealNVP scores standardized `mu` as negative log likelihood.
- TTA: brightness ×1.15, contrast ×0.85, seeded Gaussian noise (seed 42, sigma 5), each run through detector at `conf=0.05`; same-class IoU match threshold 0.30, 0.70 IoU + 0.30 confidence agreement, averaged across three variants.
- Live evidence: hardcoded normal validation P5/P95 and 0.50 YOLO + 0.20 VAE + 0.30 flow base evidence; priority = 0.80 base + 0.10 corroboration + 0.10 TTA; uncertainty = 0.50 disagreement + 0.30(1−TTA) + 0.20(1−agreement). These are review scales, not probabilities.

## Original API run and repeatability

Fixed source: `E:\SIH\Datasets\SAAD_baseline\images\test\GhostVision__Rec9_wcp_ss_port_00049_png_jpg.rf.4e0cac2a6f42385c79c4abbb0486a92d__7c484f4422.png` (SHA-256 `f1e6f50c975935c5bd75f641cf936d3ff769542b04f9481b431aa6d09204d99c`). Two sequential calls of the original coroutine loaded the same 640×640 file and produced **7 candidates** each. Candidate IDs are assigned after priority sorting, so the table below is in final API order. All rows have class ID 0. The complete two-run JSON, including pixel boxes and normalized evidence fields, is [audit/baseline_capture.json](audit/baseline_capture.json).

| API order | Box xyxy pixels | YOLO | VAE MSE | Flow NLL | TTA | Priority | Uncertainty | Profile / action |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 119.3,55.9,176.0,151.3 | 0.172228 | 0.00185853 | 115.6291 | 0.986896 | 0.607630 | 0.432233 | DETECTOR_DOMINANT / REVIEW |
| 2 | 115.3,55.2,183.7,164.2 | 0.236889 | 0.00213455 | 110.6434 | 0.981423 | 0.604673 | 0.436020 | DETECTOR_DOMINANT / REVIEW |
| 3 | 123.0,54.8,170.4,122.0 | 0.097666 | 0.00116978 | 123.3744 | 0.981607 | 0.428451 | 0.236685 | DETECTOR_DOMINANT / LOW_PRIORITY_REVIEW |
| 4 | 127.0,56.9,164.6,106.8 | 0.077364 | 0.00097159 | 122.7608 | 0.984108 | 0.344413 | 0.193091 | WEAK_EVIDENCE / LOW_PRIORITY_REVIEW |
| 5 | 130.9,60.8,159.6,97.1 | 0.069828 | 0.00075255 | 134.2990 | 0.988089 | 0.339088 | 0.191841 | WEAK_EVIDENCE / LOW_PRIORITY_REVIEW |
| 6 | 133.8,63.4,156.3,92.2 | 0.068361 | 0.00062211 | 136.2310 | 0.991596 | 0.337974 | 0.191485 | WEAK_EVIDENCE / LOW_PRIORITY_REVIEW |
| 7 | 138.3,64.2,155.8,91.2 | 0.058950 | 0.00059060 | 136.3012 | 0.991031 | 0.314771 | 0.182233 | WEAK_EVIDENCE / LOW_PRIORITY_REVIEW |

Observed maximum absolute difference between the two calls:

| Field | Maximum absolute difference |
| --- | ---: |
| yoloConfidence | 0.0 |
| vaeScore | 0.0 |
| flowScore | 0.0 |
| ttaConsistency | 0.0 |
| priority | 0.0 |
| uncertainty | 0.0 |

All seven candidate boxes, ordering, classes, profiles and recommendations also matched exactly. This establishes within-process repeatability on this GPU and environment only. Proposed same-environment regression gate: exact candidate count/order/class/profile/recommendation; box coordinate absolute difference ≤1 pixel, YOLO confidence ≤1e-5, VAE MSE ≤1e-6, flow NLL ≤1e-2, TTA ≤1e-3, normalized evidence/priority/uncertainty ≤1e-3. These limits are conservative engineering margins around an observed zero difference; they are **not** cross-device validation.

## Three-domain stage capture

The original detector, crop, VAE, flow, TTA and evidence functions were run on one fixed prepared test image from each domain. Full candidate-level outputs in detector order are in [audit/fixed_image_stage_capture.json](audit/fixed_image_stage_capture.json).

| Domain/image | SHA-256 | Dimensions | Candidates | First detector candidate: class, confidence, VAE, flow, TTA, priority, uncertainty, profile |
| --- | --- | --- | ---: | --- |
| AI4Shipwrecks: `AI4Shipwrecks__Artificial_Reef_01__8ffa726c8a.png` | `3d787b7ed78f5807860cfca874160665f727d5c4ed4fe865e49ee47a072a336f` | 1728×2476 | 14 | 0, 0.150140, 0.00914367, 163.4921, 0.970808, 0.673615, 0.274110, YOLO_FLOW |
| GhostVision: `GhostVision__Rec9_wcp_ss_port_00049_png_jpg.rf.4e0cac2a6f42385c79c4abbb0486a92d__7c484f4422.png` | `f1e6f50c975935c5bd75f641cf936d3ff769542b04f9481b431aa6d09204d99c` | 640×640 | 7 | 0, 0.236889, 0.00213455, 110.6434, 0.981423, 0.604673, 0.436020, DETECTOR_DOMINANT |
| SubPipeMini2: `SubPipeMini2__1693570200.990__65ff02946b.png` | `13f06c56478d1db7064102682a6f4248d1250a596d5c3ed68113ccb07e11c1e5` | 2500×500 | 6 | 0, 0.384133, 0.00291569, 169.4366, 0.972116, 0.707068, 0.413703, YOLO_FLOW |

## Verified and unavailable

**Verified:** original loader successfully loaded all five runtime tensors; the observed YOLO class map is anthropogenic; original direct API produced complete candidate responses twice; all six measured score families repeated exactly for the fixed GhostVision image; three-domain stage functions produced 14/7/6 detector candidates; frozen percentile values in the code match the rounded validation-normal artifact values.

**Failed then resolved:** the first API attempt could not write its redirected upload under the command sandbox (`PermissionError` on `D:\SAAD\SAAD\audit`). The same original API call succeeded after filesystem escalation. No original file was written or changed.

**Not verified:** HTTP multipart transport/CORS, browser upload and rendered review/report interactions, server persistence (not implemented), CPU or another GPU, fresh environment install, independence of SubPipeMini2 neighboring frames, and the offline experiment metrics. Because the API's upload directory was redirected only after import, its returned localhost image URL was not a valid served URL in this direct-call test. The current code returns a successful analysis even when per-candidate VAE/flow or TTA scoring errors are swallowed, so regression tests must assert non-null values. The three-domain direct stage capture is not a full API response.

**Calibration distinction:** offline v3 `engine_config.json` shares the normal-reference P5/P95 and primary weights, but uses different priority/uncertainty rules and explicitly has `tta_used=false`. The deployed API includes TTA and different action/profile rules. The `final_saad_evaluation/FINAL_VERDICT.txt` selects `YOLO_FLOW_MEAN` as its research primary policy, also different from live API ranking. Production migration must preserve the live API policy until an explicit product/model decision changes it.

No inference source, checkpoint, or original result was rewritten.
