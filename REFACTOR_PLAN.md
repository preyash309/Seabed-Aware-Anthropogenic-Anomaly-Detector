# SAAD production recovery plan — audit and baseline complete

Audit date: 2026-09-23. Status as of 2026-09-24: **Stages 1–6 implemented and locally verified, with fresh wheel installation and external deployment unverified**. Stage reports record the executed commands, results and limits. No original data or checkpoint has been changed. The originals remain read-only.

The proposed layout and future-tense steps below are retained as the original audit hypothesis. [ARCHITECTURE.md](ARCHITECTURE.md), [STAGE6_VERIFICATION.md](STAGE6_VERIFICATION.md) and [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) describe the implemented system and its remaining gates.

## What changed after examining the original project

The prior audit found no checkpoints or datasets because it inspected only the partial GitHub checkout. The complete local project at `E:\SIH` has the exact five runtime tensors, prepared datasets, calibration tables, model results, integrity reports and offline research scripts. The working frontend is at `E:\SIH Web\saad`. [SOURCE_INVENTORY.md](SOURCE_INVENTORY.md), [ASSET_MANIFEST.md](ASSET_MANIFEST.md), and [MIGRATION_MAP.md](MIGRATION_MAP.md) contain file-level evidence.

The live backend source in the clean checkout is byte-identical to the original backend inference source. The original Python environment loads its YOLO26s, ConvVAE and RealNVP checkpoints. Direct calls of the original `/api/analyze` coroutine returned seven candidates for a fixed GhostVision test image twice with zero measured score difference. Original stage functions also ran on fixed AI4Shipwrecks and SubPipeMini2 test images. See [BASELINE_VERIFICATION.md](BASELINE_VERIFICATION.md) and its two small JSON fixtures in `audit/`.

The live API's evidence policy is **not the offline v3 engine**. The normal-validation P5/P95 values and 0.50/0.20/0.30 primary weights match the v3 result artifacts (59 validation normals), but offline v3 uses different priority and uncertainty equations and does not use TTA. The final offline evaluation's `YOLO_FLOW_MEAN` primary policy is a third distinct research choice. The approved migration must preserve the live API equations and profile/action labels exactly before any policy change is considered.

## Frozen behavior to preserve

1. Load `yolo26s_generic_baseline/weights/best.pt`, `vae_normal_seabed/checkpoints/best.pt`, and `latent_normalizing_flow/{best_flow,latent_mean,latent_std}.pt` using the recorded SHA-256 values. Do not swap in `last.pt`, VAE v2 SSIM, MAR, or generic YOLO weights.
2. Decode the uploaded image with Pillow; run Ultralytics YOLO on the saved source path at `imgsz=640`, `conf=0.05`, current device. Clamp boxes to image bounds.
3. For each box, use the original grayscale/context/square crop and 256×256 bilinear resize, scale to float32 [0,1], obtain deterministic VAE `mu`, compute reconstruction MSE, standardize `mu` with saved mean/std and compute RealNVP NLL.
4. Run brightness, contrast and seeded-noise YOLO TTA and match same-class boxes using the original IoU/confidence algorithm.
5. Normalize YOLO/VAE/flow with the six frozen percentile constants, calculate the live evidence profile, priority, uncertainty, priority level and recommendation; sort by priority and renumber candidate IDs. Scores represent review evidence, not calibrated anthropogenic probabilities.
6. Preserve response fields and current ordering during numerical extraction. Fix data loss and type mismatches in the web adapter only after a baseline comparison.

The current API imports models at module import, writes uploads, returns absolute localhost image URLs, swallows per-candidate anomaly/TTA errors, and has no persistence. These are production risks to address after parity is proven. The frontend performs a real upload and displays results, but review changes live only in React state; reports come from the latest localStorage item and browser export/print. There is no server review, retrieval or report endpoint.

## Proposed repository layout

```text
backend/                 FastAPI routes, typed schemas, health/readiness, persistence
saad_inference/          exact checkpoint loaders, crop, YOLO, VAE, flow, TTA, evidence policy
web/                     React/Vite app and typed API client
research/original/       preserved original offline scripts, organized by original relative path
research/runs/           small run manifests, commands and result hashes only
config/model_manifest.*  checkpoint paths, hashes, model architecture and calibration identity
tests/                   artifact compatibility, fixed-image numerical and API contract tests
docs/                    operator, research and validation instructions
var/                     runtime uploads/database/reports, ignored by Git
```

Model weights, prepared/raw datasets and generated experiment outputs remain external assets. Research scripts stay executable with documented source dataset versions, split manifest, command, Python environment, seed, checkpoint and input/output hashes. Existing experiment outputs are indexed in [ASSET_MANIFEST.md](ASSET_MANIFEST.md); no retraining is part of the migration. Keep the original local paths recorded as provenance even after runtime paths become configurable.

## Configuration, manifest and safe repository boundaries

A future model manifest should assign a stable identifier to each of the five live tensors, record original path, required SHA-256, size, architecture/input contract, trained split/provenance when known, and the matching calibration artifact hash. It should distinguish the **live API policy** from offline v3 and final evaluation policies. Startup should validate required hashes and tensor shapes before declaring readiness.

A future `.env.example` should document `SAAD_ARTIFACT_DIR`, `SAAD_YOLO_WEIGHTS`, `SAAD_VAE_WEIGHTS`, `SAAD_FLOW_WEIGHTS`, `SAAD_LATENT_MEAN`, `SAAD_LATENT_STD`, `SAAD_UPLOAD_DIR`, `SAAD_DATABASE_URL`, `SAAD_CORS_ORIGINS`, `SAAD_API_PUBLIC_URL`, `VITE_API_BASE_URL`, and optional device choice. Defaults should permit exact baseline behavior in a local parity run while production deployment uses external mounted assets and storage. Do not read or copy the original web `.env.local` contents into Git.

A future root `.gitignore` must exclude `.env*` while allowing `.env.example`, model/checkpoint extensions, dataset mounts, raw images/labels, `var/`, uploads, databases, generated result directories, caches, virtual environments, `node_modules`, build output and temporary reports. Keep small, reviewed JSON regression fixtures and their source hashes; do not commit raw source images or generated results. Validate with `git status` before any commit/push. No Git push is authorized in this phase.

## Implementation sequence after approval

1. Create a pinned runtime environment from the observed Python 3.11 / Torch 2.11.0+cu128 / Ultralytics 8.4.140 stack and the existing web lockfile. Document GPU and CPU setup separately. Produce `.env.example`, model manifest and safe ignore rules.
2. Extract the live numerical code into `saad_inference/` in small moves, replacing absolute paths with configuration. Keep operations, defaults, preprocessing and exception semantics unchanged for the parity gate. Compare all candidate fields against both baseline fixtures. Only then improve error surfacing and lifecycle behavior with explicit tests.
3. Move FastAPI into thin routes with bounded uploads, image validation, safe stored paths, durable analysis/review records and readiness that reflects actual model state. Define versioned response schemas. Preserve old route behavior or provide a documented compatibility adapter.
4. Add one frontend API client and typed response schema, eliminate hardcoded localhost URLs, retain candidate evidence fields, support reload/direct links, and persist review actions through API calls. Provide server-backed report retrieval/export with audit metadata.
5. Preserve offline scripts under their original-relative research paths. Record environment locks, data/split manifests, commands, seeds and output hashes for each study. No training or experimental policy substitution occurs as part of production extraction.

## Verification gates

**Numerical parity:** same fixed three-domain image set; detector count, class and candidate ordering; pixel boxes and confidence; VAE MSE; RealNVP NLL; three TTA variants; normalized evidence, profile, recommendation, priority and uncertainty. Apply the same-environment tolerances in [BASELINE_VERIFICATION.md](BASELINE_VERIFICATION.md). Recheck on a second device before claiming portability. Reject missing VAE/flow/TTA values rather than treating a superficially successful API response as parity.

**API and UI:** run backend in a clean environment with externally mounted original assets; check health/readiness and model hash reporting; upload each fixed image through actual HTTP multipart and the frontend; retrieve analysis after reload; accept, reject and correct a candidate; confirm durable review state, summary updates and audit record; generate JSON/CSV/PDF reports from persisted state and compare their candidate values to the analysis record. Verify local and deployed URL/CORS behavior and upload rejection paths.

**Data/research integrity:** prepared detector counts are 6,571/1,097/544 images for train/val/test; VAE patch counts are 11,475/1,403/1,408. Existing detector train labels outnumber train images by 45. The original audit lists 685 missing annotations in source data and a SubPipeMini2 temporal-neighbor split leakage report. Inspect these before rerunning training/evaluation or interpreting domain metrics. The production refactor does not repair datasets silently.

## Current boundary

Stages 1–6 were authorized and completed on `refactor/saad-production` for local loopback use. The fresh Torch CUDA wheel installation remains unverified after interrupted downloads; see `STAGE6_VERIFICATION.md` and `RELEASE_CHECKLIST.md`. Retraining, checkpoint modification, copying datasets or results into Git, pushing, merging and external deployment remain outside the authorized work. The deployed policy remains `saad-live-api-v1`; offline v3 remains research provenance only.
