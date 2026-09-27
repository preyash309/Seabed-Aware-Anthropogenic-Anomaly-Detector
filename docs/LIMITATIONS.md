# Limitations, security and reuse

This guide describes the boundary of the verified release. For measured checks see [Evaluation](EVALUATION.md); for installation see [Setup](SETUP.md).

## Deployment and security

SAAD is a **local-loopback research application** verified on one Windows RTX 4070 Laptop GPU machine. It has no authentication or per-user authorization. Do not bind it to a public/shared network interface. Multi-user coordination, external deployment, production security controls and concurrent-review conflict handling are unverified.

Model status exposes local checkpoint paths; uploads and saved decisions are accessible to a caller of the local API. CORS origin restrictions are not access control. Keep local data private and back up SQLite together with saved image files. There is no automatic upload retention, scan deletion or authenticated backup workflow.

The original checkpoint loader uses Python-compatible deserialization (`weights_only=False`) to preserve compatibility. Supply only trusted authorized frozen assets; manifest hashes establish identity, not safety of an unknown replacement. Repository hygiene scans recognize selected signatures/file types and are not a comprehensive security assessment.

## Evidence and scientific claims

Five frozen tensors are external. The live calibration is **committed** in `config/live_api_v1.json` and validated by raw hash. No checkpoints or datasets are bundled or publicly downloadable through this repository.

Only `saad-live-api-v1` is deployed. Offline Evidence Engine v3 has different priority/uncertainty/action logic and no TTA; its results are not live API results. Scores prioritize human review and are not calibrated probabilities. Corrections/decisions are stored separately and do not alter immutable predictions.

Golden regression establishes migration parity and same-environment repeatability, not detection accuracy. No verified precision, recall, mAP, AUROC, latency or throughput benchmark is presented. Three fixed images are too few for generalization claims; candidate counts do not represent labeled object counts.

CPU numerical inference, another GPU/machine, non-Windows parity, retraining and historical offline experiment reruns are unverified. Fresh wheel installation passed on the audited machine only. Hosted CPU CI does not run model inference or attest to GPU parity.

## Data integrity and rights

The detector train split has 45 more labels than images; the source audit found 685 missing annotations. The SubPipeMini2 temporal-neighbor report found adjacent frames across splits. These findings constrain independent performance interpretation.

Dataset/model redistribution permissions and independent code ownership have not been established. The repository has no `LICENSE` file and grants no open-source reuse rights. Consult original source terms/rights holders before reuse; do not publish external assets merely because their hashes are indexed.

The 31-script archive preserves source provenance. Original research can contain absolute paths, training side effects and destructive output resets. Full guarded dataset builds and historical training/evaluation were not rerun.

## Application and reporting limits

The observed detector class is anthropogenic; no multiclass marine-debris taxonomy is verified. Inputs are raster images, not raw sonar streams or georeferenced survey formats. The browser picker and API suffix support differ; authoritative validation occurs in the backend.

A successful analysis can contain missing VAE/flow/TTA scores and `analysisStatus=partial`. Inference-time review fields are not authoritative human decisions. No reset-to-pending or delete route exists, and the event trail is application-maintained rather than cryptographically tamper-proof.

The Analytics route and decorative header controls are not verified features. Browser tests covered loopback upload/review/history/queue/downloads; hosted CI has no browser automation.

Reports differ: JSON includes review events, CSV contains candidate rows, PDF is text-only without image overlays or a complete event transcript. PDF signature/download was verified, but rendered text was not independently extracted. Report model/policy identifiers use current server settings at generation time; summary counts remain model counts rather than final accepted-object totals.

## Historical incidents and provenance strength

An intermediate environment used copied packages after a CUDA-wheel download failure; a later fresh wheel installation superseded that installation gap. During an earlier Stage 2 test, Ultralytics auto-installed `pi-heif` into the original environment. Final clean-repository validation did not repeat or undo that change. Exact pre-incident environment parity cannot be rechecked without modifying the original environment. Documented runs require `YOLO_AUTOINSTALL=False`.

One concurrent model-loaded numerical attempt exhausted CUDA/host memory; isolated reruns passed. A later full-suite process exited with native Windows code 0xC0000005 before reporting tests; allocation/startup and an unchanged isolated rerun passed. Neither failed process was a measured numerical mismatch. Calibration CRLF conversion once failed raw-hash preflight; exact LF restoration and the one-file Git attribute resolved it without changing values.

[SOURCE_INVENTORY.md](../SOURCE_INVENTORY.md) is a dated size snapshot; size equality alone cannot prove byte identity for non-archived original sources. [ASSET_MANIFEST.md](../ASSET_MANIFEST.md) records external hashes, and [research/source_manifest.json](../research/source_manifest.json) pins archived source bytes. These are the stated provenance strengths, not a claim that every original dataset image was recursively hashed.
