# Research archive and verification tools

This directory preserves original research source identity and provides read-only verification entry points. It is not the backend inference package. See [Research guide](../docs/RESEARCH.md) for script-by-script status, dataset counts and policy distinctions.

## Contents and provenance

| Path | Purpose |
| --- | --- |
| [original/](original/) | 31 original scripts in Evidence, Experiment, VAE and YOLOv26 groups; source bytes preserved. |
| [source_manifest.json](source_manifest.json) | Source/archive-relative paths, sizes and SHA-256 identities for those 31 scripts. |
| [legacy/backend_test_realnvp.py](legacy/backend_test_realnvp.py) | Historical backend smoke script, retained byte-for-byte; superseded by current tests. |
| [archive_sources.ps1](archive_sources.ps1) | Archive verification/copy helper; refuses differing content at occupied paths. |
| [verify_provenance.py](verify_provenance.py) | Check archived hashes, optionally against read-only originals. |
| [verify_fixtures.py](verify_fixtures.py) | Frozen tensor/calibration/fixed-image preflight; optional focused golden run. |
| [verify_asset_ledger.py](verify_asset_ledger.py) | Stream-hash external ledger files; no recursive dataset-image reading. |
| [verify_original_inventory.py](verify_original_inventory.py) | Compare retained original source-size records, not full byte identities. |

Original scripts retain paths and potentially destructive training/output behavior. Do not run them as installation checks or import them into production. Offline v3 remains in `original/Evidence/build_saad_evidence_engine_v3.py`; it is distinct from deployed `saad-live-api-v1`.

## Supported verification commands

From the repository root after [Setup](../docs/SETUP.md):

```powershell
.\.venv\Scripts\python.exe -B research/verify_provenance.py
.\.venv\Scripts\python.exe -B research/verify_fixtures.py
.\.venv\Scripts\python.exe -B research/verify_fixtures.py --run
```

The first command checks archive identity. Fixture preflight requires the authorized five tensors, committed calibration and three fixed external images; `--run` then invokes the two golden tests. Configure `SAAD_GOLDEN_IMAGE_DIR` for an authorized image mirror if original paths are unavailable. Stop the API before GPU regression.

On the audited machine with the original trees available read-only:

```powershell
.\.venv\Scripts\python.exe -B research/verify_provenance.py --original-root E:/SIH
.\.venv\Scripts\python.exe -B research/verify_asset_ledger.py
.\.venv\Scripts\python.exe -B research/verify_original_inventory.py
```

These E: locations are historical input provenance, not portable installation requirements. Ledger verification needs the indexed external assets/results; original inventory verification needs both original source trees. The ledger contains 202 rows / 197 unique files; the original size inventory covers 37 ML/backend and 40 web source/config files. Size matching cannot establish full byte identity.

## Reproducibility boundary

Current reproducible evaluation is the frozen fixture suite, not a historical training run. Archived result files, dataset manifests and research calibration tables remain external and are indexed in [ASSET_MANIFEST.md](../ASSET_MANIFEST.md). They are not generated afresh by these verification tools.

The separate [ml selection](../ml/README.md) retains 22 historical scripts and three guarded dataset builders. A dry run is supported; complete rebuilding and training remain unverified. No images, checkpoints, generated outputs or new screenshots are included. Dataset/model/source licensing still requires original rights-holder review.
