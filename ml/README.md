# Offline ML selection

This directory retains 22 selected historical research scripts plus the dataset safety helper. It is separate from [production inference](../backend/README.md). Nineteen scripts match the original archive byte-for-byte; three builders include clean-repository path/refusal guards. The complete original archive and hashes are under [research/](../research/README.md).

## Status by area

| Directory | Content | Current status |
| --- | --- | --- |
| [Dataset/](Dataset/) | GhostVision repair, detector/VAE preparation, safety helper | Path-planning dry runs verified; full builds unverified. |
| [YOLO/](YOLO/) | Detector training and failure analysis | Historical; not rerun during migration. |
| [VAE/](VAE/) | VAE/latent-flow training and anomaly/checkpoint checks | Historical; not the canonical runtime architecture. |
| [Evaluation/](Evaluation/) | Latent flow, fusion, domain evaluation and offline ranking | Historical; not deployed policy. |
| [Experiments/](Experiments/) | Active learning, LODO calibration, MAR, ROI/shadow studies | Historical; no current performance claim. |

[requirements.txt](requirements.txt) lists direct offline dependencies; it is not a locked, verified recipe for all historical experiments. Use the backend's audited environment for the supported dry-run checks. [Research](../docs/RESEARCH.md) lists every selected script and its boundary.

## Supported dataset planning

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -B ml/Dataset/prepare_saad_dataset.py --dry-run
.\.venv\Scripts\python.exe -B ml/Dataset/prepare_vae_dataset.py --dry-run
.\.venv\Scripts\python.exe -B ml/Dataset/fix_ghostvision.py --dry-run
```

Expected output states the read-only source and proposed output and confirms no images, labels or manifests were written. Dry runs inspect path planning; they do not establish dataset integrity or validate a complete build.

| Builder setting | Applies to |
| --- | --- |
| `SAAD_SOURCE_DATASET_ROOT` | Detector dataset source root. |
| `SAAD_PREPARED_DATASET_DIR` | VAE builder's prepared detector dataset root. |
| `SAAD_GHOSTVISION_SOURCE` | GhostVision repair source root. |
| `SAAD_RESEARCH_OUTPUT_DIR` | Proposed builder output. |
| `SAAD_ALLOW_DATASET_BUILD=YES` | Explicit opt-in required for writes, not needed for dry runs. |

Historical source defaults point to the audited external original trees. Outputs default to new children of `var/research_datasets`. The guards require opt-in, an existing source directory and a new output under that clean-repository boundary; they refuse occupied outputs. Do not set opt-in for a quick test. Full builds were not executed and no replacement dataset is claimed.

## Evaluation and training

The supported current evaluation command is [research/verify_fixtures.py](../research/verify_fixtures.py), optionally `--run`, with authorized frozen assets and images. Follow [Evaluation](../docs/EVALUATION.md); do not confuse it with offline performance studies.

Before attempting any historical experiment, inspect its absolute paths, inputs, output reset behavior and policy choice. In particular `Evaluation/evaluate_saad_by_dataset.py` resets a temporary directory destructively. The repository provides preserved source and result provenance, not an approved unattended training command. No weights were retrained or changed.

Dataset counts/annotation gaps and external result identities are in [ASSET_MANIFEST.md](../ASSET_MANIFEST.md). External data/checkpoint redistribution rights are not established, and no such assets are included here.
