# Research sources and provenance

Use this guide for historical workflow status and dataset interpretation. [Research tools](../research/README.md) covers provenance/preflight commands; [Offline ML](../ml/README.md) covers builder settings and dry runs. Production model architecture is documented separately in [Architecture](ARCHITECTURE.md).

The production service imports only `backend/` modules. `research/original/` preserves **31 byte-identical Python scripts** from the original `E:/SIH/{Evidence,Experiment,VAE,YOLOv26}` folders. `research/source_manifest.json` records each source-relative path, archived path, byte size and SHA-256. `research/verify_provenance.py --original-root E:/SIH` checked all 31 against the read-only originals on the audited machine. Archived scripts can contain original absolute paths and may train models or overwrite outputs if run; they are provenance, not safe production entry points.

The earlier GitHub `ml/` selection contains 22 original scripts: 19 are byte-identical to the archive and three dataset builders have clean-repository path and refusal guards. `ml/Dataset/safety.py` is the new helper. `ml/requirements.txt` lists direct offline dependencies but is not a verified lock for historical reruns. The complete external result/hash ledger and dataset integrity findings are in [ASSET_MANIFEST.md](../ASSET_MANIFEST.md). No checkpoint, dataset image, generated result or external calibration table is shipped.

## Supported and historical workflows

| Area | Status | Boundary |
| --- | --- | --- |
| `research/verify_fixtures.py` and `--run` | Supported fixed-image evaluation | Validates frozen assets, live calibration and image hashes; `--run` invokes the two golden tests. Requires authorized external assets. |
| `ml/Dataset/{fix_ghostvision,prepare_saad_dataset,prepare_vae_dataset}.py` | Guarded dry run | `--dry-run` performs path planning without output. Full rebuilds were not executed. |
| `ml/Evaluation/*.py`, `ml/Experiments/*.py`, `ml/VAE/*.py`, `ml/YOLO/*.py` | Historical research selection | External inputs/results, path literals or training side effects; no unattended execution support is claimed. `evaluate_saad_by_dataset.py` contains a destructive temporary reset. |
| `research/original/**/*.py` | Hash-preserved archive | Never imported by production; do not run without inspecting paths and side effects. |
| `research/legacy/backend_test_realnvp.py` | Historical smoke script from the merged backend | Retained byte-for-byte, superseded by checkpoint and golden tests; may load external models when executed. |

The original offline Evidence Engine v3 is archived at `research/original/Evidence/build_saad_evidence_engine_v3.py`. Its original result configuration remains external. Although its normal-reference percentiles and primary weights match values used by the deployed `saad-live-api-v1`, v3 has different priority bonuses/penalties, uncertainty and action/profile rules and no TTA. The original final offline comparison also explored `YOLO_FLOW_MEAN`; neither offline policy was substituted into the live API. See [Architecture](ARCHITECTURE.md) and [Evaluation](EVALUATION.md) before comparing scores.

The three dataset builders require `SAAD_ALLOW_DATASET_BUILD=YES` for writes, a valid read-only input and a **new** output below `var/research_datasets` in the clean checkout. They refuse existing outputs and destinations in the original tree. Their complete rebuilds remain unverified. A dry run from the repository root is:

```powershell
.\.venv\Scripts\python.exe -B ml/Dataset/prepare_saad_dataset.py --dry-run
.\.venv\Scripts\python.exe -B ml/Dataset/prepare_vae_dataset.py --dry-run
.\.venv\Scripts\python.exe -B ml/Dataset/fix_ghostvision.py --dry-run
```

`research/archive_sources.ps1 -SourceRoot E:/SIH` verifies archived source hashes and refuses a different file at an occupied archive path. Do not use a training run or dataset build as a quick installation check. External dataset and model redistribution rights have not been established; users must supply authorized copies matching the recorded hashes.

## Selected research-script status

| Clean repository file | Status | Input/output note |
| --- | --- | --- |
| `ml/Dataset/fix_ghostvision.py` | Guarded, dry run supported; full build unverified | Original GhostVision metadata/images; new clean-repo output only |
| `ml/Dataset/prepare_saad_dataset.py` | Guarded, dry run supported; full build unverified | Four raw source domains; new clean-repo output only |
| `ml/Dataset/prepare_vae_dataset.py` | Guarded, dry run supported; full build unverified | Prepared SAAD manifest/images; new clean-repo output only |
| `ml/Dataset/safety.py` | Supported safety helper | Blocks original or existing outputs |
| `ml/Evaluation/build_saad_evidence_engine.py` | Historical | Offline policy generation, not deployed |
| `ml/Evaluation/build_saad_evidence_ranking.py` | Historical | Offline ranking study |
| `ml/Evaluation/evaluate_latent_flow.py` | Historical | External VAE/flow assets and VAE data |
| `ml/Evaluation/evaluate_saad_by_dataset.py` | Historical | Contains a destructive temporary evaluation reset; do not run unattended |
| `ml/Evaluation/evaluate_saad_fusion.py` | Historical | External datasets and output root |
| `ml/Evaluation/final_saad_evaluation.py` | Historical | Offline comparison, including YOLO_FLOW_MEAN; not live policy |
| `ml/Experiments/active_learning_saad.py` | Historical | Acquisition and external results |
| `ml/Experiments/calibrate_saad_fusion.py` | Historical | Offline calibration |
| `ml/Experiments/calibrate_saad_lodo.py` | Historical | Leave-one-domain-out calibration |
| `ml/Experiments/evaluate_mar.py` | Historical | MAR checkpoint and outputs |
| `ml/Experiments/evaluate_roi_fusion.py` | Historical | ROI fusion assets and outputs |
| `ml/Experiments/evaluate_shadow_plausibility.py` | Historical | External dataset and output root |
| `ml/Experiments/train_mar.py` | Historical training | No retraining in migration |
| `ml/VAE/train_latent_flow.py` | Historical training | No retraining in migration |
| `ml/VAE/train_vae.py` | Historical training | No retraining in migration |
| `ml/VAE/vae_anomaly.py` | Historical | VAE anomaly evaluation |
| `ml/VAE/vae_test_checkpoint.py` | Historical smoke check | Superseded by golden/checkpoint tests |
| `ml/YOLO/failure_analysis.py` | Historical | Detector error analysis |
| `ml/YOLO/train_yolo.py` | Historical training | No retraining in migration |

## Cleanup disposition

This table records the post-merge audit of files and folders that looked obsolete. Exact deleted versions remain in Git history; no merged history is rewritten.

| Path(s) | Classification | Decision |
| --- | --- | --- |
| `backend/`, `web/src/`, `config/`, `.github/workflows/`, `tests/`, `audit/*.json` | KEEP | Production, CI and numerical contract. No inference, policy, API or frontend behavior change. |
| `SOURCE_INVENTORY.md`, `ASSET_MANIFEST.md`, `BASELINE_VERIFICATION.md` | KEEP | Historical source/asset ledgers and original baseline; verification scripts and the model manifest reference their root paths. |
| `research/original/`, `research/source_manifest.json` | ARCHIVE | Preserve 31 original hashes and source bytes; no research deletion. |
| Research verification scripts and `ml/` | KEEP | Preserve supported checks, guarded dry runs and selected historical workflows. |
| `backend/{vae_inference,realnvp_inference,tta_inference}.py` | KEEP | Compatibility imports, including original offline source references. |
| `backend/test_realnvp.py` | ARCHIVE | Move byte-for-byte to `research/legacy/`; redundant model-loading smoke script. |
| `ARCHITECTURE.md`, `docs/DEPLOYMENT.md`, `research/STATUS.md`, `ml/README.md`, `web/README.md` | CONSOLIDATE | Current facts moved into permanent architecture, setup and research docs. |
| `STAGE1_VERIFICATION.md` through `STAGE6_VERIFICATION.md`, `REFACTOR_PLAN.md` | CONSOLIDATE, then DELETE | Progress, pre-migration hypotheses and interim failures superseded by implemented architecture, evaluation, setup, research and limits; original versions remain in merged Git history. |
| `audit/MIGRATION_MAP.md` | ARCHIVE | Move the original file-by-file migration classification out of the main documentation path; its historical proposals are not current runtime instructions. |
| `PUBLICATION_REVIEW.md`, `RELEASE_CHECKLIST.md`, `GITHUB_PUBLICATION_MANIFEST.md`, `docs/PR_BODY.md` | CONSOLIDATE, then DELETE | PR #1 preparation state is obsolete after merge; fresh-install result, local verification and publication limitations are retained in permanent docs. |
| `.venv/`, `var/`, `web/node_modules/`, `web/dist/`, local `.env` files | KEEP (ignored) | Ignored local environments, storage and builds; do not delete a user's runtime state or add it to Git. |
| `datasets/.gitignore`, `ml/*/.gitignore` | KEEP | Empty mount/output sentinels and Git exclusions. |

No source or dataset redistribution license is invented by this cleanup. The original external source, model, dataset and result directories are not edited.

## External datasets and prepared splits

The original inventory establishes source roots named AI4Shipwrecks, GhostVision_fixed, Marine_PULSE, SeabedObjects-Ship-and-Airplane-dataset-master, sss-crab-pot-detection-ds and SubPipeMini2. Presence is provenance evidence, not a licensing or evaluation endorsement. The prepared detector manifest uses AI4Shipwrecks, GhostVision, Marine-PULSE and SubPipeMini2 domain labels.

Prepared detector data uses images/labels for train, val and test plus `manifest.csv`; prepared VAE data has split images and its own manifest. These directories remain external. The committed `datasets/` is only an ignored-data boundary; the runtime does not require a dataset checkout.

| Prepared set | Split | Image files | Label files |
| --- | --- | ---: | ---: |
| SAAD_baseline | train | 6,571 | 6,616 |
| SAAD_baseline | val | 1,097 | 1,097 |
| SAAD_baseline | test | 544 | 544 |
| SAAD_VAE | train | 11,475 | Not applicable to this count |
| SAAD_VAE | val | 1,403 | Not applicable to this count |
| SAAD_VAE | test | 1,408 | Not applicable to this count |

These are audited file counts, not independently rebuilt splits. The detector manifest's Anthropogenic/Normal counts are separately recorded in [ASSET_MANIFEST.md](../ASSET_MANIFEST.md); manifest rows and image/label-file counts should not be conflated. Marine-PULSE contributes only 88 normal training rows in that record, with no validation/test rows.

The integrity audit records 685 missing annotations, 45 surplus training label files and neighboring SubPipeMini2 frames across train/validation/test. Full dataset-image bytes were not recursively read or hashed. The three fixed images have independent recorded hashes and label patterns (AI4 empty, GhostVision two class-0 boxes, SubPipeMini2 one), supporting regression coverage rather than accuracy claims.

Dataset builders preserve their historical preparation logic with output safeguards; they do not establish a clean new statistical evaluation protocol. Supply authorized data privately and inspect source terms before reuse. No official dataset URL, redistribution permission or unverified performance metric is inferred from the inventory.

## Documentation after consolidation

The cleanup disposition above records PR #2's historical decisions. The new directory READMEs under `ml/` and `web/` provide focused developer navigation; they do not restore the superseded progress documents or change research source bytes. The six permanent guides remain the detailed reference, with [the root README](../README.md) as the landing page.
