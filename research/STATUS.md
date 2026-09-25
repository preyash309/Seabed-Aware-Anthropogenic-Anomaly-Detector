# Research source and reproducibility status

`research/original/` contains 31 unmodified Python scripts copied read-only from `E:/SIH/{Evidence,Experiment,VAE,YOLOv26}`. Each archived byte sequence was checked against the original SHA-256 and recorded in `source_manifest.json` with relative original and archive paths and size. The archive is provenance only: importing or executing a script may use historical absolute paths, write outputs or start training. It is never imported by production inference. Offline Evidence Engine v3 is `original/Evidence/build_saad_evidence_engine_v3.py`; the deployed service imports only `backend/saad_inference/evidence/live_api_v1.py`. Shared normal percentiles do not make those policies equivalent.

The previous GitHub selection in `ml/` was compared by basename and SHA-256 against the archive. Nineteen files were byte-identical; the three guarded builders differ by their Stage 5 safety changes. The new `ml/Dataset/safety.py` has no historical counterpart. Classification of every selected source file:

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

**Reproducible now:** the frozen three-domain inference evaluation through `research/verify_fixtures.py --run` and the API golden tests. This uses configured external assets and retains the audited numerical tolerances. **Partially reproducible:** the guarded builders' source/output path resolution and refusal modes, verified by dry run; their full rebuilds are unverified and should be compared against the existing dataset integrity reports before use. **Historical:** the remaining original and `ml/` scripts have source hashes and external result metadata, but their exact execution environments, input snapshots and full outputs have not been independently rerun in this migration. Do not infer a supported workflow merely from file presence.

Before interpreting metrics, consult `ASSET_MANIFEST.md`: the prepared detector train split has 45 more labels than images, the source audit records 685 missing annotations, and the SubPipeMini2 temporal-neighbor split leakage report limits independence claims. The external result/hash ledger is in that manifest. No dataset image collection, checkpoint or result binary belongs in Git.

## Commands

Use the isolated Python 3.11 environment described in the root README. `backend/requirements-original-windows-cu128.lock.txt` is the observed Windows package snapshot; `ml/requirements.txt` lists direct offline packages but is not a verified fresh lock. The full Torch CUDA wheel download remained unavailable on this machine, so a fresh research environment installation is **UNVERIFIED**.

```powershell
$env:SAAD_ARTIFACT_DIR='E:\SIH\SIH_Results'
$env:YOLO_AUTOINSTALL='False'
$env:YOLO_CONFIG_DIR='D:\SAAD\SAAD\var\ultralytics'
& .\.venv\Scripts\python.exe -B research/verify_fixtures.py
& .\.venv\Scripts\python.exe -B research/verify_fixtures.py --run
& .\.venv\Scripts\python.exe -B ml/Dataset/prepare_saad_dataset.py --dry-run
& .\.venv\Scripts\python.exe -B ml/Dataset/prepare_vae_dataset.py --dry-run
& .\.venv\Scripts\python.exe -B ml/Dataset/fix_ghostvision.py --dry-run
```

The source/archive checksum operation is repeatable with `research/archive_sources.ps1 -SourceRoot E:/SIH`. It refuses a different file at any occupied archive path and compares SHA-256 after copying. Do not run any training, dataset build or historical analysis script as a validation shortcut.
