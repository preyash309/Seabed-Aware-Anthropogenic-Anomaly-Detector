# Source inventory

Audit date: 2026-09-23. Inventory covers Python, TypeScript/JavaScript, configuration, dependencies, and documentation. The inventory walks source trees while pruning `node_modules`, `.git`, Python environments/caches, datasets, results, generated uploads, and build output. Dataset manifests and results appear in [ASSET_MANIFEST.md](ASSET_MANIFEST.md). `.env.local` is listed by name and size only; its contents were not copied. The clean-repository table is a source snapshot taken during the audit; newly written audit deliverables and SVG template assets are outside its source-file count.

| Tree | Source/config/doc files |
| --- | ---: |
| original ML | 37 |
| original web | 40 |
| clean repository | 77 |

## Original ML and backend

| Path relative to `E:\SIH` | Bytes | Role |
| --- | ---: | --- |
| `Evidence/active_learning_retrain.py` | 23133 | offline research/data preparation |
| `Evidence/active_learning_saad.py` | 30816 | offline research/data preparation |
| `Evidence/build_saad_evidence_engine_v2.py` | 24424 | offline research/data preparation |
| `Evidence/build_saad_evidence_engine_v3.py` | 49585 | offline research/data preparation |
| `Evidence/evaluate_mar.py` | 29116 | offline research/data preparation |
| `Evidence/evaluate_yolo_tta.py` | 23347 | offline research/data preparation |
| `Evidence/train_mar.py` | 24957 | offline research/data preparation |
| `Experiment/build_saad_evidence_engine.py` | 40225 | offline research/data preparation |
| `Experiment/build_saad_evidence_ranking.py` | 31333 | offline research/data preparation |
| `Experiment/calibrate_saad_cv.py` | 31430 | offline research/data preparation |
| `Experiment/calibrate_saad_fusion.py` | 26712 | offline research/data preparation |
| `Experiment/calibrate_saad_lodo.py` | 31467 | offline research/data preparation |
| `Experiment/evaluate_roi_fusion.py` | 69519 | offline research/data preparation |
| `Experiment/evaluate_weighted_fusion.py` | 49979 | offline research/data preparation |
| `Experiment/final_saad_evaluation.py` | 29899 | offline research/data preparation |
| `VAE/evaluate_latent_flow.py` | 61606 | offline research/data preparation |
| `VAE/evaluate_saad_fusion.py` | 56148 | offline research/data preparation |
| `VAE/evaluate_shadow_plausibility.py` | 31945 | offline research/data preparation |
| `VAE/prepare_vae_dataset.py` | 13952 | offline research/data preparation |
| `VAE/train_latent_flow.py` | 41417 | offline research/data preparation |
| `VAE/train_vae.py` | 33895 | offline research/data preparation |
| `VAE/train_vae_v2_ssim.py` | 43074 | offline research/data preparation |
| `VAE/vae_anomaly.py` | 58228 | offline research/data preparation |
| `VAE/vae_test_checkpoint.py` | 3627 | offline research/data preparation |
| `YOLOv26/evaluate_saad_by_dataset.py` | 22949 | offline research/data preparation |
| `YOLOv26/failure_analysis.py` | 40926 | offline research/data preparation |
| `YOLOv26/fix_ghostvision.py` | 15176 | offline research/data preparation |
| `YOLOv26/integrity_pass_saad.py` | 18443 | offline research/data preparation |
| `YOLOv26/prepare_saad_dataset.py` | 48500 | offline research/data preparation |
| `YOLOv26/test.py` | 33893 | offline research/data preparation |
| `YOLOv26/train_yolo.py` | 12482 | offline research/data preparation |
| `backend/main.py` | 41328 | live API/runtime |
| `backend/realnvp_inference.py` | 11408 | live API/runtime |
| `backend/requirements.txt` | 72 | dependency list |
| `backend/test_realnvp.py` | 298 | manual smoke test |
| `backend/tta_inference.py` | 9570 | live API/runtime |
| `backend/vae_inference.py` | 12101 | live API/runtime |

## Original web

| Path relative to `E:\SIH Web\saad` | Bytes | Role |
| --- | ---: | --- |
| `.env.local` | 39 | local secret/config (exclude) |
| `.gitignore` | 277 | frontend source/config/docs |
| `.prettierignore` | 102 | frontend source/config/docs |
| `.prettierrc` | 264 | frontend source/config/docs |
| `README.md` | 477 | frontend source/config/docs |
| `components.json` | 515 | frontend source/config/docs |
| `eslint.config.js` | 613 | frontend source/config/docs |
| `index.html` | 370 | frontend source/config/docs |
| `package-lock.json` | 258914 | dependency lock/config |
| `package.json` | 1162 | dependency lock/config |
| `tsconfig.app.json` | 713 | frontend source/config/docs |
| `tsconfig.json` | 205 | frontend source/config/docs |
| `tsconfig.node.json` | 615 | frontend source/config/docs |
| `vite.config.ts` | 341 | frontend source/config/docs |
| `src/App.tsx` | 8240 | frontend source/config/docs |
| `src/index.css` | 410 | frontend source/config/docs |
| `src/main.tsx` | 351 | frontend source/config/docs |
| `src/components/theme-provider.tsx` | 5286 | frontend source/config/docs |
| `src/components/evidence/CandidateList.tsx` | 5953 | frontend source/config/docs |
| `src/components/evidence/EvidencePanel.tsx` | 6243 | frontend source/config/docs |
| `src/components/layout/AppShell.tsx` | 613 | frontend source/config/docs |
| `src/components/layout/Sidebar.tsx` | 4683 | frontend source/config/docs |
| `src/components/layout/Topbar.tsx` | 2122 | frontend source/config/docs |
| `src/components/sonar/ScanPipeline.tsx` | 2969 | frontend source/config/docs |
| `src/components/sonar/SonarViewer.tsx` | 19741 | frontend source/config/docs |
| `src/components/sonar/UploadZone.tsx` | 4880 | frontend source/config/docs |
| `src/components/ui/badge.tsx` | 1915 | frontend source/config/docs |
| `src/components/ui/button.tsx` | 3230 | frontend source/config/docs |
| `src/components/ui/card.tsx` | 2620 | frontend source/config/docs |
| `src/components/ui/dialog.tsx` | 4066 | frontend source/config/docs |
| `src/components/ui/progress.tsx` | 1716 | frontend source/config/docs |
| `src/components/ui/scroll-area.tsx` | 1600 | frontend source/config/docs |
| `src/components/ui/separator.tsx` | 521 | frontend source/config/docs |
| `src/components/ui/tabs.tsx` | 3487 | frontend source/config/docs |
| `src/components/ui/tooltip.tsx` | 2836 | frontend source/config/docs |
| `src/lib/utils.ts` | 24 | frontend source/config/docs |
| `src/pages/Analysis.tsx` | 18633 | frontend source/config/docs |
| `src/pages/NewScan.tsx` | 9374 | frontend source/config/docs |
| `src/pages/Reports.tsx` | 24078 | frontend source/config/docs |
| `src/types/saad.ts` | 952 | frontend source/config/docs |

## Clean repository

| Path relative to `D:\SAAD\SAAD` | Bytes | Role |
| --- | ---: | --- |
| `ASSET_MANIFEST.md` | 32897 | checked-out source/config/docs |
| `README.md` | 2 | checked-out source/config/docs |
| `REFACTOR_PLAN.md` | 29207 | existing audit hypothesis |
| `backend/main.py` | 41328 | checked-out source/config/docs |
| `backend/realnvp_inference.py` | 11408 | checked-out source/config/docs |
| `backend/requirements.txt` | 171 | checked-out source/config/docs |
| `backend/test_realnvp.py` | 298 | checked-out source/config/docs |
| `backend/tta_inference.py` | 9570 | checked-out source/config/docs |
| `backend/vae_inference.py` | 12101 | checked-out source/config/docs |
| `ml/README.md` | 2 | checked-out source/config/docs |
| `ml/requirements.txt` | 149 | checked-out source/config/docs |
| `ml/Dataset/.gitignore` | 2 | checked-out source/config/docs |
| `ml/Dataset/fix_ghostvision.py` | 15176 | checked-out source/config/docs |
| `ml/Dataset/prepare_saad_dataset.py` | 48500 | checked-out source/config/docs |
| `ml/Dataset/prepare_vae_dataset.py` | 13952 | checked-out source/config/docs |
| `ml/Evaluation/.gitignore` | 2 | checked-out source/config/docs |
| `ml/Evaluation/build_saad_evidence_engine.py` | 40225 | checked-out source/config/docs |
| `ml/Evaluation/build_saad_evidence_ranking.py` | 31333 | checked-out source/config/docs |
| `ml/Evaluation/evaluate_latent_flow.py` | 61606 | checked-out source/config/docs |
| `ml/Evaluation/evaluate_saad_by_dataset.py` | 22949 | checked-out source/config/docs |
| `ml/Evaluation/evaluate_saad_fusion.py` | 56148 | checked-out source/config/docs |
| `ml/Evaluation/final_saad_evaluation.py` | 29899 | checked-out source/config/docs |
| `ml/Experiments/.gitignore` | 2 | checked-out source/config/docs |
| `ml/Experiments/active_learning_saad.py` | 30816 | checked-out source/config/docs |
| `ml/Experiments/calibrate_saad_fusion.py` | 26712 | checked-out source/config/docs |
| `ml/Experiments/calibrate_saad_lodo.py` | 31467 | checked-out source/config/docs |
| `ml/Experiments/evaluate_mar.py` | 29116 | checked-out source/config/docs |
| `ml/Experiments/evaluate_roi_fusion.py` | 69519 | checked-out source/config/docs |
| `ml/Experiments/evaluate_shadow_plausibility.py` | 31945 | checked-out source/config/docs |
| `ml/Experiments/train_mar.py` | 24957 | checked-out source/config/docs |
| `ml/VAE/.gitignore` | 2 | checked-out source/config/docs |
| `ml/VAE/train_latent_flow.py` | 41417 | checked-out source/config/docs |
| `ml/VAE/train_vae.py` | 33895 | checked-out source/config/docs |
| `ml/VAE/vae_anomaly.py` | 58228 | checked-out source/config/docs |
| `ml/VAE/vae_test_checkpoint.py` | 3627 | checked-out source/config/docs |
| `ml/YOLO/.gitignore` | 2 | checked-out source/config/docs |
| `ml/YOLO/failure_analysis.py` | 40926 | checked-out source/config/docs |
| `ml/YOLO/train_yolo.py` | 12482 | checked-out source/config/docs |
| `web/.gitignore` | 277 | checked-out source/config/docs |
| `web/.prettierignore` | 102 | checked-out source/config/docs |
| `web/.prettierrc` | 264 | checked-out source/config/docs |
| `web/README.md` | 477 | checked-out source/config/docs |
| `web/components.json` | 540 | checked-out source/config/docs |
| `web/eslint.config.js` | 613 | checked-out source/config/docs |
| `web/index.html` | 370 | checked-out source/config/docs |
| `web/package-lock.json` | 266211 | checked-out source/config/docs |
| `web/package.json` | 1207 | checked-out source/config/docs |
| `web/tsconfig.app.json` | 713 | checked-out source/config/docs |
| `web/tsconfig.json` | 205 | checked-out source/config/docs |
| `web/tsconfig.node.json` | 615 | checked-out source/config/docs |
| `web/vite.config.ts` | 341 | checked-out source/config/docs |
| `web/src/App.tsx` | 8240 | checked-out source/config/docs |
| `web/src/index.css` | 441 | checked-out source/config/docs |
| `web/src/main.tsx` | 351 | checked-out source/config/docs |
| `web/src/components/theme-provider.tsx` | 5286 | checked-out source/config/docs |
| `web/src/components/evidence/CandidateList.tsx` | 5953 | checked-out source/config/docs |
| `web/src/components/evidence/EvidencePanel.tsx` | 6243 | checked-out source/config/docs |
| `web/src/components/layout/AppShell.tsx` | 613 | checked-out source/config/docs |
| `web/src/components/layout/Sidebar.tsx` | 4683 | checked-out source/config/docs |
| `web/src/components/layout/Topbar.tsx` | 2122 | checked-out source/config/docs |
| `web/src/components/sonar/ScanPipeline.tsx` | 2969 | checked-out source/config/docs |
| `web/src/components/sonar/SonarViewer.tsx` | 19741 | checked-out source/config/docs |
| `web/src/components/sonar/UploadZone.tsx` | 4880 | checked-out source/config/docs |
| `web/src/components/ui/badge.tsx` | 1966 | checked-out source/config/docs |
| `web/src/components/ui/button.tsx` | 3287 | checked-out source/config/docs |
| `web/src/components/ui/card.tsx` | 2722 | checked-out source/config/docs |
| `web/src/components/ui/dialog.tsx` | 4226 | checked-out source/config/docs |
| `web/src/components/ui/progress.tsx` | 1796 | checked-out source/config/docs |
| `web/src/components/ui/scroll-area.tsx` | 1652 | checked-out source/config/docs |
| `web/src/components/ui/separator.tsx` | 543 | checked-out source/config/docs |
| `web/src/components/ui/tabs.tsx` | 3568 | checked-out source/config/docs |
| `web/src/components/ui/tooltip.tsx` | 2901 | checked-out source/config/docs |
| `web/src/lib/utils.ts` | 25 | checked-out source/config/docs |
| `web/src/pages/Analysis.tsx` | 18633 | checked-out source/config/docs |
| `web/src/pages/NewScan.tsx` | 9374 | checked-out source/config/docs |
| `web/src/pages/Reports.tsx` | 24078 | checked-out source/config/docs |
| `web/src/types/saad.ts` | 952 | checked-out source/config/docs |

## Evidence of deployed entry points

Dataset-side source documentation and configuration (raw image and annotation trees were not read):

| Original path | Bytes | SHA-256 |
| --- | ---: | --- |
| `E:\SIH\Datasets\README AI4Shipwrecks.txt` | 3052 | `368eb368d2985c82f5b442777ec2e3c4f5144a7384ef328a7e11ecc92bbc81de` |
| `E:\SIH\Datasets\SeabedObjects-Ship-and-Airplane-dataset-master\SeabedObjects-Ship-and-Airplane-dataset-master\README.md` | 616 | `b39bbabf5948a8afeb525550676278b17d8c267a06cf68ad85a9e8c379d88010` |
| `E:\SIH\Datasets\sss-crab-pot-detection-ds\README.md` | 7465 | `2e372b61798ab953c852f0fcfd1a7b87d74216451c16cad9b8a2ff92b5e989e0` |
| `E:\SIH\Datasets\SubPipeMini2\SubPipeMiniSSS\config.yaml` | 1160 | `687fee4e3295e131f65397b52c6bccdea236eb056e07a6595a51c8135f6eea66` |

The prepared `SAAD_baseline/data.yaml` and result-side YAML/JSON/CSV files are listed with hashes in [ASSET_MANIFEST.md](ASSET_MANIFEST.md). Hugging Face cache metadata was excluded.

- `E:\SIH\backend\main.py` creates the FastAPI app, loads all three model assets at import time, and exposes `POST /api/analyze`, `GET /api/health`, `GET /api/model`, `GET /`, and static `/uploads`. It imports only `vae_inference.py`, `realnvp_inference.py`, and `tta_inference.py` for inference. `test_realnvp.py` is a manual smoke script.

- `E:\SIH Web\saad\src\pages\NewScan.tsx` sends multipart uploads to the backend. `Analysis.tsx` adapts the response and manages review in React state. `Reports.tsx` exports JSON/CSV and prints a browser PDF from local state/localStorage. The original web page files match the clean repository text.

- Scripts under `Evidence`, `Experiment`, `VAE`, and `YOLOv26` are offline training, calibration, preparation, diagnostics, or evaluation. They are not imported by the live API. The original contains offline files absent from the GitHub selection, including `train_vae_v2_ssim.py`, `integrity_pass_saad.py`, `evaluate_yolo_tta.py`, `active_learning_retrain.py`, `calibrate_saad_cv.py`, and `build_saad_evidence_engine_v2.py`/`v3.py`.

- Checked-out `backend/main.py` and its three inference modules are byte-identical to the original. The clean backend requirements list is expanded; the original bundled environment is the observed runnable one. The web differences flagged by raw byte comparison are line-ending normalization; `git diff --no-index` showed no content delta for inspected cases.

## API to React contract

| Runtime producer | Consumer | Observed contract issue |
| --- | --- | --- |
| `POST /api/analyze` | `NewScan.tsx` | Upload uses hardcoded localhost API URL. |
| Result in navigation state | `Analysis.tsx` | Direct link/reload loses analysis; adapter drops detailed class, pixel box and evidence fields. |
| `priorityLevel=MEDIUM` and nullable scores | `types/saad.ts` | Declared union omits MEDIUM and score fields are nonnullable. |
| Client review updates | `Analysis.tsx` | No API call or durable review endpoint. |
| Latest report in localStorage | `Reports.tsx` | Browser exports and print are not server persisted or auditable. |
| Absolute localhost image URL | `SonarViewer.tsx` | Fails for remote deployment origins. |

No package install or frontend build was run as part of source inventory.
