# Migration map

This is the **pre-migration audit classification**. Stages 1–6 subsequently implemented the production and research layout; see [Architecture](../docs/ARCHITECTURE.md), [Evaluation](../docs/EVALUATION.md) and [Research](../docs/RESEARCH.md) for the implemented state. The original Stage 6 report remains in the merged Git history. Each original source/config/doc file below has one disposition. `ARCHIVE` means preserve the original research script and its input/output provenance in a versioned offline area; it does not mean delete the original. `EXTERNAL ASSET` means refer by validated path/hash, never commit binary data.

| Original file | Disposition | Proposed clean-repo location / reason |
| --- | --- | --- |
| `E:\SIH\Evidence/active_learning_retrain.py` | ARCHIVE | research/original/Evidence/active_learning_retrain.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\Evidence/active_learning_saad.py` | ARCHIVE | research/original/Evidence/active_learning_saad.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\Evidence/build_saad_evidence_engine_v2.py` | ARCHIVE | research/original/Evidence/build_saad_evidence_engine_v2.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\Evidence/build_saad_evidence_engine_v3.py` | ARCHIVE | research/original/Evidence/build_saad_evidence_engine_v3.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\Evidence/evaluate_mar.py` | ARCHIVE | research/original/Evidence/evaluate_mar.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\Evidence/evaluate_yolo_tta.py` | ARCHIVE | research/original/Evidence/evaluate_yolo_tta.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\Evidence/train_mar.py` | ARCHIVE | research/original/Evidence/train_mar.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\Experiment/build_saad_evidence_engine.py` | ARCHIVE | research/original/Experiment/build_saad_evidence_engine.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\Experiment/build_saad_evidence_ranking.py` | ARCHIVE | research/original/Experiment/build_saad_evidence_ranking.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\Experiment/calibrate_saad_cv.py` | ARCHIVE | research/original/Experiment/calibrate_saad_cv.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\Experiment/calibrate_saad_fusion.py` | ARCHIVE | research/original/Experiment/calibrate_saad_fusion.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\Experiment/calibrate_saad_lodo.py` | ARCHIVE | research/original/Experiment/calibrate_saad_lodo.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\Experiment/evaluate_roi_fusion.py` | ARCHIVE | research/original/Experiment/evaluate_roi_fusion.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\Experiment/evaluate_weighted_fusion.py` | ARCHIVE | research/original/Experiment/evaluate_weighted_fusion.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\Experiment/final_saad_evaluation.py` | ARCHIVE | research/original/Experiment/final_saad_evaluation.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\VAE/evaluate_latent_flow.py` | ARCHIVE | research/original/VAE/evaluate_latent_flow.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\VAE/evaluate_saad_fusion.py` | ARCHIVE | research/original/VAE/evaluate_saad_fusion.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\VAE/evaluate_shadow_plausibility.py` | ARCHIVE | research/original/VAE/evaluate_shadow_plausibility.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\VAE/prepare_vae_dataset.py` | ARCHIVE | research/original/VAE/prepare_vae_dataset.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\VAE/train_latent_flow.py` | ARCHIVE | research/original/VAE/train_latent_flow.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\VAE/train_vae.py` | ARCHIVE | research/original/VAE/train_vae.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\VAE/train_vae_v2_ssim.py` | ARCHIVE | research/original/VAE/train_vae_v2_ssim.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\VAE/vae_anomaly.py` | ARCHIVE | research/original/VAE/vae_anomaly.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\VAE/vae_test_checkpoint.py` | ARCHIVE | research/original/VAE/vae_test_checkpoint.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\YOLOv26/evaluate_saad_by_dataset.py` | ARCHIVE | research/original/YOLOv26/evaluate_saad_by_dataset.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\YOLOv26/failure_analysis.py` | ARCHIVE | research/original/YOLOv26/failure_analysis.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\YOLOv26/fix_ghostvision.py` | ARCHIVE | research/original/YOLOv26/fix_ghostvision.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\YOLOv26/integrity_pass_saad.py` | ARCHIVE | research/original/YOLOv26/integrity_pass_saad.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\YOLOv26/prepare_saad_dataset.py` | ARCHIVE | research/original/YOLOv26/prepare_saad_dataset.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\YOLOv26/test.py` | ARCHIVE | research/original/YOLOv26/test.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\YOLOv26/train_yolo.py` | ARCHIVE | research/original/YOLOv26/train_yolo.py; preserve command, inputs, outputs and hashes |
| `E:\SIH\backend/main.py` | REFACTOR | backend/main.py → API orchestration, config, evidence policy modules; numeric behavior frozen |
| `E:\SIH\backend/realnvp_inference.py` | REFACTOR | backend/realnvp_inference.py → production inference package with explicit asset paths |
| `E:\SIH\backend/requirements.txt` | REFACTOR | backend/requirements.txt plus locked runtime environment |
| `E:\SIH\backend/test_realnvp.py` | ARCHIVE | research/smoke/test_realnvp.py; replace with automated compatibility test after approval |
| `E:\SIH\backend/tta_inference.py` | REFACTOR | backend/tta_inference.py → production inference package with explicit asset paths |
| `E:\SIH\backend/vae_inference.py` | REFACTOR | backend/vae_inference.py → production inference package with explicit asset paths |
| `E:\SIH Web\saad\.env.local` | EXCLUDE | local secrets; document keys in .env.example |
| `E:\SIH Web\saad\.gitignore` | KEEP | web/.gitignore; retain current UI/design behavior |
| `E:\SIH Web\saad\.prettierignore` | KEEP | web/.prettierignore; retain current UI/design behavior |
| `E:\SIH Web\saad\.prettierrc` | KEEP | web/.prettierrc; retain current UI/design behavior |
| `E:\SIH Web\saad\README.md` | REFACTOR | web/README.md; config and setup consistency, preserve lockfile |
| `E:\SIH Web\saad\components.json` | KEEP | web/components.json; retain current UI/design behavior |
| `E:\SIH Web\saad\eslint.config.js` | KEEP | web/eslint.config.js; retain current UI/design behavior |
| `E:\SIH Web\saad\index.html` | KEEP | web/index.html; retain current UI/design behavior |
| `E:\SIH Web\saad\package-lock.json` | REFACTOR | web/package-lock.json; config and setup consistency, preserve lockfile |
| `E:\SIH Web\saad\package.json` | REFACTOR | web/package.json; config and setup consistency, preserve lockfile |
| `E:\SIH Web\saad\tsconfig.app.json` | KEEP | web/tsconfig.app.json; retain current UI/design behavior |
| `E:\SIH Web\saad\tsconfig.json` | KEEP | web/tsconfig.json; retain current UI/design behavior |
| `E:\SIH Web\saad\tsconfig.node.json` | KEEP | web/tsconfig.node.json; retain current UI/design behavior |
| `E:\SIH Web\saad\vite.config.ts` | REFACTOR | web/vite.config.ts; config and setup consistency, preserve lockfile |
| `E:\SIH Web\saad\src/App.tsx` | KEEP | web/src/App.tsx; retain current UI/design behavior |
| `E:\SIH Web\saad\src/index.css` | KEEP | web/src/index.css; retain current UI/design behavior |
| `E:\SIH Web\saad\src/main.tsx` | KEEP | web/src/main.tsx; retain current UI/design behavior |
| `E:\SIH Web\saad\src/components/theme-provider.tsx` | KEEP | web/src/components/theme-provider.tsx; retain current UI/design behavior |
| `E:\SIH Web\saad\src/components/evidence/CandidateList.tsx` | KEEP | web/src/components/evidence/CandidateList.tsx; retain current UI/design behavior |
| `E:\SIH Web\saad\src/components/evidence/EvidencePanel.tsx` | KEEP | web/src/components/evidence/EvidencePanel.tsx; retain current UI/design behavior |
| `E:\SIH Web\saad\src/components/layout/AppShell.tsx` | KEEP | web/src/components/layout/AppShell.tsx; retain current UI/design behavior |
| `E:\SIH Web\saad\src/components/layout/Sidebar.tsx` | KEEP | web/src/components/layout/Sidebar.tsx; retain current UI/design behavior |
| `E:\SIH Web\saad\src/components/layout/Topbar.tsx` | KEEP | web/src/components/layout/Topbar.tsx; retain current UI/design behavior |
| `E:\SIH Web\saad\src/components/sonar/ScanPipeline.tsx` | KEEP | web/src/components/sonar/ScanPipeline.tsx; retain current UI/design behavior |
| `E:\SIH Web\saad\src/components/sonar/SonarViewer.tsx` | KEEP | web/src/components/sonar/SonarViewer.tsx; retain current UI/design behavior |
| `E:\SIH Web\saad\src/components/sonar/UploadZone.tsx` | KEEP | web/src/components/sonar/UploadZone.tsx; retain current UI/design behavior |
| `E:\SIH Web\saad\src/components/ui/badge.tsx` | KEEP | web/src/components/ui/badge.tsx; retain current UI/design behavior |
| `E:\SIH Web\saad\src/components/ui/button.tsx` | KEEP | web/src/components/ui/button.tsx; retain current UI/design behavior |
| `E:\SIH Web\saad\src/components/ui/card.tsx` | KEEP | web/src/components/ui/card.tsx; retain current UI/design behavior |
| `E:\SIH Web\saad\src/components/ui/dialog.tsx` | KEEP | web/src/components/ui/dialog.tsx; retain current UI/design behavior |
| `E:\SIH Web\saad\src/components/ui/progress.tsx` | KEEP | web/src/components/ui/progress.tsx; retain current UI/design behavior |
| `E:\SIH Web\saad\src/components/ui/scroll-area.tsx` | KEEP | web/src/components/ui/scroll-area.tsx; retain current UI/design behavior |
| `E:\SIH Web\saad\src/components/ui/separator.tsx` | KEEP | web/src/components/ui/separator.tsx; retain current UI/design behavior |
| `E:\SIH Web\saad\src/components/ui/tabs.tsx` | KEEP | web/src/components/ui/tabs.tsx; retain current UI/design behavior |
| `E:\SIH Web\saad\src/components/ui/tooltip.tsx` | KEEP | web/src/components/ui/tooltip.tsx; retain current UI/design behavior |
| `E:\SIH Web\saad\src/lib/utils.ts` | KEEP | web/src/lib/utils.ts; retain current UI/design behavior |
| `E:\SIH Web\saad\src/pages/Analysis.tsx` | REFACTOR | web/src/pages/Analysis.tsx; typed API, durable review/retrieval/report integration |
| `E:\SIH Web\saad\src/pages/NewScan.tsx` | REFACTOR | web/src/pages/NewScan.tsx; typed API, durable review/retrieval/report integration |
| `E:\SIH Web\saad\src/pages/Reports.tsx` | REFACTOR | web/src/pages/Reports.tsx; typed API, durable review/retrieval/report integration |
| `E:\SIH Web\saad\src/types/saad.ts` | REFACTOR | web/src/types/saad.ts; typed API, durable review/retrieval/report integration |

## Existing clean-repo research selection and root documents

| Existing item | Disposition | Reason |
| --- | --- | --- |
| `README.md` | REFACTOR | update documentation/dependencies after baseline gate |
| `REFACTOR_PLAN.md` | REFACTOR | update documentation/dependencies after baseline gate |
| `ml/README.md` | REFACTOR | retain existing research selection and reconcile with original counterparts |
| `ml/requirements.txt` | REFACTOR | retain existing research selection and reconcile with original counterparts |
| `ml/Dataset/.gitignore` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/Dataset/fix_ghostvision.py` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/Dataset/prepare_saad_dataset.py` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/Dataset/prepare_vae_dataset.py` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/Evaluation/.gitignore` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/Evaluation/build_saad_evidence_engine.py` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/Evaluation/build_saad_evidence_ranking.py` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/Evaluation/evaluate_latent_flow.py` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/Evaluation/evaluate_saad_by_dataset.py` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/Evaluation/evaluate_saad_fusion.py` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/Evaluation/final_saad_evaluation.py` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/Experiments/.gitignore` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/Experiments/active_learning_saad.py` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/Experiments/calibrate_saad_fusion.py` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/Experiments/calibrate_saad_lodo.py` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/Experiments/evaluate_mar.py` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/Experiments/evaluate_roi_fusion.py` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/Experiments/evaluate_shadow_plausibility.py` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/Experiments/train_mar.py` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/VAE/.gitignore` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/VAE/train_latent_flow.py` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/VAE/train_vae.py` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/VAE/vae_anomaly.py` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/VAE/vae_test_checkpoint.py` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/YOLO/.gitignore` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/YOLO/failure_analysis.py` | KEEP | retain existing research selection and reconcile with original counterparts |
| `ml/YOLO/train_yolo.py` | KEEP | retain existing research selection and reconcile with original counterparts |

## External assets and exclusions

Dataset-side documentation and configuration have individual dispositions:

| Original file | Disposition | Proposed handling |
| --- | --- | --- |
| `E:\SIH\Datasets\README AI4Shipwrecks.txt` | ARCHIVE | Preserve in research data-source documentation. |
| `E:\SIH\Datasets\SeabedObjects-Ship-and-Airplane-dataset-master\SeabedObjects-Ship-and-Airplane-dataset-master\README.md` | ARCHIVE | Preserve source-dataset documentation and license context. |
| `E:\SIH\Datasets\sss-crab-pot-detection-ds\README.md` | ARCHIVE | Preserve source-dataset documentation and license context. |
| `E:\SIH\Datasets\SubPipeMini2\SubPipeMiniSSS\config.yaml` | EXTERNAL ASSET | Hash and reference with the raw dataset provenance. |


| Asset group | Disposition | Handling |
| --- | --- | --- |
| Five live checkpoint/normalization tensors listed in [ASSET_MANIFEST.md](../ASSET_MANIFEST.md) | EXTERNAL ASSET | Configure paths and verify SHA-256 before model load. |
| Alternate `last.pt`, VAE v2 SSIM, MAR, YOLO base/nano checkpoints | EXTERNAL ASSET | Preserve research provenance; do not substitute into runtime. |
| Prepared/raw datasets, calibration tables, experiment outputs | EXTERNAL ASSET | Keep external; reference hashes, manifests and run metadata. |
| `node_modules`, `.git`, `E:\SIH\new`, caches, uploads and generated binaries | EXCLUDE | Never migrate or commit. |

## Original proposed order (historical)

1. Pin the five live assets and calibration file by hash in a model manifest; add configurable paths with original defaults only for parity testing.
2. Extract production modules without altering numerical operations; compare every fixed candidate against `audit/baseline_capture.json` and `audit/fixed_image_stage_capture.json`.
3. Add typed API/web integration, durable analysis and review records, and reports.
4. Preserve original research scripts and outputs as provenance manifests and reproducible offline commands.
