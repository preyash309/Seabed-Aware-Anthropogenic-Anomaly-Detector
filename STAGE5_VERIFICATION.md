# Stage 5 verification — research provenance and guarded builders

Date: 2026-09-24. Branch: `refactor/saad-production`. This stage did not alter production model algorithms, checkpoints, calibration, API responses or the live evidence policy. The original source, results, datasets and Python environment remained read-only.

## Source organization

All 31 Python scripts from the original `Evidence`, `Experiment`, `VAE` and `YOLOv26` source folders were copied into `research/original/` and checked byte-for-byte by SHA-256. `research/source_manifest.json` records original-relative path, archived path, size and SHA-256 for each. `research/verify_provenance.py --original-root E:/SIH` passed for all 31 after the copy. Of the prior 22-script `ml/` selection, 19 were byte-identical to archived originals. Three dataset builders were modified only in the clean repository to add path configuration and a refusal gate. The archive preserves their historical bytes. `research/STATUS.md` classifies every selected script and distinguishes reproducible, partially reproducible and historical workflows. Existing external result hashes remain in `ASSET_MANIFEST.md`.

The supported offline evaluation is `research/verify_fixtures.py`: without `--run`, it validates five frozen assets, the live calibration and three fixed domain image hashes without writing outputs. With `--run`, it executes the existing two-test golden inference suite. `SAAD_ARTIFACT_DIR` and optional `SAAD_GOLDEN_IMAGE_DIR` provide portable external locations. This workflow uses the deployed `saad-live-api-v1` policy; archived offline v3 code is never imported by production.

The three clean-repository dataset builders support `--dry-run`. Running them without `SAAD_ALLOW_DATASET_BUILD=YES` fails before an output is created. Even with opt-in, outputs must be new directories below `var/research_datasets` in the clean repository. Existing outputs and original source paths as destinations are refused; the builders no longer call recursive deletion. Actual dataset rebuilds and training were **not** run. Other historical scripts still contain machine-specific paths and are not marked as supported entry points.

## Tests and results

- `research/archive_sources.ps1 -SourceRoot E:/SIH`: 31 archived and post-copy SHA-256 matches.
- `python -B research/verify_provenance.py --original-root E:/SIH`: 31/31 archive and original hashes matched.
- `SAAD_ARTIFACT_DIR=E:/SIH/SIH_Results python -B research/verify_fixtures.py`: five asset hashes, live policy and three image hashes passed, no outputs written.
- The three dataset builders with `--dry-run`: passed; each printed its read-only source and ignored clean-repository destination and wrote nothing.
- The same builders without opt-in: each refused execution with the expected disabled error before creating output.
- `python -B -m unittest discover -s tests -p 'test_stage5_safety.py' -v`: **3 passed** (opt-in, original destination and existing output guards).
- `python -B research/verify_fixtures.py --run`, isolated after stopping local servers: **2 golden tests passed** with every reported maximum absolute difference **0.0**.
- Full `python -B -m unittest discover -s tests -p 'test_stage*.py' -q`: **25 passed, 0 failures, 0 skips**.
- `npm run typecheck`, `npm run lint`, `npm run build`: all passed after removal of unreferenced Vite/React template icons and update of the page title. The final Vite build transformed 1,858 modules.
- `git diff --check`: passed; generated `web/dist`, `web/node_modules`, `var`, `.venv` and `web/.env.local` are ignored.

The first `verify_fixtures.py --run` attempt was made while the local API and Vite servers were running. It encountered CUDA unknown errors and memory errors, causing a failed golden run and a PowerShell out-of-memory termination. No fixture or tolerance was changed. The local servers were stopped, GPU memory returned to zero reported use, and the isolated rerun passed with zero numerical difference. The full 25-test suite then passed serially. GPU regression runs should avoid a concurrent model-loaded API process on this machine.

## Limits

The exact Torch CUDA wheel fresh download remains **UNVERIFIED** after connection interruptions; the current isolated `.venv` uses package files copied read-only from the original environment. The historical studies have source and result provenance but their training/evaluation reruns are **UNVERIFIED**. Dataset integrity caveats include 45 surplus train labels, 685 missing source annotations and SubPipeMini2 temporal-neighbor split leakage. No archived script is a production dependency. The read-only original locations were not moved, deleted or overwritten.
