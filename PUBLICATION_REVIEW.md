# SAAD GitHub publication review

Review date: 2026-09-25. Repository: `D:/SAAD/SAAD`. Branch: `refactor/saad-production`. Base: `origin/main` at `e37089f`. This review prepares a pull request; no push, pull request, merge, history rewrite, release or network deployment was performed.

## Decision and scope

**Ready for a reviewable GitHub pull request after explicit authorization to push and open it.** The demonstrated application boundary is local loopback on the audited Windows RTX 4070 machine. The fresh Python installation and all golden comparisons now pass there. External assets, authentication and other hardware remain outside this claim.

I read `STAGE1_VERIFICATION.md` through `STAGE6_VERIFICATION.md`, `RELEASE_CHECKLIST.md`, `README.md`, `ARCHITECTURE.md`, `ASSET_MANIFEST.md`, `research/STATUS.md` and the deployment guide. `EXECUTION_STATUS.md` was requested for review but is **absent** from this checkout; this document records the current execution state instead. The initial working tree was clean, with 18 local migration commits ahead of `origin/refactor/saad-production`.

## Repository and history hygiene

- The pre-publication tree had 165 tracked source/config/doc files totaling 2,574,067 bytes. No tracked file exceeded 1 MiB; the largest was `web/package-lock.json` at 266,211 bytes. No model checkpoint, tensor, dataset image, database, generated output, runtime log, local `.env` or screenshot was tracked. The two `audit/*.json` files hold numeric fixtures, original image paths and SHA-256 values, not image bytes.
- At audit start, the full reachable local Git history was checked via `git rev-list --objects --all`: 401 unique objects, 224 unique blobs, 3,149,124 aggregate blob bytes. The largest historical blob was a 258,914-byte package lock. Blob content scanning found no private-key block, GitHub/AWS token signature or credential assignment. The only historical `.env.local` blob is 39 bytes and contains `VITE_API_BASE_URL=http://127.0.0.1:8000`; it was deleted in the pre-migration history. Example `.env.example` revisions are placeholders. No history rewrite is needed on the evidence found. A separate scan of all 168 proposed current files found zero credential or private-user-path alerts.
- Absolute `E:/SIH` paths remain in archived research scripts, audit fixture source fields, historical reports and operator examples for provenance. Production `backend/` and `web/src/` contain no original-machine asset path. No concrete private user-home path was found; a generic path-search example in `REFACTOR_PLAN.md` triggered that category in the scanner.
- All 31 `research/original/` entries are `.py` source files, not media or model assets. Their SHA-256 hashes matched the read-only originals. Searches found no embedded credential or third-party license notice in the archive. Source ownership cannot be independently proved by a file scan; the publisher should confirm it owns or is authorized to publish the original research code. Dataset and model redistribution rights were not inferred, so no such bytes or screenshots are included. There is no repository `LICENSE` file; this PR does not claim an open-source reuse license.
- `.gitignore` was checked with `git check-ignore` for dataset paths, model/tensor formats, `.env.local`, uploads, SQLite and sidecars, `var/`, `web/node_modules/` and `web/dist/`. It was extended for NumPy/HDF tensors, SQLite sidecars, dataset directory variants and columnar dataset files. The exact proposed paths and exclusions are in `GITHUB_PUBLICATION_MANIFEST.md`.

No concrete secret or improperly licensed asset was identified in the proposed publication. The historical loopback-only `.env.local` and provenance paths are documented above rather than hidden.

## Fresh isolated installation

The new environment is `D:/SAAD/SAAD/var/publication-venv` (ignored by Git), created with `py -3.11 -m venv`. It did not use or modify `E:/SIH/new`. `TEMP`/`TMP` and all installer artifacts were directed into ignored `D:/SAAD/SAAD/var/` paths. The exact `torch==2.11.0+cu128` Windows wheel was downloaded from the official PyTorch `cu128` index. The first pip attempt requested the 2,753,148,611-byte wheel a second time during resolution, so that attempt was interrupted **after** one complete official wheel transfer. The completed wheel was installed into the new environment with `pip install --no-deps <wheel-path>`; the documented lockfile install then succeeded and reported Torch already satisfied. `reportlab==5.0.1` installed separately as documented. The temporary wheel was deleted after verification.

Installed runtime: Python 3.11.0; Torch 2.11.0+cu128; torchvision 0.26.0+cu128; CUDA 12.8; Ultralytics 8.4.140; FastAPI 0.141.1; NumPy 2.4.6; Pillow 12.3.0; reportlab 5.0.1; NVIDIA GeForce RTX 4070 Laptop GPU. The executable path was inside `var/publication-venv`. `pip check` returned **No broken requirements found**. This is a fresh wheel installation on this machine, not a claim about a second machine, CPU or another GPU. The older `.venv` still has the Stage 3 package-copy history and was not used for this fresh-install gate.

## Validation results

| Gate | Result | Exact evidence |
| --- | --- | --- |
| Runtime assets and calibration | **PASS** | Fresh-environment `research/verify_fixtures.py` validated all five frozen tensors, `saad-live-api-v1` calibration and three fixed domain image hashes. |
| External result ledger and original source | **PASS with scope** | `research/verify_asset_ledger.py`: 202 rows, 197 unique files, 1,090,035,579 bytes rehashed. `research/verify_provenance.py`: 31 archive hashes matched. `research/verify_original_inventory.py`: 37 ML/backend and 40 web source/config sizes matched; non-archived original source had size-only audit baselines. |
| Fresh Python dependency consistency | **PASS** | `var/publication-venv/Scripts/python.exe -m pip check`: no broken requirements. No package file was copied from the original environment. |
| Fresh backend and golden suite | **PASS** | `var/publication-venv/Scripts/python.exe -B -m unittest discover -s tests -p 'test_stage*.py' -q`: 25 tests, 0 failures, 0 skips. GhostVision API: 7 candidates; AI4Shipwrecks/GhostVision/SubPipeMini2 stage cases: 14/7/6. Every printed maximum absolute difference for boxes, confidence, VAE MSE, RealNVP NLL, TTA, normalized evidence, priority and uncertainty was **0.0**. Fixtures and tolerances were unchanged. Log: ignored `var/results/publication-fresh-tests.log`. |
| Existing isolated environment cross-check | **PASS** | The same 25 tests and the two focused golden tests passed with all reported deltas 0.0 before the fresh installation finished. |
| Frontend dependency and quality gates | **PASS** | `npm ci`: 488 packages, zero reported vulnerabilities. `npm run typecheck`, `npm run lint` and `npm run build` passed; Vite transformed 1,858 modules. |
| Real HTTP and browser | **PASS on loopback** | Existing-env built-browser upload of the fixed GhostVision image returned scan `SAAD-34BE6C1A` with seven candidates. Accepting candidate 01 persisted after reload and appeared in the report. Direct HTTP read-back found one review event. The fresh-env Uvicorn process then accepted a real multipart GhostVision upload as `SAAD-81994680` with seven candidates; the production browser opened that scan directly and rendered all seven. Both servers were stopped afterward. |
| Original locations | **PASS with audit limit** | External asset hashes and archived-source hashes matched; original inventory sizes matched. No original file was edited, moved, deleted, trained or overwritten during publication preparation. Size-only baseline coverage cannot prove byte identity for non-archived original source files. |

One unprivileged PowerShell HTTP upload attempt was denied access to the read-only source image by the command sandbox; an authorized rerun using the same fixture and loopback endpoint returned HTTP 200. That was an execution-permission failure, not an application inference failure. The first fresh pip attempt's duplicate wheel transfer was stopped; the subsequent local-wheel plus pinned-dependency installation passed. No regression tolerance was relaxed and no inference code changed.

## Proposed pull request

**Title:** `Recover frozen SAAD inference and local review application`

**Body:** `docs/PR_BODY.md` contains the ready-to-paste summary, tests, external asset requirements and deployment limits. The 31-script archive makes the full diff large; review commit `51eb080` as byte-preserved provenance and the Stage 1–6 commits as separate implementation steps. `GITHUB_PUBLICATION_MANIFEST.md` lists every included path and the excluded assets/runtime classes.

The GitHub CLI (`gh`) is not installed on this machine. After explicit authorization, from `D:/SAAD/SAAD`:

```powershell
git push -u origin refactor/saad-production
```

Then open `https://github.com/preyash309/Marine---Debris-Detection/compare/main...refactor/saad-production?expand=1`, use the title above and paste `docs/PR_BODY.md`. If GitHub CLI is later installed and authenticated, the equivalent PR command is:

```powershell
gh pr create --base main --head refactor/saad-production --title "Recover frozen SAAD inference and local review application" --body-file docs/PR_BODY.md
```

Do not merge or publish a release based on this local-loopback verification alone. Public network operation needs authentication, authorization, upload access controls and a separate deployment review. CPU/other-GPU numerical behavior, multi-user coordination, historical training reruns and external dataset rights remain outside the verified scope.
