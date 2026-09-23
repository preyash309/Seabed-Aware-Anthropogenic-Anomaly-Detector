# SAAD local release checklist

Validation date: 2026-09-24. Branch: `refactor/saad-production`. This checklist describes the audited local Windows application, not an externally exposed deployment.

| Gate | Status | Evidence |
| --- | --- | --- |
| Frozen runtime tensors and calibration | **PASS** | Five model/normalization tensors and live calibration validated by configured size/SHA-256 before load; 197 unique original asset/result files in 202 manifest rows rehashed successfully. |
| Live evidence policy | **PASS** | `saad-live-api-v1` remains runtime default; offline v3 is archived research only. |
| Golden numerical regression | **PASS** | 25 tests; fixed GhostVision API seven candidates and domain stage fixtures 14/7/6; every reported maximum absolute difference 0.0. |
| Real HTTP and production browser | **PASS** | GhostVision, SubPipeMini2 and AI4Shipwrecks browser uploads returned 7/6/14; AI4 used the built Vite bundle. |
| Durable review, correction and audit | **PASS** | Accept/reject/correct retained after browser reload and backend restart; saved prediction box unchanged; three review events. |
| Scan history and priority queue | **PASS** | Browser history/direct links loaded saved scans. Pending queue now orders globally by immutable model priority; API regression test and restarted service confirmed it. |
| JSON, CSV and PDF reports | **PASS with scope** | Browser observed all three attachment downloads; API tests checked JSON/CSV review content and PDF signature. The built report displayed the corrected box. PDF text was not independently extracted. |
| Failure responses | **PASS** | HTTP malformed upload 400, missing scan 404, unsupported report format 422; a missing-asset preflight failed before model load with the named YOLO path; automated upload size/pixel and database tests passed. |
| Frontend quality gates | **PASS** | `npm ci`, TypeScript check, ESLint and Vite production build passed; 1,858 transformed modules, zero reported install vulnerabilities. |
| Research and original integrity | **PASS with scope** | 31 archived source hashes matched originals; 37 ML/backend and 40 web original source sizes matched audit; 197 unique hashed external assets matched. Original web/backend source content beyond the archived files had size-only audit baselines. |
| Tracked-file hygiene | **PASS** | No tracked model/dataset/image/database/secret/local `.env` or generated output; no tracked file above 1 MiB. Runtime outputs and dependency directories ignored. |
| Fresh Python wheel installation | **UNVERIFIED** | Exact Torch `2.11.0+cu128` wheel download was interrupted; working isolated `.venv` contains read-only-copied package files. `pip check` passes, but a from-wheel build was not completed. |
| Other hardware and external access | **UNVERIFIED** | CPU, second GPU, fresh Windows machine, multi-user security and public deployment were not exercised. Keep this build on loopback. |

Do not push, merge or deploy from this checklist alone. The branch has reviewable local commits and no dataset, checkpoint, result or credential in Git. See [STAGE6_VERIFICATION.md](STAGE6_VERIFICATION.md) and [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).
