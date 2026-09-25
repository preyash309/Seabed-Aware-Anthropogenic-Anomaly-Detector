## Summary

- Recover the frozen SAAD YOLO26s, ConvVAE, RealNVP and TTA inference path with manifest-validated external assets and the original `/api/analyze` response.
- Keep `saad-live-api-v1` as the deployed evidence policy. Archive the distinct offline Evidence Engine v3 as research provenance.
- Add typed FastAPI routes, durable SQLite scans/reviews/audit events, and a React workflow for upload, candidate review, history, queue and JSON/CSV/PDF reports.
- Preserve 31 original research scripts with source hashes and guard the supported dataset builders.

## Verification

- 25 backend tests passed on the audited Windows RTX 4070 machine. The GhostVision API fixture returned seven candidates; the three domain fixtures returned 14/7/6. Every reported maximum numerical difference was 0.0 with unchanged fixtures and tolerances.
- Five external tensors, live calibration and three fixed image hashes validated. The external asset ledger rehashed 197 unique files; all 31 archived research scripts matched their original hashes.
- `npm ci`, frontend type checking, linting and production build passed. A built-browser GhostVision upload returned seven candidates; an accepted review survived reload and appeared in the report. Loopback HTTP readiness, scan/review-event and report retrieval passed.
- Publication hygiene and the fresh isolated Python installation result are documented in `PUBLICATION_REVIEW.md`.

## GitHub Actions

PR and main-branch CI run three independent jobs: 13 asset-independent backend tests plus `pip check` on CPU, frontend `npm ci`/typecheck/lint/build, and tracked-file/history hygiene. The 12 backend tests requiring private tensors, fixed sonar images or model-loaded routes remain local-only. GitHub Actions has no model weights, dataset or GPU golden-parity claim; [docs/CI.md](docs/CI.md) lists the exact split. The Windows RTX 4070 verification above is a separate completed local result.
## Installation and limits

The repository does **not** include the five model/normalization tensors, live calibration, datasets, test images, runtime SQLite files or generated results. Supply authorized external assets matching `config/model_manifest.json`; see `README.md` and `docs/DEPLOYMENT.md` for the pinned Windows/CUDA setup and local launch commands.

The verified operating boundary is local loopback on the audited GPU. The API has no authentication. CPU/other-GPU parity, multi-user operation, historical training reruns and external deployment remain unverified. The offline v3 policy is not the deployed policy.
