# Backend developer guide

The FastAPI backend runs frozen inference, validates uploads, persists scans/reviews and exports reports. Run commands below from the repository root. See [Setup](../docs/SETUP.md) for dependencies and [Architecture](../docs/ARCHITECTURE.md) for preprocessing, model dimensions and exact evidence rules.

## Modules and dependencies

| Module | Responsibility |
| --- | --- |
| [main.py](main.py) | Compatible Uvicorn entry point and legacy exports. |
| [config.py](config.py) | Environment settings, path confinement, manifest/calibration validation. |
| [saad_inference/registry.py](saad_inference/registry.py) | Validate assets and initialize the single model set for a process. |
| [preprocessing.py](saad_inference/preprocessing.py), [detector.py](saad_inference/detector.py) | Frozen crop geometry and YOLO inference/box decoding. |
| [vae.py](saad_inference/vae.py), [realnvp.py](saad_inference/realnvp.py) | Canonical architectures, loading and anomaly scores. |
| [tta.py](saad_inference/tta.py), [live_api_v1.py](saad_inference/evidence/live_api_v1.py) | Detector consistency and deployed evidence policy. |
| [service.py](saad_inference/service.py), [response.py](saad_inference/response.py) | End-to-end orchestration, ordering and compatible response. |
| [routes.py](routes.py), [schemas.py](schemas.py) | HTTP boundary, typed persisted/review contracts and static uploads. |
| [store.py](store.py), [reporting.py](reporting.py) | SQLite state, review events and JSON/CSV/PDF serialization. |

Dependency direction: routes → settings/registry/service/store/reporting; service → detector/crops/VAE/flow/TTA/policy/response. Research scripts are not runtime dependencies. Three `*_inference.py` files preserve older imports.

Routes load models at import time. Starting another worker loads another model set; request handlers do not reload the weights. A server cannot provide even health metadata before asset validation and loading succeed.

## Start and status

After [installation and configuration](../docs/SETUP.md):

```powershell
New-Item -ItemType Directory -Path var\ultralytics -Force | Out-Null
$env:YOLO_AUTOINSTALL='False'
$env:YOLO_CONFIG_DIR=(Resolve-Path var\ultralytics).Path
.\.venv\Scripts\python.exe -B -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

`GET /api/health` reports loaded model/device metadata. `GET /api/ready` additionally checks SQLite with `SELECT 1`. Neither proves future inference success. The default FastAPI interactive schema is at `/docs`. Keep the service on loopback; model status includes local checkpoint paths.

## API contract

| Method | Path | Input / result |
| --- | --- | --- |
| GET | `/` | Service/version metadata. |
| GET | `/api/health` | Device, model-set and live-policy metadata. |
| GET | `/api/model` | Model paths, dimensions and policy configuration. |
| GET | `/api/ready` | Model-started service plus storage readiness. |
| POST | `/api/analyze` | Multipart field `file`; original analysis response, persisted before HTTP success. |
| GET | `/api/scans` | Newest-first scan summaries; `limit` defaults to 100, range 1–500. |
| GET | `/api/review-queue` | Pending candidates by descending saved model priority; same limit range. |
| GET | `/api/scans/{scan_id}` | Saved analysis, status and separate review map. |
| PUT | `/api/scans/{scan_id}/candidates/{candidate_id}/review` | ACCEPT, REJECT or CORRECT; returns current review state. |
| GET | `/api/scans/{scan_id}/review-events` | Saved review-event list. |
| GET | `/api/scans/{scan_id}/report` | `format=json\|csv\|pdf`; JSON `download=true` adds attachment disposition; CSV/PDF always attach. |
| GET | `/uploads/{name}` | Saved image served from the configured upload directory. |

### Analysis fields

Top-level fields: `success`, `surveyId`, `filename`, `image`, `processing`, `summary`, `candidates`, `pipeline`. Image includes width, height, format and URL. Processing identifies device, detector, image size, threshold, TTA and evidence engine.

Each candidate includes `id`, `bbox` (percentage x/y/width/height), `bbox_pixels` (x1/y1/x2/y2), `classId`, `className`, `yoloConfidence`, nullable `vaeScore`/`flowScore`/`ttaConsistency`, `priority`, `uncertainty`, `priorityLevel`, `evidenceProfile`, `recommendation`, `reviewStatus` and detailed `evidence`. Models/schema allow additional compatibility fields. [Numeric fixtures](../audit/baseline_capture.json) provide captured values without image bytes.

Scan retrieval returns `scanId`, `createdAt`, `analysisStatus` (complete/partial), `analysis` and `reviews` keyed by candidate ID. Review state is authoritative for human decisions; the prediction's original `reviewStatus` is not updated. Scan IDs are `SAAD-` plus eight uppercase UUID hex characters; candidate IDs are stable within a scan, assigned after priority sorting.

### Review body

```json
{"action":"ACCEPT"}
```

```json
{"action":"CORRECT","bbox":{"x":10,"y":12,"width":20,"height":18},"note":"Box adjusted"}
```

`CORRECT` requires a box; ACCEPT/REJECT must omit it. x/y are 0–100; width/height are positive and ≤100; the full box must fit within the image with the implemented `1e-6` boundary tolerance. Optional note is limited to 2,000 characters. The browser does not currently supply a note.

### Error behavior

| Boundary | HTTP behavior |
| --- | --- |
| Missing filename, unsupported extension, empty bytes or invalid decoded image | 400 |
| Default 25 MiB byte / 50-million-pixel limit exceeded | 413 |
| Missing multipart field, invalid review schema or query values | 422 |
| Unknown scan/candidate | 404 |
| YOLO inference failure | 500 |
| Handled SQLite/storage or persistence-validation failure | 503 |
| VAE/flow or TTA scoring failure | Can return success with partial evidence and saved `analysisStatus=partial`. |

Attempted uploads are removed for handled image, detector and persistence failures. A successful scan retains its image. This is not a guarantee for arbitrary filesystem failures; the application has no comprehensive recovery or upload retention policy.

## Storage and exports

SQLite uses WAL, foreign keys, per-operation connections and transactions. Four tables hold scans (image identity and analysis JSON), candidate predictions, current reviews and review events. No review mutates a prediction. PENDING revision 0 is projected for unreviewed candidates; actions increment revision and append prior/new state atomically. Accept/reject removes an earlier corrected box. No deletion or reset endpoint exists.

The pending queue excludes reviewed candidates. Saved review state survives refresh and service restart when the database and upload files are retained. Back up both; copying SQLite state alone does not preserve images.

Reports read saved predictions and current reviews. JSON includes events and metadata; CSV has 22 candidate columns and spreadsheet-formula escaping; PDF is text-only with pagination. Current settings supply report model/policy IDs at generation time. CSV does not include the entire event trail. Reports do not turn reviewed counts into an accuracy estimate.

## Development checks

Use [tests/README.md](../tests/README.md) for the 13 asset-independent tests and complete 25-test local suite. Run the full golden suite after any numerical change, with the API stopped. A passing asset-free test slice does not establish model-startup compatibility or GPU parity. Do not modify frozen fixtures or tolerances.
