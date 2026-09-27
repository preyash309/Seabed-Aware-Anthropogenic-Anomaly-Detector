# Browser client

The original React UI is retained with a typed API client and saved scan/review integration. The stack is React 19, TypeScript 6, Vite 8 and Tailwind CSS 4; [package.json](package.json) and [package-lock.json](package-lock.json) define the installed dependencies. Use Node 24/npm 11 as audited.

## Run and build

With the configured loopback backend running, execute from this directory:

```powershell
npm ci
npm run dev -- --host 127.0.0.1
```

For the production compilation:

```powershell
npm run typecheck
npm run lint
npm run build
npm run preview -- --host 127.0.0.1 --port 4173
```

Allow `http://127.0.0.1:4173` in backend CORS before previewing. Copy [.env.example](.env.example) to ignored `.env.local` to set `VITE_API_BASE_URL` (default `http://127.0.0.1:8000`). This is a build-time setting, so rebuild after changing it. See [Setup](../docs/SETUP.md).

## Routes

| Path | Page / behavior |
| --- | --- |
| `/` | Static system overview. |
| `/scan` | Select/preview an image and submit real multipart analysis. |
| `/scan/:id` | Load saved analysis/reviews; inspect candidates and save decisions/corrections. |
| `/history` | Saved scan list from the API. |
| `/review` | Pending candidate queue from the API. |
| `/reports` | Select a saved scan and download JSON/CSV/PDF; optional `scanId` query. |
| `/analytics` | Placeholder, not a connected analytics feature. |

[App.tsx](src/App.tsx) owns routing. [pages/](src/pages/) implements these flows. Decorative header controls are not verified capabilities.

## Data flow

[lib/api.ts](src/lib/api.ts) sends requests and exposes typed transport functions. [types/saad.ts](src/types/saad.ts) describes responses; TypeScript types do not provide complete runtime response validation. Analysis checks success, survey ID and a candidate array before navigation.

[lib/adapt.ts](src/lib/adapt.ts) combines immutable prediction data with separate saved review state. It retains `modelBBox` and uses the corrected box, if present, as the display `bbox`. Image URLs are resolved against the configured API base. Human decisions are not stored as local-storage truth.

NewScan sends FormData field `file`, receives the scan ID and navigates to analysis. Analysis loads the scan on direct navigation/refresh; after a saved action it fetches updated state. History, queue and reports query the same persisted API.

## Upload and box review

The file picker accepts PNG, JPEG, WebP and TIFF MIME types. Drag/drop takes only the first file and requires a MIME type starting with `image/`. The API additionally supports BMP filenames. The UI displays size but does not enforce the server byte/pixel limits; decoded-file validation and 25 MiB / 50-million-pixel defaults are backend responsibilities.

The [SonarViewer](src/components/sonar/SonarViewer.tsx) uses the rendered `object-contain` image rectangle to place percentage boxes. Selecting a candidate exposes its original model evidence. Correction mode starts from the display box; drag/resize changes remain local until Save sends CORRECT. Cancellation does not persist. Accept/reject sends its action and clears any earlier saved correction according to the API contract.

Saved reviews survive refresh/direct navigation and backend restart when local SQLite/uploads are retained. Predictions and saved review state remain distinct; the prediction's original review-status field is not the current human state. See [Backend API](../backend/README.md#api-contract).

## Reports and validation scope

The Reports page uses saved scans and the report endpoint. JSON downloads use `download=true`; CSV/PDF are attachments. CSV candidate rows and text-only PDF summaries have different content from the full JSON report.

Local loopback browser verification covered real uploads, candidate review, correction persistence, history, queue and report downloads, including a built client. Hosted CI checks types, lint and build but has no automated browser or model-backed workflow test. No public hosting, alternate server routing or multi-user operation is claimed. See [Evaluation](../docs/EVALUATION.md) and [Limitations](../docs/LIMITATIONS.md).
