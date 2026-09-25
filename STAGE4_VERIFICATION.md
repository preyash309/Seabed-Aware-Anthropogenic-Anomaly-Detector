# Stage 4 verification — React integration and reports

Date: 2026-09-23. Branch: `refactor/saad-production`. The original ML and web folders remained read-only; all changes and generated runtime files are under the clean repository.

## Implemented

The original React layout now calls the typed API client configured by `VITE_API_BASE_URL`. Upload analysis navigates to a saved scan ID. Direct analysis and report URLs reload from SQLite. Candidate accept, reject and corrected percentage boxes are persisted separately from immutable model predictions. History and pending review queue are server backed. Report JSON, CSV and PDF are generated from stored predictions, reviews and audit events; the report records `saad-live-api-v1`. The UI displays normalized evidence using server-provided values and shows raw Flow NLL as a raw score. Device text comes from `/api/health` instead of a hardcoded GPU claim. No research policy was deployed.

## Commands and results

Run from `D:/SAAD/SAAD` after configuring the five external audited assets and the clean-repository runtime paths:

```powershell
$env:YOLO_AUTOINSTALL='False'
$env:YOLO_CONFIG_DIR='D:\SAAD\SAAD\var\ultralytics'
$env:SAAD_ARTIFACT_DIR='E:\SIH\SIH_Results'
$env:SAAD_UPLOAD_DIR='D:\SAAD\SAAD\var\uploads'
$env:SAAD_OUTPUT_DIR='D:\SAAD\SAAD\var\results'
& 'D:\SAAD\SAAD\.venv\Scripts\python.exe' -B -m unittest discover -s 'D:\SAAD\SAAD\tests' -p 'test_stage*.py' -q
```

Result: **22 passed, 0 failures, 0 skips**. The GhostVision API fixture retained seven candidates; domain stage fixtures retained 14/7/6. Every reported maximum absolute golden difference was **0.0**, including boxes, detector confidence, VAE MSE, Flow NLL, TTA, policy scores, uncertainty and priority. No tolerance changed. The Stage 4 report test asserts immutable model boxes, a corrected review box, two audit events, JSON attachment, CSV fields and a `%PDF-` response.

From `D:/SAAD/SAAD/web`: `npm ci`, `npm run typecheck`, `npm run lint`, `npm run build` all passed. The final Vite build transformed 1,858 modules. Package installation reported zero vulnerabilities. `web/vite.config.ts` uses `import.meta.dirname` to avoid the Vite native-loader warning.

## Real browser and HTTP workflow

With local Uvicorn at `127.0.0.1:8000` and Vite at `127.0.0.1:5173`, an in-app browser selected the fixed GhostVision image, submitted the upload and reached saved scan `SAAD-7B822882` with seven candidates. Candidate 01 was accepted, 02 rejected and 03 moved and resized then corrected. The page showed three of seven reviewed; after reload, the reviewed count and corrected overlay dimensions persisted. The report showed Accepted, Rejected and Corrected and the corrected percentage box. History linked to the saved scan and report; the review queue listed only pending candidates. Browser download events were observed for JSON, CSV and PDF after the JSON export was switched to a server attachment. The browser console had no error entries during the tested flow.

A second browser upload used the fixed SubPipeMini2 image and produced saved scan `SAAD-DF2B426F` with six candidates. For a tall AI4Shipwrecks scan (1728×1927) and the wide SubPipe scan (2500×500), read-only DOM geometry measurements showed the overlay frame matched the independently calculated `object-contain` image rectangle within 0.005 CSS pixels. The saved AI4 scan had five candidates; its upload was not performed in this verification session. The real backend served the sonar images by HTTP 200.

## Limits

The current machine used the original package files copied read-only into the clean repository `.venv`; a fresh pip installation of the exact Torch CUDA wheel remains **UNVERIFIED** because the 2.8 GB download was interrupted. CPU, another GPU, external deployment and multi-user access remain **UNVERIFIED**. The local API has no authentication and should remain on loopback. The React Analytics route and decorative header controls remain nonfunctional original UI elements and are not part of the verified workflow. Model evidence still uses the live API v1 policy; offline Evidence Engine v3 remains research only.
