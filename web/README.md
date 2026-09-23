# SAAD browser client

The original React layout is connected to the local SAAD API through `src/lib/api.ts`. The API base URL is `VITE_API_BASE_URL`; it defaults to `http://127.0.0.1:8000`. Copy `.env.example` to `.env.local` to change it without committing a local URL. Configure the backend's `SAAD_CORS_ORIGINS` for the web origin.

```powershell
npm ci
npm run dev -- --host 127.0.0.1
npm run typecheck
npm run lint
npm run build
```

The browser uploads to `/api/analyze`, retrieves scans by stable ID, persists reviews and corrected boxes through the API, and downloads JSON, CSV and PDF reports from the saved server state. The backend must be running and loaded with the validated external model assets. The original Analytics placeholder and decorative header controls are not verified functions.
