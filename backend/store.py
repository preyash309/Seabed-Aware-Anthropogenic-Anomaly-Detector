"""Durable immutable predictions and separate mutable human reviews."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from urllib.parse import urlsplit

from schemas import AnalysisResponse, ReviewRequest


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


class ScanNotFound(LookupError):
    pass


class CandidateNotFound(LookupError):
    pass


class ScanStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS scans (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    image_name TEXT NOT NULL,
                    image_sha256 TEXT NOT NULL,
                    analysis_status TEXT NOT NULL CHECK (analysis_status IN ('complete','partial')),
                    analysis_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    scan_id TEXT NOT NULL REFERENCES scans(id),
                    candidate_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    prediction_json TEXT NOT NULL,
                    PRIMARY KEY (scan_id, candidate_id)
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    scan_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('ACCEPTED','REJECTED','CORRECTED')),
                    corrected_box_json TEXT,
                    revision INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (scan_id, candidate_id),
                    FOREIGN KEY (scan_id, candidate_id) REFERENCES candidates(scan_id, candidate_id)
                );
                CREATE TABLE IF NOT EXISTS review_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    prior_status TEXT,
                    prior_box_json TEXT,
                    corrected_box_json TEXT,
                    note TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (scan_id, candidate_id) REFERENCES candidates(scan_id, candidate_id)
                );
                CREATE INDEX IF NOT EXISTS scans_created_idx ON scans(created_at DESC);
                CREATE INDEX IF NOT EXISTS reviews_status_idx ON reviews(status);
            """)

    def save_scan(self, analysis: dict, image_sha256: str) -> None:
        AnalysisResponse.model_validate(analysis)
        scan_id = analysis["surveyId"]
        image_name = Path(urlsplit(analysis["image"]["url"]).path).name
        candidates = analysis["candidates"]
        partial = any(
            candidate.get(key) is None
            for candidate in candidates
            for key in ("vaeScore", "flowScore", "ttaConsistency")
        )
        with self.connection() as db:
            db.execute(
                "INSERT INTO scans VALUES (?,?,?,?,?,?,?)",
                (scan_id, _now(), analysis["filename"], image_name, image_sha256,
                 "partial" if partial else "complete", _json(analysis)),
            )
            db.executemany(
                "INSERT INTO candidates VALUES (?,?,?,?)",
                [(scan_id, candidate["id"], position, _json(candidate))
                 for position, candidate in enumerate(candidates)],
            )

    def get_scan(self, scan_id: str) -> dict:
        with self.connection() as db:
            scan = db.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
            if scan is None:
                raise ScanNotFound(scan_id)
            rows = db.execute(
                "SELECT candidate_id,status,corrected_box_json,revision,updated_at "
                "FROM reviews WHERE scan_id=?", (scan_id,),
            ).fetchall()
        reviews = {
            row["candidate_id"]: {
                "status": row["status"],
                "correctedBBox": json.loads(row["corrected_box_json"])
                if row["corrected_box_json"] else None,
                "revision": row["revision"],
                "updatedAt": row["updated_at"],
            }
            for row in rows
        }
        for candidate in json.loads(scan["analysis_json"])["candidates"]:
            reviews.setdefault(candidate["id"], {
                "status": "PENDING", "correctedBBox": None,
                "revision": 0, "updatedAt": None,
            })
        return {
            "scanId": scan["id"], "createdAt": scan["created_at"],
            "analysisStatus": scan["analysis_status"],
            "analysis": json.loads(scan["analysis_json"]), "reviews": reviews,
        }

    def list_scans(self, limit: int = 100) -> list[dict]:
        with self.connection() as db:
            rows = db.execute(
                "SELECT id,created_at,filename,image_name,analysis_status,analysis_json "
                "FROM scans ORDER BY created_at DESC,id DESC LIMIT ?", (limit,),
            ).fetchall()
        return [{
            "scanId": row["id"], "createdAt": row["created_at"],
            "filename": row["filename"], "analysisStatus": row["analysis_status"],
            "imageUrl": json.loads(row["analysis_json"])["image"]["url"],
            "summary": json.loads(row["analysis_json"])["summary"],
        } for row in rows]

    def review_queue(self, limit: int = 100) -> list[dict]:
        with self.connection() as db:
            rows = db.execute(
                "SELECT c.scan_id,c.candidate_id,c.prediction_json,s.created_at,r.status "
                "FROM candidates c JOIN scans s ON s.id=c.scan_id "
                "LEFT JOIN reviews r ON r.scan_id=c.scan_id AND r.candidate_id=c.candidate_id "
                "WHERE r.status IS NULL ORDER BY s.created_at DESC,c.position",
            ).fetchall()
        pending = [{
            "scanId": row["scan_id"], "candidateId": row["candidate_id"],
            "createdAt": row["created_at"], "prediction": json.loads(row["prediction_json"]),
        } for row in rows]
        pending.sort(key=lambda item: item["prediction"]["priority"], reverse=True)
        return pending[:limit]

    def review(self, scan_id: str, candidate_id: str, request: ReviewRequest) -> dict:
        status = {"ACCEPT": "ACCEPTED", "REJECT": "REJECTED", "CORRECT": "CORRECTED"}[request.action]
        corrected = _json(request.bbox.model_dump()) if request.bbox else None
        timestamp = _now()
        with self.connection() as db:
            if db.execute("SELECT 1 FROM scans WHERE id=?", (scan_id,)).fetchone() is None:
                raise ScanNotFound(scan_id)
            if db.execute(
                "SELECT 1 FROM candidates WHERE scan_id=? AND candidate_id=?",
                (scan_id, candidate_id),
            ).fetchone() is None:
                raise CandidateNotFound(candidate_id)
            previous = db.execute(
                "SELECT status,corrected_box_json,revision FROM reviews "
                "WHERE scan_id=? AND candidate_id=?", (scan_id, candidate_id),
            ).fetchone()
            revision = previous["revision"] + 1 if previous else 1
            db.execute(
                "INSERT INTO reviews VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(scan_id,candidate_id) DO UPDATE SET "
                "status=excluded.status,corrected_box_json=excluded.corrected_box_json,"
                "revision=excluded.revision,updated_at=excluded.updated_at",
                (scan_id, candidate_id, status, corrected, revision, timestamp),
            )
            db.execute(
                "INSERT INTO review_events "
                "(scan_id,candidate_id,action,prior_status,prior_box_json,corrected_box_json,note,created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (scan_id, candidate_id, request.action,
                 previous["status"] if previous else None,
                 previous["corrected_box_json"] if previous else None,
                 corrected, request.note, timestamp),
            )
        return {
            "status": status,
            "correctedBBox": request.bbox.model_dump() if request.bbox else None,
            "revision": revision, "updatedAt": timestamp,
        }

    def events(self, scan_id: str) -> list[dict]:
        with self.connection() as db:
            if db.execute("SELECT 1 FROM scans WHERE id=?", (scan_id,)).fetchone() is None:
                raise ScanNotFound(scan_id)
            rows = db.execute(
                "SELECT * FROM review_events WHERE scan_id=? ORDER BY event_id",
                (scan_id,),
            ).fetchall()
        return [{
            "eventId": row["event_id"], "scanId": row["scan_id"],
            "candidateId": row["candidate_id"], "action": row["action"],
            "priorStatus": row["prior_status"],
            "priorBBox": json.loads(row["prior_box_json"]) if row["prior_box_json"] else None,
            "correctedBBox": json.loads(row["corrected_box_json"])
            if row["corrected_box_json"] else None,
            "note": row["note"], "createdAt": row["created_at"],
        } for row in rows]
