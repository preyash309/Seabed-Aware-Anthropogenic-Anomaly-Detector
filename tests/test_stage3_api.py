"""HTTP contract and durable review tests with the frozen real model set."""

from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import routes  # noqa: E402
from store import ScanStore  # noqa: E402


class DurableApi(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "var")
        self.addCleanup(self.temporary.cleanup)
        self.store = ScanStore(Path(self.temporary.name) / "scans.sqlite3")
        self.patch_store = patch.object(routes, "STORE", self.store)
        self.patch_store.start()
        self.addCleanup(self.patch_store.stop)
        self.client = TestClient(routes.app)

    def test_http_upload_review_restart_and_image(self):
        fixture = json.loads((ROOT / "audit/baseline_capture.json").read_text())
        image = Path(fixture["source"])
        before = set(routes.SETTINGS.upload_dir.iterdir())
        try:
            response = self.client.post(
                "/api/analyze", files={"file": (image.name, image.read_bytes(), "image/png")},
            )
            self.assertEqual(response.status_code, 200, response.text)
            analysis = response.json()
            self.assertEqual(set(analysis), {
                "success", "surveyId", "filename", "image", "processing",
                "summary", "candidates", "pipeline",
            })
            self.assertEqual(len(analysis["candidates"]), 7)
            scan_id = analysis["surveyId"]
            candidate = analysis["candidates"][0]
            self.assertEqual(candidate["id"], "candidate-01")
            self.assertEqual(self.client.get(analysis["image"]["url"]).status_code, 200)
            self.assertEqual(self.client.get("/api/ready").status_code, 200)

            detail = self.client.get(f"/api/scans/{scan_id}")
            self.assertEqual(detail.status_code, 200, detail.text)
            self.assertEqual(detail.json()["analysis"], analysis)
            self.assertEqual(detail.json()["reviews"][candidate["id"]]["status"], "PENDING")
            self.assertEqual(self.client.get("/api/scans").json()[0]["scanId"], scan_id)
            queue = self.client.get("/api/review-queue").json()
            self.assertEqual(len(queue), 7)

            review_url = f"/api/scans/{scan_id}/candidates/{candidate['id']}/review"
            accepted = self.client.put(review_url, json={"action": "ACCEPT"})
            self.assertEqual(accepted.status_code, 200, accepted.text)
            self.assertEqual(accepted.json()["status"], "ACCEPTED")
            self.assertEqual(len(self.client.get("/api/review-queue").json()), 6)
            corrected_box = {"x": 10, "y": 20, "width": 30, "height": 40}
            corrected = self.client.put(review_url, json={"action": "CORRECT", "bbox": corrected_box})
            self.assertEqual(corrected.status_code, 200, corrected.text)
            self.assertEqual(corrected.json()["revision"], 2)
            self.assertEqual(corrected.json()["correctedBBox"], corrected_box)
            self.assertEqual(len(self.client.get(f"/api/scans/{scan_id}/review-events").json()), 2)

            reopened = ScanStore(self.store.path)
            saved = reopened.get_scan(scan_id)
            self.assertEqual(saved["reviews"][candidate["id"]]["correctedBBox"], corrected_box)
            self.assertEqual(saved["analysis"]["candidates"][0]["bbox"], candidate["bbox"])
            self.assertEqual(saved["analysis"]["candidates"][0]["priority"], candidate["priority"])

            report = self.client.get(f"/api/scans/{scan_id}/report")
            self.assertEqual(report.status_code, 200, report.text)
            report_data = report.json()
            self.assertEqual(report_data["policyId"], "saad-live-api-v1")
            self.assertEqual(report_data["candidates"][0]["prediction"]["bbox"], candidate["bbox"])
            self.assertEqual(report_data["candidates"][0]["review"]["correctedBBox"], corrected_box)
            self.assertEqual(len(report_data["reviewEvents"]), 2)
            json_download = self.client.get(f"/api/scans/{scan_id}/report?format=json&download=true")
            self.assertEqual(json_download.status_code, 200)
            self.assertIn("attachment", json_download.headers["content-disposition"])
            self.assertEqual(json_download.json()["candidates"][0]["review"]["correctedBBox"], corrected_box)
            csv = self.client.get(f"/api/scans/{scan_id}/report?format=csv")
            self.assertEqual(csv.status_code, 200)
            self.assertIn("corrected_x", csv.text)
            self.assertIn("CORRECTED", csv.text)
            pdf = self.client.get(f"/api/scans/{scan_id}/report?format=pdf")
            self.assertEqual(pdf.status_code, 200)
            self.assertTrue(pdf.content.startswith(b"%PDF-"))

            partial = json.loads(json.dumps(analysis))
            partial["surveyId"] = scan_id + "-PARTIAL"
            partial["candidates"][0]["vaeScore"] = None
            reopened.save_scan(partial, "0" * 64)
            self.assertEqual(reopened.get_scan(partial["surveyId"])["analysisStatus"], "partial")
        finally:
            for saved in set(routes.SETTINGS.upload_dir.iterdir()) - before:
                if saved.is_file() and saved.suffix.lower() == ".png":
                    saved.unlink()

    def test_invalid_limits_missing_records_and_storage_failure(self):
        self.assertEqual(self.client.post(
            "/api/analyze", files={"file": ("bad.txt", b"data", "text/plain")},
        ).status_code, 400)
        self.assertEqual(self.client.post(
            "/api/analyze", files={"file": ("empty.png", b"", "image/png")},
        ).status_code, 400)
        with patch.object(routes, "SETTINGS", replace(routes.SETTINGS, max_upload_bytes=2)):
            self.assertEqual(self.client.post(
                "/api/analyze", files={"file": ("large.png", b"abc", "image/png")},
            ).status_code, 413)
        fixture = json.loads((ROOT / "audit/baseline_capture.json").read_text())
        image = Path(fixture["source"])
        with patch.object(routes, "SETTINGS", replace(routes.SETTINGS, max_image_pixels=1)):
            self.assertEqual(self.client.post(
                "/api/analyze", files={"file": (image.name, image.read_bytes(), "image/png")},
            ).status_code, 413)
        before = set(routes.SETTINGS.upload_dir.iterdir())
        with patch.object(self.store, "save_scan", side_effect=sqlite3.OperationalError("forced")):
            self.assertEqual(self.client.post(
                "/api/analyze", files={"file": (image.name, image.read_bytes(), "image/png")},
            ).status_code, 503)
        self.assertEqual(set(routes.SETTINGS.upload_dir.iterdir()), before)
        self.assertEqual(self.client.get("/api/scans/missing").status_code, 404)
        self.assertEqual(self.client.put(
            "/api/scans/missing/candidates/candidate-01/review", json={"action": "REJECT"},
        ).status_code, 404)
        self.assertEqual(self.client.put(
            "/api/scans/missing/candidates/candidate-01/review", json={"action": "CORRECT"},
        ).status_code, 422)
        self.assertEqual(self.client.put(
            "/api/scans/missing/candidates/candidate-01/review",
            json={"action": "CORRECT", "bbox": {"x": 90, "y": 90, "width": 20, "height": 20}},
        ).status_code, 422)
        with patch.object(routes, "STORE", None), patch.object(routes, "STORE_ERROR", "forced"):
            self.assertEqual(self.client.get("/api/ready").status_code, 503)
            self.assertEqual(self.client.post(
                "/api/analyze", files={"file": ("valid.png", b"abc", "image/png")},
            ).status_code, 503)


if __name__ == "__main__":
    unittest.main()
