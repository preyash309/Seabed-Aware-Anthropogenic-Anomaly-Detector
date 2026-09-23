"""Boundary checks for one model registry, route wiring, and serialization."""

import asyncio
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import main  # noqa: E402
import routes  # noqa: E402
from saad_inference.response import serialize_response  # noqa: E402


class OrchestrationBoundaries(unittest.TestCase):
    def test_registry_and_compatibility_entry_point_share_one_model_set(self):
        self.assertIs(main.yolo_model, routes.MODELS.yolo_model)
        self.assertIs(main.vae_model, routes.MODELS.vae_model)
        self.assertIs(main.flow_model, routes.MODELS.flow_model)
        self.assertIs(main.app, routes.app)
        self.assertEqual(main.SETTINGS.calibration["policy_id"], "saad-live-api-v1")

    def test_route_passes_loaded_models_to_service(self):
        upload = UploadFile(filename="sample.png", file=io.BytesIO(b"sample"))
        with patch.object(routes, "run_analysis", new_callable=AsyncMock) as mocked:
            mocked.return_value = {"success": True}
            actual = asyncio.run(routes.analyze(upload))
        self.assertEqual(actual, {"success": True})
        mocked.assert_awaited_once_with(upload, routes.MODELS, routes.SETTINGS)

    def test_response_sorts_and_renumbers_original_candidates(self):
        candidates = [
            {"priority": 0.1, "id": "old-1", "priorityLevel": "LOW", "recommendation": "LOW_PRIORITY_REVIEW"},
            {"priority": 0.9, "id": "old-2", "priorityLevel": "HIGH", "recommendation": "HIGH_PRIORITY"},
            {"priority": 0.5, "id": "old-3", "priorityLevel": "REVIEW", "recommendation": "UNCERTAIN_REVIEW"},
        ]

        class ImageStub:
            format = "PNG"

        response = serialize_response(
            candidates, "SAAD-TEST", "input.png", ImageStub(), 640, 320,
            "saved.png", "http://localhost:8000", "cuda:0",
        )
        self.assertEqual([item["priority"] for item in response["candidates"]], [0.9, 0.5, 0.1])
        self.assertEqual([item["id"] for item in response["candidates"]],
                         ["candidate-01", "candidate-02", "candidate-03"])
        self.assertEqual(response["summary"], {
            "totalCandidates": 3, "highPriority": 1, "review": 1,
            "uncertain": 1, "lowPriority": 1,
        })
        self.assertEqual(response["image"]["url"], "http://localhost:8000/uploads/saved.png")

    def test_invalid_uploads_keep_original_http_errors(self):
        from saad_inference.service import analyze

        cases = [
            (None, b"data", "No filename provided."),
            ("sample.txt", b"data", "Unsupported image format."),
            ("sample.png", b"", "Uploaded file is empty."),
            ("sample.png", b"not an image", "Invalid image:"),
        ]
        for filename, contents, expected in cases:
            with self.subTest(filename=filename, expected=expected):
                upload = UploadFile(filename=filename, file=io.BytesIO(contents))
                with self.assertRaises(HTTPException) as caught:
                    asyncio.run(analyze(upload, routes.MODELS, routes.SETTINGS))
                self.assertEqual(caught.exception.status_code, 400)
                self.assertIn(expected, caught.exception.detail)


if __name__ == "__main__":
    unittest.main()
