"""Original API and three-domain golden inference regression checks."""

import asyncio
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from fastapi import UploadFile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import main  # noqa: E402 — validates frozen assets at import


def image_path(record):
    original = Path(record.get("source", record.get("image")))
    mirror = os.environ.get("SAAD_GOLDEN_IMAGE_DIR")
    path = Path(mirror) / original.name if mirror else original
    if not path.is_file():
        raise AssertionError(f"Golden image unavailable: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != record.get("source_sha256", record.get("sha256")):
        raise AssertionError(f"Golden image SHA-256 mismatch: {path}")
    return path


class GoldenInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = json.loads((ROOT / "audit/baseline_capture.json").read_text())
        cls.stages = json.loads((ROOT / "audit/fixed_image_stage_capture.json").read_text())

    def compare(self, actual, expected, stage=False):
        fields = (
            (("yolo_confidence", 1e-5), ("vae_mse", 1e-6),
             ("realnvp_nll", 1e-2), ("tta_consistency", 1e-3)) if stage else
            (("yoloConfidence", 1e-5), ("vaeScore", 1e-6),
             ("flowScore", 1e-2), ("ttaConsistency", 1e-3),
             ("priority", 1e-3), ("uncertainty", 1e-3))
        )
        for key, tolerance in fields:
            self.assertIsNotNone(actual[key])
            self.assertAlmostEqual(actual[key], expected[key], delta=tolerance, msg=key)
        if stage:
            self.assertEqual(actual["class_id"], expected["class_id"])
            for a, b in zip(actual["bbox_xyxy"], expected["bbox_xyxy"]):
                self.assertAlmostEqual(a, b, delta=1.0)
            actual_evidence, expected_evidence = actual["evidence"], expected["evidence"]
            for key in ("priority", "uncertainty", "baseEvidence"):
                self.assertAlmostEqual(actual_evidence[key], expected_evidence[key], delta=1e-3)
            for key in ("evidenceProfile", "priorityLevel", "recommendation"):
                self.assertEqual(actual_evidence[key], expected_evidence[key])
        else:
            self.assertEqual(actual["classId"], expected["classId"])
            for key in ("priorityLevel", "evidenceProfile", "recommendation"):
                self.assertEqual(actual[key], expected[key])
            for key in ("x1", "y1", "x2", "y2"):
                self.assertAlmostEqual(actual["bbox_pixels"][key], expected["bbox_pixels"][key], delta=1)
            for key in ("yolo", "vae", "flow"):
                self.assertAlmostEqual(
                    actual["evidence"]["normalized"][key],
                    expected["evidence"]["normalized"][key], delta=1e-3,
                )

    def test_original_api(self):
        source = image_path(self.api)
        before = set(main.UPLOAD_DIR.iterdir())
        response = None
        try:
            upload = UploadFile(filename=source.name, file=io.BytesIO(source.read_bytes()))
            response = asyncio.run(main.analyze(upload))
            self.assertTrue(response["success"])
            actual, expected = response["candidates"], self.api["run_1"]
            self.assertEqual(len(actual), len(expected))
            self.assertEqual(response["summary"]["totalCandidates"], len(expected))
            for candidate, golden in zip(actual, expected):
                self.compare(candidate, golden)
        finally:
            if response is not None:
                generated = main.UPLOAD_DIR / Path(response["image"]["url"]).name
                if generated not in before and generated.is_file():
                    if hashlib.sha256(generated.read_bytes()).hexdigest() == self.api["source_sha256"]:
                        generated.unlink()

    def test_three_domain_stages(self):
        for record in self.stages:
            with self.subTest(image=record["image"]):
                source = image_path(record)
                image = Image.open(source)
                image.load()
                results = main.yolo_model.predict(
                    source=str(source), imgsz=640, conf=0.05, device=main.DEVICE, verbose=False
                )
                actual = []
                boxes = results[0].boxes if results else None
                if boxes is not None:
                    for i in range(len(boxes)):
                        raw = boxes.xyxy[i].detach().cpu().numpy()
                        box = [max(0.0, min(float(raw[j]), image.size[j % 2])) for j in range(4)]
                        confidence = float(boxes.conf[i].detach().cpu().item())
                        class_id = int(boxes.cls[i].detach().cpu().item())
                        patch = main.extract_candidate_patch(
                            image, dict(zip(("x1", "y1", "x2", "y2"), box))
                        )
                        tensor = torch.from_numpy(np.asarray(patch, dtype=np.float32) / 255.0)
                        tensor = tensor.unsqueeze(0).unsqueeze(0).to(main.DEVICE)
                        with torch.inference_mode():
                            mu, _ = main.vae_model.encode(tensor)
                            reconstruction = main.vae_model.decode(mu)
                            vae = float(torch.mean((reconstruction - tensor) ** 2).item())
                        flow = main.score_latent(
                            main.flow_model, main.flow_latent_mean, main.flow_latent_std, mu
                        )
                        actual.append({"bbox_xyxy": box, "class_id": class_id,
                                       "yolo_confidence": confidence, "vae_mse": vae,
                                       "realnvp_nll": flow})
                tta = main.compute_tta_consistency(
                    main.yolo_model, np.asarray(image.convert("RGB")),
                    [{"box": d["bbox_xyxy"], "confidence": d["yolo_confidence"],
                      "classId": d["class_id"]} for d in actual], main.DEVICE, 640,
                )
                for candidate, score in zip(actual, tta):
                    candidate["tta_consistency"] = score
                    candidate["evidence"] = main.build_evidence(
                        candidate["yolo_confidence"], candidate["vae_mse"],
                        candidate["realnvp_nll"], score,
                    )
                expected = record["detections_in_yolo_order"]
                self.assertEqual(len(actual), len(expected))
                for candidate, golden in zip(actual, expected):
                    self.compare(candidate, golden, stage=True)


if __name__ == "__main__":
    unittest.main()
