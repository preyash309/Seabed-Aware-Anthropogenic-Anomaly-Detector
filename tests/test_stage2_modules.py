"""Small behavior boundaries for extracted detector, crop and TTA modules."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from saad_inference.detector import decode_detection  # noqa: E402
from saad_inference.preprocessing import extract_candidate_patch  # noqa: E402
from saad_inference.tta import box_iou, match_detection, noise_perturbation  # noqa: E402


class ExtractedModules(unittest.TestCase):
    def test_detector_clamps_box_and_preserves_class(self):
        class Boxes:
            xyxy = torch.tensor([[-2.0, 5.0, 21.0, 15.0]])
            conf = torch.tensor([0.5])
            cls = torch.tensor([0.0])

        base = []
        confidence, class_id, pct, pixels = decode_detection(Boxes, 0, 20, 10, base)
        self.assertEqual((confidence, class_id), (0.5, 0))
        self.assertEqual(pixels, {"x1": 0.0, "y1": 5.0, "x2": 20.0, "y2": 10.0})
        self.assertEqual(pct, {"x": 0.0, "y": 50.0, "width": 100.0, "height": 50.0})
        self.assertEqual(base[0]["box"], [0.0, 5.0, 20.0, 10.0])

    def test_crop_shape_grayscale_and_rounding_path(self):
        rgb = Image.fromarray(np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3), "RGB")
        crop = extract_candidate_patch(
            rgb, {"x1": 5.25, "y1": 7.75, "x2": 15.25, "y2": 17.75}
        )
        self.assertEqual(crop.mode, "L")
        self.assertEqual(crop.size, (256, 256))

    def test_tta_match_threshold_and_seed(self):
        base = {"box": [0, 0, 10, 10], "confidence": 0.5, "classId": 0}
        variant = {"box": [0, 0, 3, 10], "confidence": 0.4, "classId": 0}
        self.assertEqual(box_iou(base["box"], variant["box"]), 0.3)
        match = match_detection(base, [variant])
        self.assertEqual(match["iou"], 0.3)
        self.assertAlmostEqual(match["confidenceAgreement"], 0.9)
        self.assertEqual(match_detection(base, [{**variant, "classId": 1}])["iou"], 0.0)

        image = np.full((8, 8, 3), 100, dtype=np.uint8)
        np.testing.assert_array_equal(noise_perturbation(image), noise_perturbation(image))


if __name__ == "__main__":
    unittest.main()
