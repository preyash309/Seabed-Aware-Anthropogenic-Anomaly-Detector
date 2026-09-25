"""Boundary behavior of the deployed live API evidence policy."""

import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from saad_inference.evidence.live_api_v1 import (  # noqa: E402
    FLOW_P5, FLOW_P95, VAE_P5, VAE_P95, YOLO_P5, YOLO_P95,
    build_evidence, percentile_score,
)


class LivePolicyBoundaries(unittest.TestCase):
    def test_missing_scores_and_missing_tta(self):
        result = build_evidence(YOLO_P5, None, None, None)
        self.assertEqual(result["normalized"], {"yolo": 0.0, "vae": 0.0, "flow": 0.0})
        self.assertEqual(result["ttaStability"], 0.5)
        self.assertAlmostEqual(result["priority"], 0.05)
        self.assertAlmostEqual(result["uncertainty"], 0.15)
        self.assertEqual(result["evidenceProfile"], "WEAK_EVIDENCE")
        self.assertEqual(result["recommendation"], "LOW_PRIORITY_REVIEW")

    def test_non_finite_raw_scores_map_to_zero(self):
        result = build_evidence(math.nan, math.inf, -math.inf, None)
        self.assertEqual(result["normalized"], {"yolo": 0.0, "vae": 0.0, "flow": 0.0})
        self.assertEqual(result["priority"], 0.05)
        self.assertEqual(percentile_score("not a number", 0.0, 1.0), 0.0)
        self.assertEqual(percentile_score(0.5, 1.0, 1.0), 0.0)

    def test_normalization_and_corroboration_threshold_ties(self):
        raw = [a + (b - a) * 0.5 for a, b in (
            (YOLO_P5, YOLO_P95), (VAE_P5, VAE_P95), (FLOW_P5, FLOW_P95)
        )]
        result = build_evidence(*raw, 0.5)
        for value in result["normalized"].values():
            self.assertAlmostEqual(value, 0.5, delta=1e-15)
        self.assertEqual(result["corroboration"], 1.0)
        self.assertAlmostEqual(result["priority"], 0.55)
        self.assertEqual(result["evidenceProfile"], "YOLO_VAE_FLOW_CORROBORATED")
        self.assertEqual(result["priorityLevel"], "MEDIUM")
        self.assertEqual(result["recommendation"], "REVIEW")

    def test_high_priority_and_uncertainty(self):
        corroborated = build_evidence(YOLO_P95, VAE_P95, FLOW_P95, 1.0)
        self.assertEqual(corroborated["priority"], 1.0)
        self.assertEqual(corroborated["uncertainty"], 0.0)
        self.assertEqual(corroborated["recommendation"], "HIGH_PRIORITY_REVIEW")

        uncertain = build_evidence(YOLO_P95, VAE_P5, FLOW_P5, 0.0)
        self.assertAlmostEqual(uncertain["priority"], 0.433, delta=1e-12)
        self.assertAlmostEqual(uncertain["uncertainty"], 0.7949747174978257, delta=1e-12)
        self.assertEqual(uncertain["priorityLevel"], "HIGH")
        self.assertEqual(uncertain["recommendation"], "HIGH_PRIORITY_UNCERTAIN")


if __name__ == "__main__":
    unittest.main()
