"""Exact state-key and tensor-shape compatibility with frozen originals."""

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from config import Settings  # noqa: E402
from saad_inference.vae import ConvVAE, _extract_state_dict as vae_state  # noqa: E402
from saad_inference.realnvp import RealNVP, _extract_state_dict as flow_state  # noqa: E402


class FrozenCheckpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = Settings.from_environment()
        cls.settings.validate_assets()

    def check_state(self, role, architecture, extractor, count):
        checkpoint = torch.load(
            self.settings.artifact_paths[role], map_location="cpu", weights_only=False
        )
        saved = {
            key.removeprefix("module."): value
            for key, value in extractor(checkpoint).items()
        }
        expected = architecture().state_dict()
        self.assertEqual(len(saved), count)
        self.assertEqual(set(saved), set(expected))
        for key in expected:
            self.assertEqual(tuple(saved[key].shape), tuple(expected[key].shape), key)
        architecture().load_state_dict(saved, strict=True)

    def test_vae_state(self):
        self.check_state("vae", ConvVAE, vae_state, 71)

    def test_realnvp_state(self):
        self.check_state("flow", RealNVP, flow_state, 56)

    def test_latent_normalization_shapes(self):
        for role in ("latent_mean", "latent_std"):
            tensor = torch.load(
                self.settings.artifact_paths[role], map_location="cpu", weights_only=False
            )
            self.assertEqual(torch.as_tensor(tensor).numel(), 128, role)


if __name__ == "__main__":
    unittest.main()
