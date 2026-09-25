"""Fail-closed configuration checks; no original files are written."""

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from config import ARTIFACT_ENV, ConfigurationError, Settings  # noqa: E402


class RuntimeConfiguration(unittest.TestCase):
    def test_requires_external_assets(self):
        empty = {key: "" for key in ("SAAD_ARTIFACT_DIR", *ARTIFACT_ENV.values())}
        with patch.dict(os.environ, empty):
            with self.assertRaisesRegex(ConfigurationError, "Set SAAD_ARTIFACT_DIR"):
                Settings.from_environment()

    def test_rejects_offline_policy(self):
        manifest = json.loads((ROOT / "config/model_manifest.json").read_text())
        calibration = json.loads((ROOT / "config/live_api_v1.json").read_text())
        manifest["default_policy_id"] = "saad-offline-evidence-v3"
        with patch.dict(os.environ, {"SAAD_ARTIFACT_DIR": str(ROOT)}):
            with patch("config._json", side_effect=[manifest, calibration]):
                with self.assertRaisesRegex(ConfigurationError, "deployed policy"):
                    Settings.from_environment()

    def test_missing_asset_fails_before_model_load(self):
        with patch.dict(os.environ, {"SAAD_ARTIFACT_DIR": str(ROOT)}):
            settings = Settings.from_environment()
        with self.assertRaisesRegex(ConfigurationError, "Missing yolo artifact"):
            settings.validate_assets()

    def test_hash_mismatch_is_rejected(self):
        if not os.environ.get("SAAD_ARTIFACT_DIR"):
            self.skipTest("external assets unavailable")
        settings = Settings.from_environment()
        altered = copy.deepcopy(settings.manifest)
        altered["artifacts"]["yolo"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ConfigurationError, "SHA-256 mismatch for yolo"):
            replace(settings, manifest=altered).validate_assets()


if __name__ == "__main__":
    unittest.main()
