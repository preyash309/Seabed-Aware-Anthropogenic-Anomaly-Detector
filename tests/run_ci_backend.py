"""Run the 13 original backend tests that need no private tensors or images."""

from pathlib import Path
import os
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
# The live policy reads calibration through Settings on import, which requires
# path declarations but never opens model files in this selected test slice.
os.environ["SAAD_ARTIFACT_DIR"] = str(ROOT)
os.environ["SAAD_DEVICE"] = "cpu"

TESTS = (
    "test_stage1_config.RuntimeConfiguration.test_requires_external_assets",
    "test_stage1_config.RuntimeConfiguration.test_rejects_offline_policy",
    "test_stage1_config.RuntimeConfiguration.test_missing_asset_fails_before_model_load",
    "test_stage2_modules.ExtractedModules",
    "test_stage2_policy.LivePolicyBoundaries",
    "test_stage5_safety.DatasetSafety",
)

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(loader.loadTestsFromName(name) for name in TESTS)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.testsRun != 13 or result.skipped:
        sys.exit(1)
