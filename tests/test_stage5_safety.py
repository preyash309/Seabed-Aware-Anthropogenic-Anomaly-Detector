"""The historical dataset builders cannot write into the original tree."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml" / "Dataset"))
from safety import SAFE_OUTPUT_ROOT, require_new_output  # noqa: E402


class DatasetSafety(unittest.TestCase):
    def test_requires_explicit_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                require_new_output(ROOT, SAFE_OUTPUT_ROOT / "new_set")

    def test_rejects_original_output_even_with_opt_in(self):
        with patch.dict(os.environ, {"SAAD_ALLOW_DATASET_BUILD": "YES"}):
            with self.assertRaisesRegex(RuntimeError, "child"):
                require_new_output(ROOT, Path(r"E:\SIH\Datasets\SAAD_baseline"))

    def test_rejects_existing_output_even_with_opt_in(self):
        with patch.dict(os.environ, {"SAAD_ALLOW_DATASET_BUILD": "YES"}):
            with patch.object(Path, "exists", return_value=True):
                with self.assertRaises(FileExistsError):
                    require_new_output(ROOT, SAFE_OUTPUT_ROOT / "new_set")


if __name__ == "__main__":
    unittest.main()
