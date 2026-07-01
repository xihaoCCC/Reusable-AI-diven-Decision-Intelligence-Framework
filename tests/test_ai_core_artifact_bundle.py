from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "src"
    / "ai_core"
    / "artifacts"
    / "exploitation_type"
    / "ctdc_xgboost"
    / "v0.1.0"
)


class AICoreArtifactBundleTests(unittest.TestCase):
    def test_minimal_release_contract_is_complete(self) -> None:
        manifest = json.loads(
            (ARTIFACT / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        feature_config = yaml.safe_load(
            (ARTIFACT / "feature_config.yaml").read_text(encoding="utf-8")
        )

        self.assertEqual(manifest["artifact_status"], "released")
        self.assertEqual(manifest["release_channel"], "research_preview")
        self.assertEqual(manifest["version"], "0.1.0")
        self.assertEqual(feature_config["task_name"], "ctdc_exploitation_type_classification")
        for file_name in manifest["files"].values():
            self.assertTrue((ARTIFACT / file_name).is_file(), file_name)

    def test_release_excludes_experimental_outputs(self) -> None:
        names = {path.name for path in ARTIFACT.iterdir() if path.is_file()}
        excluded_fragments = {
            "logistic",
            "importance",
            "threshold",
            "confusion",
            "uncalibrated",
        }

        for fragment in excluded_fragments:
            self.assertFalse(any(fragment in name for name in names), fragment)
        self.assertFalse(any(name.endswith(".png") for name in names))

    def test_release_checksums_match(self) -> None:
        failures = []
        checksum_path = ARTIFACT / "checksums.sha256"
        for line in checksum_path.read_text(encoding="ascii").splitlines():
            expected, file_name = line.split("  ", 1)
            path = ARTIFACT / file_name
            if not path.is_file():
                failures.append(file_name)
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                failures.append(file_name)

        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
