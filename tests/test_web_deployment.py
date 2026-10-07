"""Regression tests for the static deployment checker.

Run from the repository root:
    python3 -m unittest discover -s tests -v
"""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
CHECKER_PATH = REPOSITORY_ROOT / "src" / "check_web_deployment.py"

SPEC = importlib.util.spec_from_file_location(
    "masklab_web_checker",
    CHECKER_PATH,
)
CHECKER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER_MODULE)

DeploymentChecker = CHECKER_MODULE.DeploymentChecker


class DeploymentCheckerTests(unittest.TestCase):
    """Exercise valid deployments and common packaging failures."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)

        self.workspace = Path(self.temporary_directory.name)
        self.web_root = self.workspace / "web"
        self.web_root.mkdir()

        for asset in CHECKER_MODULE.REQUIRED_ASSETS:
            self.write_file(asset, b"example static asset")

        for manifest in CHECKER_MODULE.MODEL_MANIFESTS:
            self.write_model(manifest)

    def write_file(self, relative_path, content):
        path = self.web_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def write_model(self, relative_path):
        manifest_path = self.web_root / relative_path
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        manifest = {
            "modelTopology": {"example": "fixture topology"},
            "weightsManifest": [
                {
                    "paths": ["weights.bin"],
                    "weights": [
                        {
                            "name": "example_weight",
                            "shape": [1],
                            "dtype": "float32",
                        }
                    ],
                }
            ],
        }

        manifest_path.write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        (manifest_path.parent / "weights.bin").write_bytes(
            bytes([0, 0, 0, 0])
        )

    def edit_manifest(self, relative_path, change):
        path = self.web_root / relative_path
        manifest = json.loads(path.read_text(encoding="utf-8"))
        change(manifest)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def run_checker(self, web_root=None):
        checker = DeploymentChecker(
            self.web_root if web_root is None else web_root
        )
        captured_output = io.StringIO()

        with contextlib.redirect_stdout(captured_output):
            exit_code = checker.run()

        return checker, exit_code, captured_output.getvalue()

    def assert_failure(self, expected_message):
        checker, exit_code, output = self.run_checker()
        self.assertEqual(exit_code, 1)
        self.assertTrue(checker.errors)
        self.assertIn(expected_message, output)
        return checker

    def test_complete_static_bundle_passes(self):
        checker, exit_code, output = self.run_checker()

        self.assertEqual(exit_code, 0)
        self.assertEqual(checker.errors, [])
        self.assertIn("Static asset checks passed.", output)

    def test_missing_website_directory_fails(self):
        missing_directory = self.workspace / "missing"
        checker, exit_code, output = self.run_checker(missing_directory)

        self.assertEqual(exit_code, 1)
        self.assertTrue(checker.errors)
        self.assertIn("Website directory not found", output)

    def test_file_cannot_be_used_as_website_directory(self):
        regular_file = self.workspace / "regular-file"
        regular_file.write_text("not a directory", encoding="utf-8")
        _, exit_code, output = self.run_checker(regular_file)

        self.assertEqual(exit_code, 1)
        self.assertIn("Website directory not found", output)

    def test_each_required_asset_is_checked(self):
        for asset in CHECKER_MODULE.REQUIRED_ASSETS:
            with self.subTest(asset=asset):
                path = self.web_root / asset
                original = path.read_bytes()
                path.unlink()
                try:
                    self.assert_failure("Missing file")
                finally:
                    path.write_bytes(original)

    def test_empty_javascript_file_fails(self):
        (self.web_root / "app.js").write_bytes(b"")
        self.assert_failure("Empty file")

    def test_directory_instead_of_asset_fails(self):
        path = self.web_root / "style.css"
        path.unlink()
        path.mkdir()
        self.assert_failure("Missing file")

    def test_missing_classifier_manifest_fails(self):
        (self.web_root / "model" / "model.json").unlink()
        self.assert_failure("Missing file")

    def test_missing_face_detector_manifest_fails(self):
        (self.web_root / "face-model" / "model.json").unlink()
        self.assert_failure("Missing file")

    def test_empty_manifest_fails(self):
        (self.web_root / "model" / "model.json").write_bytes(b"")
        self.assert_failure("Empty file")

    def test_malformed_json_fails(self):
        self.write_file("model/model.json", b"{unfinished")
        self.assert_failure("Cannot parse model/model.json")

    def test_non_utf8_manifest_fails(self):
        self.write_file("model/model.json", bytes([255, 254, 255]))
        self.assert_failure("Cannot parse model/model.json")

    def test_json_array_is_rejected(self):
        self.write_file("model/model.json", b"[]")
        self.assert_failure("expected a JSON object")

    def test_missing_model_topology_fails(self):
        self.edit_manifest(
            "model/model.json",
            lambda manifest: manifest.pop("modelTopology"),
        )
        self.assert_failure("modelTopology is missing or invalid")

    def test_empty_model_topology_fails(self):
        self.edit_manifest(
            "model/model.json",
            lambda manifest: manifest.update(modelTopology={}),
        )
        self.assert_failure("modelTopology is missing or invalid")

    def test_missing_weights_manifest_fails(self):
        self.edit_manifest(
            "model/model.json",
            lambda manifest: manifest.pop("weightsManifest"),
        )
        self.assert_failure("weightsManifest is missing or invalid")

    def test_empty_weight_groups_fail(self):
        self.edit_manifest(
            "model/model.json",
            lambda manifest: manifest.update(weightsManifest=[]),
        )
        self.assert_failure("weightsManifest is missing or invalid")

    def test_non_object_weight_group_fails(self):
        self.edit_manifest(
            "model/model.json",
            lambda manifest: manifest.update(weightsManifest=["invalid"]),
        )
        self.assert_failure("expected an object")

    def test_empty_weight_paths_fail(self):
        self.edit_manifest(
            "model/model.json",
            lambda manifest: manifest["weightsManifest"][0].update(
                paths=[]
            ),
        )
        self.assert_failure("no weight paths declared")

    def test_invalid_weight_filename_fails(self):
        self.edit_manifest(
            "model/model.json",
            lambda manifest: manifest["weightsManifest"][0].update(
                paths=[None]
            ),
        )
        self.assert_failure("invalid weight filename")

    def test_missing_weight_specifications_fail(self):
        self.edit_manifest(
            "model/model.json",
            lambda manifest: manifest["weightsManifest"][0].pop(
                "weights"
            ),
        )
        self.assert_failure("no weight specifications declared")

    def test_missing_weight_shard_fails(self):
        (self.web_root / "model" / "weights.bin").unlink()
        self.assert_failure("Missing file")

    def test_empty_weight_shard_fails(self):
        (self.web_root / "model" / "weights.bin").write_bytes(b"")
        self.assert_failure("Empty file")

    def test_multiple_shards_are_all_required(self):
        self.edit_manifest(
            "model/model.json",
            lambda manifest: manifest["weightsManifest"][0].update(
                paths=["weights.bin", "second.bin"]
            ),
        )
        self.assert_failure("second.bin")

        self.write_file("model/second.bin", bytes([1, 2, 3, 4]))
        _, exit_code, _ = self.run_checker()
        self.assertEqual(exit_code, 0)

    def test_git_lfs_pointer_is_rejected(self):
        self.write_file(
            "model/weights.bin",
            b"version https://git-lfs.github.com/spec/v1",
        )
        self.assert_failure("Git LFS pointer found")

    def test_parent_traversal_outside_web_root_fails(self):
        outside_file = self.workspace / "outside.bin"
        outside_file.write_bytes(b"outside website")

        self.edit_manifest(
            "model/model.json",
            lambda manifest: manifest["weightsManifest"][0].update(
                paths=["../../outside.bin"]
            ),
        )
        self.assert_failure("weight path leaves website directory")

    def test_absolute_path_outside_web_root_fails(self):
        outside_file = self.workspace / "outside.bin"
        outside_file.write_bytes(b"outside website")

        self.edit_manifest(
            "model/model.json",
            lambda manifest: manifest["weightsManifest"][0].update(
                paths=[str(outside_file)]
            ),
        )
        self.assert_failure("weight path leaves website directory")

    def test_nested_weight_path_inside_website_passes(self):
        self.write_file("model/nested/weights.bin", b"nested weights")
        self.edit_manifest(
            "model/model.json",
            lambda manifest: manifest["weightsManifest"][0].update(
                paths=["nested/weights.bin"]
            ),
        )

        _, exit_code, _ = self.run_checker()
        self.assertEqual(exit_code, 0)

    def test_reports_multiple_missing_assets(self):
        (self.web_root / "index.html").unlink()
        (self.web_root / "app.js").unlink()

        checker, exit_code, output = self.run_checker()

        self.assertEqual(exit_code, 1)
        self.assertGreaterEqual(len(checker.errors), 2)
        self.assertIn("index.html", output)
        self.assertIn("app.js", output)

    def test_checker_does_not_modify_files(self):
        before = {
            path.relative_to(self.web_root): path.read_bytes()
            for path in self.web_root.rglob("*")
            if path.is_file()
        }

        self.run_checker()

        after = {
            path.relative_to(self.web_root): path.read_bytes()
            for path in self.web_root.rglob("*")
            if path.is_file()
        }

        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()