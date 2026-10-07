"""Check the static website's assets before deployment.

This checks file presence and model manifests.
It does not test camera access or prediction accuracy.
"""

import argparse
import json
import sys
from pathlib import Path


REQUIRED_ASSETS = (
    "index.html",
    "style.css",
    "app.js",
    "favicon.svg",
    "vendor/tf.min.js",
    "vendor/blazeface.min.js",
)

MODEL_MANIFESTS = (
    "model/model.json",
    "face-model/model.json",
)


class DeploymentChecker:
    """Collect deployment problems without modifying files."""

    def __init__(self, web_root):
        self.web_root = web_root.resolve()
        self.errors = []
        self.checked_files = set()

    def fail(self, message):
        self.errors.append(message)
        print(f"[FAIL] {message}")

    def check_file(self, path):
        """Require a readable, nonempty file."""
        if path in self.checked_files:
            return True

        try:
            if not path.is_file():
                self.fail(f"Missing file: {path}")
                return False

            if path.stat().st_size == 0:
                self.fail(f"Empty file: {path}")
                return False

            with path.open("rb") as stream:
                prefix = stream.read(100)

            if prefix.startswith(
                b"version https://git-lfs.github.com/spec/v1"
            ):
                self.fail(
                    f"Git LFS pointer found instead of actual content: {path}"
                )
                return False

        except OSError as error:
            self.fail(f"Cannot read {path}: {error}")
            return False

        self.checked_files.add(path)
        print(f"[PASS] {path.relative_to(self.web_root)}")
        return True

    def check_model(self, relative_path):
        """Check JSON structure and every declared weight shard."""
        manifest_path = self.web_root / relative_path

        if not self.check_file(manifest_path):
            return

        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValueError) as error:
            self.fail(f"Cannot parse {relative_path}: {error}")
            return

        if not isinstance(manifest, dict):
            self.fail(f"{relative_path}: expected a JSON object")
            return

        topology = manifest.get("modelTopology")
        if not isinstance(topology, dict) or not topology:
            self.fail(f"{relative_path}: modelTopology is missing or invalid")

        groups = manifest.get("weightsManifest")
        if not isinstance(groups, list) or not groups:
            self.fail(
                f"{relative_path}: weightsManifest is missing or invalid"
            )
            return

        for group_number, group in enumerate(groups, start=1):
            label = f"{relative_path}, weight group {group_number}"

            if not isinstance(group, dict):
                self.fail(f"{label}: expected an object")
                continue

            paths = group.get("paths")
            if not isinstance(paths, list) or not paths:
                self.fail(f"{label}: no weight paths declared")
                continue

            weights = group.get("weights")
            if not isinstance(weights, list) or not weights:
                self.fail(f"{label}: no weight specifications declared")

            for shard_name in paths:
                if not isinstance(shard_name, str) or not shard_name:
                    self.fail(f"{label}: invalid weight filename")
                    continue

                shard_path = (
                    manifest_path.parent / shard_name
                ).resolve()

                try:
                    shard_path.relative_to(self.web_root)
                except ValueError:
                    self.fail(
                        f"{label}: weight path leaves website directory: "
                        f"{shard_name}"
                    )
                    continue

                self.check_file(shard_path)

    def run(self):
        if not self.web_root.is_dir():
            self.fail(f"Website directory not found: {self.web_root}")
            return 1

        print(f"Checking website: {self.web_root}")
        print()

        for asset in REQUIRED_ASSETS:
            self.check_file(self.web_root / asset)

        for manifest in MODEL_MANIFESTS:
            self.check_model(manifest)

        print()
        if self.errors:
            print(f"Deployment checks failed: {len(self.errors)} problem(s).")
            print("Fix the reported problems before hosting.")
            return 1

        print(f"Checked {len(self.checked_files)} nonempty files.")
        print("Static asset checks passed.")
        print("Next: test model loading and camera predictions in a browser.")
        return 0


def main():
    repository_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Check MaskLab website assets before deployment."
    )
    parser.add_argument(
        "--web-root",
        type=Path,
        default=repository_root / "web",
        help="Website directory; defaults to this repository's web folder.",
    )
    arguments = parser.parse_args()

    checker = DeploymentChecker(arguments.web_root)
    return checker.run()


if __name__ == "__main__":
    sys.exit(main())