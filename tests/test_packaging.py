from __future__ import annotations

import unittest
import tomllib
from pathlib import Path

import sensenova_mlx


class PackagingTest(unittest.TestCase):
    def test_runtime_version_matches_project_metadata(self) -> None:
        root = Path(__file__).parents[1]
        config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(sensenova_mlx.__version__, config["project"]["version"])

    def test_declared_publication_files_and_download_manifest_exist(self) -> None:
        root = Path(__file__).parents[1]
        required = (
            "README.md",
            "README.zh-CN.md",
            "LICENSE",
            "NOTICE",
            "SECURITY.md",
            "CONTRIBUTING.md",
            "CITATION.cff",
            "docs/download-manifest.json",
            "docs/assets/showcase.md",
        )

        missing = [relative for relative in required if not (root / relative).is_file()]

        self.assertEqual(missing, [])

    def test_source_distribution_excludes_local_and_generated_state(self) -> None:
        root = Path(__file__).parents[1]
        config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        excludes = config["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]

        for excluded in (
            "/artifacts",
            "/dist",
            "/.venv",
            "/PROGRESS.md",
            "/release-assets",
            "/SPEC.md",
        ):
            with self.subTest(excluded=excluded):
                self.assertIn(excluded, excludes)

    def test_public_source_tools_and_docs_have_no_personal_absolute_paths(self) -> None:
        root = Path(__file__).parents[1]
        public_files = [
            root / "README.md",
            *sorted((root / "src").rglob("*.py")),
            *sorted((root / "tools").rglob("*.py")),
            *sorted((root / "docs").rglob("*.md")),
        ]
        public_files.extend(path for path in (root / "README.zh-CN.md",) if path.exists())
        forbidden = ("/" + "Users/", "/" + "private/tmp", "local" + " agent")
        offenders = []
        for path in public_files:
            text = path.read_text(encoding="utf-8")
            if any(marker in text for marker in forbidden):
                offenders.append(str(path.relative_to(root)))

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
