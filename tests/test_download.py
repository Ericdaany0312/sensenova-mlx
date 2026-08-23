from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sensenova_mlx
from sensenova_mlx.download import USER_AGENT, DownloadRecord, load_download_manifest, remaining_download_bytes


MANIFEST = Path(__file__).parents[1] / "docs" / "download-manifest.json"


class DownloadManifestTest(unittest.TestCase):
    def test_downloader_user_agent_matches_runtime_version(self) -> None:
        self.assertEqual(USER_AGENT, f"sensenova-mlx/{sensenova_mlx.__version__}")

    def test_manifest_rejects_path_traversal_and_wrong_declared_total(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sensenova-manifest-") as raw:
            path = Path(raw) / "manifest.json"
            manifest = {
                "schema_version": 1,
                "total_bytes": 10,
                "repositories": [
                    {
                        "revision": "abc",
                        "resolve_base": "https://www.modelscope.cn/models/x/y/resolve/abc",
                        "files": [
                            {"name": "../escape", "size": 9, "sha256": "0" * 64},
                        ],
                    }
                ],
            }
            path.write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ValueError, "unsafe manifest path"):
                load_download_manifest(path)

            manifest["repositories"][0]["files"][0]["name"] = "safe.bin"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "total mismatch"):
                load_download_manifest(path)

    def test_partial_symlink_is_never_followed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sensenova-symlink-") as raw:
            root = Path(raw)
            record = DownloadRecord(
                relative_path="model.bin",
                size=10,
                sha256="0" * 64,
                url="https://www.modelscope.cn/models/x/y/resolve/revision/model.bin",
            )
            (root / "model.bin.partial").symlink_to(root / "outside.bin")

            with self.assertRaisesRegex(ValueError, "symbolic link"):
                remaining_download_bytes(root, [record])

    def test_formal_manifest_is_revision_pinned_and_byte_exact(self) -> None:
        records = load_download_manifest(MANIFEST)

        self.assertEqual(sum(record.size for record in records), 51_042_331_251)
        self.assertEqual(len(records), 28)
        self.assertTrue(all("/resolve/" in record.url for record in records))
        self.assertTrue(all("/master/" not in record.url and "/main/" not in record.url for record in records))

    def test_partial_file_reduces_remaining_bytes_without_touching_it(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sensenova-download-") as raw:
            root = Path(raw)
            record = DownloadRecord(
                relative_path="model.bin",
                size=10,
                sha256="0" * 64,
                url="https://example.invalid/resolve/revision/model.bin",
            )
            partial = root / "model.bin.partial"
            partial.write_bytes(b"abc")

            self.assertEqual(remaining_download_bytes(root, [record]), 7)
            self.assertEqual(partial.read_bytes(), b"abc")


if __name__ == "__main__":
    unittest.main()
