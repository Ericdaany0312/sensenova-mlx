from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.release_audit import audit_archive


class ReleaseAuditTest(unittest.TestCase):
    def test_clean_wheel_has_no_findings(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sensenova-release-audit-") as raw:
            archive = Path(raw) / "clean.whl"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("sensenova_mlx/__init__.py", "__version__ = '0.2.0'\n")

            self.assertEqual(audit_archive(archive), [])

    def test_sdist_rejects_model_weights(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sensenova-release-audit-") as raw:
            archive = Path(raw) / "bad.tar.gz"
            payload = b"not-a-real-model"
            with tarfile.open(archive, "w:gz") as handle:
                member = tarfile.TarInfo("sensenova-mlx-0.2.0/models/model.safetensors")
                member.size = len(payload)
                handle.addfile(member, io.BytesIO(payload))

            findings = audit_archive(archive)

            self.assertTrue(any("model.safetensors" in finding for finding in findings))

    def test_archive_rejects_personal_absolute_paths_in_text(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sensenova-release-audit-") as raw:
            archive = Path(raw) / "bad.whl"
            personal_path = "/" + "Users/example/private/model"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("sensenova_mlx/config.py", f'MODEL = "{personal_path}"\n')

            findings = audit_archive(archive)

            self.assertTrue(any("personal absolute path" in finding for finding in findings))

    def test_archive_rejects_oversized_text_members(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sensenova-release-audit-") as raw:
            archive = Path(raw) / "bad.whl"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
                handle.writestr("sensenova_mlx/generated.py", b"x" * (2 * 1024 * 1024 + 1))

            findings = audit_archive(archive)

            self.assertTrue(any("oversized text member" in finding for finding in findings))

    def test_archive_rejects_links(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sensenova-release-audit-") as raw:
            archive = Path(raw) / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                member = tarfile.TarInfo("sensenova-mlx-0.2.0/docs/latest")
                member.type = tarfile.SYMTYPE
                member.linkname = "../private"
                handle.addfile(member)

            findings = audit_archive(archive)

            self.assertTrue(any("link member" in finding for finding in findings))

    def test_archive_rejects_parent_path_segments(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sensenova-release-audit-") as raw:
            archive = Path(raw) / "bad.whl"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../outside.py", "SAFE = False\n")

            findings = audit_archive(archive)

            self.assertTrue(any("unsafe path" in finding for finding in findings))


if __name__ == "__main__":
    unittest.main()
