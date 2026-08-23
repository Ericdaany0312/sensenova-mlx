from __future__ import annotations

import argparse
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


BLOCKED_DIRECTORIES = frozenset({".venv", "artifacts", "dist", "model", "models", "__pycache__"})
BLOCKED_SUFFIXES = frozenset({".bin", ".ckpt", ".gguf", ".key", ".pem", ".pt", ".pth", ".safetensors"})
TEXT_SUFFIXES = frozenset({".cff", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"})
PERSONAL_MARKERS = ("/" + "Users/", "/" + "private/tmp", "local" + " agent")
MAX_TEXT_BYTES = 2 * 1024 * 1024


def _is_text(name: str) -> bool:
    path = PurePosixPath(name)
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE", "NOTICE"}


def _archive_members(path: Path) -> Iterable[tuple[str, int, bytes | None, str]]:
    if path.suffix == ".whl" or zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                kind = "link" if stat.S_ISLNK(member.external_attr >> 16) else "file"
                payload = (
                    archive.read(member)
                    if kind == "file" and _is_text(member.filename) and member.file_size <= MAX_TEXT_BYTES
                    else None
                )
                yield member.filename, member.file_size, payload, kind
        return
    if path.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isdir():
                    continue
                kind = "file" if member.isfile() else "link" if member.issym() or member.islnk() else "special"
                payload = None
                if kind == "file" and _is_text(member.name) and member.size <= MAX_TEXT_BYTES:
                    extracted = archive.extractfile(member)
                    if extracted is not None:
                        payload = extracted.read()
                yield member.name, member.size, payload, kind
        return
    raise ValueError(f"unsupported release archive: {path}")


def audit_archive(path: Path) -> list[str]:
    findings: list[str] = []
    for name, size, payload, kind in _archive_members(path):
        normalized = PurePosixPath(name)
        if normalized.is_absolute() or ".." in normalized.parts:
            findings.append(f"{path.name}: unsafe path in {name}")
        if kind == "link":
            findings.append(f"{path.name}: link member in {name}")
        elif kind != "file":
            findings.append(f"{path.name}: special member in {name}")
        lower_parts = {part.lower() for part in normalized.parts}
        if lower_parts & BLOCKED_DIRECTORIES:
            findings.append(f"{path.name}: blocked directory in {name}")
        if normalized.suffix.lower() in BLOCKED_SUFFIXES:
            findings.append(f"{path.name}: blocked file type in {name}")
        if normalized.name.startswith(".env") or normalized.name.startswith(("id_rsa", "id_ed25519")):
            findings.append(f"{path.name}: credential-like filename in {name}")
        if kind != "file" or not _is_text(name):
            continue
        if size > MAX_TEXT_BYTES:
            findings.append(f"{path.name}: oversized text member in {name}")
            continue
        if payload is None:
            findings.append(f"{path.name}: unreadable text member in {name}")
            continue
        text = payload.decode("utf-8", errors="replace")
        if any(marker in text for marker in PERSONAL_MARKERS):
            findings.append(f"{path.name}: personal absolute path in {name}")
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit wheel and sdist contents before release")
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args(argv)
    findings = [finding for archive in args.archives for finding in audit_archive(archive)]
    if findings:
        for finding in findings:
            print(f"[blocked] {finding}")
        return 1
    for archive in args.archives:
        print(f"[clean] {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
