from __future__ import annotations

import json
import os
import re
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable
from urllib.parse import quote, urlparse

from . import __version__
from .weights import FileRecord, verify_file


USER_AGENT = f"sensenova-mlx/{__version__}"


@dataclass(frozen=True)
class DownloadRecord:
    relative_path: str
    size: int
    sha256: str
    url: str


def _safe_manifest_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ValueError(f"unsafe manifest path: {value!r}")
    return value


def _reject_symlinks(root: Path, path: Path) -> None:
    if root.is_symlink():
        raise ValueError(f"refusing symbolic link download root: {root}")
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"refusing symbolic link in download path: {current}")


def load_download_manifest(path: Path) -> tuple[DownloadRecord, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError(f"unsupported download manifest schema in {path}")
    records: list[DownloadRecord] = []
    for repository in raw["repositories"]:
        base = repository["resolve_base"].rstrip("/")
        revision = repository["revision"]
        if not base.endswith(f"/resolve/{revision}"):
            raise ValueError(f"resolve_base is not pinned to declared revision: {base}")
        parsed_base = urlparse(base)
        if parsed_base.scheme != "https" or parsed_base.netloc != "www.modelscope.cn":
            raise ValueError(f"resolve_base must use official ModelScope HTTPS: {base}")
        raw_prefix = repository.get("local_prefix", "")
        local_prefix = _safe_manifest_path(raw_prefix) if raw_prefix else ""
        for item in repository["files"]:
            name = _safe_manifest_path(item["name"])
            relative = f"{local_prefix}/{name}" if local_prefix else name
            size = int(item["size"])
            sha256 = str(item["sha256"])
            if size <= 0:
                raise ValueError(f"manifest size must be positive for {relative}: {size}")
            if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
                raise ValueError(f"invalid SHA-256 for {relative}: {sha256!r}")
            records.append(
                DownloadRecord(
                    relative_path=relative,
                    size=size,
                    sha256=sha256,
                    url=f"{base}/{quote(name, safe='/')}",
                )
            )
    paths = [record.relative_path for record in records]
    if len(paths) != len(set(paths)):
        raise ValueError("download manifest has duplicate local paths")
    declared_total = int(raw["total_bytes"])
    actual_total = sum(record.size for record in records)
    if declared_total != actual_total:
        raise ValueError(
            f"download manifest total mismatch: declared {declared_total}, actual {actual_total}"
        )
    return tuple(records)


def remaining_download_bytes(target: Path, records: Iterable[DownloadRecord]) -> int:
    remaining = 0
    for record in records:
        destination = target / record.relative_path
        partial = destination.with_name(f"{destination.name}.partial")
        _reject_symlinks(target, destination)
        _reject_symlinks(target, partial)
        if destination.exists():
            verify_file(
                destination,
                FileRecord(record.relative_path, record.size, record.sha256),
            )
            continue
        partial_size = partial.stat().st_size if partial.exists() else 0
        if partial_size > record.size:
            raise ValueError(
                f"{partial}: partial size {partial_size} exceeds expected {record.size}"
            )
        remaining += record.size - partial_size
    return remaining


def _download_one(record: DownloadRecord, target: Path, *, chunk_size: int) -> None:
    destination = target / record.relative_path
    _reject_symlinks(target, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_file(destination, FileRecord(record.relative_path, record.size, record.sha256))
        print(f"[verified] {record.relative_path}", flush=True)
        return
    partial = destination.with_name(f"{destination.name}.partial")
    _reject_symlinks(target, partial)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > record.size:
        raise ValueError(f"{partial}: partial size {offset} exceeds expected {record.size}")
    if offset == record.size:
        verify_file(partial, FileRecord(record.relative_path, record.size, record.sha256))
        partial.replace(destination)
        print(f"[verified] {record.relative_path}", flush=True)
        return

    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(record.url, headers=headers)
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=120) as response:
        if urlparse(response.geturl()).scheme != "https":
            raise RuntimeError(f"download redirected away from HTTPS: {response.geturl()}")
        status = getattr(response, "status", response.getcode())
        if offset and status != 206:
            raise RuntimeError(
                f"server did not honor Range for {record.relative_path}: HTTP {status}; "
                f"kept {partial} for inspection"
            )
        if offset:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                raise RuntimeError(
                    f"server returned wrong Content-Range for {record.relative_path}: {content_range!r}"
                )
        mode = "ab" if offset else "wb"
        downloaded = offset
        next_report = max(downloaded + 512 * 1024 * 1024, 512 * 1024 * 1024)
        with partial.open(mode) as handle:
            while True:
                block = response.read(chunk_size)
                if not block:
                    break
                handle.write(block)
                downloaded += len(block)
                if downloaded >= next_report:
                    percent = 100.0 * downloaded / record.size
                    print(f"[download] {record.relative_path} {percent:.1f}%", flush=True)
                    next_report += 512 * 1024 * 1024
            handle.flush()
            os.fsync(handle.fileno())
    if downloaded != record.size:
        raise IOError(
            f"{record.relative_path}: connection ended at {downloaded}/{record.size} bytes; resume is safe"
        )
    verify_file(partial, FileRecord(record.relative_path, record.size, record.sha256))
    partial.replace(destination)
    elapsed = time.monotonic() - started
    print(f"[verified] {record.relative_path} ({elapsed:.1f}s)", flush=True)


def download_all(
    manifest_path: Path,
    target: Path,
    *,
    headroom_bytes: int = 10 * 1024**3,
    chunk_size: int = 8 * 1024 * 1024,
) -> None:
    records = load_download_manifest(manifest_path)
    if target.is_symlink():
        raise ValueError(f"refusing symbolic link download root: {target}")
    target.mkdir(parents=True, exist_ok=True)
    remaining = remaining_download_bytes(target, records)
    free = shutil.disk_usage(target).free
    if free < remaining + headroom_bytes:
        raise OSError(
            f"insufficient free space: need {remaining + headroom_bytes} bytes including headroom, "
            f"have {free}"
        )
    print(f"[plan] {len(records)} files, {remaining} bytes remaining, target={target}", flush=True)
    for record in records:
        _download_one(record, target, chunk_size=chunk_size)
