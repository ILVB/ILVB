"""Safe CBZ/ZIP reading (in memory, zip-slip-proof, size-capped) and CBZ writing."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from manga_ar.errors import ArchiveError
from manga_ar.io.naming import is_image_name, long_path, natural_key
from manga_ar.io.writer import atomic_write_bytes
from manga_ar.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class ArchiveLimits:
    max_members: int = 5000
    max_member_bytes: int = 200 * 1024 * 1024
    max_total_bytes: int = 4096 * 1024 * 1024


@dataclass
class ArchiveMember:
    name: str  # normalised POSIX relative path inside the archive
    data: bytes


@dataclass
class ArchiveContents:
    members: list[ArchiveMember]
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def safe_member_name(raw: str) -> str:
    """Normalise a member name or raise :class:`ArchiveError` for traversal attempts."""
    name = raw.replace("\\", "/")
    if name.startswith("/") or (len(name) > 1 and name[1] == ":"):
        raise ArchiveError(f"absolute path in archive member {raw!r}")
    parts = [p for p in PurePosixPath(name).parts if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise ArchiveError(f"path traversal in archive member {raw!r}")
    if not parts:
        raise ArchiveError(f"empty archive member name {raw!r}")
    return "/".join(parts)


def read_archive(path: Path, limits: ArchiveLimits | None = None) -> ArchiveContents:
    """Read image members of a CBZ/ZIP in natural order.

    Unsafe names (absolute, ``..``) abort the whole archive: a crafted archive is not
    partially trusted. Non-image members and directories are skipped; duplicate names
    keep the first occurrence.
    """
    limits = limits or ArchiveLimits()
    try:
        zf = zipfile.ZipFile(long_path(path))
    except (zipfile.BadZipFile, OSError) as exc:
        raise ArchiveError(f"{path.name}: not a readable ZIP/CBZ ({exc})") from exc
    with zf:
        infos = zf.infolist()
        if len(infos) > limits.max_members:
            raise ArchiveError(
                f"{path.name}: {len(infos)} members exceed the limit {limits.max_members}"
            )
        seen: set[str] = set()
        selected: list[tuple[str, zipfile.ZipInfo]] = []
        contents = ArchiveContents(members=[])
        total = 0
        for info in infos:
            name = safe_member_name(info.filename)
            if info.is_dir():
                continue
            if not is_image_name(name) or name.split("/")[-1].startswith("."):
                contents.skipped.append(name)
                continue
            key = name.casefold()
            if key in seen:
                contents.warnings.append(f"duplicate member {name!r} ignored")
                continue
            seen.add(key)
            if info.file_size > limits.max_member_bytes:
                raise ArchiveError(f"{path.name}: member {name!r} exceeds the size limit")
            total += info.file_size
            if total > limits.max_total_bytes:
                raise ArchiveError(f"{path.name}: uncompressed size exceeds the limit")
            selected.append((name, info))
        selected.sort(key=lambda item: natural_key(item[0]))
        for name, info in selected:
            try:
                with zf.open(info) as fh:
                    data = fh.read(limits.max_member_bytes + 1)
            except (zipfile.BadZipFile, OSError, RuntimeError, EOFError) as exc:
                contents.warnings.append(f"member {name!r} unreadable: {exc}")
                continue
            if len(data) > limits.max_member_bytes:
                raise ArchiveError(f"{path.name}: member {name!r} inflates beyond the limit")
            contents.members.append(ArchiveMember(name=name, data=data))
    for warning in contents.warnings:
        log.warning("%s: %s", path.name, warning)
    return contents


def write_cbz(path: Path, entries: list[tuple[str, bytes]]) -> None:
    """Write ``(member_name, data)`` pairs to a CBZ atomically (stored, images are packed)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in entries:
            zf.writestr(safe_member_name(name), data)
    atomic_write_bytes(path, buf.getvalue())
