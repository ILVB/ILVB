"""Input collection (``ImageSource``): files, folders (recursive) and CBZ/ZIP archives.

Every page becomes a :class:`PageJob` carrying a lazy byte reader, its relative output
directory and the group it belongs to (an archive or an input folder), so outputs keep
the relative layout and CBZ repacking knows which pages form one book. Order is natural
(page2 before page10) within each input; inputs keep the order they were given in.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from manga_ar.errors import ArchiveError, ImageLoadError
from manga_ar.io.archive import ArchiveLimits, read_archive
from manga_ar.io.loader import read_bytes
from manga_ar.io.naming import display_name, is_archive_name, is_image_name, natural_key
from manga_ar.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class PageJob:
    name: str  # display name, unique in the batch ("ch1/p01.png", "vol.cbz/p01.png")
    stem: str  # file stem used for output names
    rel_dir: PurePosixPath  # output directory relative to the output root
    group: str  # book the page belongs to (archive path or input folder)
    group_name: str  # stem of that book (CBZ output name)
    group_dir: PurePosixPath  # where that book's CBZ goes, relative to the output root
    member: str  # path of the page inside its book (CBZ member name base)
    read: Callable[[], bytes] = field(repr=False)


@dataclass
class InputProblem:
    name: str
    error: str
    status: str = "skipped"  # "skipped" (not an image) | "failed" (unreadable)


@dataclass
class CollectedInputs:
    jobs: list[PageJob] = field(default_factory=list)
    problems: list[InputProblem] = field(default_factory=list)


def _file_reader(path: Path) -> Callable[[], bytes]:
    return lambda: read_bytes(path)


def _bytes_reader(data: bytes) -> Callable[[], bytes]:
    return lambda: data


def collect_inputs(
    inputs: Sequence[Path],
    limits: ArchiveLimits | None = None,
    exclude: Path | None = None,
) -> CollectedInputs:
    """``exclude``: a directory never read from (the output root, when it lies inside an
    input folder, so a rerun does not translate its own outputs)."""
    out = CollectedInputs()
    excluded = exclude.resolve() if exclude is not None else None
    seen: set[str] = set()

    def add(job: PageJob) -> None:
        if job.name in seen:  # same page given twice: process once
            log.info("duplicate input %s ignored", job.name)
            return
        seen.add(job.name)
        out.jobs.append(job)

    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            _collect_dir(path, add, out, limits, excluded)
        elif path.is_file():
            if is_archive_name(path.name):
                _collect_archive(path, PurePosixPath(), display_name(path.name), add, out, limits)
            elif is_image_name(path.name):
                add(
                    PageJob(
                        name=display_name(path.name),
                        stem=path.stem,
                        rel_dir=PurePosixPath(),
                        group=str(path.parent.resolve()),
                        group_name=path.parent.resolve().name or "pages",
                        group_dir=PurePosixPath(),
                        member=path.name,
                        read=_file_reader(path),
                    )
                )
            else:
                out.problems.append(InputProblem(display_name(path.name), "unsupported file type"))
        else:
            out.problems.append(
                InputProblem(display_name(str(path)), "no such file or directory", "failed")
            )
    return out


def _collect_dir(
    root: Path,
    add: Callable[[PageJob], None],
    out: CollectedInputs,
    limits: ArchiveLimits | None,
    excluded: Path | None,
) -> None:
    files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and not _hidden(p.relative_to(root))
        and not (excluded is not None and p.resolve().is_relative_to(excluded))
    ]
    files.sort(key=lambda p: tuple(natural_key(part) for part in p.relative_to(root).parts))
    for path in files:
        rel = PurePosixPath(path.relative_to(root).as_posix())
        name = display_name(f"{root.name}/{rel}")
        if is_archive_name(path.name):
            _collect_archive(path, PurePosixPath(root.name) / rel.parent, name, add, out, limits)
        elif is_image_name(path.name):
            add(
                PageJob(
                    name=name,
                    stem=path.stem,
                    rel_dir=PurePosixPath(root.name) / rel.parent,
                    group=str(root.resolve()),
                    group_name=root.name,
                    group_dir=PurePosixPath(),
                    member=str(rel),
                    read=_file_reader(path),
                )
            )
        else:
            out.problems.append(InputProblem(name, "unsupported file type"))


def _collect_archive(
    path: Path,
    rel_parent: PurePosixPath,
    name: str,
    add: Callable[[PageJob], None],
    out: CollectedInputs,
    limits: ArchiveLimits | None,
) -> None:
    try:
        contents = read_archive(path, limits)
    except (ArchiveError, ImageLoadError) as exc:
        out.problems.append(InputProblem(name, str(exc), "failed"))
        return
    stem = path.stem
    for skipped in contents.skipped:
        log.info("%s: skipped non-image member %s", name, skipped)
    if not contents.members:
        out.problems.append(InputProblem(name, "archive contains no images", "failed"))
    for member in contents.members:
        mpath = PurePosixPath(member.name)
        add(
            PageJob(
                name=f"{name}/{member.name}",
                stem=mpath.stem,
                rel_dir=rel_parent / stem / mpath.parent,
                group=str(path.resolve()),
                group_name=stem,
                group_dir=rel_parent,
                member=member.name,
                read=_bytes_reader(member.data),
            )
        )


def _hidden(rel: Path) -> bool:
    """Skip dot-files/directories (including MangaAR's own ``.mangaar`` work dirs)."""
    return any(part.startswith(".") for part in rel.parts)
