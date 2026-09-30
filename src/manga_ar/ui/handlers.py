"""GUI logic, independent of Gradio (handler tests call this directly).

The Gradio layer only maps widgets to these methods; all work goes through the same
:class:`~manga_ar.pipeline.Pipeline` and :func:`~manga_ar.pipeline.rerender` the CLI uses.
Every run gets its own directory inside a temporary workspace that is deleted on exit.
"""

from __future__ import annotations

import collections
import logging
import shutil
import tempfile
import threading
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from manga_ar.cancel import CancelToken
from manga_ar.config import AppConfig, resolve_cache_dir
from manga_ar.errors import ConfigError, MangaArError
from manga_ar.io.archive import read_archive
from manga_ar.io.naming import is_archive_name, is_image_name, natural_key
from manga_ar.logging_setup import get_logger
from manga_ar.models.manager import ModelManager
from manga_ar.pipeline import Pipeline, ProgressFn, Stages, build_stages, rerender
from manga_ar.report import BatchReport
from manga_ar.schemas import Flag, PageDocument, Region
from manga_ar.typeset.fonts import FontRegistry

log = get_logger(__name__)
COLUMNS = ["id", "type", "source text", "Arabic text", "font", "size", "skip", "flags"]
EDITABLE = {3, 4, 5, 6}


@dataclass
class GuiSettings:
    source: str = "auto"
    preset: str = "balanced"
    device: str = "auto"
    providers: list[str] = field(default_factory=list)  # empty: configuration default
    font: str = ""
    digits: str = "western"
    sfx: str = "skip"
    erase_untranslated: bool = False
    output_format: str = "png"
    glossary: str | None = None
    tm: str | None = None

    def overrides(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "input.source_lang": self.source,
            "runtime.device": self.device,
            "translate.providers": list(self.providers) or None,
            "typeset.font": self.font or None,
            "typeset.digits": self.digits,
            "detect.sfx": self.sfx,
            "inpaint.erase_untranslated": self.erase_untranslated,
            "output.format": self.output_format,
            "translate.glossary_file": self.glossary or None,
            "translate.tm_file": self.tm or None,
            "output.resume": False,
        }


@dataclass
class PageEntry:
    label: str
    sidecar: Path


@dataclass
class TranslateOutcome:
    report: BatchReport
    pages: list[PageEntry]
    bundle: Path | None
    summary: str


@dataclass
class PageView:
    rows: list[list[Any]]
    original: Path
    result: Path
    doc: PageDocument


class RingLogHandler(logging.Handler):
    """Keeps the last ``capacity`` formatted records for the Diagnostics tab."""

    def __init__(self, capacity: int = 400) -> None:
        super().__init__(logging.INFO)
        self.records: collections.deque[str] = collections.deque(maxlen=capacity)
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


class GuiController:
    def __init__(self, cfg: AppConfig, workspace: Path | None = None) -> None:
        self.base_cfg = cfg
        self.workspace = workspace or Path(tempfile.mkdtemp(prefix="mangaar-gui-"))
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.cancel_token = CancelToken()
        self.registry = FontRegistry()
        self.manager = ModelManager(resolve_cache_dir(cfg), offline=cfg.runtime.offline)
        self.logs = RingLogHandler()
        logging.getLogger("manga_ar").addHandler(self.logs)
        self._stages: dict[str, Stages] = {}
        self._runs = 0
        self.pages: list[PageEntry] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------ settings
    def font_choices(self) -> list[str]:
        return sorted(k for k, f in self.registry.fonts.items() if f.role == "arabic")

    def config_for(self, settings: GuiSettings) -> AppConfig:
        clean = {k: v for k, v in settings.overrides().items() if v is not None}
        try:
            return self.base_cfg.replace(**clean)
        except ConfigError as exc:
            raise MangaArError(f"invalid settings: {exc}") from exc

    def _stages_for(self, cfg: AppConfig) -> Stages:
        key = cfg.config_hash() + cfg.runtime.device + str(cfg.runtime.offline)
        if key not in self._stages:
            self._stages.clear()  # one model set in memory at a time
            self._stages[key] = build_stages(cfg, self.manager)
        return self._stages[key]

    # ----------------------------------------------------------- translate
    def translate(
        self, files: Sequence[str | Path], settings: GuiSettings, progress: ProgressFn | None = None
    ) -> TranslateOutcome:
        if not files:
            raise MangaArError("upload at least one image, folder or CBZ/ZIP archive")
        with self._lock:
            self._runs += 1
            run = self.workspace / f"run-{self._runs:03d}"
        inputs = self._stage_inputs(files, run / "input")
        if not inputs:
            raise MangaArError("none of the uploaded files is an image or a CBZ/ZIP archive")
        cfg = self.config_for(settings)
        self.cancel_token.reset()
        pipeline = Pipeline(cfg, self._stages_for(cfg), cancel=self.cancel_token, progress=progress)
        out = run / "output"
        report = pipeline.run(inputs, out)
        self.pages = self._scan_pages(out)
        bundle = self._bundle(out, run / "mangaar_output.zip") if self.pages else None
        counts = ", ".join(f"{k}: {v}" for k, v in sorted(report.counts.items()))
        summary = f"**{len(report.pages)} page(s)** — {counts} (exit code {report.exit_code})"
        if report.cancelled:
            summary += " — cancelled"
        return TranslateOutcome(report, self.pages, bundle, summary)

    def cancel(self) -> str:
        self.cancel_token.cancel()
        return "Cancelling after the current region…"

    @staticmethod
    def _stage_inputs(files: Sequence[str | Path], target: Path) -> list[Path]:
        target.mkdir(parents=True, exist_ok=True)
        staged: list[Path] = []
        for raw in files:
            src = Path(raw)
            if src.is_dir():
                dest = target / src.name
                shutil.copytree(src, dest, dirs_exist_ok=True)
                staged.append(dest)
            elif is_image_name(src.name) or is_archive_name(src.name):
                dest = target / src.name
                n = 1
                while dest.exists():  # two uploads with the same name
                    dest = target / f"{src.stem}-{n}{src.suffix}"
                    n += 1
                shutil.copy2(src, dest)
                staged.append(dest)
            else:
                log.warning("ignored upload %s (not an image or archive)", src.name)
        return sorted(staged, key=lambda p: natural_key(p.name))  # page2 before page10

    @staticmethod
    def _scan_pages(out: Path) -> list[PageEntry]:
        entries = []
        sidecars = sorted(
            out.rglob("*.mangaar.json"),
            key=lambda p: tuple(natural_key(part) for part in p.relative_to(out).parts),
        )
        for sidecar in sidecars:
            try:
                doc = PageDocument.load(sidecar)
            except MangaArError:
                continue
            entries.append(PageEntry(doc.source, sidecar))
        return entries

    @staticmethod
    def _bundle(out: Path, target: Path) -> Path:
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(out.rglob("*")):
                rel = path.relative_to(out)
                if (
                    path.is_dir()
                    or rel.parts[0] in {".mangaar", "debug"}
                    or ".mangaar" in rel.parts
                ):
                    continue
                zf.write(path, rel.as_posix())
        return target

    # ------------------------------------------------------------- review
    def page(self, sidecar: str | Path) -> PageView:
        path = Path(sidecar)
        doc = PageDocument.load(path)
        return PageView(self.rows(doc), *self._images(doc, path), doc)

    @staticmethod
    def rows(doc: PageDocument) -> list[list[Any]]:
        rows = []
        for r in sorted(doc.regions, key=lambda x: x.reading_order):
            arabic = r.override.text if r.override.text is not None else _translation(r)
            rows.append(
                [
                    r.id,
                    r.type.value,
                    r.source_text,
                    arabic,
                    r.override.font or "",
                    r.override.size_px or "",
                    bool(r.override.skip),
                    " ".join(sorted(f.value for f in r.flags)),
                ]
            )
        return rows

    def _images(self, doc: PageDocument, sidecar: Path) -> tuple[Path, Path]:
        base = sidecar.parent
        if not doc.source_image or not doc.output:
            raise MangaArError(f"{sidecar.name}: page has no work images")
        original = base / doc.source_image
        result = base / doc.output
        if doc.output_member:  # CBZ page: extract a preview
            contents = read_archive(result)
            member = next((m for m in contents.members if m.name == doc.output_member), None)
            if member is None:
                raise MangaArError(f"{doc.output_member} missing from {result.name}")
            preview = self.workspace / "previews" / f"{sidecar.stem}.png"
            preview.parent.mkdir(parents=True, exist_ok=True)
            preview.write_bytes(member.data)
            result = preview
        return original, result

    def apply_edits(self, sidecar: str | Path, rows: Sequence[Sequence[Any]]) -> PageView:
        """Write table edits into region overrides, then re-render the page."""
        path = Path(sidecar)
        doc = PageDocument.load(path)
        fonts = set(self.registry.fonts)
        by_id = {r.id: r for r in doc.regions}
        for row in rows:
            if not row or str(row[0]) not in by_id:
                continue
            region = by_id[str(row[0])]
            arabic = str(row[3] or "").strip()
            region.override.text = arabic if arabic and arabic != _translation(region) else None
            font = str(row[4] or "").strip()
            if font and font not in fonts:
                raise MangaArError(f"{region.id}: unknown font {font!r}")
            region.override.font = font or None
            region.override.size_px = _size(row[5], region.id)
            region.override.skip = _truthy(row[6])
        rerender(path, self.base_cfg, registry=self.registry, doc=doc)
        return self.page(path)

    def retranslate(self, sidecar: str | Path, region_id: str, settings: GuiSettings) -> PageView:
        path = Path(sidecar)
        doc = PageDocument.load(path)
        region = next((r for r in doc.regions if r.id == region_id.strip()), None)
        if region is None:
            raise MangaArError(f"no region {region_id!r} on this page")
        if not region.source_text:
            raise MangaArError(f"{region.id} has no source text to translate")
        cfg = self.config_for(settings)
        translator = self._stages_for(cfg).translator
        result = translator.translate_text(
            region.source_text, region.source_lang or doc.lang or "ja"
        )
        if result is None:
            raise MangaArError(f"{region.id}: every translation provider failed")
        region.translation = result
        region.override.text = None
        region.flags.discard(Flag.UNTRANSLATED)
        rerender(path, self.base_cfg, registry=self.registry, doc=doc)
        return self.page(path)

    # --------------------------------------------------------- diagnostics
    def diagnostics(self, check_network: bool = False) -> str:
        from manga_ar.doctor import run_doctor

        report = run_doctor(self.base_cfg, check_network=check_network).render()
        return report + "\n\n--- recent log ---\n" + "\n".join(self.logs.records)

    def cleanup(self) -> None:
        logging.getLogger("manga_ar").removeHandler(self.logs)
        self._stages.clear()
        shutil.rmtree(self.workspace, ignore_errors=True)


def _translation(region: Region) -> str:
    if region.translation is not None and Flag.UNTRANSLATED not in region.flags:
        return region.translation.text
    return ""


def _size(value: Any, region_id: str) -> int | None:
    if value in (None, "", 0, "0"):
        return None
    try:
        size = int(float(value))
    except (TypeError, ValueError) as exc:
        raise MangaArError(f"{region_id}: size must be a number of pixels") from exc
    if not 6 <= size <= 400:
        raise MangaArError(f"{region_id}: size {size}px outside 6-400")
    return size


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "x"}
    return bool(value)
