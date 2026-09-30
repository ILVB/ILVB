"""GUI handler tests (no browser): translate → review → edit → re-render → re-translate."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from manga_ar.config import AppConfig, load_config
from manga_ar.errors import MangaArError
from manga_ar.io.writer import write_image
from manga_ar.pipeline import Stages
from manga_ar.schemas import PageDocument
from manga_ar.ui.handlers import GuiController, GuiSettings, _size, _truthy
from tests.e2e.fakes import FakeTranslator, fake_stages, page


@pytest.fixture
def controller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GuiController:
    ctl = GuiController(load_config(environ={}), tmp_path / "ws")
    translator = FakeTranslator()

    def stages_for(cfg: AppConfig) -> Stages:
        return fake_stages(cfg, translator=translator)

    monkeypatch.setattr(ctl, "_stages_for", stages_for)
    yield ctl
    ctl.cleanup()


def _upload(tmp_path: Path) -> list[str]:
    a, b = tmp_path / "up" / "p1.png", tmp_path / "up" / "p10.png"
    write_image(a, page(), "png")
    write_image(b, page((80,)), "png")
    (tmp_path / "up" / "x.txt").write_text("ignored", encoding="utf-8")
    return [str(b), str(a), str(tmp_path / "up" / "x.txt")]  # unsorted on purpose


def test_translate_review_edit_rerender(controller: GuiController, tmp_path: Path) -> None:
    seen: list[float] = []
    outcome = controller.translate(
        _upload(tmp_path), GuiSettings(), progress=lambda f, m: seen.append(f)
    )
    assert outcome.report.exit_code == 0 and len(outcome.pages) == 2
    assert seen and seen[-1] == 1.0 and "2 page(s)" in outcome.summary
    assert outcome.bundle is not None
    with zipfile.ZipFile(outcome.bundle) as zf:
        names = set(zf.namelist())
    assert {"p1_ar.png", "p10_ar.png", "report.md"} <= names
    assert [p.name for p in outcome.report.pages] == ["p1.png", "p10.png"]  # natural order
    assert not any(".mangaar/" in n for n in names)

    sidecar = next(p.sidecar for p in outcome.pages if p.label == "p1.png")
    view = controller.page(sidecar)
    assert [r[3] for r in view.rows] == ["مرحبا", "إلى اللقاء"]
    before = np.asarray(Image.open(view.result))
    rows = [list(r) for r in view.rows]
    rows[0][3] = "أهلا وسهلا"  # edit Arabic text
    rows[1][4] = "Cairo"  # font override
    rows[1][5] = "24"  # size override
    edited = controller.apply_edits(sidecar, rows)
    doc = PageDocument.load(sidecar)
    assert doc.regions[0].override.text == "أهلا وسهلا"
    assert doc.regions[1].layout is not None and doc.regions[1].layout.font == "Cairo"
    assert doc.regions[1].layout.size_px == 24
    assert edited.rows[0][3] == "أهلا وسهلا"
    assert not np.array_equal(before, np.asarray(Image.open(edited.result)))

    rows = [list(r) for r in edited.rows]
    rows[0][6] = True  # skip region 0
    rows[0][3] = "مرحبا"  # back to the translation: override cleared
    controller.apply_edits(sidecar, rows)
    doc = PageDocument.load(sidecar)
    assert doc.regions[0].override.skip and doc.regions[0].layout is None
    assert doc.regions[0].override.text is None


def test_retranslate_and_errors(controller: GuiController, tmp_path: Path) -> None:
    outcome = controller.translate(_upload(tmp_path), GuiSettings())
    sidecar = outcome.pages[0].sidecar
    doc = PageDocument.load(sidecar)
    rid = doc.regions[0].id
    doc.regions[0].override.text = "نص"
    doc.save(sidecar)
    view = controller.retranslate(sidecar, rid, GuiSettings())
    assert view.rows[0][3] == "مرحبا" and view.doc.regions[0].override.text is None
    with pytest.raises(MangaArError, match="no region"):
        controller.retranslate(sidecar, "nope", GuiSettings())
    rows = [list(r) for r in view.rows]
    rows[0][4] = "NotAFont"
    with pytest.raises(MangaArError, match="unknown font"):
        controller.apply_edits(sidecar, rows)
    with pytest.raises(MangaArError, match="upload"):
        controller.translate([], GuiSettings())
    with pytest.raises(MangaArError, match="none of the uploaded"):
        controller.translate([str(tmp_path / "up" / "x.txt")], GuiSettings())
    with pytest.raises(MangaArError, match="invalid settings"):
        controller.config_for(GuiSettings(providers=["bogus"]))


def test_cbz_page_preview_cancel_diagnostics_cleanup(
    controller: GuiController, tmp_path: Path
) -> None:
    files = _upload(tmp_path)[:1]
    outcome = controller.translate(files, GuiSettings(output_format="cbz", font="Tajawal"))
    view = controller.page(outcome.pages[0].sidecar)
    assert view.result.parent.name == "previews" and view.result.is_file()
    assert "Cancelling" in controller.cancel() and controller.cancel_token.cancelled
    outcome = controller.translate(files, GuiSettings())  # a new run resets the token
    assert not outcome.report.cancelled
    text = controller.diagnostics()
    assert "python:" in text and "recent log" in text
    assert "NotoNaskhArabic" in controller.font_choices()
    ws = controller.workspace
    controller.cleanup()
    assert not ws.exists()


def test_small_parsers() -> None:
    assert _size("", "r") is None and _size("24", "r") == 24 and _size(30.0, "r") == 30
    with pytest.raises(MangaArError):
        _size("big", "r")
    with pytest.raises(MangaArError):
        _size(1000, "r")
    assert _truthy("yes") and _truthy(True) and not _truthy("false") and not _truthy(0)
