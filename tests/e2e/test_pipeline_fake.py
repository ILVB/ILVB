"""End-to-end pipeline runs with all-fake stages (fast; Gate P6)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from manga_ar.cancel import CancelToken
from manga_ar.config import AppConfig, load_config
from manga_ar.io.writer import write_image
from manga_ar.pipeline import Pipeline, rerender
from manga_ar.report import EXIT_FATAL, EXIT_OK, EXIT_PARTIAL
from manga_ar.schemas import Flag, PageDocument
from tests.e2e.fakes import FakeDetector, FakeInpainter, FakeOcr, FakeTranslator, fake_stages, page
from tests.fixtures.synth import corrupt_png_crc


def _cfg(**over: Any) -> AppConfig:
    return load_config(overrides=over, environ={})


def _run(cfg: AppConfig, inputs: list[Path], out: Path, **stage_kw: Any) -> Any:
    return Pipeline(cfg, fake_stages(cfg, **stage_kw)).run(inputs, out)


def _pixels(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def test_batch_with_corrupted_files_exit_2_and_layout(tmp_path: Path) -> None:
    src = tmp_path / "chapter"
    for name in ("page10.png", "page2.png", "sub/page1.png"):
        write_image(src / name, page(), "png")
    (src / "empty.png").write_bytes(b"")
    (src / "truncated.png").write_bytes((src / "page2.png").read_bytes()[:60])
    (src / "badcrc.png").write_bytes(corrupt_png_crc((src / "page2.png").read_bytes()))
    (src / "notes.txt").write_text("not an image", encoding="utf-8")
    out = tmp_path / "out"
    report = _run(_cfg(**{"input.truncated": "strict"}), [src], out)
    assert report.exit_code == EXIT_PARTIAL
    by_name = {p.name: p for p in report.pages}
    for bad in ("empty.png", "truncated.png", "badcrc.png", "notes.txt"):
        assert by_name[f"chapter/{bad}"].status == "skipped", bad
        assert by_name[f"chapter/{bad}"].error
    ok = [p.name for p in report.pages if p.status == "ok"]
    assert ok == ["chapter/page2.png", "chapter/page10.png", "chapter/sub/page1.png"]  # natural
    for rel in ("page2", "page10", "sub/page1"):
        assert (out / "chapter" / f"{rel}_ar.png").is_file()
        doc = PageDocument.load(out / "chapter" / f"{rel}_ar.mangaar.json")
        assert doc.status == "ok" and all(r.layout is not None for r in doc.regions)
    saved = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert saved["exit_code"] == EXIT_PARTIAL and saved["counts"]["skipped"] == 4
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "empty.png" in md and "truncated.png" in md
    # the text blocks were replaced: black code pixels gone, Arabic ink present
    final = _pixels(out / "chapter" / "page2_ar.png")
    assert final[140:160, 75:100].min() < 128 or final[130:170, 40:140].min() < 128
    assert not (final[135:165, 70:110] == 0).all()


def test_all_failed_is_fatal_and_missing_input(tmp_path: Path) -> None:
    (tmp_path / "bad.png").write_bytes(b"\x89PNG broken")
    report = _run(_cfg(), [tmp_path / "bad.png", tmp_path / "missing.png"], tmp_path / "o")
    assert report.exit_code == EXIT_FATAL
    assert {p.status for p in report.pages} == {"skipped", "failed"}


def test_resume_skips_and_config_change_reprocesses(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    write_image(src, page(), "png")
    out = tmp_path / "out"
    det = FakeDetector()
    assert _run(_cfg(), [src], out, detector=det).exit_code == EXIT_OK
    assert det.calls == 1
    report = _run(_cfg(**{"output.resume": True}), [src], out, detector=det)
    assert report.pages[0].status == "resumed" and det.calls == 1
    report = _run(_cfg(**{"output.resume": True, "output.force": True}), [src], out, detector=det)
    assert report.pages[0].status == "ok" and det.calls == 2
    changed = _cfg(**{"output.resume": True, "typeset.font": "Cairo"})
    report = _run(changed, [src], out, detector=det)
    assert report.pages[0].status == "ok" and det.calls == 3  # config hash differs
    # output deleted but sidecar + work images intact: re-rendered without detection
    (out / "in_ar.png").unlink()
    report = _run(changed, [src], out, detector=det)
    assert report.pages[0].status == "resumed" and det.calls == 3
    assert (out / "in_ar.png").is_file()


def test_degraded_page_retries_translation_on_resume(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    write_image(src, page(), "png")
    out = tmp_path / "out"
    offline = FakeTranslator(table={}, offline=False)
    report = _run(_cfg(), [src], out, translator=offline)
    assert report.pages[0].status == "degraded" and report.exit_code == EXIT_OK
    assert any("models download local-mt" in n for n in report.notes)
    first = _pixels(out / "in_ar.png")
    assert (first[135:165, 70:110] == 0).all()  # untranslated: original pixels kept
    det = FakeDetector()
    report = _run(_cfg(**{"output.resume": True}), [src], out, detector=det)
    assert det.calls == 0 and report.pages[0].status == "resumed"
    doc = PageDocument.load(out / "in_ar.mangaar.json")
    assert doc.status == "ok" and all(r.translation is not None for r in doc.regions)


def test_region_failures_are_isolated(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    write_image(src, page((40, 60, 80), w=560), "png")
    out = tmp_path / "out"
    report = _run(_cfg(), [src], out, ocr=FakeOcr(fail_widths={60}), inpainter=FakeInpainter({80}))
    p = report.pages[0]
    assert p.status == "degraded" and p.translated == 1
    doc = PageDocument.load(out / "in_ar.mangaar.json")
    flags = {r.bbox.width: r.flags for r in doc.regions}
    assert Flag.OCR_FAILED in flags[60] and Flag.SKIPPED in flags[80]
    final = _pixels(out / "in_ar.png")
    assert (final[135:165, 370:450] == 0).all()  # failed inpaint: original text kept
    assert any("inpainting failed" in w for w in p.warnings)


def test_erase_untranslated_and_no_text_page(tmp_path: Path) -> None:
    write_image(tmp_path / "a.png", page((50,)), "png")  # code 50 has no translation
    blank = np.full((200, 300, 3), 255, np.uint8)
    blank[0:4, 0:4] = (0, 90, 200)
    write_image(tmp_path / "blank.png", blank, "png")
    out = tmp_path / "out"
    report = _run(_cfg(**{"inpaint.erase_untranslated": True}), [tmp_path], out)
    by = {p.name.split("/")[-1]: p for p in report.pages}
    assert by["blank.png"].status == "no_text"
    assert np.array_equal(_pixels(out / tmp_path.name / "blank_ar.png"), blank)
    erased = _pixels(out / tmp_path.name / "a_ar.png")
    assert erased[135:165, 65:115].min() == 255  # erased, not restored


def test_cbz_in_cbz_out_and_rerender_member(tmp_path: Path) -> None:
    book = tmp_path / "vol1.cbz"
    with zipfile.ZipFile(book, "w") as zf:
        for name in ("p10.png", "p2.png"):
            zf.writestr(name, _png(page()))
        zf.writestr("readme.txt", "hi")
    out = tmp_path / "out"
    report = _run(_cfg(**{"output.format": "cbz"}), [book], out)
    assert report.exit_code == EXIT_OK and report.archives == ["vol1_ar.cbz"]
    with zipfile.ZipFile(out / "vol1_ar.cbz") as zf:
        assert zf.namelist() == ["p2_ar.png", "p10_ar.png"]
    sidecar = out / "vol1" / "p2_ar.mangaar.json"
    doc = PageDocument.load(sidecar)
    doc.regions[0].override.text = "نص جديد"
    before = zipfile.ZipFile(out / "vol1_ar.cbz").read("p2_ar.png")
    result = rerender(sidecar, _cfg(), doc=doc)
    after = zipfile.ZipFile(out / "vol1_ar.cbz").read("p2_ar.png")
    assert before != after and result.doc.regions[0].layout is not None
    assert result.doc.regions[0].layout.lines == ["نص جديد"]
    with zipfile.ZipFile(out / "vol1_ar.cbz") as zf:
        assert zf.namelist() == ["p2_ar.png", "p10_ar.png"]
    # CBZ pages resume by re-rendering from their sidecars (no detection)
    det = FakeDetector()
    report = _run(
        _cfg(**{"output.format": "cbz", "output.resume": True}), [book], out, detector=det
    )
    assert det.calls == 0 and {p.status for p in report.pages} == {"resumed"}


def _png(arr: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def test_cbz_to_images_preserves_member_dirs(tmp_path: Path) -> None:
    book = tmp_path / "vol2.zip"
    with zipfile.ZipFile(book, "w") as zf:
        zf.writestr("ch1/01.png", _png(page()))
    out = tmp_path / "out"
    report = _run(_cfg(**{"output.format": "jpg"}), [book], out)
    assert report.exit_code == EXIT_OK
    assert (out / "vol2" / "ch1" / "01_ar.jpg").is_file()


def test_rerender_with_font_override(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    write_image(src, page(), "png")
    out = tmp_path / "out"
    _run(_cfg(), [src], out)
    sidecar = out / "in_ar.mangaar.json"
    result = rerender(sidecar, _cfg(), {"typeset.font": "Cairo"})
    assert {r.layout.font for r in result.doc.regions if r.layout} == {"Cairo"}
    assert PageDocument.load(sidecar).settings["typeset"]["font"] == "Cairo"


def test_cancellation_between_pages(tmp_path: Path) -> None:
    for k in range(3):
        write_image(tmp_path / "in" / f"p{k}.png", page(), "png")
    token = CancelToken()
    seen: list[str] = []

    def progress(fraction: float, message: str) -> None:
        seen.append(message)
        if len(seen) == 2:
            token.cancel()

    cfg = _cfg()
    report = Pipeline(cfg, fake_stages(cfg), cancel=token, progress=progress).run(
        [tmp_path / "in"], tmp_path / "out"
    )
    assert report.cancelled and report.exit_code == EXIT_PARTIAL
    assert [p.status for p in report.pages] == ["ok", "cancelled"]


def test_grayscale_debug_sfx_and_language_vote(tmp_path: Path) -> None:
    img = page(gray=True)
    img[40:260, 330:350] = 0  # tall solid block → fake SFX
    Image.fromarray(img[..., 0], "L").save(tmp_path / "g.png")
    out = tmp_path / "out"
    ocr = FakeOcr()
    cfg = _cfg(**{"runtime.debug": True, "detect.sfx": "translate"})
    report = Pipeline(cfg, fake_stages(cfg, ocr=ocr)).run([tmp_path / "g.png"], out)
    assert report.pages[0].lang == "zh" and ocr.probes == 1
    assert Image.open(out / "g_ar.png").mode == "L"  # grayscale in → grayscale out
    doc = PageDocument.load(out / "g_ar.mangaar.json")
    assert any(Flag.FROM_SFX in r.flags for r in doc.regions)
    names = {p.name for p in (out / "debug" / "g").iterdir()}
    assert {"01_detections.png", "04_inpaint_diff.png", "07_text_layer.png"} <= names


def test_page_crash_is_isolated(tmp_path: Path) -> None:
    class Boom(FakeDetector):
        def detect(self, rgb: np.ndarray) -> list:  # type: ignore[type-arg]
            if rgb.shape[1] == 300:
                raise RuntimeError("detector exploded")
            return super().detect(rgb)

    write_image(tmp_path / "in" / "a.png", page(), "png")
    write_image(tmp_path / "in" / "b.png", page(w=300, widths=(40,)), "png")
    report = _run(_cfg(), [tmp_path / "in"], tmp_path / "out", detector=Boom())
    assert [p.status for p in report.pages] == ["ok", "failed"]
    assert "detector exploded" in (report.pages[1].error or "")
    assert report.exit_code == EXIT_PARTIAL


def test_output_dir_inside_input_is_not_reprocessed(tmp_path: Path) -> None:
    write_image(tmp_path / "a.png", page(), "png")
    out = tmp_path / "translated"
    _run(_cfg(), [tmp_path], out)
    report = _run(_cfg(), [tmp_path], out)
    assert [p.name.split("/")[-1] for p in report.pages] == ["a.png"]


@pytest.mark.parametrize("fmt", ["png", "webp"])
def test_formats(tmp_path: Path, fmt: str) -> None:
    write_image(tmp_path / "a.png", page(), "png")
    report = _run(_cfg(**{"output.format": fmt}), [tmp_path / "a.png"], tmp_path / "o")
    assert (tmp_path / "o" / f"a_ar.{fmt}").is_file() and report.exit_code == EXIT_OK


def test_region_language_override_on_resume(tmp_path: Path) -> None:
    """E6: a per-region source-language override (sidecar/GUI) drives translation."""
    src = tmp_path / "in.png"
    write_image(src, page(), "png")
    out = tmp_path / "out"
    _run(_cfg(), [src], out, translator=FakeTranslator(table={}))
    sidecar = out / "in_ar.mangaar.json"
    doc = PageDocument.load(sidecar)
    doc.regions[0].override.source_lang = "ko"
    doc.save(sidecar)
    translator = FakeTranslator()
    _run(_cfg(**{"output.resume": True}), [src], out, translator=translator)
    assert sorted(translator.sources) == ["ko", "zh"]
