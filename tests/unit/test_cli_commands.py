"""In-process CLI tests for translate / rerender / gui with fake pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from manga_ar import cli
from manga_ar.config import AppConfig
from manga_ar.io.writer import write_image
from manga_ar.models.manager import ModelManager
from manga_ar.pipeline import Stages
from manga_ar.schemas import PageDocument
from tests.e2e.fakes import fake_stages, page


@pytest.fixture(autouse=True)
def _fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    def build(cfg: AppConfig, manager: ModelManager, tm: Any = None) -> Stages:
        return fake_stages(cfg)

    monkeypatch.setattr("manga_ar.pipeline.build_stages", build)


def test_translate_command_flags_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_image(tmp_path / "in" / "a.png", page(), "png")
    (tmp_path / "in" / "bad.png").write_bytes(b"")
    out = tmp_path / "out"
    code = cli.main(
        [
            "translate", str(tmp_path / "in"), "-o", str(out), "--source", "zh",
            "--preset", "fast", "--providers", "tm,local", "--font", "Cairo",
            "--digits", "arabic_indic", "--sfx", "translate", "--format", "webp",
            "--reading-order", "comic_ltr", "--erase-untranslated", "--debug", "--quiet",
            "--offline",
        ]
    )  # fmt: skip
    assert code == cli.EXIT_PARTIAL
    text = capsys.readouterr().out
    assert "SKIPPED in/bad.png" in text and "report:" in text
    doc = PageDocument.load(out / "in" / "a_ar.mangaar.json")
    s = doc.settings
    assert s["preset"] == "fast" and s["typeset"]["font"] == "Cairo"
    assert s["translate"]["providers"] == ["tm", "local"] and s["output"]["format"] == "webp"
    assert s["inpaint"]["erase_untranslated"] and s["detect"]["sfx"] == "translate"
    assert doc.reading_order_mode == "comic_ltr" and (out / "in" / "a_ar.webp").is_file()
    assert (out / "debug").is_dir()
    code = cli.main(["translate", str(tmp_path / "in" / "a.png"), "-o", str(out), "--quiet",
                     "--resume"])  # fmt: skip
    assert code == cli.EXIT_OK
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["pages"][0]["status"] == "ok"  # settings changed → reprocessed


def test_rerender_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_image(tmp_path / "a.png", page(), "png")
    out = tmp_path / "out"
    assert cli.main(["translate", str(tmp_path / "a.png"), "-o", str(out), "--quiet"]) == 0
    sidecar = out / "a_ar.mangaar.json"
    assert cli.main(["rerender", str(sidecar), "--font", "Amiri"]) == cli.EXIT_OK
    assert {r.layout.font for r in PageDocument.load(sidecar).regions if r.layout} == {"Amiri"}
    missing = tmp_path / "missing.mangaar.json"
    assert cli.main(["rerender", str(sidecar), str(missing)]) == cli.EXIT_PARTIAL
    assert cli.main(["rerender", str(missing)]) == cli.EXIT_FATAL
    assert "FAIL" in capsys.readouterr().out


def test_gui_command_uses_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_launch(cfg: AppConfig, **kwargs: Any) -> None:
        calls.update(kwargs)

    import manga_ar.ui.gradio_app as app

    monkeypatch.setattr(app, "launch", fake_launch)
    assert cli.main(["gui", "--no-browser", "--port", "7999"]) == cli.EXIT_OK
    assert calls == {"host": "127.0.0.1", "port": 7999, "open_browser": False}


def test_bad_provider_list_is_a_usage_error() -> None:
    with pytest.raises(SystemExit):
        cli.main(["translate", "x", "-o", "y", "--providers", ","])
    assert cli.main(["translate", "x", "-o", "y", "--providers", "bogus", "--quiet"]) == 1
