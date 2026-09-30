from __future__ import annotations

import pytest

from manga_ar.cli import main


def test_doctor_offline_ok(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["doctor", "--no-network", "--offline"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "opencv distributions" in out and "primary font" in out and "overall: OK" in out


def test_fonts_check_and_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fonts", "check"]) == 0
    out = capsys.readouterr().out
    assert "NotoNaskhArabic" in out and "BASIC+RAQM" in out
    assert "BalooBhaijaan2" in out and "RAQM-only" in out
    assert main(["fonts", "list"]) == 0


def test_models_list_and_offline_download(
    capsys: pytest.CaptureFixture[str], tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MANGAAR_CACHE_DIR", str(tmp_path))
    assert main(["models", "list"]) == 0
    assert "ctd" in capsys.readouterr().out
    assert main(["models", "download", "mt-ja-en", "--offline"]) == 1
    assert "offline" in capsys.readouterr().out


def test_bad_config_is_fatal(tmp_path: object, capsys: pytest.CaptureFixture[str]) -> None:
    from pathlib import Path

    bad = Path(str(tmp_path)) / "bad.yaml"
    bad.write_text("nope: 1\n", encoding="utf-8")
    assert main(["doctor", "--no-network", "--config", str(bad)]) == 1
    assert "unknown keys" in capsys.readouterr().err
