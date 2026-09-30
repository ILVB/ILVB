"""CLI end-to-end runs with the real stages (classical detector, RapidOCR, OpenCV/solid
inpainting, translation memory, Arabic typesetting). Marked ``slow``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from manga_ar import synth
from manga_ar.io.writer import write_image
from manga_ar.schemas import Flag, PageDocument

pytestmark = pytest.mark.slow
pytest.importorskip("rapidocr_onnxruntime")


def _cli(
    *args: str, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    full_env = {**os.environ, "PYTHONIOENCODING": "utf-8", **(env or {})}
    return subprocess.run(
        [sys.executable, "-m", "manga_ar", *args],
        cwd=cwd,
        env=full_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
        check=False,
    )


@pytest.fixture(scope="module")
def batch(tmp_path_factory: pytest.TempPathFactory, cache_dir: Path) -> Path:
    try:
        synth.find_cjk_font("zh", cache_dir)
    except Exception as exc:  # noqa: BLE001 - skip when the fixture font is unavailable
        pytest.skip(f"CJK font unavailable: {exc}")
    root = tmp_path_factory.mktemp("cli")
    src = root / "chapter 1"  # a space in the path on purpose
    write_image(src / "page10.png", synth.demo_page(cache_dir).image, "png")
    write_image(src / "page2.png", synth.demo_page(cache_dir).image, "png")
    (src / "zero.png").write_bytes(b"")
    (src / "broken.jpg").write_bytes(b"\xff\xd8\xff\xe0 truncated")
    (root / "tm.json").write_text(
        json.dumps(synth.DEMO_TRANSLATIONS, ensure_ascii=False), encoding="utf-8"
    )
    return root


def test_translate_corrupted_batch_exit_2_then_resume(batch: Path) -> None:
    out = batch / "out"
    args = [
        "translate",
        "chapter 1",
        "-o",
        "out",
        "--source",
        "zh",
        "--preset",
        "fast",
        "--providers",
        "tm",
        "--tm",
        "tm.json",
        "--offline",
        "--quiet",
    ]
    first = _cli(*args, cwd=batch)
    assert first.returncode == 2, first.stdout + first.stderr
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    status = {p["name"]: p["status"] for p in report["pages"]}
    assert status["chapter 1/zero.png"] == "skipped"
    assert status["chapter 1/broken.jpg"] == "skipped"
    assert [n for n, s in status.items() if s == "ok"] == [
        "chapter 1/page2.png",
        "chapter 1/page10.png",
    ]
    assert "zero.png" in (out / "report.md").read_text(encoding="utf-8")
    doc = PageDocument.load(out / "chapter 1" / "page2_ar.mangaar.json")
    assert [r.translation.text for r in doc.regions if r.translation] and all(
        r.layout is not None for r in doc.regions
    )
    second = _cli(*args, "--resume", cwd=batch)
    assert second.returncode == 2
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert {p["status"] for p in report["pages"] if "page" in p["name"]} == {"resumed"}

    rerender = _cli(
        "rerender", str(out / "chapter 1" / "page2_ar.mangaar.json"), "--font", "Almarai", cwd=batch
    )
    assert rerender.returncode == 0, rerender.stdout + rerender.stderr
    doc = PageDocument.load(out / "chapter 1" / "page2_ar.mangaar.json")
    assert {r.layout.font for r in doc.regions if r.layout} == {"Almarai"}


def test_offline_without_local_model_keeps_originals_and_explains(batch: Path) -> None:
    """E4: offline with no usable offline provider → UNTRANSLATED, originals kept, the
    report explains how to install the local model. With the Marian weights installed
    (W-004) the same command translates everything."""
    out = batch / "offline"
    result = _cli(
        "translate",
        "chapter 1/page2.png",
        "-o",
        "offline",
        "--source",
        "zh",
        "--providers",
        "local",
        "--offline",
        "--preset",
        "fast",
        "--quiet",
        cwd=batch,
    )
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    page = report["pages"][0]
    if page["status"] == "ok":  # local MT installed on this host: full offline success
        assert result.returncode == 0 and page["translated"] == page["regions"]
        return
    assert result.returncode == 0 and page["status"] == "degraded"
    assert page["flags"].get(Flag.UNTRANSLATED.value) == page["regions"]
    assert any("models download local-mt" in n for n in report["notes"])


def test_demo_command(tmp_path: Path) -> None:
    result = _cli("demo", "-o", str(tmp_path / "demo"), cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "regions translated: 3/3" in result.stdout
    assert (tmp_path / "demo" / "output" / "demo_zh_ar.png").is_file()


def test_missing_input_is_fatal(tmp_path: Path) -> None:
    result = _cli("translate", "nope.png", "-o", "out", "--offline", "--quiet", cwd=tmp_path)
    assert result.returncode == 1
    assert "FAILED nope.png" in result.stdout
