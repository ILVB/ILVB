"""GUI launch smoke test through gradio_client: upload → translate → edit → re-render."""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("gradio")
pytest.importorskip("gradio_client")

from manga_ar.config import load_config
from manga_ar.io.writer import write_image
from manga_ar.schemas import PageDocument
from tests.e2e.fakes import fake_stages, page

SETTINGS = ["auto", "balanced", "auto", ["tm"], "NotoNaskhArabic", "western", "skip", False, "png"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def gui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Any, str]]:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # the client must never reach the Hub
    from manga_ar.ui.gradio_app import launch
    from manga_ar.ui.handlers import GuiController

    cfg = load_config(environ={})
    controller = GuiController(cfg, tmp_path / "ws")
    monkeypatch.setattr(controller, "_stages_for", lambda c: fake_stages(c))
    port = _free_port()
    app, _ = launch(cfg, port=port, open_browser=False, block=False, controller=controller)
    try:
        yield controller, f"http://127.0.0.1:{port}/"
    finally:
        app.close()
        controller.cleanup()


def test_gui_translate_edit_rerender(gui: tuple[Any, str], tmp_path: Path) -> None:
    from gradio_client import Client, handle_file

    _controller, url = gui
    assert url.startswith("http://127.0.0.1:")  # never bound to a public interface
    upload = tmp_path / "p1.png"
    write_image(upload, page(), "png")
    client = Client(url, verbose=False)
    summary, pick, _edit, bundle, pair, report_md = client.predict(
        [handle_file(str(upload))], None, *SETTINGS, None, None, api_name="/translate"
    )
    assert "ok: 1" in summary and Path(bundle).name == "mangaar_output.zip"
    assert "MangaAR batch report" in report_md
    assert len(pair) == 2 and all(Path(p).is_file() for p in pair)
    sidecar = pick["value"]
    table, original, result, _ = client.predict(sidecar, api_name="/load_page")
    assert Path(result).is_file()
    rows = table["data"] if isinstance(table, dict) else table
    assert rows[0][3] == "مرحبا" and Path(original).is_file()
    rows[0][3] = "تم التعديل"
    headers = table["headers"] if isinstance(table, dict) else None
    payload = {"headers": headers, "data": rows} if headers else rows
    new_table, new_result, status = client.predict(sidecar, payload, api_name="/rerender")
    assert "Re-rendered" in status and Path(new_result).is_file()
    new_rows = new_table["data"] if isinstance(new_table, dict) else new_table
    assert new_rows[0][3] == "تم التعديل"
    doc = PageDocument.load(Path(sidecar))
    assert doc.regions[0].override.text == "تم التعديل"
    layout = doc.regions[0].layout
    assert layout is not None and " ".join(layout.lines) == "تم التعديل"
    assert "overall" in client.predict(api_name="/diagnostics")
    assert os.environ.get("GRADIO_ANALYTICS_ENABLED") == "False"
