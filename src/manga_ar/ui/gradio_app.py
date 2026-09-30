"""Local Gradio GUI: Translate, Review & Edit, Settings, Diagnostics.

Bound to 127.0.0.1 with ``share=False``; analytics are disabled before Gradio is imported
(no telemetry). All events share one queue worker (``default_concurrency_limit=1``)
because several ML runtimes are not thread-safe; only Cancel bypasses the queue.
Callbacks just translate widget values for :class:`~manga_ar.ui.handlers.GuiController`.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import Any

# No telemetry: analytics must be disabled before Gradio is imported.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import gradio as gr

from manga_ar import __version__
from manga_ar.config import AppConfig
from manga_ar.errors import MangaArError
from manga_ar.logging_setup import get_logger
from manga_ar.ui.handlers import COLUMNS, EDITABLE, GuiController, GuiSettings

log = get_logger(__name__)
PROVIDERS = ["tm", "google", "mymemory", "libretranslate", "local"]


def _settings(*values: Any) -> GuiSettings:
    (source, preset, device, providers, font, digits, sfx, erase, fmt, glossary, tm) = values
    return GuiSettings(
        source=source,
        preset=preset,
        device=device,
        providers=list(providers or []),
        font=font or "",
        digits=digits,
        sfx=sfx,
        erase_untranslated=bool(erase),
        output_format=fmt,
        glossary=_path(glossary),
        tm=_path(tm),
    )


def _path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return str(value.get("path") or value.get("name") or "") or None
    return str(getattr(value, "name", value))


def _files(value: Any) -> list[str]:
    if not value:
        return []
    items = value if isinstance(value, list) else [value]
    return [p for p in (_path(v) for v in items) if p]


def build_app(controller: GuiController) -> gr.Blocks:
    cfg = controller.base_cfg
    fonts = controller.font_choices()
    # delete_cache: files Gradio copies for display are removed hourly and on shutdown.
    with gr.Blocks(title="MangaAR", analytics_enabled=False, delete_cache=(3600, 3600)) as app:
        gr.Markdown(
            f"# MangaAR {__version__}\n"
            "Translate manga, manhwa, manhua and comic pages into Arabic, locally and for "
            "free. Translate only content you have the right to translate."
        )
        with gr.Tab("Translate"):
            uploads = gr.File(
                label="Pages (images) or CBZ/ZIP archives",
                file_count="multiple",
                file_types=["image", ".cbz", ".zip"],
            )
            folder = gr.File(label="…or a whole folder", file_count="directory")
            with gr.Row():
                run_btn = gr.Button("Translate", variant="primary")
                cancel_btn = gr.Button("Cancel", variant="stop")
            status = gr.Markdown()
            with gr.Row():
                page_pick = gr.Dropdown(label="Page", choices=[], interactive=True)
                bundle = gr.File(label="Download results (ZIP)", interactive=False)
            slider = gr.ImageSlider(label="Before / after", type="filepath")
            report_md = gr.Markdown()
        with gr.Tab("Review & Edit"):
            gr.Markdown(
                "Edit the Arabic text, font or size of any region (or tick *skip*), then "
                "**Re-render page**. Only typesetting runs again."
            )
            edit_pick = gr.Dropdown(label="Page", choices=[], interactive=True)
            table = gr.Dataframe(
                headers=COLUMNS,
                datatype=["str", "str", "str", "str", "str", "str", "bool", "str"],
                type="array",
                interactive=True,
                static_columns=[i for i in range(len(COLUMNS)) if i not in EDITABLE],
                wrap=True,
            )
            with gr.Row():
                rerender_btn = gr.Button("Re-render page", variant="primary")
                region_box = gr.Textbox(label="Region id", placeholder="e.g. page01-r3")
                retranslate_btn = gr.Button("Re-translate region")
            edit_status = gr.Markdown()
            with gr.Row():
                original_img = gr.Image(label="Original", type="filepath", interactive=False)
                result_img = gr.Image(label="Result", type="filepath", interactive=False)
        with gr.Tab("Settings"):
            with gr.Row():
                source = gr.Dropdown(["auto", "ja", "ko", "zh"], value="auto", label="Source")
                preset = gr.Dropdown(
                    ["fast", "balanced", "quality"], value=cfg.preset, label="Preset"
                )
                device = gr.Dropdown(
                    ["auto", "cpu", "cuda", "mps"], value=cfg.runtime.device, label="Device"
                )
            providers = gr.CheckboxGroup(
                PROVIDERS, value=list(cfg.translate.providers), label="Translation providers"
            )
            with gr.Row():
                font = gr.Dropdown(fonts, value=cfg.typeset.font, label="Arabic font")
                digits = gr.Dropdown(
                    ["western", "arabic_indic"], value=cfg.typeset.digits, label="Digits"
                )
                sfx = gr.Dropdown(["skip", "translate"], value=cfg.detect.sfx, label="SFX")
                fmt = gr.Dropdown(
                    ["png", "jpg", "webp", "cbz"], value=cfg.output.format, label="Output"
                )
            erase = gr.Checkbox(
                value=cfg.inpaint.erase_untranslated,
                label="Erase text that could not be translated",
            )
            with gr.Row():
                glossary = gr.File(
                    label="Glossary (JSON/YAML/CSV)", file_types=[".json", ".yaml", ".yml", ".csv"]
                )
                tm = gr.File(
                    label="Translation memory (JSON/YAML/CSV)",
                    file_types=[".json", ".yaml", ".yml", ".csv"],
                )
        with gr.Tab("Diagnostics"):
            doctor_btn = gr.Button("Run self-check")
            diag = gr.Textbox(label="Doctor + recent log", lines=28, max_lines=60)

        settings_inputs = [
            source,
            preset,
            device,
            providers,
            font,
            digits,
            sfx,
            erase,
            fmt,
            glossary,
            tm,
        ]

        def on_translate(
            files: Any,
            folder_files: Any,
            *values: Any,
            progress: gr.Progress = gr.Progress(),  # noqa: B008 - Gradio injects progress
        ) -> Any:
            try:
                outcome = controller.translate(
                    _files(files) + _files(folder_files),
                    _settings(*values),
                    progress=lambda f, m: progress(f, desc=m),
                )
            except MangaArError as exc:
                raise gr.Error(str(exc)) from exc
            choices = [(p.label, str(p.sidecar)) for p in outcome.pages]
            first = choices[0][1] if choices else None
            pair = _pair(first)
            return (
                outcome.summary,
                gr.update(choices=choices, value=first),
                gr.update(choices=choices, value=first),
                str(outcome.bundle) if outcome.bundle else None,
                pair,
                outcome.report.to_markdown(),
            )

        def _pair(sidecar: str | None) -> Any:
            if not sidecar:
                return None
            view = controller.page(sidecar)
            return (str(view.original), str(view.result))

        def on_page(sidecar: str | None) -> Any:
            try:
                return _pair(sidecar)
            except MangaArError as exc:
                raise gr.Error(str(exc)) from exc

        def on_edit_page(sidecar: str | None) -> Any:
            if not sidecar:
                return [], None, None, ""
            try:
                view = controller.page(sidecar)
            except MangaArError as exc:
                raise gr.Error(str(exc)) from exc
            return view.rows, str(view.original), str(view.result), ""

        def on_rerender(sidecar: str | None, rows: Any) -> Any:
            if not sidecar:
                raise gr.Error("choose a page first")
            try:
                view = controller.apply_edits(sidecar, _rows(rows))
            except MangaArError as exc:
                raise gr.Error(str(exc)) from exc
            return view.rows, str(view.result), f"Re-rendered ({view.doc.status})."

        def on_retranslate(sidecar: str | None, region_id: str, *values: Any) -> Any:
            if not sidecar or not region_id:
                raise gr.Error("choose a page and enter a region id")
            try:
                view = controller.retranslate(sidecar, region_id, _settings(*values))
            except MangaArError as exc:
                raise gr.Error(str(exc)) from exc
            return view.rows, str(view.result), f"{region_id} re-translated."

        run_btn.click(
            on_translate,
            [uploads, folder, *settings_inputs],
            [status, page_pick, edit_pick, bundle, slider, report_md],
            api_name="translate",
        )
        cancel_btn.click(controller.cancel, None, status, queue=False, api_name="cancel")
        page_pick.change(on_page, page_pick, slider, api_name="preview")
        edit_pick.change(
            on_edit_page,
            edit_pick,
            [table, original_img, result_img, edit_status],
            api_name="load_page",
        )
        rerender_btn.click(
            on_rerender, [edit_pick, table], [table, result_img, edit_status], api_name="rerender"
        )
        retranslate_btn.click(
            on_retranslate,
            [edit_pick, region_box, *settings_inputs],
            [table, result_img, edit_status],
            api_name="retranslate",
        )
        doctor_btn.click(controller.diagnostics, None, diag, api_name="diagnostics")
    app.queue(default_concurrency_limit=1)
    return app


def _rows(value: Any) -> list[list[Any]]:
    if value is None:
        return []
    if hasattr(value, "values") and hasattr(value, "columns"):  # pandas DataFrame
        return [list(r) for r in value.values.tolist()]
    if isinstance(value, dict) and "data" in value:
        return [list(r) for r in value["data"]]
    return [list(r) for r in value]


def launch(
    cfg: AppConfig,
    host: str = "127.0.0.1",
    port: int | None = 7860,
    open_browser: bool = True,
    block: bool = True,
    workspace: Path | None = None,
    controller: GuiController | None = None,
) -> tuple[gr.Blocks, GuiController]:
    """Start the GUI. ``block=False`` returns immediately (tests, embedding)."""
    controller = controller or GuiController(cfg, workspace)
    os.environ.setdefault("GRADIO_TEMP_DIR", str(controller.workspace / "uploads"))
    atexit.register(controller.cleanup)
    if host not in {"127.0.0.1", "localhost", "::1"}:
        log.warning("GUI bound to %s: it is reachable from other machines on the network", host)
    app = build_app(controller)
    app.launch(
        server_name=host,
        server_port=port,
        share=False,
        inbrowser=open_browser,
        prevent_thread_lock=not block,
        allowed_paths=[str(controller.workspace)],
        footer_links=[],
        quiet=not block,
    )
    return app, controller
