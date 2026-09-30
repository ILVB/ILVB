"""Command-line interface: ``manga-arabic`` (also ``python -m manga_ar``)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from manga_ar import __version__
from manga_ar.config import AppConfig, load_config, resolve_cache_dir
from manga_ar.errors import MangaArError
from manga_ar.logging_setup import configure_logging, ensure_utf8_streams

EXIT_OK, EXIT_FATAL, EXIT_PARTIAL = 0, 1, 2


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="FILE",
        help="extra YAML config file (repeatable, later wins)",
    )
    parser.add_argument(
        "--offline", action="store_true", default=None, help="never use the network"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manga-arabic",
        description="Translate Manga/Manhwa/Manhua/comic pages to Arabic (free, offline-capable).",
    )
    parser.add_argument("--version", action="version", version=f"manga-arabic {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="environment self-check")
    _common(p)
    p.add_argument("--no-network", action="store_true", help="skip reachability checks")

    p = sub.add_parser("models", help="list, download or verify models")
    _common(p)
    p.add_argument("action", choices=["list", "download", "verify"])
    p.add_argument("names", nargs="*", help="model names or groups (default: all / 'default')")

    p = sub.add_parser("fonts", help="list fonts or check their glyph coverage")
    _common(p)
    p.add_argument("action", choices=["list", "check"])

    p = sub.add_parser("translate", help="translate pages, folders or CBZ/ZIP archives")
    _common(p)
    p.add_argument("inputs", nargs="+", type=Path, help="images, folders or .cbz/.zip files")
    p.add_argument("-o", "--output", type=Path, required=True, help="output directory")
    p.add_argument("--source", choices=["auto", "ja", "ko", "zh"], default=None)
    p.add_argument("--preset", choices=["fast", "balanced", "quality"], default=None)
    p.add_argument(
        "--reading-order", choices=["auto", "manga_rtl", "comic_ltr", "webtoon_ttb"], default=None
    )
    p.add_argument(
        "--providers", type=_csv, default=None, help="comma list, e.g. tm,google,mymemory,local"
    )
    p.add_argument("--glossary", type=Path, default=None, help="glossary JSON/YAML/CSV")
    p.add_argument("--tm", type=Path, default=None, help="translation memory JSON/YAML/CSV")
    p.add_argument("--detector", choices=["auto", "classical", "hybrid", "rapid", "craft", "ctd"])
    _typeset_flags(p)
    p.add_argument("--sfx", choices=["skip", "translate"], default=None)
    p.add_argument("--format", choices=["png", "jpg", "webp", "cbz"], default=None)
    p.add_argument("--resume", action="store_true", default=None, help="skip finished pages")
    p.add_argument("--force", action="store_true", default=None, help="reprocess everything")
    p.add_argument("--debug", action="store_true", default=None, help="write debug artifacts")
    p.add_argument("--quiet", action="store_true", help="no progress bar")

    p = sub.add_parser("rerender", help="re-typeset pages from their sidecar JSON")
    _common(p)
    p.add_argument("sidecars", nargs="+", type=Path, help="<page>_ar.mangaar.json files")
    _typeset_flags(p)

    p = sub.add_parser("demo", help="generate a synthetic page and run the whole pipeline")
    _common(p)
    p.add_argument("-o", "--output", type=Path, default=Path("mangaar-demo"))

    p = sub.add_parser("gui", help="start the local web GUI (127.0.0.1 only by default)")
    _common(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    return parser


def _csv(value: str) -> list[str]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return items


def _typeset_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--font", default=None, help="Arabic font key (see `fonts list`)")
    p.add_argument("--digits", choices=["western", "arabic_indic"], default=None)
    p.add_argument(
        "--erase-untranslated",
        action="store_true",
        default=None,
        help="erase text that could not be translated instead of keeping the original",
    )


def _typeset_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "typeset.font": args.font,
        "typeset.digits": args.digits,
        "inpaint.erase_untranslated": args.erase_untranslated,
    }


def config_from_args(args: argparse.Namespace, extra: dict[str, Any] | None = None) -> AppConfig:
    overrides: dict[str, Any] = {
        "runtime.offline": getattr(args, "offline", None),
        "runtime.device": getattr(args, "device", None),
        "runtime.log_level": getattr(args, "log_level", None),
    }
    overrides.update(extra or {})
    return load_config(files=[Path(f) for f in getattr(args, "config", [])], overrides=overrides)


def cmd_doctor(args: argparse.Namespace) -> int:
    from manga_ar.doctor import run_doctor

    cfg = config_from_args(args)
    report = run_doctor(cfg, check_network=not args.no_network)
    sys.stdout.write(report.render() + "\n")
    return EXIT_OK if report.ok else EXIT_FATAL


def cmd_models(args: argparse.Namespace) -> int:
    from manga_ar.models.manager import ModelManager
    from manga_ar.models.registry import GROUPS, REGISTRY

    cfg = config_from_args(args)
    manager = ModelManager(resolve_cache_dir(cfg), offline=cfg.runtime.offline)
    if args.action == "list":
        for row in manager.status():
            flag = " (copyleft/NC: opt-in)" if row["copyleft_or_nc"] else ""
            state = "present" if row["present"] else "missing"
            sys.stdout.write(
                f"{row['name']:14s} {state:8s} {row['license']}{flag}\n"
                f"{'':14s} {row['description']}\n"
            )
        sys.stdout.write(
            "groups: " + ", ".join(f"{g}={'+'.join(m)}" for g, m in GROUPS.items()) + "\n"
        )
        return EXIT_OK
    names = args.names or (["default"] if args.action == "download" else list(REGISTRY))
    failures = 0
    for name in names:
        targets = GROUPS.get(name, (name,))
        for target in targets:
            try:
                if args.action == "download":
                    if REGISTRY.get(target) and REGISTRY[target].kind == "easyocr":
                        from manga_ar.ocr.easyocr_engine import download_easyocr_models

                        download_easyocr_models(manager)
                    path = manager.ensure(target)
                    sys.stdout.write(f"ok {target} → {path}\n")
                else:
                    ok, detail = manager.verify(target)
                    failures += 0 if ok else 1
                    sys.stdout.write(f"{'ok' if ok else 'FAIL'} {target}: {detail}\n")
            except MangaArError as exc:
                failures += 1
                sys.stdout.write(f"FAIL {target}: {exc}\n")
    return EXIT_OK if failures == 0 else EXIT_FATAL


def cmd_fonts(args: argparse.Namespace) -> int:
    from manga_ar.typeset.fonts import FontRegistry

    cfg = config_from_args(args)
    reg = FontRegistry()
    bad = 0
    for key, info in reg.fonts.items():
        if args.action == "list":
            sys.stdout.write(f"{key:20s} {info.source:8s} {info.license:10s} {info.path}\n")
            continue
        if info.role == "symbol":
            sys.stdout.write(f"{key:20s} symbol font, {len(info.cmap)} glyphs\n")
            continue
        cov = info.coverage
        assert cov is not None
        status = "BASIC+RAQM" if cov.basic_ok else "RAQM-only"
        details = {
            "missing_contextual": len(cov.contextual),
            "isolated_via_base_letters": bool(cov.isolated),
            "missing_punct": [f"U+{c:04X}" for c in cov.punctuation],
            "missing_indic_digits": len(cov.indic_digits),
        }
        sys.stdout.write(f"{key:20s} {status:10s} {json.dumps(details, ensure_ascii=False)}\n")
        if key == cfg.typeset.font and not cov.basic_ok:
            bad += 1
    return EXIT_OK if bad == 0 else EXIT_FATAL


def _progress_bar(enabled: bool) -> tuple[Any, Any]:
    if not enabled:
        return None, None
    from tqdm import tqdm

    bar = tqdm(total=1000, unit="‰", bar_format="{l_bar}{bar}| {desc}", file=sys.stderr)

    def update(fraction: float, message: str) -> None:
        bar.n = round(fraction * 1000)
        bar.set_description_str(message[-60:])
        bar.refresh()

    return bar, update


def cmd_translate(args: argparse.Namespace) -> int:
    from manga_ar.pipeline import Pipeline

    extra: dict[str, Any] = {
        "preset": args.preset,
        "input.source_lang": args.source,
        "input.reading_order": args.reading_order,
        "translate.providers": args.providers,
        "translate.glossary_file": str(args.glossary) if args.glossary else None,
        "translate.tm_file": str(args.tm) if args.tm else None,
        "detect.detector": args.detector,
        "detect.sfx": args.sfx,
        "output.format": args.format,
        "output.resume": args.resume,
        "output.force": args.force,
        "runtime.debug": args.debug,
        **_typeset_overrides(args),
    }
    cfg = config_from_args(args, extra)
    bar, update = _progress_bar(not args.quiet and sys.stderr.isatty())
    try:
        report = Pipeline(cfg, progress=update).run(args.inputs, args.output)
    finally:
        if bar is not None:
            bar.close()
    counts = ", ".join(f"{k}: {v}" for k, v in sorted(report.counts.items()))
    sys.stdout.write(f"{len(report.pages)} page(s) — {counts}\n")
    for page in report.pages:
        if not page.succeeded:
            sys.stdout.write(f"  {page.status.upper()} {page.name}: {page.error or ''}\n")
    for note in report.notes:
        sys.stdout.write(f"note: {note}\n")
    sys.stdout.write(f"report: {args.output / 'report.md'}\n")
    return report.exit_code


def cmd_rerender(args: argparse.Namespace) -> int:
    from manga_ar.pipeline import rerender
    from manga_ar.typeset.fonts import FontRegistry

    cfg = config_from_args(args)
    registry = FontRegistry()
    ok = 0
    for sidecar in args.sidecars:
        try:
            result = rerender(sidecar, cfg, _typeset_overrides(args), registry)
        except MangaArError as exc:
            sys.stdout.write(f"FAIL {sidecar}: {exc}\n")
            continue
        ok += 1
        sys.stdout.write(f"ok {sidecar} → {result.output}\n")
    if ok == len(args.sidecars):
        return EXIT_OK
    return EXIT_PARTIAL if ok else EXIT_FATAL


def cmd_demo(args: argparse.Namespace) -> int:
    from manga_ar.demo import run_demo

    cfg = config_from_args(args)
    result = run_demo(cfg, args.output)
    sys.stdout.write(f"input:  {result.input_path}\n")
    sys.stdout.write(f"output: {result.output_path}\n")
    sys.stdout.write(f"regions translated: {result.translated}/{result.expected}\n")
    if result.exit_code == EXIT_OK:
        sys.stdout.write("demo OK: open the output image to see the Arabic lettering\n")
    else:
        sys.stdout.write(f"demo incomplete; see {args.output / 'output' / 'report.md'}\n")
    return result.exit_code


def cmd_gui(args: argparse.Namespace) -> int:
    from manga_ar.ui.gradio_app import launch

    cfg = config_from_args(args)
    launch(cfg, host=args.host, port=args.port, open_browser=not args.no_browser)
    return EXIT_OK


COMMANDS = {
    "doctor": cmd_doctor,
    "models": cmd_models,
    "fonts": cmd_fonts,
    "translate": cmd_translate,
    "rerender": cmd_rerender,
    "demo": cmd_demo,
    "gui": cmd_gui,
}


def main(argv: Sequence[str] | None = None) -> int:
    ensure_utf8_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        level = getattr(args, "log_level", None) or "INFO"
        configure_logging(level)
        return COMMANDS[args.command](args)
    except MangaArError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return EXIT_FATAL
    except KeyboardInterrupt:
        sys.stderr.write("interrupted\n")
        return EXIT_FATAL
