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
    return parser


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


COMMANDS = {"doctor": cmd_doctor, "models": cmd_models, "fonts": cmd_fonts}


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
