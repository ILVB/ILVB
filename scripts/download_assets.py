"""Download (and SHA-256 verify) the OFL CJK fonts used by synthetic fixtures and the demo.

Fonts land in ``<cache>/models/fonts`` where ``<cache>`` is ``$MANGAAR_CACHE_DIR`` or
``<project>/.cache``. They are never committed. Optional: ``--models`` also fetches the
default model group (EasyOCR + LaMa).

Usage: python scripts/download_assets.py [--models]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from manga_ar.errors import ModelUnavailableError
from manga_ar.models.manager import ModelManager

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", action="store_true", help="also fetch EasyOCR + LaMa")
    args = parser.parse_args()
    cache = Path(os.environ.get("MANGAAR_CACHE_DIR", ROOT / ".cache"))
    manager = ModelManager(cache)
    groups = ["fonts"] + (["default"] if args.models else [])
    ok = True
    for group in groups:
        try:
            for path in manager.ensure_group(group):
                print(f"ok {path}")
        except ModelUnavailableError as exc:
            print(f"FAILED {group}: {exc}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
