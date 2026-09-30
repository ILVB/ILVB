"""Download OFL CJK fonts used to generate synthetic test fixtures.

The fonts are *not* committed. They land in ``<cache>/fonts`` where ``<cache>`` is
``$MANGAAR_CACHE_DIR`` or ``<project>/.cache``. Each file is verified against a pinned
SHA-256. If a download is impossible (offline, blocked host), the fixture generator falls
back to system CJK fonts (see ``tests/fixtures/synth.py``).

Usage: python scripts/download_assets.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/SubsetOTF"
# name -> (url path, sha256). Licence: SIL OFL 1.1 (notofonts/noto-cjk).
FONTS: dict[str, tuple[str, str]] = {
    "NotoSansJP-Regular.otf": (
        "JP/NotoSansJP-Regular.otf",
        "dff723ba59d57d136764a04b9b2d03205544f7cd785a711442d6d2d085ac5073",
    ),
    "NotoSansKR-Regular.otf": (
        "KR/NotoSansKR-Regular.otf",
        "69975a0ac8472717870aefeab0a4d52739308d90856b9955313b2ad5e0148d68",
    ),
    "NotoSansSC-Regular.otf": (
        "SC/NotoSansSC-Regular.otf",
        "faa6c9df652116dde789d351359f3d7e5d2285a2b2a1f04a2d7244df706d5ea9",
    ),
}


def cache_dir() -> Path:
    env = os.environ.get("MANGAAR_CACHE_DIR")
    return Path(env) if env else ROOT / ".cache"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, rel: str, expected: str, dest_dir: Path, force: bool) -> bool:
    dest = dest_dir / name
    if dest.exists() and not force and sha256(dest) == expected:
        print(f"ok (cached) {name}")
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(f"{BASE}/{rel}", stream=True, timeout=(5, 60)) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    fh.write(chunk)
    except requests.RequestException as exc:
        print(f"FAILED {name}: {exc}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False
    digest = sha256(tmp)
    if digest != expected:
        print(f"FAILED {name}: sha256 mismatch {digest}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dest)
    print(f"ok {name} sha256={digest}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()
    dest_dir = cache_dir() / "fonts"
    dest_dir.mkdir(parents=True, exist_ok=True)
    results = [fetch(n, rel, h, dest_dir, args.force) for n, (rel, h) in FONTS.items()]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
