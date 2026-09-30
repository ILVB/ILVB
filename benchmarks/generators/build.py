"""Build the synthetic benchmark and its manifest: ``python -m benchmarks.generators.build``.

--check verifies existing files against the committed manifest (no regeneration). The test
split is written and hashed without being read back or printed (sealed at creation).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from benchmarks.datasets import DATA_DIR, MANIFEST_DIR, sha256
from benchmarks.generators import GENERATOR_VERSION
from benchmarks.generators.fonts import FONTS
from benchmarks.generators.page import CATEGORIES, make_page
from benchmarks.generators.texts import TextPools, silver_refs

PAGES_PER_CATEGORY = {"dev": 4, "val": 4, "test": 5}
LANGS = ("ja", "zh", "ko", "en")
MIN_REGIONS = 400


def _seed(split: str, category: str, idx: int) -> int:
    digest = hashlib.sha256(f"{GENERATOR_VERSION}:{split}:{category}:{idx}".encode()).hexdigest()
    return int(digest[:8], 16)


def _png(rgb: Any) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG", compress_level=6)
    return buf.getvalue()


def _write(path: Path, data: bytes) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": path.name, "sha256": hashlib.sha256(data).hexdigest()}


def build(splits: tuple[str, ...]) -> dict[str, Any]:
    refs = silver_refs()
    out: dict[str, Any] = {
        "name": GENERATOR_VERSION,
        "fonts": {k: v.sha256 for k, v in FONTS.items()},
        "splits": {},
    }
    for split in splits:
        pools = TextPools(split)
        root = DATA_DIR / GENERATOR_VERSION / split
        pages: list[dict[str, Any]] = []
        regions = 0
        for category in CATEGORIES:
            for idx in range(PAGES_PER_CATEGORY[split]):
                lang = "ja" if category == "vertical_ja" else LANGS[idx % len(LANGS)]
                pid = f"{split}-{category}-{idx:02d}"
                img, clean, gt = make_page(pid, category, lang, split, _seed(split, category, idx),
                                           pools, refs)  # fmt: skip
                gt["image"], gt["clean"] = f"{pid}.png", f"{pid}.clean.png"
                blob = json.dumps(gt, ensure_ascii=False, sort_keys=True).encode("utf-8")
                files = {
                    "image": _write(root / gt["image"], _png(img)),
                    "clean": _write(root / gt["clean"], _png(clean)),
                    "gt": _write(root / f"{pid}.gt.json", blob),
                }
                regions += len(gt["regions"])
                pages.append({"page_id": pid, "category": category, "lang": lang,
                              "regions": len(gt["regions"]), "files": files})  # fmt: skip
        joined = "".join(f["sha256"] for p in pages for f in p["files"].values())
        out["splits"][split] = {"pages": pages, "regions": regions,
                                "sha256": hashlib.sha256(joined.encode()).hexdigest()}  # fmt: skip
    return out


def check(name: str = GENERATOR_VERSION) -> list[str]:
    man = json.loads((MANIFEST_DIR / f"{name}.json").read_text(encoding="utf-8"))
    problems = []
    for split, data in man["splits"].items():
        for page in data["pages"]:
            for f in page["files"].values():
                path = DATA_DIR / name / split / f["path"]
                if not path.is_file() or sha256(path) != f["sha256"]:
                    problems.append(f"{split}/{f['path']}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify files against the manifest")
    ap.add_argument("--write-manifest", action="store_true", help="(re)write the manifest")
    args = ap.parse_args(argv)
    if args.check:
        problems = check()
        sys.stdout.write("OK\n" if not problems else "\n".join(problems) + "\n")
        return 0 if not problems else 1
    man = build(("dev", "val", "test"))
    total = sum(s["regions"] for s in man["splits"].values())
    if total < MIN_REGIONS:
        sys.stderr.write(f"only {total} regions (< {MIN_REGIONS})\n")
        return 1
    target = MANIFEST_DIR / f"{GENERATOR_VERSION}.json"
    if args.write_manifest or not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    elif json.loads(target.read_text(encoding="utf-8")) != man:
        sys.stderr.write("regenerated files differ from the committed manifest\n")
        return 1
    counts = {s: (len(d["pages"]), d["regions"]) for s, d in man["splits"].items()}
    sys.stdout.write(f"{GENERATOR_VERSION}: (pages, regions) per split {counts}; total {total}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
