"""Fetch the heavy model weights this host cannot reach (ADR-0007, item 2).

Usage (on a host with access to huggingface.co and download.pytorch.org):
    python scripts/download_heavy_weights.py [--only nllb,llm,embed,lpips] [--check]

Everything goes into the model cache ($MANGAAR_CACHE_DIR/models, else the platformdirs
cache). Nothing is committed except `benchmarks/heavy_weights.lock.json`.
- First run: the SHA-256 of every downloaded file is written to the lock. Commit the lock:
  from then on the weights are pinned.
- Later runs and `--check` verify every file against the lock and fail on any mismatch.

- nllb: facebook/nllb-200-distilled-600M, converted locally to CTranslate2 int8
  (CC-BY-NC-4.0: non-commercial, opt-in only)
- llm: Qwen/Qwen2.5-7B-Instruct-GGUF, Q4_K_M (Apache-2.0)
- embed: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0)
- lpips: torchvision AlexNet ImageNet weights (BSD-3-Clause); the lpips linear layers ship
  inside the lpips wheel

Downloads total about 8 GB (NLLB about 2.5 GB before conversion, GGUF about 4.7 GB).
Conversion needs `ctranslate2`, `transformers` and `sentencepiece`; LPIPS needs `torchvision`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "benchmarks" / "heavy_weights.lock.json"


@dataclass(frozen=True)
class Weight:
    key: str
    repo: str
    license: str
    patterns: tuple[str, ...] = ()


WEIGHTS = {
    "nllb": Weight(
        "nllb",
        "facebook/nllb-200-distilled-600M",
        "CC-BY-NC-4.0",
        ("*.json", "*.model", "pytorch_model.bin"),
    ),
    "llm": Weight("llm", "Qwen/Qwen2.5-7B-Instruct-GGUF", "Apache-2.0", ("*q4_k_m*.gguf",)),
    "embed": Weight(
        "embed",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "Apache-2.0",
        ("*.json", "*.txt", "*.model", "model.safetensors", "1_Pooling/*"),
    ),
    "lpips": Weight("lpips", "torchvision:alexnet:IMAGENET1K_V1", "BSD-3-Clause"),
}


def cache_root() -> Path:
    env = os.environ.get("MANGAAR_CACHE_DIR")
    if env:
        return Path(env).expanduser() / "models"
    from platformdirs import user_cache_dir

    return Path(user_cache_dir("manga-arabic", appauthor=False)) / "models"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hf(weight: Weight, dest: Path) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download

    revision = HfApi().model_info(weight.repo).sha  # pin the exact commit that was fetched
    snapshot_download(weight.repo, revision=revision, local_dir=dest,
                      allow_patterns=list(weight.patterns))  # fmt: skip
    return {"repo": weight.repo, "revision": revision}


def fetch_nllb(dest: Path) -> dict[str, Any]:
    src = dest.parent / "nllb-hf"
    meta = _hf(WEIGHTS["nllb"], src)
    from ctranslate2.converters import TransformersConverter

    TransformersConverter(str(src)).convert(str(dest), quantization="int8", force=True)
    return {**meta, "converted": "ctranslate2 int8"}


def fetch_lpips(dest: Path) -> dict[str, Any]:
    os.environ["TORCH_HOME"] = str(dest)
    from torchvision.models import AlexNet_Weights, alexnet

    alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)  # torchvision checks the hash prefix
    return {"url": AlexNet_Weights.IMAGENET1K_V1.url}


FETCHERS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "nllb": fetch_nllb,
    "llm": lambda d: _hf(WEIGHTS["llm"], d),
    "embed": lambda d: _hf(WEIGHTS["embed"], d),
    "lpips": fetch_lpips,
}


def hash_tree(dest: Path) -> dict[str, str]:
    return {p.relative_to(dest).as_posix(): sha256(p)
            for p in sorted(dest.rglob("*")) if p.is_file()}  # fmt: skip


def verify(key: str, dest: Path, entry: dict[str, Any]) -> list[str]:
    found = hash_tree(dest) if dest.is_dir() else {}
    problems = [f"{key}/{name}: missing" for name in entry["files"] if name not in found]
    problems += [f"{key}/{name}: SHA-256 mismatch" for name, digest in entry["files"].items()
                 if name in found and found[name] != digest]  # fmt: skip
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=",".join(WEIGHTS), help="comma-separated keys")
    ap.add_argument("--check", action="store_true", help="verify only, never download")
    args = ap.parse_args(argv)
    keys = [k for k in args.only.split(",") if k]
    unknown = set(keys) - set(WEIGHTS)
    if unknown:
        ap.error(f"unknown keys {sorted(unknown)}")
    lock: dict[str, Any] = json.loads(LOCK.read_text("utf-8")) if LOCK.is_file() else {}
    root, problems, recorded = cache_root(), [], []
    for key in keys:
        dest = root / f"heavy-{key}"
        if key in lock:
            if not args.check and not dest.is_dir():
                FETCHERS[key](dest)
            problems += verify(key, dest, lock[key])
            continue
        if args.check:
            problems.append(f"{key}: not in {LOCK.name} (run without --check first)")
            continue
        meta = FETCHERS[key](dest)
        lock[key] = {**meta, "license": WEIGHTS[key].license, "files": hash_tree(dest)}
        recorded.append(key)
    if recorded:
        LOCK.write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        sys.stdout.write(f"recorded {recorded} in {LOCK}: commit it to pin these weights\n")
    for p in problems:
        sys.stdout.write(f"FAIL {p}\n")
    sys.stdout.write("OK\n" if not problems else f"{len(problems)} problem(s)\n")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
