"""Source-text pools per split (meaning-level split assignment by hash, no leakage)."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from benchmarks.generators import GENERATOR_VERSION

ROOT = Path(__file__).resolve().parents[2]
TEXT_DIR = ROOT / "benchmarks" / "data" / "text"
SPLITS = ("dev", "val", "test")


@lru_cache(maxsize=1)
def bank() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((TEXT_DIR / "phrases.yaml").read_text(encoding="utf-8"))
    return data


def _rank(keys: list[str]) -> list[str]:
    return sorted(
        keys, key=lambda k: hashlib.sha256(f"{GENERATOR_VERSION}:{k}".encode()).hexdigest()
    )


def split_of_keys(keys: list[str]) -> dict[str, str]:
    """Deterministic, balanced split: the hash-ranked keys are cut into thirds."""
    ranked = _rank(keys)
    n = len(ranked)
    out = {}
    for i, key in enumerate(ranked):
        out[key] = SPLITS[min(2, i * 3 // n)]
    return out


class TextPools:
    def __init__(self, split: str) -> None:
        data = bank()
        self.split = split
        assignment = split_of_keys(list(data["phrases"]))
        self.ids = sorted(k for k, s in assignment.items() if s == split)
        self.phrases: dict[str, dict[str, str]] = data["phrases"]
        self.sfx_words: dict[str, list[tuple[str, str]]] = {}
        for lang, words in data["sfx"].items():
            keys = [f"sfx-{lang}-{i:02d}" for i in range(len(words))]
            sfx_split = split_of_keys(keys)
            self.sfx_words[lang] = [(k, w) for k, w in zip(keys, words, strict=True)
                                    if sfx_split[k] == split]  # fmt: skip

    def dialogue(self, rng: np.random.Generator, lang: str) -> tuple[str, str]:
        key = self.ids[int(rng.integers(0, len(self.ids)))]
        return key, self.phrases[key][lang]

    def sfx(self, rng: np.random.Generator, lang: str) -> tuple[str, str]:
        words = self.sfx_words[lang]
        return words[int(rng.integers(0, len(words)))]
