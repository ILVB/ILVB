"""Translation memory: exact matches of normalised source text → Arabic (offline)."""

from __future__ import annotations

import csv
import io
import json
import unicodedata
from pathlib import Path

import yaml

from manga_ar.errors import ConfigError, ProviderError


def tm_key(text: str) -> str:
    """Normalisation used for TM lookups: NFKC, no whitespace."""
    return "".join(unicodedata.normalize("NFKC", text).split())


def load_pairs(path: Path) -> dict[str, str]:
    """Read ``source → target`` pairs from JSON (object), YAML (mapping) or CSV (2 columns)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            data = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(text) or {}
        elif suffix == ".csv":
            data = {
                row[0]: row[1]
                for row in csv.reader(io.StringIO(text))
                if len(row) >= 2 and not row[0].startswith("#")
            }
        else:
            raise ConfigError(f"{path}: unsupported format (use .json, .yaml or .csv)")
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"{path}: invalid file ({exc})") from exc
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise ConfigError(f"{path}: expected a mapping of strings")
    return data


class TranslationMemory:
    name = "tm"
    network = False

    def __init__(self, pairs: dict[str, str] | None = None) -> None:
        self.entries = {tm_key(k): v for k, v in (pairs or {}).items()}

    @classmethod
    def from_file(cls, path: Path) -> TranslationMemory:
        return cls(load_pairs(path))

    def available(self) -> bool:
        return bool(self.entries)

    def supports(self, src: str, tgt: str) -> bool:
        return tgt == "ar"

    def lookup(self, text: str) -> str | None:
        return self.entries.get(tm_key(text))

    def translate(self, text: str, src: str, tgt: str) -> str:
        hit = self.lookup(text)
        if hit is None:
            raise ProviderError("tm: no exact match", retryable=False)
        return hit
