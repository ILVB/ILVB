"""Glossary: protect terms (names, honorifics, places) with placeholders that survive
translation, then substitute the configured Arabic rendering (S5)."""

from __future__ import annotations

import re
from pathlib import Path

from manga_ar.translate.tm import load_pairs

_TOKEN = re.compile(r"ZQX(\d+)X", re.IGNORECASE)


class Glossary:
    def __init__(self, pairs: dict[str, str] | None = None) -> None:
        # longest terms first so "先輩さん" wins over "さん"
        self.pairs = dict(sorted((pairs or {}).items(), key=lambda kv: -len(kv[0])))

    @classmethod
    def from_file(cls, path: Path) -> Glossary:
        return cls(load_pairs(path))

    def __bool__(self) -> bool:
        return bool(self.pairs)

    def protect(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace glossary terms by placeholder tokens (ZQX1X, ZQX2X, …)."""
        mapping: dict[str, str] = {}
        out = text
        for term, arabic in self.pairs.items():
            if term and term in out:
                token = f"ZQX{len(mapping) + 1}X"
                mapping[token] = arabic
                out = out.replace(term, f" {token} ")
        return " ".join(out.split()) if mapping else text, mapping

    @staticmethod
    def restore(translated: str, mapping: dict[str, str]) -> tuple[str, bool]:
        """Substitute placeholders; the flag says whether every placeholder survived."""
        if not mapping:
            return translated, True
        found = {m.group(0).upper() for m in _TOKEN.finditer(translated)}
        survived = set(mapping) <= found

        def sub(m: re.Match[str]) -> str:
            return mapping.get(m.group(0).upper(), "")

        return " ".join(_TOKEN.sub(sub, translated).split()), survived
