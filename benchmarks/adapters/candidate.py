"""Adapter for the current source tree (v0.2.0 candidate).

The candidate keeps v0.1.0's stage entry points, so it reuses the v0.1.0 adapter. The
one difference so far: under ``engine.profile: v2`` English is a source language the
pipeline can be told explicitly (the English OCR route, step 1.0.5). Under ``legacy`` it
goes through v0.1.0's ``auto`` path, as the frozen baseline did, so that legacy runs stay
byte-identical.
"""

from __future__ import annotations

from typing import Any

from benchmarks.adapters import v010


class Engine(v010.Engine):
    def __init__(self, overrides: dict[str, Any], tm_pairs: dict[str, str] | None = None) -> None:
        super().__init__(overrides, tm_pairs)
        if self.cfg.engine.profile == "v2":
            self.source_langs = v010.SOURCE_LANGS | {"en"}
