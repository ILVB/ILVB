"""Adapter for the current source tree (v0.2.0 candidate).

Until Phase 1 changes the public pipeline API, the candidate exposes the same stage
entry points as v0.1.0, so the v0.1.0 adapter implementation applies unchanged. When the
API diverges, the candidate-specific implementation lives here (the baseline adapter in
`v010.py` never changes).
"""

from __future__ import annotations

from benchmarks.adapters.v010 import Engine

__all__ = ["Engine"]
