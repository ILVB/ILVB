"""Deterministic synthetic manga pages with exact ground truth (step 0.3.2).

Self-contained on purpose: nothing here imports ``manga_ar``, so product changes can never
move the yardstick. Output is pinned by SHA-256 in the dataset manifest.
"""

GENERATOR_VERSION = "synthetic_v1"
