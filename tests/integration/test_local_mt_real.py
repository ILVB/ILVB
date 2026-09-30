"""Offline translation through the real local model (``-m integration``).

Needs the OPUS-MT weights once (``manga-arabic models download local-mt``); the network
policy of the build host blocks huggingface.co (DECISIONS W-004), so this skips there.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from manga_ar.errors import ModelUnavailableError, ProviderError
from manga_ar.models.manager import ModelManager
from manga_ar.translate.local_mt import LocalMtProvider
from manga_ar.translate.normalize_ar import arabic_ratio

pytestmark = pytest.mark.integration


def test_local_mt_offline(cache_dir: Path) -> None:
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("transformers not installed (extra: offline-mt)")
    manager = ModelManager(cache_dir)
    try:
        for name in ("mt-ja-en", "mt-en-ar"):
            manager.ensure(name)
    except ModelUnavailableError as exc:
        pytest.skip(f"local MT weights unavailable (W-004): {exc}")
    prov = LocalMtProvider(ModelManager(cache_dir, offline=True))
    try:
        out = prov.translate_many(["ありがとう", "待ってくれ"], "ja", "ar")
    except ProviderError as exc:
        pytest.fail(f"offline translation failed: {exc}")
    assert all(arabic_ratio(t) >= 0.6 for t in out)
