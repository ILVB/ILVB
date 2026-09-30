"""Live translation smoke test (``-m network``): ≥ 20 sentences ja/ko/zh → Arabic.

Blocked on the build host by the network policy (DECISIONS W-003); run on a machine
with internet access: pytest -m network tests/integration/test_translate_live.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from manga_ar.config import load_config
from manga_ar.models.manager import ModelManager
from manga_ar.schemas import BBox, OcrResult, Region, RegionType
from manga_ar.translate.normalize_ar import arabic_ratio, has_cjk
from manga_ar.translate.service import build_translation_service

pytestmark = pytest.mark.network

SENTENCES = {
    "ja": [
        "ありがとう",
        "待ってくれ",
        "本当に行くの?",
        "大丈夫だよ",
        "気をつけて",
        "どこへ行く?",
        "もう遅いよ",
        "信じてる",
    ],
    "ko": [
        "고마워요",
        "기다려 줘",
        "정말 갈 거야?",
        "괜찮아",
        "조심해",
        "어디 가니?",
        "너무 늦었어",
    ],
    "zh": ["谢谢你", "等一下", "你真的要去吗?", "没关系", "小心点", "你去哪里?", "太晚了"],
}


def _reachable(url: str) -> bool:
    try:
        return requests.head(url, timeout=5).status_code < 500
    except requests.RequestException:
        return False


@pytest.mark.parametrize("lang", ["ja", "ko", "zh"])
def test_live_translation(lang: str, tmp_path: Path) -> None:
    if not (
        _reachable("https://translate.google.com/")
        or _reachable("https://api.mymemory.translated.net/")
    ):
        pytest.skip("no free translation endpoint reachable from this host (W-003)")
    cfg = load_config(
        overrides={"translate.providers": ["google", "mymemory"], "translate.cache": False},
        environ={},
    )
    svc = build_translation_service(cfg, ModelManager(tmp_path), tmp_path)
    regions = [
        Region(
            id=f"r{i}",
            type=RegionType.BUBBLE,
            bbox=BBox(0, 0, 1, 1),
            reading_order=i,
            ocr=OcrResult("test", t, lang=lang),
        )
        for i, t in enumerate(SENTENCES[lang])
    ]
    svc.translate_regions(regions, lang)
    for r in regions:
        assert r.translation is not None and r.translation.text, r.id
        assert arabic_ratio(r.translation.text) >= 0.6 and not has_cjk(r.translation.text)
