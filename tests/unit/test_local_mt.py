"""LocalMtProvider flow with fake transformers objects (no model download)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from manga_ar.errors import ModelUnavailableError, ProviderError
from manga_ar.models.manager import ModelManager
from manga_ar.translate.local_mt import LocalMtProvider


class FakeTensor:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    def to(self, device: str) -> FakeTensor:
        return self


class FakeTokenizer:
    def __init__(self, table: dict[str, str]) -> None:
        self.table = table
        self.src_lang = "en"

    def __call__(self, texts: list[str], **kw: Any) -> dict[str, FakeTensor]:
        return {"input_ids": FakeTensor(texts)}

    def batch_decode(self, out: list[str], skip_special_tokens: bool = True) -> list[str]:
        return out

    def get_lang_id(self, lang: str) -> int:
        return {"ar": 7}[lang]


class FakeModel:
    def __init__(self, tok: FakeTokenizer) -> None:
        self.tok = tok
        self.kwargs: dict[str, Any] = {}

    def generate(self, input_ids: FakeTensor, **kw: Any) -> list[str]:
        self.kwargs = kw
        return [self.tok.table.get(t, f"?{t}") for t in input_ids.texts]


def _provider(model: str, tables: dict[str, dict[str, str]]) -> LocalMtProvider:
    prov = LocalMtProvider(ModelManager(Path("/nonexistent"), offline=True), model)
    loaded: list[str] = []

    def fake_pipeline(name: str) -> tuple[FakeTokenizer, FakeModel]:
        loaded.append(name)
        if name not in tables:
            raise ModelUnavailableError(f"{name} missing")
        tok = FakeTokenizer(tables[name])
        return tok, FakeModel(tok)

    prov._pipeline = fake_pipeline  # type: ignore[method-assign]
    prov.loaded = loaded  # type: ignore[attr-defined]
    return prov


def test_marian_pivot_through_english() -> None:
    prov = _provider(
        "marian",
        {
            "mt-ja-en": {"ありがとう": "thank you", "待って": "wait"},
            "mt-en-ar": {"thank you": "شكرا", "wait": "انتظر"},
        },
    )
    assert prov.translate_many(["ありがとう", "待って"], "ja", "ar") == ["شكرا", "انتظر"]
    assert prov.loaded == ["mt-ja-en", "mt-en-ar"]  # type: ignore[attr-defined]
    assert prov.translate("待って", "ja", "ar") == "انتظر"
    assert prov.supports("ko", "ar") and not prov.supports("ja", "en")


def test_m2m100_direct_with_forced_language() -> None:
    prov = _provider("m2m100", {"mt-m2m100": {"谢谢": "شكرا"}})
    assert prov.translate("谢谢", "zh", "ar") == "شكرا"


def test_missing_model_is_provider_error() -> None:
    prov = _provider("marian", {"mt-en-ar": {}})
    with pytest.raises(ProviderError):
        prov.translate("ありがとう", "ja", "ar")
    assert prov.translate_many([], "ja", "ar") == []


def test_available_offline_requires_models(tmp_path: Path) -> None:
    prov = LocalMtProvider(ModelManager(tmp_path, offline=True))
    assert prov.available() is False
    for name in ("ja-en", "en-ar"):
        d = tmp_path / "models" / "hf" / f"Helsinki-NLP--opus-mt-{name}"
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}", encoding="utf-8")
    assert prov.available() is True
