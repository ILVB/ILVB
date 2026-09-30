"""deep-translator adapters with the library stubbed (no network)."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
import requests
from deep_translator import exceptions as dt

from manga_ar.errors import ProviderError, RateLimitError
from manga_ar.translate import providers as mod


class StubClient:
    behaviour: Any = "ok"
    seen: ClassVar[list[tuple[str, str, Any]]] = []

    def __init__(self, source: str, target: str, **kw: Any) -> None:
        StubClient.seen.append((source, target, kw))

    def translate(self, text: str) -> Any:
        b = StubClient.behaviour
        if isinstance(b, BaseException):
            raise b
        return None if b == "none" else f"ar:{text}"


@pytest.fixture(autouse=True)
def stub(monkeypatch: pytest.MonkeyPatch) -> None:
    import deep_translator

    for name in ("GoogleTranslator", "MyMemoryTranslator", "LibreTranslator"):
        monkeypatch.setattr(deep_translator, name, StubClient)
    StubClient.behaviour = "ok"
    StubClient.seen = []


def test_language_codes_and_proxy() -> None:
    g = mod.GoogleProvider(proxy="http://127.0.0.1:9")
    assert g.translate("你好", "zh", "ar") == "ar:你好"
    src, tgt, kw = StubClient.seen[-1]
    assert (src, tgt) == ("zh-CN", "ar") and kw["proxies"]["https"] == "http://127.0.0.1:9"
    m = mod.MyMemoryProvider()
    m.translate("안녕", "ko", "ar")
    assert StubClient.seen[-1][:2] == ("ko-KR", "ar-SA")
    assert g.supports("ja", "ar") and not g.supports("fr", "ar")


def test_libretranslate_needs_url() -> None:
    assert not mod.LibreTranslateProvider(None).available()
    lt = mod.LibreTranslateProvider("http://localhost:5000")
    assert lt.available()
    lt.translate("你好", "zh", "ar")
    assert StubClient.seen[-1][2]["custom_url"] == "http://localhost:5000"


@pytest.mark.parametrize(
    ("exc", "kind", "retryable"),
    [
        (dt.TooManyRequests(), RateLimitError, True),
        (dt.NotValidLength("x", 0, 5000), ProviderError, False),
        (dt.RequestError(), ProviderError, True),
        (requests.ConnectionError("proxy"), ProviderError, True),
    ],
)
def test_error_mapping(exc: BaseException, kind: type, retryable: bool) -> None:
    StubClient.behaviour = exc
    with pytest.raises(kind) as info:
        mod.GoogleProvider().translate("x", "ja", "ar")
    assert info.value.retryable is retryable  # type: ignore[attr-defined]


def test_invalid_response_is_retryable() -> None:
    StubClient.behaviour = "none"
    with pytest.raises(ProviderError) as info:
        mod.MyMemoryProvider().translate("x", "ja", "ar")
    assert info.value.retryable
