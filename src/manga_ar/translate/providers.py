"""Free online providers via deep-translator (Google web endpoint, MyMemory, LibreTranslate).

These endpoints are unofficial/free and may change or be restricted at any time; the
offline local model exists for that reason. No API keys, no accounts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from manga_ar.errors import ProviderError, RateLimitError

# Per-provider language codes (verified against deep-translator 1.11.4).
GOOGLE_CODES = {"ja": "ja", "ko": "ko", "zh": "zh-CN", "zh-TW": "zh-TW", "en": "en", "ar": "ar"}
MYMEMORY_CODES = {
    "ja": "ja-JP",
    "ko": "ko-KR",
    "zh": "zh-CN",
    "zh-TW": "zh-TW",
    "en": "en-GB",
    "ar": "ar-SA",
}
LIBRE_CODES = {"ja": "ja", "ko": "ko", "zh": "zh", "zh-TW": "zt", "en": "en", "ar": "ar"}


def _map_error(exc: Exception, provider: str) -> ProviderError:
    from deep_translator import exceptions as dt

    if isinstance(exc, dt.TooManyRequests):
        return RateLimitError(f"{provider}: HTTP 429 / quota exceeded")
    if isinstance(
        exc,
        (
            dt.NotValidLength,
            dt.NotValidPayload,
            dt.LanguageNotSupportedException,
            dt.InvalidSourceOrTargetLanguage,
            dt.AuthorizationException,
            dt.ApiKeyException,
        ),
    ):
        return ProviderError(f"{provider}: {type(exc).__name__}: {exc}", retryable=False)
    return ProviderError(f"{provider}: {type(exc).__name__}: {exc}", retryable=True)


class _DeepTranslatorProvider(ABC):
    name = "base"
    network = True
    codes: ClassVar[dict[str, str]] = {}

    def __init__(self, proxy: str | None = None) -> None:
        self.proxies = {"https": proxy, "http": proxy} if proxy else None

    def available(self) -> bool:
        import importlib.util

        return importlib.util.find_spec("deep_translator") is not None

    def supports(self, src: str, tgt: str) -> bool:
        return src in self.codes and tgt in self.codes

    @abstractmethod
    def _client(self, src: str, tgt: str) -> Any:
        """Build the deep-translator client for provider-specific language codes."""

    def translate(self, text: str, src: str, tgt: str) -> str:
        import requests

        try:
            out = self._client(self.codes[src], self.codes[tgt]).translate(text)
        except requests.RequestException as exc:
            raise ProviderError(f"{self.name}: network error {type(exc).__name__}") from exc
        except Exception as exc:
            raise _map_error(exc, self.name) from exc
        if not isinstance(out, str):
            raise ProviderError(f"{self.name}: empty/invalid response", retryable=True)
        return out


class GoogleProvider(_DeepTranslatorProvider):
    name = "google"
    codes = GOOGLE_CODES

    def _client(self, src: str, tgt: str) -> Any:
        from deep_translator import GoogleTranslator

        return GoogleTranslator(source=src, target=tgt, proxies=self.proxies)


class MyMemoryProvider(_DeepTranslatorProvider):
    name = "mymemory"
    codes = MYMEMORY_CODES

    def _client(self, src: str, tgt: str) -> Any:
        from deep_translator import MyMemoryTranslator

        return MyMemoryTranslator(source=src, target=tgt, proxies=self.proxies)


class LibreTranslateProvider(_DeepTranslatorProvider):
    name = "libretranslate"
    codes = LIBRE_CODES

    def __init__(self, base_url: str | None, proxy: str | None = None) -> None:
        super().__init__(proxy)
        self.base_url = base_url

    def available(self) -> bool:
        return bool(self.base_url) and super().available()

    def _client(self, src: str, tgt: str) -> Any:
        from deep_translator import LibreTranslator

        return LibreTranslator(
            source=src,
            target=tgt,
            custom_url=self.base_url,
            use_free_api=False,
            proxies=self.proxies,
        )
