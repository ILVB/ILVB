"""Offline machine translation with transformers (the only provider usable ``--offline``).

``marian``: OPUS-MT src→en then en→ar (pivot). ``m2m100``: direct src→ar.
Models come from the Hugging Face Hub on first use (``manga-arabic models download
local-mt``) and then load from the local cache with no network access.
"""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from typing import Any

from manga_ar.errors import ModelUnavailableError, ProviderError
from manga_ar.models.manager import ModelManager


class LocalMtProvider:
    name = "local"
    network = False

    def __init__(self, manager: ModelManager, model: str = "marian", device: str = "cpu") -> None:
        self.manager = manager
        self.model = model
        self.device = device
        self._lock = threading.Lock()

    def _required(self, src: str) -> list[str]:
        if self.model == "m2m100":
            return ["mt-m2m100"]
        return ([] if src == "en" else [f"mt-{src}-en"]) + ["mt-en-ar"]

    def available(self) -> bool:
        if importlib.util.find_spec("transformers") is None:
            return False
        if not self.manager.offline:
            return True
        return all(self.manager.is_present(n) for n in self._required("ja"))

    def supports(self, src: str, tgt: str) -> bool:
        return tgt == "ar" and src in {"ja", "ko", "zh", "en"}

    def _pipeline(self, name: str) -> tuple[Any, Any]:
        def build(path: Path) -> tuple[Any, Any]:
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                torch.manual_seed(0)
                tok = AutoTokenizer.from_pretrained(str(path))
                model = AutoModelForSeq2SeqLM.from_pretrained(str(path)).to(self.device).eval()
            except (OSError, ValueError, RuntimeError, ImportError) as exc:
                raise ModelUnavailableError(
                    f"local MT model {name!r} could not load: {exc}"
                ) from exc
            return tok, model

        return self.manager.load(name, build)

    def _generate(
        self,
        name: str,
        texts: list[str],
        forced_lang: str | None = None,
        src_lang: str | None = None,
    ) -> list[str]:
        import torch

        tok, model = self._pipeline(name)
        if src_lang is not None and hasattr(tok, "src_lang"):
            tok.src_lang = src_lang
        batch = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        kwargs: dict[str, Any] = {"max_new_tokens": 256, "num_beams": 4, "do_sample": False}
        if forced_lang is not None:
            kwargs["forced_bos_token_id"] = tok.get_lang_id(forced_lang)
        with torch.inference_mode():
            out = model.generate(**batch, **kwargs)
        return [str(t) for t in tok.batch_decode(out, skip_special_tokens=True)]

    def translate_many(self, texts: list[str], src: str, tgt: str) -> list[str]:
        """Translate a list in one model batch (no index markers needed)."""
        if not texts:
            return []
        with self._lock:
            try:
                if self.model == "m2m100":
                    return self._generate("mt-m2m100", texts, forced_lang="ar", src_lang=src)
                english = texts if src == "en" else self._generate(f"mt-{src}-en", texts)
                return self._generate("mt-en-ar", english)
            except ModelUnavailableError as exc:
                raise ProviderError(str(exc), retryable=False) from exc
            except (RuntimeError, ValueError, IndexError) as exc:
                raise ProviderError(f"local MT failed: {exc}", retryable=False) from exc

    def translate(self, text: str, src: str, tgt: str) -> str:
        return self.translate_many([text], src, tgt)[0]
