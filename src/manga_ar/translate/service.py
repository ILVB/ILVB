"""Translation orchestration: dedupe, glossary, cache, batching, failover, validation (S5).

Chain (default): tm → google → mymemory → libretranslate (if URL) → local. Offline mode
keeps only non-network providers. Every network call goes through a
:class:`ResilientCaller` (limiter, breaker, deadline, backoff). A region whose text no
provider could translate acceptably is flagged ``UNTRANSLATED`` (never rendered).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from manga_ar.config import AppConfig
from manga_ar.errors import ProviderError
from manga_ar.logging_setup import get_logger
from manga_ar.models.manager import ModelManager
from manga_ar.schemas import Flag, Region, RegionType, TranslationAttempt, TranslationResult
from manga_ar.translate.base import Translator
from manga_ar.translate.batching import chunk, decode, encode
from manga_ar.translate.cache import TranslationCache
from manga_ar.translate.glossary import Glossary
from manga_ar.translate.normalize_ar import normalize_ar
from manga_ar.translate.resilience import (
    Backoff,
    CircuitBreaker,
    Clock,
    RealClock,
    ResilientCaller,
    TokenBucket,
)
from manga_ar.translate.tm import TranslationMemory
from manga_ar.translate.validate import translation_problems

log = get_logger(__name__)
TARGET = "ar"


@dataclass
class _Job:
    source: str
    protected: str
    mapping: dict[str, str]
    decorations: list[str]
    regions: list[Region] = field(default_factory=list)
    attempts: list[TranslationAttempt] = field(default_factory=list)
    result: str | None = None
    provider: str | None = None
    from_cache: bool = False
    glossary_lost: bool = False
    placeholder_failures: int = 0


class TranslationService:
    def __init__(
        self,
        providers: Sequence[Translator],
        cfg: AppConfig,
        cache: TranslationCache | None = None,
        glossary: Glossary | None = None,
        clock: Clock | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.providers = list(providers)
        self.cfg = cfg
        self.tcfg = cfg.translate
        self.cache = cache
        self.glossary = glossary or Glossary()
        self.clock = clock or RealClock()
        self.rng = rng or random.Random(cfg.runtime.seed)
        self.callers: dict[str, ResilientCaller] = {}
        for p in self.providers:
            if p.network:
                self.callers[p.name] = ResilientCaller(
                    p.name,
                    limiter=TokenBucket(self.tcfg.rate_per_sec, self.tcfg.burst, self.clock),
                    breaker=CircuitBreaker(
                        self.tcfg.breaker_threshold, self.tcfg.breaker_cooldown, self.clock
                    ),
                    backoff=Backoff(
                        self.tcfg.backoff_base,
                        self.tcfg.backoff_factor,
                        self.tcfg.backoff_cap,
                        self.tcfg.max_attempts,
                    ),
                    deadline=self.tcfg.total_timeout,
                    clock=self.clock,
                    rng=self.rng,
                )

    # ----------------------------------------------------------------- chain
    def chain(self, src: str) -> list[Translator]:
        out = []
        for p in self.providers:
            if self.cfg.runtime.offline and p.network:
                continue
            if p.available() and p.supports(src, TARGET):
                out.append(p)
        return out

    def has_offline_provider(self, src: str) -> bool:
        return any(
            not p.network and p.name != "tm" and p.available() and p.supports(src, TARGET)
            for p in self.providers
        )

    # ------------------------------------------------------------- regions
    def translate_regions(self, regions: Sequence[Region], src: str) -> None:
        """Fill ``region.translation`` (or flag ``UNTRANSLATED``) for eligible regions."""
        jobs: dict[str, _Job] = {}
        for r in sorted(regions, key=lambda x: x.reading_order):
            if r.type == RegionType.SFX or r.override.skip or Flag.PASS_THROUGH in r.flags:
                continue
            if r.override.text is not None:
                continue  # the user's own Arabic wins
            text = r.ocr.text if r.ocr is not None else ""
            if not text:
                r.flag(Flag.UNTRANSLATED)
                continue
            if Flag.OCR_SUSPECT in r.flags and not self.tcfg.translate_suspect:
                r.flag(Flag.UNTRANSLATED)  # never translate flagged garbage silently
                continue
            job = jobs.get(text)
            if job is None:
                protected, mapping = self.glossary.protect(text)
                decorations = list(r.ocr.decorations) if r.ocr is not None else []
                job = _Job(text, protected, mapping, decorations)
                jobs[text] = job
            job.regions.append(r)
        pending = list(jobs.values())
        pending = self._run_chain(pending, src)
        # Placeholders lost on every provider: degrade gracefully without the glossary.
        degraded = [j for j in pending if j.mapping and j.placeholder_failures]
        for j in degraded:
            j.protected, j.mapping, j.glossary_lost = j.source, {}, True
        if degraded:
            self._run_chain(degraded, src)
        for job in jobs.values():
            self._finalize(job)

    def translate_text(self, text: str, src: str) -> TranslationResult | None:
        """Translate a single text (GUI "re-translate region")."""
        from manga_ar.schemas import BBox, OcrResult

        probe = Region(
            id="probe",
            type=RegionType.BUBBLE,
            bbox=BBox(0, 0, 1, 1),
            ocr=OcrResult(engine="user", text=text, lang=src),
        )
        self.translate_regions([probe], src)
        return probe.translation if Flag.UNTRANSLATED not in probe.flags else None

    def _run_chain(self, jobs: list[_Job], src: str) -> list[_Job]:
        pending = jobs
        for provider in self.chain(src):
            if not pending:
                break
            pending = self._run_provider(provider, pending, src)
        return pending

    def _finalize(self, job: _Job) -> None:
        for r in job.regions:
            if job.result is not None and job.provider is not None:
                r.translation = TranslationResult(
                    job.provider, job.result, job.from_cache, list(job.attempts)
                )
                r.flags.discard(Flag.UNTRANSLATED)
                if job.glossary_lost:
                    r.flag(Flag.GLOSSARY_DEGRADED)
            else:
                r.translation = TranslationResult("none", "", False, list(job.attempts))
                r.flag(Flag.UNTRANSLATED)

    # ------------------------------------------------------------ providers
    def _run_provider(self, provider: Translator, jobs: list[_Job], src: str) -> list[_Job]:
        todo = []
        for j in jobs:
            if self.cache is not None and provider.name != "tm":
                hit = self.cache.get(provider.name, src, TARGET, j.protected)
                if hit is not None and self._accept(j, hit, provider.name, src, from_cache=True):
                    continue
            todo.append(j)
        if not todo:
            return []
        if isinstance(provider, TranslationMemory):
            for j in todo:
                hit = provider.lookup(j.source)
                if hit is None:
                    j.attempts.append(TranslationAttempt("tm", False, "no exact match"))
                else:
                    self._accept(j, hit, "tm", src, from_cache=False, trusted=True)
            return [j for j in todo if j.result is None]
        many = getattr(provider, "translate_many", None)
        if callable(many):
            start = self.clock.time()
            try:
                outputs = many([j.protected for j in todo], src, TARGET)
            except ProviderError as exc:
                for j in todo:
                    j.attempts.append(
                        TranslationAttempt(
                            provider.name, False, str(exc), self.clock.time() - start
                        )
                    )
                return todo
            for j, out in zip(todo, outputs, strict=True):
                self._accept(j, out, provider.name, src, from_cache=False)
            return [j for j in todo if j.result is None]
        if self.tcfg.batch and len(todo) > 1:
            limit = self.tcfg.chunk_chars.get(provider.name, 1500)
            for idxs in chunk([j.protected for j in todo], limit):
                group = [todo[i] for i in idxs]
                if len(group) == 1:
                    self._single(provider, group[0], src)
                    continue
                self._batch(provider, group, src)
        else:
            for j in todo:
                self._single(provider, j, src)
        return [j for j in todo if j.result is None]

    def _call(self, provider: Translator, text: str, src: str) -> str:
        caller = self.callers.get(provider.name)
        if caller is None:
            return provider.translate(text, src, TARGET)
        return caller.call(lambda: provider.translate(text, src, TARGET))

    def _single(self, provider: Translator, job: _Job, src: str) -> None:
        start = self.clock.time()
        try:
            out = self._call(provider, job.protected, src)
        except ProviderError as exc:
            job.attempts.append(
                TranslationAttempt(provider.name, False, str(exc), self.clock.time() - start)
            )
            return
        self._accept(job, out, provider.name, src, from_cache=False)

    def _batch(self, provider: Translator, group: list[_Job], src: str) -> None:
        start = self.clock.time()
        try:
            response = self._call(provider, encode([j.protected for j in group]), src)
        except ProviderError as exc:
            for j in group:
                j.attempts.append(
                    TranslationAttempt(provider.name, False, str(exc), self.clock.time() - start)
                )
            return
        segments = decode(response, len(group))
        if segments is None:
            log.info(
                "%s: batch integrity check failed; translating %d regions one by one",
                provider.name,
                len(group),
            )
            for j in group:
                self._single(provider, j, src)
            return
        for j, seg in zip(group, segments, strict=True):
            if not self._accept(j, seg, provider.name, src, from_cache=False):
                self._single(provider, j, src)  # retry this segment alone once

    def _accept(
        self,
        job: _Job,
        raw: str,
        provider: str,
        src: str,
        *,
        from_cache: bool,
        trusted: bool = False,
    ) -> bool:
        if trusted:
            restored, survived = raw, True
        else:
            restored, survived = self.glossary.restore(raw, job.mapping)
        problems = translation_problems(job.source, restored, self.tcfg)
        if job.mapping and not survived:
            problems.append("glossary-placeholder-lost")
            job.placeholder_failures += 1
        if problems:
            job.attempts.append(TranslationAttempt(provider, False, ",".join(problems)))
            return False
        job.result = normalize_ar(restored, self.cfg.typeset.digits, job.decorations)
        job.provider = provider
        job.from_cache = from_cache
        job.attempts.append(TranslationAttempt(provider, True, "cache" if from_cache else None))
        if self.cache is not None and not from_cache and provider != "tm":
            self.cache.put(provider, src, TARGET, job.protected, raw)
        return True


def build_translation_service(
    cfg: AppConfig,
    manager: ModelManager,
    cache_dir: Path,
    tm_pairs: dict[str, str] | None = None,
    clock: Clock | None = None,
) -> TranslationService:
    """Providers in configured order; TM from ``tm_pairs`` and/or ``translate.tm_file``."""
    from manga_ar.translate.local_mt import LocalMtProvider
    from manga_ar.translate.providers import (
        GoogleProvider,
        LibreTranslateProvider,
        MyMemoryProvider,
    )

    t = cfg.translate
    pairs: dict[str, str] = {}
    if t.tm_file:
        from manga_ar.translate.tm import load_pairs

        pairs.update(load_pairs(Path(t.tm_file)))
    pairs.update(tm_pairs or {})

    def make(name: str) -> Translator:
        if name == "tm":
            return TranslationMemory(pairs)
        if name == "google":
            return GoogleProvider(t.proxy)
        if name == "mymemory":
            return MyMemoryProvider(t.proxy)
        if name == "libretranslate":
            return LibreTranslateProvider(t.libretranslate_url, t.proxy)
        return LocalMtProvider(manager, t.local_model)

    providers = [make(name) for name in t.providers]
    cache = TranslationCache(cache_dir / "translations.sqlite3") if t.cache else None
    glossary = Glossary.from_file(Path(t.glossary_file)) if t.glossary_file else Glossary()
    return TranslationService(providers, cfg, cache, glossary, clock)
