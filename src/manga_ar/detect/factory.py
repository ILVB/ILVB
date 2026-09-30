"""Build the configured text detector (with graceful fallback to the classical one)."""

from __future__ import annotations

from manga_ar.config import AppConfig
from manga_ar.detect.base import TextDetector
from manga_ar.detect.classical import ClassicalDetector
from manga_ar.errors import ModelUnavailableError
from manga_ar.logging_setup import get_logger
from manga_ar.models.manager import ModelManager

log = get_logger(__name__)


def resolve_detector(cfg: AppConfig) -> str:
    if cfg.detect.detector != "auto":
        return cfg.detect.detector
    return "db_primary" if cfg.engine.profile == "v2" else "classical"


def build_detector(cfg: AppConfig, manager: ModelManager) -> TextDetector:
    """``auto``/``classical`` → classical; ``hybrid``/``ctd``/``rapid``/``craft`` → ML-assisted
    detectors (fallback: classical).

    ``auto`` is classical under the ``legacy`` profile (v0.1.0, DECISIONS D-011) and
    ``db_primary`` under ``v2`` (ADR-0004: dev/val evidence on synthetic_v1). Both need no
    download. comic-text-detector is opt-in because of its GPL weights.
    """
    name = resolve_detector(cfg)
    classical = ClassicalDetector(cfg.detect)
    if name == "classical":
        return classical
    try:
        if name == "ctd":
            from manga_ar.detect.comic_text_detector import ComicTextDetector

            return ComicTextDetector(cfg.detect, manager, classical)
        if name == "rapid":
            from manga_ar.detect.rapid_detector import RapidDbDetector

            return RapidDbDetector(cfg.detect, classical)
        if name == "db_primary":
            from manga_ar.detect.db_primary import DbPrimaryDetector
            from manga_ar.detect.rapid_detector import RapidDbDetector

            return DbPrimaryDetector(RapidDbDetector(cfg.detect, classical), classical)
        if name == "hybrid":
            from manga_ar.detect.hybrid import HybridDetector
            from manga_ar.detect.rapid_detector import RapidDbDetector

            return HybridDetector(classical, RapidDbDetector(cfg.detect, classical))
        from manga_ar.detect.craft_detector import CraftDetector

        return CraftDetector(cfg.detect, manager, classical, offline=cfg.runtime.offline)
    except ModelUnavailableError as exc:
        log.warning("detector %r unavailable (%s); using the classical detector", name, exc)
        return classical
