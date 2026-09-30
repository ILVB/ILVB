"""Build the OCR engine set and router from configuration."""

from __future__ import annotations

from manga_ar.config import AppConfig
from manga_ar.models.manager import ModelManager
from manga_ar.ocr.base import OcrEngine
from manga_ar.ocr.easyocr_engine import EasyOcrEngine
from manga_ar.ocr.manga_ocr_engine import MangaOcrEngine
from manga_ar.ocr.paddle_engine import PaddleOcrEngine
from manga_ar.ocr.rapid_engine import RapidOcrEngine
from manga_ar.ocr.router import OcrRouter


def build_engines(cfg: AppConfig, manager: ModelManager, device: str) -> dict[str, OcrEngine]:
    return {
        "manga_ocr": MangaOcrEngine(manager, device),
        "easyocr": EasyOcrEngine(manager, device, offline=cfg.runtime.offline),
        "rapid": RapidOcrEngine(cfg.runtime.num_threads),
        "paddle": PaddleOcrEngine(device),
    }


def build_router(cfg: AppConfig, manager: ModelManager, device: str) -> OcrRouter:
    return OcrRouter(
        build_engines(cfg, manager, device), cfg.ocr, cfg.upgrade("ocr", "clean_texture")
    )
