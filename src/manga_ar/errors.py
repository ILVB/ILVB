"""Exception hierarchy. Every error raised by MangaAR derives from :class:`MangaArError`."""

from __future__ import annotations


class MangaArError(Exception):
    """Base class for all MangaAR errors."""


class ConfigError(MangaArError):
    """Invalid or inconsistent configuration."""


class ImageLoadError(MangaArError):
    """An input image could not be decoded or violates a safety limit."""


class ArchiveError(ImageLoadError):
    """An archive (CBZ/ZIP) is malformed or contains unsafe members."""


class DetectionError(MangaArError):
    """Text detection or bubble segmentation failed."""


class OcrError(MangaArError):
    """Text recognition failed."""


class InpaintError(MangaArError):
    """Text removal failed."""


class TranslationError(MangaArError):
    """Translation failed (all providers exhausted or output rejected)."""


class ProviderError(TranslationError):
    """A single translation provider failed. ``retryable`` hints the resilience layer."""

    def __init__(self, message: str, *, retryable: bool = True, retry_after: float | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


class RateLimitError(ProviderError):
    """HTTP 429 / quota exhaustion. ``retry_after`` carries the server hint in seconds."""


class DeadlineExceededError(ProviderError):
    """A network call exceeded its hard deadline."""


class CircuitOpenError(ProviderError):
    """The provider's circuit breaker is open; the call was not attempted."""

    def __init__(self, message: str):
        super().__init__(message, retryable=False)


class TypesetError(MangaArError):
    """Arabic typesetting failed (e.g. no font can render the text)."""


class ModelUnavailableError(MangaArError):
    """A model is missing (offline, download failed, checksum mismatch, not installed)."""


class CancelledError(MangaArError):
    """Cooperative cancellation was requested (GUI cancel button)."""
