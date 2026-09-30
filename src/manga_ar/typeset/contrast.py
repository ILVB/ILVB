"""Text colour, outline and contrast (A10)."""

from __future__ import annotations

Rgb = tuple[int, int, int]
BLACK: Rgb = (0, 0, 0)
WHITE: Rgb = (255, 255, 255)


def relative_luminance(rgb: Rgb) -> float:
    """WCAG 2.x relative luminance."""

    def channel(c: int) -> float:
        v = c / 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast_ratio(a: Rgb, b: Rgb) -> float:
    la, lb = relative_luminance(a), relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def choose_colors(
    background: Rgb, target: float, size: int, outline_frac: float, force_outline: bool
) -> tuple[Rgb, Rgb | None, int]:
    """Black or white text (whichever contrasts more with ``background``); an outline in
    the opposite colour when forced (text over artwork) or when contrast < ``target``."""
    text = (
        BLACK if contrast_ratio(BLACK, background) >= contrast_ratio(WHITE, background) else WHITE
    )
    best = contrast_ratio(text, background)
    if force_outline or best < target:
        outline = WHITE if text == BLACK else BLACK
        return text, outline, max(1, round(outline_frac * size))
    return text, None, 0
