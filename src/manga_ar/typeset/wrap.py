"""Balanced wrapping on the LOGICAL string (A1, A7).

Words are joined into logical lines; each candidate line is measured after shaping +
bidi (its real visual width). A dynamic programme picks, for a fixed number of lines,
the break points minimising squared slack against per-line spans (constant for the
rectangle strategy, following the bubble outline for the shape strategy), penalising a
single-word last line (orphan).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

INF = float("inf")


@dataclass
class Wrap:
    lines: list[str]  # logical lines, top → bottom
    widths: list[int]
    cost: float


def tokenize(text: str) -> list[str]:
    """Whitespace-separated words in logical order (punctuation stays attached)."""
    return text.split()


def wrap_fixed(
    words: list[str],
    spans: list[int],
    measure: Callable[[int, int], int],
    orphan_penalty: float = 0.35,
) -> Wrap | None:
    """Best wrap into exactly ``len(spans)`` lines, each fitting its span; else ``None``.

    ``measure(i, j)`` returns the visual width of ``" ".join(words[i:j])``.
    """
    n_lines = len(spans)
    n = len(words)
    if n_lines == 0 or n < n_lines:
        return None
    widths: dict[tuple[int, int], int] = {}

    def width(i: int, j: int) -> int:
        key = (i, j)
        if key not in widths:
            widths[key] = measure(i, j)
        return widths[key]

    # best[k][j]: minimal cost placing words[:j] on the first k lines
    best = [[INF] * (n + 1) for _ in range(n_lines + 1)]
    back = [[-1] * (n + 1) for _ in range(n_lines + 1)]
    best[0][0] = 0.0
    for k in range(1, n_lines + 1):
        span = spans[k - 1]
        remaining_lines = n_lines - k
        for j in range(k, n - remaining_lines + 1):
            for i in range(k - 1, j):
                if best[k - 1][i] == INF:
                    continue
                w = width(i, j)
                if w > span:
                    continue
                slack = (span - w) / max(1, span)
                cost = best[k - 1][i] + slack * slack
                if k == n_lines and n_lines > 1 and j - i == 1 and n > 2:
                    cost += orphan_penalty
                if cost < best[k][j]:
                    best[k][j] = cost
                    back[k][j] = i
    if best[n_lines][n] == INF:
        return None
    lines: list[str] = []
    line_widths: list[int] = []
    j = n
    for k in range(n_lines, 0, -1):
        i = back[k][j]
        lines.append(" ".join(words[i:j]))
        line_widths.append(width(i, j))
        j = i
    lines.reverse()
    line_widths.reverse()
    return Wrap(lines, line_widths, best[n_lines][n])
