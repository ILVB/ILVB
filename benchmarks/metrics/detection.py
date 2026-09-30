"""Region matching and detection precision/recall/F1 at IoU 0.5 (G-OCR-2)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

Point = tuple[float, float]
Box = tuple[float, float, float, float]


def box_of(polygon: Sequence[Point]) -> Box:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(
    gt: Sequence[Box], pred: Sequence[Box], threshold: float = 0.5
) -> list[tuple[int, int, float]]:
    """One-to-one greedy matching by decreasing IoU; pairs below ``threshold`` are dropped."""
    pairs = sorted(
        ((iou(g, p), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred)),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    used_g: set[int] = set()
    used_p: set[int] = set()
    out = []
    for score, gi, pi in pairs:
        if score < threshold:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        out.append((gi, pi, score))
    return out


@dataclass(frozen=True)
class DetectionCounts:
    true_positives: int
    gt_total: int
    pred_total: int

    def __add__(self, other: DetectionCounts) -> DetectionCounts:
        return DetectionCounts(
            self.true_positives + other.true_positives,
            self.gt_total + other.gt_total,
            self.pred_total + other.pred_total,
        )

    @property
    def precision(self) -> float:
        return self.true_positives / self.pred_total if self.pred_total else 0.0

    @property
    def recall(self) -> float:
        return self.true_positives / self.gt_total if self.gt_total else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0
