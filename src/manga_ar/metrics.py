"""Evaluation metrics shared by the test-suite and ``scripts/benchmark.py``."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from manga_ar.schemas import BBox

BoolArray = npt.NDArray[np.bool_]


def match_boxes(
    truth: Sequence[BBox], pred: Sequence[BBox], iou_threshold: float = 0.5
) -> list[tuple[int, int]]:
    """Greedy one-to-one matching by IoU (highest first). Returns (truth_i, pred_j)."""
    pairs = sorted(
        ((t.iou(p), i, j) for i, t in enumerate(truth) for j, p in enumerate(pred)),
        reverse=True,
    )
    used_t: set[int] = set()
    used_p: set[int] = set()
    out: list[tuple[int, int]] = []
    for iou, i, j in pairs:
        if iou < iou_threshold:
            break
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        out.append((i, j))
    return out


def precision_recall(n_truth: int, n_pred: int, n_matched: int) -> tuple[float, float]:
    precision = n_matched / n_pred if n_pred else 1.0
    recall = n_matched / n_truth if n_truth else 1.0
    return precision, recall


def mask_iou(a: BoolArray, b: BoolArray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(truth: str, pred: str) -> float:
    """Character error rate (edit distance / reference length), spaces ignored."""
    t = "".join(truth.split())
    p = "".join(pred.split())
    return levenshtein(t, p) / max(1, len(t))


def similarity(a: str, b: str) -> float:
    """Normalised Levenshtein similarity in [0, 1]."""
    return 1.0 - levenshtein(a, b) / max(1, len(a), len(b))


def order_exact(truth_order: Sequence[int], pred_order: Sequence[int]) -> bool:
    return list(truth_order) == list(pred_order)
