"""OCR text metrics: normalisation, character/word edit counts (micro-averaged CER/WER)."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import jiwer


def normalize(text: str) -> str:
    """Gate normalisation (G-OCR-1): NFKC, then all whitespace removed."""
    return "".join(unicodedata.normalize("NFKC", text).split())


@dataclass(frozen=True)
class EditCounts:
    edits: int
    length: int

    def __add__(self, other: EditCounts) -> EditCounts:
        return EditCounts(self.edits + other.edits, self.length + other.length)

    @property
    def rate(self) -> float:
        return self.edits / self.length if self.length else 0.0


def char_edits(reference: str, hypothesis: str) -> EditCounts:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return EditCounts(len(hyp), 0)
    if not hyp:
        return EditCounts(len(ref), len(ref))
    out = jiwer.process_characters(ref, hyp)
    return EditCounts(out.substitutions + out.deletions + out.insertions, len(ref))


def word_edits(reference: str, hypothesis: str) -> EditCounts:
    ref = " ".join(unicodedata.normalize("NFKC", reference).split())
    hyp = " ".join(unicodedata.normalize("NFKC", hypothesis).split())
    if not ref:
        return EditCounts(len(hyp.split()), 0)
    if not hyp:
        n = len(ref.split())
        return EditCounts(n, n)
    out = jiwer.process_words(ref, hyp)
    return EditCounts(out.substitutions + out.deletions + out.insertions, len(ref.split()))
