from __future__ import annotations

import numpy as np
import pytest

from manga_ar.ocr.langid import (
    classify_text,
    is_passthrough,
    script_consistency,
    vote_language,
)
from manga_ar.ocr.paddle_engine import normalize_paddle_output
from manga_ar.ocr.postprocess import extract_decorations, normalize_ocr_text, postprocess
from manga_ar.ocr.reflow import glyph_cells, merge_stroke_runs, reflow_column
from manga_ar.ocr.suspicion import assess


def test_decorations_and_normalisation() -> None:
    assert extract_decorations("好き♡だよ♪") == ("好きだよ", ["♡", "♪"])
    assert normalize_ocr_text("え ？！ 本 当", "ja") == "え?!本当"
    assert normalize_ocr_text("そう。。。", "ja") == "そう…"
    assert normalize_ocr_text("待って〜〜", "ja") == "待って~~"
    assert normalize_ocr_text("정말  갈 거야", "ko") == "정말 갈 거야"
    assert normalize_ocr_text("ｱｲｳ", "ja") == "アイウ"  # half-width katakana → NFKC
    assert normalize_ocr_text("!!!!!!", "ja") == "!!!"
    assert postprocess("ね♡", "ja", preserve_decorations=False) == ("ね", [])


def test_langid() -> None:
    assert classify_text("ありがとう") == "ja"
    assert classify_text("本当に行くの") == "ja"
    assert classify_text("고마워요") == "ko"
    assert classify_text("谢谢你") == "zh"
    assert classify_text("OK!") is None
    assert vote_language(["谢谢", "ありがとう", "ありがとう"]) == "ja"
    assert vote_language(["123"]) is None
    assert is_passthrough("OK 123!") and not is_passthrough("OKだ")
    assert script_consistency("고마워", "ko") == 1.0
    assert script_consistency("hello", "ja") == 0.0


def test_suspicion_rules() -> None:
    assert assess("", "ja", 0.9, 1000, 20, 0.35).reasons == ["empty"]
    loop = assess("だだだだだだだ", "ja", 0.99, 5000, 20, 0.35)
    assert loop.suspicious and "repetition-loop" in loop.reasons
    dense = assess("あ" * 60, "ja", 0.9, 400, 20, 0.35)
    assert dense.suspicious and any(r.startswith("too-dense") for r in dense.reasons)
    good = assess("大丈夫だよ", "ja", 0.9, 30 * 130, 26, 0.35)
    assert not good.suspicious and good.score > 0.8
    wrong = assess("hello", "ko", 0.99, 3000, 20, 0.35)
    assert wrong.suspicious


def test_merge_stroke_runs() -> None:
    # ど (0-18), こ top stroke (24-27) + bottom stroke (33-37), う dash (42-44) + curve (46-62)
    runs = [(0, 18), (24, 27), (33, 37), (42, 44), (46, 62)]
    assert merge_stroke_runs(runs, 20.0) == [(0, 18), (24, 37), (42, 62)]
    assert merge_stroke_runs([(0, 20), (26, 46)], 20.0) == [(0, 20), (26, 46)]  # never fuse


def test_glyph_cells_and_reflow() -> None:
    col = np.full((150, 40, 3), 255, np.uint8)
    for k in range(3):
        col[10 + 45 * k : 40 + 45 * k, 8:32] = 0
    assert len(glyph_cells(col, 30)) == 3
    strip = reflow_column(col, size=30)
    assert strip.shape[1] > strip.shape[0]
    # touching glyphs are split into square cells
    tall = np.full((130, 40, 3), 255, np.uint8)
    tall[5:125, 8:32] = 0
    assert len(glyph_cells(tall, 30)) == 4


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, []),
        ([{"rec_texts": ["今日", "は"], "rec_scores": [0.9, 0.8]}], [("今日", 0.9), ("は", 0.8)]),
        ([[[[[0, 0], [1, 0], [1, 1], [0, 1]], ("안녕", 0.95)]]], [("안녕", 0.95)]),
        ([None], []),
    ],
)
def test_paddle_output_normaliser(raw: object, expected: list[tuple[str, float]]) -> None:
    assert normalize_paddle_output(raw) == expected


def test_paddle_result_object_with_json() -> None:
    class Result:
        def __init__(self) -> None:
            self.json = {"res": {"rec_texts": ["你好"], "rec_scores": [0.7]}}

    assert normalize_paddle_output([Result()]) == [("你好", 0.7)]
