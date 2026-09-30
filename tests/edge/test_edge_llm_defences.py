"""E-12 and E-13 on the v2 prompt and output defences (fake model replies, no LLM)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from manga_ar.config import load_config
from manga_ar.translate.v2.defences import accept, strip_meta
from manga_ar.translate.v2.prompt import (
    CLOSE,
    OPEN,
    RULES,
    ContractError,
    Segment,
    build_prompt,
    parse_reply,
)

ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(environ={}).translate
SUITE = yaml.safe_load((ROOT / "benchmarks/data/suites/injection.yaml").read_text("utf-8"))


def _reply(text: str) -> str:
    return json.dumps({"items": [{"id": "r1", "translation": text, "glossary_used": [],
                                  "confidence": 0.9}]}, ensure_ascii=False)  # fmt: skip


def _outcome(source: str, raw: str) -> str | None:
    """The accepted rendering for one region, or None when the reply is rejected."""
    try:
        translations = parse_reply(raw, ("r1",))
    except ContractError:
        return None
    return accept(source, translations["r1"], CFG)[0]


@pytest.mark.parametrize("item", SUITE["items"], ids=lambda i: i["id"])
def test_e13_bubble_text_stays_inside_the_data_block(item: dict[str, Any]) -> None:
    prompt = build_prompt([Segment("r1", item["text"])])
    assert prompt.system == RULES  # instructions never depend on the data
    assert prompt.user.count(OPEN) == 1 and prompt.user.count(CLOSE) == 1
    block = prompt.user.split(OPEN, 1)[1].rsplit(CLOSE, 1)[0]
    assert "<" not in block  # no source text can open or close a delimiter
    (entry,) = json.loads(block)
    assert entry["id"] == "r1" and "‮" not in entry["text"]  # bidi overrides removed


@pytest.mark.parametrize("item", SUITE["items"], ids=lambda i: i["id"])
def test_e13_obeying_the_injection_is_never_accepted(item: dict[str, Any]) -> None:
    raw = item["obey"] if item["obey"].lstrip().startswith("{") else _reply(item["obey"])
    assert _outcome(item["text"], raw) is None
    assert _outcome(item["text"], _reply(item["honest"])) == item["honest"]


def test_e12_refusal_meta_loops_and_broken_json() -> None:
    src = "ありがとう"
    assert _outcome(src, _reply("عذرًا، لا يمكنني ترجمة هذا")) is None  # refusal (Arabic)
    assert _outcome(src, _reply("Here is the translation: شكرًا لك")) == "شكرًا لك"  # stripped
    assert strip_meta("الترجمة: «شكرًا»") == "شكرًا"
    assert _outcome(src, _reply("شكرا شكرا شكرا شكرا شكرا")) is None  # repetition loop
    assert _outcome(src, "not json at all") is None
    assert _outcome(src, '{"items": [{"id": "r1"}]}') is None  # no translation string
    assert _outcome(src, "```json\n" + _reply("شكرًا لك") + "\n```") == "شكرًا لك"
