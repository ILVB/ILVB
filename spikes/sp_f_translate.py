"""SP-F: deep-translator JA/KO/ZH→AR with deadline + backoff; local MT offline smoke."""

from __future__ import annotations

import concurrent.futures as cf
import random
import time

SENTENCES = {"ja": "今日はいい天気ですね", "ko": "오늘은 날씨가 좋네요", "zh": "今天天气很好"}


def with_deadline(fn, timeout: float):  # type: ignore[no-untyped-def]
    pool = cf.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(fn)
    try:
        return fut.result(timeout=timeout)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def backoff_call(fn, attempts: int = 3):  # type: ignore[no-untyped-def]
    for i in range(attempts):
        try:
            return with_deadline(fn, 25.0)
        except Exception as exc:
            delay = random.uniform(0, min(8.0, 1.0 * 2**i))
            print(
                f"  attempt {i + 1} failed: {type(exc).__name__}: {str(exc)[:120]}; sleep {delay:.1f}s"
            )
            time.sleep(delay)
    return None


def main() -> None:
    from deep_translator import GoogleTranslator, MyMemoryTranslator

    for name, factory in (
        (
            "google",
            lambda src: GoogleTranslator(source=src if src != "zh" else "zh-CN", target="ar"),
        ),
        (
            "mymemory",
            lambda src: MyMemoryTranslator(
                source={"ja": "ja-JP", "ko": "ko-KR", "zh": "zh-CN"}[src], target="ar-SA"
            ),
        ),
    ):
        for src, text in SENTENCES.items():
            print(f"{name} {src}:")
            out = backoff_call(lambda: factory(src).translate(text), attempts=2)
            print(f"  -> {out!r}")
    print("local MT (Marian ja-en):")
    try:
        from transformers import MarianMTModel, MarianTokenizer

        tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
        model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
        batch = tok([SENTENCES["ja"]], return_tensors="pt")
        print("  ->", tok.batch_decode(model.generate(**batch), skip_special_tokens=True))
    except Exception as exc:
        print(f"  FAILED {type(exc).__name__}: {str(exc)[:200]}")


if __name__ == "__main__":
    main()
