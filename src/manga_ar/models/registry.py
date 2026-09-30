"""Static registry of every model MangaAR can use, with source, checksum and licence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["file", "hf", "easyocr", "bundled"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: Kind
    description: str
    license: str
    url: str = ""
    sha256: str = ""
    size: int = 0
    filename: str = ""
    repo_id: str = ""
    revision: str = "main"
    copyleft_or_nc: bool = False


REGISTRY: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec(
            name="lama",
            kind="file",
            description="LaMa big-lama TorchScript inpainting model (IOPaint release)",
            license="Apache-2.0",
            url="https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
            sha256="344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9",
            size=205_669_692,
            filename="lama/big-lama.pt",
        ),
        ModelSpec(
            name="ctd",
            kind="file",
            description="comic-text-detector ONNX (text blocks + text mask). Opt-in.",
            license="GPL-3.0 (weights from zyddnys/manga-image-translator release)",
            url=(
                "https://github.com/zyddnys/manga-image-translator/releases/download/"
                "beta-0.3/comictextdetector.pt.onnx"
            ),
            sha256="1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f",
            size=94_669_756,
            filename="ctd/comictextdetector.pt.onnx",
            copyleft_or_nc=True,
        ),
        ModelSpec(
            name="easyocr",
            kind="easyocr",
            description="EasyOCR CRAFT detector + ja/ko/zh recognisers (GitHub releases, MD5 "
            "verified by easyocr)",
            license="Apache-2.0",
            filename="easyocr",
        ),
        ModelSpec(
            name="rapid",
            kind="bundled",
            description="PP-OCR det/cls/rec (Chinese) ONNX models bundled in rapidocr_onnxruntime",
            license="Apache-2.0",
        ),
        ModelSpec(
            name="manga-ocr",
            kind="hf",
            description="manga-ocr Japanese recogniser",
            license="Apache-2.0",
            repo_id="kha-white/manga-ocr-base",
        ),
        ModelSpec(
            name="mt-ja-en",
            kind="hf",
            description="Marian OPUS-MT Japanese→English (pivot)",
            license="CC-BY-4.0",
            repo_id="Helsinki-NLP/opus-mt-ja-en",
        ),
        ModelSpec(
            name="mt-ko-en",
            kind="hf",
            description="Marian OPUS-MT Korean→English (pivot)",
            license="CC-BY-4.0",
            repo_id="Helsinki-NLP/opus-mt-ko-en",
        ),
        ModelSpec(
            name="mt-zh-en",
            kind="hf",
            description="Marian OPUS-MT Chinese→English (pivot)",
            license="CC-BY-4.0",
            repo_id="Helsinki-NLP/opus-mt-zh-en",
        ),
        ModelSpec(
            name="mt-en-ar",
            kind="hf",
            description="Marian OPUS-MT English→Arabic (pivot)",
            license="CC-BY-4.0",
            repo_id="Helsinki-NLP/opus-mt-en-ar",
        ),
        ModelSpec(
            name="mt-m2m100",
            kind="hf",
            description="M2M100 418M direct many-to-many translation",
            license="MIT",
            repo_id="facebook/m2m100_418M",
        ),
        ModelSpec(
            name="font-cjk-jp",
            kind="file",
            description="Noto Sans JP (subset OTF) for synthetic fixtures / demo pages",
            license="OFL-1.1",
            url="https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/SubsetOTF/JP/"
            "NotoSansJP-Regular.otf",
            sha256="dff723ba59d57d136764a04b9b2d03205544f7cd785a711442d6d2d085ac5073",
            size=4_533_028,
            filename="fonts/NotoSansJP-Regular.otf",
        ),
        ModelSpec(
            name="font-cjk-kr",
            kind="file",
            description="Noto Sans KR (subset OTF) for synthetic fixtures / demo pages",
            license="OFL-1.1",
            url="https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/SubsetOTF/KR/"
            "NotoSansKR-Regular.otf",
            sha256="69975a0ac8472717870aefeab0a4d52739308d90856b9955313b2ad5e0148d68",
            size=4_644_748,
            filename="fonts/NotoSansKR-Regular.otf",
        ),
        ModelSpec(
            name="font-cjk-sc",
            kind="file",
            description="Noto Sans SC (subset OTF) for synthetic fixtures / demo pages",
            license="OFL-1.1",
            url="https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/SubsetOTF/SC/"
            "NotoSansSC-Regular.otf",
            sha256="faa6c9df652116dde789d351359f3d7e5d2285a2b2a1f04a2d7244df706d5ea9",
            size=8_331_336,
            filename="fonts/NotoSansSC-Regular.otf",
        ),
    )
}

# Named download bundles for `manga-arabic models download <group>`.
GROUPS: dict[str, tuple[str, ...]] = {
    "local-mt": ("mt-ja-en", "mt-ko-en", "mt-zh-en", "mt-en-ar"),
    "local-mt-m2m100": ("mt-m2m100",),
    "default": ("easyocr", "lama"),
    "fonts": ("font-cjk-jp", "font-cjk-kr", "font-cjk-sc"),
}
