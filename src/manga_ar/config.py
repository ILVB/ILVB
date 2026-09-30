"""Layered, validated, hashable configuration.

Layers (later wins): packaged ``default.yaml`` → preset overlay → user YAML files →
environment → explicit overrides (CLI). The merged mapping is converted into frozen
dataclasses with strict type checking; unknown keys raise :class:`ConfigError`.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml

from manga_ar import __version__
from manga_ar.errors import ConfigError

Preset = Literal["fast", "balanced", "quality"]
Lang = Literal["auto", "ja", "ko", "zh"]
ReadingOrder = Literal["auto", "manga_rtl", "comic_ltr", "webtoon_ttb"]

# Preset overlays are applied on top of the defaults, before user files.
PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "detect": {"detector": "classical"},
        "inpaint": {"use_lama": False, "residual_check": False},
        "ocr": {"verify_multi_engine": False},
    },
    "balanced": {},
    "quality": {
        "inpaint": {"use_lama": True, "residual_check": True},
        "ocr": {"verify_multi_engine": True},
    },
}

# Keys that never influence the produced pixels/text; excluded from the config hash.
_HASH_EXCLUDED: dict[str, set[str]] = {
    "runtime": {"device", "debug", "cache_dir", "log_level", "num_threads", "offline"},
    "output": {"resume", "force"},
}


@dataclass(frozen=True)
class RuntimeConfig:
    device: Literal["auto", "cpu", "cuda", "mps"]
    offline: bool
    debug: bool
    seed: int
    cache_dir: str | None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    num_threads: int

    def __post_init__(self) -> None:
        _check(self.num_threads >= 0, "runtime.num_threads must be >= 0")


@dataclass(frozen=True)
class InputConfig:
    source_lang: Lang
    reading_order: ReadingOrder
    max_pixels: int
    min_side: int
    max_side: int
    truncated: Literal["strict", "lenient"]
    archive_max_members: int
    archive_max_member_mb: int
    archive_max_total_mb: int

    def __post_init__(self) -> None:
        _check(self.max_pixels > 0, "input.max_pixels must be > 0")
        _check(0 < self.min_side <= self.max_side, "input.min_side must be in (0, max_side]")
        _check(self.archive_max_members > 0, "input.archive_max_members must be > 0")


@dataclass(frozen=True)
class TilingConfig:
    enabled: bool
    aspect_trigger: float
    height_trigger: int
    tile_height: int
    overlap: int

    def __post_init__(self) -> None:
        _check(self.tile_height >= 256, "tiling.tile_height must be >= 256")
        _check(0 <= self.overlap < self.tile_height // 2, "tiling.overlap must be < tile_height/2")


@dataclass(frozen=True)
class DetectConfig:
    detector: Literal["auto", "classical", "hybrid", "rapid", "craft", "ctd"]
    sfx: Literal["skip", "translate"]
    min_glyph_px: int
    max_glyph_frac: float
    min_region_area: int
    nms_iou: float
    bubble_roi_scale: float
    leak_max_area_ratio: float
    leak_max_page_fraction: float
    leak_min_solidity: float
    suppress_furigana: bool

    def __post_init__(self) -> None:
        _check(0 < self.nms_iou <= 1, "detect.nms_iou must be in (0, 1]")
        _check(0 < self.leak_max_page_fraction <= 1, "detect.leak_max_page_fraction in (0, 1]")
        _check(0 < self.max_glyph_frac <= 1, "detect.max_glyph_frac must be in (0, 1]")


@dataclass(frozen=True)
class OcrConfig:
    engines: dict[str, list[str]]
    padding: float
    min_confidence: float
    max_upscale: float
    min_crop_px: int
    preserve_decorations: bool
    verify_multi_engine: bool
    pass_through_latin: bool

    def __post_init__(self) -> None:
        known = {"manga_ocr", "easyocr", "rapid", "paddle"}
        for lang, names in self.engines.items():
            _check(lang in {"ja", "ko", "zh"}, f"ocr.engines: unknown language {lang!r}")
            bad = set(names) - known
            _check(not bad, f"ocr.engines.{lang}: unknown engines {sorted(bad)}")
        _check(0 <= self.padding <= 0.5, "ocr.padding must be in [0, 0.5]")
        _check(1.0 <= self.max_upscale <= 3.0, "ocr.max_upscale must be in [1, 3]")


@dataclass(frozen=True)
class InpaintConfig:
    strategy: Literal["auto", "solid", "telea", "ns", "lama"]
    use_lama: bool
    uniform_std: float
    gradient_residual_std: float
    dilate_min: int
    dilate_max: int
    allowed_erode: int
    feather_px: int
    lama_context: int
    lama_max_side: int
    residual_check: bool
    residual_max_passes: int
    erase_untranslated: bool

    def __post_init__(self) -> None:
        _check(0 <= self.dilate_min <= self.dilate_max, "inpaint.dilate_min <= dilate_max")
        _check(self.allowed_erode >= 0, "inpaint.allowed_erode must be >= 0")
        _check(self.lama_max_side >= 128, "inpaint.lama_max_side must be >= 128")


@dataclass(frozen=True)
class TranslateConfig:
    providers: list[str]
    target: str
    tm_file: str | None
    glossary_file: str | None
    libretranslate_url: str | None
    local_model: Literal["marian", "m2m100"]
    proxy: str | None
    connect_timeout: float
    total_timeout: float
    rate_per_sec: float
    burst: int
    backoff_base: float
    backoff_factor: float
    backoff_cap: float
    max_attempts: int
    breaker_threshold: int
    breaker_cooldown: float
    batch: bool
    chunk_chars: dict[str, int]
    cache: bool
    min_arabic_ratio: float
    max_length_ratio: float

    def __post_init__(self) -> None:
        known = {"tm", "google", "mymemory", "libretranslate", "local"}
        bad = set(self.providers) - known
        _check(not bad, f"translate.providers: unknown providers {sorted(bad)}")
        _check(len(self.providers) == len(set(self.providers)), "translate.providers: duplicates")
        _check(self.target == "ar", "translate.target must be 'ar'")
        _check(0 < self.connect_timeout <= self.total_timeout, "connect_timeout <= total_timeout")
        _check(self.rate_per_sec > 0 and self.burst >= 1, "translate rate/burst must be positive")
        _check(self.max_attempts >= 1, "translate.max_attempts must be >= 1")
        _check(self.breaker_threshold >= 1, "translate.breaker_threshold must be >= 1")
        _check(0 < self.min_arabic_ratio <= 1, "translate.min_arabic_ratio must be in (0, 1]")
        for name, limit in self.chunk_chars.items():
            _check(limit > 0, f"translate.chunk_chars.{name} must be > 0")
        _check(self.chunk_chars.get("google", 0) < 5000, "google chunk limit must be < 5000")
        _check(self.chunk_chars.get("mymemory", 0) < 500, "mymemory chunk limit must be < 500")


@dataclass(frozen=True)
class TypesetConfig:
    font: str
    fallback_fonts: list[str]
    symbol_fonts: list[str]
    render_path: Literal["basic", "raqm"]
    line_spacing: float
    line_spacing_floor: float
    min_size_px: int
    min_size_page_frac: float
    max_size_page_frac: float
    max_size_safe_frac: float
    padding_frac: float
    padding_min_px: int
    alignment: Literal["center", "right"]
    digits: Literal["western", "arabic_indic"]
    outline_frac: float
    shadow: bool
    contrast_target: float
    strip_harakat: bool
    strategy: Literal["shape", "rect"]

    def __post_init__(self) -> None:
        _check(self.line_spacing_floor >= 1.0, "typeset.line_spacing_floor must be >= 1.0")
        _check(
            self.line_spacing >= self.line_spacing_floor, "typeset.line_spacing must be >= floor"
        )
        _check(self.min_size_px >= 6, "typeset.min_size_px must be >= 6")
        _check(0 < self.max_size_safe_frac <= 1, "typeset.max_size_safe_frac must be in (0, 1]")


@dataclass(frozen=True)
class OutputConfig:
    format: Literal["png", "jpg", "webp", "cbz"]
    jpeg_quality: int
    webp_quality: int
    suffix: str
    write_sidecar: bool
    keep_icc: bool
    resume: bool
    force: bool

    def __post_init__(self) -> None:
        _check(92 <= self.jpeg_quality <= 100, "output.jpeg_quality must be in [92, 100]")
        _check(1 <= self.webp_quality <= 100, "output.webp_quality must be in [1, 100]")
        _check(bool(self.suffix) and "/" not in self.suffix, "output.suffix must be a name part")


@dataclass(frozen=True)
class AppConfig:
    """The complete, validated configuration of one run."""

    schema: int
    preset: Preset
    runtime: RuntimeConfig
    input: InputConfig
    tiling: TilingConfig
    detect: DetectConfig
    ocr: OcrConfig
    inpaint: InpaintConfig
    translate: TranslateConfig
    typeset: TypesetConfig
    output: OutputConfig
    sources: tuple[str, ...] = field(default=(), compare=False)

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data.pop("sources", None)
        return data

    def config_hash(self) -> str:
        """SHA-256 over output-affecting settings plus the MangaAR version (E19)."""
        data = self.to_dict()
        for section, keys in _HASH_EXCLUDED.items():
            for key in keys:
                data[section].pop(key, None)
        data["_version"] = __version__
        blob = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def replace(self, **overrides: Any) -> AppConfig:
        """Return a copy with dotted-key overrides applied and re-validated."""
        data = self.to_dict()
        _deep_merge(data, _dotted_to_nested(overrides))
        return build_config(data, sources=self.sources)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def default_config_path() -> Path:
    return Path(str(resources.files("manga_ar") / "data" / "default.yaml"))


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must contain a mapping at top level")
    return data


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> None:
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            # dict-valued leaves (ocr.engines, translate.chunk_chars) merge key-wise too
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def _dotted_to_nested(flat: Mapping[str, Any]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for dotted, value in flat.items():
        parts = dotted.split(".")
        node = nested
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return nested


def env_overrides(environ: Mapping[str, str]) -> dict[str, Any]:
    """Translate environment variables into a nested override mapping."""
    out: dict[str, Any] = {}
    if environ.get("MANGAAR_CACHE_DIR"):
        out["runtime.cache_dir"] = environ["MANGAAR_CACHE_DIR"]
    if environ.get("MANGAAR_OFFLINE"):
        out["runtime.offline"] = environ["MANGAAR_OFFLINE"].strip().lower() in {"1", "true", "yes"}
    if environ.get("MANGAAR_DEVICE"):
        out["runtime.device"] = environ["MANGAAR_DEVICE"].strip()
    for key, raw in environ.items():
        if not key.startswith("MANGAAR__"):
            continue
        dotted = ".".join(p.lower() for p in key[len("MANGAAR__") :].split("__") if p)
        if not dotted:
            continue
        try:
            out[dotted] = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ConfigError(f"environment variable {key}: invalid value {raw!r}") from exc
    return _dotted_to_nested(out)


def load_config(
    files: typing.Sequence[str | Path] = (),
    overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Build the run configuration from all layers.

    ``overrides`` uses dotted keys (``{"translate.providers": ["local"]}``); ``None`` values
    are ignored so CLI parsers can pass unset flags straight through.
    """
    environ = os.environ if environ is None else environ
    clean_overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    base = _read_yaml(default_config_path())
    file_layers = [_read_yaml(Path(f)) for f in files]
    env_layer = env_overrides(environ)
    cli_layer = _dotted_to_nested(clean_overrides)

    # The preset is itself a layered value; resolve it first so its overlay sits right
    # above the defaults.
    preset = base.get("preset", "balanced")
    for layer in (*file_layers, env_layer, cli_layer):
        preset = layer.get("preset", preset)
    if preset not in PRESETS:
        raise ConfigError(f"unknown preset {preset!r}; expected one of {sorted(PRESETS)}")
    merged = copy.deepcopy(base)
    _deep_merge(merged, PRESETS[preset])
    for layer in (*file_layers, env_layer, cli_layer):
        _deep_merge(merged, layer)
    merged["preset"] = preset
    return build_config(merged, sources=tuple(str(f) for f in files))


def build_config(data: Mapping[str, Any], sources: tuple[str, ...] = ()) -> AppConfig:
    """Strictly convert a complete mapping into :class:`AppConfig`."""
    obj = _build(AppConfig, dict(data), "config", skip={"sources"})
    return dataclasses.replace(obj, sources=sources)


_T = typing.TypeVar("_T")


def _build(cls: type[_T], data: Any, path: str, skip: set[str] | None = None) -> _T:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path} must be a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.name not in (skip or set())}  # type: ignore[arg-type]
    unknown = set(data) - set(fields)
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
    missing = set(fields) - set(data)
    if missing:
        raise ConfigError(f"{path}: missing keys {sorted(missing)}")
    kwargs = {name: _coerce(hints[name], data[name], f"{path}.{name}") for name in fields}
    return cls(**kwargs)


def _coerce(tp: Any, value: Any, path: str) -> Any:
    origin = get_origin(tp)
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        return _build(tp, value, path)
    if origin is Literal:
        allowed = get_args(tp)
        if value not in allowed:
            raise ConfigError(f"{path}: {value!r} not in {list(allowed)}")
        return value
    if origin in (Union, types.UnionType):
        args = get_args(tp)
        if value is None and type(None) in args:
            return None
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(arg, value, path)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError("; ".join(errors) or f"{path}: invalid value {value!r}")
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        (item_tp,) = get_args(tp)
        return [_coerce(item_tp, v, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        key_tp, val_tp = get_args(tp)
        return {
            _coerce(key_tp, k, f"{path}.<key>"): _coerce(val_tp, v, f"{path}.{k}")
            for k, v in value.items()
        }
    if origin is tuple:
        return tuple(value)
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    raise ConfigError(f"{path}: unsupported type {tp!r}")  # pragma: no cover - schema bug


def resolve_cache_dir(cfg: AppConfig) -> Path:
    """Cache directory: config/env value, else the platformdirs user cache."""
    if cfg.runtime.cache_dir:
        return Path(cfg.runtime.cache_dir).expanduser()
    from platformdirs import user_cache_dir

    return Path(user_cache_dir("manga-arabic", appauthor=False))
