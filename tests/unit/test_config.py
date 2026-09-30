from __future__ import annotations

from pathlib import Path

import pytest

from manga_ar.config import default_config_path, env_overrides, load_config
from manga_ar.errors import ConfigError

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.usefixtures("clean_env")


def test_defaults_load_and_validate() -> None:
    cfg = load_config(environ={})
    assert cfg.preset == "balanced"
    assert cfg.translate.providers[0] == "tm"
    assert cfg.translate.chunk_chars["google"] == 4500
    assert cfg.typeset.font == "NotoNaskhArabic"


def test_repo_copy_of_defaults_is_identical() -> None:
    assert (ROOT / "configs" / "default.yaml").read_bytes() == default_config_path().read_bytes()


def test_preset_overlay_applies_before_user_files(tmp_path: Path) -> None:
    fast = load_config(overrides={"preset": "fast"}, environ={})
    assert fast.detect.detector == "classical" and fast.inpaint.use_lama is False
    user = tmp_path / "u.yaml"
    user.write_text("preset: fast\ninpaint:\n  use_lama: true\n", encoding="utf-8")
    cfg = load_config(files=[user], environ={})
    assert cfg.preset == "fast" and cfg.inpaint.use_lama is True  # file beats preset
    quality = load_config(overrides={"preset": "quality"}, environ={})
    assert quality.inpaint.residual_check and quality.ocr.verify_multi_engine


def test_layer_order_file_env_cli(tmp_path: Path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text(
        "translate:\n  rate_per_sec: 0.5\n  chunk_chars:\n    google: 3000\n", encoding="utf-8"
    )
    env = {"MANGAAR__translate__rate_per_sec": "0.25", "MANGAAR_CACHE_DIR": str(tmp_path)}
    cfg = load_config(files=[f], environ=env)
    assert cfg.translate.rate_per_sec == 0.25  # env beats file
    assert cfg.translate.chunk_chars["google"] == 3000  # dict leaves merge key-wise
    assert cfg.translate.chunk_chars["mymemory"] == 450
    assert cfg.runtime.cache_dir == str(tmp_path)
    cfg2 = load_config(files=[f], environ=env, overrides={"translate.rate_per_sec": 2.0})
    assert cfg2.translate.rate_per_sec == 2.0  # CLI beats env
    cfg3 = load_config(environ=env, overrides={"translate.rate_per_sec": None})
    assert cfg3.translate.rate_per_sec == 0.25  # None means "flag not given"


def test_env_parsing() -> None:
    out = env_overrides(
        {
            "MANGAAR_OFFLINE": "true",
            "MANGAAR_DEVICE": "cpu",
            "MANGAAR__typeset__font": "Amiri",
            "OTHER": "x",
        }
    )
    assert out == {"runtime": {"offline": True, "device": "cpu"}, "typeset": {"font": "Amiri"}}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"translate.bogus": 1}, "unknown keys"),
        ({"translate.rate_per_sec": "fast"}, "expected a number"),
        ({"runtime.device": "tpu"}, "not in"),
        ({"output.jpeg_quality": 50}, "jpeg_quality"),
        ({"translate.providers": ["google", "google"]}, "duplicates"),
        ({"translate.providers": ["deepl"]}, "unknown providers"),
        ({"translate.chunk_chars": {"google": 5000}}, "< 5000"),
        ({"preset": "ultra"}, "unknown preset"),
        ({"ocr.engines": {"ja": ["tesseract"]}}, "unknown engines"),
        ({"runtime.offline": "yes"}, "true/false"),
    ],
)
def test_invalid_values_rejected(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(overrides=overrides, environ={})


def test_invalid_yaml_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("translate: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(files=[bad], environ={})
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(files=[tmp_path / "missing.yaml"], environ={})


def test_hash_ignores_runtime_only_keys_and_tracks_output_keys() -> None:
    base = load_config(environ={})
    same = load_config(
        overrides={"runtime.debug": True, "runtime.device": "cpu", "output.force": True}, environ={}
    )
    other = load_config(overrides={"typeset.font": "Amiri"}, environ={})
    assert base.config_hash() == same.config_hash()
    assert base.config_hash() != other.config_hash()
    assert len(base.config_hash()) == 64


def test_replace_revalidates() -> None:
    cfg = load_config(environ={})
    assert cfg.replace(**{"typeset.font": "Cairo"}).typeset.font == "Cairo"
    with pytest.raises(ConfigError):
        cfg.replace(**{"typeset.line_spacing": 0.5})
