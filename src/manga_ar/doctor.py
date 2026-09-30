"""Environment self-check (``manga-arabic doctor``)."""

from __future__ import annotations

import importlib.metadata as md
import platform
import sys
from dataclasses import dataclass, field

from manga_ar import __version__
from manga_ar.config import AppConfig, resolve_cache_dir
from manga_ar.models.device import resolve_device
from manga_ar.models.manager import ModelManager
from manga_ar.typeset.fonts import FontRegistry

OPENCV_DISTS = (
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
)
OPTIONAL_PACKAGES = (
    "torch",
    "onnxruntime",
    "easyocr",
    "rapidocr_onnxruntime",
    "manga-ocr",
    "paddleocr",
    "transformers",
    "gradio",
)
REACHABILITY = {
    "pypi": "https://pypi.org/simple/",
    "github": "https://github.com/",
    "huggingface": "https://huggingface.co/",
    "google-translate": "https://translate.google.com/",
    "mymemory": "https://api.mymemory.translated.net/",
}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    essential: bool = False


@dataclass
class DoctorReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks if c.essential)

    def render(self) -> str:
        lines = []
        for c in self.checks:
            mark = "OK  " if c.ok else ("FAIL" if c.essential else "WARN")
            lines.append(f"[{mark}] {c.name}: {c.detail}")
        lines.append("")
        lines.append("overall: " + ("OK" if self.ok else "PROBLEMS FOUND (see FAIL lines)"))
        return "\n".join(lines)


def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def run_doctor(cfg: AppConfig, check_network: bool = True) -> DoctorReport:
    rep = DoctorReport()
    rep.checks.append(
        Check(
            "python",
            (3, 10) <= sys.version_info[:2] < (3, 13),
            f"{platform.python_version()} on {platform.system()} {platform.machine()}",
            essential=True,
        )
    )
    rep.checks.append(Check("manga-arabic", True, __version__))
    for dist in (
        "numpy",
        "pillow",
        "fonttools",
        "arabic-reshaper",
        "python-bidi",
        "deep-translator",
    ):
        v = _version(dist)
        rep.checks.append(Check(dist, v is not None, v or "not installed", essential=True))
    installed_cv = [d for d in OPENCV_DISTS if _version(d)]
    rep.checks.append(
        Check(
            "opencv distributions",
            len(installed_cv) == 1,
            ", ".join(f"{d}=={_version(d)}" for d in installed_cv)
            or "none installed"
            + (
                ""
                if len(installed_cv) <= 1
                else " — conflict! run: pip uninstall -y "
                + " ".join(installed_cv)
                + " && pip install opencv-python-headless"
            ),
            essential=True,
        )
    )
    try:
        import cv2

        rep.checks.append(Check("cv2 import", True, cv2.__version__, essential=True))
    except ImportError as exc:
        rep.checks.append(Check("cv2 import", False, str(exc), essential=True))
    from PIL import features

    raqm = bool(features.check("raqm"))
    rep.checks.append(
        Check("libraqm (RAQM layout)", True, "available" if raqm else "not available (BASIC only)")
    )
    for dist in OPTIONAL_PACKAGES:
        v = _version(dist)
        rep.checks.append(Check(f"optional {dist}", v is not None, v or "not installed"))
    rep.checks.append(
        Check("device", True, f"{cfg.runtime.device} -> {resolve_device(cfg.runtime.device)}")
    )
    cache = resolve_cache_dir(cfg)
    rep.checks.append(Check("cache dir", True, str(cache)))
    manager = ModelManager(cache, offline=True)
    for row in manager.status():
        note = " [copyleft/non-commercial, opt-in]" if row["copyleft_or_nc"] else ""
        rep.checks.append(
            Check(
                f"model {row['name']}",
                bool(row["present"]),
                ("present" if row["present"] else "not downloaded") + note,
            )
        )
    reg = FontRegistry()
    primary = cfg.typeset.font
    fonts_ok = primary in reg.fonts and reg.fonts[primary].basic_ok
    rep.checks.append(
        Check(
            "primary font",
            fonts_ok,
            f"{primary}: "
            + ("covers Arabic presentation forms" if fonts_ok else "missing/unsuitable"),
            essential=True,
        )
    )
    usable = [k for k, f in reg.fonts.items() if f.role == "arabic" and f.basic_ok]
    rep.checks.append(Check("Arabic fonts (BASIC path)", bool(usable), ", ".join(usable)))
    sym = [k for k, f in reg.fonts.items() if f.role == "symbol"]
    rep.checks.append(Check("symbol fonts", bool(sym), ", ".join(sym)))
    if check_network and not cfg.runtime.offline:
        rep.checks.extend(_reachability())
    else:
        rep.checks.append(Check("network", True, "skipped (offline)"))
    return rep


def _reachability() -> list[Check]:
    import requests

    out = []
    for name, url in REACHABILITY.items():
        try:
            resp = requests.head(url, timeout=(4, 6), allow_redirects=True)
            ok = resp.status_code < 500
            detail = f"HTTP {resp.status_code}"
        except requests.RequestException as exc:
            ok = False
            detail = f"unreachable ({type(exc).__name__})"
        out.append(Check(f"reach {name}", ok, detail))
    return out
