"""Environment capture recorded with every benchmark run (step 0.4, PD-5).

OS, CPU, RAM, GPU/VRAM, Python and library versions, git commit/dirty state, and SHA-256
of every model file in the cache and every generator font. Hashes are memoised by
(path, size, mtime) in benchmarks/cache/ so repeated runs stay fast.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "numpy", "pillow", "opencv-python-headless", "fonttools", "torch", "torchvision",
    "onnxruntime", "rapidocr-onnxruntime", "easyocr", "manga-ocr", "paddleocr",
    "transformers", "ctranslate2", "llama-cpp-python", "sentence-transformers", "faiss-cpu",
    "uharfbuzz", "arabic-reshaper", "python-bidi", "pyphen", "scikit-image", "lpips",
    "sacrebleu", "jiwer", "nltk", "pydantic", "gradio", "deep-translator",
)  # fmt: skip


def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _ram_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 2**20, 2)
    except OSError:
        return None
    return None


def _cpu() -> dict[str, Any]:
    model = platform.processor() or ""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"model": model, "logical_cores": os.cpu_count()}


def _gpu() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda": False, "mps": False, "devices": []}
    try:
        import torch

        info["cuda"] = bool(torch.cuda.is_available())
        mps = getattr(torch.backends, "mps", None)
        info["mps"] = bool(mps is not None and mps.is_available())
        for i in range(torch.cuda.device_count() if info["cuda"] else 0):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append(
                {"name": props.name, "vram_gb": round(props.total_memory / 2**30, 2)}
            )
    except ImportError:
        info["torch"] = "not installed"
    return info


def _git() -> dict[str, Any]:
    def run(*args: str) -> str:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
        return out.stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _hash_files(paths: list[Path], memo_path: Path) -> dict[str, str]:
    memo: dict[str, Any] = {}
    if memo_path.is_file():
        memo = json.loads(memo_path.read_text(encoding="utf-8"))
    out = {}
    for path in sorted(paths):
        st = path.stat()
        key = f"{path}|{st.st_size}|{st.st_mtime_ns}"
        if key not in memo:
            memo[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        out[str(path)] = memo[key]
    memo_path.parent.mkdir(parents=True, exist_ok=True)
    memo_path.write_text(json.dumps(memo), encoding="utf-8")
    return out


def capture() -> dict[str, Any]:
    cache = Path(os.environ.get("MANGAAR_CACHE_DIR", ROOT / ".cache"))
    models_dir = cache / "models"
    model_files = (
        [p for p in models_dir.rglob("*") if p.is_file() and not p.name.endswith(".verified")]
        if models_dir.is_dir()
        else []
    )
    fonts = (
        list((cache / "bench_fonts").glob("*.[ot]tf")) if (cache / "bench_fonts").is_dir() else []
    )
    hashes = _hash_files(model_files + fonts, ROOT / "benchmarks" / "cache" / "hash_memo.json")
    return {
        "os": {"system": platform.system(), "release": platform.release(),
               "machine": platform.machine(), "platform": platform.platform()},
        "cpu": _cpu(),
        "ram_gb": _ram_gb(),
        "gpu": _gpu(),
        "python": sys.version.split()[0],
        "packages": {p: _version(p) for p in PACKAGES},
        "git": _git(),
        "models": {str(Path(k).relative_to(cache)): v for k, v in hashes.items()},
    }  # fmt: skip


def write(path: Path) -> dict[str, Any]:
    env = capture()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return env


if __name__ == "__main__":
    data = write(ROOT / "benchmarks" / "env.json")
    cpu = data["cpu"]
    sys.stdout.write(
        f"{cpu['model']} x{cpu['logical_cores']}, {data['ram_gb']} GB, gpu={data['gpu']}, "
        f"{len(data['models'])} model/font files hashed\n"
    )
