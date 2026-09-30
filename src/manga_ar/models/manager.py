"""Single place that resolves, downloads, verifies, caches and lazily loads models.

Downloads use retry with exponential backoff, resume partial ``.part`` files with HTTP
Range requests, check free disk space and verify SHA-256. ``offline=True`` never touches
the network and raises :class:`ModelUnavailableError` with manual-download instructions.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TypeVar

from manga_ar.errors import ModelUnavailableError
from manga_ar.logging_setup import get_logger
from manga_ar.models.registry import GROUPS, REGISTRY, ModelSpec

log = get_logger(__name__)
T = TypeVar("T")


class _Response(Protocol):
    status_code: int
    headers: Any

    def iter_content(self, chunk_size: int) -> Any: ...

    def close(self) -> None: ...


class HttpSession(Protocol):
    def get(self, url: str, *, headers: dict[str, str], stream: bool, timeout: Any) -> Any: ...


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelManager:
    """Resolve and provide models; thread-safe lazy loading."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        offline: bool = False,
        session: HttpSession | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_attempts: int = 4,
        rng: random.Random | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.models_dir = cache_dir / "models"
        self.offline = offline
        self._session = session
        self._sleep = sleep
        self._max_attempts = max_attempts
        self._rng = rng or random.Random(0)
        self._loaded: dict[str, Any] = {}
        self._lock = threading.RLock()

    # ----------------------------------------------------------------- paths
    def spec(self, name: str) -> ModelSpec:
        try:
            return REGISTRY[name]
        except KeyError as exc:
            raise ModelUnavailableError(
                f"unknown model {name!r}; known: {sorted(REGISTRY)}"
            ) from exc

    def path(self, name: str) -> Path:
        spec = self.spec(name)
        if spec.kind == "file":
            return self.models_dir / spec.filename
        if spec.kind == "hf":
            return self.models_dir / "hf" / spec.repo_id.replace("/", "--")
        if spec.kind == "easyocr":
            return self.models_dir / "easyocr"
        return self.models_dir / "bundled" / name

    def is_present(self, name: str) -> bool:
        spec = self.spec(name)
        path = self.path(name)
        if spec.kind == "file":
            return path.is_file() and (not spec.size or path.stat().st_size == spec.size)
        if spec.kind == "hf":
            return (path / "config.json").is_file()
        if spec.kind == "easyocr":
            return path.is_dir() and any(path.glob("*.pth"))
        return _bundled_available(name)

    # -------------------------------------------------------------- resolve
    def ensure(self, name: str) -> Path:
        """Return a local path for ``name``, downloading it if allowed and needed."""
        spec = self.spec(name)
        if spec.kind == "file":
            return self._ensure_file(spec)
        if spec.kind == "hf":
            return self._ensure_hf(spec)
        path = self.path(name)
        if spec.kind == "easyocr":
            path.mkdir(parents=True, exist_ok=True)
            return path
        if not _bundled_available(name):
            raise ModelUnavailableError(
                f"model {name!r} ships inside an optional package that is not installed; "
                "install it with: pip install 'manga-arabic-translator[rapid]'"
            )
        return path

    def ensure_group(self, group: str) -> list[Path]:
        names = GROUPS.get(group, (group,))
        return [self.ensure(n) for n in names]

    def _ensure_file(self, spec: ModelSpec) -> Path:
        dest = self.path(spec.name)
        if dest.is_file() and self._verified(dest, spec):
            return dest
        if self.offline:
            raise ModelUnavailableError(self._manual_instructions(spec, "offline mode is on"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(dest.parent).free
        if spec.size and free < int(spec.size * 1.1):
            raise ModelUnavailableError(
                self._manual_instructions(
                    spec, f"not enough disk space ({free} bytes free, {spec.size} needed)"
                )
            )
        last_error = "unknown error"
        for attempt in range(1, self._max_attempts + 1):
            try:
                self._download(spec, dest)
                return dest
            except ModelUnavailableError:
                raise
            except Exception as exc:  # noqa: BLE001 - any transport error is retried
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "download of %s failed (attempt %d/%d): %s",
                    spec.name,
                    attempt,
                    self._max_attempts,
                    last_error,
                )
                if attempt < self._max_attempts:
                    self._sleep(self._rng.uniform(0, min(30.0, 2.0**attempt)))
        raise ModelUnavailableError(self._manual_instructions(spec, last_error))

    def _download(self, spec: ModelSpec, dest: Path) -> None:
        part = dest.with_name(dest.name + ".part")
        offset = part.stat().st_size if part.exists() else 0
        if spec.size and offset > spec.size:
            part.unlink()
            offset = 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        session = self._session or _default_session()
        resp = session.get(spec.url, headers=headers, stream=True, timeout=(10, 60))
        try:
            if resp.status_code == 416:  # range not satisfiable: the part is complete
                pass
            elif resp.status_code == 200:
                offset = 0  # server ignored the range: restart
                with part.open("wb") as fh:
                    for chunk in resp.iter_content(1 << 20):
                        fh.write(chunk)
            elif resp.status_code == 206:
                with part.open("ab") as fh:
                    for chunk in resp.iter_content(1 << 20):
                        fh.write(chunk)
            else:
                raise OSError(f"HTTP {resp.status_code} for {spec.url}")
        finally:
            resp.close()
        digest = sha256_file(part)
        if spec.sha256 and digest != spec.sha256:
            part.unlink(missing_ok=True)
            raise OSError(f"sha256 mismatch for {spec.name}: got {digest}")
        part.replace(dest)
        self._mark_verified(dest)
        log.info("downloaded %s → %s", spec.name, dest)

    def _verified(self, path: Path, spec: ModelSpec) -> bool:
        """Hash once, then trust a marker keyed by size+mtime (avoids re-hashing 200 MB)."""
        if spec.size and path.stat().st_size != spec.size:
            return False
        marker = path.with_name(path.name + ".verified")
        stat = path.stat()
        stamp = f"{stat.st_size}:{stat.st_mtime_ns}:{spec.sha256}"
        if marker.is_file() and marker.read_text(encoding="utf-8") == stamp:
            return True
        if spec.sha256 and sha256_file(path) != spec.sha256:
            log.warning("cached %s failed checksum verification; re-downloading", spec.name)
            return False
        marker.write_text(stamp, encoding="utf-8")
        return True

    def _mark_verified(self, path: Path) -> None:
        spec = next((s for s in REGISTRY.values() if self.path(s.name) == path), None)
        if spec is not None:
            self._verified(path, spec)

    def _ensure_hf(self, spec: ModelSpec) -> Path:
        dest = self.path(spec.name)
        if (dest / "config.json").is_file():
            return dest
        if self.offline:
            raise ModelUnavailableError(self._manual_instructions(spec, "offline mode is on"))
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ModelUnavailableError("huggingface_hub is not installed") from exc
        try:
            snapshot_download(
                repo_id=spec.repo_id,
                revision=spec.revision,
                local_dir=str(dest),
                allow_patterns=[
                    "*.json",
                    "*.txt",
                    "*.model",
                    "*.spm",
                    "*.safetensors",
                    "vocab*",
                    "source*",
                    "target*",
                ],
            )
        except Exception as exc:  # noqa: BLE001 - hub raises many transport/HTTP types
            raise ModelUnavailableError(
                self._manual_instructions(spec, f"{type(exc).__name__}: {exc}")
            ) from exc
        if not (dest / "config.json").is_file():
            raise ModelUnavailableError(self._manual_instructions(spec, "config.json missing"))
        return dest

    def _manual_instructions(self, spec: ModelSpec, reason: str) -> str:
        dest = self.path(spec.name)
        if spec.kind == "hf":
            how = (
                f"download the repository https://huggingface.co/{spec.repo_id} (safetensors, "
                f"json, tokenizer files) into {dest}"
            )
        else:
            how = f"download {spec.url} and save it as {dest} (sha256 {spec.sha256})"
        return f"model {spec.name!r} unavailable: {reason}. To install it manually, {how}."

    # ------------------------------------------------------------------ load
    def load(self, name: str, loader: Callable[[Path], T]) -> T:
        """Resolve ``name`` and build it once with ``loader(path)``; cached thereafter."""
        with self._lock:
            if name in self._loaded:
                return self._loaded[name]  # type: ignore[no-any-return]
            path = self.ensure(name)
            obj = loader(path)
            self._loaded[name] = obj
            return obj

    def unload(self, name: str) -> None:
        with self._lock:
            self._loaded.pop(name, None)

    def status(self) -> list[dict[str, Any]]:
        rows = []
        for name, spec in REGISTRY.items():
            rows.append(
                {
                    "name": name,
                    "kind": spec.kind,
                    "present": self.is_present(name),
                    "license": spec.license,
                    "copyleft_or_nc": spec.copyleft_or_nc,
                    "path": str(self.path(name)),
                    "description": spec.description,
                }
            )
        return rows

    def verify(self, name: str) -> tuple[bool, str]:
        spec = self.spec(name)
        path = self.path(name)
        if spec.kind != "file":
            return self.is_present(name), "presence check only"
        if not path.is_file():
            return False, "missing"
        digest = sha256_file(path)
        ok = not spec.sha256 or digest == spec.sha256
        return ok, digest

    def dump_status(self) -> str:
        return json.dumps(self.status(), indent=1, ensure_ascii=False)


def _bundled_available(name: str) -> bool:
    import importlib.util

    module = {"rapid": "rapidocr_onnxruntime"}.get(name, name)
    return importlib.util.find_spec(module) is not None


def _default_session() -> HttpSession:
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = "manga-arabic-translator (model download)"
    return session
