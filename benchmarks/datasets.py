"""Dataset access with a sealed test split (step 0.3.3).

dev (tuning) and val (model selection) load freely. The test split loads only with a
`SealedCapability`, which only `tools/validate_phase1.py` issues (and logs). Content is
verified against the committed manifest's SHA-256 values on every load.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.schema import GtPage

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "benchmarks" / "manifests"
DATA_DIR = ROOT / "benchmarks" / "cache"


class SealedSplitError(PermissionError):
    pass


class ManifestError(RuntimeError):
    pass


class SealedCapability:
    """Proof that the caller is the gate. Construct only via ``_issue`` (gate script)."""

    __slots__ = ("purpose",)
    _token = object()

    def __init__(self, token: object, purpose: str) -> None:
        if token is not SealedCapability._token:
            raise SealedSplitError("the test split is sealed; only the gate may open it")
        self.purpose = purpose


def _issue(purpose: str) -> SealedCapability:  # called by tools/validate_phase1.py only
    return SealedCapability(SealedCapability._token, purpose)


@dataclass(frozen=True)
class Item:
    page_id: str
    split: str
    category: str
    lang: str
    image: Path
    clean: Path
    gt_path: Path

    def gt(self) -> GtPage:
        return GtPage.model_validate_json(self.gt_path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(name: str = "synthetic_v1") -> dict[str, Any]:
    data: dict[str, Any] = json.loads((MANIFEST_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return data


def load_split(
    split: str, name: str = "synthetic_v1", capability: SealedCapability | None = None,
    verify: bool = True,
) -> list[Item]:  # fmt: skip
    if split == "test" and not isinstance(capability, SealedCapability):
        raise SealedSplitError("the test split is sealed; only the gate may open it")
    man = manifest(name)
    root = DATA_DIR / name / split
    items = []
    for entry in man["splits"][split]["pages"]:
        pid = entry["page_id"]
        paths = {k: root / entry["files"][k]["path"] for k in ("image", "clean", "gt")}
        if verify:
            for key, path in paths.items():
                if not path.is_file():
                    raise ManifestError(
                        f"{path} missing: run python -m benchmarks.generators.build"
                    )
                if sha256(path) != entry["files"][key]["sha256"]:
                    raise ManifestError(f"{path} does not match the manifest (hash)")
        items.append(Item(pid, split, entry["category"], entry["lang"],
                          paths["image"], paths["clean"], paths["gt"]))  # fmt: skip
    return items
