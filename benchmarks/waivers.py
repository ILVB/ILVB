"""Human waivers for Gate 1 (benchmarks/waivers.yaml) and how a check's status is decided."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
WAIVERS = ROOT / "benchmarks" / "waivers.yaml"
HEAVY_LOCK = ROOT / "benchmarks" / "heavy_weights.lock.json"
Status = Literal["PASS", "FAIL", "WAIVED"]


class Waiver(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    granted: str
    by: str
    text: str
    status: str
    models: list[str]
    lifted_when: str
    checks: dict[str, str]

    def active(self, lock: Path = HEAVY_LOCK) -> bool:
        """Active until every listed model is recorded in the heavy-weights lock."""
        pinned = json.loads(lock.read_text("utf-8")) if lock.is_file() else {}
        return not all(m in pinned for m in self.models)


def load(path: Path = WAIVERS) -> list[Waiver]:
    data = yaml.safe_load(path.read_text("utf-8"))
    if data.get("schema") != 1:
        raise ValueError(f"{path}: unsupported schema")
    return [Waiver.model_validate(w) for w in data["waivers"]]


def decide(check: str, measured: bool | None, waivers: list[Waiver],
           lock: Path = HEAVY_LOCK) -> tuple[Status, str]:  # fmt: skip
    """Status of one gate check. ``measured`` is the verdict, or None if not measurable.

    Not measurable → WAIVED only under an active waiver naming the check, else FAIL
    (NOT-MEASURABLE). A measured check always keeps its real verdict.
    """
    if measured is not None:
        return ("PASS" if measured else "FAIL"), ""
    for w in waivers:
        if check in w.checks and w.active(lock):
            return "WAIVED", f"{w.status} (waived: {w.id}, {w.granted})"
    return "FAIL", "NOT-MEASURABLE"
