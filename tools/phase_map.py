"""Phase path sets and dependency rules enforced by the governance tools (docs/PHASE_MAP.md).

Single source of truth for `tools/audit_phase_order.py` and `tools/assert_gate.py`.
"""

from __future__ import annotations

import re
import sys
from fnmatch import fnmatch
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 (the project supports >= 3.10)
    import tomli as tomllib

GATE_TAG = "phase1-gate-passed"
APPROVAL_FILE = ".gates/phase1.approved"
GATE_JSON = "benchmarks/results/phase1_gate.json"
APPROVAL_PHRASE = "PHASE 1 APPROVED"

# Phase 2 (Edu-Reader) paths: none may exist in any commit before the gate tag.
PHASE2_PATHS = (
    "src/manga_ar/reader/*",  # reader server, viewer and its static assets
    "src/manga_ar/edu/*",  # NLP/dictionary core and Edu-Card
    "data/*",  # Phase 2 lexicons (e.g. data/sfx_lexicon_en_ar.tsv)
    "web/*",  # standalone viewer assets, if split out
    "tests/edu/*",
    "tests/reader/*",
    "docs/PLAN_PHASE2.md",
)

# Files whose change after the approved commit invalidates Gate 1 (they produce results).
PHASE1_SCOPED = ("src/manga_ar/*", "benchmarks/*", "tools/validate_phase1.py", "configs/*")
PHASE1_UNSCOPED = (
    *PHASE2_PATHS,
    "src/manga_ar/ui/*",  # existing GUI: [compat] changes only, no results impact
    "benchmarks/results/*",  # gate outputs and logs themselves
    "benchmarks/cache/*",
)

# Dependencies that belong to Phase 2 and may not appear before the gate tag.
FORBIDDEN_ANYWHERE = frozenset({"spacy", "wordfreq", "lemminflect", "pyinflect", "wn"})
FORBIDDEN_ANYWHERE_PREFIXES = ("en-core-web-", "spacy-")
FORBIDDEN_DIRECT = frozenset({"fastapi", "uvicorn", "starlette"})  # reader server stack
BENCH_ONLY = frozenset({"nltk"})  # METEOR only, in the [bench] extra
FORBIDDEN_EXTRAS = frozenset({"edu"})

_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def load_toml(text: str) -> dict[str, Any]:
    return tomllib.loads(text)


def _match(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch(path, p) or path == p.rstrip("/*") for p in patterns)


def is_phase2_path(path: str) -> bool:
    return _match(path, PHASE2_PATHS)


def is_phase1_scoped(path: str) -> bool:
    return _match(path, PHASE1_SCOPED) and not _match(path, PHASE1_UNSCOPED)


def dist_name(requirement: str) -> str:
    m = _NAME.match(requirement)
    return re.sub(r"[-_.]+", "-", m.group(1)).lower() if m else ""


def _forbidden(name: str) -> bool:
    return name in FORBIDDEN_ANYWHERE or name.startswith(FORBIDDEN_ANYWHERE_PREFIXES)


def pyproject_violations(pyproject: dict[str, Any]) -> list[str]:
    """Phase 2 dependencies declared in pyproject.toml (any group)."""
    project = pyproject.get("project", {})
    groups: dict[str, list[str]] = {"dependencies": list(project.get("dependencies", []))}
    for extra, reqs in project.get("optional-dependencies", {}).items():
        groups[f"[{extra}]"] = list(reqs)
    out = [f"extra {e!r} exists" for e in project.get("optional-dependencies", {})
           if e in FORBIDDEN_EXTRAS]  # fmt: skip
    for group, reqs in groups.items():
        for req in reqs:
            name = dist_name(req)
            if _forbidden(name) or name in FORBIDDEN_DIRECT:
                out.append(f"{name} declared in {group}")
            elif name in BENCH_ONLY and group != "[bench]":
                out.append(f"{name} declared outside [bench] ({group})")
    return out


def lock_violations(lock: dict[str, Any]) -> list[str]:
    """Phase 2 packages resolved anywhere in uv.lock (transitive included)."""
    names = {dist_name(p.get("name", "")) for p in lock.get("package", [])}
    return [f"{n} in uv.lock" for n in sorted(names) if _forbidden(n)]
