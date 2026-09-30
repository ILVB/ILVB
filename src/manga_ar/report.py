"""Batch report: per-page status, warnings and timings → ``report.json`` + ``report.md``.

Exit codes (E18): 0 = every page produced an output; 2 = partial success (some pages
failed or were skipped); 1 = nothing was produced (all failed, or a fatal error).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from manga_ar import __version__
from manga_ar.io.writer import atomic_write_text

EXIT_OK, EXIT_FATAL, EXIT_PARTIAL = 0, 1, 2
PageStatus = Literal["ok", "degraded", "resumed", "no_text", "failed", "skipped", "cancelled"]
SUCCESS: frozenset[str] = frozenset({"ok", "degraded", "resumed", "no_text"})


@dataclass
class PageOutcome:
    name: str
    status: PageStatus
    output: str | None = None
    sidecar: str | None = None
    lang: str | None = None
    regions: int = 0
    translated: int = 0
    flags: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status in SUCCESS


@dataclass
class BatchReport:
    config_hash: str
    preset: str
    output_dir: str
    pages: list[PageOutcome] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    archives: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    cancelled: bool = False
    version: str = __version__

    def add(self, outcome: PageOutcome) -> None:
        self.pages.append(outcome)

    def note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(p.status for p in self.pages))

    @property
    def exit_code(self) -> int:
        if not self.pages:
            return EXIT_FATAL
        ok = sum(p.succeeded for p in self.pages)
        if ok == 0:
            return EXIT_FATAL
        if ok < len(self.pages) or self.cancelled:
            return EXIT_PARTIAL
        return EXIT_OK

    # ----------------------------------------------------------------- output
    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["counts"] = self.counts
        data["exit_code"] = self.exit_code
        return data

    def to_markdown(self) -> str:
        lines = [
            "# MangaAR batch report",
            "",
            f"- MangaAR {self.version}, preset `{self.preset}`, config hash "
            f"`{self.config_hash[:12]}`",
            f"- Output: `{self.output_dir}`",
            f"- Pages: {len(self.pages)} ("
            + ", ".join(f"{k}: {v}" for k, v in sorted(self.counts.items()))
            + f"), exit code {self.exit_code}, {self.elapsed_s:.1f} s",
        ]
        if self.cancelled:
            lines.append("- The run was cancelled; remaining pages were not processed.")
        if self.archives:
            lines.append("- Archives written: " + ", ".join(f"`{a}`" for a in self.archives))
        if self.notes:
            lines += ["", "## Notes", ""] + [f"- {n}" for n in self.notes]
        lines += [
            "",
            "## Pages",
            "",
            "| page | status | lang | regions | translated | flags | time |",
            "|---|---|---|---|---|---|---|",
        ]
        for p in self.pages:
            flags = ", ".join(f"{k}×{v}" for k, v in sorted(p.flags.items())) or "—"
            total = sum(p.timings.values())
            lines.append(
                f"| {_md(p.name)} | {p.status} | {p.lang or '—'} | {p.regions} | "
                f"{p.translated} | {flags} | {total:.2f} s |"
            )
        problems = [p for p in self.pages if p.error or p.warnings]
        if problems:
            lines += ["", "## Errors and warnings", ""]
            for p in problems:
                if p.error:
                    lines.append(f"- **{_md(p.name)}**: {_md(p.error)}")
                for w in p.warnings:
                    lines.append(f"- {_md(p.name)}: {_md(w)}")
        return "\n".join(lines) + "\n"

    def save(self, directory: Path) -> tuple[Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        jpath, mpath = directory / "report.json", directory / "report.md"
        atomic_write_text(jpath, json.dumps(self.to_json(), ensure_ascii=False, indent=1))
        atomic_write_text(mpath, self.to_markdown())
        return jpath, mpath


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")
