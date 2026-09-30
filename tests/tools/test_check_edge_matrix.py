"""0.5: the edge-case matrix checker, including negative controls."""

from __future__ import annotations

from pathlib import Path

from tools import check_edge_matrix as cem


def _setup(root: Path, rows: dict[str, tuple[str, str]], tests: str) -> None:
    (root / "docs").mkdir()
    lines = ["| ID | Case | Owner | Tests | Status |", "|---|---|---|---|---|"]
    lines += [f"| {eid} | case | 1.0 | {t} | {s} |" for eid, (t, s) in rows.items()]
    (root / "docs" / "EDGE_CASE_MATRIX.md").write_text("\n".join(lines) + "\n", "utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_edge.py").write_text(tests, "utf-8")


def _all_covered() -> tuple[dict[str, tuple[str, str]], str]:
    rows, body = {}, ""
    for eid in cem.IDS:
        name = f"test_e{eid[2:]}_case"
        rows[eid] = (f"`tests/test_edge.py::{name}`", "covered")
        body += f"def {name}() -> None:\n    assert True\n\n"
    return rows, body


def test_id_parsing() -> None:
    assert cem.id_of("test_e05_punctuation_only") == ["E-05"]
    assert cem.id_of("test_e01_e18_combo") == ["E-01", "E-18"]
    assert cem.id_of("test_hello105") == [] and cem.id_of("test_e5_x") == []


def test_full_coverage_passes(tmp_path: Path) -> None:
    rows, body = _all_covered()
    _setup(tmp_path, rows, body)
    assert cem.problems(tmp_path) == []


def test_negative_controls(tmp_path: Path) -> None:
    rows, body = _all_covered()
    rows["E-02"] = ("", "covered")  # claimed but the test is not listed
    rows["E-03"] = ("`tests/test_edge.py::test_e03_missing`", "covered")  # does not exist
    del rows["E-04"]  # no row
    body = body.replace("def test_e05_case", "def test_other_case")  # E-05 untested
    _setup(tmp_path, rows, body)
    found = "\n".join(cem.problems(tmp_path))
    assert "E-02: test tests/test_edge.py::test_e02_case is not listed" in found
    assert "E-03: listed test tests/test_edge.py::test_e03_missing does not exist" in found
    assert "E-04: no row" in found
    assert "E-05: untested" in found and "E-05: status 'covered' does not match" in found


def test_repository_matrix_lists_every_id() -> None:
    root = Path(__file__).resolve().parents[2]
    rows = cem.parse_matrix((root / cem.MATRIX).read_text("utf-8"))
    assert set(rows) == set(cem.IDS)
    assert not [p for p in cem.problems(root) if "untested" not in p and "status" not in p]
