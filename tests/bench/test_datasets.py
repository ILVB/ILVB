"""0.3.3: manifest, stratified page-level splits, sealed test split, hash verification."""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import pytest

from benchmarks import datasets
from benchmarks.generators.page import CATEGORIES


def test_manifest_counts_and_stratification() -> None:
    man = datasets.manifest()
    total = sum(s["regions"] for s in man["splits"].values())
    assert total >= 400
    for split, per_cat in (("dev", 4), ("val", 4), ("test", 5)):
        cats = Counter(p["category"] for p in man["splits"][split]["pages"])
        assert cats == {c: per_cat for c in CATEGORIES}, split
    ids = [p["page_id"] for s in man["splits"].values() for p in s["pages"]]
    assert len(ids) == len(set(ids))  # no page in two splits
    recorded = Path(datasets.MANIFEST_DIR / "synthetic_v1.test.sha256").read_text().split()[0]
    assert recorded == man["splits"]["test"]["sha256"]


def test_test_split_is_sealed() -> None:
    with pytest.raises(datasets.SealedSplitError):
        datasets.load_split("test", verify=False)
    with pytest.raises(datasets.SealedSplitError):
        datasets.SealedCapability(object(), "sneaky")


@pytest.mark.integration
def test_dev_loads_and_tampering_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = datasets.DATA_DIR / "synthetic_v1" / "dev"
    if not src.is_dir():
        pytest.skip("dataset not built: python -m benchmarks.generators.build")
    items = datasets.load_split("dev")
    assert len(items) == 40 and items[0].gt().regions
    copy = tmp_path / "synthetic_v1" / "dev"
    shutil.copytree(src, copy)
    monkeypatch.setattr(datasets, "DATA_DIR", tmp_path)
    victim = copy / f"{items[0].page_id}.gt.json"
    victim.write_text(victim.read_text(encoding="utf-8").replace('"text": "', '"text": "X', 1),
                      encoding="utf-8")  # fmt: skip
    with pytest.raises(datasets.ManifestError, match="hash"):
        datasets.load_split("dev")
