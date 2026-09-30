from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from manga_ar.errors import ArchiveError
from manga_ar.io.archive import ArchiveLimits, read_archive, safe_member_name, write_cbz
from tests.fixtures.synth import gradient_rgb, png_bytes

PNG = png_bytes(gradient_rgb())


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


@pytest.mark.parametrize(
    "name", ["../evil.png", "a/../../evil.png", "/abs/evil.png", "C:\\evil.png", "..\\evil.png"]
)
def test_zip_slip_rejected(tmp_path: Path, name: str) -> None:
    with pytest.raises(ArchiveError):
        safe_member_name(name)
    zpath = tmp_path / "evil.cbz"
    with zipfile.ZipFile(zpath, "w") as zf:
        info = zipfile.ZipInfo("ok.png")
        zf.writestr(info, PNG)
        zf.writestr(zipfile.ZipInfo(name), PNG)
    with pytest.raises(ArchiveError):
        read_archive(zpath)


def test_natural_order_nested_skips_and_duplicates(tmp_path: Path) -> None:
    zpath = _zip(
        tmp_path / "ch.cbz",
        {
            "vol/page10.png": PNG,
            "vol/page2.png": PNG,
            "vol/Page1.PNG": PNG,
            "notes.txt": b"hi",
            "vol/.hidden.png": PNG,
            "__MACOSX/": b"",
            "vol/page2.PNG": PNG,
        },
    )
    contents = read_archive(zpath)
    assert [m.name for m in contents.members] == [
        "vol/Page1.PNG",
        "vol/page2.png",
        "vol/page10.png",
    ]
    assert "notes.txt" in contents.skipped
    assert any("duplicate" in w for w in contents.warnings)


def test_limits(tmp_path: Path) -> None:
    zpath = _zip(tmp_path / "many.cbz", {f"p{i}.png": PNG for i in range(6)})
    with pytest.raises(ArchiveError, match="members exceed"):
        read_archive(zpath, ArchiveLimits(max_members=5))
    with pytest.raises(ArchiveError, match="size limit"):
        read_archive(zpath, ArchiveLimits(max_member_bytes=100))
    with pytest.raises(ArchiveError, match="uncompressed size"):
        read_archive(zpath, ArchiveLimits(max_total_bytes=len(PNG) * 3))
    bad = tmp_path / "bad.cbz"
    bad.write_bytes(b"not a zip")
    with pytest.raises(ArchiveError, match="not a readable"):
        read_archive(bad)


def test_write_cbz_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "o" / "x_ar.cbz"
    write_cbz(out, [("b/p2_ar.png", PNG), ("b/p10_ar.png", PNG)])
    names = [m.name for m in read_archive(out).members]
    assert names == ["b/p2_ar.png", "b/p10_ar.png"]
    with zipfile.ZipFile(io.BytesIO(out.read_bytes())) as zf:
        assert zf.testzip() is None
