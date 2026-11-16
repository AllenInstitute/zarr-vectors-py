"""Range reads work where os has no pread (Windows)."""

import os

from zarr_vectors.core._cells_on_disk import _read_ranges


def test_read_ranges_without_pread(tmp_path, monkeypatch):
    path = tmp_path / "cell"
    data = bytes(range(256)) * 4
    path.write_bytes(data)
    monkeypatch.delattr(os, "pread", raising=False)
    got = _read_ranges([(str(path), 5, 7), (str(path), 0, -1), (str(path), 1020, 8)])
    assert got[0] == data[5:12]
    assert got[1] == data
    assert isinstance(got[2], ValueError)  # short read
