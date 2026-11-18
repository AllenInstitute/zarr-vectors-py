"""HeaderRegistry dict-only contract tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tests._deprecated import open_zv
from zarr_vectors.headers import HeaderRegistry
from zarr_vectors.types.points import write_points


def _make_store(tmp_path: Path) -> str:
    store = str(tmp_path / "pts.zv")
    rng = np.random.default_rng(0)
    write_points(
        store,
        rng.uniform(0, 100, size=(10, 3)).astype(np.float32),
        chunk_shape=(100., 100., 100.),
    )
    return store


class TestHeaderRegistry:

    def test_add_get_roundtrip(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        reg = HeaderRegistry(store)

        header = {
            "format_name": "trk",
            "voxel_size": [1.0, 1.0, 1.0],
            "dimensions": [256, 256, 256],
        }
        reg.add("trk", header)

        out = reg.get("trk")
        assert out == header

    def test_available_and_has(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        reg = HeaderRegistry(store)
        assert reg.available_formats == []
        assert not reg.has("trk")

        reg.add("trk", {"format_name": "trk"})
        reg.add("swc", {"format_name": "swc", "comment_lines": ["# test"]})

        assert reg.has("trk") and reg.has("swc")
        assert reg.available_formats == ["swc", "trk"]

    def test_overwrite(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        reg = HeaderRegistry(store)
        reg.add("trk", {"format_name": "trk", "version": 1})
        reg.add("trk", {"format_name": "trk", "version": 2})
        assert reg.get("trk")["version"] == 2

    def test_remove(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        reg = HeaderRegistry(store)
        reg.add("trk", {"format_name": "trk"})
        assert reg.has("trk")
        reg.remove("trk")
        assert not reg.has("trk")

    def test_get_missing_raises(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        reg = HeaderRegistry(store)
        try:
            reg.get("trk")
            assert False, "expected KeyError"
        except KeyError:
            pass


class TestLazyStoreHeaders:

    def test_headers_property_returns_dicts(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        reg = HeaderRegistry(store)
        reg.add("trk", {"format_name": "trk", "voxel_size": [1.0, 1.0, 1.0]})

        zv = open_zv(store)
        headers = zv.headers
        assert "trk" in headers
        assert isinstance(headers["trk"], dict)
        assert headers["trk"]["voxel_size"] == [1.0, 1.0, 1.0]
