"""Tests for the raw-cell option on native-sharded per-chunk arrays.

``create_vertices_array(..., compress=False)`` and
``create_attribute_array(..., compress=False)``, used inside
:meth:`Group.native_sharded_arrays`, must write ``sharding_indexed``
cells with no inner compressor — so a reader can byte-range-read rows
within a cell (one fragment's vertices) instead of fetching and
decompressing the whole cell. The default (``compress=True``) must be
unaffected: cells still get Zarr's default codec (``zstd``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from zarr.storage import LocalStore

from zarr_vectors.core.arrays import (
    create_attribute_array,
    create_vertices_array,
    read_chunk_vertices,
    write_chunk_attributes,
    write_chunk_vertices,
)
from zarr_vectors.core.store import FsGroup


def _make_sharded_level_group(
    tmp_path: Path, *, grid_shape=(4, 4, 4), shard_shape=(2, 2, 2)
) -> FsGroup:
    root = FsGroup(tmp_path / "store.zarr", create=True)
    lg = root.create_group("0")
    lg._native_sharded_config = {
        "grid_shape": tuple(grid_shape),
        "shard_shape": tuple(shard_shape),
    }
    return lg


def _inner_codec_names(zarr_json: dict) -> list[str]:
    sharding = zarr_json["codecs"][0]
    assert sharding["name"] == "sharding_indexed"
    return [c["name"] for c in sharding["configuration"]["codecs"]]


class TestRawCellVertices:
    def test_round_trip(self, tmp_path: Path) -> None:
        lg = _make_sharded_level_group(tmp_path)
        create_vertices_array(lg, dtype="float32", compress=False)

        g0 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        write_chunk_vertices(lg, (0, 0, 0), [g0])
        g1 = np.array([[10, 20, 30]], dtype=np.float32)
        write_chunk_vertices(lg, (1, 0, 0), [g1])

        out0 = read_chunk_vertices(lg, (0, 0, 0), dtype=np.float32, ndim=3)
        out1 = read_chunk_vertices(lg, (1, 0, 0), dtype=np.float32, ndim=3)
        np.testing.assert_array_equal(out0[0], g0)
        np.testing.assert_array_equal(out1[0], g1)

    def test_no_inner_compressor_on_disk(self, tmp_path: Path) -> None:
        lg = _make_sharded_level_group(tmp_path)
        create_vertices_array(lg, dtype="float32", compress=False)
        write_chunk_vertices(
            lg, (0, 0, 0), [np.zeros((2, 3), dtype=np.float32)]
        )

        store = LocalStore(str(tmp_path / "store.zarr"))
        arr = zarr.open_array(store=store, path="0/vertices", mode="r")
        assert _inner_codec_names(arr.metadata.to_dict()) == ["vlen-bytes"]

    def test_cell_codec_attribute_stamped(self, tmp_path: Path) -> None:
        lg = _make_sharded_level_group(tmp_path)
        create_vertices_array(lg, dtype="float32", compress=False)

        store = LocalStore(str(tmp_path / "store.zarr"))
        arr = zarr.open_array(store=store, path="0/vertices", mode="r")
        assert dict(arr.attrs).get("cell_codec") == "raw"

    def test_cell_bytes_are_uncompressed(self, tmp_path: Path) -> None:
        """The shard's index locates a cell whose payload is the raw
        vlen-bytes single-item framing (``count``, ``length``, bytes) —
        no zstd frame — so a reader can slice fragment rows directly.
        """
        lg = _make_sharded_level_group(tmp_path)
        create_vertices_array(lg, dtype="float32", compress=False)
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        write_chunk_vertices(lg, (0, 0, 0), [pts])

        shard_path = tmp_path / "store.zarr" / "0" / "vertices" / "c" / "0" / "0" / "0"
        data = shard_path.read_bytes()

        import struct

        num_cells = 8  # prod(shard_shape) = 2*2*2
        index_len = num_cells * 16 + 4
        index = struct.unpack("<" + "QQ" * num_cells, data[-index_len:-4])
        offset, length = index[0], index[1]
        assert offset != 0xFFFFFFFFFFFFFFFF, "cell (0,0,0) should be populated"

        cell = data[offset : offset + length]
        count, payload_len = struct.unpack("<II", cell[:8])
        assert count == 1
        payload = cell[8 : 8 + payload_len]
        np.testing.assert_array_equal(
            np.frombuffer(payload, dtype="<f4"), pts.reshape(-1)
        )

    def test_default_compress_true_keeps_zstd(self, tmp_path: Path) -> None:
        lg = _make_sharded_level_group(tmp_path)
        create_vertices_array(lg, dtype="float32")  # compress defaults True

        store = LocalStore(str(tmp_path / "store.zarr"))
        arr = zarr.open_array(store=store, path="0/vertices", mode="r")
        assert _inner_codec_names(arr.metadata.to_dict()) == [
            "vlen-bytes",
            "zstd",
        ]
        assert dict(arr.attrs).get("cell_codec") != "raw"

    def test_vertex_fragments_unaffected_by_vertices_compress_flag(
        self, tmp_path: Path
    ) -> None:
        """``compress=False`` targets the vertices array only —
        ``vertex_fragments`` (read whole, benefits from compression)
        keeps the default codec regardless.
        """
        lg = _make_sharded_level_group(tmp_path)
        create_vertices_array(lg, dtype="float32", compress=False)

        store = LocalStore(str(tmp_path / "store.zarr"))
        frags = zarr.open_array(
            store=store, path="0/vertex_fragments", mode="r"
        )
        assert _inner_codec_names(frags.metadata.to_dict()) == [
            "vlen-bytes",
            "zstd",
        ]


class TestRawCellAttributes:
    def test_round_trip(self, tmp_path: Path) -> None:
        lg = _make_sharded_level_group(tmp_path)
        create_attribute_array(lg, "radius", dtype="float32", compress=False)

        vals = np.array([1.5, 2.5], dtype=np.float32)
        write_chunk_attributes(lg, "radius", (0, 0, 0), [vals], dtype=np.float32)

        # write_chunk_attributes has no dedicated per-chunk reader here;
        # confirm the underlying cell round-trips via read_bytes directly.
        raw = lg.read_bytes("vertex_attributes/radius", "0.0.0")
        np.testing.assert_array_equal(np.frombuffer(raw, dtype="<f4"), vals)

    def test_no_inner_compressor_on_disk(self, tmp_path: Path) -> None:
        lg = _make_sharded_level_group(tmp_path)
        create_attribute_array(lg, "radius", dtype="float32", compress=False)

        store = LocalStore(str(tmp_path / "store.zarr"))
        arr = zarr.open_array(
            store=store, path="0/vertex_attributes/radius", mode="r"
        )
        assert _inner_codec_names(arr.metadata.to_dict()) == ["vlen-bytes"]
        assert dict(arr.attrs).get("cell_codec") == "raw"

    def test_default_compress_true_keeps_zstd(self, tmp_path: Path) -> None:
        lg = _make_sharded_level_group(tmp_path)
        create_attribute_array(lg, "radius", dtype="float32")

        store = LocalStore(str(tmp_path / "store.zarr"))
        arr = zarr.open_array(
            store=store, path="0/vertex_attributes/radius", mode="r"
        )
        assert _inner_codec_names(arr.metadata.to_dict()) == [
            "vlen-bytes",
            "zstd",
        ]
