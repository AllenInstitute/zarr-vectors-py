"""Tests for the single-level skeleton geometry type.

Covers the pure path-decomposition helper, the per-chunk writer, the
directed cross-chunk parent->child edges, and a pull-by-id read round-trip
(with a hand-built object_index standing in for the external reduce).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.core.arrays import (
    read_links,
    write_object_attributes,
    write_object_index,
)
from zarr_vectors.core.paths import links_group_path, links_path
from zarr_vectors.types.skeletons import (
    decompose_tree_to_paths,
    init_skeleton_store,
    read_skeleton_by_segment_id,
    write_skeleton_chunk,
    write_skeleton_cross_chunk_links,
)


def read_cross_links(level_group, *, delta: int = 0) -> list:
    """Records spanning more than one chunk.

    ``read_links`` returns the whole family — intra-chunk links are the
    all-zero-offsets array, not a separate family — so tests that care
    about the pre-merge ``cross_chunk_links`` set filter for it.
    """
    return [
        record for record in read_links(level_group, delta=delta)
        if len({tuple(cc) for cc, _vi in record}) > 1
    ]


def _init(tmp_path: Path):
    return init_skeleton_store(
        str(tmp_path / "skel.zv"),
        chunk_shape=(100.0, 100.0, 100.0),
        bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
        ndim=3,
        attribute_dtypes={"radius": "float32"},
    )


class TestDecompose:
    def test_linear_chain(self) -> None:
        piece = {
            "positions": np.array(
                [[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32,
            ),
            "edges": np.array([[1, 0], [2, 1]], dtype=np.int64),
        }
        opos, _oattrs, frags, blinks, _new = decompose_tree_to_paths(piece)
        assert len(frags) == 1
        assert frags[0] == (0, 3)
        assert blinks == []

    def test_branching_tree(self) -> None:
        # 0->1->2 with a branch 1->3.
        piece = {
            "positions": np.arange(12, dtype=np.float32).reshape(4, 3),
            "edges": np.array([[1, 0], [2, 1], [3, 1]], dtype=np.int64),
        }
        _opos, _oattrs, frags, blinks, _new = decompose_tree_to_paths(piece)
        # One main chain (0,1,2) + one branch fragment (3) linked to vertex 1.
        assert (0, 3) in frags
        assert any(count == 1 for _s, count in frags)
        assert len(blinks) == 1
        child_o, parent_o = blinks[0]
        assert parent_o == 1  # branch attaches at order-index 1


class TestPerChunkWrite:
    def test_vertices_and_links_round_trip(self, tmp_path: Path) -> None:
        from zarr_vectors.core.arrays import (
            read_chunk_links, read_chunk_vertices,
        )

        root, lg = _init(tmp_path)
        piece = {
            "segment_id": 42,
            "positions": np.array(
                [[10, 10, 10], [20, 10, 10], [30, 10, 10]], dtype=np.float32,
            ),
            "edges": np.array([[1, 0], [2, 1]], dtype=np.int64),
            "attributes": {"radius": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
        }
        records, _anchors = write_skeleton_chunk(lg, (0, 0, 0), [piece])
        assert records == [(42, (0, 0, 0), 0)]

        verts = read_chunk_vertices(lg, (0, 0, 0), ndim=3)
        assert np.allclose(np.concatenate(verts), piece["positions"])
        # One linear fragment → no explicit branch link rows.
        links = read_chunk_links(lg, (0, 0, 0), delta=0)
        assert all(len(g) == 0 for g in links)


class TestDirectedCrossChunk:
    def test_parent_child_order_preserved(self, tmp_path: Path) -> None:
        root, lg = _init(tmp_path)
        # Parent in chunk (1,0,0), child in chunk (0,0,0): canonical sort
        # would swap them; directed storage must not.
        links = [(((1, 0, 0), 0), ((0, 0, 0), 0))]
        write_skeleton_cross_chunk_links(lg, links, ndim=3)

        # directed/store is family-wide, so it lives on the <delta> group.
        meta = lg.read_array_meta(links_group_path(0))
        assert meta["directed"] is True
        out = read_cross_links(lg, delta=0)
        assert out == [(((1, 0, 0), 0), ((0, 0, 0), 0))]

        # The parent is the source cell and the child sits one chunk back
        # along -x.  A canonical sort would instead have led with the
        # child, filing the record at cell 0.0.0 under offsets "+1.0.0" —
        # the negative offset is the direction surviving on disk.
        # (``0.0.0``, the intra array, is pre-created empty by init.)
        segments = lg[links_group_path(0)].children()
        assert "-1.0.0" in segments
        assert "+1.0.0" not in segments
        assert lg.chunk_exists(links_path(0, ((-1, 0, 0),)), "1.0.0")


class TestReadBySegmentId:
    def test_round_trip(self, tmp_path: Path) -> None:
        root, lg = _init(tmp_path)
        piece = {
            "segment_id": 77,
            "positions": np.array(
                [[10, 10, 10], [20, 10, 10], [30, 10, 10], [40, 10, 10]],
                dtype=np.float32,
            ),
            "edges": np.array([[1, 0], [2, 1], [3, 2]], dtype=np.int64),
            "attributes": {"radius": np.array(
                [1.0, 2.0, 3.0, 4.0], dtype=np.float32,
            )},
        }
        write_skeleton_chunk(lg, (0, 0, 0), [piece])

        # Stand in for the external object-index reduce: one object (oid 0)
        # whose single fragment is (chunk, fragment_idx)=(0,0,0),0), and a
        # sorted segment_id attribute mapping oid 0 -> segment 77.
        write_object_index(lg, {0: [((0, 0, 0), 0)]}, sid_ndim=3, total_objects=1)
        write_object_attributes(
            lg, "segment_id", np.array([77], dtype=np.uint64),
        )

        store_path = str(tmp_path / "skel.zv")
        skel = read_skeleton_by_segment_id(store_path, 77)
        assert skel is not None
        assert skel["fragment_count"] == 1
        assert np.allclose(skel["positions"], piece["positions"])
        # Implicit within-path edges reconstructed as [child, parent].
        assert {tuple(e) for e in skel["edges"]} == {(1, 0), (2, 1), (3, 2)}
        assert np.allclose(skel["attributes"]["radius"], [1.0, 2.0, 3.0, 4.0])

    def test_missing_segment_returns_none(self, tmp_path: Path) -> None:
        root, lg = _init(tmp_path)
        write_object_index(lg, {}, sid_ndim=3, total_objects=1)
        write_object_attributes(
            lg, "segment_id", np.array([5], dtype=np.uint64),
        )
        skel = read_skeleton_by_segment_id(str(tmp_path / "skel.zv"), 999)
        assert skel is None
