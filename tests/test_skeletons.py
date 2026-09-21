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


class TestReadBySegmentIdCrossChunk:
    """``read_skeleton_by_segment_id`` must also read the OTHER population
    in the ``links/0/`` family: boundary-crossing parent->child edges
    written by ``write_skeleton_cross_chunk_links``.

    Before this fix, edge assembly only ever consulted
    ``read_chunk_link_fragment`` (intra-chunk branch links, keyed by the
    ``link_fragments/`` sidecar, which -- per its own comment -- only
    ever partitions the all-zero-offsets array).  Crossings were written
    correctly and never read back: silent data loss, one edge per chunk
    boundary an object crosses.
    """

    def test_crossing_edge_is_read_back(self, tmp_path: Path) -> None:
        root, lg = _init(tmp_path)
        piece_a = {
            "segment_id": 77,
            "positions": np.array(
                [[10, 10, 10], [20, 10, 10]], dtype=np.float32,
            ),
            "edges": np.array([[1, 0]], dtype=np.int64),
        }
        piece_b = {
            "segment_id": 77,
            "positions": np.array(
                [[110, 10, 10], [120, 10, 10]], dtype=np.float32,
            ),
            "edges": np.array([[1, 0]], dtype=np.int64),
        }
        write_skeleton_chunk(lg, (0, 0, 0), [piece_a])
        write_skeleton_chunk(lg, (1, 0, 0), [piece_b])
        # Parent = A's last vertex (chunk-local 1); child = B's first
        # vertex (chunk-local 0) -- endpoint 0 is the parent, per
        # write_skeleton_cross_chunk_links's own docstring.
        write_skeleton_cross_chunk_links(
            lg, [(((0, 0, 0), 1), ((1, 0, 0), 0))], ndim=3,
        )
        write_object_index(
            lg, {0: [((0, 0, 0), 0), ((1, 0, 0), 0)]},
            sid_ndim=3, total_objects=1,
        )
        write_object_attributes(
            lg, "segment_id", np.array([77], dtype=np.uint64),
        )

        skel = read_skeleton_by_segment_id(str(tmp_path / "skel.zv"), 77)
        assert skel is not None
        edges = {tuple(e) for e in skel["edges"]}
        # Global 0,1 = fragment A; global 2,3 = fragment B.  The
        # crossing must come back as [child, parent] = (2, 1) --
        # storage leads with the parent, so a reader that forgets to
        # swap would instead report (1, 2).
        assert edges == {(1, 0), (3, 2), (2, 1)}
        assert (1, 2) not in edges, "cross-chunk edge direction reversed"
        # A connected 4-node tree has exactly 3 edges; a stray
        # duplicate would inflate the list without changing the set.
        assert len(skel["edges"]) == 3

    def test_crossing_read_exactly_once_with_two_fragments_in_source_chunk(
        self, tmp_path: Path,
    ) -> None:
        """The chunk holding the crossing's source cell has TWO of this
        object's fragments in it.  A reader that iterates manifest
        entries rather than distinct chunks would fetch that cell's
        cross-chunk record once per fragment and emit the same edge
        twice; this is exactly the shape of the object built for
        ``viz_data/out_fixed.zv``.
        """
        root, lg = _init(tmp_path)
        piece_a1 = {
            "segment_id": 77,
            "positions": np.array(
                [[10, 10, 10], [20, 10, 10]], dtype=np.float32,
            ),
            "edges": np.array([[1, 0]], dtype=np.int64),
        }
        piece_a2 = {
            "segment_id": 77,
            "positions": np.array(
                [[30, 10, 10], [40, 10, 10]], dtype=np.float32,
            ),
            "edges": np.array([[1, 0]], dtype=np.int64),
        }
        piece_b = {
            "segment_id": 77,
            "positions": np.array([[110, 10, 10]], dtype=np.float32),
            "edges": np.zeros((0, 2), dtype=np.int64),
        }
        write_skeleton_chunk(lg, (0, 0, 0), [piece_a1, piece_a2])
        write_skeleton_chunk(lg, (1, 0, 0), [piece_b])
        # Parent = A2's second vertex (chunk-local 3: A1 occupies 0-1,
        # A2 occupies 2-3); child = B's only vertex.
        write_skeleton_cross_chunk_links(
            lg, [(((0, 0, 0), 3), ((1, 0, 0), 0))], ndim=3,
        )
        write_object_index(
            lg,
            {0: [((0, 0, 0), 0), ((0, 0, 0), 1), ((1, 0, 0), 0)]},
            sid_ndim=3, total_objects=1,
        )
        write_object_attributes(
            lg, "segment_id", np.array([77], dtype=np.uint64),
        )

        skel = read_skeleton_by_segment_id(str(tmp_path / "skel.zv"), 77)
        assert skel is not None
        edges = [tuple(e) for e in skel["edges"]]
        # Global: A1 -> 0,1; A2 -> 2,3; B -> 4.  Crossing: child=4,
        # parent=3.
        assert edges.count((4, 3)) == 1, (
            "crossing read once per manifest entry instead of once per "
            "distinct chunk"
        )
        assert set(edges) == {(1, 0), (3, 2), (4, 3)}

    def test_negative_offset_segment(self, tmp_path: Path) -> None:
        """The source cell of a crossing is the PARENT's chunk, and the
        parent's chunk sorting after the child's is equally valid (see
        ``TestDirectedCrossChunk``) -- a fix that only derives positive
        offsets correctly would miss half of all crossings.
        """
        root, lg = _init(tmp_path)
        piece_c = {
            "segment_id": 77,
            "positions": np.array(
                [[110, 10, 10], [120, 10, 10]], dtype=np.float32,
            ),
            "edges": np.array([[1, 0]], dtype=np.int64),
        }
        piece_d = {
            "segment_id": 77,
            "positions": np.array(
                [[10, 10, 10], [20, 10, 10]], dtype=np.float32,
            ),
            "edges": np.array([[1, 0]], dtype=np.int64),
        }
        write_skeleton_chunk(lg, (1, 0, 0), [piece_c])
        write_skeleton_chunk(lg, (0, 0, 0), [piece_d])
        # Parent in chunk (1,0,0); child in chunk (0,0,0) -- the source
        # cell is (1,0,0) and the offset to the child is "-1.0.0".
        write_skeleton_cross_chunk_links(
            lg, [(((1, 0, 0), 0), ((0, 0, 0), 1))], ndim=3,
        )
        segments = lg[links_group_path(0)].children()
        assert "-1.0.0" in segments, "fixture did not land the intended offset"

        write_object_index(
            lg, {0: [((1, 0, 0), 0), ((0, 0, 0), 0)]},
            sid_ndim=3, total_objects=1,
        )
        write_object_attributes(
            lg, "segment_id", np.array([77], dtype=np.uint64),
        )

        skel = read_skeleton_by_segment_id(str(tmp_path / "skel.zv"), 77)
        assert skel is not None
        edges = {tuple(e) for e in skel["edges"]}
        # Global 0,1 = C (the parent fragment); global 2,3 = D.  Parent
        # local 0 -> global 0; child local 1 -> global 3.
        assert (3, 0) in edges
        assert (0, 3) not in edges, "negative-offset edge direction reversed"

    def test_two_objects_sharing_the_crossing_cell(self, tmp_path: Path) -> None:
        """Both objects' crossing records are filed in the SAME cell (same
        source chunk, same offsets segment).  The per-object filter
        (``_to_global`` returning ``None`` for a foreign index) must keep
        each object's read from picking up the other's edge.
        """
        root, lg = _init(tmp_path)
        piece_a0 = {
            "segment_id": 77,
            "positions": np.array([[10, 10, 10]], dtype=np.float32),
            "edges": np.zeros((0, 2), dtype=np.int64),
        }
        piece_b0 = {
            "segment_id": 88,
            "positions": np.array([[20, 10, 10]], dtype=np.float32),
            "edges": np.zeros((0, 2), dtype=np.int64),
        }
        piece_a1 = {
            "segment_id": 77,
            "positions": np.array([[110, 10, 10]], dtype=np.float32),
            "edges": np.zeros((0, 2), dtype=np.int64),
        }
        piece_b1 = {
            "segment_id": 88,
            "positions": np.array([[120, 10, 10]], dtype=np.float32),
            "edges": np.zeros((0, 2), dtype=np.int64),
        }
        write_skeleton_chunk(lg, (0, 0, 0), [piece_a0, piece_b0])
        write_skeleton_chunk(lg, (1, 0, 0), [piece_a1, piece_b1])
        write_skeleton_cross_chunk_links(
            lg,
            [
                (((0, 0, 0), 0), ((1, 0, 0), 0)),  # object 77
                (((0, 0, 0), 1), ((1, 0, 0), 1)),  # object 88
            ],
            ndim=3,
        )
        write_object_index(
            lg,
            {
                0: [((0, 0, 0), 0), ((1, 0, 0), 0)],
                1: [((0, 0, 0), 1), ((1, 0, 0), 1)],
            },
            sid_ndim=3, total_objects=2,
        )
        write_object_attributes(
            lg, "segment_id", np.array([77, 88], dtype=np.uint64),
        )

        skel_a = read_skeleton_by_segment_id(str(tmp_path / "skel.zv"), 77)
        skel_b = read_skeleton_by_segment_id(str(tmp_path / "skel.zv"), 88)
        assert skel_a is not None and skel_b is not None
        assert {tuple(e) for e in skel_a["edges"]} == {(1, 0)}
        assert {tuple(e) for e in skel_b["edges"]} == {(1, 0)}
        assert len(skel_a["edges"]) == 1, "object 77 picked up a foreign edge"
        assert len(skel_b["edges"]) == 1, "object 88 picked up a foreign edge"
