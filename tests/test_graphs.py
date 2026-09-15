"""Step 09 tests: graph/skeleton write/read core API."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.types.graphs import write_graph, read_graph
from zarr_vectors.types.skeletons import (
    init_skeleton_store,
    write_skeleton_chunk,
    write_skeleton_cross_chunk_links,
)
from zarr_vectors.exceptions import ArrayError


class TestGeneralGraph:

    def test_single_chunk(self, tmp_path: Path) -> None:
        pos = np.array([[10,10,10],[20,20,20],[30,30,30],[40,40,40]], dtype=np.float32)
        edges = np.array([[0,1],[1,2],[2,3],[0,2]], dtype=np.int64)
        s = write_graph(str(tmp_path / "g.zv"), pos, edges, chunk_shape=(100.,100.,100.))
        assert s["node_count"] == 4 and s["edge_count"] == 4
        r = read_graph(str(tmp_path / "g.zv"))
        assert r["node_count"] == 4 and r["edge_count"] > 0

    def test_cross_chunk(self, tmp_path: Path) -> None:
        pos = np.array([[10,50,50],[20,50,50],[110,50,50],[120,50,50]], dtype=np.float32)
        edges = np.array([[0,1],[2,3],[0,2],[1,3]], dtype=np.int64)
        s = write_graph(str(tmp_path / "g.zv"), pos, edges, chunk_shape=(100.,100.,100.))
        assert s["cross_edge_count"] >= 2

    def test_empty_edges(self, tmp_path: Path) -> None:
        pos = np.array([[1,2,3],[4,5,6]], dtype=np.float32)
        s = write_graph(str(tmp_path / "g.zv"), pos, np.zeros((0,2), dtype=np.int64),
                        chunk_shape=(100.,100.,100.))
        assert s["edge_count"] == 0

    def test_bad_edge_shape(self, tmp_path: Path) -> None:
        pos = np.array([[1,2,3],[4,5,6]], dtype=np.float32)
        try:
            write_graph(str(tmp_path / "g.zv"), pos, np.zeros((3,3), dtype=np.int64),
                        chunk_shape=(100.,100.,100.))
            assert False
        except ArrayError:
            pass


class TestSkeleton:

    def test_7node_tree(self, tmp_path: Path) -> None:
        pos = np.array([[50,50,50],[40,40,40],[60,60,60],[30,30,30],
                         [45,35,35],[65,65,65],[25,25,25]], dtype=np.float32)
        edges = np.array([[1,0],[2,0],[3,1],[4,1],[5,2],[6,3]], dtype=np.int64)
        s = write_graph(str(tmp_path / "s.zv"), pos, edges, chunk_shape=(200.,200.,200.), kind="skeleton")
        r = read_graph(str(tmp_path / "s.zv"))
        assert r["node_count"] == 7 and r["edge_count"] == 6

    def test_with_attributes(self, tmp_path: Path) -> None:
        pos = np.array([[50,50,50],[40,40,40],[60,60,60]], dtype=np.float32)
        edges = np.array([[1,0],[2,0]], dtype=np.int64)
        radius = np.array([5.0,3.0,3.0], dtype=np.float32)
        write_graph(str(tmp_path / "s.zv"), pos, edges, chunk_shape=(200.,200.,200.),
                    kind="skeleton", vertex_attributes={"radius": radius})

    def test_cross_chunk(self, tmp_path: Path) -> None:
        pos = np.array([[10,50,50],[20,50,50],[30,50,50],[110,50,50],[120,50,50]], dtype=np.float32)
        edges = np.array([[1,0],[2,1],[3,2],[4,3]], dtype=np.int64)
        s = write_graph(str(tmp_path / "s.zv"), pos, edges, chunk_shape=(100.,100.,100.), kind="skeleton")
        assert s["cross_edge_count"] >= 1
        r = read_graph(str(tmp_path / "s.zv"))
        assert r["node_count"] == 5

    def test_multiple_objects(self, tmp_path: Path) -> None:
        pos = np.array([[10,10,10],[20,20,20],[30,30,30],[50,50,50],[60,60,60]], dtype=np.float32)
        edges = np.array([[1,0],[2,1],[4,3]], dtype=np.int64)
        oids = np.array([0,0,0,1,1], dtype=np.int64)
        s = write_graph(str(tmp_path / "g.zv"), pos, edges, chunk_shape=(100.,100.,100.),
                        kind="skeleton", object_ids=oids)
        assert s["object_count"] == 2


def _edge_endpoints(positions, edges):
    """Edges as sets of coordinate pairs, so a reordered read still compares.

    A reader is free to return vertices in storage order rather than input
    order, so index equality says nothing; what has to survive the round
    trip is which *points* are joined.
    """
    return {
        frozenset((tuple(positions[a]), tuple(positions[b])))
        for a, b in edges
    }


class TestSkeletonReadOrder:
    """The implicit-sequential convention is stated in READ order.

    ``parent[i] = i - 1`` holds over the rows a reader concatenates, not
    over the depth-first numbering the writer assigned.  The two diverge
    as soon as the chunk holding the root sorts after another chunk, and
    the old writer decided what to store in DFS space -- so a real edge
    went unwritten and the root silently inherited the previous chunk's
    last vertex as its parent.
    """

    def test_root_in_a_later_chunk(self, tmp_path: Path) -> None:
        # Root at high coordinates, so its chunk sorts last while its DFS
        # index is 0.
        pos = np.array([[350,350,350],[50,50,50],[60,60,60],[70,70,70]],
                       dtype=np.float32)
        edges = np.array([[1,0],[2,1],[3,2]], dtype=np.int64)
        store = str(tmp_path / "root_last.zv")
        write_graph(store, pos, edges, chunk_shape=(200.,200.,200.),
                    kind="skeleton")

        r = read_graph(store)
        assert r["edge_count"] == 3, "read back an edge that was never written"
        assert _edge_endpoints(r["positions"], r["edges"]) ==             _edge_endpoints(pos, edges)

    def test_multi_object_forest(self, tmp_path: Path) -> None:
        """Two trees, one object id each -- the supported spelling.

        Each object is its own fragment, so each root starts a fragment
        and has no predecessor to inherit.
        """
        pos = np.array([[10,10,10],[20,20,20],[100,100,100],[110,110,110]],
                       dtype=np.float32)
        edges = np.array([[1,0],[3,2]], dtype=np.int64)
        store = str(tmp_path / "forest.zv")
        write_graph(store, pos, edges, chunk_shape=(500.,500.,500.),
                    kind="skeleton", object_ids=np.array([0,0,1,1]))

        r = read_graph(store)
        assert r["edge_count"] == 2, "the second tree's root grew a parent"
        assert _edge_endpoints(r["positions"], r["edges"]) ==             _edge_endpoints(pos, edges)

    def test_multi_object_roots_in_different_chunks(self, tmp_path: Path) -> None:
        pos = np.array([[310,10,10],[10,10,10],[20,20,20],
                        [30,30,30],[330,30,30]], dtype=np.float32)
        edges = np.array([[1,0],[2,1],[4,3]], dtype=np.int64)
        store = str(tmp_path / "mixed.zv")
        write_graph(store, pos, edges, chunk_shape=(200.,200.,200.),
                    kind="skeleton", object_ids=np.array([0,0,0,1,1]))

        r = read_graph(store)
        assert r["edge_count"] == 3
        assert _edge_endpoints(r["positions"], r["edges"]) ==             _edge_endpoints(pos, edges)

    def test_a_forest_in_one_object_is_refused(self, tmp_path: Path) -> None:
        """Inexpressible, so it raises rather than inventing an edge.

        A second root inside one object's fragment has no parent but does
        have a predecessor the reader will hand it.  There is nowhere to
        record "this one is a root", so the write is rejected and the
        message names the two ways out.
        """
        pos = np.array([[10,10,10],[20,20,20],[100,100,100],[110,110,110]],
                       dtype=np.float32)
        edges = np.array([[1,0],[3,2]], dtype=np.int64)
        store = tmp_path / "bad_forest.zv"
        try:
            # No object_ids: every vertex defaults to object 0, so both
            # trees share one fragment.
            write_graph(str(store), pos, edges, chunk_shape=(500.,500.,500.),
                        kind="skeleton")
            assert False, "expected ArrayError"
        except ArrayError as e:
            assert "roots" in str(e)
            assert "object_ids" in str(e) and "kind='graph'" in str(e)

    def test_a_forest_is_writable_as_a_graph(self, tmp_path: Path) -> None:
        """The escape the error names actually works."""
        pos = np.array([[10,10,10],[20,20,20],[100,100,100],[110,110,110]],
                       dtype=np.float32)
        edges = np.array([[1,0],[3,2]], dtype=np.int64)
        store = str(tmp_path / "as_graph.zv")
        write_graph(store, pos, edges, chunk_shape=(500.,500.,500.))

        r = read_graph(store)
        assert r["edge_count"] == 2
        assert _edge_endpoints(r["positions"], r["edges"]) ==             _edge_endpoints(pos, edges)


def _directed_edge_endpoints(positions, edges):
    """Edges as ORDERED (child_pos, parent_pos) pairs.

    ``_edge_endpoints`` above uses ``frozenset``, so it cannot see an
    inverted tree -- which is exactly how the cross-chunk endpoint swap
    covered by ``TestReadGraphOnDirectedSkeletonFamily`` stayed hidden.
    """
    return {
        (tuple(positions[c]), tuple(positions[p])) for c, p in edges
    }


class TestReadGraphOnDirectedSkeletonFamily:
    """``read_graph`` must also read stores built by the OTHER skeleton
    writer -- ``init_skeleton_store`` / ``write_skeleton_chunk`` /
    ``write_skeleton_cross_chunk_links`` (``types/skeletons.py``, the
    writer ``zarr_vectors_tools``' precomputed ingest actually uses) --
    not just its own ``write_graph(kind="skeleton")``.

    Both stamp ``links_convention: "implicit_sequential_with_branches"``,
    so ``read_graph`` treats either as fair game.  But the two writers use
    OPPOSITE endpoint policies (see ``docs/spec/geometry_types/
    skeleton.md``'s "Known inconsistency"): ``write_graph`` uses an
    undirected canonical family (``directed=False``), whose ``perm_idx``
    column restores exact ``[child, parent]`` input order regardless of
    physical chunk placement -- so a single "column 0 is the child" rule
    is already correct for it (see ``test_root_in_a_later_chunk`` above).
    ``write_skeleton_cross_chunk_links`` instead uses a ``directed=True``
    family with no ``perm_idx``: intra records are ``[child, parent]``
    but CROSS records lead with the **parent** -- so that same blanket
    rule silently corrupts every cross-chunk edge it reads from THIS
    writer's stores: the correct intra-derived parent gets overwritten
    with the wrong (reversed) cross-derived one, and the edge it
    overwrote is lost outright, not merely reordered.
    """

    def test_directed_family_cross_chunk_edge_not_reversed(
        self, tmp_path: Path,
    ) -> None:
        root, lg = init_skeleton_store(
            str(tmp_path / "sk.zv"),
            chunk_shape=(100.0, 100.0, 100.0),
            bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
            ndim=3,
            attribute_dtypes={},
        )
        piece_a = {
            "segment_id": 1,
            "positions": np.array(
                [[10, 10, 10], [20, 10, 10]], dtype=np.float32,
            ),
            "edges": np.array([[1, 0]], dtype=np.int64),
        }
        piece_b = {
            "segment_id": 1,
            "positions": np.array(
                [[110, 10, 10], [120, 10, 10]], dtype=np.float32,
            ),
            "edges": np.array([[1, 0]], dtype=np.int64),
        }
        write_skeleton_chunk(lg, (0, 0, 0), [piece_a])
        write_skeleton_chunk(lg, (1, 0, 0), [piece_b])
        # Parent = A's last vertex; child = B's first vertex -- a single
        # chain of 4 nodes spanning the chunk boundary.
        write_skeleton_cross_chunk_links(
            lg, [(((0, 0, 0), 1), ((1, 0, 0), 0))], ndim=3,
        )

        r = read_graph(str(tmp_path / "sk.zv"))
        assert r["node_count"] == 4
        # The whole point: exactly 3 edges recovering the single chain,
        # not 2 (the pre-fix result, which lost the (20,10,10)->(10,10,10)
        # edge outright when the buggy cross-derived assignment
        # overwrote the correct implicit one).
        assert r["edge_count"] == 3
        expected = {
            ((20, 10, 10), (10, 10, 10)),
            ((110, 10, 10), (20, 10, 10)),
            ((120, 10, 10), (110, 10, 10)),
        }
        actual = _directed_edge_endpoints(r["positions"], r["edges"])
        assert actual == expected

    def test_undirected_family_still_correct(self, tmp_path: Path) -> None:
        """The existing writer/reader pair must not regress: a
        ``directed=False`` family's cross-chunk edges need no swap.
        """
        pos = np.array(
            [[350, 350, 350], [50, 50, 50], [60, 60, 60], [70, 70, 70]],
            dtype=np.float32,
        )
        edges = np.array([[1, 0], [2, 1], [3, 2]], dtype=np.int64)
        store = str(tmp_path / "root_last.zv")
        write_graph(
            store, pos, edges, chunk_shape=(200.0, 200.0, 200.0),
            kind="skeleton",
        )
        r = read_graph(store)
        assert r["edge_count"] == 3
        assert _directed_edge_endpoints(r["positions"], r["edges"]) == \
            _directed_edge_endpoints(pos, edges)


class TestMultiObjectLinkIndices:
    """A chunk's vertices are stored object by object, and links index that.

    ``build_vertex_chunk_mapping`` numbered vertices in ascending global
    order while the write loop below it emitted one fragment per object --
    so in any chunk holding more than one object, every edge pointed at
    the wrong vertex.  ``meshes.py`` already regrouped before building the
    mapping; ``graphs.py`` did not.
    """

    def test_interleaved_objects_keep_their_edges(self, tmp_path: Path) -> None:
        pos = np.array([[10,10,10],[20,20,20],[30,30,30],[40,40,40]],
                       dtype=np.float32)
        # Object ids interleaved, so grouping by object reorders the chunk.
        oids = np.array([1,0,1,0], dtype=np.int64)
        edges = np.array([[0,2],[1,3]], dtype=np.int64)
        store = str(tmp_path / "interleaved.zv")
        write_graph(store, pos, edges, chunk_shape=(500.,500.,500.),
                    object_ids=oids)

        r = read_graph(store)
        assert r["edge_count"] == 2
        assert _edge_endpoints(r["positions"], r["edges"]) ==             _edge_endpoints(pos, edges)


class TestBboxFilter:

    def test_bbox(self, tmp_path: Path) -> None:
        pos = np.array([[10,10,10],[50,50,50],[150,150,150]], dtype=np.float32)
        edges = np.array([[0,1],[1,2]], dtype=np.int64)
        write_graph(str(tmp_path / "g.zv"), pos, edges, chunk_shape=(200.,200.,200.))
        r = read_graph(str(tmp_path / "g.zv"), bbox=(np.array([0,0,0]), np.array([100,100,100])))
        assert r["node_count"] == 2
