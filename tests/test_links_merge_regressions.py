"""Regressions for defects found from a downstream consumer's vantage.

Both were invisible to the rest of the suite because core never exercised
the path a downstream caller reaches for first.  Each test here fails
loudly against the pre-fix code.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from zarr_vectors.core.arrays import (
    create_links_family,
    finalize_links,
    link_family_policy,
    list_link_offsets,
    read_chunk_links,
    read_chunk_vertices,
    read_links,
    vertices_dtype,
    write_link_cells,
)
from zarr_vectors.core.paths import is_intra, links_group_path, parse_offsets
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.types.graphs import write_graph
from zarr_vectors.types.lines import write_lines
from zarr_vectors.types.points import write_points


def _graph_spanning_two_chunks(n_edges: int) -> str:
    """Store with ``n_edges`` edges, every one straddling a chunk boundary."""
    pos: list[list[float]] = []
    edges: list[list[int]] = []
    for i in range(n_edges):
        pos += [[float(i), 0.0, 0.0], [20.0 + i, 0.0, 0.0]]
        edges.append([2 * i, 2 * i + 1])
    path = os.path.join(tempfile.mkdtemp(), "g.zv")
    write_graph(
        path, np.array(pos, np.float32), np.array(edges, np.int64),
        chunk_shape=(10.0, 10.0, 10.0),
    )
    return path


def _cross_segments(lg, delta: int = 0):
    """The non-intra offsets segments of a family, with their offsets."""
    policy = link_family_policy(lg, delta)
    assert policy is not None
    link_width, sid_ndim, _directed, _store = policy
    out = []
    for seg in list_link_offsets(lg, delta):
        offsets = parse_offsets(seg, sid_ndim=sid_ndim, link_width=link_width)
        if not is_intra(offsets):
            out.append(offsets)
    return out


class TestReadChunkLinksPhysicalWidth:
    """``read_chunk_links`` must decode at the stored (physical) row width.

    A links_has_perm array stores ``1 + L`` per row.  Decoding at the
    logical width L does not reliably raise: with N records the cell holds
    ``(1 + L) * N`` elements and ``(1 + L) * N == N (mod L)``, so it
    silently yields ``(1 + L) * N / L`` fabricated rows whenever
    ``N % L == 0`` — every even N at L=2.  Loud on odd N, silent on even,
    which is why nothing caught it.
    """

    @pytest.mark.parametrize("n_edges", [1, 2, 3, 4, 5, 6])
    def test_row_count_matches_truth(self, n_edges: int) -> None:
        lg = get_resolution_level(open_store(_graph_spanning_two_chunks(n_edges)), 0)
        truth = len(read_links(lg, delta=0))
        assert truth == n_edges

        for offsets in _cross_segments(lg):
            groups = read_chunk_links(lg, (0, 0, 0), offsets=offsets)
            got = sum(int(np.asarray(g).shape[0]) for g in groups)
            assert got == truth, (
                f"read_chunk_links fabricated rows: {got} != {truth}. "
                f"Decoded at the logical width instead of the stored one."
            )

    def test_returns_the_perm_column(self) -> None:
        # The rows are 1 + L wide and the caller must be able to see it;
        # silently handing back L-wide rows would misread perm_idx as a
        # vertex index.
        lg = get_resolution_level(open_store(_graph_spanning_two_chunks(2)), 0)
        for offsets in _cross_segments(lg):
            groups = read_chunk_links(lg, (0, 0, 0), offsets=offsets)
            rows = np.concatenate([np.asarray(g) for g in groups], axis=0)
            assert rows.shape[1] == 3, rows  # [perm_idx, vi_0, vi_1]

    def test_intra_has_no_perm_column(self) -> None:
        # The intra array is never canonical-sorted, so it carries no perm
        # column and stays byte-identical to the pre-merge links/<delta>/.
        path = os.path.join(tempfile.mkdtemp(), "g.zv")
        pos = np.array(
            [[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]], dtype=np.float32,
        )
        write_graph(
            path, pos, np.array([[0, 1], [1, 2]], np.int64),
            chunk_shape=(100.0, 100.0, 100.0),
        )
        lg = get_resolution_level(open_store(path), 0)
        groups = read_chunk_links(lg, (0, 0, 0))
        rows = np.concatenate([np.asarray(g) for g in groups], axis=0)
        assert rows.shape[1] == 2, rows


class TestDecentralizedManifestProtocol:
    """Workers skip the shared manifest; the coordinator rebuilds it.

    ``nonempty_chunks`` is array-wide state, so stamping it is a
    read-modify-write that two workers writing *disjoint* cells still race
    on — the loser's key vanishes while its payload sits on disk.  Workers
    therefore pass ``record_presence=False`` and ``finalize_links``
    rebuilds the manifest from the store listing.
    """

    def _store(self) -> str:
        path = os.path.join(tempfile.mkdtemp(), "p.zv")
        pos = np.array(
            [[5.0, 5, 5], [15.0, 5, 5], [25.0, 5, 5], [35.0, 5, 5]],
            dtype=np.float32,
        )
        write_points(
            path, pos, chunk_shape=(10.0, 10.0, 10.0),
            bounds=([0.0, 0.0, 0.0], [40.0, 40.0, 40.0]),
        )
        return path

    def test_disjoint_workers_plus_finalize_keeps_every_record(self) -> None:
        lg = get_resolution_level(open_store(self._store(), mode="r+"), 0)
        # Two workers, disjoint source chunks, same offsets array.
        write_link_cells(lg, [[((0, 0, 0), 0), ((1, 0, 0), 0)]], sid_ndim=3)
        write_link_cells(lg, [[((2, 0, 0), 0), ((3, 0, 0), 0)]], sid_ndim=3)
        finalize_links(lg, delta=0)
        assert len(read_links(lg, delta=0)) == 2

    def test_reads_see_nothing_until_finalize(self) -> None:
        # Not a bug — the documented contract.  Pinning it so the worker
        # half can't be "fixed" back into stamping the shared manifest.
        lg = get_resolution_level(open_store(self._store(), mode="r+"), 0)
        write_link_cells(lg, [[((0, 0, 0), 0), ((1, 0, 0), 0)]], sid_ndim=3)
        assert read_links(lg, delta=0) == []
        finalize_links(lg, delta=0)
        assert len(read_links(lg, delta=0)) == 1

    def test_record_presence_false_is_honoured_by_every_per_chunk_writer(
        self,
    ) -> None:
        # record_presence=False must suppress the nonempty_chunks stamp on
        # EVERY array a per-chunk write touches, or a concurrent writer still
        # races on the un-suppressed one (the leak was: write_chunk_links
        # honoured it on the links cell but not on the link_fragments sidecar,
        # and write_chunk_fragment_attributes had no opt-out at all).
        from zarr_vectors.core.arrays import (
            create_fragment_attribute_array,
            create_links_array,
            create_vertices_array,
            write_chunk_fragment_attributes,
            write_chunk_links,
            write_chunk_vertices,
        )
        from zarr_vectors.core.paths import intra_offsets, links_path
        from zarr_vectors.core.store import create_store

        # Empty store: no prior write may pre-stamp any manifest, or the
        # opt-out under test would be masked by an earlier presence entry.
        path = os.path.join(tempfile.mkdtemp(), "rp.zv")
        root = create_store(
            path, bounds=([0.0, 0.0, 0.0], [40.0, 40.0, 40.0]),
            chunk_shape=(10.0, 10.0, 10.0), geometry_types=["graph"], ndim=3,
        )
        lg = get_resolution_level(root, 0)
        create_vertices_array(lg, dtype="float32")
        create_links_array(lg, link_width=2, sid_ndim=3)
        create_fragment_attribute_array(lg, "segment_id", dtype="uint64")

        write_chunk_vertices(
            lg, (0, 0, 0), [np.zeros((3, 3), np.float32)], record_presence=False,
        )
        write_chunk_links(
            lg, (0, 0, 0), [np.array([[0, 1], [1, 2]], np.int64)],
            record_presence=False,
        )
        write_chunk_fragment_attributes(
            lg, "segment_id", (0, 0, 0), np.array([7], np.uint64),
            dtype=np.uint64, record_presence=False,
        )

        intra = links_path(0, intra_offsets(3, 2))
        for name in (
            "vertices", "vertex_fragments", intra, "link_fragments",
            "fragment_attributes/segment_id",
        ):
            assert lg.list_chunks(name) == [], (
                f"{name} stamped its manifest despite record_presence=False"
            )


class TestCreateLinksFamily:
    """Policy must be stampable without materialising an offsets array."""

    def _store(self) -> str:
        path = os.path.join(tempfile.mkdtemp(), "p.zv")
        pos = np.array(
            [[5.0, 5, 5], [15.0, 5, 5], [25.0, 5, 5], [35.0, 5, 5]],
            dtype=np.float32,
        )
        write_points(
            path, pos, chunk_shape=(10.0, 10.0, 10.0),
            bounds=([0.0, 0.0, 0.0], [40.0, 40.0, 40.0]),
        )
        return path

    def test_stamps_group_only(self) -> None:
        lg = get_resolution_level(open_store(self._store(), mode="r+"), 0)
        create_links_family(lg, delta=0, link_width=2, sid_ndim=3, directed=True)
        assert link_family_policy(lg, 0) == (2, 3, True, "canonical")
        # A cross-only family must not be forced to invent an intra array.
        assert list_link_offsets(lg, 0) == []

    def test_policy_survives_worker_writes(self) -> None:
        lg = get_resolution_level(open_store(self._store(), mode="r+"), 0)
        create_links_family(lg, delta=0, link_width=2, sid_ndim=3, directed=True)
        write_link_cells(
            lg, [[((0, 0, 0), 0), ((1, 0, 0), 0)]], sid_ndim=3, directed=True,
        )
        finalize_links(lg, delta=0)
        assert link_family_policy(lg, 0)[2] is True

    def test_conflicting_restamp_raises(self) -> None:
        lg = get_resolution_level(open_store(self._store(), mode="r+"), 0)
        create_links_family(lg, delta=0, link_width=2, sid_ndim=3, directed=True)
        with pytest.raises(Exception, match="family-wide"):
            create_links_family(
                lg, delta=0, link_width=2, sid_ndim=3, directed=False,
            )


class TestPublicIntrospection:
    """The four policy fields must be reachable without private imports.

    A consumer needs all of them to call parse_offsets / links_has_perm /
    create_links_array correctly; re-deriving them from raw meta keys is
    how downstreams drift out of sync with the format.
    """

    def test_link_family_policy_is_public_and_complete(self) -> None:
        lg = get_resolution_level(open_store(_graph_spanning_two_chunks(2)), 0)
        link_width, sid_ndim, directed, store = link_family_policy(lg, 0)
        assert (link_width, sid_ndim) == (2, 3)
        assert directed is False and store == "canonical"

    def test_link_family_policy_none_when_absent(self) -> None:
        path = os.path.join(tempfile.mkdtemp(), "p.zv")
        pos = np.array([[5.0, 5, 5]], dtype=np.float32)
        write_points(
            path, pos, chunk_shape=(10.0, 10.0, 10.0),
            bounds=([0.0, 0.0, 0.0], [40.0, 40.0, 40.0]),
        )
        lg = get_resolution_level(open_store(path), 0)
        assert link_family_policy(lg, 0) is None

    def test_iter_link_cells_enumerates_without_reading_records(self) -> None:
        from zarr_vectors.core.arrays import iter_link_cells

        lg = get_resolution_level(open_store(_graph_spanning_two_chunks(3)), 0)
        cells = list(iter_link_cells(lg, 0))
        assert cells, "iter_link_cells must expose the cell census"
        for seg, offsets, src_chunk, groups in cells:
            assert seg in list_link_offsets(lg, 0)
            assert len(src_chunk) == 3
            assert isinstance(offsets, tuple)
            assert groups is not None

    def test_cell_endpoint_chunks_inverts_placement(self) -> None:
        from zarr_vectors.core.arrays import cell_endpoint_chunks

        # Unscaled: the anchor is the identity, so endpoint k is src + o_k.
        one = (1, 1, 1)
        chunks = cell_endpoint_chunks((0, 0, 0), ((0, 0, 1),), one, one)
        assert chunks == ((0, 0, 0), (0, 0, 1))
        # Negative offsets resolve backwards.
        chunks = cell_endpoint_chunks((2, 0, 0), ((-1, 0, 0),), one, one)
        assert chunks == ((2, 0, 0), (1, 0, 0))


class TestVerticesDtypeIsHonoured:
    """Readers must decode at the dtype the store declares.

    A ``vertices/`` cell is a flat buffer with no inline header, so the
    Zarr ``data_type`` (``variable_length_bytes``) describes the container
    and says nothing about the payload; the element type lives only in the
    array's ``dtype`` attribute.  ``read_chunk_vertices`` used to default
    to ``float32`` and several callers hardcoded it, so a ``float64`` store
    decoded to garbage at twice the row count — silently, because an
    assumed dtype is not checkable against anything.

    This is the code half of the question that prompted the vlen-metadata
    doc fix: the spec says readers MUST honour the stored dtype, and this
    is what makes that true.
    """

    def _float64_store(self) -> str:
        path = os.path.join(tempfile.mkdtemp(), "f64.zv")
        pos = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float64,
        )
        write_points(
            path, pos, chunk_shape=(100.0, 100.0, 100.0),
            bounds=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
            dtype="float64",
        )
        return path

    def test_declared_dtype_is_readable(self) -> None:
        lg = get_resolution_level(open_store(self._float64_store()), 0)
        assert vertices_dtype(lg) == np.dtype("float64")

    def test_default_read_honours_declared_dtype(self) -> None:
        lg = get_resolution_level(open_store(self._float64_store()), 0)
        got = np.concatenate(
            [np.asarray(g) for g in read_chunk_vertices(lg, (0, 0, 0))], axis=0,
        )
        assert got.shape == (3, 3), got.shape
        np.testing.assert_allclose(
            got, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        )

    def test_wrong_dtype_corrupts_silently(self) -> None:
        # Pins WHY the default matters: an explicit wrong dtype does not
        # raise, it just returns different numbers.  Nothing in the blob
        # can catch it.
        lg = get_resolution_level(open_store(self._float64_store()), 0)
        bad = np.concatenate(
            [np.asarray(g)
             for g in read_chunk_vertices(lg, (0, 0, 0), dtype=np.float32)],
            axis=0,
        )
        assert not np.allclose(bad, [[1.0, 2.0, 3.0]] * 3)

    def test_float32_store_still_reads_float32(self) -> None:
        path = os.path.join(tempfile.mkdtemp(), "f32.zv")
        pos = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        write_points(
            path, pos, chunk_shape=(100.0, 100.0, 100.0),
            bounds=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
        )
        lg = get_resolution_level(open_store(path), 0)
        assert vertices_dtype(lg) == np.dtype("float32")
        got = np.concatenate(
            [np.asarray(g) for g in read_chunk_vertices(lg, (0, 0, 0))], axis=0,
        )
        np.testing.assert_allclose(got, [[1.0, 2.0, 3.0]])


class TestLinesCrossChunkEndpointIndices:
    """``write_lines`` must link each line to its OWN endpoints.

    Link endpoints are chunk-local vertex indices.  ``write_lines`` appends
    each cross-chunk endpoint as its own single-vertex fragment, and a
    chunk's vertices are its fragments concatenated in order — so the k-th
    line's endpoint sits at chunk-local index k.  Hardcoding 0 made every
    line in a chunk link to that chunk's *first* vertex: three distinct
    lines all decoded to the identical record.

    Pre-existing; invisible because ``read_lines`` reconstructs geometry
    from ``object_index/`` manifests, not from the links.
    """

    def _store(self, n: int = 3) -> str:
        eps = np.array(
            [[[float(i + 1)] * 3, [15.0 + i, float(i + 1), float(i + 1)]]
             for i in range(n)],
            dtype=np.float32,
        )
        path = os.path.join(tempfile.mkdtemp(), "l.zv")
        write_lines(
            path, eps, chunk_shape=(10.0, 10.0, 10.0),
            bounds=([0.0, 0.0, 0.0], [30.0, 30.0, 30.0]),
        )
        return path

    def test_endpoints_are_distinct_per_line(self) -> None:
        lg = get_resolution_level(open_store(self._store(3)), 0)
        records = read_links(lg, delta=0)
        assert len(records) == 3
        # Three lines crossing the same boundary must not collapse onto
        # one another.
        assert len({tuple(r) for r in records}) == 3
        vis = sorted(r[0][1] for r in records)
        assert vis == [0, 1, 2], vis

    def test_endpoints_resolve_to_their_own_line(self) -> None:
        lg = get_resolution_level(open_store(self._store(3)), 0)

        def chunk_verts(cc):
            return np.concatenate(
                [np.asarray(g) for g in read_chunk_vertices(lg, cc)], axis=0,
            )

        for (ca, va), (cb, vb) in read_links(lg, delta=0):
            pa = chunk_verts(ca)[va]
            pb = chunk_verts(cb)[vb]
            # In this fixture a line's two endpoints share y and z.
            assert abs(pa[1] - pb[1]) < 1e-6, (pa, pb)
            assert abs(pa[2] - pb[2]) < 1e-6, (pa, pb)


def test_links_group_is_a_group_not_an_array() -> None:
    # links/<delta> carries the family policy and holds one array per
    # offsets segment.  _is_per_chunk_array must stay depth-aware, or
    # _ensure_array_dir clobbers the group with an array.
    from zarr_vectors.core.arrays import _is_per_chunk_array

    assert _is_per_chunk_array(links_group_path(0)) is False
    assert _is_per_chunk_array(f"{links_group_path(0)}/0.0.0") is True
    assert _is_per_chunk_array(f"{links_group_path(-1)}/0.0.+1") is True
    assert _is_per_chunk_array("link_attributes/w/0") is False
    assert _is_per_chunk_array("link_attributes/w/0/0.0.+1") is True


class TestIntraCellDtypeIsHonoured:
    """``read_links`` must decode each cell at the dtype its array declares.

    ``write_chunk_links`` stamps the payload dtype onto the offsets array and
    ``iter_link_cells`` honours it, but ``read_links`` /
    ``read_links_for_tuple`` used to decode every cell at a hard-coded int64.
    An int32-declared array — which is what BRIDGE stamps for its per-chunk
    node graph via ``create_node_graph_array`` — then yields a quarter of the
    rows on an even element count and raises ``cannot reshape array of size N``
    on an odd one.
    """

    @staticmethod
    def _int32_intra_store(n_edges: int) -> tuple[str, np.ndarray]:
        """Store whose intra links array is declared (and written) int32."""
        from zarr_vectors.core.arrays import (
            create_links_array,
            create_vertices_array,
            write_chunk_links,
            write_chunk_vertices,
        )
        from zarr_vectors.core.store import create_store

        path = os.path.join(tempfile.mkdtemp(), "s.zarrvectors")
        root = create_store(
            path, bounds=([0.0, 0.0, 0.0], [64.0, 64.0, 64.0]),
            chunk_shape=(32.0, 32.0, 32.0), geometry_types=["point_cloud"],
        )
        lg = get_resolution_level(root, 0)

        n_verts = n_edges + 1
        pos = np.stack([
            np.linspace(0.0, 30.0, n_verts),
            np.zeros(n_verts), np.zeros(n_verts),
        ], axis=1).astype(np.float32)
        create_vertices_array(lg, dtype="float32")
        write_chunk_vertices(lg, (0, 0, 0), [pos], dtype=np.float32)

        edges = np.stack([
            np.arange(n_edges, dtype=np.int32),
            np.arange(1, n_edges + 1, dtype=np.int32),
        ], axis=1)
        create_links_array(lg, 2, dtype="int32", delta=0, sid_ndim=3)
        write_chunk_links(lg, (0, 0, 0), [edges], dtype=np.int32)
        return path, edges

    @pytest.mark.parametrize("n_edges", [43, 44])
    def test_odd_and_even_edge_counts_round_trip(self, n_edges: int) -> None:
        # 43 is the case that RAISED pre-fix; 44 is the one that silently
        # returned a quarter of the rows.
        path, edges = self._int32_intra_store(n_edges)
        lg = get_resolution_level(open_store(path), 0)

        records = read_links(lg, delta=0)
        assert len(records) == n_edges, (
            f"expected {n_edges} records, got {len(records)}"
        )
        got = np.asarray([[va, vb] for (_ca, va), (_cb, vb) in records], dtype=np.int32)
        assert np.array_equal(got, edges)

    def test_agrees_with_iter_link_cells(self) -> None:
        # iter_link_cells always honoured the stamp; the two readers must not
        # disagree about the same bytes.
        from zarr_vectors.core.arrays import iter_link_cells

        path, edges = self._int32_intra_store(43)
        lg = get_resolution_level(open_store(path), 0)

        via_cells = np.concatenate(
            [np.asarray(g) for _seg, _off, _src, groups in iter_link_cells(lg, 0)
             for g in groups],
            axis=0,
        )
        via_read = np.asarray(
            [[va, vb] for (_ca, va), (_cb, vb) in read_links(lg, delta=0)],
        )
        assert via_cells.shape == via_read.shape == edges.shape
        assert np.array_equal(via_cells, via_read)

    def test_int64_cells_are_unaffected(self) -> None:
        # The fallback must stay int64 for arrays written before the stamp.
        lg = get_resolution_level(open_store(_graph_spanning_two_chunks(5)), 0)
        assert len(read_links(lg, delta=0)) == 5
