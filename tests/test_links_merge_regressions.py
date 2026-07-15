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
