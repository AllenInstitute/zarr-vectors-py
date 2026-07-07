"""Tests for the 0.8 per-tuple ``cross_chunk_links`` cell layout.

Covers the new canonical-sort key, ``perm_idx`` round-trip, per-tuple
fast-path reads, attribute alignment, and the canonical-sort
invariant enforced by the consistency validator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.core.arrays import (
    CrossChunkLinkPartition,
    create_cross_chunk_link_attributes_array,
    create_cross_chunk_links_array,
    read_cross_chunk_link_attributes,
    read_cross_chunk_link_attributes_for_tuple,
    read_cross_chunk_links,
    read_cross_chunk_links_for_tuple,
    write_cross_chunk_link_attributes,
    write_cross_chunk_links,
)
from zarr_vectors.core.paths import (
    cross_chunk_links_path, format_cell_key, parse_cell_key,
)
from zarr_vectors.core.store import create_store, get_resolution_level
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.spatial.boundary import (
    apply_perm_inverse, canonical_sort,
)


def _new_lg(tmp_path: Path):
    root = create_store(
        str(tmp_path / "store.zv"),
        bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
        chunk_shape=(100.0, 100.0, 100.0),
        geometry_types=["graph"],
        ndim=3,
    )
    return get_resolution_level(root, 0)


class TestPermIdxRoundTrip:
    """Lehmer perm_idx must round-trip every permutation for L=1..4."""

    @pytest.mark.parametrize("L", [1, 2, 3, 4])
    def test_every_permutation_round_trips(self, L: int) -> None:
        import itertools
        for p in itertools.permutations(range(L)):
            sorted_rec, perm_idx = canonical_sort([
                ((i,), 100 + i) for i in p
            ])
            recovered = apply_perm_inverse(sorted_rec, perm_idx, L)
            expected = [((i,), 100 + i) for i in p]
            assert recovered == expected, (L, p, perm_idx, recovered)


class TestEdgesL2:
    """L=2 edges: 6-D cell keys, direction preserved."""

    def test_cell_key_dimensionality(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        create_cross_chunk_links_array(lg, delta=0)
        edges = [
            [((0, 0, 0), 5), ((1, 0, 0), 10)],
            [((1, 0, 0), 5), ((2, 0, 0), 7)],
        ]
        partition = write_cross_chunk_links(lg, edges, sid_ndim=3, delta=0)
        # Cell keys are 6 dotted components (2 chunks × 3 coords).
        for ckey in partition.cell_indices:
            assert len(ckey.split(".")) == 6

    def test_direction_preserved_across_canonical_sort(
        self, tmp_path: Path,
    ) -> None:
        lg = _new_lg(tmp_path)
        create_cross_chunk_links_array(lg, delta=0)
        # Write an edge whose canonical-min endpoint is NOT input[0].
        edges = [
            [((2, 0, 0), 8), ((0, 0, 0), 3)],   # canonical reorders endpoints
        ]
        write_cross_chunk_links(lg, edges, sid_ndim=3, delta=0)
        out = read_cross_chunk_links(lg, delta=0)
        assert out[0] == (((2, 0, 0), 8), ((0, 0, 0), 3))

    def test_pair_lookup_any_input_order(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        create_cross_chunk_links_array(lg, delta=0)
        edges = [
            [((0, 0, 0), 1), ((4, 4, 4), 9)],
            [((4, 4, 4), 2), ((0, 0, 0), 7)],
            [((1, 0, 0), 3), ((2, 0, 0), 6)],
        ]
        write_cross_chunk_links(lg, edges, sid_ndim=3, delta=0)

        # Lookup (0,0,0)-(4,4,4) — should return only edges 0 and 1
        # regardless of order we pass the chunks.
        r_ab = read_cross_chunk_links_for_tuple(
            lg, [(0, 0, 0), (4, 4, 4)], delta=0,
        )
        r_ba = read_cross_chunk_links_for_tuple(
            lg, [(4, 4, 4), (0, 0, 0)], delta=0,
        )
        assert len(r_ab) == 2 == len(r_ba)
        assert set(r_ab) == set(r_ba)

        # Lookup an unrelated pair — empty.
        r_xy = read_cross_chunk_links_for_tuple(
            lg, [(5, 5, 5), (6, 6, 6)], delta=0,
        )
        assert r_xy == []


class TestTrianglesL3:
    """L=3 triangle faces: 9-D cell keys, winding preserved across 3+ chunks."""

    def test_face_spanning_three_chunks_cell_key_9d(
        self, tmp_path: Path,
    ) -> None:
        lg = _new_lg(tmp_path)
        create_cross_chunk_links_array(lg, delta=0, link_width=3)
        faces = [
            [((0, 0, 0), 5), ((1, 0, 0), 12), ((1, 1, 1), 7)],
        ]
        partition = write_cross_chunk_links(
            lg, faces, sid_ndim=3, delta=0, link_width=3,
        )
        ckey = next(iter(partition.cell_indices))
        assert len(ckey.split(".")) == 9

    def test_winding_preserved_for_triangle_with_3_chunks(
        self, tmp_path: Path,
    ) -> None:
        lg = _new_lg(tmp_path)
        create_cross_chunk_links_array(lg, delta=0, link_width=3)
        # Faces written in two different windings of the SAME triangle.
        face_a = [((0, 0, 0), 5), ((1, 0, 0), 12), ((1, 1, 1), 7)]
        face_b = [((1, 1, 1), 7), ((1, 0, 0), 12), ((0, 0, 0), 5)]  # opposite winding
        write_cross_chunk_links(
            lg, [face_a, face_b], sid_ndim=3, delta=0, link_width=3,
        )
        out = read_cross_chunk_links(lg, delta=0)
        # Both faces share the same canonical cell — but must be
        # read back distinct (different windings).
        assert len(out) == 2
        out_set = set(out)
        assert tuple(tuple(t) for t in face_a) in out_set
        assert tuple(tuple(t) for t in face_b) in out_set


class TestCanonicalSortInvariant:
    """The validator must flag a malformed store whose cell key isn't sorted."""

    def test_hand_crafted_unsorted_cell_key_flagged(
        self, tmp_path: Path,
    ) -> None:
        # Validator skips levels with no vertex data, so use write_graph
        # to populate a real store, then inject a malformed cell key.
        from zarr_vectors.types.graphs import write_graph
        from zarr_vectors.validate.consistency import validate_consistency
        from zarr_vectors.core.store import open_store, get_resolution_level

        store_path = tmp_path / "store.zv"
        positions = np.array([
            [50.0, 50.0, 50.0],
            [150.0, 50.0, 50.0],
        ], dtype=np.float32)
        edges = np.array([[0, 1]], dtype=np.int64)
        write_graph(
            str(store_path), positions, edges,
            chunk_shape=(100.0, 100.0, 100.0),
        )
        root = open_store(str(store_path), mode="r+")
        lg = get_resolution_level(root, 0)
        # Re-write a cell under a NON-canonical key to simulate
        # corruption (a writer that forgot to sort).
        family = cross_chunk_links_path(0)
        bad_key = format_cell_key([(1, 0, 0), (0, 0, 0)])  # swapped
        from zarr_vectors.encoding.ragged import encode_ragged_blob
        blob = encode_ragged_blob(
            [np.array([0, 5, 10], dtype=np.int64)], np.dtype(np.int64),
        )
        lg.write_bytes(family, bad_key, blob)

        result = validate_consistency(str(store_path))
        bad_messages = [
            e for e in result.errors
            if "canonical-sort invariant" in e
        ]
        assert bad_messages, f"expected canonical-sort error; got {result.errors}"


class TestAttributesAlignment:
    """Cell-aligned attribute writes preserve per-record correspondence."""

    def test_attributes_aligned_via_partition(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        create_cross_chunk_links_array(lg, delta=0)
        create_cross_chunk_link_attributes_array(lg, "weight", delta=0)

        # 3 records distributed across 2 cells.
        records = [
            [((0, 0, 0), 5), ((1, 0, 0), 10)],     # cell A
            [((0, 0, 0), 6), ((1, 0, 0), 11)],     # cell A
            [((1, 0, 0), 5), ((2, 0, 0), 7)],      # cell B
        ]
        weights = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        partition = write_cross_chunk_links(
            lg, records, sid_ndim=3, delta=0,
        )
        write_cross_chunk_link_attributes(
            lg, "weight", weights, num_links=3, delta=0,
            partition=partition,
        )

        # Global read order matches link read order.
        link_back = read_cross_chunk_links(lg, delta=0)
        weight_back = read_cross_chunk_link_attributes(
            lg, "weight", delta=0,
        )
        assert weight_back.shape == (3,)
        # Pair link rec ↔ weight by sort order; weights must match.
        expected_weight_per_record = {
            tuple(tuple(t) for t in rec): weights[i]
            for i, rec in enumerate(records)
        }
        for link_rec, w in zip(link_back, weight_back):
            assert np.isclose(w, expected_weight_per_record[link_rec])

    def test_for_tuple_attribute_read(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        create_cross_chunk_links_array(lg, delta=0)
        create_cross_chunk_link_attributes_array(lg, "weight", delta=0)
        records = [
            [((0, 0, 0), 5), ((1, 0, 0), 10)],
            [((0, 0, 0), 6), ((1, 0, 0), 11)],
            [((1, 0, 0), 5), ((2, 0, 0), 7)],
        ]
        weights = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        partition = write_cross_chunk_links(
            lg, records, sid_ndim=3, delta=0,
        )
        write_cross_chunk_link_attributes(
            lg, "weight", weights, num_links=3, delta=0,
            partition=partition,
        )

        w_ab = read_cross_chunk_link_attributes_for_tuple(
            lg, "weight", [(0, 0, 0), (1, 0, 0)], delta=0,
        )
        assert sorted(w_ab.tolist()) == [
            pytest.approx(0.1), pytest.approx(0.2),
        ]
        w_bc = read_cross_chunk_link_attributes_for_tuple(
            lg, "weight", [(1, 0, 0), (2, 0, 0)], delta=0,
        )
        assert w_bc.tolist() == [pytest.approx(0.3)]

    def test_append_requires_partition(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        create_cross_chunk_links_array(lg, delta=0)
        create_cross_chunk_link_attributes_array(lg, "weight", delta=0)

        records = [[((0, 0, 0), 5), ((1, 0, 0), 10)]]
        p0 = write_cross_chunk_links(lg, records, sid_ndim=3, delta=0)
        write_cross_chunk_link_attributes(
            lg, "weight", np.array([0.1], dtype=np.float32),
            num_links=1, delta=0, partition=p0,
        )

        # Forgetting partition= in append must raise.
        new_records = [[((0, 0, 0), 6), ((1, 0, 0), 11)]]
        p1 = write_cross_chunk_links(
            lg, new_records, sid_ndim=3, delta=0, mode="append",
        )
        with pytest.raises(ArrayError, match="partition"):
            write_cross_chunk_link_attributes(
                lg, "weight", np.array([0.2], dtype=np.float32),
                num_links=2, delta=0, mode="append",
            )
        # Passing partition works.
        write_cross_chunk_link_attributes(
            lg, "weight", np.array([0.2], dtype=np.float32),
            num_links=2, delta=0, mode="append", partition=p1,
        )
        back = read_cross_chunk_link_attributes(lg, "weight", delta=0)
        assert back.shape == (2,)


class TestDirectedStoreMeta:
    """directed / store flags are stamped in the family .zattrs."""

    def test_defaults(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        write_cross_chunk_links(
            lg, [[((0, 0, 0), 1), ((1, 0, 0), 2)]], sid_ndim=3, delta=0,
        )
        meta = lg.read_array_meta(cross_chunk_links_path(0))
        assert meta["directed"] is False
        assert meta["store"] == "canonical"
        assert meta["num_links"] == 1
        assert meta["num_physical_records"] == 1

    def test_directed_flag_and_counts(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        write_cross_chunk_links(
            lg, [[((0, 0, 0), 1), ((1, 0, 0), 2)]], sid_ndim=3, delta=0,
            directed=True, store="duplicate",
        )
        meta = lg.read_array_meta(cross_chunk_links_path(0))
        assert meta["directed"] is True
        assert meta["store"] == "duplicate"
        assert meta["num_links"] == 1
        assert meta["num_physical_records"] == 2  # both orderings on disk


class TestDirectedReads:
    """directed families keep A→B and B→A as distinct directional cells."""

    def test_both_orientations_distinct(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        fwd = [((0, 0, 0), 1), ((1, 0, 0), 2)]
        rev = [((1, 0, 0), 9), ((0, 0, 0), 8)]
        p = write_cross_chunk_links(
            lg, [fwd, rev], sid_ndim=3, delta=0, directed=True,
        )
        assert set(p.cell_indices) == {"0.0.0.1.0.0", "1.0.0.0.0.0"}
        # for_tuple must respect input order (no canonical sort).
        got_fwd = read_cross_chunk_links_for_tuple(
            lg, [(0, 0, 0), (1, 0, 0)], delta=0,
        )
        got_rev = read_cross_chunk_links_for_tuple(
            lg, [(1, 0, 0), (0, 0, 0)], delta=0,
        )
        assert got_fwd == [(((0, 0, 0), 1), ((1, 0, 0), 2))]
        assert got_rev == [(((1, 0, 0), 9), ((0, 0, 0), 8))]

    def test_directed_full_scan_input_order(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        # canonical would swap (2,) before (0,); directed must not.
        write_cross_chunk_links(
            lg, [[((2, 0, 0), 8), ((0, 0, 0), 3)]], sid_ndim=3, delta=0,
            directed=True,
        )
        out = read_cross_chunk_links(lg, delta=0)
        assert out == [(((2, 0, 0), 8), ((0, 0, 0), 3))]

    def test_append_mismatched_directed_raises(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        write_cross_chunk_links(
            lg, [[((0, 0, 0), 1), ((1, 0, 0), 2)]], sid_ndim=3, delta=0,
            directed=True,
        )
        with pytest.raises(ArrayError, match="directed"):
            write_cross_chunk_links(
                lg, [[((0, 0, 0), 3), ((1, 0, 0), 4)]], sid_ndim=3, delta=0,
                mode="append", directed=False,
            )


class TestDuplicateStore:
    """store='duplicate' fans each record across incident-chunk cells."""

    def test_edge_in_both_prefix_cells(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        p = write_cross_chunk_links(
            lg, [[((0, 0, 0), 1), ((4, 4, 4), 9)]], sid_ndim=3, delta=0,
            store="duplicate",
        )
        assert set(p.cell_indices) == {"0.0.0.4.4.4", "4.4.4.0.0.0"}
        # Incidence read from either chunk finds the record by prefix.
        from_a = read_cross_chunk_links_for_tuple(
            lg, [(0, 0, 0), (4, 4, 4)], delta=0,
        )
        assert from_a == [(((0, 0, 0), 1), ((4, 4, 4), 9))]
        # Full scan returns one copy per cell (documented duplicate behavior).
        assert len(read_cross_chunk_links(lg, delta=0)) == 2

    def test_attributes_replicate_across_copies(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        create_cross_chunk_link_attributes_array(lg, "weight", delta=0)
        records = [
            [((0, 0, 0), 1), ((4, 4, 4), 9)],
            [((1, 0, 0), 2), ((2, 0, 0), 3)],
        ]
        weights = np.array([0.5, 0.7], dtype=np.float32)
        partition = write_cross_chunk_links(
            lg, records, sid_ndim=3, delta=0, store="duplicate",
        )
        write_cross_chunk_link_attributes(
            lg, "weight", weights, num_links=2, delta=0, partition=partition,
        )
        # Links and attributes are both physically duplicated and stay
        # row-aligned under the global read.
        links = read_cross_chunk_links(lg, delta=0)
        attrs = read_cross_chunk_link_attributes(lg, "weight", delta=0)
        assert len(links) == len(attrs) == 4  # 2 records × 2 cells each
        expected = {
            tuple(tuple(t) for t in rec): weights[i]
            for i, rec in enumerate(records)
        }
        for rec, w in zip(links, attrs):
            assert np.isclose(w, expected[rec])

    def test_duplicate_attributes_require_partition(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        create_cross_chunk_link_attributes_array(lg, "weight", delta=0)
        write_cross_chunk_links(
            lg, [[((0, 0, 0), 1), ((4, 4, 4), 9)]], sid_ndim=3, delta=0,
            store="duplicate",
        )
        with pytest.raises(ArrayError, match="duplicate"):
            write_cross_chunk_link_attributes(
                lg, "weight", np.array([0.5], dtype=np.float32),
                num_links=1, delta=0,   # no partition → must raise
            )


class TestCellKeyHelpers:
    """parse_cell_key ↔ format_cell_key round-trip."""

    def test_round_trip_l2(self) -> None:
        chunks = [(0, 1, 2), (3, 4, 5)]
        key = format_cell_key(chunks)
        recovered = parse_cell_key(key, sid_ndim=3, link_width=2)
        assert recovered == ((0, 1, 2), (3, 4, 5))

    def test_round_trip_l3(self) -> None:
        chunks = [(0, 0, 0), (1, 0, 0), (1, 1, 1)]
        key = format_cell_key(chunks)
        recovered = parse_cell_key(key, sid_ndim=3, link_width=3)
        assert recovered == ((0, 0, 0), (1, 0, 0), (1, 1, 1))

    def test_arity_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_cell_key("0.1.2.3", sid_ndim=3, link_width=2)
