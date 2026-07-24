"""Tests for the ``links/<delta>/<offsets>/`` cell layout.

Covers the offsets-segment placement, ``perm_idx`` round-trip, per-tuple
fast-path reads, attribute alignment, and the canonical-sort invariant
enforced by the consistency validator.

Connectivity is ONE family per delta: ``links/<delta>`` is a group whose
children are one rank-D array per relative-offset segment, and a record
lives in the cell of its **source** chunk.  An intra-chunk link is just a
record whose offsets are all zero, so there is no separate cross-chunk
family and no dotted multi-chunk cell key any more — the relationship
lives in the path segment, the source chunk in the cell key.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.core.arrays import (
    LinkPartition,
    create_link_attributes_array,
    create_links_array,
    list_link_offsets,
    read_link_attributes,
    read_link_attributes_for_tuple,
    read_links,
    read_links_for_tuple,
    write_chunk_links,
    write_link_attributes,
    write_links,
)
from zarr_vectors.core.paths import (
    format_offsets,
    intra_offsets,
    links_group_path,
    parse_offsets,
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
    """L=2 edges: one 3-component offset per segment, direction preserved."""

    def test_offsets_segment_arity(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        create_links_array(lg, link_width=2, delta=0, sid_ndim=3)
        edges = [
            [((0, 0, 0), 5), ((1, 0, 0), 10)],
            [((1, 0, 0), 5), ((2, 0, 0), 7)],
        ]
        partition = write_links(lg, edges, sid_ndim=3, delta=0)
        assert isinstance(partition, LinkPartition)
        # A key is (offsets_segment, source_chunk).  L=2 → link_width-1 = 1
        # offset of sid_ndim = 3 components; the source chunk carries the
        # other 3.  The pre-merge layout dotted both into one 6-D cell key.
        for seg, src in partition.cell_indices:
            assert len(seg.split("_")) == 1
            assert len(seg.split(".")) == 3
            assert len(src) == 3
            # And it decodes back to a single 3-tuple offset.
            offsets = parse_offsets(seg, sid_ndim=3, link_width=2)
            assert len(offsets) == 1 and len(offsets[0]) == 3

    def test_both_edges_share_one_offsets_array(self, tmp_path: Path) -> None:
        # Both edges step +1 along x, so they share one offsets array and
        # differ only by cell — that factoring is the point of the layout.
        lg = _new_lg(tmp_path)
        write_links(lg, [
            [((0, 0, 0), 5), ((1, 0, 0), 10)],
            [((1, 0, 0), 5), ((2, 0, 0), 7)],
        ], sid_ndim=3, delta=0)
        assert list_link_offsets(lg, 0) == ["+1.0.0"]

    def test_direction_preserved_across_canonical_sort(
        self, tmp_path: Path,
    ) -> None:
        lg = _new_lg(tmp_path)
        create_links_array(lg, link_width=2, delta=0, sid_ndim=3)
        # Write an edge whose canonical-min endpoint is NOT input[0].
        edges = [
            [((2, 0, 0), 8), ((0, 0, 0), 3)],   # canonical reorders endpoints
        ]
        write_links(lg, edges, sid_ndim=3, delta=0)
        out = read_links(lg, delta=0)
        assert out[0] == (((2, 0, 0), 8), ((0, 0, 0), 3))
        # Stored canonically: source is the lex-min chunk, offset positive.
        assert "+2.0.0" in list_link_offsets(lg, 0)

    def test_pair_lookup_any_input_order(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        create_links_array(lg, link_width=2, delta=0, sid_ndim=3)
        edges = [
            [((0, 0, 0), 1), ((4, 4, 4), 9)],
            [((4, 4, 4), 2), ((0, 0, 0), 7)],
            [((1, 0, 0), 3), ((2, 0, 0), 6)],
        ]
        write_links(lg, edges, sid_ndim=3, delta=0)

        # Lookup (0,0,0)-(4,4,4) — should return only edges 0 and 1
        # regardless of order we pass the chunks.
        r_ab = read_links_for_tuple(
            lg, [(0, 0, 0), (4, 4, 4)], delta=0,
        )
        r_ba = read_links_for_tuple(
            lg, [(4, 4, 4), (0, 0, 0)], delta=0,
        )
        assert len(r_ab) == 2 == len(r_ba)
        assert set(r_ab) == set(r_ba)

        # Lookup an unrelated pair — empty.
        r_xy = read_links_for_tuple(
            lg, [(5, 5, 5), (6, 6, 6)], delta=0,
        )
        assert r_xy == []

    def test_intra_tuple_lookup_reads_all_zero_offsets(
        self, tmp_path: Path,
    ) -> None:
        # An all-equal tuple resolves to the intra array — the family that
        # used to be a standalone links/<delta> array.
        lg = _new_lg(tmp_path)
        write_links(lg, [
            [((1, 1, 1), 4), ((1, 1, 1), 5)],
            [((1, 1, 1), 6), ((2, 1, 1), 7)],
        ], sid_ndim=3, delta=0)
        got = read_links_for_tuple(lg, [(1, 1, 1), (1, 1, 1)], delta=0)
        assert got == [(((1, 1, 1), 4), ((1, 1, 1), 5))]


class TestTrianglesL3:
    """L=3 faces: two offsets per segment, winding preserved across chunks."""

    def test_face_spanning_three_chunks_segment_arity(
        self, tmp_path: Path,
    ) -> None:
        lg = _new_lg(tmp_path)
        create_links_array(lg, link_width=3, delta=0, sid_ndim=3)
        faces = [
            [((0, 0, 0), 5), ((1, 0, 0), 12), ((1, 1, 1), 7)],
        ]
        partition = write_links(
            lg, faces, sid_ndim=3, delta=0, link_width=3,
        )
        (seg, src), = partition.cell_indices
        # L=3 → two offsets joined by "_", each 3 dotted components; the
        # source chunk is the third endpoint.  (The pre-merge layout dotted
        # all three chunks into one 9-D cell key.)
        assert len(seg.split("_")) == 2
        assert all(len(part.split(".")) == 3 for part in seg.split("_"))
        assert len(src) == 3
        offsets = parse_offsets(seg, sid_ndim=3, link_width=3)
        assert len(offsets) == 2 and all(len(o) == 3 for o in offsets)

    def test_winding_preserved_for_triangle_with_3_chunks(
        self, tmp_path: Path,
    ) -> None:
        lg = _new_lg(tmp_path)
        create_links_array(lg, link_width=3, delta=0, sid_ndim=3)
        # Faces written in two different windings of the SAME triangle.
        face_a = [((0, 0, 0), 5), ((1, 0, 0), 12), ((1, 1, 1), 7)]
        face_b = [((1, 1, 1), 7), ((1, 0, 0), 12), ((0, 0, 0), 5)]  # opposite winding
        p = write_links(
            lg, [face_a, face_b], sid_ndim=3, delta=0, link_width=3,
        )
        out = read_links(lg, delta=0)
        # Both faces share the same canonical cell — but must be
        # read back distinct (different windings).
        assert len(out) == 2
        out_set = set(out)
        assert tuple(tuple(t) for t in face_a) in out_set
        assert tuple(tuple(t) for t in face_b) in out_set
        # Both windings really did land in ONE (offsets, cell) bucket, so
        # perm_idx is the only thing telling them apart.
        assert set(p.cell_indices) == {("+1.0.0_+1.+1.+1", (0, 0, 0))}
        assert sorted(p.cell_indices[("+1.0.0_+1.+1.+1", (0, 0, 0))]) == [0, 1]


class TestCanonicalSortInvariant:
    """The validator must flag a store whose offsets segment isn't canonical."""

    def _graph_store(self, tmp_path: Path):
        from zarr_vectors.types.graphs import write_graph
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
            bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
        )
        root = open_store(str(store_path), mode="r+")
        return store_path, get_resolution_level(root, 0)

    def test_hand_crafted_negative_offset_segment_flagged(
        self, tmp_path: Path,
    ) -> None:
        # Validator skips levels with no vertex data, so use write_graph
        # to populate a real store, then inject a malformed offsets array.
        # Under the offset layout the canonical-sort invariant is a
        # property of the DIRECTORY NAME: a canonical family stores each
        # undirected record once, under the lexicographically-positive
        # offset.  A cell filed under "0.0.-1" is the writer-forgot-to-sort
        # corruption the pre-merge swapped cell key stood for.
        from zarr_vectors.validate.consistency import validate_consistency

        store_path, lg = self._graph_store(tmp_path)
        bad_offsets = ((0, 0, -1),)
        create_links_array(
            lg, link_width=2, delta=0, sid_ndim=3, offsets=bad_offsets,
        )
        write_chunk_links(
            lg, (0, 0, 0), [np.array([[5, 10]], dtype=np.int64)],
            delta=0, offsets=bad_offsets,
        )
        assert format_offsets(bad_offsets) in list_link_offsets(lg, 0)

        result = validate_consistency(str(store_path))
        bad_messages = [
            e for e in result.errors
            if "lexicographically negative" in e
        ]
        assert bad_messages, (
            f"expected lex-negative offset error; got {result.errors}"
        )

    def test_hand_crafted_unsorted_offsets_flagged(
        self, tmp_path: Path,
    ) -> None:
        # A canonical L=3 record sorts its endpoints, so its offsets are
        # non-decreasing.  "0.0.+1_0.0.0" descends — the direct analogue of
        # the pre-merge unsorted cell key.
        from zarr_vectors.validate.consistency import validate_consistency
        from zarr_vectors.core.store import open_store, get_resolution_level
        from zarr_vectors.types.meshes import write_mesh

        store_path = tmp_path / "mesh.zv"
        verts = np.array([
            [50, 50, 50], [60, 50, 50], [50, 60, 50],
        ], dtype="f4")
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        write_mesh(
            str(store_path), verts, faces,
            chunk_shape=(100.0, 100.0, 100.0),
            bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
        )
        root = open_store(str(store_path), mode="r+")
        lg = get_resolution_level(root, 0)

        bad_offsets = ((0, 0, 1), (0, 0, 0))   # descends: violates the sort
        assert format_offsets(bad_offsets) == "0.0.+1_0.0.0"
        create_links_array(
            lg, link_width=3, delta=0, sid_ndim=3, offsets=bad_offsets,
        )
        write_chunk_links(
            lg, (0, 0, 0), [np.array([[0, 5, 10]], dtype=np.int64)],
            delta=0, offsets=bad_offsets,
        )

        result = validate_consistency(str(store_path))
        bad_messages = [
            e for e in result.errors
            if "canonical-sort invariant" in e
        ]
        assert bad_messages, (
            f"expected canonical-sort error; got {result.errors}"
        )

    def test_well_formed_store_not_flagged(self, tmp_path: Path) -> None:
        # The negative control: without the injected array the same store
        # must be clean, so the tests above are detecting the corruption
        # rather than something write_graph always emits.
        from zarr_vectors.validate.consistency import validate_consistency

        store_path, _lg = self._graph_store(tmp_path)
        result = validate_consistency(str(store_path))
        assert not [
            e for e in result.errors
            if "canonical-sort" in e or "lexicographically negative" in e
        ], result.errors


class TestValidatorModeAware:
    """Directed / duplicate families must not trip the canonical-sort check."""

    def _graph_store(self, tmp_path: Path):
        from zarr_vectors.types.graphs import write_graph
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
            bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
        )
        root = open_store(str(store_path), mode="r+")
        return store_path, get_resolution_level(root, 0)

    def test_directed_not_flagged(self, tmp_path: Path) -> None:
        from zarr_vectors.validate.consistency import validate_consistency

        store_path, lg = self._graph_store(tmp_path)
        # directed/store is family-wide, so flipping it means dropping the
        # family rather than replacing one offsets array within it.
        lg.delete_subtree(links_group_path(0))
        # Directed edge stored in non-canonical (input) order: it files
        # under a lex-negative offset, which is legal for a directed family.
        write_links(
            lg, [[((1, 0, 0), 1), ((0, 0, 0), 0)]], sid_ndim=3, delta=0,
            directed=True,
        )
        assert list_link_offsets(lg, 0) == ["-1.0.0"]
        result = validate_consistency(str(store_path))
        assert not [e for e in result.errors if "canonical-sort" in e], (
            result.errors
        )
        assert not [
            e for e in result.errors if "lexicographically negative" in e
        ], result.errors

    def test_duplicate_not_flagged_counts_ok(self, tmp_path: Path) -> None:
        from zarr_vectors.validate.consistency import validate_consistency

        store_path, lg = self._graph_store(tmp_path)
        lg.delete_subtree(links_group_path(0))
        write_links(
            lg, [[((0, 0, 0), 0), ((1, 0, 0), 1)]], sid_ndim=3, delta=0,
            store="duplicate",
        )
        # A duplicate family leads with each incident chunk, so one copy
        # legitimately sits under a lex-negative offset.
        assert sorted(list_link_offsets(lg, 0)) == ["+1.0.0", "-1.0.0"]
        result = validate_consistency(str(store_path))
        assert not [e for e in result.errors if "canonical-sort" in e], (
            result.errors
        )
        assert not [
            e for e in result.errors if "lexicographically negative" in e
        ], result.errors
        assert not [
            e for e in result.errors if "num_physical_records" in e
        ], result.errors

    def test_physical_count_mismatch_flagged(self, tmp_path: Path) -> None:
        from zarr_vectors.validate.consistency import validate_consistency

        store_path, lg = self._graph_store(tmp_path)
        write_links(
            lg, [[((0, 0, 0), 0), ((1, 0, 0), 1)]], sid_ndim=3, delta=0,
        )
        family = links_group_path(0)
        meta = lg.read_array_meta(family)
        meta["num_physical_records"] = 99  # corrupt the recorded count
        lg.write_array_meta(family, meta)
        result = validate_consistency(str(store_path))
        assert [
            e for e in result.errors if "num_physical_records" in e
        ], result.errors


class TestAttributesAlignment:
    """Cell-aligned attribute writes preserve per-record correspondence."""

    def test_attributes_aligned_via_partition(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        create_links_array(lg, link_width=2, delta=0, sid_ndim=3)
        create_link_attributes_array(lg, "weight", delta=0, sid_ndim=3)

        # 3 records distributed across 2 cells.
        records = [
            [((0, 0, 0), 5), ((1, 0, 0), 10)],     # cell A
            [((0, 0, 0), 6), ((1, 0, 0), 11)],     # cell A
            [((1, 0, 0), 5), ((2, 0, 0), 7)],      # cell B
        ]
        weights = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        partition = write_links(
            lg, records, sid_ndim=3, delta=0,
        )
        write_link_attributes(
            lg, "weight", weights, num_links=3, delta=0,
            partition=partition,
        )

        # Global read order matches link read order.
        link_back = read_links(lg, delta=0)
        weight_back = read_link_attributes(
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
        create_links_array(lg, link_width=2, delta=0, sid_ndim=3)
        create_link_attributes_array(lg, "weight", delta=0, sid_ndim=3)
        records = [
            [((0, 0, 0), 5), ((1, 0, 0), 10)],
            [((0, 0, 0), 6), ((1, 0, 0), 11)],
            [((1, 0, 0), 5), ((2, 0, 0), 7)],
        ]
        weights = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        partition = write_links(
            lg, records, sid_ndim=3, delta=0,
        )
        write_link_attributes(
            lg, "weight", weights, num_links=3, delta=0,
            partition=partition,
        )

        w_ab = read_link_attributes_for_tuple(
            lg, "weight", [(0, 0, 0), (1, 0, 0)], delta=0,
        )
        assert sorted(w_ab.tolist()) == [
            pytest.approx(0.1), pytest.approx(0.2),
        ]
        w_bc = read_link_attributes_for_tuple(
            lg, "weight", [(1, 0, 0), (2, 0, 0)], delta=0,
        )
        assert w_bc.tolist() == [pytest.approx(0.3)]

    def test_append_requires_partition(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        create_links_array(lg, link_width=2, delta=0, sid_ndim=3)
        create_link_attributes_array(lg, "weight", delta=0, sid_ndim=3)

        records = [[((0, 0, 0), 5), ((1, 0, 0), 10)]]
        p0 = write_links(lg, records, sid_ndim=3, delta=0)
        write_link_attributes(
            lg, "weight", np.array([0.1], dtype=np.float32),
            num_links=1, delta=0, partition=p0,
        )

        # Forgetting partition= in append must raise.
        new_records = [[((0, 0, 0), 6), ((1, 0, 0), 11)]]
        p1 = write_links(
            lg, new_records, sid_ndim=3, delta=0, mode="append",
        )
        with pytest.raises(ArrayError, match="partition"):
            write_link_attributes(
                lg, "weight", np.array([0.2], dtype=np.float32),
                num_links=2, delta=0, mode="append",
            )
        # Passing partition works.
        write_link_attributes(
            lg, "weight", np.array([0.2], dtype=np.float32),
            num_links=2, delta=0, mode="append", partition=p1,
        )
        back = read_link_attributes(lg, "weight", delta=0)
        assert back.shape == (2,)


class TestDirectedStoreMeta:
    """directed / store flags are stamped on the links/<delta> family group."""

    def test_defaults(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        write_links(
            lg, [[((0, 0, 0), 1), ((1, 0, 0), 2)]], sid_ndim=3, delta=0,
        )
        # Policy is family-wide, so it lives on the <delta> GROUP — every
        # offsets array beneath it decodes against these.
        meta = lg.read_array_meta(links_group_path(0))
        assert meta["directed"] is False
        assert meta["store"] == "canonical"
        assert meta["num_links"] == 1
        assert meta["num_physical_records"] == 1
        assert meta["link_width"] == 2
        assert meta["sid_ndim"] == 3

    def test_directed_flag_and_counts(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        write_links(
            lg, [[((0, 0, 0), 1), ((1, 0, 0), 2)]], sid_ndim=3, delta=0,
            directed=True, store="duplicate",
        )
        meta = lg.read_array_meta(links_group_path(0))
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
        p = write_links(
            lg, [fwd, rev], sid_ndim=3, delta=0, directed=True,
        )
        # Opposite offsets at different cells — the offset sign carries the
        # direction that the pre-merge layout carried in the cell-key order.
        assert set(p.cell_indices) == {
            ("+1.0.0", (0, 0, 0)),
            ("-1.0.0", (1, 0, 0)),
        }
        # for_tuple must respect input order (no canonical sort).
        got_fwd = read_links_for_tuple(
            lg, [(0, 0, 0), (1, 0, 0)], delta=0,
        )
        got_rev = read_links_for_tuple(
            lg, [(1, 0, 0), (0, 0, 0)], delta=0,
        )
        assert got_fwd == [(((0, 0, 0), 1), ((1, 0, 0), 2))]
        assert got_rev == [(((1, 0, 0), 9), ((0, 0, 0), 8))]

    def test_directed_full_scan_input_order(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        # canonical would swap (2,) before (0,); directed must not.
        write_links(
            lg, [[((2, 0, 0), 8), ((0, 0, 0), 3)]], sid_ndim=3, delta=0,
            directed=True,
        )
        out = read_links(lg, delta=0)
        assert out == [(((2, 0, 0), 8), ((0, 0, 0), 3))]
        assert list_link_offsets(lg, 0) == ["-2.0.0"]

    def test_append_mismatched_directed_raises(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        write_links(
            lg, [[((0, 0, 0), 1), ((1, 0, 0), 2)]], sid_ndim=3, delta=0,
            directed=True,
        )
        with pytest.raises(ArrayError, match="directed"):
            write_links(
                lg, [[((0, 0, 0), 3), ((1, 0, 0), 4)]], sid_ndim=3, delta=0,
                mode="append", directed=False,
            )


class TestDuplicateStore:
    """store='duplicate' fans each record across incident-chunk cells."""

    def test_edge_in_both_offset_cells(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        p = write_links(
            lg, [[((0, 0, 0), 1), ((4, 4, 4), 9)]], sid_ndim=3, delta=0,
            store="duplicate",
        )
        # Each distinct incident chunk leads once, under opposite offsets.
        assert set(p.cell_indices) == {
            ("+4.+4.+4", (0, 0, 0)),
            ("-4.-4.-4", (4, 4, 4)),
        }
        # Incidence read from either chunk finds the record.
        from_a = read_links_for_tuple(
            lg, [(0, 0, 0), (4, 4, 4)], delta=0,
        )
        assert from_a == [(((0, 0, 0), 1), ((4, 4, 4), 9))]
        # Full scan returns one copy per cell (documented duplicate behavior).
        assert len(read_links(lg, delta=0)) == 2

    def test_attributes_replicate_across_copies(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        create_link_attributes_array(lg, "weight", delta=0, sid_ndim=3)
        records = [
            [((0, 0, 0), 1), ((4, 4, 4), 9)],
            [((1, 0, 0), 2), ((2, 0, 0), 3)],
        ]
        weights = np.array([0.5, 0.7], dtype=np.float32)
        partition = write_links(
            lg, records, sid_ndim=3, delta=0, store="duplicate",
        )
        write_link_attributes(
            lg, "weight", weights, num_links=2, delta=0, partition=partition,
        )
        # Links and attributes are both physically duplicated and stay
        # row-aligned under the global read.
        links = read_links(lg, delta=0)
        attrs = read_link_attributes(lg, "weight", delta=0)
        assert len(links) == len(attrs) == 4  # 2 records × 2 cells each
        expected = {
            tuple(tuple(t) for t in rec): weights[i]
            for i, rec in enumerate(records)
        }
        for rec, w in zip(links, attrs):
            assert np.isclose(w, expected[rec])

    def test_duplicate_attributes_require_partition(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        create_link_attributes_array(lg, "weight", delta=0, sid_ndim=3)
        write_links(
            lg, [[((0, 0, 0), 1), ((4, 4, 4), 9)]], sid_ndim=3, delta=0,
            store="duplicate",
        )
        with pytest.raises(ArrayError, match="duplicate"):
            write_link_attributes(
                lg, "weight", np.array([0.5], dtype=np.float32),
                num_links=1, delta=0,   # no partition → must raise
            )


class TestOffsetsSegmentHelpers:
    """format_offsets ↔ parse_offsets round-trip at the arities used here."""

    def test_round_trip_l2(self) -> None:
        offsets = ((3, 4, 5),)
        seg = format_offsets(offsets)
        assert seg == "+3.+4.+5"
        assert parse_offsets(seg, sid_ndim=3, link_width=2) == offsets

    def test_round_trip_l3(self) -> None:
        offsets = ((1, 0, 0), (1, 1, 1))
        seg = format_offsets(offsets)
        assert seg == "+1.0.0_+1.+1.+1"
        assert parse_offsets(seg, sid_ndim=3, link_width=3) == offsets

    def test_intra_round_trip(self) -> None:
        # The all-zero offsets: what used to be a standalone links/<delta>.
        offsets = intra_offsets(3, 2)
        assert format_offsets(offsets) == "0.0.0"
        assert parse_offsets("0.0.0", sid_ndim=3, link_width=2) == offsets

    def test_arity_mismatch_raises(self) -> None:
        # One offset present but link_width=3 needs two.
        with pytest.raises(ValueError):
            parse_offsets("0.1.2", sid_ndim=3, link_width=3)
        # Four components where sid_ndim=3 is expected.
        with pytest.raises(ValueError):
            parse_offsets("0.1.2.3", sid_ndim=3, link_width=2)
