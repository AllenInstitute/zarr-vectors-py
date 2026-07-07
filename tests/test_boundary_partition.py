"""Unit tests for the cross-chunk record partitioner.

Covers :func:`partition_cross_records_by_tuple` across the
``directed`` × ``store`` matrix: canonical (default) placement,
directed input-order placement, and ``duplicate`` per-incident-chunk
placement.  The key correctness guarantee is that every stored cell —
regardless of mode — round-trips back to the original input record via
``parse_cell_key`` + ``apply_perm_inverse`` (exactly what the reader
does).
"""

from __future__ import annotations

import itertools

import pytest

from zarr_vectors.core.paths import parse_cell_key
from zarr_vectors.exceptions import ChunkingError
from zarr_vectors.spatial.boundary import (
    apply_perm_inverse,
    canonical_sort,
    partition_cross_records_by_tuple,
)


def _reconstruct(cell_key, entry, *, sid_ndim, link_width):
    """Rebuild a record's input-order endpoints from one stored cell entry.

    Mirrors what ``read_cross_chunk_links`` does on read.
    """
    vi_in_cell, perm_idx, _input_idx = entry
    chunks = parse_cell_key(cell_key, sid_ndim=sid_ndim, link_width=link_width)
    endpoints = list(zip(chunks, vi_in_cell))
    return apply_perm_inverse(endpoints, perm_idx, link_width)


class TestCanonicalUndirected:
    """Default mode must match the legacy canonical_sort placement."""

    def test_matches_canonical_sort(self) -> None:
        records = [
            [((2, 0, 0), 8), ((0, 0, 0), 3)],
            [((1, 1, 1), 5), ((1, 1, 1), 2)],  # same chunk, tie-break on vi
        ]
        buckets = partition_cross_records_by_tuple(records, 2, 3)
        # One entry per input record (no duplication).
        assert sum(len(v) for v in buckets.values()) == len(records)
        for rec in records:
            sorted_rec, perm_idx = canonical_sort(rec)
            from zarr_vectors.core.paths import format_cell_key
            key = format_cell_key([c for c, _ in sorted_rec])
            vi = [v for _, v in sorted_rec]
            assert (vi, perm_idx) in [(e[0], e[1]) for e in buckets[key]]

    def test_round_trips(self) -> None:
        records = [
            [((3, 0, 0), 8), ((0, 2, 0), 3)],
            [((0, 0, 5), 1), ((9, 9, 9), 4)],
        ]
        buckets = partition_cross_records_by_tuple(records, 2, 3)
        for key, entries in buckets.items():
            for entry in entries:
                rec = records[entry[2]]
                got = _reconstruct(key, entry, sid_ndim=3, link_width=2)
                assert got == [(tuple(c), v) for c, v in rec]


class TestDirected:
    """directed=True keeps input order; A→B and B→A are distinct cells."""

    def test_input_order_preserved_perm_zero(self) -> None:
        # canonical would swap this (chunk (2,) > (0,)), directed must not.
        rec = [((2, 0, 0), 8), ((0, 0, 0), 3)]
        buckets = partition_cross_records_by_tuple(
            [rec], 2, 3, directed=True,
        )
        assert len(buckets) == 1
        (key, entries), = buckets.items()
        assert key == "2.0.0.0.0.0"          # input order, not canonical
        assert entries[0][1] == 0             # perm_idx identity
        assert entries[0][0] == [8, 3]

    def test_both_orientations_distinct_cells(self) -> None:
        fwd = [((0, 0, 0), 1), ((1, 0, 0), 2)]
        rev = [((1, 0, 0), 2), ((0, 0, 0), 1)]
        buckets = partition_cross_records_by_tuple(
            [fwd, rev], 2, 3, directed=True,
        )
        assert set(buckets) == {"0.0.0.1.0.0", "1.0.0.0.0.0"}

    def test_round_trips(self) -> None:
        records = [
            [((5, 0, 0), 8), ((0, 0, 1), 3), ((2, 2, 2), 7)],  # L=3 face
        ]
        buckets = partition_cross_records_by_tuple(
            records, 3, 3, directed=True,
        )
        for key, entries in buckets.items():
            for entry in entries:
                got = _reconstruct(key, entry, sid_ndim=3, link_width=3)
                assert got == [(tuple(c), v) for c, v in records[entry[2]]]


class TestDuplicate:
    """store='duplicate' emits one cell per distinct incident chunk."""

    def test_l2_both_orderings(self) -> None:
        rec = [((0, 0, 0), 1), ((4, 4, 4), 9)]
        buckets = partition_cross_records_by_tuple(
            [rec], 2, 3, store="duplicate",
        )
        # Two distinct chunks → both orderings present.
        assert set(buckets) == {"0.0.0.4.4.4", "4.4.4.0.0.0"}
        # Every copy carries the same input index and round-trips.
        for key, entries in buckets.items():
            assert entries[0][2] == 0
            got = _reconstruct(key, entries[0], sid_ndim=3, link_width=2)
            assert got == [((0, 0, 0), 1), ((4, 4, 4), 9)]

    def test_l2_order_independent(self) -> None:
        # Same undirected edge given both ways → identical cell set.
        a = partition_cross_records_by_tuple(
            [[((0, 0, 0), 1), ((4, 4, 4), 9)]], 2, 3, store="duplicate",
        )
        b = partition_cross_records_by_tuple(
            [[((4, 4, 4), 9), ((0, 0, 0), 1)]], 2, 3, store="duplicate",
        )
        assert set(a) == set(b)

    def test_triangle_three_distinct_chunks(self) -> None:
        rec = [((0, 0, 0), 1), ((1, 0, 0), 2), ((2, 0, 0), 3)]
        buckets = partition_cross_records_by_tuple(
            [rec], 3, 3, store="duplicate",
        )
        # 3 distinct chunks → 3 cells, each leading with a distinct chunk.
        assert len(buckets) == 3
        # Each cell's leading chunk (first sid_ndim components) is distinct.
        leads = {tuple(key.split(".")[:3]) for key in buckets}
        assert leads == {("0", "0", "0"), ("1", "0", "0"), ("2", "0", "0")}

    def test_repeated_chunk_dedups_leads(self) -> None:
        # [A, A, B] touches 2 distinct chunks → 2 cells, not 3.
        rec = [((0, 0, 0), 1), ((0, 0, 0), 2), ((5, 0, 0), 3)]
        buckets = partition_cross_records_by_tuple(
            [rec], 3, 3, store="duplicate",
        )
        assert len(buckets) == 2

    def test_all_copies_round_trip(self) -> None:
        cases = [
            (3, [((0, 0, 0), 1), ((1, 0, 0), 2), ((2, 0, 0), 3)]),
            (2, [((3, 0, 0), 4), ((0, 0, 0), 5)]),
        ]
        for lw, rec in cases:
            b = partition_cross_records_by_tuple([rec], lw, 3, store="duplicate")
            for key, entries in b.items():
                for entry in entries:
                    got = _reconstruct(key, entry, sid_ndim=3, link_width=lw)
                    assert got == [(tuple(c), v) for c, v in rec]


class TestDirectedDuplicate:
    """directed + duplicate: independent copies, each recovers input order."""

    def test_copies_recover_input_direction(self) -> None:
        rec = [((1, 0, 0), 7), ((0, 0, 0), 4)]   # direction 1→0
        buckets = partition_cross_records_by_tuple(
            [rec], 2, 3, directed=True, store="duplicate",
        )
        assert set(buckets) == {"1.0.0.0.0.0", "0.0.0.1.0.0"}
        for key, entries in buckets.items():
            got = _reconstruct(key, entries[0], sid_ndim=3, link_width=2)
            assert got == [((1, 0, 0), 7), ((0, 0, 0), 4)]


class TestValidation:
    def test_arity_mismatch_record(self) -> None:
        with pytest.raises(ChunkingError):
            partition_cross_records_by_tuple(
                [[((0, 0, 0), 1)]], 2, 3,
            )

    def test_arity_mismatch_chunk(self) -> None:
        with pytest.raises(ChunkingError):
            partition_cross_records_by_tuple(
                [[((0, 0), 1), ((1, 1), 2)]], 2, 3,
            )

    def test_unknown_store(self) -> None:
        with pytest.raises(ChunkingError):
            partition_cross_records_by_tuple(
                [[((0, 0, 0), 1), ((1, 0, 0), 2)]], 2, 3, store="bogus",
            )


class TestExhaustiveRoundTrip:
    """Every permutation of a record, in every mode, must round-trip."""

    @pytest.mark.parametrize("L", [1, 2, 3, 4])
    @pytest.mark.parametrize("directed", [False, True])
    @pytest.mark.parametrize("store", ["canonical", "duplicate"])
    def test_matrix(self, L: int, directed: bool, store: str) -> None:
        for p in itertools.permutations(range(L)):
            rec = [((i, 0, 0), 100 + i) for i in p]
            buckets = partition_cross_records_by_tuple(
                [rec], L, 3, directed=directed, store=store,
            )
            assert buckets, (L, directed, store, p)
            for key, entries in buckets.items():
                for entry in entries:
                    got = _reconstruct(key, entry, sid_ndim=3, link_width=L)
                    assert got == [(tuple(c), v) for c, v in rec], (
                        L, directed, store, p, key, entry,
                    )
