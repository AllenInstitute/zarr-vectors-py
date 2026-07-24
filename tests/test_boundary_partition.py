"""Unit tests for the link record partitioner.

Covers :func:`partition_records_by_offset` across the ``directed`` ×
``store`` matrix: canonical (default) placement, directed input-order
placement, and ``duplicate`` per-incident-chunk placement.  The key
correctness guarantee is that every stored cell — regardless of mode —
round-trips back to the original input record via ``parse_offsets`` +
``apply_perm_inverse`` (exactly what the reader does).

Placement is by ``(offsets_segment, source_chunk)``: the record lives in
the array cell of its source chunk, and the segment says where the other
endpoints sit relative to it.  Intra-chunk records are not a separate
family — they are simply the all-zero offsets.
"""

from __future__ import annotations

import itertools

import pytest

from zarr_vectors.core.paths import format_offsets, parse_offsets
from zarr_vectors.exceptions import ChunkingError
from zarr_vectors.spatial.boundary import (
    anchor_chunk,
    apply_perm_inverse,
    canonical_sort,
    partition_records_by_offset,
)

# A default (unscaled) store: every level shares the root chunk_shape, so
# the source anchor is the identity and offsets are plain differences.
UNSCALED = (1, 1, 1)


def _partition(records, link_width, sid_ndim, **kwargs):
    """``partition_records_by_offset`` on a default unscaled 3-D store."""
    scale = (1,) * sid_ndim
    return partition_records_by_offset(
        records, link_width, sid_ndim,
        scale_src=scale, scale_trg=scale, **kwargs,
    )


def _endpoint_chunks(seg, src_chunk, *, sid_ndim, link_width,
                     scale_src=UNSCALED, scale_trg=UNSCALED):
    """Rebuild a cell's L endpoint chunks from its segment + source.

    Mirrors ``arrays._cell_endpoint_chunks``: endpoint 0 *is* the source
    (its offset is the implicit zero), endpoint k>0 is ``anchor + o_k``.
    """
    offsets = parse_offsets(seg, sid_ndim=sid_ndim, link_width=link_width)
    anchor = anchor_chunk(src_chunk, scale_src, scale_trg)
    return (tuple(src_chunk),) + tuple(
        tuple(a + o for a, o in zip(anchor, off)) for off in offsets
    )


def _reconstruct(key, entry, *, sid_ndim, link_width,
                 scale_src=UNSCALED, scale_trg=UNSCALED):
    """Rebuild a record's input-order endpoints from one stored entry.

    Mirrors what ``read_links`` does on read.
    """
    seg, src_chunk = key
    vi_in_src, perm_idx, _input_idx = entry
    chunks = _endpoint_chunks(
        seg, src_chunk, sid_ndim=sid_ndim, link_width=link_width,
        scale_src=scale_src, scale_trg=scale_trg,
    )
    endpoints = list(zip(chunks, vi_in_src))
    return apply_perm_inverse(endpoints, perm_idx, link_width)


class TestCanonicalUndirected:
    """Default mode must match the legacy canonical_sort placement."""

    def test_matches_canonical_sort(self) -> None:
        # Cross-chunk records only: an intra-chunk record is deliberately
        # NOT canonical-sorted (identity placement preserves input order —
        # see _cell_placements), so it is covered separately below.
        records = [
            [((2, 0, 0), 8), ((0, 0, 0), 3)],
            [((0, 0, 1), 5), ((0, 0, 0), 2)],
            [((5, 0, 0), 8), ((0, 0, 1), 3), ((2, 2, 2), 7)],   # L=3 face
        ]
        for rec in records:
            link_width = len(rec)
            buckets = _partition([rec], link_width, 3)
            # One entry per input record (no duplication).
            assert sum(len(v) for v in buckets.values()) == 1
            sorted_rec, perm_idx = canonical_sort(rec)
            src = sorted_rec[0][0]
            seg = format_offsets([
                tuple(c - s for c, s in zip(cc, src))
                for cc, _vi in sorted_rec[1:]
            ])
            vi = [v for _, v in sorted_rec]
            assert (vi, perm_idx) in [(e[0], e[1]) for e in buckets[(seg, src)]]

    def test_intra_chunk_not_canonical_sorted(self) -> None:
        # canonical_sort would swap these on the vi tie-break; the
        # partitioner must not — intra offsets are all-zero, so there is
        # nothing to canonicalise and input order is data.
        rec = [((1, 1, 1), 5), ((1, 1, 1), 2)]
        sorted_rec, perm_idx = canonical_sort(rec)
        assert [v for _, v in sorted_rec] == [2, 5] and perm_idx == 1
        buckets = _partition([rec], 2, 3)
        assert buckets == {("0.0.0", (1, 1, 1)): [([5, 2], 0, 0)]}

    def test_intra_chunk_record_is_all_zero_offsets(self) -> None:
        # Both endpoints in one chunk → the intra array, identity placement.
        buckets = _partition([[((1, 1, 1), 5), ((1, 1, 1), 2)]], 2, 3)
        assert set(buckets) == {("0.0.0", (1, 1, 1))}
        (entry,), = buckets.values()
        assert entry == ([5, 2], 0, 0)   # input order preserved, perm identity

    def test_cross_chunk_record_is_positive_offset(self) -> None:
        # A canonical family stores each undirected record once, under the
        # lexicographically-positive offset — whichever way it is passed.
        fwd = _partition([[((0, 0, 0), 1), ((0, 0, 1), 2)]], 2, 3)
        rev = _partition([[((0, 0, 1), 2), ((0, 0, 0), 1)]], 2, 3)
        assert set(fwd) == set(rev) == {("0.0.+1", (0, 0, 0))}

    def test_round_trips(self) -> None:
        records = [
            [((3, 0, 0), 8), ((0, 2, 0), 3)],
            [((0, 0, 5), 1), ((9, 9, 9), 4)],
        ]
        buckets = _partition(records, 2, 3)
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
        buckets = _partition([rec], 2, 3, directed=True)
        assert len(buckets) == 1
        (key, entries), = buckets.items()
        # Source is input endpoint 0; the other endpoint is 2 chunks back.
        assert key == ("-2.0.0", (2, 0, 0))
        assert entries[0][1] == 0             # perm_idx identity
        assert entries[0][0] == [8, 3]

    def test_both_orientations_distinct_cells(self) -> None:
        fwd = [((0, 0, 0), 1), ((1, 0, 0), 2)]
        rev = [((1, 0, 0), 2), ((0, 0, 0), 1)]
        buckets = _partition([fwd, rev], 2, 3, directed=True)
        # Opposite offsets at different cells — the offset sign is what
        # carries the direction.
        assert set(buckets) == {
            ("+1.0.0", (0, 0, 0)),
            ("-1.0.0", (1, 0, 0)),
        }

    def test_directed_intra_chunk_still_all_zero(self) -> None:
        # Direction within one chunk is carried by the row's vi order.
        buckets = _partition(
            [[((1, 1, 1), 9), ((1, 1, 1), 4)]], 2, 3, directed=True,
        )
        assert set(buckets) == {("0.0.0", (1, 1, 1))}

    def test_round_trips(self) -> None:
        records = [
            [((5, 0, 0), 8), ((0, 0, 1), 3), ((2, 2, 2), 7)],  # L=3 face
        ]
        buckets = _partition(records, 3, 3, directed=True)
        for key, entries in buckets.items():
            for entry in entries:
                got = _reconstruct(key, entry, sid_ndim=3, link_width=3)
                assert got == [(tuple(c), v) for c, v in records[entry[2]]]


class TestDuplicate:
    """store='duplicate' emits one cell per distinct incident chunk."""

    def test_l2_both_orderings(self) -> None:
        rec = [((0, 0, 0), 1), ((4, 4, 4), 9)]
        buckets = _partition([rec], 2, 3, store="duplicate")
        # Two distinct chunks → each leads once, under opposite offsets.
        assert set(buckets) == {
            ("+4.+4.+4", (0, 0, 0)),
            ("-4.-4.-4", (4, 4, 4)),
        }
        # Every copy carries the same input index and round-trips.
        for key, entries in buckets.items():
            assert entries[0][2] == 0
            got = _reconstruct(key, entries[0], sid_ndim=3, link_width=2)
            assert got == [((0, 0, 0), 1), ((4, 4, 4), 9)]

    def test_l2_order_independent(self) -> None:
        # Same undirected edge given both ways → identical cell set.
        a = _partition(
            [[((0, 0, 0), 1), ((4, 4, 4), 9)]], 2, 3, store="duplicate",
        )
        b = _partition(
            [[((4, 4, 4), 9), ((0, 0, 0), 1)]], 2, 3, store="duplicate",
        )
        assert set(a) == set(b)

    def test_triangle_three_distinct_chunks(self) -> None:
        rec = [((0, 0, 0), 1), ((1, 0, 0), 2), ((2, 0, 0), 3)]
        buckets = _partition([rec], 3, 3, store="duplicate")
        # 3 distinct chunks → 3 cells, each sourced at a distinct chunk.
        assert len(buckets) == 3
        assert {src for _seg, src in buckets} == {
            (0, 0, 0), (1, 0, 0), (2, 0, 0),
        }

    def test_repeated_chunk_dedups_leads(self) -> None:
        # [A, A, B] touches 2 distinct chunks → 2 cells, not 3.
        rec = [((0, 0, 0), 1), ((0, 0, 0), 2), ((5, 0, 0), 3)]
        buckets = _partition([rec], 3, 3, store="duplicate")
        assert len(buckets) == 2
        assert {src for _seg, src in buckets} == {(0, 0, 0), (5, 0, 0)}

    def test_intra_chunk_record_not_duplicated(self) -> None:
        # One distinct chunk → one cell even under duplicate.
        rec = [((2, 2, 2), 1), ((2, 2, 2), 2)]
        buckets = _partition([rec], 2, 3, store="duplicate")
        assert set(buckets) == {("0.0.0", (2, 2, 2))}

    def test_all_copies_round_trip(self) -> None:
        cases = [
            (3, [((0, 0, 0), 1), ((1, 0, 0), 2), ((2, 0, 0), 3)]),
            (2, [((3, 0, 0), 4), ((0, 0, 0), 5)]),
        ]
        for lw, rec in cases:
            b = _partition([rec], lw, 3, store="duplicate")
            for key, entries in b.items():
                for entry in entries:
                    got = _reconstruct(key, entry, sid_ndim=3, link_width=lw)
                    assert got == [(tuple(c), v) for c, v in rec]


class TestDirectedDuplicate:
    """directed + duplicate: independent copies, each recovers input order."""

    def test_copies_recover_input_direction(self) -> None:
        rec = [((1, 0, 0), 7), ((0, 0, 0), 4)]   # direction 1→0
        buckets = _partition(
            [rec], 2, 3, directed=True, store="duplicate",
        )
        assert set(buckets) == {
            ("-1.0.0", (1, 0, 0)),
            ("+1.0.0", (0, 0, 0)),
        }
        for key, entries in buckets.items():
            got = _reconstruct(key, entries[0], sid_ndim=3, link_width=2)
            assert got == [((1, 0, 0), 7), ((0, 0, 0), 4)]


class TestCrossLevel:
    """cross_level=True forces the source to input endpoint 0."""

    def test_source_is_endpoint_zero(self) -> None:
        # Canonical would lead with (0,0,0); cross-level must not, so the
        # anchor's scale_src stays the owning level's.
        rec = [((2, 0, 0), 8), ((0, 0, 0), 3)]
        buckets = _partition([rec], 2, 3, cross_level=True)
        assert set(buckets) == {("-2.0.0", (2, 0, 0))}
        (entries,), = [list(v) for v in buckets.values()],
        assert entries[0][1] == 0          # identity placement, no perm

    def test_anchor_rescales_source(self) -> None:
        # A coarser target level whose chunks are 2× the source's: chunk
        # (4,0,0) at the source anchors to (2,0,0) in the target grid, so
        # a target endpoint at (3,0,0) is offset +1, not -1.
        rec = [((4, 0, 0), 1), ((3, 0, 0), 2)]
        buckets = partition_records_by_offset(
            [rec], 2, 3,
            scale_src=(1, 1, 1), scale_trg=(2, 2, 2), cross_level=True,
        )
        assert set(buckets) == {("+1.0.0", (4, 0, 0))}
        # And it round-trips through the same anchor arithmetic.
        (key, entries), = buckets.items()
        got = _reconstruct(
            key, entries[0], sid_ndim=3, link_width=2,
            scale_src=(1, 1, 1), scale_trg=(2, 2, 2),
        )
        assert got == [((4, 0, 0), 1), ((3, 0, 0), 2)]


class TestValidation:
    def test_arity_mismatch_record(self) -> None:
        with pytest.raises(ChunkingError):
            _partition([[((0, 0, 0), 1)]], 2, 3)

    def test_arity_mismatch_chunk(self) -> None:
        with pytest.raises(ChunkingError):
            _partition([[((0, 0), 1), ((1, 1), 2)]], 2, 3)

    def test_unknown_store(self) -> None:
        with pytest.raises(ChunkingError):
            _partition(
                [[((0, 0, 0), 1), ((1, 0, 0), 2)]], 2, 3, store="bogus",
            )

    def test_scale_rank_mismatch(self) -> None:
        # scale_src/scale_trg must have sid_ndim components — a wrong rank
        # would silently mis-anchor every record.
        with pytest.raises(ChunkingError, match="scale_src/scale_trg rank"):
            partition_records_by_offset(
                [[((0, 0, 0), 1), ((1, 0, 0), 2)]], 2, 3,
                scale_src=(1, 1), scale_trg=(1, 1, 1),
            )


class TestExhaustiveRoundTrip:
    """Every permutation of a record, in every mode, must round-trip."""

    @pytest.mark.parametrize("L", [1, 2, 3, 4])
    @pytest.mark.parametrize("directed", [False, True])
    @pytest.mark.parametrize("store", ["canonical", "duplicate"])
    def test_matrix(self, L: int, directed: bool, store: str) -> None:
        for p in itertools.permutations(range(L)):
            rec = [((i, 0, 0), 100 + i) for i in p]
            buckets = _partition(
                [rec], L, 3, directed=directed, store=store,
            )
            assert buckets, (L, directed, store, p)
            for key, entries in buckets.items():
                for entry in entries:
                    got = _reconstruct(key, entry, sid_ndim=3, link_width=L)
                    assert got == [(tuple(c), v) for c, v in rec], (
                        L, directed, store, p, key, entry,
                    )
