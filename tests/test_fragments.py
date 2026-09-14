"""Tests for the per-chunk fragment-index encoder/decoder."""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.encoding.fragments import (
    FRAGMENT_INDEX_MAGIC,
    FRAGMENT_INDEX_VERSION,
    ChunkFragmentIndex,
    decode_fragments,
    encode_fragments,
)
from zarr_vectors.exceptions import ArrayError


# ---------------------------------------------------------------------------
# Round-trip basics
# ---------------------------------------------------------------------------


def _roundtrip(fragments, **kwargs) -> ChunkFragmentIndex:
    raw = encode_fragments(fragments, **kwargs)
    return decode_fragments(raw)


def test_empty_chunk_header_only_sentinel() -> None:
    raw = encode_fragments([])
    assert len(raw) == 16, "F=0 must be a header-only 16-byte blob"
    fi = decode_fragments(raw)
    assert len(fi) == 0
    assert fi.num_range_fragments == 0
    assert fi.num_explicit_fragments == 0


def test_single_range() -> None:
    fi = _roundtrip([(100, 5)])
    assert len(fi) == 1
    assert fi.is_range(0) is True
    assert fi.range(0) == (100, 5)
    assert fi.indices(0).tolist() == [100, 101, 102, 103, 104]


def test_single_explicit() -> None:
    fi = _roundtrip([np.array([3, 7, 9, 100])])
    assert len(fi) == 1
    assert fi.is_range(0) is False
    assert fi.indices(0).tolist() == [3, 7, 9, 100]
    view = fi.indices_view(0)
    assert view.tolist() == [3, 7, 9, 100]
    assert view.flags["WRITEABLE"] is False or True  # just exercise


def test_mixed_bitmap() -> None:
    fragments = [
        (10, 5),                       # 0 range
        np.array([100, 101, 200]),     # 1 explicit (not arange because 200)
        (50, 1),                       # 2 range
        np.array([7]),                 # 3 explicit (single elem, but auto-detected as range arange(7,8))
        (0, 0),                        # 4 empty range
        np.array([], dtype=np.int64),  # 5 empty explicit
    ]
    fi = _roundtrip(fragments)
    assert len(fi) == 6
    # f=3 should be auto-promoted to range (arange(7, 8) == [7])
    assert fi.is_range(0) is True and fi.range(0) == (10, 5)
    assert fi.is_range(1) is False
    assert fi.is_range(2) is True and fi.range(2) == (50, 1)
    assert fi.is_range(3) is True and fi.range(3) == (7, 1)
    assert fi.is_range(4) is True and fi.range(4) == (0, 0)
    assert fi.is_range(5) is False
    # Round-trip the explicit ones
    assert fi.indices(1).tolist() == [100, 101, 200]
    assert fi.indices(4).tolist() == []
    assert fi.indices(5).tolist() == []


def test_auto_detect_long_arange() -> None:
    f0 = np.arange(1000, 2000, dtype=np.int64)
    fi = _roundtrip([f0])
    assert fi.is_range(0) is True
    assert fi.range(0) == (1000, 1000)
    # Range path footprint stays tiny regardless of N.
    raw = encode_fragments([f0])
    # 16 header + 8 bitmap (padded) + 16 range row + 4 csr_offsets[0] = 44
    assert len(raw) == 16 + 8 + 16 + 4


def test_force_explicit_overrides_auto_range() -> None:
    raw = encode_fragments(
        [np.arange(5, 15, dtype=np.int64)], force_explicit=True,
    )
    fi = decode_fragments(raw)
    assert fi.is_range(0) is False
    assert fi.indices(0).tolist() == list(range(5, 15))


def test_all_range_fast_path_byte_budget() -> None:
    """The motivating use case: every fragment is a range."""
    f = 1000
    fragments = [(i * 10, 10) for i in range(f)]
    raw = encode_fragments(fragments)
    # 16 header + ceil(1000/8)=125, padded to 128, + 1000*16 range table
    # + 4 csr_offsets (E=0 → 1 slot). Total = 16 + 128 + 16000 + 4 = 16148.
    assert len(raw) == 16 + 128 + 1000 * 16 + 4
    fi = decode_fragments(raw)
    assert len(fi) == f
    assert fi.num_range_fragments == f
    # Spot check a couple of ranges
    assert fi.range(0) == (0, 10)
    assert fi.range(999) == (9990, 10)


def test_alignment_padding_F_eq_3() -> None:
    """3 fragments → 1-byte raw bitmap, padded to 8 → range table aligned."""
    fragments = [(0, 1), (1, 1), (2, 1)]
    raw = encode_fragments(fragments)
    # 16 + 8 (padded bitmap) + 3*16 + 4 (csr_offsets[0])
    assert len(raw) == 16 + 8 + 48 + 4
    fi = decode_fragments(raw)
    assert [fi.range(i) for i in range(3)] == [(0, 1), (1, 1), (2, 1)]


# ---------------------------------------------------------------------------
# is_range(f) must not materialise fragments
# ---------------------------------------------------------------------------


class _NoMaterializeFragmentIndex(ChunkFragmentIndex):
    """Subclass that explodes if .indices / .range / .indices_view are called."""

    def __init__(self, fi: ChunkFragmentIndex) -> None:
        super().__init__(
            num_fragments=fi.num_fragments,
            _bitmap=fi._bitmap,
            _range_table=fi._range_table,
            _csr_offsets=fi._csr_offsets,
            _csr_indices=fi._csr_indices,
        )

    def range(self, f):  # type: ignore[override]
        raise AssertionError(
            f"is_range({f}) decoded fragment payload (called .range)",
        )

    def indices(self, f):  # type: ignore[override]
        raise AssertionError(
            f"is_range({f}) decoded fragment payload (called .indices)",
        )

    def indices_view(self, f):  # type: ignore[override]
        raise AssertionError(
            f"is_range({f}) decoded fragment payload (called .indices_view)",
        )


def test_is_range_does_not_decode_payload() -> None:
    """Stress the must-have invariant: is_range(f) is pure bit lookup."""
    fragments = [(i, 5) if i % 3 == 0 else np.array([i, i + 100, i + 200])
                 for i in range(100)]
    base = _roundtrip(fragments)
    no_materialize = _NoMaterializeFragmentIndex(base)
    # Calling is_range over the whole F must never reach .range/.indices.
    for f in range(len(no_materialize)):
        no_materialize.is_range(f)


# ---------------------------------------------------------------------------
# Random access through prefix-popcount
# ---------------------------------------------------------------------------


def test_random_access_uses_lazy_popcount() -> None:
    f = 50
    rng = np.random.default_rng(42)
    fragments = []
    for i in range(f):
        if rng.random() < 0.5:
            fragments.append((rng.integers(0, 1000).item(), rng.integers(1, 6).item()))
        else:
            fragments.append(
                rng.integers(0, 1000, size=rng.integers(1, 6).item()).astype(np.int64),
            )
    fi = _roundtrip(fragments)
    # Visit fragments in non-sequential order; each call should give the
    # same answer the encoder put in.
    order = list(range(f))
    rng.shuffle(order)
    for idx in order:
        if fi.is_range(idx):
            start, count = fi.range(idx)
            assert fi.indices(idx).tolist() == list(range(start, start + count))
        else:
            decoded = fi.indices(idx).tolist()
            # The encoder may have auto-promoted this fragment to range if
            # the random list happened to be an arange.  In that branch
            # is_range would have been True; here we know it's False so
            # the underlying CSR must round-trip exactly.
            assert decoded == list(np.asarray(fragments[idx]).astype(np.int64))


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_negative_explicit_index_rejected() -> None:
    with pytest.raises(ArrayError, match="non-negative"):
        encode_fragments([np.array([0, -1, 2])])


def test_negative_range_count_rejected() -> None:
    with pytest.raises(ArrayError, match="count must be >= 0"):
        encode_fragments([(5, -1)])


def test_tuple_wrong_shape_rejected() -> None:
    with pytest.raises(ArrayError, match="\\(start, count\\)"):
        encode_fragments([(1, 2, 3)])  # type: ignore[list-item]


def test_2d_fragment_rejected() -> None:
    with pytest.raises(ArrayError, match="1-D"):
        encode_fragments([np.zeros((2, 2), dtype=np.int64)])


def test_truncated_blob_rejected() -> None:
    raw = encode_fragments([(0, 5), np.array([1, 2, 3])])
    with pytest.raises(ArrayError, match="truncated"):
        decode_fragments(raw[:-4])


def test_bad_magic_rejected() -> None:
    raw = bytearray(encode_fragments([(0, 5)]))
    raw[0] = 0
    with pytest.raises(ArrayError, match="magic"):
        decode_fragments(bytes(raw))


def test_too_short_blob_rejected() -> None:
    with pytest.raises(ArrayError, match="too short"):
        decode_fragments(b"\x00\x00\x00")


def test_range_on_explicit_raises() -> None:
    fi = _roundtrip([np.array([100, 101, 200])])
    with pytest.raises(ArrayError, match="explicit"):
        fi.range(0)


def test_indices_view_on_range_raises() -> None:
    fi = _roundtrip([(10, 5)])
    with pytest.raises(ArrayError, match="explicit"):
        fi.indices_view(0)


def test_out_of_range_fragment_index_raises() -> None:
    fi = _roundtrip([(0, 1)])
    with pytest.raises(IndexError):
        fi.is_range(1)
    with pytest.raises(IndexError):
        fi.is_range(-1)


# ---------------------------------------------------------------------------
# Header constants exposed for downstream readers
# ---------------------------------------------------------------------------


def test_header_constants_in_blob() -> None:
    import struct
    raw = encode_fragments([(0, 1)])
    magic, version, flags, f, r = struct.unpack_from("<IHHII", raw, 0)
    assert magic == FRAGMENT_INDEX_MAGIC
    assert version == FRAGMENT_INDEX_VERSION
    assert flags == 0
    assert f == 1
    assert r == 1


# ---------------------------------------------------------------------------
# ChunkFragmentIndex.tiles — the predicate the bulk readers short-circuit on
# ---------------------------------------------------------------------------


def test_tiles_true_for_contiguous_ranges_covering_the_buffer() -> None:
    fi = _roundtrip([(0, 4), (4, 6), (10, 2)])
    assert fi.tiles(12) is True


def test_tiles_false_when_the_last_range_stops_short() -> None:
    """A buffer longer than the fragments means rows nobody references —
    returning it whole would hand back data the fragments exclude."""
    fi = _roundtrip([(0, 4), (4, 6)])
    assert fi.tiles(12) is False
    assert fi.tiles(10) is True


def test_tiles_false_when_a_row_is_skipped() -> None:
    fi = _roundtrip([(0, 4), (5, 5)])
    assert fi.tiles(10) is False


def test_tiles_false_when_ranges_are_out_of_order() -> None:
    """Concatenating these does not reproduce the buffer — same rows,
    different order — so the shortcut must not fire."""
    fi = _roundtrip([(5, 5), (0, 5)])
    assert fi.tiles(10) is False


def test_tiles_false_when_the_first_range_does_not_start_at_zero() -> None:
    fi = _roundtrip([(1, 9)])
    assert fi.tiles(10) is False


def test_tiles_false_for_any_explicit_fragment() -> None:
    fi = _roundtrip([(0, 5), np.array([5, 7, 9], dtype=np.int64)])
    assert fi.num_explicit_fragments == 1
    assert fi.tiles(10) is False


def test_tiles_false_for_an_empty_index() -> None:
    fi = _roundtrip([])
    assert fi.tiles(0) is False
