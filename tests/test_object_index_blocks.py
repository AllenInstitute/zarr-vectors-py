"""Tests for the object-index manifest block encoder/decoder."""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.encoding.fragments import (
    MANIFEST_MODE_EXPLICIT,
    MANIFEST_MODE_RANGE,
    MANIFEST_MODE_SINGLE,
    decode_object_manifest_blocks,
    encode_object_manifest_blocks,
)
from zarr_vectors.exceptions import ArrayError

# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_empty_manifest_is_four_bytes() -> None:
    raw = encode_object_manifest_blocks([], sid_ndim=3)
    assert raw == b"\x00\x00\x00\x00"
    blocks = decode_object_manifest_blocks(raw, sid_ndim=3)
    assert blocks == []


def test_single_block_single_mode() -> None:
    raw = encode_object_manifest_blocks([((1, 2, 3), 7)], sid_ndim=3)
    blocks = decode_object_manifest_blocks(raw, sid_ndim=3)
    assert blocks == [((1, 2, 3), 7)]


def test_single_block_range_mode() -> None:
    raw = encode_object_manifest_blocks(
        [((1, 2, 3), (4, 4))], sid_ndim=3,
    )
    blocks = decode_object_manifest_blocks(raw, sid_ndim=3)
    assert blocks == [((1, 2, 3), (4, 4))]


def test_single_block_explicit_mode() -> None:
    raw = encode_object_manifest_blocks(
        [((1, 2, 3), np.array([2, 7]))], sid_ndim=3,
    )
    blocks = decode_object_manifest_blocks(raw, sid_ndim=3)
    assert len(blocks) == 1
    coords, frag_ref = blocks[0]
    assert coords == (1, 2, 3)
    assert isinstance(frag_ref, np.ndarray)
    assert frag_ref.tolist() == [2, 7]


def test_multi_chunk_manifest_walkthrough() -> None:
    """Mirrors the example from plan §4: object 42 spread across 3 chunks."""
    raw = encode_object_manifest_blocks(
        [
            ((0, 0, 0), np.arange(3, 6)),  # auto-detected range (3, 3)
            ((1, 0, 0), 0),                # single
            ((1, 0, 1), np.array([2, 7])), # explicit
        ],
        sid_ndim=3,
    )
    blocks = decode_object_manifest_blocks(raw, sid_ndim=3)
    assert blocks[0] == ((0, 0, 0), (3, 3))   # auto-promoted to range
    assert blocks[1] == ((1, 0, 0), 0)
    assert blocks[2][0] == (1, 0, 1)
    assert isinstance(blocks[2][1], np.ndarray)
    assert blocks[2][1].tolist() == [2, 7]


def test_force_explicit_overrides_auto_range() -> None:
    raw = encode_object_manifest_blocks(
        [((0, 0, 0), np.arange(3, 8))], sid_ndim=3, force_explicit=True,
    )
    blocks = decode_object_manifest_blocks(raw, sid_ndim=3)
    assert isinstance(blocks[0][1], np.ndarray)
    assert blocks[0][1].tolist() == [3, 4, 5, 6, 7]


def test_sid_ndim_2d_round_trip() -> None:
    raw = encode_object_manifest_blocks(
        [((0, 0), (3, 3)), ((1, 1), 0)], sid_ndim=2,
    )
    blocks = decode_object_manifest_blocks(raw, sid_ndim=2)
    assert blocks == [((0, 0), (3, 3)), ((1, 1), 0)]


# ---------------------------------------------------------------------------
# Range short-circuit really saves bytes
# ---------------------------------------------------------------------------


def test_range_mode_smaller_than_repeated_single_mode() -> None:
    """An object owning fragments 0..99 in chunk (0,0,0) should encode
    much smaller as one range block than as 100 single-mode blocks.
    Mirrors the example from plan §4."""
    chunk = (0, 0, 0)
    range_blob = encode_object_manifest_blocks(
        [(chunk, (0, 100))], sid_ndim=3,
    )
    singles_blob = encode_object_manifest_blocks(
        [(chunk, i) for i in range(100)], sid_ndim=3,
    )
    assert len(range_blob) < len(singles_blob)
    # And quantitatively: range is one block (4 + 24 coords + 1 mode + 16 payload = 45 bytes);
    # singles is 100 blocks (4 + 100*(24 + 1 + 8) = 4 + 3300 = 3304 bytes).
    assert len(range_blob) == 45
    assert len(singles_blob) == 4 + 100 * (24 + 1 + 8)


# ---------------------------------------------------------------------------
# Fragment re-use across two objects is implicit (no on-disk cross-ref)
# ---------------------------------------------------------------------------


def test_two_objects_can_reference_same_chunk_local_fragment() -> None:
    """Re-use is implicit: two manifests both pointing at fragment 5 in
    chunk (1,2,3) are valid and decode independently."""
    obj_a = encode_object_manifest_blocks([((1, 2, 3), 5)], sid_ndim=3)
    obj_b = encode_object_manifest_blocks([((1, 2, 3), 5)], sid_ndim=3)
    blocks_a = decode_object_manifest_blocks(obj_a, sid_ndim=3)
    blocks_b = decode_object_manifest_blocks(obj_b, sid_ndim=3)
    assert blocks_a == blocks_b == [((1, 2, 3), 5)]


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_wrong_chunk_coord_rank_rejected() -> None:
    with pytest.raises(ArrayError, match="sid_ndim=3"):
        encode_object_manifest_blocks([((1, 2), 0)], sid_ndim=3)


def test_negative_single_fragment_index_rejected() -> None:
    with pytest.raises(ArrayError, match=">= 0"):
        encode_object_manifest_blocks([((0, 0, 0), -1)], sid_ndim=3)


def test_negative_range_count_rejected() -> None:
    with pytest.raises(ArrayError, match=">= 0"):
        encode_object_manifest_blocks([((0, 0, 0), (5, -1))], sid_ndim=3)


def test_negative_explicit_index_rejected() -> None:
    with pytest.raises(ArrayError, match="non-negative"):
        encode_object_manifest_blocks(
            [((0, 0, 0), np.array([0, -1, 2]))], sid_ndim=3,
        )


def test_truncated_manifest_rejected() -> None:
    raw = encode_object_manifest_blocks(
        [((1, 2, 3), 7), ((4, 5, 6), (0, 3))], sid_ndim=3,
    )
    with pytest.raises(ArrayError, match="truncated"):
        decode_object_manifest_blocks(raw[:-4], sid_ndim=3)


def test_trailing_bytes_rejected() -> None:
    raw = encode_object_manifest_blocks([((1, 2, 3), 7)], sid_ndim=3)
    with pytest.raises(ArrayError, match="trailing bytes"):
        decode_object_manifest_blocks(raw + b"\x00\x00", sid_ndim=3)


def test_too_short_blob_rejected() -> None:
    with pytest.raises(ArrayError, match="too short"):
        decode_object_manifest_blocks(b"\x00\x00", sid_ndim=3)


def test_mode_constants_match_spec() -> None:
    """Lock down the wire format mode tags."""
    assert MANIFEST_MODE_SINGLE == 0
    assert MANIFEST_MODE_RANGE == 1
    assert MANIFEST_MODE_EXPLICIT == 2
