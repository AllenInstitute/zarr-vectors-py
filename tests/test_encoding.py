"""Step 02 tests: ragged encoding round-trips and compression config."""

from __future__ import annotations

import numpy as np

from zarr_vectors.encoding.ragged import (
    decode_ragged_blob,
    decode_ragged_ints,
    decode_ragged_floats,
    encode_ragged_blob,
    encode_ragged_ints,
    encode_ragged_floats,
)
from zarr_vectors.exceptions import ArrayError


# ---------------------------------------------------------------------------
# Fragment round-trips
# ---------------------------------------------------------------------------

class TestRaggedFloatEncoding:

    def test_single_group_3d(self) -> None:
        groups = [np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)]
        raw, offsets = encode_ragged_floats(groups, dtype=np.float32)
        decoded = decode_ragged_floats(raw, offsets, dtype=np.float32, ncols=3)
        assert len(decoded) == 1
        np.testing.assert_array_equal(decoded[0], groups[0])

    def test_multiple_groups_varying_size(self) -> None:
        rng = np.random.default_rng(99)
        groups = [
            rng.uniform(size=(5, 3)).astype(np.float32),
            rng.uniform(size=(100, 3)).astype(np.float32),
            rng.uniform(size=(1, 3)).astype(np.float32),
            rng.uniform(size=(42, 3)).astype(np.float32),
        ]
        raw, offsets = encode_ragged_floats(groups, dtype=np.float32)
        assert len(offsets) == 4
        decoded = decode_ragged_floats(raw, offsets, dtype=np.float32, ncols=3)
        assert len(decoded) == 4
        for orig, dec in zip(groups, decoded):
            np.testing.assert_array_equal(dec, orig)

    def test_empty_group(self) -> None:
        groups = [
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),  # empty
            np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
        ]
        raw, offsets = encode_ragged_floats(groups, dtype=np.float32)
        decoded = decode_ragged_floats(raw, offsets, dtype=np.float32, ncols=3)
        assert len(decoded) == 3
        assert decoded[1].shape == (0, 3)
        np.testing.assert_array_equal(decoded[0], groups[0])
        np.testing.assert_array_equal(decoded[2], groups[2])

    def test_empty_chunk(self) -> None:
        raw, offsets = encode_ragged_floats([], dtype=np.float32)
        assert raw == b""
        assert len(offsets) == 0
        decoded = decode_ragged_floats(raw, offsets, dtype=np.float32, ncols=3)
        assert decoded == []

    def test_single_vertex(self) -> None:
        groups = [np.array([[10.0, 20.0, 30.0]], dtype=np.float64)]
        raw, offsets = encode_ragged_floats(groups, dtype=np.float64)
        decoded = decode_ragged_floats(raw, offsets, dtype=np.float64, ncols=3)
        np.testing.assert_array_equal(decoded[0], groups[0])

    def test_1d_scalars(self) -> None:
        groups = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0], dtype=np.float32),
        ]
        raw, offsets = encode_ragged_floats(groups, dtype=np.float32)
        decoded = decode_ragged_floats(raw, offsets, dtype=np.float32, ncols=1)
        assert decoded[0].shape == (3,)
        assert decoded[1].shape == (2,)
        np.testing.assert_array_equal(decoded[0], groups[0])
        np.testing.assert_array_equal(decoded[1], groups[1])

    def test_large_group(self) -> None:
        rng = np.random.default_rng(123)
        big = rng.uniform(size=(100_000, 3)).astype(np.float32)
        groups = [big]
        raw, offsets = encode_ragged_floats(groups, dtype=np.float32)
        decoded = decode_ragged_floats(raw, offsets, dtype=np.float32, ncols=3)
        np.testing.assert_array_equal(decoded[0], big)

    def test_dtype_int32(self) -> None:
        groups = [np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)]
        raw, offsets = encode_ragged_floats(groups, dtype=np.int32)
        decoded = decode_ragged_floats(raw, offsets, dtype=np.int32, ncols=3)
        np.testing.assert_array_equal(decoded[0], groups[0])


# ---------------------------------------------------------------------------
# Ragged int round-trips
# ---------------------------------------------------------------------------

class TestRaggedIntEncoding:

    def test_edge_lists(self) -> None:
        groups = [
            np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64),
            np.array([[3, 4]], dtype=np.int64),
        ]
        raw, offsets = encode_ragged_ints(groups)
        decoded = decode_ragged_ints(raw, offsets, ncols=2)
        assert len(decoded) == 2
        np.testing.assert_array_equal(decoded[0], groups[0])
        np.testing.assert_array_equal(decoded[1], groups[1])

    def test_triangle_faces(self) -> None:
        groups = [
            np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        ]
        raw, offsets = encode_ragged_ints(groups)
        decoded = decode_ragged_ints(raw, offsets, ncols=3)
        np.testing.assert_array_equal(decoded[0], groups[0])

    def test_flat_ints(self) -> None:
        groups = [
            np.array([0, 1, 5, 12], dtype=np.int64),
            np.array([2, 3], dtype=np.int64),
        ]
        raw, offsets = encode_ragged_ints(groups)
        decoded = decode_ragged_ints(raw, offsets, ncols=1)
        np.testing.assert_array_equal(decoded[0], groups[0])
        np.testing.assert_array_equal(decoded[1], groups[1])

    def test_empty(self) -> None:
        raw, offsets = encode_ragged_ints([])
        decoded = decode_ragged_ints(raw, offsets, ncols=2)
        assert decoded == []


# ---------------------------------------------------------------------------
# Object index round-trips
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Self-describing ragged blob (inline offset header) round-trips
# ---------------------------------------------------------------------------

class TestRaggedBlob:

    def test_round_trip_ints(self) -> None:
        groups = [
            np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
            np.array([[3, 4]], dtype=np.int64),
        ]
        blob = encode_ragged_blob(groups, np.dtype(np.int64))
        decoded = decode_ragged_blob(blob, np.dtype(np.int64), ncols=2)
        assert len(decoded) == 2
        np.testing.assert_array_equal(decoded[0], groups[0])
        np.testing.assert_array_equal(decoded[1], groups[1])

    def test_empty(self) -> None:
        blob = encode_ragged_blob([], np.dtype(np.int64))
        decoded = decode_ragged_blob(blob, np.dtype(np.int64), ncols=2)
        assert decoded == []

