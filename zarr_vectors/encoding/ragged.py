"""Ragged array encoding and decoding for ZV spatial chunks.

Each spatial chunk in a ZV store holds a variable number of fragments,
each of variable length.  This module serialises lists of numpy arrays into
flat byte buffers with an accompanying offset array, and deserialises them
back.

The encoding is simple concatenation of raw bytes.  The offset array records
where each group starts so that individual groups can be extracted without
decoding the whole chunk (enabling HTTP range reads on cloud stores).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from zarr_vectors.exceptions import ArrayError

# ---------------------------------------------------------------------------
# Ragged float encoding (vertex positions, float/int attributes)
# ---------------------------------------------------------------------------

def encode_ragged_floats(
    groups: list[npt.NDArray],
    dtype: np.dtype,
) -> tuple[bytes, npt.NDArray[np.int64]]:
    """Encode a list of ragged float arrays into a single byte buffer.

    Args:
        groups: List of arrays, each shape ``(N_k, D)`` or ``(N_k,)`` for
            fragment *k*.  All arrays must have the same number of
            columns (D) and be castable to *dtype*.
        dtype: Target numpy dtype for serialisation.

    Returns:
        raw_bytes: Concatenated byte buffer of all groups.
        offsets: ``(K,)`` int64 array of byte offsets where each group starts.
            The end of group *k* is ``offsets[k+1]`` (or ``len(raw_bytes)``
            for the last group).

    Raises:
        ArrayError: If groups have inconsistent column counts.
    """
    if not groups:
        return b"", np.empty(0, dtype=np.int64)

    dtype = np.dtype(dtype)

    # Fast path: every group already an array of the target dtype and a
    # common shape past the first axis -- what every writer hands over
    # -- so the buffer is one concatenation and one ``tobytes``, and the
    # offsets follow from the row counts.  A line store writes half a
    # million one-row groups; serialising each on its own was a third of
    # the write.  Anything else (a mixed-shape list, a group that needs
    # casting) takes the per-group path below, byte for byte the same.
    first = groups[0]
    if (
        isinstance(first, np.ndarray)
        and first.dtype == dtype
        and first.ndim >= 1
        and all(
            isinstance(g, np.ndarray)
            and g.dtype == dtype
            and g.shape[1:] == first.shape[1:]
            for g in groups
        )
    ):
        row_bytes = int(dtype.itemsize) * int(np.prod(first.shape[1:], dtype=np.int64))
        counts = np.fromiter((g.shape[0] for g in groups), dtype=np.int64, count=len(groups))
        offsets = np.empty(len(groups), dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts[:-1] * row_bytes, out=offsets[1:])
        buffer = np.concatenate(groups, axis=0) if len(groups) > 1 else first
        return np.ascontiguousarray(buffer).tobytes(), offsets

    parts: list[bytes] = []
    offsets_list: list[int] = []
    current_offset = 0

    for g in groups:
        offsets_list.append(current_offset)
        arr = np.asarray(g, dtype=dtype)
        if arr.ndim == 0:
            arr = arr.reshape(0)
        raw = arr.tobytes()
        parts.append(raw)
        current_offset += len(raw)

    return b"".join(parts), np.array(offsets_list, dtype=np.int64)


def decode_ragged_floats(
    raw_bytes: bytes,
    offsets: npt.NDArray[np.int64],
    dtype: np.dtype,
    ncols: int,
) -> list[npt.NDArray]:
    """Decode a byte buffer back into a list of ragged float arrays.

    Args:
        raw_bytes: The concatenated byte buffer produced by
            :func:`encode_ragged_floats`.
        offsets: ``(K,)`` byte offset array.
        dtype: The numpy dtype used during encoding.
        ncols: Number of columns per row (e.g. 3 for XYZ positions).
            Use 1 for flat per-vertex scalars.

    Returns:
        List of arrays, each shape ``(N_k, ncols)`` (or ``(N_k,)`` when
        *ncols* is 1).

    Raises:
        ArrayError: If offsets are out of range or data does not divide evenly.
    """
    if len(offsets) == 0:
        return []

    dtype = np.dtype(dtype)
    itemsize = dtype.itemsize
    total_len = len(raw_bytes)
    groups: list[npt.NDArray] = []

    for i in range(len(offsets)):
        start = int(offsets[i])
        end = int(offsets[i + 1]) if i + 1 < len(offsets) else total_len

        if start < 0 or end > total_len or start > end:
            raise ArrayError(
                f"Invalid offset range [{start}, {end}) for buffer of length {total_len}"
            )

        segment = raw_bytes[start:end]
        n_elements = len(segment) // itemsize

        if len(segment) % itemsize != 0:
            raise ArrayError(
                f"Segment length {len(segment)} is not divisible by dtype "
                f"itemsize {itemsize}"
            )

        arr = np.frombuffer(segment, dtype=dtype)

        if ncols > 1:
            if n_elements % ncols != 0:
                raise ArrayError(
                    f"Element count {n_elements} is not divisible by ncols={ncols}"
                )
            arr = arr.reshape(-1, ncols)

        groups.append(arr)

    return groups


# ---------------------------------------------------------------------------
# Ragged integer encoding (links, groupings)
# ---------------------------------------------------------------------------

def encode_ragged_ints(
    groups: list[npt.NDArray],
    dtype: np.dtype = np.dtype(np.int64),
) -> tuple[bytes, npt.NDArray[np.int64]]:
    """Encode ragged integer arrays into a byte buffer with offsets.

    Identical to :func:`encode_ragged_floats` but with a default integer
    dtype.  Used for link arrays, grouping arrays, and similar.

    Args:
        groups: List of integer arrays, each shape ``(M_k, L)`` or ``(M_k,)``.
        dtype: Integer dtype (default int64).

    Returns:
        raw_bytes, offsets — same semantics as :func:`encode_ragged_floats`.
    """
    return encode_ragged_floats(groups, dtype=dtype)


def decode_ragged_ints(
    raw_bytes: bytes,
    offsets: npt.NDArray[np.int64],
    dtype: np.dtype = np.dtype(np.int64),
    ncols: int = 1,
) -> list[npt.NDArray]:
    """Decode ragged integer arrays from a byte buffer.

    Args:
        raw_bytes: Byte buffer from :func:`encode_ragged_ints`.
        offsets: ``(K,)`` byte offset array.
        dtype: Integer dtype used during encoding.
        ncols: Number of columns per row (e.g. 2 for edges, 3 for tri faces).

    Returns:
        List of integer arrays.
    """
    return decode_ragged_floats(raw_bytes, offsets, dtype=dtype, ncols=ncols)


# v0.5 ``encode_object_index`` / ``decode_object_index`` and
# ``encode_vertex_offsets`` / ``decode_vertex_offsets`` were removed in
# 0.6.0.  Object manifests are now encoded as chunk-block payloads (see
# :func:`zarr_vectors.encoding.fragments.encode_object_manifest_blocks`),
# and per-chunk vertex grouping moved to a fragment-index blob under
# ``vertex_fragments/`` (see :mod:`zarr_vectors.encoding.fragments`).
# 0.5.x stores are not readable by this build — rewrite from source.


# ---------------------------------------------------------------------------
# Self-describing ragged blob (inline header)
#
# Used by per-chunk link blobs where per-vertex-group boundaries cannot
# be derived from another array.  Layout:
#
#   [K : int64]              # number of groups
#   [off_0, ..., off_{K-1}]  # K int64 byte offsets into the data section
#   [data]                   # concatenated raw bytes
#
# Each off_k is the byte offset of group k inside ``data``.  The
# end of the last group is ``len(data)``.
# ---------------------------------------------------------------------------

def encode_ragged_blob(
    groups: list[npt.NDArray],
    dtype: np.dtype,
) -> bytes:
    """Encode a list of ragged arrays with inline offset header."""
    data_bytes, offsets = encode_ragged_floats(groups, dtype)
    k = len(offsets)
    header = np.empty(1 + k, dtype=np.int64)
    header[0] = k
    if k:
        header[1:] = offsets
    return header.tobytes() + data_bytes


def decode_ragged_blob(
    raw_bytes: bytes,
    dtype: np.dtype,
    ncols: int = 1,
) -> list[npt.NDArray]:
    """Decode an inline-header ragged blob produced by
    :func:`encode_ragged_blob`."""
    if len(raw_bytes) < 8:
        return []
    k = int(np.frombuffer(raw_bytes[:8], dtype=np.int64)[0])
    header_len = 8 * (1 + k)
    if k == 0:
        return []
    offsets = np.frombuffer(
        raw_bytes[8:header_len], dtype=np.int64,
    ).copy()
    data = raw_bytes[header_len:]
    return decode_ragged_floats(data, offsets, dtype, ncols)
