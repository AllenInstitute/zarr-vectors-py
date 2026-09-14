"""Fragment-index encoding and decoding for per-chunk ragged structures.

A *fragment index* describes the F fragments inside one spatial chunk's
``vertex_fragments/<i.j.k>`` or ``link_fragments/<i.j.k>`` blob.  Each
fragment is either:

* a contiguous **range** ``[start, start+count)`` of row indices into the
  chunk's ``vertices/<i.j.k>`` (or ``links/0/<i.j.k>``) array, or
* an explicit **list** of row indices, allowing two fragments in the same
  chunk to re-use the same underlying vertex/link.

The on-disk byte layout is fixed at version 1:

.. code-block:: text

    HEADER (16 bytes, 8-byte-aligned)
      uint32 magic            = 0x5A56_4647  ('ZVFG')
      uint16 version          = 1
      uint16 flags            = 0            (reserved)
      uint32 num_fragments F
      uint32 num_range_fragments R           (popcount of bitmap; redundant)

    RANGE BITMAP
      ceil(F/8) bytes, padded to next 8-byte boundary
      bit f (LSB-first within byte f//8) = 1 iff fragment f is a range

    RANGE TABLE (R entries × 16 bytes)
      int64 start, int64 count   per range fragment, fragment-index order

    EXPLICIT CSR (E = F − R entries)
      uint32 explicit_offsets[E+1]   running offsets into explicit_indices
      int64  explicit_indices[T]     concatenated indices, T = explicit_offsets[E]

The layout is designed so that :meth:`ChunkFragmentIndex.is_range` is a single
bit lookup — no fragment payload has to be decoded to answer "is this
fragment a contiguous range?".  Random access to a fragment's representation
goes through a one-time prefix-popcount cache built lazily on first call.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import numpy.typing as npt

from zarr_vectors.exceptions import ArrayError

# Public constants ---------------------------------------------------

#: Magic number written at the start of every fragment-index blob.
FRAGMENT_INDEX_MAGIC: int = 0x5A56_4647  # 'ZVFG'

#: On-disk format version this module reads and writes.
FRAGMENT_INDEX_VERSION: int = 1

#: Header length in bytes (kept 8-byte-aligned for downstream int64 fields).
_HEADER_BYTES: int = 16

#: Header struct: magic, version, flags, num_fragments, num_range_fragments.
_HEADER_STRUCT = struct.Struct("<IHHII")
assert _HEADER_STRUCT.size == _HEADER_BYTES


def _bitmap_padded_length(num_fragments: int) -> int:
    """Number of bytes the range bitmap occupies on disk, including padding
    to the next 8-byte boundary so the subsequent int64 range table is
    naturally aligned."""
    raw = (num_fragments + 7) // 8
    return (raw + 7) & ~7


# Encoding -----------------------------------------------------------


Fragment = npt.NDArray[np.integer] | tuple[int, int]


def _classify_fragment(
    fragment: Fragment, *, force_explicit: bool,
) -> tuple[bool, tuple[int, int] | np.ndarray]:
    """Return ``(is_range, payload)`` for one input fragment.

    ``payload`` is either ``(start, count)`` for the range path or an
    ``np.ndarray[int64]`` for the explicit path.  Auto-detection looks
    for any non-negative array where ``arr == arange(arr[0], arr[0]+len(arr))``;
    pass ``force_explicit=True`` to bypass the check.
    """
    if isinstance(fragment, tuple):
        if len(fragment) != 2:
            raise ArrayError(
                f"Fragment tuple must have shape (start, count), got {fragment!r}",
            )
        start, count = int(fragment[0]), int(fragment[1])
        if count < 0:
            raise ArrayError(f"Fragment count must be >= 0, got {count}")
        return True, (start, count)

    arr = np.asarray(fragment)
    if arr.ndim != 1:
        raise ArrayError(
            f"Fragment array must be 1-D, got shape {arr.shape}",
        )
    arr = arr.astype(np.int64, copy=False)
    if not force_explicit and arr.size > 0:
        start = int(arr[0])
        # Cheap arange-detect: compare to arange.  For small fragments
        # this is bounded; for very long fragments we still want the
        # range path because it's by far the more compressible
        # representation.
        if start >= 0 and np.array_equal(
            arr, np.arange(start, start + arr.size, dtype=np.int64),
        ):
            return True, (start, int(arr.size))
    if arr.size == 0:
        # Zero-length explicit slice — preserve the explicit path so the
        # caller can distinguish an empty list from an empty range if
        # they care; but for the on-disk encoding both shapes have the
        # same cost (1 csr_offset slot, 0 indices), so this is mostly a
        # round-tripping detail.
        return False, arr
    if int(arr.min()) < 0:
        raise ArrayError(
            "Explicit fragment indices must be non-negative",
        )
    return False, arr


def encode_fragments(
    fragments: Sequence[Fragment],
    *,
    force_explicit: bool = False,
) -> bytes:
    """Encode F fragments into the v1 byte layout.

    Args:
        fragments: One entry per fragment.  Each entry is either a
            ``(start, count)`` tuple (always emitted as a range), or a
            1-D integer array (auto-detected as a range when it equals
            ``arange(arr[0], arr[0]+len(arr))``, else explicit).
        force_explicit: If True, never auto-promote arrays to the range
            path.  Tuples are still emitted as ranges — pass an array if
            you need the explicit path unconditionally.

    Returns:
        The fragment-index blob, suitable for writing as the single
        chunk of a 1-D uint8 zarr v3 array.
    """
    f = len(fragments)
    if f == 0:
        # Header-only sentinel for empty chunks.
        return _HEADER_STRUCT.pack(
            FRAGMENT_INDEX_MAGIC,
            FRAGMENT_INDEX_VERSION,
            0,                # flags
            0,                # num_fragments
            0,                # num_range_fragments
        )

    classified: list[tuple[bool, tuple[int, int] | np.ndarray]] = [
        _classify_fragment(frag, force_explicit=force_explicit)
        for frag in fragments
    ]

    # Bitmap (range bits) ------------------------------------------------
    bitmap_len = _bitmap_padded_length(f)
    bitmap = bytearray(bitmap_len)
    r = 0
    for i, (is_range, _) in enumerate(classified):
        if is_range:
            bitmap[i >> 3] |= 1 << (i & 7)
            r += 1
    e = f - r

    # Range table --------------------------------------------------------
    range_table = np.empty((r, 2), dtype=np.int64)
    r_idx = 0
    for is_range, payload in classified:
        if is_range:
            start, count = payload  # type: ignore[misc]
            range_table[r_idx, 0] = start
            range_table[r_idx, 1] = count
            r_idx += 1

    # Explicit CSR -------------------------------------------------------
    explicit_indices_list: list[np.ndarray] = []
    explicit_offsets = np.empty(e + 1, dtype=np.uint32)
    explicit_offsets[0] = 0
    e_idx = 0
    for is_range, payload in classified:
        if not is_range:
            arr = payload  # type: ignore[assignment]
            explicit_indices_list.append(arr)
            e_idx += 1
            explicit_offsets[e_idx] = explicit_offsets[e_idx - 1] + arr.size

    if explicit_indices_list:
        explicit_indices = np.concatenate(
            explicit_indices_list, dtype=np.int64,
        )
    else:
        explicit_indices = np.empty(0, dtype=np.int64)

    # Pack --------------------------------------------------------------
    header = _HEADER_STRUCT.pack(
        FRAGMENT_INDEX_MAGIC,
        FRAGMENT_INDEX_VERSION,
        0,    # flags
        f,
        r,
    )
    parts: list[bytes] = [
        header,
        bytes(bitmap),
        range_table.tobytes(),
        explicit_offsets.tobytes(),
        explicit_indices.tobytes(),
    ]
    return b"".join(parts)


# Decoding -----------------------------------------------------------


@dataclass(frozen=True)
class ChunkFragmentIndex:
    """Decoded view of one chunk's fragment-index blob.

    The dataclass holds zero-copy numpy views onto the source bytes
    (where dtype-alignment allows) plus a lazily-built prefix-popcount
    cache for random access.  ``is_range(f)`` is always a single bit
    lookup — no fragment payload is materialised until ``range(f)``,
    ``indices(f)`` or ``indices_view(f)`` is called.
    """

    num_fragments: int
    _bitmap: np.ndarray            # uint8, length ceil(F/8) (unpadded view)
    _range_table: np.ndarray       # (R, 2) int64
    _csr_offsets: np.ndarray       # (E+1,) uint32
    _csr_indices: np.ndarray       # (T,) int64
    # Lazy cache: prefix-popcount of the *bitmap* (inclusive at index i+1).
    # _popcount_prefix[k] == number of range fragments in [0, k).
    # Wrapped in a list so the frozen dataclass can mutate it.
    _popcount_cache: list = field(default_factory=list)

    # Public API --------------------------------------------------------

    def __len__(self) -> int:
        return self.num_fragments

    @property
    def num_range_fragments(self) -> int:
        return int(self._range_table.shape[0])

    @property
    def num_explicit_fragments(self) -> int:
        return self.num_fragments - self.num_range_fragments

    @property
    def vertex_extent(self) -> int:
        """Number of underlying vertices this index spans.

        ``max(referenced vertex index) + 1`` across every fragment — the
        length of the ``vertices`` array the chunk's per-vertex attributes
        align 1:1 with.

        Computed as two reductions over the whole ``_range_table`` /
        ``_csr_indices`` buffers rather than a per-fragment loop, because
        the number of fragments is unbounded: a BRIDGE chunk carries one
        range fragment plus a path-fragment twin per polyline, and at
        ~187k twins the loop form cost ~0.9 s **per attribute per read**,
        which is what made a neighbourhood load scale with how much of the
        store was already built.
        """
        extent = 0
        if self._range_table.size:
            ends = self._range_table[:, 0] + self._range_table[:, 1]
            extent = int(ends.max())
        if self._csr_indices.size:
            extent = max(extent, int(self._csr_indices.max()) + 1)
        return extent

    def tiles(self, n_rows: int) -> bool:
        """Do the fragments partition ``[0, n_rows)`` in fragment order?

        True only when every fragment is a range, the first starts at row
        0, each one begins where the last ended, and the last ends at
        ``n_rows``.  Under that condition concatenating the fragments
        reproduces the underlying buffer exactly — same rows, same order
        — so a caller that only wants the concatenation can skip the
        partition entirely.  It is the layout every bulk writer produces,
        because it assigns rows bin by bin; a level written per object
        and then edited is where it stops holding.

        Decided from the range table as two whole-array comparisons
        rather than a per-fragment loop, because the fragment count is
        unbounded — the check has to stay cheap on the chunks it does
        *not* fire for.
        """
        if self.num_explicit_fragments:
            return False
        table = self._range_table
        if table.shape[0] != self.num_fragments or table.shape[0] == 0:
            return False
        starts = table[:, 0]
        ends = starts + table[:, 1]
        return bool(
            starts[0] == 0
            and int(ends[-1]) == n_rows
            and np.array_equal(ends[:-1], starts[1:])
        )

    def is_range(self, f: int) -> bool:
        """Return True if fragment ``f`` is a contiguous range.

        One bit lookup — does not decode any fragment payload.
        """
        if f < 0 or f >= self.num_fragments:
            raise IndexError(
                f"Fragment index {f} out of range [0, {self.num_fragments})",
            )
        return bool((self._bitmap[f >> 3] >> (f & 7)) & 1)

    def range(self, f: int) -> tuple[int, int]:
        """Return ``(start, count)`` for a range fragment.

        Raises:
            ArrayError: If fragment ``f`` is not a range.
        """
        if not self.is_range(f):
            raise ArrayError(
                f"Fragment {f} is explicit, not a range; "
                f"use .indices(f) or .indices_view(f)",
            )
        row = self._popcount_prefix()[f]
        start = int(self._range_table[row, 0])
        count = int(self._range_table[row, 1])
        return start, count

    def indices(self, f: int) -> npt.NDArray[np.int64]:
        """Return the indices of fragment ``f`` as a 1-D ``int64`` array.

        For range fragments this materialises ``arange(start, start+count)``.
        For explicit fragments this returns a copy of the CSR slice (use
        :meth:`indices_view` for a zero-copy view).
        """
        if self.is_range(f):
            start, count = self.range(f)
            return np.arange(start, start + count, dtype=np.int64)
        prefix = self._popcount_prefix()
        e_idx = f - prefix[f]
        a = int(self._csr_offsets[e_idx])
        b = int(self._csr_offsets[e_idx + 1])
        return self._csr_indices[a:b].copy()

    def indices_view(self, f: int) -> npt.NDArray[np.int64]:
        """Return a zero-copy view onto the indices of an explicit fragment.

        Raises:
            ArrayError: If fragment ``f`` is a range — there is no
                backing array to view; call :meth:`indices` instead.
        """
        if self.is_range(f):
            raise ArrayError(
                f"Fragment {f} is a range; .indices_view requires an "
                f"explicit fragment.  Use .indices(f) to materialise.",
            )
        prefix = self._popcount_prefix()
        e_idx = f - prefix[f]
        a = int(self._csr_offsets[e_idx])
        b = int(self._csr_offsets[e_idx + 1])
        return self._csr_indices[a:b]

    # Internals ---------------------------------------------------------

    def _popcount_prefix(self) -> npt.NDArray[np.int32]:
        """Return (and lazily build) the bitmap prefix-popcount.

        ``_popcount_prefix[i]`` == count of set bits in bitmap[0..i)
        (i.e. how many fragments in ``[0, i)`` are ranges).  Useful for
        translating a fragment index ``f`` into a row of ``_range_table``
        (``prefix[f]`` if ``is_range(f)``) or ``_csr_offsets``
        (``f - prefix[f]`` if not).
        """
        if self._popcount_cache:
            return self._popcount_cache[0]
        # One pass over the bitmap, then a cumulative sum.  This is
        # O(F) once per ChunkFragmentIndex; subsequent queries are O(1).
        f = self.num_fragments
        bits = np.unpackbits(
            self._bitmap, bitorder="little",
        )[:f].astype(np.int32, copy=False)
        prefix = np.empty(f + 1, dtype=np.int32)
        prefix[0] = 0
        np.cumsum(bits, out=prefix[1:])
        self._popcount_cache.append(prefix)
        return prefix


def decode_fragments(raw: bytes) -> ChunkFragmentIndex:
    """Parse a v1 fragment-index blob into a :class:`ChunkFragmentIndex`."""
    if len(raw) < _HEADER_BYTES:
        raise ArrayError(
            f"Fragment-index blob too short: {len(raw)} < {_HEADER_BYTES}",
        )
    magic, version, flags, f, r = _HEADER_STRUCT.unpack_from(raw, 0)
    if magic != FRAGMENT_INDEX_MAGIC:
        raise ArrayError(
            f"Bad fragment-index magic: got 0x{magic:08X}, "
            f"want 0x{FRAGMENT_INDEX_MAGIC:08X}",
        )
    if version != FRAGMENT_INDEX_VERSION:
        raise ArrayError(
            f"Unsupported fragment-index version {version}; "
            f"this code reads version {FRAGMENT_INDEX_VERSION}",
        )
    if flags != 0:
        raise ArrayError(
            f"Unsupported fragment-index flags 0x{flags:04X}; expected 0",
        )
    if r > f:
        raise ArrayError(
            f"num_range_fragments {r} exceeds num_fragments {f}",
        )

    if f == 0:
        return ChunkFragmentIndex(
            num_fragments=0,
            _bitmap=np.empty(0, dtype=np.uint8),
            _range_table=np.empty((0, 2), dtype=np.int64),
            _csr_offsets=np.zeros(1, dtype=np.uint32),
            _csr_indices=np.empty(0, dtype=np.int64),
        )

    offset = _HEADER_BYTES

    bitmap_raw_bytes = (f + 7) // 8
    bitmap_padded = _bitmap_padded_length(f)
    if len(raw) < offset + bitmap_padded:
        raise ArrayError(
            f"Fragment-index blob truncated in bitmap region",
        )
    # Copy out the unpadded portion as our canonical bitmap.  Copying
    # is cheap (≤ ceil(F/8) bytes) and avoids retaining the whole input
    # buffer just for a tiny view.
    bitmap = np.frombuffer(
        raw, dtype=np.uint8, count=bitmap_raw_bytes, offset=offset,
    ).copy()
    offset += bitmap_padded

    range_table_bytes = r * 16
    if len(raw) < offset + range_table_bytes:
        raise ArrayError(
            f"Fragment-index blob truncated in range table",
        )
    range_table = np.frombuffer(
        raw, dtype=np.int64, count=r * 2, offset=offset,
    ).reshape(r, 2).copy()
    offset += range_table_bytes

    e = f - r
    csr_offsets_bytes = (e + 1) * 4
    if len(raw) < offset + csr_offsets_bytes:
        raise ArrayError(
            f"Fragment-index blob truncated in CSR offsets",
        )
    csr_offsets = np.frombuffer(
        raw, dtype=np.uint32, count=e + 1, offset=offset,
    ).copy()
    offset += csr_offsets_bytes

    t = int(csr_offsets[e]) if e > 0 else 0
    csr_indices_bytes = t * 8
    if len(raw) < offset + csr_indices_bytes:
        raise ArrayError(
            f"Fragment-index blob truncated in CSR indices",
        )
    csr_indices = np.frombuffer(
        raw, dtype=np.int64, count=t, offset=offset,
    ).copy()

    return ChunkFragmentIndex(
        num_fragments=f,
        _bitmap=bitmap,
        _range_table=range_table,
        _csr_offsets=csr_offsets,
        _csr_indices=csr_indices,
    )


def read_fragment_index(
    level_group,
    array_name: str,
    chunk_coords: tuple[int, ...],
) -> ChunkFragmentIndex:
    """Read and decode the fragment-index blob for one chunk.

    ``level_group`` is a :class:`zarr_vectors.core.group.FsGroup` (or
    equivalent) exposing :meth:`read_bytes` and
    :meth:`chunk_exists`.  Returns an empty :class:`ChunkFragmentIndex` when
    the chunk's blob is absent — callers that need to distinguish
    "missing" from "empty fragment list" should check
    ``level_group.chunk_exists(array_name, chunk_key)`` first.
    """
    chunk_key = ".".join(str(c) for c in chunk_coords)
    raw = level_group.read_bytes(array_name, chunk_key)
    return decode_fragments(raw)


# ---------------------------------------------------------------------------
# Object-index manifest blocks
# ---------------------------------------------------------------------------
#
# Each object's manifest is a sequence of per-chunk *blocks*.  A block
# carries one chunk's coordinates plus a fragment reference encoded in
# one of three modes:
#
#   mode 0  uint8                         single fragment
#           int64 fragment_index
#   mode 1  uint8                         contiguous range
#           int64 start, int64 count
#   mode 2  uint8                         explicit list
#           uint32 count
#           int64 fragment_indices[count]
#
# The manifest is preceded by:
#
#   uint32 num_blocks B
#
# All fragment references are *chunk-local* — they index into the
# ``vertex_fragments/<chunk_coords>`` array of the block's named chunk
# only, never across chunks.  This preserves chunk-write independence:
# chunks can be written without coordinating fragment numbering with
# any other chunk.
#
# An empty manifest is 4 bytes: ``B=0``.

# Mode tags
MANIFEST_MODE_SINGLE = 0
MANIFEST_MODE_RANGE = 1
MANIFEST_MODE_EXPLICIT = 2


ObjectManifestBlock = tuple[
    tuple[int, ...],  # chunk_coords
    npt.NDArray[np.integer] | tuple[int, int] | int,  # fragment ref
]


def _encode_one_block(
    chunk_coords: tuple[int, ...],
    fragment_ref,
    sid_ndim: int,
    *,
    force_explicit: bool,
) -> bytes:
    if len(chunk_coords) != sid_ndim:
        raise ArrayError(
            f"chunk_coords {chunk_coords} has rank {len(chunk_coords)}, "
            f"expected sid_ndim={sid_ndim}",
        )
    coords_bytes = np.asarray(chunk_coords, dtype=np.int64).tobytes()

    # int → single
    if isinstance(fragment_ref, (int, np.integer)):
        idx = int(fragment_ref)
        if idx < 0:
            raise ArrayError(
                f"fragment_index must be >= 0, got {idx}",
            )
        return (
            coords_bytes
            + struct.pack("<B", MANIFEST_MODE_SINGLE)
            + struct.pack("<q", idx)
        )

    # tuple → range
    if isinstance(fragment_ref, tuple):
        if len(fragment_ref) != 2:
            raise ArrayError(
                f"Fragment range tuple must be (start, count), got {fragment_ref!r}",
            )
        start, count = int(fragment_ref[0]), int(fragment_ref[1])
        if count < 0:
            raise ArrayError(f"range count must be >= 0, got {count}")
        return (
            coords_bytes
            + struct.pack("<B", MANIFEST_MODE_RANGE)
            + struct.pack("<qq", start, count)
        )

    # array → auto-detect (range vs explicit)
    arr = np.asarray(fragment_ref)
    if arr.ndim != 1:
        raise ArrayError(
            f"Fragment ref array must be 1-D, got shape {arr.shape}",
        )
    arr = arr.astype(np.int64, copy=False)
    if not force_explicit and arr.size > 0:
        start = int(arr[0])
        if start >= 0 and np.array_equal(
            arr, np.arange(start, start + arr.size, dtype=np.int64),
        ):
            return (
                coords_bytes
                + struct.pack("<B", MANIFEST_MODE_RANGE)
                + struct.pack("<qq", start, int(arr.size))
            )
    if arr.size > 0 and int(arr.min()) < 0:
        raise ArrayError(
            "Explicit fragment indices must be non-negative",
        )
    return (
        coords_bytes
        + struct.pack("<B", MANIFEST_MODE_EXPLICIT)
        + struct.pack("<I", int(arr.size))
        + arr.tobytes()
    )


def encode_object_manifest_blocks(
    blocks: Sequence[ObjectManifestBlock],
    sid_ndim: int,
    *,
    force_explicit: bool = False,
) -> bytes:
    """Encode one object's manifest into the v0.6 block format.

    Args:
        blocks: List of ``(chunk_coords, fragment_ref)`` pairs.  Each
            block names one spatial chunk and the fragment(s) the
            object owns inside that chunk.  ``fragment_ref`` may be:

            - an ``int`` (single fragment index, mode 0),
            - a ``(start, count)`` tuple (contiguous range, mode 1), or
            - a 1-D integer array (auto-detected: arange → range,
              otherwise explicit list).
        sid_ndim: Rank of the chunk-coordinate space.  All
            ``chunk_coords`` tuples must be of this length.
        force_explicit: If True, never auto-promote arrays to range.

    Returns:
        The manifest bytes — a 4-byte block count followed by ``B``
        per-block payloads.  An empty list returns 4 bytes (``B=0``).
    """
    block_bytes = [
        _encode_one_block(c, f, sid_ndim, force_explicit=force_explicit)
        for c, f in blocks
    ]
    return struct.pack("<I", len(blocks)) + b"".join(block_bytes)


def decode_object_manifest_blocks(
    raw: bytes,
    sid_ndim: int,
) -> list[
    tuple[
        tuple[int, ...],
        int | tuple[int, int] | npt.NDArray[np.int64],
    ]
]:
    """Decode a v0.6 manifest blob into ``(chunk_coords, fragment_ref)``
    pairs.

    Each returned ``fragment_ref`` is one of:

    - ``int`` for mode-0 blocks,
    - ``(start, count)`` tuple for mode-1 blocks,
    - ``np.ndarray[int64]`` for mode-2 blocks.

    Callers that prefer a uniform shape (e.g. always an array of indices)
    should map over the result.
    """
    if len(raw) < 4:
        raise ArrayError(
            f"Manifest blob too short: {len(raw)} < 4",
        )
    (b,) = struct.unpack_from("<I", raw, 0)
    offset = 4
    coords_bytes = sid_ndim * 8

    result: list[tuple[tuple[int, ...], int | tuple[int, int] | np.ndarray]] = []
    for _ in range(b):
        if len(raw) < offset + coords_bytes + 1:
            raise ArrayError("Manifest blob truncated at block header")
        coords_arr = np.frombuffer(
            raw, dtype=np.int64, count=sid_ndim, offset=offset,
        )
        coords = tuple(int(c) for c in coords_arr)
        offset += coords_bytes
        (mode,) = struct.unpack_from("<B", raw, offset)
        offset += 1

        if mode == MANIFEST_MODE_SINGLE:
            if len(raw) < offset + 8:
                raise ArrayError("Manifest blob truncated in single-mode payload")
            (idx,) = struct.unpack_from("<q", raw, offset)
            offset += 8
            result.append((coords, int(idx)))

        elif mode == MANIFEST_MODE_RANGE:
            if len(raw) < offset + 16:
                raise ArrayError("Manifest blob truncated in range-mode payload")
            start, count = struct.unpack_from("<qq", raw, offset)
            offset += 16
            result.append((coords, (int(start), int(count))))

        elif mode == MANIFEST_MODE_EXPLICIT:
            if len(raw) < offset + 4:
                raise ArrayError(
                    "Manifest blob truncated in explicit-mode header",
                )
            (count,) = struct.unpack_from("<I", raw, offset)
            offset += 4
            indices_bytes = count * 8
            if len(raw) < offset + indices_bytes:
                raise ArrayError(
                    "Manifest blob truncated in explicit-mode indices",
                )
            indices = np.frombuffer(
                raw, dtype=np.int64, count=count, offset=offset,
            ).copy()
            offset += indices_bytes
            result.append((coords, indices))

        else:
            raise ArrayError(f"Unknown manifest block mode {mode}")

    if offset != len(raw):
        raise ArrayError(
            f"Manifest blob has {len(raw) - offset} trailing bytes after "
            f"{b} blocks",
        )
    return result
