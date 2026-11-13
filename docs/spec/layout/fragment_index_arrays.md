# Fragment-index arrays

This page specifies the byte layout of the fragment-index blob stored in
each cell of `vertex_fragments/` and `link_fragments/`, and the arrays that
hold it. What fragments are, and how objects and readers use them, is on
[Fragments](../object_model/fragments.md).

## Terms

**Fragment index**
: The blob in one cell of `vertex_fragments/` (or `link_fragments/`) that
  partitions the rows of the matching `vertices/` cell (or intra-chunk
  link cell) into `F` [fragments](../object_model/fragments.md). Each
  fragment is a **range fragment**, a contiguous run of rows
  `[start, start + count)`, or an **explicit fragment**, a list of row
  indices.

**Range bitmap**
: One bit per fragment, set when that fragment is a range.

**Range table**
: One `(start, count)` pair of `int64` per range fragment, in fragment
  order.

**Explicit CSR**
: The explicit fragments' row indices in compressed-sparse-row form: an
  offsets array and one flat array of indices.

## Introduction

A reader that wants some of a chunk's rows, rather than all of them, reads
the chunk's fragment index first. The layout answers "is fragment `f` a
range?" with one bit test, so the common case (every fragment a range)
needs no per-fragment decoding. Explicit fragments let two fragments name
the same row, which is how [shared fragments](../object_model/fragments.md)
at coarser pyramid levels avoid duplicating rows.

The reference implementation is `zarr_vectors.encoding.fragments`
(`encode_fragments`, `decode_fragments`, `ChunkFragmentIndex`); all three
are internal.

## Technical reference

### Byte layout (version 1)

All integers are little-endian.

```text
HEADER (16 bytes)
  uint32 magic                = 0x5A564647  ('ZVFG' as bytes 47 46 56 5A)
  uint16 version              = 1
  uint16 flags                = 0
  uint32 num_fragments        F
  uint32 num_range_fragments  R

RANGE BITMAP
  ceil(F/8) bytes, zero-padded to the next multiple of 8 bytes
  bit f (least significant bit first within byte f >> 3) = 1 if fragment f is a range

RANGE TABLE (R × 16 bytes)
  int64 start, int64 count    one pair per range fragment, in fragment order

EXPLICIT CSR (E = F − R)
  uint32 explicit_offsets[E + 1]
  int64  explicit_indices[T]  T = explicit_offsets[E]
```

The header is the struct `<IHHII`. Its 16 bytes and the bitmap padding keep
the range table 8-byte aligned.

A cell with no fragments is the 16-byte header alone, with `F = 0` and
`R = 0`. A cell with fragments always carries `explicit_offsets[0] = 0`, even
when `E = 0`, so an index of only ranges ends with four zero bytes.

**Range table rows.** Row `r` belongs to the `r`-th set bit of the bitmap,
not to fragment `r`. The row of range fragment `f` is the number of set bits
before bit `f`. `ChunkFragmentIndex` builds that prefix count once, on first
use, and then answers each lookup in constant time.

**Explicit indices.** The `e`-th explicit fragment (counting explicit
fragments only) owns
`explicit_indices[explicit_offsets[e]:explicit_offsets[e+1]]`. Indices are
absolute rows of the chunk's payload cell, must be non-negative, and may
repeat across fragments. A zero-length explicit fragment is allowed.

**Encoder choices.** `encode_fragments` writes a `(start, count)` tuple as
a range, and writes an integer array as a range when it equals
`arange(arr[0], arr[0] + len(arr))`; pass `force_explicit=True` to keep
arrays explicit. The bulk writers emit only ranges.

### Worked example

Three fragments: rows 0 to 3, the rows `[12, 7, 19]`, and rows 20 to 27.

```python
import numpy as np
from zarr_vectors.encoding.fragments import encode_fragments, decode_fragments

blob = encode_fragments([(0, 4), np.array([12, 7, 19]), (20, 8)])
len(blob)                              # 88
fidx = decode_fragments(blob)
[fidx.is_range(f) for f in range(3)]   # [True, False, True]
fidx.range(2)                          # (20, 8)
fidx.indices(1)                        # array([12,  7, 19])
```

`F = 3`, `R = 2`, `E = 1`, `T = 3`:

```text
offset  bytes                     meaning
0x00    47 46 56 5A               magic 'ZVFG'
0x04    01 00                     version = 1
0x06    00 00                     flags = 0
0x08    03 00 00 00               F = 3
0x0C    02 00 00 00               R = 2
0x10    05 00 00 00 00 00 00 00   bitmap: bits 0 and 2 set, padded to 8 bytes
0x18    00 00 00 00 00 00 00 00   range[0].start = 0
0x20    04 00 00 00 00 00 00 00   range[0].count = 4
0x28    14 00 00 00 00 00 00 00   range[1].start = 20
0x30    08 00 00 00 00 00 00 00   range[1].count = 8
0x38    00 00 00 00               explicit_offsets[0] = 0
0x3C    03 00 00 00               explicit_offsets[1] = 3
0x40    0C 00 00 00 00 00 00 00   explicit_indices[0] = 12
0x48    07 00 00 00 00 00 00 00   explicit_indices[1] = 7
0x50    13 00 00 00 00 00 00 00   explicit_indices[2] = 19
```

Resolving fragment 1: bit 1 of `0x05` is 0, so it is explicit. One set bit
precedes it, so it is explicit fragment `1 - 1 = 0`, and its indices are
`explicit_indices[0:3] = [12, 7, 19]`.

### `vertex_fragments/` and `link_fragments/` arrays

Both are per-chunk arrays with the layout given on
[Chunk arrays](chunk_arrays.md): `data_type` `variable_length_bytes`, shape
equal to the level's chunk grid, one fragment-index blob per cell. The
attributes of `vertex_fragments/`:

```json
{
  "nonempty_chunks": ["0.0.0", "0.0.1"],
  "zv_array": "vertex_fragments",
  "encoding": "fragment_index_v1"
}
```

`link_fragments/` carries `"zv_array": "link_fragments"` and the same
`encoding`.

- `vertex_fragments/` partitions the rows of `vertices/`. Every writer
  creates it with `vertices/`.
- `link_fragments/` partitions the rows of the intra-chunk link array at
  level delta 0, `links/0/<all-zero offsets>/`, and no other link array.
  It exists only in stores that write that array: graphs, skeletons and
  meshes. Its cells are keyed by chunk alone, so only one link array per
  chunk can own them; see
  [Links](../object_model/links.md#cell-encoding-two-branches-one-condition).

### Validation

A blob is well formed when:

1. it starts with the magic, version `1` and flags `0`;
2. `R ≤ F`, and `R` equals the number of set bits among the first `F`
   bits of the bitmap;
3. it is long enough for the bitmap, range table, `E + 1` offsets and `T`
   indices it declares;
4. `explicit_offsets` starts at 0 and never decreases;
5. every range `[start, start + count)` and every explicit index lies in
   `[0, rows in the payload cell)`.

`decode_fragments` raises `ArrayError` for a bad magic, version or flags,
for `R > F`, and for a blob too short for its declared sections. It does
not recount the bitmap against `R`.

In the validator, [L1](../validation/l1_structural.md) warns when a level
has no `vertex_fragments/`. [L3](../validation/l3_consistency.md) decodes
every `vertices/` cell through its fragment index and reports a cell that
fails to decode. When a level sets `fragments_tile`, L3 also checks that
each cell's fragments are ranges that cover the cell's rows in order
(`ChunkFragmentIndex.tiles`).
