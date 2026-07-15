# Fragment-index arrays

```{admonition} Format change in ZVF 0.6.0
:class: note

Prior to ZVF 0.6.0 the per-chunk spatial index was a fixed-shape
`int64` array named `vertex_group_offsets/<i.j.k>` storing one
`(offset, count)` row per bin. ZVF 0.6.0 replaced it with the
binary **fragment-index** format described on this page. The
semantic role is the same — locating vertices within a chunk —
but the new format adds two capabilities: *explicit* (non-contiguous)
fragments, and *fragment sharing* across object manifests at
coarsened pyramid levels.

0.5.x stores are not readable by 0.6+ readers; rewrite from source.
```

## Terms

**Fragment**
: A logical grouping of rows inside one chunk's `vertices/<i.j.k>`
  (or `links/0/<all-zero offsets>/<i.j.k>`) array. A fragment is either a contiguous
  range of rows or an explicit list of row indices. At level 0 with
  the default writer, each non-empty bin emits exactly one fragment;
  at coarsened pyramid levels a fragment may represent a metavertex
  shared between several objects' manifests.

**Range fragment**
: A fragment that owns a contiguous run `[start, start + count)` of
  row indices into the chunk's payload array. The historical analogue
  of a pre-0.6 `(offset, count)` row.

**Explicit fragment**
: A fragment that owns an arbitrary, non-contiguous list of row
  indices. Enables a single chunk row to participate in more than
  one fragment — the primitive behind shared metavertices at
  coarsened levels.

**Fragment index** (the array `vertex_fragments/<i.j.k>`)
: The binary blob inside one chunk that describes the F fragments
  in that chunk. Encoded in the v1 layout described below
  (magic `'ZVFG'` = `0x5A56_4647`).

**Range bitmap**
: A bitmap of length F (padded to bytes) inside the fragment index,
  where bit `f` is set iff fragment `f` is a range. Lets readers
  classify a fragment in one bit lookup.

**Range table**
: The packed array of `(start, count)` int64 pairs, one row per
  range fragment, stored in fragment-index order.

**CSR (explicit)**
: The compressed-sparse-row layout used for explicit fragments:
  a `uint32 offsets[E+1]` array of running offsets, plus a flat
  `int64 indices[T]` array of concatenated row indices.

**Shared fragment**
: A fragment named by more than one object's manifest. Stored once;
  vertex rows are not duplicated. Marked at the level via
  `LevelMetadata.shared_fragments=True` and at the store via the
  `CAP_SHARED_FRAGMENTS` capability token.

### What happened to `offset` and `count`?

| Pre-0.6 | Post-0.6 |
|---------|----------|
| `vertex_group_offsets[chunk][b, 0]` (offset) | `range_table[r, 0]` (start) for the range fragment that represents bin `b` |
| `vertex_group_offsets[chunk][b, 1]` (count) | `range_table[r, 1]` (count) |
| `count == 0` (empty bin) | No fragment is emitted for that bin |
| `count == -1` (fill, never written) | No fragment is emitted for that bin |

In the new format, empty bins do not occupy a slot; the fixed
`B_per_chunk × 16-byte` overhead of the pre-0.6 layout is gone.
Variable-width fragments and arbitrary index lists become possible
because the index no longer relies on a fixed-shape int64 table.

---

## Introduction

The fragment index is the per-chunk spatial acceleration structure
of ZVF. Without it, a bounding-box query would have to load every
chunk's full vertex payload and filter client-side. With it, readers
load only the byte ranges that overlap the query.

The format was rewritten in ZVF 0.6.0 for two reasons:

1. **Shared metavertices at coarsened levels.** When the pyramid
   builder coarsens N objects whose paths share a metavertex, the
   pre-0.6 design forced the metavertex's vertex row to be duplicated
   per object. The new format lets each object's manifest *name* the
   same fragment in the same chunk; the underlying vertex row is
   stored once.

2. **Eliminate fixed empty-bin overhead.** The pre-0.6
   `vertex_group_offsets/` array stored a `(0, 0)` or `(-1, -1)`
   row for every bin in every chunk — even bins that never held a
   vertex. For sparse stores this dominated index size. The new
   format stores only as many fragments as the chunk needs.

Trade-off: parsing a fragment index requires a tiny decoder
(presented below) instead of a single `int64` Zarr array read.
Decode cost is dominated by a one-time O(F) prefix-popcount built
lazily on first access; subsequent lookups are O(1).

This page describes the byte layout, the decoder algorithm, and the
read path that resolves an object's manifest down to vertex rows.
See [`zarr_vectors.encoding.fragments`](../../api/index.rst) for the
reference implementation.

---

## Design rationale

The fragment-index format makes three structural choices that are not
obvious from the byte layout alone: it supports two fragment kinds
(range and explicit), it discriminates between them with a one-bit-
per-fragment bitmap rather than a tag inside each fragment, and it
deliberately does *not* run a general-purpose compressor over the
result. This section explains why.

### When fragment sharing matters — and when it doesn't

The "reuse" question is whether a single row of `vertices/<chunk>`
(or `links/0/<all-zero offsets>/<chunk>`) can be named by more than one logical owner —
two object manifests, or two adjacent bins, or a parent and a child
in a pyramid. The answer depends on the level and on the writer:

| Scenario | Sharing needed? | Why |
|----------|----------------|-----|
| Level 0, default writer | No | Each vertex belongs to exactly one bin; manifests refer to disjoint row ranges. A contiguous `(start, count)` per non-empty bin is sufficient. |
| Level 0, custom writer that splits one bin across two manifests (e.g. a polyline with both endpoints in the same bin claimed by separate objects) | Yes, but rare | Almost always cheaper to duplicate the few rows than pay the explicit-fragment overhead. |
| Coarsened pyramid level, no merging (`shared_fragments=False`) | No | Coarsening preserves disjoint manifests; only the spatial granularity changes. |
| Coarsened pyramid level, metanode-style merging (`shared_fragments=True`) | **Yes — the dominant case the format exists for** | A coarsened metavertex commonly belongs to *N* parent objects' paths. Duplicating its vertex row *N* times costs N× storage for the row plus N× the bytes through the network on any bbox read that touches that chunk. |

The metanode case drives the whole design. At level 0 the format
collapses to "one range fragment per non-empty bin" — structurally
identical to the pre-0.6 `(offset, count)` table for non-empty rows.
At coarsened levels the same byte layout absorbs sharing without
duplicating rows. The single format covers both cases without a
mode switch at the level or chunk level — only at the per-fragment
bit level.

### Why two fragment kinds at all

Storing every fragment as an explicit index list (the most general
form) seems simpler. It isn't, for three reasons:

1. **Storage cost.** A typical level-0 bin holds dozens to hundreds
   of vertices. As a range fragment it costs 16 bytes regardless of
   `count`. As an explicit list it costs 4 bytes (CSR offset slot) +
   8 bytes × `count` — so `count = 50` is 404 bytes vs. 16 bytes, a
   ~25× blowup on the *common case* in service of a feature only
   coarsened levels need.

2. **Decode cost.** Range fragments materialise as
   `np.arange(start, start + count)` — zero allocation if the caller
   just wants `vertices[start:start+count]`. Explicit fragments
   require a gather load and a CSR offset lookup. Forcing every
   fragment through the gather path adds per-fragment overhead that
   compounds across thousands of fragments in a typical chunk.

3. **Format predictability.** The level-0 stable case maps cleanly
   to the pre-0.6 `(offset, count)` model and to other formats'
   contiguous-row conventions (Arrow run-end, Parquet `RunLength`,
   skeleton polyline conventions). Keeping that representation
   first-class makes the format legible at a glance and makes
   level-0 reads byte-identical in their per-chunk hot path to what
   the pre-0.6 format produced.

Conversely, forcing *every* fragment into a range would forbid the
shared-metavertex case the rewrite was undertaken for. Hence two
kinds, paid for only where they're earned.

### Why the bitmap is the discriminator

Given two kinds, the format needs a way for the reader to ask "what
kind is fragment `f`?" before deciding which table to consult. Three
plausible designs:

| Design | Classify cost | Bytes for `F = 256` fragments, `E = 4` explicit | Random-access by `f` |
|--------|--------------|----------------------------------------------|---------------------|
| Per-fragment tag byte | O(1) | 256 B tags | O(1) |
| List of explicit fragment IDs (sorted) | O(log E) bsearch | `4 × 4 = 16` B | Needs bsearch per query |
| One-bit-per-fragment bitmap | O(1) | `256 / 8 = 32` B | O(1) bit test |
| Detect dynamically from the indices | O(`count`) scan | 0 B but always-explicit storage | Forces all-explicit |

The bitmap dominates: it's a constant fraction of the explicit-tag
cost (1/8th the bytes), it preserves O(1) classification by fragment
index (a single byte fetch + shift + mask), and it never forces a
scan over the index list. The "list of explicit IDs" approach saves
bytes when `E` is tiny, but loses the O(1) classify-by-`f` property
that fragment lookup needs — readers wouldn't know "is `f = 137`
explicit?" without a binary search.

The bitmap pays its modest fixed cost (`ceil(F/8)` bytes) in
exchange for *structural* compression: with one bit per fragment
the format declares "this run is contiguous" without storing the
run elements themselves. The range table then carries only two
int64 values for that run, regardless of length. The combination
— bitmap classifier + per-kind dense tables — is the structural
analogue of run-length encoding, applied to fragment identities
rather than to vertex values.

### Why no outer compressor

The blob is binary-packed already (8-byte-aligned header, dense
bitmap, packed int64 ranges, CSR explicit). For typical fragment
counts (tens to low hundreds per chunk), it's small — usually 100
B to a few KB — and the *latency* cost of running a general
decompressor (LZ4/Zstd/Blosc) on a bbox-query hot path dominates
the bytes-saved gain. Specifically:

- The range table and CSR indices carry near-incompressible payloads
  (sparse int64 row indices into the chunk).
- The bitmap is already near its entropy bound for the typical
  case of "mostly ranges, scattered explicits".
- Decode is on the read-amplification path: a bbox query in a
  region with N overlapping chunks pays the decompression cost N
  times before any vertex byte is touched.

Stores that need to minimise on-disk size MAY wrap the chunk write
in their own compression layer (e.g. an outer Zstd codec at the
Zarr level or a content-addressed-blob layer above the kvstore).
The fragment-index format is independent of that choice. Readers
that don't know about the outer wrapper still parse the raw blob
correctly once it's been unwrapped.

### Reading the format as a compression scheme

It is useful to think of the v1 layout as a small structural
compression scheme rather than as a data structure:

- The **bitmap** encodes a 1-bit "is this row range a constant
  arithmetic progression?" flag per fragment — RLE-of-flags rather
  than RLE-of-values.
- The **range table** stores the two parameters (start, count) that
  reconstruct the implicit arithmetic progression — a tight
  parametric form.
- The **CSR** stores the explicit override list for the fragments
  where the parametric form doesn't apply.
- The **header** carries the popcount that lets the decoder allocate
  the popcount-prefix lookup in one pass.

The reader pays the cost of the explicit override only when the
writer chose to use it. The bitmap is what makes "is this a range?"
a free question — and that, more than any specific byte saving, is
the property the format exists to provide.

---

## Technical reference

### On-disk byte layout (v1)

```
HEADER (16 bytes)
  uint32 magic            = 0x5A56_4647  ('ZVFG')
  uint16 version          = 1
  uint16 flags            = 0            (reserved)
  uint32 num_fragments      F
  uint32 num_range_fragments R           (popcount of bitmap; redundant)

RANGE BITMAP
  ceil(F/8) bytes, padded to next 8-byte boundary
  bit f (LSB-first within byte f >> 3) = 1 iff fragment f is a range

RANGE TABLE
  R entries × 16 bytes
  int64 start, int64 count   per range fragment, in fragment-index order

EXPLICIT CSR (E = F − R)
  uint32 explicit_offsets[E+1]   running offsets into explicit_indices
  int64  explicit_indices[T]     concatenated indices, T = explicit_offsets[E]
```

An empty chunk (no fragments) is a 16-byte header-only blob with
`F = 0`, `R = 0`, no bitmap, no range table, no CSR.

### Header

The 16-byte header is the struct `<IHHII` (little-endian: `uint32`,
`uint16`, `uint16`, `uint32`, `uint32`). It is sized so that the
subsequent range table (`int64` rows) is naturally 8-byte-aligned —
the alignment also drives the bitmap-padding rule below.

Invariant: `num_range_fragments == popcount(range_bitmap[0:F])`.
Readers SHOULD validate this; the reference decoder raises on
mismatch.

### Range bitmap

`ceil(F/8)` raw bytes, then zero-padded to the next 8-byte boundary
so the following range table starts on an `int64`-aligned offset.
Bits are LSB-first within each byte:

```python
is_range_f = (bitmap[f >> 3] >> (f & 7)) & 1
```

Padding bytes are zero on write; decoders MUST ignore them.

### Range table

`R` rows of 16 bytes each: `int64 start`, `int64 count`.

```{important}
**Row order matters.** Row `r` corresponds to the `r`-th *set bit*
of the bitmap when scanned from `f = 0` upward — **not** to the
`f`-th fragment. A naive reader that indexes `range_table[f]`
will get the wrong row whenever any earlier fragment is explicit.
The reference decoder uses a prefix-popcount of the bitmap to map
`f → r` in O(1); see the decoder algorithm below.
```

`start` is the row index in `vertices/<chunk_coords>` of the first
row of the fragment; `count` is the number of rows.

### Explicit CSR

For the `E = F − R` explicit fragments, indices are stored CSR-style:

- `uint32 explicit_offsets[E+1]` — running offsets. Strictly
  monotone non-decreasing. `explicit_offsets[0] == 0`.
- `int64 explicit_indices[T]` — flat array of length
  `T = explicit_offsets[E]`. The indices of explicit fragment with
  *explicit-index* `e` (its position among explicit fragments, not
  among all fragments) are `explicit_indices[offsets[e] : offsets[e+1]]`.

Indices are **absolute row indices** into `vertices/<chunk_coords>`,
not deltas, not bin-relative offsets. Indices MUST be non-negative
and SHOULD lie in `[0, len(vertices[chunk]))`.

A zero-length explicit fragment is legal (`offsets[e+1] == offsets[e]`)
and is the way to round-trip an empty fragment that the writer wanted
to keep distinguishable from an empty range.

### Decoder algorithm

The reference `ChunkFragmentIndex` class holds zero-copy views over the
underlying bytes plus a lazy prefix-popcount cache. The cache is
the only non-obvious piece:

```python
def _popcount_prefix(bitmap, F):
    """O(F) prefix-popcount of the bitmap.
    prefix[i] = number of range fragments in [0, i).
    """
    bits   = np.unpackbits(bitmap, bitorder="little")[:F].astype(np.int32)
    prefix = np.empty(F + 1, dtype=np.int32)
    prefix[0] = 0
    np.cumsum(bits, out=prefix[1:])
    return prefix
```

Built once on first access; subsequent fragment lookups are O(1):

```python
def is_range(f):
    return (bitmap[f >> 3] >> (f & 7)) & 1

def range(f):                            # range fragment f → (start, count)
    r = prefix[f]                        # row in range_table
    return range_table[r, 0], range_table[r, 1]

def indices(f):                          # any fragment f → row indices
    if is_range(f):
        s, c = range(f)
        return np.arange(s, s + c)
    e = f - prefix[f]                    # explicit-index of f
    a, b = csr_offsets[e], csr_offsets[e + 1]
    return csr_indices[a:b]
```

`is_range(f)` is a single bit lookup and does not warm the cache.
`range(f)` and `indices(f)` warm it on first call.

### Query path: manifest → fragments → vertex rows

To read all vertices belonging to one object:

1. Decode the object's **manifest** from `object_index/data` — a
   sequence of per-chunk blocks, each naming `(chunk_coords,
   fragment_ref)` where `fragment_ref` is one of the three manifest
   block modes (single / range / explicit). See
   [Object manifest](../object_model/object_manifest.md) for the
   manifest blob format.

2. For each block, load the chunk's `vertex_fragments/<chunk_coords>`
   blob and `decode_fragments` it.

3. Resolve `fragment_ref` to a set of fragment indices `f`:
   - mode 0 (single): `[fragment_ref]`
   - mode 1 (range): `range(start, start + count)`
   - mode 2 (explicit): the stored `int64` list

4. For each `f`, compute the row indices via `fidx.indices(f)`.

5. Read `vertices/<chunk_coords>[rows, :]` and concatenate across
   blocks in manifest order.

### Bounding-box query path

For spatial (bbox) queries rather than per-object reads:

**Level 0, default writer.** Each non-empty bin has emitted exactly
one range fragment. The bbox query is:

```python
for chunk_coord in overlapping_chunks(bbox):
    fidx = read_fragment_index(VERTEX_FRAGMENTS, chunk_coord)
    for f in range(fidx.num_fragments):
        start, count = fidx.range(f)
        # Optional bin-bbox prefilter: the writer's fragment-order
        # invariant lets readers map f → bin and skip bin bboxes
        # outside the query without reading vertex rows.
        verts = vertices[chunk_coord][start : start + count]
        yield verts  # final point-in-bbox filter applied client-side
```

**Coarsened levels with `shared_fragments=True`.** The query walks
every fragment in overlapping chunks; sharing is invisible to the
spatial read path. A range fragment whose `count` covers a
post-coarsening metavertex still resolves to one contiguous row
slice. An explicit fragment's row list materialises via
`fidx.indices(f)` and gather-loads from `vertices/<chunk_coord>`.

### `vertex_fragments/` array schema

| Property | Value |
|----------|-------|
| zarr `node_type` | `array` |
| dtype | `uint8` |
| logical shape | one variable-length blob per chunk, addressed by chunk coordinate |
| codec pipeline | none — chunks are written as opaque bytes via the FsGroup `write_bytes` path |
| group metadata | `{"zv_array": "vertex_fragments", "encoding": "fragment_index_v1"}` |

Each chunk holds one fragment-index blob in the v1 byte layout
described above. The metadata `encoding: "fragment_index_v1"` tag
identifies the format and is the discriminator that future versions
of ZVF will rev when the binary layout changes.

The blob is intentionally not compressed by default. It is already
binary-packed (header + bitmap + dense int64 range table + uint32/int64
CSR), and general-purpose compression typically adds latency to hot
decode paths without meaningful ratio gain. Stores that must minimise
on-disk size MAY wrap chunk writes in their own compression layer; the
fragment-index format is independent of any outer compression.

### `link_fragments/` array schema

Identical structure, parallel role for exactly one link array: the
**intra-chunk array at delta 0**, `links/0/<all-zero offsets>/`.

| Property | Value |
|----------|-------|
| zarr `node_type` | `array` |
| dtype | `uint8` |
| logical shape | one blob per chunk |
| codec pipeline | none |
| group metadata | `{"zv_array": "link_fragments", "encoding": "fragment_index_v1"}` |

`link_fragments/<chunk_coords>` describes the fragment partition of
`links/0/<all-zero offsets>/<chunk_coords>` rows. Present iff the geometry
type has connectivity (polyline / streamline / graph / skeleton / mesh) and
at least one chunk has been written.

Every other link array — any non-zero offsets segment, and every
`delta != 0` — uses an inline self-describing ragged blob and has **no**
`link_fragments/` sibling. The writer picks between the two on exactly one
condition, `delta == 0 and is_intra(offsets)`; see
[Links](../object_model/links.md#cell-encoding-two-branches-one-condition).

```{important}
**Why only that one array may write it.** `link_fragments/<chunk>` is keyed
by **chunk alone** — the key carries no `<delta>` and no `<offsets>`
segment — and the write is an unconditional replace. The path therefore
admits exactly **one** writer per chunk.

That is safe only because the enclosing branch condition allows exactly one
array to reach it: the intra array at delta 0. A chunk is the source cell
of at most one such array, since "delta 0, all offsets zero" names a single
array per family. If a second offsets array — say `links/0/0.0.+1/` — ever
wrote this sidecar for the same source chunk, it would **silently clobber**
the intra array's fragment index: no error, no key collision to detect,
just an index describing the wrong array's rows. The intra cell would then
decode its flat blob against another array's group boundaries and return
wrong-length link groups.

The branch condition in `write_chunk_links` and this write must therefore
stay together. Widening the condition without re-keying the sidecar (to
include the delta and offsets) reintroduces the clobber.
```

Note also that link groups need **not** be 1:1 with the chunk's vertex
fragments. A chunk may legitimately have many vertex fragments but few link
groups — BRIDGE, for instance, stores streamlines as vertex fragments and
its node graph as link fragments. Readers derive per-group link ranges from
`link_fragments/`, never from `vertex_fragments/`, so no write-time 1:1
guard applies.

### Write-time invariants

A writer SHALL maintain:

1. **Magic and version.** Every blob begins with `'ZVFG'`
   (`0x5A56_4647`) followed by `version = 1`.
2. **Popcount agreement.** The header's `num_range_fragments` equals
   the popcount of the range bitmap over the first `F` bits.
3. **Bitmap padding.** Bytes beyond `ceil(F/8)` in the range bitmap
   region are zero.
4. **Range table order.** Row `r` of the range table corresponds to
   the `r`-th set bit of the bitmap (scan order `f = 0, 1, …`).
5. **CSR monotonicity.** `explicit_offsets[0] == 0` and
   `explicit_offsets[i+1] >= explicit_offsets[i]`.
6. **Index bounds.** Range fragments' `[start, start + count)` and
   explicit fragments' indices lie in `[0, len(vertices[chunk_coords]))`.
7. **Non-negative explicit indices.** Every entry of
   `explicit_indices` is `>= 0`.

At level 0, the default writer additionally emits one range fragment
per non-empty bin in ascending bin-flat-index order, preserving the
pre-0.6 fragment order convention. At coarsened levels, fragment order
follows the metavertex emission order chosen by the coarsening
pipeline.

### Worked example

A chunk holds three fragments:

- **Fragment 0** is a range starting at row 0, count 4 — the
  vertices of one non-empty bin at level 0.
- **Fragment 1** is explicit, with rows `[12, 7, 19]` — a shared
  metavertex emitted by the coarsening pipeline that references
  rows of `vertices/<chunk>` not contiguous in bin order.
- **Fragment 2** is a range starting at row 20, count 8 — another
  non-empty bin at level 0.

So `F = 3`, `R = 2`, `E = 1`, `T = 3`. Annotated hex dump:

```
offset  bytes                                       meaning
------  ------------------------------------------  --------------------
0x00    47 46 56 5A                                 magic 'ZVFG'
0x04    01 00                                       version = 1
0x06    00 00                                       flags = 0
0x08    03 00 00 00                                 F = 3
0x0C    02 00 00 00                                 R = 2
0x10    05 00 00 00 00 00 00 00                     bitmap (bits 0,2 set;
                                                    padded to 8 bytes)
0x18    00 00 00 00 00 00 00 00                     range[0].start = 0
0x20    04 00 00 00 00 00 00 00                     range[0].count = 4
0x28    14 00 00 00 00 00 00 00                     range[1].start = 20
0x30    08 00 00 00 00 00 00 00                     range[1].count = 8
0x38    00 00 00 00                                 csr_offsets[0] = 0
0x3C    03 00 00 00                                 csr_offsets[1] = 3
0x40    0C 00 00 00 00 00 00 00                     csr_indices[0] = 12
0x48    07 00 00 00 00 00 00 00                     csr_indices[1] = 7
0x50    13 00 00 00 00 00 00 00                     csr_indices[2] = 19
```

Total blob size: 88 bytes.

Walking through `fidx.indices(1)`:

1. `is_range(1)` reads bit 1 of byte 0 of the bitmap: `0x05 >> 1 & 1
   = 0` → explicit.
2. The lazy prefix-popcount yields `prefix = [0, 1, 1, 2]`.
3. `e = 1 - prefix[1] = 1 - 1 = 0` — fragment 1 is the *zeroth*
   explicit fragment.
4. `a, b = csr_offsets[0], csr_offsets[1] = 0, 3`.
5. Return `csr_indices[0:3] = [12, 7, 19]`. Done.
