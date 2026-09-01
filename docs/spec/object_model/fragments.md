# Fragments

## Terms

**Fragment**
: The on-disk unit that packages rows of `vertices/<chunk>` for
  indexing. Each fragment is one entry in the chunk's fragment-index
  blob and is either a contiguous range of rows or an explicit list
  of row indices. See [Fragment-index arrays](../layout/fragment_index_arrays.md)
  for the byte layout.

**Fragment address**
: The pair `(chunk_coords, fragment_index)` that uniquely identifies
  a fragment in one level of a store. Chunk coordinates are a D-tuple
  (not a flat ravel), and `fragment_index` is chunk-local — it indexes
  the chunk's `vertex_fragments/<chunk_coords>` blob only.

**Primary fragment**
: The first fragment of an object in traversal order (for sequential
  types) or the fragment containing the root vertex (for tree types).
  Useful for preview rendering; no longer the only addressable entry
  in `object_index/`.

**Object fragment list**
: The ordered list of fragment addresses belonging to one object.
  Encoded as the per-object manifest in `object_index/data` (see
  [Object manifest](object_manifest.md)).

---

## Introduction

Fragments are the bridge between the chunk-level I/O of Zarr and the
bin-level spatial queries of Zarr Vectors. A small binary **fragment index**
sits alongside each chunk's vertex payload, naming the row ranges
that belong to each fragment.

At level 0 with the default writer, fragments line up with bins
one-to-one: each non-empty bin emits exactly one range fragment.
The coarsening pipeline relaxes that correspondence — at coarsened
pyramid levels with `LevelMetadata.shared_fragments=True` a single
fragment may represent a metavertex referenced by several objects'
manifests, and the bin model is supplanted by the fragment as the
addressing primitive.

This page describes the level-0 write path (where fragments and bins
align one-to-one), the fragment addressing scheme, the read path, and
what changes at coarsened levels.

---

## Technical reference

### Fragment creation during write (level 0)

When a chunk is written, its vertices are first sorted into bin
order; then one range fragment is emitted per non-empty bin:

```python
def build_level0_fragments(positions, chunk_coord, chunk_shape, bin_shape):
    """Compute fragment list for one chunk's vertices."""
    D          = len(chunk_shape)
    bpc_shape  = tuple(int(c / b) for c, b in zip(chunk_shape, bin_shape))

    # 1. Compute bin flat index for each vertex
    local      = positions - np.array(chunk_coord) * np.array(chunk_shape)
    bin_coords = np.floor(local / np.array(bin_shape)).astype(int)
    bin_flats  = np.ravel_multi_index(bin_coords.T, bpc_shape)

    # 2. Sort vertices into bin order (stable sort within each bin)
    order      = np.argsort(bin_flats, kind="stable")
    sorted_pos = positions[order]
    sorted_bin = bin_flats[order]

    # 3. Emit one range fragment per non-empty bin
    unique_bins, first_occ, counts = np.unique(
        sorted_bin, return_index=True, return_counts=True,
    )
    fragments = [
        (int(first_occ[i]), int(counts[i]))
        for i in range(len(unique_bins))
    ]

    # 4. Encode and write
    blob = encode_fragments(fragments)
    return sorted_pos, order, blob
```

The `order` permutation is applied to attribute arrays in the same
way: `sorted_attributes = attributes[order]`. Attribute row `k` always
corresponds to vertex row `k` in the chunk.

Empty bins do not contribute a fragment — the pre-0.6 `(0, 0)` and
`(-1, -1)` filler rows are gone. The total fragment count `F` in a
level-0 chunk equals the number of non-empty bins, bounded above by
`B_per_chunk`.

### Fragment addressing

A fragment is uniquely identified within one level of a store by:

1. **`chunk_coords`**: the D-tuple `(i, j, k, …)` naming the chunk.
   No flat ravel; the tuple is stored as `int64` words in the
   manifest block (see [Object manifest](object_manifest.md)).
2. **`fragment_index`**: an `int` in `[0, F)`, chunk-local. F may
   differ from chunk to chunk; readers learn F from the chunk's
   `vertex_fragments/` blob.

There is no global flat fragment index, and there is no need for one:
manifests reference fragments by their `(chunk_coords,
fragment_index)` pair, and the pair is decoded directly from the
manifest blob without a ravel-multi-index step.

### Fragment access pattern

Reading a specific fragment given its address:

```python
def read_fragment_vertices(level_group, chunk_coords, fragment_index):
    """Read the vertices of a single fragment."""
    fidx     = read_fragment_index(
        level_group, VERTEX_FRAGMENTS, chunk_coords,
    )
    rows     = fidx.indices(fragment_index)
    vertices = level_group.read_vertices(chunk_coords)
    return vertices[rows, :]
```

For range fragments, `fidx.indices(f)` materialises `arange(start,
start + count)`. For explicit fragments, it returns a copy of the
CSR slice; hot loops can use `fidx.indices_view(f)` for a zero-copy
view over the underlying bytes. See
[Fragment-index arrays](../layout/fragment_index_arrays.md) for the
decoder algorithm.

Because each chunk of `vertices/` is independently stored, reading
any fragment in a chunk reads the chunk's vertex payload in full;
the fragment index then selects rows from the in-memory array.
Subsequent fragment reads within the same chunk are served from
the in-process chunk cache.

### Fragments and the object model

For discrete-object geometry types (polyline, streamline, graph,
skeleton, mesh), `object_index/data` stores each object's full
manifest — the ordered list of `(chunk_coords, fragment_ref)` blocks
that span the object. The first block names the object's primary
fragment:

- For **polylines and streamlines**, blocks are in traversal order.
  A single chunk's contribution may be one fragment (mode 0), a
  contiguous range of adjacent-bin fragments (mode 1), or an
  arbitrary list (mode 2 — most common after coarsening).
- For **graphs and skeletons**, block order is implementation-defined;
  connectivity is recovered from the `links/<delta>/<offsets>/` family
  (intra-chunk edges and boundary-crossing edges alike — they differ
  only by which offsets array holds them).

### Fragments across multiple chunks

Reading a multi-chunk object is a single decode of the object's
manifest blob followed by parallel chunk reads — no link-walk loop:

```python
def read_object(level_group, object_id):
    blocks = decode_object_manifest(level_group, object_id)
    chunks = [
        read_chunk_fragments(level_group, coords, fragment_ref)
        for coords, fragment_ref in blocks
    ]
    return np.concatenate(chunks, axis=0)
```

Pre-0.6 stores required walking `cross_chunk_links/` forward from
each chunk to discover the next chunk; the walk was a sequence of
dependent reads. Post-0.6 the manifest enumerates every chunk
directly. The links family still encodes geometric edges across
chunks — as records with non-zero offsets in `links/0/<offsets>/` —
but is not used for chunk discovery during object reads.

### Fragment count at coarser levels

The intuition "`B_per_chunk` shrinks with `bin_ratio` until each
chunk is one fragment at the all-coarse level" is true only when the
default writer is in use *and* the level does not share fragments.

For levels with `shared_fragments=False`:

```
Level 0: bpc_shape = (4, 4, 4) → max F per chunk = 64
Level 1: bpc_shape = (2, 2, 2) → max F per chunk = 8   (bin_ratio = (2,2,2))
Level 2: bpc_shape = (1, 1, 1) → max F per chunk = 1   (bin_ratio = (4,4,4))
```

For levels with `shared_fragments=True`, `F` equals the number of
distinct metavertices in the chunk after coarsening — there is no
fixed relationship to `B_per_chunk`. `F` is typically smaller than
`B_per_chunk` (each metavertex represents many original vertices)
but can in principle exceed it when many objects converge on
non-aligned positions inside one bin.

### Invariants maintained by the writer

A writer SHALL preserve:

1. **Bin order at level 0.** Vertices within a chunk are stored in
   ascending bin-flat-index order. Attribute row `k` corresponds to
   vertex row `k`.
2. **Fragment-blob well-formedness.** Every chunk's `vertex_fragments/`
   blob conforms to the v1 byte layout — magic `'ZVFG'`, version 1,
   `R == popcount(bitmap[0:F])`, monotone CSR offsets. See
   [Fragment-index arrays](../layout/fragment_index_arrays.md) §"Write-time
   invariants".
3. **Index bounds.** Every range fragment's `[start, start + count)`
   and every explicit fragment's indices lie in `[0, len(vertices[chunk_coords]))`.
4. **Fragment order at level 0 (default writer).** Fragments are
   emitted in ascending bin-flat-index order, so readers that need
   a bin ↔ fragment map at level 0 can derive it from the writer's
   emission order without a separate sidecar.
5. **Attribute row alignment.** Attribute arrays for chunk `c` are
   row-aligned with `vertices/<c>`. Attribute access for fragment
   `f` goes via `attributes[chunk][fidx.indices(f)]`; no separate
   attribute index is maintained.
6. **Manifest consistency.** The manifest in `object_index/data`
   for object `oid` names only fragments that exist in the named
   chunks' fragment-index blobs.
7. **Sharing discipline.** When `LevelMetadata.shared_fragments=False`,
   no fragment is named by more than one object's manifest. When
   `True`, sharing is permitted and readers MUST NOT assume
   uniqueness.

### Pre-0.6 ↔ post-0.6 cross-reference

The pre-0.6 layout used the term **fragment (fragment)** for a
`(chunk_flat, bin_flat)` pair backed by a `(B, 2)` int64 table
named `vertex_group_offsets/`. The v0.6 fragment is the post-rename
equivalent on disk; users migrating off pre-0.6 stores may find this
mapping useful:

| Pre-0.6 term | Post-0.6 equivalent |
|--------------|---------------------|
| fragment address `(chunk_flat, bin_flat)` | Fragment address `(chunk_coords, fragment_index)` |
| `vertex_group_offsets[chunk][b, 0]` (`offset`) | `range_table[r, 0]` (`start`) for the range fragment representing bin `b` |
| `vertex_group_offsets[chunk][b, 1]` (`count`) | `range_table[r, 1]` (`count`) |
| Bin with `count == 0` | No fragment emitted for that bin |
| Bin with `count == -1` (fill) | No fragment emitted for that bin |
| One row per bin per chunk (fixed shape) | One fragment per non-empty bin (variable shape) |
| `(B_per_chunk × 16-byte)` empty-bin overhead | Zero overhead for empty bins |
| n/a | Explicit fragment (mode-2; arbitrary index list) |
| n/a | Shared fragment (named by ≥ 2 manifests) |
| Object manifest reconstructed via `cross_chunk_links/` walk | Object manifest enumerated directly in `object_index/data` |
