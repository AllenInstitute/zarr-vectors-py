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
  Encoded as the per-object manifest, one row of `object_index/manifests`
  (see [Object manifest](object_manifest.md)).

---

## Introduction

Fragments are the bridge between the chunk-level I/O of Zarr and the
bin-level spatial queries of Zarr Vectors. A small binary **fragment index**
sits alongside each chunk's vertex payload, naming the row ranges
that belong to each fragment.

What a fragment *is* depends on the geometry type and writer — the format
only says that a fragment is a set of rows of the chunk's `vertices` cell:

| Writer | Fragments in a chunk |
|--------|----------------------|
| `write_points`, no objects | one fragment per **bin**, in row-major bin order — including empty bins, which are zero-count ranges |
| `write_points` with `object_ids` / objects / groups | one fragment per **object** in the chunk, ordered by object id |
| `write_polylines`, `write_lines`, `write_graph`, `write_mesh` | one fragment per object per chunk (a polyline split by an attribute bin contributes one per piece) |
| coarsened level, `shared_fragments=True` | one fragment per distinct **metavertex** in the chunk; several objects' manifests may name it |

So "bins ↔ fragments one-to-one" holds only for undifferentiated point clouds
(the case `bin_shape` is designed for; see [Bin shape](../chunking/bin_shape.md)).
For every other type `bin_shape` is stored as `base_bin_shape` but the writers
split at chunk boundaries only.

This page describes the point-cloud bin write path, the fragment addressing
scheme, the read path, and what changes at coarsened levels.

---

## Technical reference

### Fragment creation during write (point cloud, no objects)

When a point-cloud chunk is written, its vertices are sorted into bin order
and **one range fragment is emitted for every bin of the chunk**, empty or not.
The fragment index therefore *is* the row-major (C-order) bin index within the
chunk, and a chunk has exactly `B_per_chunk` fragments:

```python
def build_pointcloud_fragments(positions, chunk_coord, chunk_shape, bin_shape):
    """Fragment list for one chunk's vertices (undifferentiated point cloud)."""
    bpc_shape = tuple(int(round(c / b)) for c, b in zip(chunk_shape, bin_shape))
    n_bins    = int(np.prod(bpc_shape))

    # 1. Row-major bin index of each vertex within the chunk
    #    (zarr_vectors.spatial.chunking.bin_to_fragment_index)
    bin_coords = np.floor(positions / np.array(bin_shape)).astype(int)
    local      = bin_coords - np.array(chunk_coord) * np.array(bpc_shape)
    frag_idx   = np.ravel_multi_index(local.T, bpc_shape)

    # 2. Sort vertices into bin order (stable within a bin)
    order      = np.argsort(frag_idx, kind="stable")

    # 3. One (start, count) range per bin; count == 0 for an empty bin
    counts     = np.bincount(frag_idx, minlength=n_bins)
    starts     = np.cumsum(counts) - counts
    fragments  = list(zip(starts.tolist(), counts.tolist()))

    return order, encode_fragments(fragments)
```

A worked example: `chunk_shape = (200, 200, 200)`, `bin_shape = (100, 100, 200)`
gives `bpc_shape = (2, 2, 1)` and four fragments per chunk. If the chunk holds
3 points in bin `(0,0,0)`, 2 in bin `(1,0,0)` and none elsewhere, its fragment
index is `[(0, 3), (3, 0), (3, 2), (5, 0)]` — fragment 2 is bin `(1, 0, 0)`.

The `order` permutation is applied to attribute arrays in the same way:
`sorted_attributes = attributes[order]`. Attribute row `k` always corresponds
to vertex row `k` in the chunk.

Consequences worth knowing:

- Empty bins **do** occupy a fragment slot (a 16-byte `(start, 0)` range
  entry), so a point-cloud chunk's fragment index has a fixed size of
  `16 + 8 + 16·B_per_chunk + 4` bytes (header, bitmap rounded up to one
  8-byte word for `B ≤ 64`, range table, one CSR offset). The pre-0.6
  `(0, 0)` / `(-1, -1)` filler rows are gone, but the slot per bin is not.
- `rebin_level` is different: it rewrites a chunk with one fragment per
  **occupied** bin only, sorted by bin coordinate, so after a rebin fragment
  `f` is no longer the bin with linear index `f`. Do not derive a bin from a
  fragment index without checking the store was not rebinned.
- For stores written with objects (or any non-point type) there is no
  bin ↔ fragment relationship at all.

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
from zarr_vectors.core.arrays import read_fragment

vertices = read_fragment(level_group, chunk_coords, fragment_index,
                         dtype="float32", ndim=3)          # (n, 3)
```

which is, in outline:

```python
fidx     = read_fragment_index(level_group, VERTEX_FRAGMENTS, chunk_coords)
rows     = fidx.indices(fragment_index)
cell     = np.frombuffer(read_bytes("vertices", chunk_coords), dtype).reshape(-1, ndim)
vertices = cell[rows, :]
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
skeleton, mesh), `object_index/manifests` stores each object's full
manifest — the ordered list of `(chunk_coords, fragment_ref)` blocks
that span the object. The first block names the object's primary
fragment:

- For **polylines and streamlines**, blocks are in traversal order.
  A single chunk's contribution may be one fragment (mode 0), a
  contiguous range of fragments (mode 1), or an arbitrary list (mode 2).
  The core writers currently emit mode 0 only; the other modes are format
  options readers must accept.
- For **graphs and skeletons**, block order is implementation-defined;
  connectivity is recovered from the `links/<delta>/<offsets>/` family
  (intra-chunk edges and boundary-crossing edges alike — they differ
  only by which offsets array holds them).

### Fragments across multiple chunks

Reading a multi-chunk object is a single decode of the object's
manifest blob followed by parallel chunk reads — no link-walk loop:

```python
from zarr_vectors.core.arrays import read_object_vertices

def read_object(level_group, object_id):
    fragments = read_object_vertices(level_group, object_id, dtype="float32", ndim=3)
    return np.concatenate(fragments, axis=0)    # manifest order
```

Pre-0.6 stores required walking `cross_chunk_links/` forward from
each chunk to discover the next chunk; the walk was a sequence of
dependent reads. Post-0.6 the manifest enumerates every chunk
directly. The links family still encodes geometric edges across
chunks — as records with non-zero offsets in `links/0/<offsets>/` —
but is not used for chunk discovery during object reads.

### Fragment count at coarser levels

For a level with `shared_fragments=False`, a fragment is whatever the writer
that produced the level emitted (above): for an undifferentiated point cloud
`F == B_per_chunk` at *that level's* bin shape, which shrinks as the bin
grows:

```
Level 0: bpc_shape = (4, 4, 4) → F per chunk = 64
Level 1: bpc_shape = (2, 2, 2) → F per chunk = 8   (bin 2x the base)
Level 2: bpc_shape = (1, 1, 1) → F per chunk = 1   (bin 4x the base)
```

assuming the chunk shape is unchanged. A level may carry its own coarser
`chunk_shape` (an integer multiple of the root's), in which case the bins per
chunk are `level_chunk_shape / level_bin_shape`.

For levels with `shared_fragments=True` — every level `coarsen_level` /
`build_pyramid` writes — `F` equals the number of distinct metavertices in the
chunk after coarsening; there is no fixed relationship to `B_per_chunk`. `F`
is typically smaller than `B_per_chunk` but can in principle exceed it when
many objects converge on non-aligned positions inside one bin.

### Invariants maintained by the writer

A writer SHALL preserve:

1. **Row alignment.** Attribute row `k` corresponds to vertex row `k` of the
   same cell. For an undifferentiated point cloud the vertices are
   additionally stored in ascending bin-flat-index order.
2. **Fragment-blob well-formedness.** Every chunk's `vertex_fragments/`
   blob conforms to the v1 byte layout — magic `'ZVFG'`, version 1,
   `R == popcount(bitmap[0:F])`, monotone CSR offsets. See
   [Fragment-index arrays](../layout/fragment_index_arrays.md) §"Write-time
   invariants".
3. **Index bounds.** Every range fragment's `[start, start + count)`
   and every explicit fragment's indices lie in `[0, len(vertices[chunk_coords]))`.
4. **Fragment order (undifferentiated point clouds).** Fragments are
   emitted in ascending bin-flat-index order, one per bin including empty
   ones, so fragment `f` *is* bin `f` and no separate sidecar is needed —
   until `rebin_level` has been run (see above).
5. **Attribute row alignment.** `vertex_attributes/<name>` cell `c` is
   row-aligned with `vertices` cell `c`. Attribute access for fragment `f`
   goes via `attributes[chunk][fidx.indices(f)]`; no separate attribute index
   is maintained.
6. **Manifest consistency.** The manifest in `object_index/manifests` for
   object `oid` names only fragments that exist in the named chunks'
   fragment-index blobs.
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
| Object manifest reconstructed via `cross_chunk_links/` walk | Object manifest enumerated directly in `object_index/manifests` |
