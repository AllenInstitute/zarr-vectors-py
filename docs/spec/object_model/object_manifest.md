# Object manifest

## Terms

**Object manifest**
: The ordered list of fragments that make up one object at one level,
  each named by `(chunk_coords, fragment_index)`. Reading the fragments
  in manifest order gives the object's vertices in order.

**Manifest block**
: One entry of a manifest: a chunk's coordinates and a reference to
  one or more fragments in that chunk. An object that leaves a chunk and
  returns has more than one block for it.

**Object index**
: The `object_index/` group of a level, which holds every object's
  manifest and the object ids.

**Object row**
: The position of an object in the object index. Object attributes are
  aligned to it (see [Object attributes](object_attributes.md)).

**Shared fragment**
: A fragment named by more than one object's manifest; see
  [Fragment-index arrays](../layout/fragment_index_arrays.md).

---

## Introduction

The object index turns an object id into a read plan: the manifest names
every chunk and fragment the object occupies, so an object is read with
one manifest lookup and one read per chunk, without scanning chunks or
following links. Fragment references are local to their chunk, so chunks
can be written without coordinating fragment numbering. Links between an
object's vertices are stored separately (see [Links](links.md)) and are
not needed to find them.

The object index has three layouts. This page specifies each, how a
reader tells them apart, the manifest encoding, and the checks the
validators make.

---

## Technical reference

### Layouts

| `layout` | Arrays under `object_index/` | Object ids |
|----------|------------------------------|------------|
| `vlen_manifests_v2` | `manifests`, `object_ids` | Row `k` holds the object with id `object_ids[k]` |
| `vlen_manifests_v1` | `manifests` | Row `k` holds object `k` |
| `dense_manifests_v1` | `manifest_spans`, `manifest_blocks`, `object_ids` | Row `k` holds the object with id `object_ids[k]` |

A reader dispatches on the `layout` attribute of `object_index/`. An
index with no `layout`, or one the reader does not know, is not read:
zarr-vectors-py raises `ArrayError`.

zarr-vectors-py writes `vlen_manifests_v2` for a new index unless the
root attributes say `manifest_layout: "dense"` or the writer is asked
for `layout="dense"` (see [Choosing the layout](#choosing-the-layout)).
It writes `vlen_manifests_v1` when rows are reserved in advance for
parallel writers (`reserve_object_rows`) and from the streaming writer
`zarr_vectors.core.streaming.ObjectIndexAppender`, where row `k` is
object `k` by construction.

### `object_index/` attributes

The attributes of a `vlen_manifests_v2` index written for three objects
with ids 5, 9 and 12, of which object 12 has an empty manifest:

```json
{
  "zv_array": "object_index",
  "num_objects": 3,
  "num_present": 2,
  "sid_ndim": 3,
  "layout": "vlen_manifests_v2",
  "object_ids_sorted": true
}
```

| Key | Meaning |
|-----|---------|
| `zv_array` | `"object_index"` |
| `layout` | One of the layouts above |
| `sid_ndim` | Number of chunk coordinates in a block |
| `num_objects` | Number of object rows |
| `num_present` | Number of rows with a non-empty manifest |
| `object_ids_sorted` | `true` when `object_ids` is in ascending order |
| `block_dtype` | Dense layout only, and only when `"int32"` (see [Dense layout](#dense-layout)) |

### Object ids and absent objects

`object_ids` is an `int64` array of shape `(num_objects,)`. Ids are
non-negative and need not be contiguous. An object with an empty
manifest is absent from the level: a level built by
[sparsification](../multiscale/sparsity.md) keeps a row for every id of
its source level and empties the rows of the objects it drops.
`read_object_manifest` raises `ArrayError` for an id with no row.

### Vlen layouts

`manifests` is a `variable_length_bytes` array (codec `vlen-bytes`) of
shape `(num_objects,)`; cell `k` is row `k`'s manifest blob. A row that
was never written reads as empty and counts as absent. zarr-vectors-py
chunks `manifests` and `object_ids` at 16,384 rows.

```{warning}
Zarr v3 has no finished specification for variable-length byte arrays
(see
[zarr-extensions](https://github.com/zarr-developers/zarr-extensions/tree/main/data-types)).
`vlen-bytes` stores may need re-encoding if the specification settles
on an incompatible encoding.
```

#### Manifest blob

All integers are little-endian.

```
uint32 num_blocks B

For each of the B blocks:
  int64  chunk_coords[sid_ndim]
  uint8  mode
  if mode == 0:   # single fragment
    int64 fragment_index
  elif mode == 1: # contiguous range of fragments
    int64 start, int64 count
  elif mode == 2: # explicit list
    uint32 count
    int64  fragment_indices[count]
```

An empty manifest is 4 bytes, `B = 0`. A range block names fragments
`start` to `start + count - 1`; every index is non-negative.

`zarr_vectors.encoding.fragments.encode_object_manifest_blocks(blocks,
sid_ndim, *, force_explicit=False)` encodes a list of
`(chunk_coords, fragment_ref)` pairs. An `int` gives mode 0, a
`(start, count)` tuple mode 1, and a 1-D integer array mode 1 when it is
a run of consecutive indices and mode 2 otherwise (always mode 2 with
`force_explicit=True`). `decode_object_manifest_blocks(raw, sid_ndim)`
returns the pairs with the same three reference types. The
`write_*` functions write one mode-0 block per fragment.

### Dense layout

The dense layout stores every manifest in fixed-width integer arrays:

| Array | Type | Contents |
|-------|------|----------|
| `manifest_spans` | `int64`, shape `(num_objects, 2)` | Row `k` is `(start, count)`: row `k`'s manifest is rows `start` to `start + count - 1` of `manifest_blocks` |
| `manifest_blocks` | `int64` or `int32`, shape `(n_blocks, sid_ndim + 1)` | One row per fragment: `sid_ndim` chunk coordinates, then the fragment index |
| `object_ids` | `int64`, shape `(num_objects,)` | As in `vlen_manifests_v2` |

An object's block rows, in order, are its manifest. An empty manifest
has `count == 0`, and its `start` is not read. Block rows need not be
contiguous or ordered by object: an append adds rows to both arrays, and
a patch appends an object's new blocks and rewrites only its span, so
the rows it replaced stay in `manifest_blocks`, unreferenced, until the
index is rewritten in full.

Each array's chunking and codecs are in its own `zarr.json`. zarr-vectors-py
chunks spans and ids at 16,384 rows and blocks at 65,536 rows, and
compresses all three with `zstd` at level 9: a write session that
compresses with `zstd` at a lower level is raised to 9, one with another
codec keeps it, and one with no compression gets `zstd` level 9. An
index keeps its chunking and codecs until it is rewritten in full.

`manifest_blocks` is `int32` only when the index was created so, for a
store whose chunk coordinates and fragment indices all fit. The array's
`data_type` states its dtype, and `object_index/` then carries
`block_dtype: "int32"` for information. A reader widens the blocks to
`int64`. An index keeps its dtype: appends and patches write into the
dtype it has, and zarr-vectors-py raises `ArrayError` before storing a
value outside `[-2**31, 2**31 - 1]` in an `int32` index. Spans and ids
are always `int64`.

Every manifest is read with a gather:

```python
spans  = object_index["manifest_spans"][:]            # (n, 2)
blocks = object_index["manifest_blocks"][:]           # (b, sid_ndim + 1)
counts = spans[:, 1]
offsets = np.concatenate([[0], np.cumsum(counts)])    # CSR over objects
rows = np.repeat(spans[:, 0], counts) + (
    np.arange(offsets[-1]) - np.repeat(offsets[:-1], counts))
chunk_coords, fragment_idx = blocks[rows, :-1], blocks[rows, -1]
```

### Choosing the layout

The layout of a new index is, in order of precedence:

1. the `layout=` argument (`"vlen"` or `"dense"`) of the writer, such as
   `write_object_index` or `write_object_manifests`;
2. the layout of the index already there, which a rewrite keeps;
3. the root's `manifest_layout`: `"dense"` gives the dense layout;
   absent means vlen.

`create_store(..., manifest_layout="dense")` writes
`manifest_layout: "dense"` to the root attributes and adds
`dense_manifests` to `format_capabilities`; writing a dense index adds
the capability as well. `create_store(..., manifest_block_dtype="int32")`
writes `manifest_block_dtype: "int32"`, which makes `int32` the default
for dense indexes created later. Each key is written only with its
non-default value. `block_dtype=` on the writer overrides it for one
index.

`zarr_vectors.building.store_layout(root)` reports a store's layout,
including the `object_index` layout of level 0.

### Reading an object

```python
from zarr_vectors.building import read_object_vertices
from zarr_vectors.core.arrays import read_object_manifest
from zarr_vectors.core.store import get_resolution_level, open_store

lg = get_resolution_level(open_store("tracts.zarrvectors"), 0)
print(read_object_manifest(lg, 0)[:3])
# e.g. [((1, 0, 1), 0), ((1, 1, 1), 0), ((1, 0, 1), 1)]
fragments = read_object_vertices(lg, 0)   # one (n_i, D) array per manifest entry
```

`read_object_manifest(level_group, object_id)` returns the manifest as
`(chunk_coords, fragment_index)` pairs in every layout, with range and
explicit blocks expanded to one pair per fragment.
`read_object_vertices` reads those fragments, prefetching each chunk
once.

### Shared fragments and coarser levels

Each level has its own object index, which names that level's
fragments. At a level with `shared_fragments: true`, the manifests of
different objects may name the same fragment; see
[Shared fragments](fragments.md#shared-fragments).

### Validation

**L1** reports whether a level has `object_index/`.

**L3**, for each level:

- A dense index has `manifest_spans`, `manifest_blocks` and `object_ids`,
  and no `manifests`; `manifest_spans` is `int64`; `manifest_blocks` is
  `int64` or `int32` and agrees with `block_dtype` where it is stated;
  `manifest_spans` has at least `num_objects` rows of 2;
  `manifest_blocks` has `sid_ndim + 1` columns; no span has a negative
  count, and no non-empty span reaches past the end of
  `manifest_blocks`.
- Every `(chunk_coords, fragment_index)` of every manifest names a chunk
  that holds vertices, and a fragment index below that chunk's fragment
  count.
