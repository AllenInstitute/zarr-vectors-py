# Object manifest

## Terms

**Object manifest**
: The ordered list of fragments that make up one object at one level,
  each named by `(chunk_coords, fragment_index)`. Reading the fragments
  in manifest order gives the object's vertices in order.

**Manifest block**
: One manifest entry: a chunk's coordinates and one or more fragments in
  that chunk.

**Object index**
: A level's `object_index/` group, holding every object's manifest and
  the object ids.

**Object row**
: An object's position in the object index. Object attributes are
  aligned to it (see [Object attributes](object_attributes.md)).

---

## Introduction

The object index maps an object id to the chunks and fragments that hold
it, so a reader needs one manifest lookup and one read per chunk.
Fragment references are local to their chunk, so chunks can be written
independently. Links between vertices are stored separately
([Links](links.md)).

---

## Technical reference

### Layouts

| `layout` | Arrays under `object_index/` | Row `k` holds |
|----------|------------------------------|---------------|
| `vlen_manifests_v2` | `manifests`, `object_ids` | id `object_ids[k]` |
| `vlen_manifests_v1` | `manifests` | object `k` |
| `dense_manifests_v1` | `manifest_spans`, `manifest_blocks`, `object_ids` | id `object_ids[k]` |

A reader dispatches on the `layout` attribute of `object_index/`. A
missing or unknown `layout` raises `ArrayError`.

zarr-vectors-py writes `vlen_manifests_v1` from `reserve_object_rows`
and from `zarr_vectors.core.streaming.ObjectIndexAppender`. Otherwise
the layout is, in order of precedence:

1. the writer's `layout=` argument (`"vlen"` or `"dense"`), for example
   on `write_object_index` or `write_object_manifests`;
2. the layout of the existing index, which a rewrite keeps;
3. the root's `manifest_layout`: `"dense"` gives the dense layout;
   absent gives `vlen_manifests_v2`.

| `create_store` argument | Root attributes written | Effect |
|-------------------------|-------------------------|--------|
| `manifest_layout="dense"` | `manifest_layout: "dense"`; `dense_manifests` in `format_capabilities` | New indexes are dense |
| `manifest_block_dtype="int32"` | `manifest_block_dtype: "int32"` | New dense indexes use `int32` blocks; `block_dtype=` on a writer overrides it |

Each key is written only with its non-default value. Writing a dense
index also adds `dense_manifests`. `zarr_vectors.building.store_layout(root)`
reports the level-0 `object_index` layout.

### `object_index/` attributes

A `vlen_manifests_v2` index for ids 5, 9 and 12, where 12 has an empty
manifest:

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
| `num_present` | Rows with a non-empty manifest |
| `object_ids_sorted` | `true` when `object_ids` is ascending |
| `block_dtype` | Dense layout only, and only when `"int32"` |

### Object ids and absent objects

`object_ids` is `int64`, shape `(num_objects,)`. Ids are non-negative
and need not be contiguous. An object with an empty manifest is absent
from the level. A [sparsified](../multiscale/sparsity.md) level keeps a
row for every id of its source level and empties the rows it drops.
`read_object_manifest` raises `ArrayError` for an id with no row.

### Vlen layouts

`manifests` is a `variable_length_bytes` array (codec `vlen-bytes`),
shape `(num_objects,)`; cell `k` is row `k`'s manifest blob. An unwritten
row reads as empty. Both arrays are chunked at 16,384 rows.

```{warning}
Zarr v3 has no finished specification for variable-length byte arrays
(see
[zarr-extensions](https://github.com/zarr-developers/zarr-extensions/tree/main/data-types)).
`vlen-bytes` stores may need re-encoding if the specification settles
on an incompatible encoding.
```

Manifest blob, all integers little-endian:

```
uint32 num_blocks B

For each of the B blocks:
  int64  chunk_coords[sid_ndim]
  uint8  mode
  if mode == 0:   # single fragment
    int64 fragment_index
  elif mode == 1: # fragments start .. start + count - 1
    int64 start, int64 count
  elif mode == 2: # explicit list
    uint32 count
    int64  fragment_indices[count]
```

An empty manifest is 4 bytes (`B = 0`). Every index is non-negative.

`zarr_vectors.encoding.fragments.encode_object_manifest_blocks(blocks,
sid_ndim, *, force_explicit=False)` encodes `(chunk_coords,
fragment_ref)` pairs:

| `fragment_ref` | Mode |
|----------------|------|
| `int` | 0 |
| `(start, count)` tuple | 1 |
| 1-D integer array of consecutive indices | 1 (2 with `force_explicit=True`) |
| any other 1-D integer array | 2 |

`decode_object_manifest_blocks(raw, sid_ndim)` returns the same three
reference types. The `write_*` functions write one mode-0 block per
fragment.

### Dense layout

| Array | Type | Contents |
|-------|------|----------|
| `manifest_spans` | `int64`, `(num_objects, 2)` | Row `k` is `(start, count)`: rows `start` to `start + count - 1` of `manifest_blocks` |
| `manifest_blocks` | `int64` or `int32`, `(n_blocks, sid_ndim + 1)` | One row per fragment: chunk coordinates, then fragment index |
| `object_ids` | `int64`, `(num_objects,)` | As in `vlen_manifests_v2` |

An empty manifest has `count == 0`; its `start` is not read. Block rows
need not be contiguous or in object order: a patch appends new blocks
and rewrites only the span, leaving the old rows unreferenced.

zarr-vectors-py chunks spans and ids at 16,384 rows and blocks at 65,536
rows. It compresses all three with `zstd` level 9, unless the session
uses another codec (a lower `zstd` level is raised to 9). An index keeps
its chunking, codecs and dtype until rewritten in full.

For an `int32` index, `object_index/` carries `block_dtype: "int32"` for
information; the array's `data_type` is authoritative. Readers widen
blocks to `int64`. Writing a value outside `[-2**31, 2**31 - 1]` to an
`int32` index raises `ArrayError`.

A reader gets every manifest at once by gathering `manifest_blocks` rows
with the spans.

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

`read_object_manifest` returns one `(chunk_coords, fragment_index)`
pair per fragment in every layout. `read_object_vertices` reads each
chunk once.

Each level has its own object index naming that level's fragments. At
a level with `shared_fragments: true`, different objects may name the
same fragment (see [Shared fragments](fragments.md#shared-fragments)).

### Validation

**L1** reports whether a level has `object_index/`.

**L3**, for each level:

| Check | Applies to |
|-------|------------|
| `manifest_spans`, `manifest_blocks` and `object_ids` exist; `manifests` does not | dense |
| `manifest_spans` is `int64` with at least `num_objects` rows of 2 | dense |
| `manifest_blocks` is `int64` or `int32`, matches `block_dtype` where stated, and has `sid_ndim + 1` columns | dense |
| No span has a negative count or reaches past the end of `manifest_blocks` | dense |
| Every manifest entry names a chunk with vertices and a fragment index below its fragment count | all layouts |
