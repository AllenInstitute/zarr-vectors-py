# Level groups

This page specifies the group that holds one resolution level, `<N>/`, and
every key of its attributes.

## Terms

**Level group**
: The Zarr group `<N>/` at the store root holding one resolution level's
  arrays. `N` is the **level index**: `0` is full resolution, and larger
  indices are coarser.

**Level attributes**
: The `attributes` of the level group's `zarr.json`. Zarr Vectors keys are
  under `zarr_vectors_level`; `LevelMetadata` in
  `zarr_vectors.core.metadata` is the in-memory form.

## Introduction

A level's attributes say how many vertices it holds, which arrays it has,
and how it was made from its parent level. Its `bin_ratio` and `bin_shape`
are in the root `multiscales` block instead (see
[Multiscale metadata](../multiscale/multiscale_metadata.md));
`read_level_metadata` rebuilds both.

## Technical reference

### Example

The `zarr.json` of level 1 of a streamline pyramid built with
`build_pyramid(..., factors=[(2, 2), (2, 2)])` from 20 streamlines:

```javascript
{
  "attributes": {
    "zarr_vectors_level": {
      "level": 1,
      "object_sparsity": 0.5,
      "vertex_count": 23,
      "coarsening_method": "per_object",
      "parent_level": 0,
      "arrays_present": ["vertices", "object_index"],
      "preserves_object_ids": true,
      "inherited_num_objects": 20,
      "shared_fragments": true
    }
  },
  ...
}
```

Level 0 of the same store has `"level": 0`, `"object_sparsity": 1.0`,
`"coarsening_method": "none"`, `"parent_level": null` and
`"fragments_tile": true`.

### `zarr_vectors_level` keys

Always written:

| Key | Type | Meaning |
|-----|------|---------|
| `level` | integer | The level index, equal to the group name |
| `vertex_count` | integer | Vertices at this level |
| `arrays_present` | list of strings | Array families present (`vertices`, `links`, `object_index`, …), never `<family>/<name>` |
| `object_sparsity` | number in `(0, 1]` | Fraction of the parent level's objects kept; `1.0` at level 0. Default `1.0` |
| `coarsening_method` | string | `none` at level 0; `per_object` from `build_pyramid`; `manual` from `add_resolution_level`. Default `"none"` |
| `parent_level` | integer or `null` | Level this one was made from; `null` at level 0 and required above it. Default `null` |

Written only when set:

| Key | Type | Meaning |
|-----|------|---------|
| `chunk_shape` | list of numbers | This level's chunk extent, when not the root's; a whole multiple of it |
| `chunk_dims`, `chunk_attribute_name`, `chunk_attribute_values` | list, string, list | Chunk-key axis names, the vertex attribute forming the leading axis, and one label per bin of it; see [Attribute chunking](../chunking/attribute_chunking.md) |
| `chunk_sources` | list of objects | The stores each value of a leading `source` axis reads from; see [Chunk sources](../chunking/chunk_sources.md) |
| `preserves_object_ids` | `true` | Dropped objects keep their ID as an empty manifest, so IDs match across levels |
| `inherited_num_objects` | integer | Size of the object ID space taken from the parent level |
| `shared_fragments` | `true` | A fragment may be named by more than one object's manifest |
| `fragments_tile` | `true` | Each chunk's vertex fragments are ranges covering the `vertices/` cell in order, so the cell reads without its fragment index |
| `link_fragments_aligned` | `true` | Link fragment `k` of each intra-chunk link cell holds exactly the records whose first endpoint is in vertex fragment `k` |
| `octree_chunks` | object | The level has [octree chunks](../chunking/octree_chunks.md) |

The bulk writers set `fragments_tile` and `link_fragments_aligned`. Any
later write to the arrays they describe removes them. Absent means unknown,
not false.

### Build-time keys

`zarr_vectors_presence` sits beside `zarr_vectors_level`, only while the
level is being written. Its value is `"deferred"`: the level's per-chunk
arrays carry no `nonempty_chunks`, so concurrent writers share no state,
and a reader takes presence from the store (see
[Chunk arrays](chunk_arrays.md)). `zarr_vectors.building.defer_presence`
sets it; `zarr_vectors.building.rebuild_presence` writes every array's
`nonempty_chunks` and removes it.

### Adding and removing levels

Level groups are named `0`, `1`, `2`, … with no prefix.
`list_resolution_levels(root)` returns the level indices in ascending
order.

```python
from zarr_vectors.core.store import (
    open_store, list_resolution_levels, add_resolution_level,
    remove_resolution_level, read_level_metadata,
)

root = open_store("tracts.zarrvectors", mode="r+")
add_resolution_level(root, level_index=1, bin_ratio=(2, 2, 2))
list_resolution_levels(root)             # [0, 1]
read_level_metadata(root, 1).bin_shape   # (100.0, 100.0, 100.0)
remove_resolution_level(root, level_index=1)
```

`add_resolution_level` writes an empty level (`vertex_count` 0,
`arrays_present` `[]`, `coarsening_method` `"manual"`) and its
`multiscales` entry, but no arrays. Level 0 cannot be removed. To build
populated levels, use `build_pyramid`; see
[Pyramid construction](../multiscale/pyramid_construction.md).

### Validation

Writing a level runs `LevelMetadata.validate()`. It requires non-negative
`level` and `vertex_count`, `object_sparsity` in `(0, 1]`, and positive
`chunk_shape` entries. Level 0 has no `bin_shape` or `parent_level`; higher
levels need both, with `bin_ratio` entries of at least 1. The
attribute-chunking and `chunk_sources` keys must agree. The writer also
checks that `chunk_shape` is a whole multiple of the root's and that the
level's bins divide it.

| Level | Check |
|-------|-------|
| [L1](../validation/l1_structural.md) | Error with no level groups, or a level without `vertices/`; warning on empty level attributes |
| [L2](../validation/l2_metadata.md) | Parses each level with `read_level_metadata`; repeats the chunk-shape check; warns when presence is deferred |
| [L3](../validation/l3_consistency.md) | Checks `fragments_tile` and `link_fragments_aligned` against the data |
