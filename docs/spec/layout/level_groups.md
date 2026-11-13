# Level groups

This page specifies the group that holds one resolution level, `<N>/`, and
every key of its attributes.

## Terms

**Level group**
: The Zarr group `<N>/` at the store root holding one resolution level's
  arrays. `N` is the **level index**: `0` is the full-resolution level, and
  larger indices are coarser.

**Level attributes**
: The `attributes` of the level group's `zarr.json`. Zarr Vectors keys are
  under `zarr_vectors_level`; `LevelMetadata` in
  `zarr_vectors.core.metadata` is the in-memory form.

## Introduction

A level group is a plain Zarr v3 group named by its index. Its attributes
say how many vertices it holds, which array families it has, and how it was
made from its parent level. A level's bin geometry (`bin_ratio`,
`bin_shape`) is not in its attributes: it is the level's entry in the root
`multiscales` block (see [Multiscale metadata](../multiscale/multiscale_metadata.md)),
and `read_level_metadata` rebuilds both from there.

## Technical reference

### Example

The `zarr.json` of level 1 of a streamline pyramid built with
`build_pyramid(..., factors=[(2, 2), (2, 2)])` from 20 streamlines:

```json
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
  "zarr_format": 3,
  "node_type": "group"
}
```

Level 0 of the same store:

```json
{
  "zarr_vectors_level": {
    "level": 0,
    "object_sparsity": 1.0,
    "vertex_count": 1200,
    "coarsening_method": "none",
    "parent_level": null,
    "arrays_present": ["vertices", "object_index"],
    "fragments_tile": true
  }
}
```

### `zarr_vectors_level` keys

Always written:

| Key | Type | Meaning |
|-----|------|---------|
| `level` | integer | The level index, equal to the group name |
| `vertex_count` | integer | Vertices at this level |
| `arrays_present` | list of strings | Array families at this level, by family name (`vertices`, `vertex_attributes`, `links`, `object_index`, …), never `<family>/<name>` |
| `object_sparsity` | number in `(0, 1]` | Fraction of the parent level's objects kept; `1.0` at level 0 |
| `coarsening_method` | string | `none` at level 0; `per_object` from `build_pyramid`; `manual` from `add_resolution_level` |
| `parent_level` | integer or `null` | Level this one was made from; `null` at level 0 and required above it |

Written only when set:

| Key | Type | Meaning |
|-----|------|---------|
| `chunk_shape` | list of numbers | This level's chunk extent, when it differs from the root's; each entry a whole multiple of the root's |
| `chunk_dims` | list of strings | Names of the chunk-key axes, leading non-spatial axis first, on a level [chunked by an attribute](../chunking/attribute_chunking.md) |
| `chunk_attribute_name` | string | The vertex attribute that forms the leading chunk axis |
| `chunk_attribute_values` | list | One label per bin of that axis, in bin order |
| `chunk_sources` | list of objects | The stores each value of a leading `source` axis reads from; see [Chunk sources](../chunking/chunk_sources.md) |
| `preserves_object_ids` | `true` | Dropped objects keep their ID as an empty manifest, so IDs match across levels |
| `inherited_num_objects` | integer | Size of the object ID space taken from the parent level |
| `shared_fragments` | `true` | A fragment may be named by more than one object's manifest |
| `fragments_tile` | `true` | In every chunk, the vertex fragments are ranges that cover the `vertices/` cell in order, so the cell can be read without its fragment index |
| `link_fragments_aligned` | `true` | In every chunk, link fragment `k` of the intra-chunk link array holds exactly the records whose first endpoint is in vertex fragment `k` |
| `octree_chunks` | object | The level has [octree chunks](../chunking/octree_chunks.md) |

`fragments_tile` and `link_fragments_aligned` are set by the bulk writers
after their chunk writes and removed by any later write to the arrays they
describe. Absent means unknown, not false.

A reader treats an absent `object_sparsity` as `1.0`, `coarsening_method`
as `"none"`, and `parent_level` as `null`.

### Build-time keys

One key sits at the top level of the level attributes, beside
`zarr_vectors_level`, and only while the level is being written:

| Key | Value | Meaning |
|-----|-------|---------|
| `zarr_vectors_presence` | `"deferred"` | The level's per-chunk arrays carry no `nonempty_chunks`, so concurrent writers share no state; a reader takes presence from the store (see [Chunk arrays](chunk_arrays.md)) |

`zarr_vectors.building.defer_presence` sets it, and
`zarr_vectors.building.rebuild_presence` writes every array's
`nonempty_chunks` and removes it.

### Naming and order

Level groups are named `0`, `1`, `2`, … with no prefix. Root children whose
names are not integers, such as `parametric/`, are not levels.
`list_resolution_levels(root)` returns the level indices in ascending
order.

### Adding and removing levels

```python
from zarr_vectors.core.store import (
    open_store, list_resolution_levels, list_available_ratios,
    add_resolution_level, remove_resolution_level, read_level_metadata,
)

root = open_store("tracts.zarrvectors", mode="r+")
list_available_ratios(root)          # [(1, 1, 1)]
add_resolution_level(root, level_index=1, bin_ratio=(2, 2, 2))
list_resolution_levels(root)         # [0, 1]
read_level_metadata(root, 1).bin_shape   # (100.0, 100.0, 100.0)
remove_resolution_level(root, level_index=1)
```

`add_resolution_level` writes an empty level (`vertex_count` 0,
`arrays_present` `[]`, `coarsening_method` `"manual"`) and its
`multiscales` entry; it writes no arrays. Level 0 cannot be removed.
To build populated levels, use `build_pyramid` or `coarsen_level` in
`zarr_vectors.multiresolution.coarsen`; see
[Pyramid construction](../multiscale/pyramid_construction.md).

### Validation

`LevelMetadata.validate()`, run whenever a level is written, requires:

- `level` and `vertex_count` of at least 0;
- `object_sparsity` in `(0, 1]`;
- at level 0, no `bin_shape` and no `parent_level`;
- above level 0, a `bin_shape` and a `parent_level`, and `bin_ratio`
  entries of at least 1;
- positive `chunk_shape` entries, and attribute-chunking and
  `chunk_sources` keys that agree with each other.

Writing a level also checks that its `chunk_shape` is a whole multiple of
the root's and that its bins divide it.

[L1](../validation/l1_structural.md) fails a store with no level groups,
fails a level without `vertices/`, and warns on a level with empty
attributes. [L2](../validation/l2_metadata.md) parses each level with
`read_level_metadata`, repeats the chunk-shape check, and warns when a
level's presence is deferred. [L3](../validation/l3_consistency.md)
checks `fragments_tile` and `link_fragments_aligned` against the data.
