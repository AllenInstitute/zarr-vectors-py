# Root metadata

This page lists every key in the root attributes of a Zarr Vectors store.

## Terms

**Root attributes**
: The `attributes` of the root `zarr.json`. They hold three keys:
  `zarr_vectors`, `multiscales` and `ome`.

**Capability token**
: A string in `format_capabilities` or `required_capabilities` naming an
  optional feature the store uses. The tokens are the `CAP_*` constants in
  `zarr_vectors.constants`.

## Introduction

The `zarr_vectors` block says what the store holds and how it is laid out.
The axes are in `multiscales`. `ome` lets an OME collection name the store
by path. Per-level facts are in each [level's attributes](level_groups.md).

`read_root_metadata(root)` in `zarr_vectors.core.store` returns a
`RootMetadata` (`zarr_vectors.core.metadata`), with defaults filled in for
absent keys.

## Technical reference

### Example

The root attributes of a three-level streamline pyramid
(`write_polylines` with `chunk_shape=(200, 200, 200)` and
`bin_shape=(50, 50, 50)`, then `build_pyramid(..., factors=[(2, 2), (2, 2)])`):

```javascript
{
  "zarr_vectors": {
    "zv_version": "0.9.4",
    "format_capabilities": ["preserved_object_ids", "shared_fragments", "multiscale_links"],
    "chunk_shape": [200.0, 200.0, 200.0],
    "bounds": [[0.0, 0.0, 0.0], [400.0, 400.0, 400.0]],
    "geometry_types": ["streamline"],
    "base_bin_shape": [50.0, 50.0, 50.0],
    "links_convention": "implicit_sequential",
    "object_index_convention": "standard",
    "cross_chunk_strategy": "explicit_links",
    "cross_level_depth": 1,
    "cross_level_storage": "explicit"
  },
  "multiscales": [{"version": "0.4", "name": "default", "axes": [...], "datasets": [...], ...}],
  "ome": {"version": "0.6", "type": "collection", "name": "streamline_pyr", "nodes": [...], ...}
}
```

### `zarr_vectors` keys

Always written:

| Key | Type | Meaning |
|-----|------|---------|
| `zv_version` | string | Format version of the writer, `"0.9.4"` |
| `format_capabilities` | list of strings | Capability tokens the store uses; `[]` when none |
| `chunk_shape` | list of numbers | Spatial extent of one chunk per axis; see [Chunk shape](../chunking/chunk_shape.md) |
| `bounds` | `[min_corner, max_corner]` | Spatial bounds the chunk grid covers |
| `geometry_types` | list of strings | `point_cloud`, `line`, `polyline`, `streamline`, `graph`, `skeleton` or `mesh` |

Written when set; a reader uses the default when absent:

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `base_bin_shape` | list of numbers | `chunk_shape` | Bin shape at level 0; see [Bin shape](../chunking/bin_shape.md) |
| `links_convention` | string | `"implicit_sequential"` | `explicit`, `implicit_sequential` or `implicit_sequential_with_branches`; each type's value is in [Geometry types](../geometry_types/index.md) |
| `object_index_convention` | string | `"standard"` | `standard` or `identity` |
| `cross_chunk_strategy` | string | `"explicit_links"` | `explicit_links`, `boundary_deduplication` or `both` |
| `cross_level_depth` | integer | `1` | Largest level delta of cross-level links `build_pyramid` writes; `0` none, `-1` all |
| `cross_level_storage` | string | `"explicit"` | `none`, `implicit` or `explicit` |
| `reduction_factor` | integer | `8` | Pyramid reduction factor, at least 2 |
| `crs` | object | none | Coordinate reference system, as given to `create_store(crs=...)` |
| `attribute_specs` | object | none | Declared attributes by scope (`vertex`, `object`, `link`), for example `{"vertex": {"intensity": {"dtype": "float32"}}}` |
| `shard_shape` | integer or list | unsharded | Cells per shard per axis; see [Sharding](../chunking/sharding.md) |
| `octree_sharding` | object | `sharding_indexed` | Configuration of the `zarr_octree_sharding` codec; needs `shard_shape` |
| `manifest_layout` | string | `"vlen"` | Written only as `"dense"`; see [Object manifest](../object_model/object_manifest.md) |
| `manifest_block_dtype` | string | `"int64"` | Written only as `"int32"` |
| `required_capabilities` | list of strings | `[]` | Capabilities a reader must implement to read the store |

The axes are not in this block: they are `multiscales[0].axes`.

### Capability tokens

| Token | Added when |
|-------|------------|
| `preserved_object_ids` | A level keeps object IDs stable across levels (`preserves_object_ids` in its [attributes](level_groups.md)) |
| `shared_fragments` | A level's fragments may be named by more than one object |
| `multiscale_links` | The store may hold cross-level links (`links/<delta>/` with `delta != 0`) |
| `dense_manifests` | `manifest_layout` is `"dense"` |
| `octree_sharding` | `octree_sharding` is set |
| `octree_chunks` | A level has [octree chunks](../chunking/octree_chunks.md); also in `required_capabilities` |
| `fragment_index` | Not added by any writer |

`octree_chunks` is the only token written to `required_capabilities`. The
reader does not refuse a store that requires a capability it lacks.

### `multiscales` and `ome`

`multiscales` is an OME-NGFF 0.4 multiscales block with one dataset per
level. It is the only place a level's `bin_ratio` and `bin_shape` are
stored: as the dataset's `scale` and twice its `translation`. See
[Multiscale metadata](../multiscale/multiscale_metadata.md).

`ome` is an RFC 8 collection node (`version` `"0.6"`, `type`
`"collection"`). Its `name` is the store's directory name without
extension, unless `create_store(name=...)` sets one. It holds the axes as
one coordinate system named `world`, and one
`{"type": "zv:level", "name": "<n>"}` node per level. Nothing in this
package reads `ome`, and a store without it is valid.
`zarr_vectors.building.stamp_ome_node(store)` adds it without rewriting
data.

### Reading root metadata

```python
from zarr_vectors.core.store import open_store, read_root_metadata

root = open_store("tracts.zarrvectors")
meta = read_root_metadata(root)
meta.geometry_types             # ['streamline']
meta.effective_bin_shape        # (50.0, 50.0, 50.0)
meta.reduction_factor           # 8, the default: the key is absent
```

`read_root_metadata` raises `MetadataError` when `zv_version`,
`chunk_shape`, `bounds`, `geometry_types` or `multiscales[0].axes` is
missing.

### Validation

[L1](../validation/l1_structural.md) fails a store with no `zarr_vectors`
key and warns when there is no `ome` key.
[L2](../validation/l2_metadata.md) checks that:

- `chunk_shape` and `bounds` have one entry per axis, and `chunk_shape`
  entries are positive;
- `links_convention`, `object_index_convention` and `cross_chunk_strategy`
  are known values;
- `base_bin_shape`, when present, has one positive entry per axis, each a
  whole fraction of `chunk_shape` (tolerance `1e-9`);
- `geometry_types` is not empty, and `octree_chunks` is in both capability
  lists exactly while a level has octree chunks (warnings).

`RootMetadata.validate()`, which the validator does not call, also
requires `zv_version` of at least `0.9.0`, at least two space axes with
unique names, `reduction_factor` of at least 2, known `cross_level_storage`,
`manifest_layout` and `manifest_block_dtype` values, and `shard_shape`
whenever `octree_sharding` is set.
