# Root metadata

This page lists every key in the root attributes of a Zarr Vectors store.

## Terms

**Root attributes**
: The `attributes` of the root `zarr.json`. They hold four keys:
  `zarr_conventions`, `zarr_vectors`, `multiscales` and `ome`, and
  optionally `proj:projjson` and `user_metadata`.

**Convention registration**
: An entry in `zarr_conventions` naming a convention the node follows.
  Format 0.10.0 stores its metadata as the
  [Zarr Vectors convention](https://github.com/Andrew-Keenlyside/zarr-vectors)
  (v0.1.0), and every node that carries a `zarr_vectors` attribute
  registers it.

**Capability token**
: A string in `format_capabilities` or `required_capabilities` naming an
  optional feature the store uses. The tokens are the `CAP_*` constants in
  `zarr_vectors.constants`.

## Introduction

The `zarr_vectors` object, with `kind` `"root"`, says what the store
holds and how it is laid out, including its axes. `multiscales` lists the
levels. `ome` lets an OME collection name the store by path. Per-level
facts are in each [level's attributes](level_groups.md).

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
  "zarr_conventions": [
    {"uuid": "66268a61-7833-4995-bbcb-233bb417dae5", "name": "zarr_vectors", ...},
    {"uuid": "d35379db-88df-4056-af3a-620245f8e347", "name": "multiscales", ...}
  ],
  "zarr_vectors": {
    "kind": "root",
    "zv_version": "0.10.0",
    "axes": [
      {"name": "x", "type": "space"},
      {"name": "y", "type": "space"},
      {"name": "z", "type": "space"}
    ],
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
  "multiscales": {"layout": [{"asset": "0"}, {"asset": "1"}, {"asset": "2"}]},
  "ome": {"version": "0.6", "type": "collection", "name": "streamline_pyr", "nodes": [...], ...}
}
```

### `zarr_vectors` keys

Always written:

| Key | Type | Meaning |
|-----|------|---------|
| `kind` | string | `"root"` |
| `zv_version` | string | Format version of the writer, `"0.10.0"` |
| `axes` | list of objects | One `{name, type, unit?}` per axis; see [Dimensionality](../foundations/dimensionality.md) |
| `format_capabilities` | list of strings | Capability tokens the store uses; `[]` when none |
| `chunk_shape` | list of numbers | Spatial extent of one chunk per axis; see [Chunk shape](../chunking/chunk_shape.md) |
| `bounds` | `[min_corner, max_corner]` | Spatial bounds the chunk grid covers. `create_store` without `bounds=` uses `0` to `128` on each axis. A write outside the bounds raises unless the writer gets `out_of_bounds="expand"` (grow the bounds) or `"ignore"` (drop the vertices) |
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
| `attribute_specs` | object | none | Declared attributes by scope (`vertex`, `object`, `link`), for example `{"vertex": {"intensity": {"dtype": "float32"}}}` |
| `shard_shape` | integer or list | unsharded | Cells per shard per axis; see [Sharding](../chunking/sharding.md) |
| `octree_sharding` | object | `sharding_indexed` | Configuration of the `zarr_octree_sharding` codec; needs `shard_shape` |
| `manifest_layout` | string | `"vlen"` | Written only as `"dense"`; see [Object manifest](../object_model/object_manifest.md) |
| `manifest_block_dtype` | string | `"int64"` | Written only as `"int32"` |
| `required_capabilities` | list of strings | `[]` | Capabilities a reader must implement to read the store |
| `coordinate_offset` | list of numbers | zeros | World position of the stored origin: world = stored + offset. Written by `init_skeleton_store(coordinate_offset=...)` |

There is no `crs` key. A coordinate reference system is the root
attribute `proj:projjson`; see below.

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

### `zarr_conventions`

The root registers the Zarr Vectors convention with its full entry:

```json
{
  "schema_url": "https://raw.githubusercontent.com/Andrew-Keenlyside/zarr-vectors/refs/tags/v0.1.0/schema.json",
  "spec_url": "https://github.com/Andrew-Keenlyside/zarr-vectors/blob/v0.1.0/README.md",
  "uuid": "66268a61-7833-4995-bbcb-233bb417dae5",
  "name": "zarr_vectors",
  "description": "Spatially chunked, multi-resolution N-dimensional vector data (point clouds, meshes, skeletons, streamlines, graphs)"
}
```

It also registers the
[multiscales](https://github.com/zarr-conventions/multiscales) convention
(uuid `d35379db-88df-4056-af3a-620245f8e347`) and, when the store has a
CRS, the [proj](https://github.com/zarr-conventions/proj) convention
(uuid `f17cb550-5864-4468-aeb7-f3180cfb622f`). Every other node with a
`zarr_vectors` attribute registers the short form
`{"uuid": "66268a61-7833-4995-bbcb-233bb417dae5"}`. A writer keeps
entries for other conventions that are already there.

### `multiscales`, `proj:projjson` and `ome`

`multiscales` is the multiscales convention's block, with one `layout`
entry per level: `{"layout": [{"asset": "0"}, {"asset": "1"}, ...]}`.
It has no `transform`, because every level stores vertices in level-0
coordinates. A coarser level's `bin_ratio` and `bin_shape` are in its
[level attributes](level_groups.md). See
[Multiscale metadata](../multiscale/multiscale_metadata.md).

`proj:projjson` holds the coordinate reference system given to
`create_store(crs=...)`, as a PROJJSON object. It is absent when the store
has no CRS.

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
`chunk_shape`, `bounds`, `geometry_types` or `axes` is missing. It raises
`UnsupportedFormatVersionError` (a `MetadataError`) for a 0.9.x store,
whose root `zarr_vectors` object has no `kind`, and for a store newer
than 0.10; see [Migrating from 0.9.x](#migrating-from-09x) below.

To change the block after creation, use `set_bounds(root, bounds)`
(shrinking needs `force=True`) and
`update_root_metadata(root, add_capabilities=[...], **fields)` from
`zarr_vectors.core.store` rather than editing the attributes directly.

### `user_metadata`

Application metadata goes under `user_metadata`, one namespace per
application, never in `zarr_vectors`. `Dataset.metadata["myapp"]`
reads and writes `attributes["user_metadata"]["myapp"]`. The library
does not read or validate it.

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

L2 also runs `RootMetadata.validate()` (readers do not), which
requires `zv_version` of at least `0.10.0`, at least two space axes with
unique names, `reduction_factor` of at least 2, known `cross_level_storage`,
`manifest_layout` and `manifest_block_dtype` values, and `shard_shape`
whenever `octree_sharding` is set.

### Migrating from 0.9.x

Format 0.10.0 moves metadata; every cell is byte-for-byte as in 0.9.4.
This package refuses a 0.9.x store with `UnsupportedFormatVersionError`.
Migrate it in place, rewriting only `zarr.json` files:

```python
from zarr_vectors.building import migrate_metadata_0_10

migrate_metadata_0_10("tracts.zarrvectors", dry_run=True)  # counts only
migrate_metadata_0_10("tracts.zarrvectors")
```

It returns the number of nodes rewritten by role. The root is rewritten
last, so an interrupted run leaves a store that is still refused as
0.9.x, and running it again finishes the job. On a 0.10.0 store it
changes nothing and returns zeros.

| 0.9.x | 0.10.0 |
|-------|--------|
| Root `zarr_vectors` | The same object with `kind: "root"` and `zv_version` `"0.10.0"` |
| Root `zarr_vectors.crs` | Root `proj:projjson`, plus the proj registration |
| Root `multiscales[0].axes` | Root `zarr_vectors.axes` |
| Root `multiscales` (NGFF 0.4 list) | `{"layout": [{"asset": "<level>"}, ...]}` |
| Level `zarr_vectors_level` | Level `zarr_vectors` with `kind: "level"` |
| Level `zarr_vectors_presence` | The level object's `presence` |
| Array `zv_array` and its fields at the top level | `zarr_vectors` with `kind` and the same fields |
| `zv_array: "attribute"` | `kind: "vertex_attribute"` |
| `zv_array: "groupings_attribute"` | `kind: "group_attribute"` |
| `parametric/` `parametric_types` | `zarr_vectors` with `kind: "parametric"` and `parametric_types` |
| No registration | `zarr_conventions` on every node above |

A coarse level whose bins were only in its old NGFF entry gets them
copied into its level object (`bin_ratio` = `scale`, `bin_shape` = 2 ×
`translation`). A 0.9.x reader cannot read a migrated store.
