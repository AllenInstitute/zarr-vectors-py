# Root metadata

This page lists every key in the root attributes of a Zarr Vectors store:
the `zarr_vectors` block, and where the `multiscales` and `ome` blocks
beside it are specified.

## Terms

**Root attributes**
: The `attributes` object of the `zarr.json` at the store root. It holds
  three keys: `zarr_vectors`, `multiscales` and `ome`.

**Capability token**
: A string in `format_capabilities` or `required_capabilities` naming an
  optional feature the store uses. The tokens are the `CAP_*` constants in
  `zarr_vectors.constants`.

## Introduction

A reader opens the root attributes first. The `zarr_vectors` block says
what the store holds (geometry types, bounds, chunk and bin shape) and how
it is laid out; the axes are in `multiscales`; and `ome` lets an OME
collection name the store by path. Per-level facts are in each
[level's attributes](level_groups.md).

`RootMetadata` in `zarr_vectors.core.metadata` is the in-memory form:
`read_root_metadata(root)` builds one from the root attributes, filling
defaults for keys that are absent.

## Technical reference

### Example

The root attributes of a three-level streamline pyramid
(`write_polylines` with `chunk_shape=(200, 200, 200)` and
`bin_shape=(50, 50, 50)`, then `build_pyramid(..., factors=[(2, 2), (2, 2)])`):

```json
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
  "multiscales": [
    {
      "version": "0.4",
      "name": "default",
      "axes": [
        {"name": "x", "type": "space"},
        {"name": "y", "type": "space"},
        {"name": "z", "type": "space"}
      ],
      "datasets": [
        {"path": "0", "coordinateTransformations": [
          {"type": "scale", "scale": [1.0, 1.0, 1.0]},
          {"type": "translation", "translation": [25.0, 25.0, 25.0]}]},
        {"path": "1", "coordinateTransformations": [
          {"type": "scale", "scale": [2.0, 2.0, 2.0]},
          {"type": "translation", "translation": [50.0, 50.0, 50.0]}]},
        {"path": "2", "coordinateTransformations": [
          {"type": "scale", "scale": [4.0, 4.0, 4.0]},
          {"type": "translation", "translation": [100.0, 100.0, 100.0]}]}
      ],
      "metadata": {"format": "zarr_vectors"}
    }
  ],
  "ome": {
    "version": "0.6",
    "type": "collection",
    "name": "streamline_pyr",
    "attributes": {
      "scene": {
        "coordinateSystems": [
          {"id": "world", "name": "world", "axes": [
            {"name": "x", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "z", "type": "space"}]}
        ],
        "coordinateTransformations": []
      }
    },
    "nodes": [
      {"type": "zv:level", "name": "0"},
      {"type": "zv:level", "name": "1"},
      {"type": "zv:level", "name": "2"}
    ]
  }
}
```

### `zarr_vectors` keys

Written by every store:

| Key | Type | Meaning |
|-----|------|---------|
| `zv_version` | string | Format version of the writer, `"0.9.4"` |
| `format_capabilities` | list of strings | Capability tokens the store uses; `[]` when none |
| `chunk_shape` | list of numbers | Spatial extent of one chunk per axis; see [Chunk shape](../chunking/chunk_shape.md) |
| `bounds` | `[min_corner, max_corner]` | Spatial bounds the chunk grid is allocated for |
| `geometry_types` | list of strings | Geometry types in the store: `point_cloud`, `line`, `polyline`, `streamline`, `graph`, `skeleton` or `mesh` |

Written when set (each is absent otherwise, and a reader uses the default):

| Key | Type | Default when absent | Meaning |
|-----|------|---------------------|---------|
| `base_bin_shape` | list of numbers | `chunk_shape` (one bin per chunk) | Bin shape at level 0; see [Bin shape](../chunking/bin_shape.md) |
| `links_convention` | string | `"implicit_sequential"` | `explicit`, `implicit_sequential` or `implicit_sequential_with_branches` |
| `object_index_convention` | string | `"standard"` | `standard` or `identity` |
| `cross_chunk_strategy` | string | `"explicit_links"` | `explicit_links`, `boundary_deduplication` or `both` |
| `cross_level_depth` | integer | `1` | Largest level delta of cross-level links `build_pyramid` writes; `0` none, `-1` all |
| `cross_level_storage` | string | `"explicit"` | `none`, `implicit` or `explicit` |
| `reduction_factor` | integer | `8` | Pyramid reduction factor, at least 2 |
| `crs` | object | none | Coordinate reference system, stored as given to `create_store(crs=...)` |
| `attribute_specs` | object | none | Declared attributes by scope (`vertex`, `object`, `link`), for example `{"vertex": {"intensity": {"dtype": "float32"}}}`; only non-empty scopes are written |
| `shard_shape` | integer or list | unsharded | Cells per shard per axis, used by every writer that allocates a per-chunk array; see [Sharding](../chunking/sharding.md) |
| `octree_sharding` | object | `sharding_indexed` | Configuration of the `zarr_octree_sharding` codec; needs `shard_shape` |
| `manifest_layout` | string | `"vlen"` | Written only as `"dense"`; see [Object manifest](../object_model/object_manifest.md) |
| `manifest_block_dtype` | string | `"int64"` | Written only as `"int32"` |
| `required_capabilities` | list of strings | `[]` | Capabilities a reader must implement to read the store |

The writers for each geometry type set `links_convention`,
`object_index_convention` and `cross_chunk_strategy`: point clouds, lines,
polylines and streamlines write `implicit_sequential`; graphs and meshes
`explicit`; skeletons `implicit_sequential_with_branches`.
`build_pyramid` adds `cross_level_depth` and `cross_level_storage`.

The axes are not in this block. They are `multiscales[0].axes`, and
`RootMetadata.spatial_index_dims` reads them from there.

### Capability tokens

| Token | Added when |
|-------|------------|
| `preserved_object_ids` | A level keeps object IDs stable across levels (`preserves_object_ids` in its [attributes](level_groups.md)) |
| `shared_fragments` | A level's fragments may be named by more than one object |
| `multiscale_links` | The store may hold cross-level links (`links/<delta>/` with `delta != 0`) |
| `dense_manifests` | `manifest_layout` is `"dense"` |
| `octree_sharding` | `octree_sharding` is set |
| `octree_chunks` | A level has [octree chunks](../chunking/octree_chunks.md); listed in `required_capabilities` too |
| `fragment_index` | Defined in `zarr_vectors.constants`; no writer adds it |

This package writes `required_capabilities` only for `octree_chunks`, and
does not refuse a store whose list names a capability it lacks.

### `multiscales` and `ome`

`multiscales` is an OME-NGFF 0.4 multiscales block with one dataset per
level. Each level's `bin_ratio` and `bin_shape` are stored only here, as
the dataset's `scale` and twice its `translation`. It is specified on
[Multiscale metadata](../multiscale/multiscale_metadata.md).

`ome` is an RFC 8 collection node: `version` `"0.6"`, `type`
`"collection"`, a `name` (the store's directory name without extension
unless `create_store(name=...)` sets one), the axes as one coordinate
system named `world`, and one `{"type": "zv:level", "name": "<n>"}` entry
per level. Nothing in this package reads it. A store without one is valid;
`zarr_vectors.building.stamp_ome_node(store)` adds it without rewriting
data.

### Reading root metadata

```python
from zarr_vectors.core.store import open_store, read_root_metadata

root = open_store("tracts.zarrvectors")
sorted(root.attrs.to_dict())    # ['multiscales', 'ome', 'zarr_vectors']
meta = read_root_metadata(root)
meta.geometry_types             # ['streamline']
meta.effective_bin_shape        # (50.0, 50.0, 50.0)
meta.reduction_factor           # 8, the default: the key is absent
```

`read_root_metadata` requires `zv_version`, `chunk_shape`, `bounds`,
`geometry_types` and `multiscales[0].axes`, and raises `MetadataError`
when one is missing.

### Validation

[L1](../validation/l1_structural.md) fails a store whose root attributes
have no `zarr_vectors` key and warns when they have no `ome` key.

[L2](../validation/l2_metadata.md) parses the block with
`read_root_metadata` and checks that:

- `chunk_shape` and `bounds` have one entry per axis, and `chunk_shape`
  entries are positive;
- `links_convention`, `object_index_convention` and `cross_chunk_strategy`
  are known values;
- `geometry_types` is not empty (a warning);
- `base_bin_shape`, when present, has one positive entry per axis, each a
  whole fraction of the matching `chunk_shape` entry (tolerance `1e-9`);
- `octree_chunks` is in both capability lists exactly while a level has
  octree chunks (a warning either way).

`RootMetadata.validate()` checks more, including `zv_version` of at least
`0.9.0`, at least two space axes with unique names, `reduction_factor` of
at least 2, known `cross_level_storage`, `manifest_layout` and
`manifest_block_dtype` values, and that `octree_sharding` comes with
`shard_shape`; the validator does not call it.
