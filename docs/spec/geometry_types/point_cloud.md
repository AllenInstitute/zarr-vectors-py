# Point cloud (`point_cloud`)

## Terms

**Point cloud**
: A collection of spatial positions (vertices) with no connectivity. Each
  vertex is independent. Per-vertex scalar or vector attributes may be
  stored alongside the positions.

**`GEOM_POINT_CLOUD`**
: The geometry type constant `"point_cloud"`, stored in the
  `geometry_types` list of the root attributes.

**Per-vertex attribute**
: A named array under `vertex_attributes/<name>/` holding one row per
  vertex, in the same chunk and fragment order as `vertices/`.

## Introduction

The point cloud is the simplest Zarr Vectors geometry: positions plus
optional per-vertex attributes, with no connectivity. It suits any dataset
whose vertices are independent measurements: lidar scans, synchrotron point
data, single-molecule localisation microscopy, spatial transcriptomics.

Without objects, the writer files each chunk's vertices into one fragment
per bin, so a reader can fetch a sub-chunk region without decoding the whole
chunk. Points may also be assigned to objects and groups, in which case
fragments follow objects instead of bins.

## Technical reference

### Arrays present

At level 0 of a store written by `write_points`:

| Path | Present | Description |
|------|---------|-------------|
| `vertices/` | Always | Vertex positions, one fragment per occupied bin (or per object), per chunk |
| `vertex_fragments/` | Always | Fragment index per chunk; see [Fragment-index arrays](../layout/fragment_index_arrays.md) |
| `vertex_attributes/<name>/` | With `vertex_attributes=` | One row per vertex; vector attributes have a `row_shape` |
| `object_index/` | With `object_ids=` | Object manifests; see [Object manifest](../object_model/object_manifest.md) |
| `object_attributes/<name>/` | With `object_attributes=` | One row per object |
| `groups/` | With `groups=` | Group-to-object lists |

A point cloud has no `links/` group.

### Root attributes

A point-cloud store has no type-specific root keys. The writer sets these
values in the `zarr_vectors` block (the shared keys are described in
[Root metadata](../layout/root_metadata.md)):

```json
{
  "geometry_types": ["point_cloud"],
  "base_bin_shape": [50.0, 50.0, 50.0],
  "links_convention": "implicit_sequential",
  "object_index_convention": "standard",
  "cross_chunk_strategy": "explicit_links"
}
```

`base_bin_shape` is written only when `bin_shape=` is given.

### Write and read API

```python
import numpy as np
from zarr_vectors.building import write_points
from zarr_vectors.types.points import read_points

rng = np.random.default_rng(0)
positions = rng.uniform(0, 1000, (100_000, 3)).astype(np.float32)
intensity = rng.uniform(0, 1, 100_000).astype(np.float32)
label = rng.integers(0, 8, 100_000).astype(np.int32)
colour = rng.integers(0, 256, (100_000, 3)).astype(np.uint8)

summary = write_points(
    "scan.zarrvectors",
    positions,
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
    vertex_attributes={"intensity": intensity, "label": label, "colour": colour},
)
print(summary)
# {'vertex_count': 100000, 'chunk_count': 125, 'object_count': 0,
#  'group_count': 0, 'bins_per_chunk': (4, 4, 4)}

# Whole store; attribute_names=None reads no attributes
result = read_points("scan.zarrvectors", attribute_names=["intensity", "colour"])
print(result["vertex_count"], result["positions"].shape)   # 100000 (100000, 3)
print(result["vertex_attributes"]["colour"].shape)        # (100000, 3)

# Spatial bounding-box query
result = read_points(
    "scan.zarrvectors",
    bbox=(np.array([100.0, 100.0, 100.0]), np.array([200.0, 200.0, 200.0])),
)
print(result["vertex_count"])   # 99
```

`read_points` also takes `level=`, `object_ids=`, `group_ids=`, `chunks=`
and `attribute_filter=`; the filters combine with AND. It returns:

| Key | Type | Description |
|-----|------|-------------|
| `positions` | `ndarray (N, D)` | Vertex positions |
| `vertex_attributes` | `dict[str, ndarray]` | The attributes named in `attribute_names` |
| `vertex_count` | `int` | Vertices returned |
| `object_ids` | `ndarray (N,)` | Owning object of each vertex; only when `object_ids=` or `group_ids=` is given |

Attributes may have any numeric dtype. A vector attribute, such as an RGB
colour, is stored with its per-vertex shape in the array's `row_shape`
attribute.

How coarser levels are built is described in
[Pyramid construction](../multiscale/pyramid_construction.md).

### Validation

- L1: `vertices/` missing at a level is an error; `vertex_fragments/`
  missing is a warning.
- L3: for a store whose only geometry type is `point_cloud`, that has no
  `object_index/`, and whose chunks hold more than one bin, a chunk with
  more fragments than bins is an error, and in the first three chunks of
  each level a fragment whose points fall outside its bin is a warning.
- L4: the check for a `links/0/` family warns on every point-cloud store,
  with or without links; see
  [Level 4](../validation/overview.md#level-4-geometry-conventions).

See [Validation overview](../validation/overview.md).
