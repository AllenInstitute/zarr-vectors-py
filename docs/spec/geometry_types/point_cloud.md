# Point cloud (`point_cloud`)

## Terms

**Point cloud**
: A set of vertex positions with no connectivity. The type constant is
  `GEOM_POINT_CLOUD`, `"point_cloud"`.

## Introduction

A point cloud stores positions and optional per-vertex attributes. It suits
data whose points are independent: lidar, synchrotron point data,
single-molecule localisation microscopy, spatial transcriptomics.

Without objects, the writer stores one fragment per occupied bin in each
chunk, so a reader can fetch part of a chunk. With `object_ids=`, fragments
follow objects instead of bins. What every type shares is on
[Geometry types](index.md#common-to-every-type).

## Technical reference

### Arrays present

At level 0 of a store written by `write_points`:

| Path | Present | Description |
|------|---------|-------------|
| `vertices/` | Always | One fragment per occupied bin (or per object) per chunk |
| `vertex_attributes/<name>/` | With `vertex_attributes=` | Vector attributes, such as RGB colour, have a `row_shape` |
| `object_index/` | With `object_ids=` | One manifest per object |
| `object_attributes/<name>/` | With `object_attributes=` | |
| `groups/` | With `groups=` | Group-to-object lists |

A point cloud has no `links/` group. Its root attributes:

```javascript
{
  "geometry_types": ["point_cloud"],
  "base_bin_shape": [50.0, 50.0, 50.0],
  "links_convention": "implicit_sequential",
  ...
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
colour = rng.integers(0, 256, (100_000, 3)).astype(np.uint8)

summary = write_points(
    "scan.zarrvectors",
    positions,
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
    vertex_attributes={"intensity": intensity, "colour": colour},
)
print(summary)
# {'vertex_count': 100000, 'chunk_count': 125, 'object_count': 0,
#  'group_count': 0, 'bins_per_chunk': (4, 4, 4)}

result = read_points(
    "scan.zarrvectors",
    bbox=(np.array([100.0, 100.0, 100.0]), np.array([200.0, 200.0, 200.0])),
    attribute_names=["colour"],
)
print(result["vertex_count"], result["vertex_attributes"]["colour"].shape)   # 99 (99, 3)
```

`read_points` returns:

| Key | Type | Description |
|-----|------|-------------|
| `positions` | `ndarray (N, D)` | Vertex positions |
| `vertex_attributes` | `dict[str, ndarray]` | The attributes named in `attribute_names`; none by default |
| `vertex_count` | `int` | Vertices returned |
| `object_ids` | `ndarray (N,)` | Owning object of each vertex; only when `object_ids=` or `group_ids=` is given |

### Validation

| Level | Check |
|-------|-------|
| L3 | For a store whose only type is `point_cloud`, with no `object_index/` and more than one bin per chunk: a chunk with more fragments than bins is an error; in the first three chunks of each level, a fragment whose points fall outside its bin is a warning |
| L4 | The check for a `links/0/` family warns on every point-cloud store, with or without links; see [Level 4](../validation/overview.md#level-4-geometry-conventions) |
