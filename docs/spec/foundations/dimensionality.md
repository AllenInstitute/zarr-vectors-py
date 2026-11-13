# Dimensionality

This page describes how a store's number of spatial dimensions is
declared, where it appears on disk, and what the validator checks
against it.

## Terms

**Spatial dimensionality (D)**
: The number of axes vertices are positioned and chunked in. It is the
  length of `multiscales[0].axes` in the root attributes; the Python
  package exposes it as `RootMetadata.sid_ndim`. `D` is fixed for the
  store and shared by every level.

**Axis**
: One entry of `multiscales[0].axes`: a `name`, a `type` (such as
  `"space"` or `"time"`) and an optional `unit`.

**`chunk_shape`, `bin_shape`, `bounds`**
: Per-axis values, each with exactly `D` entries. See
  [Chunk shape](../chunking/chunk_shape.md) and
  [Bin shape](../chunking/bin_shape.md).

---

## Introduction

A store is not tied to three dimensions. `D` is set when the store is
created, and every spatial rule (chunk assignment, bins, cell
addressing, bounding-box queries) is written for `D` axes. Most stores
are 3-D; 2-D stores and stores with a fourth axis such as time use the
same layout and code paths.

---

## Technical reference

### Declaring dimensionality

`create_store` takes `D` from `ndim`, or from the length of `axes`,
`bounds` or `chunk_shape`, whichever is given; if none is given, `D`
is 3. The type writers such as `write_points` take it from
`chunk_shape` and the positions. If the inputs disagree,
`MetadataError` is raised, for example
`Inconsistent dimensionality: ndim=3, chunk_shape=2`.

When no `axes` are given, they are generated from the names `x`, `y`,
`z`, `w` in that order, each with `"type": "space"` and no unit. A 2-D
point cloud written with `chunk_shape=(500.0, 500.0)` records:

```json
"axes": [
  {"name": "x", "type": "space"},
  {"name": "y", "type": "space"}
]
```

The axis order is the column order of the positions and the order of
every per-axis list (`chunk_shape`, `bin_shape`, `bounds`).

### Where `D` appears on disk

`D` is not stored as a separate key. It is implied by:

| Item | Shape |
|------|-------|
| `multiscales[0].axes` (root attributes) | `D` entries |
| `chunk_shape`, `bounds[0]`, `bounds[1]`, `base_bin_shape` (root `zarr_vectors` block) | `D` entries each |
| `bin_shape`, `bin_ratio` (each level's `zarr_vectors_level` block) | `D` entries each |
| Per-chunk arrays (`vertices`, `vertex_fragments`, `links/<delta>/<offsets>`, attributes) | rank `D`, one cell per spatial chunk; rank `D + 1` when the level is [chunked by an attribute](../chunking/attribute_chunking.md) |
| Positions inside a `vertices` cell | rows of `D` values |
| The `<offsets>` path segment of a links array | `link_width - 1` offsets of `D` signed components each (see [Links](../object_model/links.md)) |
| Dense object-index `manifest_spans` / `manifest_blocks` | `D + 1` columns (see [Object manifest](../object_model/object_manifest.md)) |

The cell layouts are in [Chunk arrays](../layout/chunk_arrays.md).

### Chunk and cell addressing

A vertex at position `p` is in the spatial chunk

```
chunk[i] = floor(p[i] / chunk_shape[i])     for i in 0 .. D-1
```

and that chunk is stored in the per-chunk array's cell
`chunk - origin`, where `origin[i] = floor(bounds_min[i] / chunk_shape[i])`
is recorded in the array's `chunk_grid_origin` attribute (absent means
zero). This lets stores with negative coordinates use a 0-indexed
array. A bounding-box query reads the cells from
`floor(lo / chunk_shape)` to `floor(hi / chunk_shape)` on each axis.
Bin addressing within a chunk follows the same per-axis rule; see
[Bin shape](../chunking/bin_shape.md).

### A store with a time axis

A fourth axis is chunked and binned like the others. This example
declares units on the axes with `create_store`, writes 1000 points,
and reads the first spatial chunk for `0 <= t < 5`:

```python
import numpy as np
from zarr_vectors.core.store import create_store
from zarr_vectors.types.points import read_points, write_points

rng = np.random.default_rng(0)
positions = np.column_stack([
    rng.uniform(0, 400, (1000, 3)),     # x, y, z in micrometres
    rng.uniform(0, 20, 1000),           # t in seconds
]).astype(np.float32)

create_store(
    "timeseries.zarrvectors",
    axes=[
        {"name": "x", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "t", "type": "time", "unit": "second"},
    ],
    chunk_shape=(200.0, 200.0, 200.0, 10.0),
    bounds=([0.0, 0.0, 0.0, 0.0], [400.0, 400.0, 400.0, 20.0]),
    geometry_types=["point_cloud"],
)
write_points(
    "timeseries.zarrvectors",
    positions,
    chunk_shape=(200.0, 200.0, 200.0, 10.0),
    bin_shape=(50.0, 50.0, 50.0, 5.0),
)

result = read_points(
    "timeseries.zarrvectors",
    bbox=(np.array([0.0, 0.0, 0.0, 0.0]),
          np.array([200.0, 200.0, 200.0, 5.0])),
)
print(result["positions"].shape)
```

```
(33, 4)
```

The `vertices` array of this store has shape `[2, 2, 2, 2]`.

### Validation

Checks that involve `D`, by level (see
[Validation](../validation/overview.md)):

- **Level 1** (`validate/structure.py`): every per-chunk array in a
  level has the same rank as `vertices`, and every stored cell key has
  that many components; a level chunked by an attribute must have rank
  `D + 1`.
- **Level 2** (`validate/metadata.py`): `D >= 1`; `chunk_shape`, both
  `bounds` corners and `base_bin_shape` have `D` entries; each level's
  `bin_shape` and `bin_ratio` have `D` entries.
- **Level 3** (`validate/consistency.py`): every decoded vertex
  fragment has `D` columns.
