# Dimensionality

How a store declares its number of spatial dimensions, `D`, where `D`
appears on disk, and what the validator checks.

## Terms

**Dimensionality (`D`)**
: The number of axes vertices are positioned and chunked in: the length
  of `multiscales[0].axes` in the root attributes
  (`RootMetadata.sid_ndim` in Python). `D` is fixed for the store and
  shared by every level.

**Axis**
: One entry of `multiscales[0].axes`: a `name`, a `type` (such as
  `"space"` or `"time"`) and an optional `unit`.

## Introduction

Every spatial rule is written for `D` axes. Most stores are 3-D; 2-D
stores and stores with a time axis use the same layout and code.

## Technical reference

### Declaring `D`

`create_store` takes `D` from `ndim`, or from the length of `axes`,
`bounds` or `chunk_shape`; the default is 3. The type writers take it
from `chunk_shape` and the positions. Inputs that disagree raise
`MetadataError`, for example
`Inconsistent dimensionality: ndim=3, chunk_shape=2`.

Without `axes`, the axes are named `x`, `y`, `z`, `w` in order, each
with `"type": "space"` and no unit. Axis order is the column order of
the positions and the order of every per-axis list.

### Where `D` appears

`D` is not stored as its own key. It is implied by:

| Item | Size |
|------|-------|
| `multiscales[0].axes` | `D` entries |
| `chunk_shape`, `bounds[0]`, `bounds[1]`, `base_bin_shape` (root `zarr_vectors` block) | `D` entries each |
| `bin_shape`, `bin_ratio` (each level's `zarr_vectors_level` block) | `D` entries each |
| Per-chunk arrays (`vertices`, `vertex_fragments`, links, attributes) | rank `D`; rank `D + 1` when [chunked by an attribute](../chunking/attribute_chunking.md) |
| Positions in a `vertices` cell | rows of `D` values |
| `<offsets>` path segment of a links array | `link_width - 1` offsets of `D` components each (see [Links](../object_model/links.md)) |
| Dense object-index `manifest_spans` / `manifest_blocks` | `D + 1` columns (see [Object manifest](../object_model/object_manifest.md)) |

### Chunk and cell addressing

A vertex at position `p` is in spatial chunk

```
chunk[i] = floor(p[i] / chunk_shape[i])     for i in 0 .. D-1
```

stored in cell `chunk - origin`. The origin,
`floor(bounds_min / chunk_shape)`, is the array's `chunk_grid_origin`
attribute (absent means zero), so negative coordinates still give a
0-indexed array. Bins follow the same rule; see
[Bin shape](../chunking/bin_shape.md).

### Example: a time axis

```python
import numpy as np
import zarr
from zarr_vectors.core.store import create_store
from zarr_vectors.types.points import write_points

rng = np.random.default_rng(0)
positions = rng.uniform(0, 20, (1000, 4)).astype(np.float32)  # x, y, z, t
chunk_shape = (10.0, 10.0, 10.0, 10.0)

create_store(
    "timeseries.zarrvectors",
    axes=[{"name": n, "type": "space", "unit": "micrometer"} for n in "xyz"]
    + [{"name": "t", "type": "time", "unit": "second"}],
    chunk_shape=chunk_shape,
    bounds=([0.0] * 4, [20.0] * 4),
)
write_points("timeseries.zarrvectors", positions, chunk_shape=chunk_shape)
print(zarr.open_array("timeseries.zarrvectors/0/vertices").shape)
```

```
(2, 2, 2, 2)
```

### Validation

| Level | Check |
|---|---|
| 1 (`validate/structure.py`) | Every per-chunk array in a level has the rank of `vertices`, and every cell key has that many components. A level chunked by an attribute has rank `D + 1`. |
| 2 (`validate/metadata.py`) | `D >= 1`. `chunk_shape`, both `bounds` corners and `base_bin_shape` have `D` entries; so do each level's `bin_shape` and `bin_ratio`. |
| 3 (`validate/consistency.py`) | Every decoded vertex fragment has `D` columns. |

See [Validation](../validation/overview.md).
