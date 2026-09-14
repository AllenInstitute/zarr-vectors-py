# Point cloud (`point_cloud`)

## Terms

**Point cloud**
: An unordered collection of spatial positions (vertices) with no
  connectivity information. Each vertex is independent. Per-vertex scalar
  or vector attributes may be stored alongside positions.

**`GEOM_POINT_CLOUD`**
: The geometry type constant `"point_cloud"`. Stored in root `.zattrs`
  under `"geometry_type"`.

**Per-vertex attribute**
: A named array, stored under `attributes/<name>/`, whose length equals
  the number of vertices in the store. Each element corresponds to one
  vertex, in fragment order.

---

## Introduction

The point cloud type is the simplest Zarr Vectors geometry: positions plus optional
per-vertex attributes, no connectivity, no object model. It is suitable for
any dataset where vertices are independent measurements — lidar scans,
synchrotron absorption point data, single-molecule localisation microscopy,
gene expression spatial transcriptomics, or any other spatially indexed
scalar field sampled at discrete positions.

Point clouds support the full Zarr Vectors spatial indexing hierarchy (chunks, bins,
fragment index) and full multi-resolution pyramids via spatial coarsening into
metanodes.

---

## Technical reference

### Arrays present

| Array path | Required | Description |
|-----------|----------|-------------|
| `vertices/` | Yes | Vertex positions, shape `(N, D)` float32 per chunk |
| `vertex_fragments/` | Yes | Fragment index — `uint8` blob per chunk; see [Fragment-index arrays](../layout/fragment_index_arrays.md) |
| `attributes/<name>/` | No | Per-vertex scalar or vector attributes |

No `links/`, `object_index/`, or `object_attributes/` arrays are
present. The `groupings/` array is also absent; point clouds do not have
a discrete object model. (`cross_chunk_links/` does not exist for any
type — that family was merged into `links/`.)

### Root `.zattrs` required keys

```json
{
  "zarr_vectors_version": "1.0",
  "geometry_type":        "point_cloud",
  "spatial_dims":         3,
  "base_bin_shape":       [50.0, 50.0, 50.0],
  "chunk_shape":          [200.0, 200.0, 200.0]
}
```

No type-specific metadata keys beyond the shared schema.

### Write API

```python
import numpy as np
from zarr_vectors.types.points import write_points

positions  = np.random.default_rng(0).uniform(0, 1000, (100_000, 3)).astype(np.float32)
intensity  = np.random.default_rng(1).uniform(0, 1, 100_000).astype(np.float32)
label      = np.random.default_rng(2).integers(0, 8, 100_000).astype(np.int32)

write_points(
    "scan.zarrvectors",
    positions,
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
    attributes={
        "intensity": intensity,    # float32 scalar
        "label":     label,        # int32 categorical
    },
    coordinate_system="RAS",
    axis_units="micrometer",
)
```

### Read API

```python
from zarr_vectors.types.points import read_points

# Read entire store
result = read_points("scan.zarrvectors")
print(result["vertex_count"])             # int
print(result["positions"].shape)          # (N, 3)
print(result["attributes"]["intensity"])  # (N,) float32

# Spatial bbox query
result = read_points(
    "scan.zarrvectors",
    bbox=(np.array([100., 100., 100.]), np.array([200., 200., 200.])),
)

# Coarser level
result = read_points("scan.zarrvectors", level=1)
```

### Return dict schema

| Key | Type | Description |
|-----|------|-------------|
| `vertex_count` | `int` | Total vertices in result |
| `positions` | `ndarray (N, D) float32` | Vertex positions |
| `attributes` | `dict[str, ndarray]` | Per-vertex attributes |
| `level` | `int` | Resolution level read |
| `bbox_used` | `tuple or None` | Effective bbox after clipping |

### Attribute dtypes

Any numeric dtype supported by Zarr is valid for attribute arrays.
Common choices:

| Attribute type | Recommended dtype |
|---------------|-------------------|
| Continuous scalar (intensity, FA, concentration) | `float32` |
| Integer label / class ID | `int32` |
| Boolean mask | `uint8` (0 or 1) |
| High-precision scalar (double-precision required) | `float64` |
| RGB colour | `uint8` with attribute shape `(N, 3)` |

### Validation rules

L1: `vertices/` exists at each level (an error if absent);
`vertex_fragments/` is a warning if absent.

L3 (point-cloud-specific): because the store is an undifferentiated
point cloud, L3 additionally enforces the bin layout — a chunk's
fragment count must not exceed the product of `bins_per_chunk`, and a
spot check of the first 3 chunks per level warns when a fragment's
points fall outside their bin's bounds. These checks apply **only** to
stores that are point-cloud-only and have no `object_index`.

L4: a point cloud requires no `links_convention`, and the presence of a
`links/0` array emits a **warning**.

Absence of `links/` / `object_index/` is not otherwise checked, and
attribute-to-vertex length alignment is not checked at any level. See
[Validation overview](../validation/overview.md).

L3:
- For every non-empty chunk, the sum of range-fragment `count` values
  plus explicit-fragment index counts equals the number of vertex rows
  in that chunk.
- Attribute array length matches `vertices/` array length at every chunk.

### Multi-resolution behaviour

Point cloud pyramids use pure spatial coarsening (no object thinning).
At each coarser level, vertices within each bin are replaced by a single
metanode at the bin centroid. Per-vertex attributes are aggregated using
the declared strategy (default: `mean` for float attributes, `majority`
for integer attributes).

`object_sparsity` is always `1.0` for point clouds and cannot be set
otherwise.
