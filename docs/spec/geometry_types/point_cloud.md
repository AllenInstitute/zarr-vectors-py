# Point cloud (`point_cloud`)

## Terms

**Point cloud**
: An unordered collection of spatial positions (vertices) with no
  connectivity information. Each vertex is independent. Per-vertex scalar
  or vector attributes may be stored alongside positions.

**`GEOM_POINT_CLOUD`**
: The geometry type constant `"point_cloud"`. Stored in the root
  `zarr_vectors` attribute block as `"geometry_types": ["point_cloud"]`.

**Per-vertex attribute**
: A named array, stored under `vertex_attributes/<name>/`, aligned row for
  row with `vertices/` (one row per vertex, in fragment order).

---

## Introduction

The point cloud type is the simplest Zarr Vectors geometry: positions plus optional
per-vertex attributes, no connectivity. It is suitable for
any dataset where vertices are independent measurements — lidar scans,
synchrotron absorption point data, single-molecule localisation microscopy,
gene expression spatial transcriptomics, or any other spatially indexed
scalar field sampled at discrete positions.

`write_points` can optionally record an object assignment per point
(`object_ids=`), per-object attributes, and groups of objects; the store
is then no longer an "undifferentiated" cloud (it gains `object_index/`,
and fragments become per-object rather than per-bin). Without those
arguments there is no object model.

Point clouds support the full Zarr Vectors spatial indexing hierarchy (chunks, bins,
fragment index) and multi-resolution pyramids via spatial coarsening into
metavertices (see [Pyramid construction](../multiscale/pyramid_construction.md)).

---

## Technical reference

### Arrays present

| Array path | Required | Description |
|-----------|----------|-------------|
| `vertices/` | Yes | Vertex positions, `(N, D)` per chunk, `float32` by default (`dtype=` selects another) |
| `vertex_fragments/` | Yes | Fragment index per chunk; see [Fragment-index arrays](../layout/fragment_index_arrays.md) |
| `vertex_attributes/<name>/` | No | Per-vertex scalar or vector attributes |
| `fragment_attributes/<name>/` | No | Per-fragment attributes, if `fragment_attributes=` was given |
| `object_index/`, `object_attributes/<name>/`, `groups/`, `group_attributes/<name>/` | No | Only when `object_ids=` / `object_attributes=` / `groups=` / `group_attributes=` were given |

No `links/` family is written.

### Root metadata

Format parameters are in the root `zarr_vectors` attribute block and the
axes in `multiscales[0].axes` (see
[Root metadata](../layout/root_metadata.md)). For a point cloud:

```json
{
  "zarr_vectors": {
    "zv_version": "0.9.2",
    "geometry_types": ["point_cloud"],
    "chunk_shape": [200.0, 200.0, 200.0],
    "base_bin_shape": [50.0, 50.0, 50.0],
    "bounds": [[0, 0, 0], [1000, 1000, 1000]],
    "links_convention": "implicit_sequential",
    "object_index_convention": "standard",
    "cross_chunk_strategy": "explicit_links"
  }
}
```

(`geometry_types` is a list; there is no `geometry_type` or
`spatial_dims` key. `base_bin_shape` is present only when `bin_shape=`
was passed. Values shown are illustrative.)

### Write API

```python
import numpy as np
from zarr_vectors.types.points import write_points

rng = np.random.default_rng(0)
positions = rng.uniform(0, 1000, (100_000, 3)).astype(np.float32)
intensity = rng.uniform(0, 1, 100_000).astype(np.float32)
label     = rng.integers(0, 8, 100_000).astype(np.int32)

summary = write_points(
    "scan.zarrvectors",
    positions,
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
    vertex_attributes={
        "intensity": intensity,    # float32 scalar
        "label":     label,        # int32 categorical
    },
)
# summary: {"vertex_count", "chunk_count", "object_count",
#           "group_count", "bins_per_chunk"}
```

Signature (keyword-only after `positions`): `chunk_shape`, `bin_shape`,
`bounds`, `vertex_attributes`, `object_ids`, `object_attributes`,
`fragment_attributes`, `groups`, `group_attributes`, `dtype="float32"`,
`backend`, `chunk_by_attribute`, `out_of_bounds="raise"`, `compressor`,
`shard_shape`. The older `attributes=` keyword still works but emits a
`DeprecationWarning`. There are no `coordinate_system` / `axis_units`
keywords.

- If `chunk_shape` is omitted, all points go in a single chunk.
- If `bin_shape` is omitted it equals `chunk_shape` (one bin per chunk).
- A new store's `bounds` default to the extent of `positions` when
  `bounds=` is not given; `out_of_bounds` (`"raise"` / `"ignore"` /
  `"expand"`) decides what happens to points outside the store's bounds
  on a later write.
- String / object-dtype attributes are dictionary-encoded automatically.

### Read API

```python
import numpy as np
from zarr_vectors.types.points import read_points

# Read entire store. Attributes must be requested by name:
# attribute_names=None (the default) returns none.
result = read_points("scan.zarrvectors", attribute_names=["intensity"])
print(result["vertex_count"])                   # int
print(result["positions"].shape)                # (N, 3)
print(result["vertex_attributes"]["intensity"]) # (N,) float32

# Spatial bbox query
result = read_points(
    "scan.zarrvectors",
    bbox=(np.array([100., 100., 100.]), np.array([200., 200., 200.])),
)

# Coarser level (requires a pyramid; see multiscale docs)
result = read_points("scan.zarrvectors", level=1)
```

`read_points` also accepts `object_ids=`, `group_ids=`, `chunks=`,
`attribute_filter=` and `backend=`; all supplied filters are ANDed.

For new code, the dataset API (`zarr_vectors.open(...)`, `Level.read()`)
wraps these functions and resolves `attributes="all"` for you.

### Return dict schema

| Key | Type | Description |
|-----|------|-------------|
| `vertex_count` | `int` | Total vertices in result |
| `positions` | `ndarray (N, D)` | Vertex positions (store dtype, `float32` by default) |
| `vertex_attributes` | `dict[str, ndarray]` | Requested per-vertex attributes |
| `object_ids` | `ndarray (N,)` | Present only for reads by object / group id |

### Attribute dtypes

Any numeric dtype supported by Zarr is valid for attribute arrays;
string attributes are dictionary-encoded as integer codes plus a
`categories` table.
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

L4: a point cloud requires no particular `links_convention`, and the
presence of a `links/0` array emits a **warning** (only when no graph,
skeleton, or mesh type is also declared).

Absence of `links/` / `object_index/` is not otherwise checked, and
attribute-to-vertex length alignment is not checked at any level. See
[Validation overview](../validation/overview.md).

### Multi-resolution behaviour

Core pyramids (`build_pyramid`, default `method="per_object"`) bin
vertices into metavertices at the bin centroid and carry no per-vertex
attributes to the coarser level. Attribute aggregation modes
(`mean`, `sum`, `mode`, `count`, `min`, `max`) are a vocabulary used by
`zarr-vectors-tools` coarseners, not by core; see
[Pyramid construction](../multiscale/pyramid_construction.md).

Object thinning does not apply to a point cloud without `object_ids`
(see [Object sparsity](../multiscale/sparsity.md)).
