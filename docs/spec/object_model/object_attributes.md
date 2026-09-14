# Object attributes

## Terms

**Per-vertex attribute**
: A named array stored under `attributes/<name>/` whose length at each
  chunk equals the number of vertices in that chunk. Attribute `k` in
  chunk `(cx, cy, cz)` corresponds to vertex `k` in the same chunk (in
  fragment order).

**Per-fragment attribute**
: A named array stored under `fragment_attributes/<name>/` whose per-chunk
  byte length equals `num_fragments_in_chunk × dtype.itemsize × ncols`.
  Element `k` corresponds to fragment `k` as encoded in
  `vertex_fragments/<chunk>`. Opt-in; the common use case is materializing
  the OID owning each fragment without round-tripping the
  `object_index/manifests` reverse lookup.

**Per-object attribute**
: A named array stored under `object_attributes/<name>/` whose length
  equals `n_objects`. Element `k` corresponds to object ID `k`.

**Attribute dtype**
: The numeric data type of an attribute array, declared in the array's
  `zarr.json`. Any dtype supported by Zarr v3 is valid.

**Attribute aggregation**
: The strategy used to compute a per-vertex attribute value for a metanode
  during pyramid construction. See
  [Pyramid construction](../multiscale/pyramid_construction.md).

---

## Introduction

Zarr Vectors supports three levels of attribute granularity: per-vertex,
per-fragment, and per-object. Per-vertex attributes assign one value (or
vector) to each vertex independently. Per-fragment attributes assign one
value per fragment within a chunk (opt-in; useful for materializing
parent-IDs like the OID owning each fragment). Per-object attributes
assign one value to each discrete object as a whole (a streamline, a
skeleton, a mesh surface).

All three levels are optional and independent. A store may have any
subset: only per-vertex, only per-object, only per-fragment, or any
combination. A store may carry per-object `mean_fa` without any per-vertex
FA values, or vice versa.

This page documents the array schemas for both attribute types, naming rules,
dtype recommendations, and behaviour during multi-resolution coarsening.

---

## Technical reference

### Per-vertex attributes (`attributes/<name>/`)

#### Array schema

```
path:        <N>/attributes/<name>/
shape:       (*chunk_grid_shape, N_max)       for scalar attributes
             (*chunk_grid_shape, N_max, K)     for K-dimensional vector attributes
chunk_shape: (1, 1, …, 1, N_max)
             (1, 1, …, 1, N_max, K)
fill_value:  0.0  (float types) or  0  (integer types)
```

`N_max` must match the `N_max` of the corresponding `vertices/` array.
Attribute element `k` in chunk `(cx, cy, cz)` corresponds to vertex `k`
in `vertices[cx, cy, cz]`.

#### Naming rules

Attribute names must be valid Python identifiers (alphanumeric and
underscores, not starting with a digit). Reserved names used by the Zarr Vectors
spec:

| Name | Type | Used by |
|------|------|---------|
| `swc_type` | int32 | `skeleton` — SWC compartment type |
| `radius` | float32 | `skeleton` — estimated tube radius |
| `swc_id` | int64 | `skeleton` — original SWC row ID |

All other names are available for application-defined attributes.

#### Common per-vertex attributes

| Attribute | Dtype | Description |
|-----------|-------|-------------|
| `intensity` | float32 | Absorption / fluorescence intensity |
| `fa` | float32 | Fractional anisotropy (per streamline vertex) |
| `md` | float32 | Mean diffusivity |
| `label` | int32 | Semantic segmentation label |
| `confidence` | float32 | Detection confidence score |
| `color` | uint8 (N, 3) | RGB colour per vertex |
| `normal` | float32 (N, 3) | Surface normal per vertex (mesh) |
| `uv` | float32 (N, 2) | UV texture coordinates (mesh) |

#### Writing per-vertex attributes

```python
import numpy as np
from zarr_vectors.types.points import write_points

n = 100_000
positions  = np.random.default_rng(0).uniform(0, 1000, (n, 3)).astype(np.float32)
intensity  = np.random.default_rng(1).uniform(0, 1, n).astype(np.float32)
color      = np.random.default_rng(2).integers(0, 256, (n, 3)).astype(np.uint8)

write_points(
    "scan.zarrvectors", positions,
    chunk_shape=(200., 200., 200.),
    attributes={
        "intensity": intensity,   # scalar float32
        "color":     color,       # vector uint8 (N, 3)
    },
)
```

#### Reading per-vertex attributes

```python
from zarr_vectors.types.points import read_points

result = read_points("scan.zarrvectors")
intensity = result["attributes"]["intensity"]   # shape (N,)
color     = result["attributes"]["color"]       # shape (N, 3)

# Bbox query preserves attribute alignment
result = read_points("scan.zarrvectors",
                     bbox=(lo, hi),
                     attributes=["intensity"])  # read subset of attributes
```

### Per-object attributes (`object_attributes/<name>/`)

#### Array schema

```
path:        <N>/object_attributes/<name>/
shape:       (n_objects,)     for scalar attributes
             (n_objects, K)   for K-dimensional vector attributes
chunk_shape: (65536,)  or  (65536, K)
fill_value:  0.0  or  0
```

`n_objects` equals `object_index.shape[0]`. Element `k` corresponds to
object ID `k`.

#### Writing per-object attributes

```python
from zarr_vectors.types.polylines import write_polylines
import numpy as np

n_polylines = 500
polylines   = [...]   # list of (N_i, 3) arrays
lengths     = np.array([p.shape[0] for p in polylines], dtype=np.int32)
mean_fa     = np.random.default_rng(0).uniform(0.2, 0.8, n_polylines).astype(np.float32)

write_polylines(
    "tracts.zarrvectors", polylines,
    chunk_shape=(200., 200., 200.),
    object_attributes={
        "length":  lengths,    # int32 per polyline
        "mean_fa": mean_fa,    # float32 per polyline
    },
)
```

#### Reading per-object attributes

```python
from zarr_vectors.types.polylines import read_polylines

# Read object attributes without fetching vertex data
from zarr_vectors.core.store import open_store
root = open_store("tracts.zarrvectors", mode="r")
mean_fa = root["0"]["object_attributes"]["mean_fa"][:]

# Read object attributes alongside vertices
result = read_polylines(
    "tracts.zarrvectors",
    object_ids=[0, 5, 10],
    include_object_attributes=True,
)
print(result["object_attributes"]["mean_fa"])   # shape (3,)
```

#### Adding attributes to an existing store

Per-object attributes can be added to an existing store without rewriting
vertex data:

```python
from zarr_vectors.core.attributes import add_object_attribute

add_object_attribute(
    "tracts.zarrvectors",
    name="cluster_id",
    values=cluster_ids,   # shape (n_objects,) int32
    level=0,
)
```

Per-vertex attributes require rewriting the entire `attributes/` sub-group
for the affected level (because vertex ordering is fixed at write time).

### Attribute alignment at all resolution levels

Per-vertex attributes must be present at every resolution level where they
are declared. When building a pyramid:

```python
from zarr_vectors.multiresolution.coarsen import build_pyramid

build_pyramid(
    "scan.zarrvectors",
    factors=[(2.0, 1.0)],
    agg_mode="mean",  # 0.4+: a single global mode (per-attribute via manual coarsen_level)
)
```

The `attribute_aggregation` dict specifies the per-attribute aggregation
strategy. Attributes not listed use `mean` for float types and `majority`
for integer types.

For per-object attributes at levels with `object_sparsity < 1.0`, only
the retained objects' attribute values are stored. The mapping from
coarse-level object IDs to level-0 IDs is stored in
`object_attributes/level0_object_id/`.

### Groupings and group attributes

Groupings assign objects to named sets (e.g. tract bundles, cell types).
The `groupings/` array stores a 2-D padded array mapping group ID →
object IDs. The `groupings_attributes/` sub-group stores per-group metadata:

```python
write_polylines(
    "tracts.zarrvectors", polylines,
    chunk_shape=(200., 200., 200.),
    groups={
        "corticospinal": cst_ids,
        "arcuate":       af_ids,
    },
    groupings_attributes={
        "tract_name": ["corticospinal", "arcuate"],   # stored as string array
        "n_objects":  [len(cst_ids), len(af_ids)],
    },
)
```

Group names are stored in `groupings_attributes/name/` as a UTF-8 string
array. Group IDs are integer indices into this array.

### Validation

L1: Each declared attribute group (`attributes/`, `object_attributes/`)
has a `zarr.json` and at least one named sub-array.

L3:
- For each per-vertex attribute at level `N`: shape is consistent with
  the `vertices/` array at the same level.
- For each per-object attribute at level `N`: shape is `(n_objects,)` or
  `(n_objects, K)` where `n_objects == object_index.shape[0]`.
- Attribute values are finite (no NaN or Inf unless explicitly declared
  as the fill value).
