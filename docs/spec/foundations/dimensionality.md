# Dimensionality

## Terms

**Spatial dimensionality (D)**
: The number of spatial axes in the coordinate system of a Zarr Vectors store. The
  most common value is `D = 3` (x, y, z). Values of `D = 2` (planar data)
  and `D = 4` (e.g. x, y, z, t for time-series tractography) are also valid.
  `D` is the number of axes in `multiscales[0].axes` (called `sid_ndim`
  in the code) and must be consistent across all arrays and all resolution
  levels.

**Position array shape**
: The shape of the vertex data within a single chunk cell of `vertices/`:
  `(N, D)`, where `N` is the number of vertices in that chunk and `D` is
  the spatial dimensionality. `N` varies per chunk (ragged); `D` is fixed
  for the store.

**`chunk_shape` / `bin_shape` tuple length**
: Both `chunk_shape` and `bin_shape` are D-tuples. They must have exactly
  `D` elements, one per spatial axis.

**Axis names**
: Optional string labels for each spatial dimension, stored in the
  OME-Zarr-compatible `axes` field of the `multiscales` metadata block
  (`multiscales[0].axes`, the only place axes are stored). Defaults, when
  no axes are given: `x`, `y`, `z`, `w` (first `D` of them).

**Axis units**
: Optional physical unit strings associated with each axis (e.g.
  `"micrometer"`, `"nanometer"`, `"voxel"`). Stored in the `axes` field
  alongside axis names.

---

## Introduction

The Zarr Vectors is not inherently three-dimensional. The spatial
dimensionality `D` is a property of each individual store, declared at
write time and embedded in the root metadata. All format rules (chunk grid,
bin grid, fragment index addressing, bounding-box queries) generalise
straightforwardly to arbitrary `D`.

In practice, the vast majority of Zarr Vectors stores are three-dimensional (x, y,
z), matching the dimensionality of the volumetric image data they are
registered against. Two-dimensional stores arise in planar microscopy
contexts. Four-dimensional stores (`x, y, z, t`) are an emerging use case
for time-series tractography and dynamic cell tracking.

This page documents how dimensionality is declared, enforced, and used
throughout the format.

---

## Technical reference

### Declaring dimensionality

`D` is resolved when a store is created, from whichever of `ndim`, `axes`,
`chunk_shape` and `bounds` are supplied to `create_store`; they must agree
or `MetadataError("Inconsistent dimensionality ...")` is raised. If none is
given it defaults to 3. The type-level writers infer it from the data
(`positions.shape[1]`, `endpoints.shape[2]`, ...) and pass it on. It is
stored as the length of the axes list and of `chunk_shape` / `bounds`,
not as its own key:

```json
{
  "zarr_vectors": {
    "zv_version": "0.9.2",
    "geometry_types": ["point_cloud"],
    "chunk_shape": [200.0, 200.0, 200.0],
    "base_bin_shape": [50.0, 50.0, 50.0],
    "bounds": [[0, 0, 0], [1000, 1000, 1000]]
  },
  "multiscales": [{"axes": [
    {"name": "x", "type": "space"},
    {"name": "y", "type": "space"},
    {"name": "z", "type": "space"}
  ]}]
}
```

There is no `spatial_dims` key. There is no mechanism to change `D`
after a store is created. Default axis names exist only up to four
dimensions (`DEFAULT_AXES_NAMES = ("x", "y", "z", "w")`); with `D > 4`
`create_store` raises `MetadataError` unless explicit `axes` are passed.

### Effect on array shapes

Each logical array is a single Zarr v3 **vlen-bytes** array whose shape is
the level's chunk grid (rank `D`, one cell per spatial chunk); the
per-chunk payloads are variable-length byte blobs. The table shows the
logical shape of what is *in* a cell, and how it varies with `D`:

| Array | Content of one cell | Notes |
|-------|---------------------|-------|
| `vertices/` | `(N, D)` | N varies per chunk |
| `vertex_fragments/` | fragment-index blob | size grows with the number of fragments in the chunk, independent of D |
| `link_fragments/` | fragment-index blob | same structure, indexes the intra-chunk (all-zero offsets) link rows |
| `links/<delta>/<offsets>/` | `(R, L)` rows | R varies; `L` = `link_width` (2 for edges, ≥ 3 for faces); independent of D |
| `object_index/manifests` | one manifest blob per object | each entry's `chunk_coords` have `D` components |

The `<offsets>` **path segment** does depend on D: it carries
`link_width - 1` offsets, each of `sid_ndim` signed components (see
[Links](../object_model/links.md)). The array *contents* do not.

### Effect on chunk and bin addressing

The chunk coordinate of a vertex with position `(p_0, p_1, …, p_{D-1})`
is:

```
chunk_coord[i] = floor(p[i] / chunk_shape[i])   for i in 0..D-1
```

The bin coordinate within a chunk is:

```
bin_coord[i] = floor((p[i] % chunk_shape[i]) / bin_shape[i])   for i in 0..D-1
```

The number of bins per chunk is:

```
bins_per_chunk = product(chunk_shape[i] / bin_shape[i]   for i in 0..D-1)
```

For `D = 3` with `chunk_shape = (200, 200, 200)` and
`bin_shape = (50, 50, 50)`: `bins_per_chunk = 4 × 4 × 4 = 64`.

For `D = 2` with `chunk_shape = (500, 500)` and `bin_shape = (100, 100)`:
`bins_per_chunk = 5 × 5 = 25`.

For an undifferentiated point cloud the fragment index within a chunk is the
bin's linear index; the bin coordinate is flattened to a scalar using
C-order (row-major) ravelling, independent of `D`:

```
bin_flat_index = ravel_multi_index(bin_coord, bins_per_chunk_shape)
```

where `bins_per_chunk_shape = tuple(chunk_shape[i] / bin_shape[i] for i in range(D))`.

### Bounding-box queries

A bounding box is specified as a pair of D-dimensional arrays:

```python
bbox = (
    np.array([100.0, 100.0, 100.0]),   # lower corner, shape (D,)
    np.array([300.0, 300.0, 200.0]),   # upper corner, shape (D,)
)
```

For `D = 2`:

```python
bbox = (np.array([0.0, 0.0]), np.array([500.0, 500.0]))
```

The query engine computes the chunk range as:

```
chunk_min[i] = floor(bbox[0][i] / chunk_shape[i])
chunk_max[i] = floor(bbox[1][i] / chunk_shape[i])
```

and then the bin range within each overlapping chunk. All arithmetic is
D-dimensional throughout.

### Axis metadata

Axis names and units live in `multiscales[0].axes` (see
[Multiscale metadata](../multiscale/multiscale_metadata.md)). Example for a
3-D store in micrometres:

```javascript
{
  "multiscales": [
    {
      "version": "0.4",
      "axes": [
        {"name": "x", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "z", "type": "space", "unit": "micrometer"}
      ],
      "datasets": [...]
    }
  ]
}
```

Axis order in the `axes` array must match the axis order of `chunk_shape`,
`bin_shape`, and vertex coordinates (column `i` of a position is axis `i`).
`zarr-vectors-py` defaults to `x, y, z, w` order. There is no
`z, y, x` default. To set names and units, create the store with
`create_store(..., axes=[{"name": ..., "type": "space", "unit": ...}, ...])`,
or declare `zv.Schema(axes=(zv.Axis("x", "micrometer"), ...))`. The
type-level `write_*` functions have no axes argument.

### Four-dimensional stores

For a fourth axis (e.g. time), the extra axis is treated identically to
the spatial axes for chunking and binning purposes. `chunk_shape` and
`bin_shape` include a component for it, and the default fourth axis name
is `w` unless you declare another:

```python
import numpy as np
from zarr_vectors.types.points import write_points, read_points

rng = np.random.default_rng(0)
positions_4d = rng.uniform(0, 100, (5000, 4)).astype(np.float32)   # (N, 4)

write_points(
    "timeseries.zarrvectors",
    positions_4d,
    chunk_shape=(50.0, 50.0, 50.0, 50.0),
    bin_shape=(10.0, 10.0, 10.0, 10.0),
)

result = read_points(
    "timeseries.zarrvectors",
    bbox=(np.array([0.0, 0.0, 0.0, 0.0]),
          np.array([100.0, 100.0, 100.0, 25.0])),   # last axis in [0, 25]
)
```

There are no special-case code paths for `D = 4`; the general D-dimensional
logic handles it uniformly. Two limits to know: Draco mesh encoding applies
only to 3-D stores (it is silently skipped otherwise), and auto-generated
axis names stop at four.

### Validation

The validators enforce:

- L2: `sid_ndim >= 1`; `len(chunk_shape)`, the two `bounds` corners and
  `base_bin_shape` (if present) all equal `sid_ndim`; per-level `bin_shape`
  / `bin_ratio` lengths are checked too, but only where the level carries
  those values as flat attributes (see the "Known gap" note in
  [L2](../validation/l2_metadata.md)).
- L3: every decoded vertex fragment is 2-D with second dimension equal to
  `sid_ndim`.
