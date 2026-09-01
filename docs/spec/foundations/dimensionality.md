# Dimensionality

## Terms

**Spatial dimensionality (D)**
: The number of spatial axes in the coordinate system of a Zarr Vectors store. The
  most common value is `D = 3` (x, y, z). Values of `D = 2` (planar data)
  and `D = 4` (e.g. x, y, z, t for time-series tractography) are also valid.
  `D` is declared in the store's root `.zattrs` metadata and must be
  consistent across all arrays and all resolution levels.

**Position array shape**
: The shape of the `vertices/` array within a single chunk group: `(N, D)`,
  where `N` is the number of vertices in that chunk and `D` is the spatial
  dimensionality. `N` varies per chunk (ragged); `D` is fixed for the store.

**`chunk_shape` / `bin_shape` tuple length**
: Both `chunk_shape` and `bin_shape` are D-tuples. They must have exactly
  `D` elements, one per spatial axis.

**Axis names**
: Optional string labels for each spatial dimension, stored in the
  OME-Zarr-compatible `axes` field of the `multiscales` metadata block.
  Typical values: `["x", "y", "z"]`, `["r", "a", "s"]` (RAS), `["col",
  "row", "slice"]` (voxel index).

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

`D` is inferred from the length of `chunk_shape` at write time and stored in
the root `.zattrs` under the key `"spatial_dims"`:

```json
{
  "zarr_vectors_version": "1.0",
  "geometry_type": "point_cloud",
  "spatial_dims": 3,
  "base_bin_shape": [50.0, 50.0, 50.0],
  "chunk_shape": [200.0, 200.0, 200.0]
}
```

There is no mechanism to change `D` after a store is created. Adding a
resolution level with a different dimensionality will raise `ValueError`.

### Effect on array shapes

Every dimensionality-dependent array shape in the format uses `D`
explicitly. The table below shows how shapes vary with `D`:

| Array | Shape | Notes |
|-------|-------|-------|
| `vertices/` (per chunk) | `(N, D)` | N varies per chunk |
| `vertex_fragments/` (per chunk) | opaque `uint8` blob, length depends on `F` | Fragment-index binary blob; size grows with the number of fragments in the chunk, independent of D |
| `link_fragments/` (per chunk) | opaque `uint8` blob, length depends on `F` | Same structure as `vertex_fragments/` but indexes `links/0/<chunk>` rows |
| `links/<delta>/<offsets>/` (per source chunk) | `(R, L)` | R varies; `L` = `link_width` (2 for edges, ≥ 3 for faces); independent of D |
| `object_index/data` | ragged `uint8` (concatenated manifest blobs) | Each blob's `chunk_coords` words have `D` int64s; total size grows with manifest count and complexity |
| `object_index/offsets` | `(n_objects,)` `int64` | Independent of D |

The `<offsets>` **path segment** does depend on D: it carries
`link_width - 1` offsets, each of `sid_ndim` signed components (see
[Links](../object_model/links.md)). The array *contents* do not.

The `chunk_grid` of the `vertices/` Zarr array reflects `D`:

```json
{
  "chunk_grid": {
    "name": "regular",
    "configuration": { "chunk_shape": [65536, 3] }
  }
}
```

Here `65536` is a soft cap on vertices per chunk (the actual number is
ragged; the array is extended as needed); `3` is `D`.

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

The fragment index array is always `(bins_per_chunk, 2)` regardless of `D`; the
bin coordinate is flattened to a scalar index using C-order (row-major)
ravelling:

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

For stores that will be visualised or processed by OME-Zarr-aware tools,
axis names and units should be provided via the `multiscales` metadata
block. Example for a 3-D RAS store in micrometres:

```json
{
  "multiscales": [
    {
      "version": "0.5",
      "axes": [
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"}
      ],
      "datasets": [...]
    }
  ]
}
```

Axis order in the `axes` array must match the axis order of `chunk_shape`,
`bin_shape`, and vertex coordinates. `zarr-vectors-py` defaults to `z, y,
x` order (slowest to fastest varying), matching the OME-Zarr convention for
volumetric image data.

### Four-dimensional stores

For time-series data (`D = 4`, axes `t, z, y, x`), the temporal axis is
treated identically to the spatial axes for chunking and binning purposes.
`chunk_shape` and `bin_shape` include a temporal component:

```python
write_points(
    "timeseries.zarrvectors",
    positions_4d,                        # shape (N, 4): (t, z, y, x)
    chunk_shape=(10.0, 200.0, 200.0, 200.0),   # 10 time units × 200³ space
    bin_shape=(5.0, 50.0, 50.0, 50.0),
)
```

Bounding-box queries may include a temporal range:

```python
result = read_points(
    "timeseries.zarrvectors",
    bbox=(np.array([0.0, 0.0, 0.0, 0.0]),
          np.array([5.0, 200.0, 200.0, 200.0])),   # t in [0, 5)
)
```

There are no special-case code paths for `D = 4`; the general D-dimensional
logic handles it uniformly.

### Validation

The validator enforces:

- `spatial_dims` in `.zattrs` equals the length of `base_bin_shape` and
  `chunk_shape` (L2).
- Every `vertices/` array's second dimension equals `spatial_dims` (L3).
- Every `bin_shape` and `chunk_shape` tuple at every resolution level has
  length `spatial_dims` (L2).
- Bounding-box arguments passed to read functions have shape `(D,)` (runtime
  assertion, not a stored validation level).
