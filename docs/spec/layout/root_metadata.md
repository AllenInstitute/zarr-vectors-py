# Root metadata

## Terms

**Root `.zattrs`**
: The JSON metadata file at the store root path. Contains all Zarr Vectors-level
  metadata that applies to the store as a whole — geometry type, spatial
  dimensionality, base bin shape, chunk shape, coordinate system, and
  OME-Zarr multiscale metadata.

**`zarr_vectors_version`**
: A semantic version string declaring the Zarr Vectors specification version the store
  conforms to. Current value: `"1.0"`. Readers should check this field and
  warn on unknown versions.

**`geometry_type`**
: A string constant identifying the geometry type stored in this store.
  One of: `"point_cloud"`, `"line"`, `"polyline"`, `"streamline"`,
  `"graph"`, `"skeleton"`, `"mesh"`. A single Zarr Vectors store holds exactly one
  geometry type.

**`base_bin_shape`**
: The bin shape at the full-resolution level (level 0), as a D-tuple of
  positive floats. Bin shapes at coarser levels are computed from this value
  and the per-level `bin_ratio`.

**`chunk_shape`**
: The spatial extent of each chunk, as a D-tuple of positive floats. Fixed
  across all resolution levels.

**`coordinate_system`**
: A string identifying the spatial coordinate system of the vertex
  coordinates. Informational only; no coordinate transformation is applied
  by `zarr-vectors-py`. Common values: `"RAS"`, `"LPS"`, `"LAS"`,
  `"voxel"`.

**`multiscales`**
: An OME-Zarr-compatible array of per-level multiscale metadata objects.
  Described in full in [Multiscale metadata](../multiscale/multiscale_metadata.md).

---

## Introduction

The root `.zattrs` file is the entry point for any tool reading a Zarr Vectors store.
It declares what kind of data the store contains, how it is laid out
spatially, and how its resolution levels relate to one another. A reader can
determine everything it needs to open and query the store from this single
file plus the per-level `.zattrs` files.

This page documents every key in the root `.zattrs`, distinguishing required
keys (whose absence should cause a validation error) from recommended and
optional keys.

---

## Technical reference

### Complete example (streamline store, 3-D, two resolution levels)

```json
{
  "zarr_vectors_version": "1.0",
  "geometry_type":        "streamline",
  "spatial_dims":         3,
  "base_bin_shape":       [50.0, 50.0, 50.0],
  "chunk_shape":          [200.0, 200.0, 200.0],
  "coordinate_system":    "RAS",
  "axis_units":           "micrometer",
  "bounding_box": {
    "min": [0.0, 0.0, 0.0],
    "max": [8000.0, 6000.0, 4000.0]
  },
  "creation_timestamp":   "2024-11-15T14:32:00Z",
  "source_description":   "Human brain white-matter tractography, subject 001",
  "multiscales": [
    {
      "version": "0.5",
      "axes": [
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"}
      ],
      "datasets": [
        {
          "path":            "0",
          "level":           0,
          "bin_ratio":       [1, 1, 1],
          "bin_shape":       [50.0, 50.0, 50.0],
          "object_sparsity": 1.0,
          "coordinateTransformations": [
            {"type": "scale",       "scale":       [1.0, 1.0, 1.0]},
            {"type": "translation", "translation": [25.0, 25.0, 25.0]}
          ]
        },
        {
          "path":            "1",
          "level":           1,
          "bin_ratio":       [2, 2, 2],
          "bin_shape":       [100.0, 100.0, 100.0],
          "object_sparsity": 0.5,
          "coordinateTransformations": [
            {"type": "scale",       "scale":       [2.0, 2.0, 2.0]},
            {"type": "translation", "translation": [50.0, 50.0, 50.0]}
          ]
        }
      ],
      "type": "zarr_vectors_multiscale"
    }
  ]
}
```

### Required keys

| Key | Type | Description |
|-----|------|-------------|
| `zarr_vectors_version` | `string` | Spec version. Must be `"1.0"` for stores conforming to this document. |
| `geometry_type` | `string` | One of the seven geometry type constants. |
| `spatial_dims` | `integer` | Number of spatial dimensions D. Must equal `len(base_bin_shape)` and `len(chunk_shape)`. |
| `base_bin_shape` | `[float, …]` | D-tuple. Bin shape at level 0. Each element must be positive and must evenly divide the corresponding element of `chunk_shape`. |
| `chunk_shape` | `[float, …]` | D-tuple. Spatial extent of each chunk. Constant across all levels. Each element must be positive. |
| `multiscales` | `[object, …]` | OME-Zarr-compatible multiscale metadata array. Must contain one entry per resolution level. |

### Recommended keys

| Key | Type | Default if absent | Description |
|-----|------|-------------------|-------------|
| `coordinate_system` | `string` | `"unknown"` | Coordinate system identifier. |
| `axis_units` | `string` or `[string, …]` | `"unknown"` | Physical units for the spatial axes. Scalar if all axes share the same unit; D-tuple if axes have different units. |
| `bounding_box` | `{min: […], max: […]}` | Not computed | Bounding box of all vertices at level 0 in the declared coordinate system. |
| `creation_timestamp` | `string` (ISO 8601) | Not recorded | UTC timestamp of store creation. |
| `source_description` | `string` | — | Human-readable description of the data source. |

### Optional keys

| Key | Type | Description |
|-----|------|-------------|
| `notes` | `string` | Free-text notes for human readers. |
| `custom_metadata` | `object` | Application-specific metadata. `zarr-vectors-py` does not read or validate this field. |
| `draco_compressed` | `boolean` | Set to `true` if any array uses the Draco codec. Informational. |

### Validation rules

The L1 validator checks that `zarr_vectors_version`, `geometry_type`,
`spatial_dims`, `base_bin_shape`, and `chunk_shape` are present and have
the correct types.

The L2 validator checks:

1. `geometry_type` is one of the seven recognised constants.
2. `len(base_bin_shape) == spatial_dims` and `len(chunk_shape) == spatial_dims`.
3. For each dimension `i`: `chunk_shape[i] % base_bin_shape[i] == 0`
   (bin shape evenly divides chunk shape). Floating-point comparison uses a
   tolerance of `1e-6 × chunk_shape[i]`.
4. All elements of `base_bin_shape` and `chunk_shape` are strictly positive.
5. `multiscales` is a non-empty list and each entry references a path that
   exists as a group in the store.
6. Level 0 is present in `multiscales` with `bin_ratio == [1, 1, …, 1]`.
7. `bin_ratio` values are monotonically non-decreasing across levels (level 0
   has the smallest bins; coarser levels have larger bins).

### Writing root metadata from Python

`zarr-vectors-py` writes root metadata automatically during `write_*`
calls. To inspect or modify it directly:

```python
from zarr_vectors.core.store import open_store

root = open_store("scan.zarrvectors", mode="r")
print(root.attrs.asdict())      # read all root .zattrs

root = open_store("scan.zarrvectors", mode="r+")
root.attrs["source_description"] = "Corrected label v2"
```

To regenerate the `multiscales` block after manually adding a resolution
level:

```python
from zarr_vectors.core.multiscale import write_multiscale_metadata

root = open_store("scan.zarrvectors", mode="r+")
write_multiscale_metadata(root)
```
