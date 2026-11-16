# Multiscale metadata

## Terms

**`multiscales` block**
: An array of objects in root `.zattrs` that describes the resolution
  pyramid of a Zarr Vectors store. The structure follows the
  [OME-Zarr NGFF multiscales specification](https://ngff.openmicroscopy.org/),
  extended with Zarr Vectors-specific keys (`bin_ratio`, `bin_shape`,
  `object_sparsity`).

**Coordinate transform**
: An affine-like spatial transformation declared per resolution level in
  the `multiscales` block. Every level stores its vertices in level-0
  coordinates (a coarse level's metavertices are centroids of level-0
  positions), so every level carries level 0's transform: `scale` 1 and
  level 0's `translation`. A coarse level's `bin_ratio` and `bin_shape`
  are recorded in its own `zarr_vectors_level` attributes, not in the
  transform.

**Scale transform**
: A multiplicative factor per axis applied to a stored coordinate. Stored
  coordinates are level-0 coordinates at every level, so the scale is
  `[1.0, …, 1.0]` at every level. (Stores written before this was fixed
  carry `scale = bin_ratio` and `translation = bin_shape / 2` on coarse
  levels; readers take `bin_ratio` / `bin_shape` from that entry when the
  level's own attributes lack them, and `write_multiscale_metadata`
  moves them across and rewrites the entry.)

**Translation transform**
: An additive offset per axis applied after the scale transform; the same
  at every level as at level 0.

**OME-Zarr compatibility**
: The property that an OME-Zarr-aware tool (Neuroglancer, napari, OME-Zarr
  validators) can read the `multiscales` block of a Zarr Vectors store and correctly
  interpret the resolution pyramid, coordinate transforms, and axis
  metadata, without needing to understand Zarr Vectors-specific keys.

---

## Introduction

The `multiscales` block serves two purposes. First, it is the machine-
readable index of the resolution pyramid: it lists every level, its path,
and the parameters of the coarsening operation that produced it. Second,
it encodes the spatial coordinate transforms that relate each coarser level
back to physical space, following the OME-Zarr convention so that viewers
can display multi-resolution data with correct physical alignment.

By following OME-Zarr conventions, Zarr Vectors stores are discoverable by the
existing ecosystem of OME-Zarr tools — not just `zarr-vectors-py`. A viewer
that understands OME-Zarr multiscales (such as Neuroglancer with the
`zarr_vectors` data source, or a napari plugin) can open any Zarr Vectors store and
correctly interpret the resolution pyramid without Zarr Vectors-specific code.

---

## Technical reference

### Full schema

```json
"multiscales": [
  {
    "version":  "0.5",
    "name":     "scan",
    "type":     "zarr_vectors_multiscale",
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
          {"type": "scale",       "scale":       [1.0, 1.0, 1.0]},
          {"type": "translation", "translation": [25.0, 25.0, 25.0]}
        ]
      }
    ],
    "coordinateTransformations": [
      {"type": "scale", "scale": [1.0, 1.0, 1.0]}
    ]
  }
]
```

### Key descriptions

#### Top-level `multiscales` fields

| Key | Required | Type | Description |
|-----|----------|------|-------------|
| `version` | Yes | `string` | OME-Zarr NGFF version. `"0.5"` for current Zarr Vectors stores. |
| `name` | No | `string` | Human-readable name for the dataset. |
| `type` | Yes | `string` | Must be `"zarr_vectors_multiscale"` for Zarr Vectors stores. OME-Zarr tools ignore unknown `type` values. |
| `axes` | Yes | `[axis, …]` | D-length array of axis descriptors. See axis schema below. |
| `datasets` | Yes | `[dataset, …]` | One entry per resolution level, in ascending level-index order. |
| `coordinateTransformations` | No | `[transform, …]` | Global-level transforms applied to all levels (e.g. a global scale). |

#### `axes` schema

```json
{
  "name":  "z",
  "type":  "space",
  "unit":  "micrometer"
}
```

`type` must be `"space"` for spatial axes. `"time"` is valid for a temporal
axis in a 4-D store. `unit` follows the OME-Zarr unit vocabulary
(`"micrometer"`, `"nanometer"`, `"millimeter"`, `"voxel"`, etc.).

#### `datasets` entry schema

| Key | Required | Type | Description |
|-----|----------|------|-------------|
| `path` | Yes | `string` | Path to the level group relative to the store root. |
| `level` | Yes | `integer` | Level index. Must match the numeric suffix of `path`. |
| `bin_ratio` | Yes (ZV) | `[int, …]` | D-tuple. Ratio of this level's bin shape to `base_bin_shape`. |
| `bin_shape` | Yes (ZV) | `[float, …]` | D-tuple. Effective bin shape at this level. |
| `object_sparsity` | No (ZV) | `float` | Fraction of objects retained. Default `1.0`. |
| `coordinateTransformations` | Yes | `[transform, …]` | Per-level coordinate transforms. |

Fields marked `(ZV)` are Zarr Vectors extensions. OME-Zarr-aware tools will ignore
them; `zarr-vectors-py` requires them.

### Coordinate transforms

Zarr Vectors uses a pair of transforms per level:

```
[scale, translation]
```

Applied in order (scale first, then translation):

```
physical_position = metanode_position × scale + translation
```

**Scale:** `[1.0, …, 1.0]` at every level. A coarse level's vertices are
centroids of level-0 positions, already in level-0 coordinates; scaling
them by the bin ratio would draw level 1 at twice its size.

**Translation:** level 0's translation, repeated at every level, so all
levels land in the same place.

### OME-Zarr compatibility notes

Zarr Vectors `multiscales` blocks are valid OME-Zarr NGFF 0.5 multiscales. An
OME-Zarr reader will:

- Correctly read `axes`, `datasets[].path`, and `coordinateTransformations`.
- Ignore `type: "zarr_vectors_multiscale"` (not a standard OME-Zarr type).
- Ignore `bin_ratio`, `bin_shape`, `object_sparsity` (Zarr Vectors extensions).
- Attempt to open the `vertices/` array at each level as an image array
  (it will find a 5-D float32 array, which is valid but may not render
  sensibly in a volume viewer).

This compatibility allows Zarr Vectors stores to be indexed and discovered by OME-Zarr
metadata tools and registries, improving interoperability with the broader
NGFF ecosystem.

### Writing and updating multiscale metadata

`zarr-vectors-py` writes the `multiscales` block automatically during
`write_*` and `build_pyramid` calls. To regenerate after manual modifications:

```python
from zarr_vectors.core.multiscale import (
    write_multiscale_metadata,
    get_level_scale,
    get_level_translation,
)
from zarr_vectors.core.store import open_store

root = open_store("scan.zarrvectors", mode="r+")
write_multiscale_metadata(root)

# Inspect individual level transforms
print(get_level_scale(root, level=1))        # [1.0, 1.0, 1.0]
print(get_level_translation(root, level=1))  # [25.0, 25.0, 25.0] (level 0's)
```

### Validation

L2 validation checks:

- `multiscales` is a non-empty list.
- Exactly one entry has `level == 0` and `bin_ratio == [1,…,1]`.
- `datasets` entries are ordered by ascending `level`.
- For each entry: `path` references an existing group; `bin_ratio`,
  `bin_shape`, and `coordinateTransformations` are present and consistent.
- `coordinateTransformations` contains exactly one `scale` and one
  `translation` transform in that order.
- Every level's transform equals level 0's (stored coordinates are
  level-0 coordinates at every level).
