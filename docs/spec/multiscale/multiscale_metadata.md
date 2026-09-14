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
  the `multiscales` block. Zarr Vectors uses two transforms per level: a `scale`
  transform (encoding `bin_ratio`) and a `translation` transform (encoding
  the centroid offset of the coarsened bins).

**Scale transform**
: A multiplicative factor per axis applied to convert a coordinate in
  level-`N` space back to level-0 physical space. For a level with
  `bin_ratio = [2, 2, 2]`, the scale is `[2.0, 2.0, 2.0]` — a point in
  level-1 coordinates must be multiplied by 2 to recover its level-0
  physical position.

**Translation transform**
: An additive offset per axis applied after the scale transform. For Zarr Vectors
  metanodes (binned vertices), the translation encodes the shift from the
  bin origin to the bin centroid: `translation[d] = bin_shape[d] / 2`.

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
          {"type": "scale",       "scale":       [2.0, 2.0, 2.0]},
          {"type": "translation", "translation": [50.0, 50.0, 50.0]}
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

**Scale:** encodes the bin ratio. For level `N` with `bin_ratio = [r_0, …, r_{D-1}]`:

```json
{"type": "scale", "scale": [r_0, r_1, ..., r_{D-1}]}
```

A viewer multiplies metanode coordinates by the scale to recover physical
coordinates.

**Translation:** encodes the bin centroid offset. For level `N` with
`bin_shape = [b_0, …, b_{D-1}]`:

```json
{"type": "translation", "translation": [b_0/2, b_1/2, ..., b_{D-1}/2]}
```

A metanode at a bin's origin has its reported coordinate at the bin centre
after applying the translation.

**Combined for level 1, `bin_ratio=[2,2,2]`, `bin_shape=[100,100,100]`:**

A metanode at position `[50.0, 100.0, 75.0]` in level-1 coordinates maps to
physical position `[50.0×2 + 50.0, 100.0×2 + 50.0, 75.0×2 + 50.0]` =
`[150.0, 250.0, 200.0]` µm.

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
print(get_level_scale(root, level=1))        # [2.0, 2.0, 2.0]
print(get_level_translation(root, level=1))  # [50.0, 50.0, 50.0]
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
- `scale[d] == bin_ratio[d]` for all `d`.
- `translation[d] == bin_shape[d] / 2` for all `d` (within tolerance).
