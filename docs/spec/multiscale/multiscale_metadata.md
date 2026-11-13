# Multiscale metadata

## Terms

**`multiscales` block**
: A list in the root attributes that describes the resolution pyramid of
  a Zarr Vectors store in the layout of the
  [OME-NGFF multiscales specification](https://ngff.openmicroscopy.org/)
  version 0.4.

**Coordinate transform**
: A `scale` or `translation` entry in a dataset's
  `coordinateTransformations` list.

---

## Introduction

The `multiscales` block lists every level with a `scale` and a
`translation` transform, so NGFF-aware tools can find the pyramid. It
also records each coarser level's bin geometry, which the
[level attributes](../layout/level_groups.md) do not carry.

---

## Technical reference

### Example

A streamline store with `base_bin_shape` `[50, 50, 50]` and a pyramid
built with `factors=[(2.0, 1.0), (2.0, 2.0)]`:

```javascript
"multiscales": [
  {
    "version": "0.4",
    "name": "default",
    "axes": [{"name": "x", "type": "space"}, ...],  // y and z alike
    "datasets": [
      {
        "path": "0",
        "coordinateTransformations": [
          {"type": "scale", "scale": [1.0, 1.0, 1.0]},
          {"type": "translation", "translation": [25.0, 25.0, 25.0]}
        ]
      },
      {
        "path": "1",
        "coordinateTransformations": [
          {"type": "scale", "scale": [2.0, 2.0, 2.0]},
          {"type": "translation", "translation": [50.0, 50.0, 50.0]}
        ]
      },
      ...   // path "2": scale [4.0, 4.0, 4.0], translation [100.0, 100.0, 100.0]
    ],
    "metadata": {"format": "zarr_vectors"}
  }
]
```

### Keys

| Key | Value |
|-----|-------|
| `version` | `"0.4"` |
| `name` | `"default"` |
| `axes` | One per spatial axis, from the root axes: `name`, `type` (default `"space"`), and `unit` when the axis has one (for example `create_store(..., unit="micrometer")`) |
| `datasets` | One per level, in ascending order |
| `datasets[].path` | The level group's name (`"0"`, `"1"`, …) |
| `datasets[].coordinateTransformations` | `scale`, then `translation` unless every offset is zero |
| `metadata.format` | `"zarr_vectors"`; there is no `type` key, which NGFF reserves for the downsampling method |

There is no top-level `coordinateTransformations` and no per-dataset
`level`, `bin_ratio`, `bin_shape` or `object_sparsity`.

### Transform values

| Level | `scale` | `translation` |
|-------|---------|---------------|
| 0 | all `1.0` | half the root's effective bin shape (`base_bin_shape`, or `chunk_shape` if there is none) |
| coarser | the level's `bin_ratio`; else `bin_shape / base_bin_shape`; else all `1.0` | half the level's `bin_shape`; else as level 0 |

For a coarser level, `read_level_metadata(root, level)` sets
`bin_ratio = scale` and `bin_shape = 2 × translation`. For level 0 it
reports neither; level 0 uses the root's `base_bin_shape`.

Vertex positions are in level-0 coordinates at every level (a
metavertex is the centroid of the vertices it replaces). The transforms
describe each level's bin grid; readers do not apply them to positions.

### Writing and reading the block

The `write_*` functions and `build_pyramid` write the block. To rewrite
it from the current level metadata:

```python
from zarr_vectors.core.multiscale import (
    get_level_scale,
    get_level_translation,
    write_multiscale_metadata,
)
from zarr_vectors.core.store import open_store

root = open_store("scan.zarrvectors", mode="r+")
write_multiscale_metadata(root)

print(get_level_scale(root, level=1))        # [2.0, 2.0, 2.0]
print(get_level_translation(root, level=1))  # [50.0, 50.0, 50.0]
```

`read_multiscale_metadata(root)` returns the block, or `None` when the
root has none.

### Validation

No validator reads the `multiscales` block.
