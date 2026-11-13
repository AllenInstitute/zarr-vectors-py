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

The `multiscales` block lists every resolution level and gives each one
a `scale` and a `translation` transform. It follows the OME-NGFF layout
so that NGFF-aware tools can discover the pyramid. It is also where a
coarser level's bin geometry is recorded: the level's attributes
([Level groups](../layout/level_groups.md)) do not carry `bin_shape` or
`bin_ratio`, and `read_level_metadata` takes them from this block.

---

## Technical reference

### Example

The block of a streamline store with `base_bin_shape` `[50, 50, 50]`
and a pyramid built with `factors=[(2.0, 1.0), (2.0, 2.0)]`:

```json
"multiscales": [
  {
    "version": "0.4",
    "name": "default",
    "axes": [
      {"name": "x", "type": "space"},
      {"name": "y", "type": "space"},
      {"name": "z", "type": "space"}
    ],
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
      {
        "path": "2",
        "coordinateTransformations": [
          {"type": "scale", "scale": [4.0, 4.0, 4.0]},
          {"type": "translation", "translation": [100.0, 100.0, 100.0]}
        ]
      }
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
| `axes` | One entry per spatial axis, taken from the root metadata's axes: `name`, `type` (default `"space"`), and `unit` only when the axis has one (for example in a store created with `create_store(..., unit="micrometer")`) |
| `datasets` | One entry per resolution level, in ascending level order |
| `datasets[].path` | The level group's name (`"0"`, `"1"`, …) |
| `datasets[].coordinateTransformations` | A `scale` transform, followed by a `translation` transform unless every offset is zero |
| `metadata.format` | `"zarr_vectors"`, which marks the block as a Zarr Vectors pyramid. NGFF reserves `type` for the downsampling method, so the block has no `type` key |

The block has no top-level `coordinateTransformations` and no
per-dataset `level`, `bin_ratio`, `bin_shape` or `object_sparsity` key.

### Transform values

For level 0:

- `scale` is all `1.0`.
- `translation` is half the root's effective bin shape
  (`base_bin_shape`, or `chunk_shape` when the store declares no bin
  shape).

For a coarser level:

- `scale` is the level's `bin_ratio`; when the level has no `bin_ratio`,
  `bin_shape / base_bin_shape` per axis; when it has neither, all `1.0`.
- `translation` is half the level's `bin_shape`, or half the root's
  effective bin shape when the level has none.

A reader recovers a coarser level's bin geometry from its transforms:
`bin_ratio = scale` and `bin_shape = 2 × translation`.
`read_level_metadata(root, level)` does this, and for level 0 reports
neither (level 0 takes `base_bin_shape` from the root).

Vertex positions are stored in level-0 coordinates at every level: a
metavertex is the centroid of the positions it replaces. The transforms
describe each level's bin grid relative to level 0; a reader of Zarr
Vectors vertices does not apply them to positions.

### Writing and reading the block

The `write_*` functions and `build_pyramid` write the block. To rewrite
it from the levels' current metadata:

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

The validators in `zarr_vectors.validate` do not read the `multiscales`
block.
