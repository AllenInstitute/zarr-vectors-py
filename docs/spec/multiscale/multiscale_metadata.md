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
`translation` transform, so NGFF-aware tools can find the pyramid. It is
also where the store's axes are stored. Each coarser level's bin
geometry (`bin_shape`, `bin_ratio`) is in its
[level attributes](../layout/level_groups.md), not in this block.

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
          {"type": "scale", "scale": [1.0, 1.0, 1.0]},
          {"type": "translation", "translation": [25.0, 25.0, 25.0]}
        ]
      },
      ...   // path "2": the same transforms
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
| `axes` | One per spatial axis, from the root axes: `name`, `type` (default `"space"`), and `unit` when the axis has one (for example `create_store(..., unit="micrometer")`). The only place the store's axes are stored |
| `datasets` | One per level, in ascending order |
| `datasets[].path` | The level group's name (`"0"`, `"1"`, …) |
| `datasets[].coordinateTransformations` | `scale`, then `translation` unless every offset is zero |
| `metadata.format` | `"zarr_vectors"`; there is no `type` key, which NGFF reserves for the downsampling method |

There is no top-level `coordinateTransformations` and no per-dataset
`level`, `bin_ratio`, `bin_shape` or `object_sparsity`.

### Transform values

Every level has level 0's transforms:

| `scale` | `translation` |
|---------|---------------|
| all `1.0` | half the root's effective bin shape (`base_bin_shape`, or `chunk_shape` if there is none) |

Vertex positions are in level-0 coordinates at every level (a
metavertex is the centroid of the vertices it replaces), so every level
is drawn in the same place. A coarser level's `bin_shape` and
`bin_ratio` are in its `zarr_vectors_level` attributes, and
`read_level_metadata(root, level)` reads them from there. For level 0 it
reports neither; level 0 uses the root's `base_bin_shape`.

Older stores wrote `scale = bin_ratio` and `translation = bin_shape / 2`
on coarser levels and kept no bin geometry in the level attributes. For
such a level, `read_level_metadata` takes `bin_ratio` from the scale and
`bin_shape` from twice the translation, and `write_multiscale_metadata`
copies them into the level attributes before rewriting the transforms.

### Writing and reading the block

`create_store` writes the block, and creating a level adds its
`datasets` entry. To rewrite it from the current levels:

```python
from zarr_vectors.core.multiscale import (
    get_level_scale,
    get_level_translation,
    write_multiscale_metadata,
)
from zarr_vectors.core.store import open_store

root = open_store("scan.zarrvectors", mode="r+")
write_multiscale_metadata(root)

print(get_level_scale(root, level=1))        # [1.0, 1.0, 1.0]
print(get_level_translation(root, level=1))  # [25.0, 25.0, 25.0]
```

`read_multiscale_metadata(root)` returns the block, or `None` when the
root has none.

### Validation

No validator checks the `datasets` or their transforms. The root
metadata, and so every validation level, needs `multiscales[0].axes`.
