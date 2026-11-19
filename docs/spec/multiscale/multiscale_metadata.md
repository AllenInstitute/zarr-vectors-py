# Multiscale metadata

## Terms

**`multiscales` block**
: An object in the root attributes that lists the resolution levels of a
  Zarr Vectors store, in the layout of the
  [multiscales convention](https://github.com/zarr-conventions/multiscales/blob/v0.1/README.md)
  v0.1. The root registers that convention in `zarr_conventions` (uuid
  `d35379db-88df-4056-af3a-620245f8e347`).

**Layout entry**
: One element of `multiscales.layout`, naming one level group.

---

## Introduction

The `multiscales` block lists every level, so tools that know the
multiscales convention can find the pyramid. It holds nothing else. The
axes are in the root `zarr_vectors.axes` (see
[Root metadata](../layout/root_metadata.md)), and each coarser level's bin
geometry (`bin_shape`, `bin_ratio`) is in its
[level attributes](../layout/level_groups.md).

---

## Technical reference

### Example

A streamline store with a three-level pyramid:

```json
"multiscales": {
  "layout": [
    {"asset": "0"},
    {"asset": "1"},
    {"asset": "2"}
  ]
}
```

### Keys

| Key | Value |
|-----|-------|
| `layout` | One entry per level, in ascending order |
| `layout[].asset` | The level group's name (`"0"`, `"1"`, …). Level `"0"` is always present |

There is no `transform` and no bin key. A writer keeps any other keys
already present in an entry.

### No transform

Vertex positions are in level-0 coordinates at every level (a
metavertex is the centroid of the vertices it replaces), so every level
is drawn in the same place and no coordinate transform relates them. The
multiscales convention's `transform` says how coordinates change between
levels, so a Zarr Vectors store writes none.

Bin geometry is stored once, outside this block: level 0's bin is the
root `base_bin_shape` (or `chunk_shape` if there is none), and a coarser
level's `bin_shape` and `bin_ratio` are in its `zarr_vectors` level
object. `read_level_metadata(root, level)` reads them from there. For
level 0 it reports neither.

Before 0.10.0 this block was an OME-NGFF 0.4 `multiscales` list that also
held the axes and gave every level level 0's `scale` and `translation`.
`migrate_metadata_0_10` replaces it; see
[Root metadata](../layout/root_metadata.md#migrating-from-09x).

### Writing and reading the block

`create_store` writes the block with level 0, and creating a level adds
its entry. To rewrite it from the current levels:

```python
from zarr_vectors.core.multiscale import (
    read_multiscale_metadata,
    write_multiscale_metadata,
)
from zarr_vectors.core.store import open_store

root = open_store("scan.zarrvectors", mode="r+")
write_multiscale_metadata(root)
read_multiscale_metadata(root)   # {'layout': [{'asset': '0'}, {'asset': '1'}]}
```

`read_multiscale_metadata(root)` returns the block, or `None` when the
root has none. `get_level_scale` and `get_level_translation` still
return level 0's transform for a listed level (`scale` all `1.0`,
`translation` half the root's effective bin shape), derived from the root
rather than stored, and `None` for a level that is not listed.

### Validation

No validator checks the `multiscales` block against the level groups.
