# Multiscale metadata

## Terms

**`multiscales` block**
: An array of objects in the root attributes (`zarr.json`) that describes the resolution
  pyramid of a Zarr Vectors store. The structure follows the
  [OME-Zarr NGFF multiscales specification](https://ngff.openmicroscopy.org/).
  Zarr Vectors' own per-level parameters (`vertex_count`,
  `object_sparsity`, `coarsening_method`, `parent_level`, …) live in each
  level group's `zarr_vectors_level` attribute; `bin_ratio` and
  `bin_shape` are *derived* from the `multiscales` transforms (see below).

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
readable index of the resolution pyramid: it lists every level and its
path. (The parameters of the coarsening operation that produced a level
are in that level's `zarr_vectors_level` attribute.) Second,
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

As written by `zarr-vectors-py` for a two-level, 3-D store with
`base_bin_shape = [50, 50, 50]` and a level 1 built with
`coarsen_factor=2`:

```json
"multiscales": [
  {
    "version":  "0.4",
    "name":     "default",
    "axes": [
      {"name": "x", "type": "space", "unit": "micrometer"},
      {"name": "y", "type": "space", "unit": "micrometer"},
      {"name": "z", "type": "space", "unit": "micrometer"}
    ],
    "datasets": [
      {
        "path": "0",
        "coordinateTransformations": [
          {"type": "scale",       "scale":       [1.0, 1.0, 1.0]},
          {"type": "translation", "translation": [25.0, 25.0, 25.0]}
        ]
      },
      {
        "path": "1",
        "coordinateTransformations": [
          {"type": "scale",       "scale":       [1.0, 1.0, 1.0]},
          {"type": "translation", "translation": [25.0, 25.0, 25.0]}
        ]
      }
    ],
    "metadata": {"format": "zarr_vectors"}
  }
]
```

Per-level parameters that are *not* in this block live on the level
group, `<level>/zarr.json` attributes, under `zarr_vectors_level`:

```json
"zarr_vectors_level": {
  "level": 1, "vertex_count": 51200, "object_sparsity": 0.5,
  "coarsening_method": "per_object", "parent_level": 0,
  "arrays_present": ["vertices", "vertex_fragments", "object_index"]
}
```

`bin_shape` and `bin_ratio` are deliberately **not** stored there:
`create_resolution_level` strips them, and `read_level_metadata`
rebuilds them from the transform (`bin_ratio = scale`,
`bin_shape = 2 × translation`). Level 0 has no `bin_ratio` / `bin_shape`
of its own; it inherits `base_bin_shape` from the root `zarr_vectors`
block.

> **Note.** `multiscales` is one of three root attribute blocks. The
> others are `zarr_vectors` (format parameters) and, since format 0.9.2,
> an optional RFC 8 `ome` node that repeats the axes and level list for
> OME collections. `multiscales[0].axes` is the **only** place the
> store's axes are stored.

### Key descriptions

#### Top-level `multiscales` fields

| Key | Written | Description |
|-----|---------|-------------|
| `version` | always | `"0.4"` — the NGFF layout the block follows (bare-root `multiscales`, not nested under `ome`). |
| `name` | always | `"default"` unless a different name was already present. |
| `axes` | always | One descriptor per spatial dimension. The canonical axis store. |
| `datasets` | always | One entry per resolution level, in ascending level order. |
| `metadata.format` | always | `"zarr_vectors"`: the format discriminator. (NGFF reserves `type` for the downsampling method, so it is not used.) |

#### `axes` schema

```json
{"name": "z", "type": "space", "unit": "micrometer"}
```

`name` and `type` are always present (`type` defaults to `"space"`);
`unit` is written only when the axis carries one. Default axis names,
when none are given, are `x`, `y`, `z`, `w` (`DEFAULT_AXES_NAMES`);
stores with more than four dimensions must pass `axes` explicitly. See
[Dimensionality](../foundations/dimensionality.md).

#### `datasets` entry schema

| Key | Type | Description |
|-----|------|-------------|
| `path` | `string` | Level group name — a bare integer, `"0"`, `"1"`, …. |
| `coordinateTransformations` | `[transform, …]` | `scale` first, then `translation` when any translation component is non-zero. |

There are no `level`, `bin_ratio`, `bin_shape` or `object_sparsity`
keys in a dataset entry.

### Coordinate transforms

Zarr Vectors writes up to two transforms per level:

```
[scale, translation]
```

(`translation` is omitted when it is all zeros.)

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

The block follows the NGFF 0.4 `multiscales` layout, but note that in
NGFF `scale` means world-units-per-index, whereas here it is a
dimensionless bin ratio. A generic OME-Zarr reader will:

- Read `axes`, `datasets[].path`, and `coordinateTransformations`.
- Ignore `metadata.format`.
- Not be able to render the level arrays: each `vertices/` array is a
  vlen-bytes array whose shape is the level's *chunk grid*, not an
  image.

The `ome` root node (format 0.9.2) is the part aimed at OME collections:
it lists one `zv:level` leaf per level and declares a `world`
coordinate system carrying the axes and units. See
[OME-Zarr comparison](../comparisons/ome_zarr.md).

### Writing and updating multiscale metadata

`zarr-vectors-py` creates the `multiscales` block when a store is
created and adds a `datasets` entry for each level created by a write
or by `build_pyramid`. To regenerate it after manual modifications:

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

No validator checks the `multiscales` block's transforms directly. L1
requires the `zarr_vectors` root block; L2 reads `multiscales[0].axes`
(a store without it cannot be parsed) and checks per-level `chunk_shape`
nesting; L5 checks levels are contiguous from 0. Scale and translation
values are not cross-checked. See
[Validation overview](../validation/overview.md).
