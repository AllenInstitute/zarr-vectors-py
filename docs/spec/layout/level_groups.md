# Level groups

## Terms

**Level group**
: The Zarr group at path `<N>/` representing one resolution level.
  Contains all array groups for that level plus a `zarr.json` and `.zattrs`.

**Level index (N)**
: The integer suffix of a level group name. Level 0 is
  always the full-resolution (finest) level. Higher indices correspond to
  progressively coarser representations.

**Per-level `.zattrs`**
: The `.zattrs` file inside `<N>/`. Stores the parameters of the
  coarsening operation applied to produce this level from level 0:
  `bin_ratio`, `object_sparsity`, `sparsity_strategy`, and the actual
  `bin_shape` in effect at this level.

**`bin_ratio`**
: A D-tuple of positive integers declaring how much larger the bin shape is
  at this level relative to the `base_bin_shape` at level 0. A `bin_ratio`
  of `[2, 2, 2]` means the bin shape is `2 × base_bin_shape` in every
  dimension.

**`object_sparsity`**
: A float in `(0, 1]` declaring what fraction of objects from level 0 are
  retained at this level. Only meaningful for geometry types with discrete
  objects (polyline, streamline, graph, skeleton, mesh). Point cloud stores
  always have `object_sparsity = 1.0`.

---

## Introduction

Each resolution level is self-contained: a reader can open and query any
single level without needing to open any other level. The per-level `.zattrs`
declares how the level was produced, enabling a reader to correctly interpret
coordinate transforms and to infer the effective bin shape without re-reading
the root metadata.

This page documents the per-level metadata schema, the naming rules for level
groups, and the relationship between level index and coarsening factor.

---

## Technical reference

### Per-level `.zattrs` schema

Full example for level 1, produced from a 3-D streamline store:

```json
{
  "level":              1,
  "bin_ratio":          [2, 2, 2],
  "bin_shape":          [100.0, 100.0, 100.0],
  "object_sparsity":    0.5,
  "sparsity_strategy":  "spatial_coverage",
  "vertex_count":       1280450,
  "object_count":       25000,
  "chunk_count":        3200
}
```

#### Required keys

| Key | Type | Description |
|-----|------|-------------|
| `level` | `integer` | Must equal `N` in `<N>`. |
| `bin_ratio` | `[int, …]` | D-tuple. Ratio of this level's bin shape to `base_bin_shape`. Level 0 must have `[1, 1, …, 1]`. |
| `bin_shape` | `[float, …]` | D-tuple. Effective bin shape: `base_bin_shape[i] × bin_ratio[i]`. Redundant given `base_bin_shape` and `bin_ratio` but stored explicitly for readers that do not access the root metadata. |

#### Recommended keys

| Key | Type | Description |
|-----|------|-------------|
| `object_sparsity` | `float` | Fraction of objects retained. Defaults to `1.0` if absent. Must be in `(0.0, 1.0]`. |
| `sparsity_strategy` | `string` | One of `"spatial_coverage"`, `"length"`, `"attribute"`, `"random"`. |
| `vertex_count` | `integer` | Total vertex count across all chunks at this level. |
| `object_count` | `integer` | Total object count at this level. |
| `chunk_count` | `integer` | Number of non-empty chunks at this level. |

### Level 0 special case

Level 0 is always the full-resolution level. Its per-level `.zattrs` must
satisfy:

- `"level": 0`
- `"bin_ratio": [1, 1, …, 1]` (D ones)
- `"bin_shape"` equals `base_bin_shape` from root `.zattrs`
- `"object_sparsity": 1.0` (all objects present)

### Naming convention and ordering

Level directories are named `0`, `1`, …,
`N`. The naming convention encodes the level index; the
directory listing order on the file system is not relied upon.

The `list_resolution_levels()` function returns levels sorted by index:

```python
from zarr_vectors.core.store import open_store, list_resolution_levels

root   = open_store("scan.zarrvectors", mode="r")
levels = list_resolution_levels(root)
# Returns: [0, 1, 2]
```

### Relationship between level index and coarsening factor

The coarsening factor at level `N` is the product of the `bin_ratio`
components: `product(bin_ratio[i] for i in range(D))`. For isotropic
ratios this equals `bin_ratio[0] ** D`.

| Level | `bin_ratio` | Coarsening factor (3-D) |
|-------|-------------|------------------------|
| 0 | [1, 1, 1] | 1× |
| 1 | [2, 2, 2] | 8× |
| 2 | [4, 4, 4] | 64× |
| 3 | [8, 8, 8] | 512× |

There is no requirement that bin ratios double at each level. Any positive-
integer ratio is valid, including anisotropic ratios:

```json
{"bin_ratio": [1, 2, 2], "bin_shape": [50.0, 100.0, 100.0]}
```

### Level group `zarr.json`

The `zarr.json` at a level group path is a plain Zarr v3 group node:

```json
{
  "zarr_format": 3,
  "node_type": "group"
}
```

No Zarr Vectors-specific keys are placed in the group's `zarr.json`; all Zarr Vectors
metadata lives in `.zattrs`.

### Adding and removing levels at runtime

```python
from zarr_vectors.core.store import (
    open_store,
    add_resolution_level,
    remove_resolution_level,
    list_available_ratios,
)

root = open_store("scan.zarrvectors", mode="r+")

# Inspect existing levels
print(list_available_ratios(root))   # [(1,1,1), (2,2,2)]

# Create an empty level shell (arrays are not populated)
level_group = add_resolution_level(root, level_index=2, bin_ratio=(4, 4, 4))

# Remove a level (deletes all arrays within it)
remove_resolution_level(root, level_index=2)
```

`add_resolution_level` creates the level group and its `.zattrs` with the
declared `bin_ratio` and `bin_shape`, but does not populate any arrays.
To populate, call `coarsen_level()` afterwards:

```python
from zarr_vectors.multiresolution.coarsen import coarsen_level

coarsen_level(
    "scan.zarrvectors",
    source_level=0,
    target_level=2,
    bin_ratio=(4, 4, 4),
    object_sparsity=0.25,
)
```

### Validation

Level group validation (L1):

- `0/` must exist.
- Each `<N>/` must contain a `zarr.json` identifying a group node.
- Each `<N>/` must contain a `.zattrs` with the required keys.

Level group validation (L2):

- `level` in `.zattrs` equals `N` in the directory name.
- `bin_shape[i] == base_bin_shape[i] × bin_ratio[i]` for all `i`
  (within floating-point tolerance).
- `bin_ratio` components are all positive integers.
- `object_sparsity` is in `(0.0, 1.0]`.
- Bin ratios are non-decreasing across levels (level 0 ≤ level 1 ≤ …
  component-wise).
