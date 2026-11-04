# Level groups

## Terms

**Level group**
: The Zarr group at path `<N>/` representing one resolution level.
  Contains all arrays for that level; its `zarr.json` carries the level's
  metadata under the `zarr_vectors_level` attribute.

**Level index (N)**
: The integer name of a level group. Level 0 is always the full-resolution
  (finest) level. Higher indices correspond to progressively coarser
  representations. Root-level groups whose name is not an integer
  (`parametric/`, …) are not levels.

**`zarr_vectors_level`**
: The attribute holding a level's own metadata (schema below). It does **not**
  contain the level's `bin_shape` or `bin_ratio` — those are derived from the
  root `multiscales` block.

**`bin_ratio`**
: The level's bin shape divided by the root `base_bin_shape`, per axis, **always
  relative to level 0**. A `bin_ratio` of `[2, 2, 2]` means the bin shape is
  `2 × base_bin_shape`. It is a float (a fractional coarsen factor is legal) and
  is stored as the NGFF `scale` of the level's dataset entry.

**`object_sparsity`**
: A float in `(0, 1]` declaring what fraction of objects from the parent
  level are retained at this level. Only meaningful for geometry types with
  discrete objects. Point clouds written without objects keep `1.0`.

**Per-level `chunk_shape`**
: An optional override of the root `chunk_shape` for this level. Each axis must
  be a positive integer multiple of the root's (nested chunk grids), so coarser
  levels can carry larger chunks.

---

## Introduction

A level group holds one complete set of arrays (vertices, fragments, links,
attributes, object index) for one resolution. Each level's *arrays* are
self-contained, but its *geometry* is not: to interpret a level a reader also
needs the root attributes — `chunk_shape` and `bounds` (for the chunk grid),
`base_bin_shape` and the level's `multiscales` transform (for the bin shape),
and the axes.

This page documents the per-level metadata schema, the naming rules for level
groups, and the relationship between level index and coarsening factor.

---

## Technical reference

### Per-level `zarr_vectors_level` schema

The level's `zarr.json` carries, in its `"attributes"` block:

```json
{
  "zarr_vectors_level": {
    "level":             1,
    "vertex_count":      1280450,
    "arrays_present":    ["vertices", "vertex_attributes", "links", "object_index"],
    "coarsening_method": "per_object",
    "parent_level":      0,
    "object_sparsity":   0.5,
    "preserves_object_ids":  true,
    "inherited_num_objects": 50000,
    "shared_fragments":      true
  }
}
```

#### Required keys

| Key | Type | Description |
|-----|------|-------------|
| `level` | `integer` | Must equal `N` in `<N>`. `>= 0`. |
| `vertex_count` | `integer` | Total vertex count across all chunks at this level. `>= 0`. Stamped at level creation (usually `0`) and corrected once the level is written; the L3 validator compares it with the vertices actually stored. |
| `arrays_present` | `[string, …]` | **Family** names present at this level: `"vertices"`, `"vertex_attributes"`, `"fragment_attributes"`, `"links"`, `"object_index"`, … — never `"<family>/<name>"`. Readers gate on the family. Re-derive it from disk with `building.refresh_arrays_present` rather than maintaining it by hand. |

#### Optional keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `object_sparsity` | `float` | `1.0` | Fraction of the parent's objects retained; in `(0.0, 1.0]`. |
| `coarsening_method` | `string` | `"none"` | `"none"`, `"per_object"` or `"manual"` (open set: other methods may be registered by companion packages). |
| `parent_level` | `integer` or `null` | `null` | The level this one was derived from. **Required for level `> 0`; must be absent for level 0.** |
| `chunk_shape` | `[float, …]` | root's | Per-level override, omitted when equal to the root's. |
| `preserves_object_ids` | `boolean` | `false` | True for levels from the per-object pyramid: every parent id keeps a manifest row. Requires `parent_level` and `inherited_num_objects`; never valid at level 0. See [Object manifest](../object_model/object_manifest.md#object-ids-and-rows). |
| `inherited_num_objects` | `integer` | absent | Size of the parent's object id space. |
| `shared_fragments` | `boolean` | `false` | Fragments may be named by more than one object's manifest. |
| `fragments_tile` | `boolean` | `false` | Every chunk's fragment index tiles its vertex buffer, so bulk reads may skip `vertex_fragments`. A claim about the absence of later edits: cleared by any later write to `vertices` / `vertex_fragments`. |
| `chunk_dims`, `chunk_attribute_name`, `chunk_attribute_values` | | absent | Attribute chunking; see [Attribute-based chunking](../chunking/attribute_chunking.md). The last two must be both set or both absent. |

Keys marked boolean are written only when true.

#### Derived values: `bin_shape` and `bin_ratio`

These two are **not** stored in `zarr_vectors_level`. `create_resolution_level`
removes them from the payload and writes them as the level's NGFF transform in
the root `multiscales[0].datasets` (see [Root metadata](root_metadata.md#multiscales)):

```
bin_ratio = scale
bin_shape = 2 × translation
```

`read_level_metadata(root, level)` reassembles them. Level 0 reports neither
(`bin_shape = None`, `bin_ratio = None`: it inherits `base_bin_shape`). Raw
reads of the level's attributes will not find them, and
`update_level_metadata` refuses to write them (or `chunk_shape` / `bounds`)
because that would produce keys no reader consults.

### Level 0 special case

Level 0 is always the full-resolution level. Its `zarr_vectors_level` must satisfy:

- `level == 0`
- no `parent_level`
- `object_sparsity == 1.0`
- not `preserves_object_ids`

and its transform in `multiscales` is `scale = 1` with `translation =
base_bin_shape / 2`.

### Naming convention and ordering

Level directories are named `0`, `1`, …, `N`; the directory listing order is
not relied upon. `list_resolution_levels` returns the integer-named groups
sorted by index:

```python
from zarr_vectors.core.store import open_store, list_resolution_levels

root   = open_store("scan.zarrvectors", mode="r")
levels = list_resolution_levels(root)
# Returns: [0, 1, 2]
```

Levels need not be contiguous in principle, but contiguous numbering from 0 is
what every writer produces and what the tools assume.

### Relationship between level index and coarsening factor

Three different "factors" meet in a level and are easy to confuse:

| Quantity | Measured against | Where it appears |
|----------|------------------|------------------|
| `bin_ratio` | **level 0** (cumulative) | NGFF `scale`; `read_level_metadata` |
| `coarsen_factor` | the **parent** level | argument of `coarsen_level`; first element of each `build_pyramid(factors=…)` pair |
| `chunk_scale_factor` | the **parent** level's chunk grid | argument of `coarsen_level`; compounds like `coarsen_factor` |

So `build_pyramid(store, factors=[(2.0, 1.0), (2.0, 1.0), (2.0, 1.0)])` yields
bins at 2×, 4× and 8× the root bin: `bin_ratio` = `[2,2,2]`, `[4,4,4]`,
`[8,8,8]` for levels 1–3. `compute_bin_ratio` converts a bin shape to an
*integer* ratio and raises if it is fractional; `read_level_metadata` does not
round.

The coarsening factor of a level, as a volume ratio of bins, is
`product(bin_ratio)`:

| Level | `bin_ratio` | Bin volume vs level 0 (3-D) |
|-------|-------------|------------------------|
| 0 | [1, 1, 1] | 1× |
| 1 | [2, 2, 2] | 8× |
| 2 | [4, 4, 4] | 64× |
| 3 | [8, 8, 8] | 512× |

There is no requirement that bin ratios double at each level. Any positive
ratio `≥ 1` is valid, including anisotropic ones, subject to each level's bin
shape tiling its chunk shape:

```json
{"scale": [1.0, 2.0, 2.0], "translation": [25.0, 50.0, 50.0]}
```

### Level group `zarr.json`

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": { "zarr_vectors_level": { "level": 1, "...": "..." } }
}
```

All Zarr Vectors metadata of the level lives in that one attribute.

### Adding and removing levels

```python
from zarr_vectors.core.store import (
    open_store,
    add_resolution_level,
    remove_resolution_level,
    list_available_ratios,
)

root = open_store("scan.zarrvectors", mode="r+")

print(list_available_ratios(root))   # [(1, 1, 1), (2.0, 2.0, 2.0)]

# Create an empty level shell (arrays are not populated)
level_group = add_resolution_level(root, level_index=2, bin_ratio=(4, 4, 4))

# Remove a level (deletes everything under it); level 0 cannot be removed
remove_resolution_level(root, level_index=2)
```

`add_resolution_level` creates the level group, its `zarr_vectors_level` block
(`coarsening_method="manual"`, `parent_level = level_index - 1` unless given)
and the matching `multiscales` entry, but populates no arrays. It exists for
callers who write a custom level themselves. Both functions refresh the root
`ome` node's level list.

To build coarser levels from the data, use the coarseners, which *create* their
target level — the target must not already exist:

```python
from zarr_vectors.multiresolution.coarsen import build_pyramid, coarsen_level

build_pyramid("scan.zarrvectors", factors=[(2.0, 1.0), (2.0, 2.0)])
#   level 1: bins 2x, all objects;   level 2: bins 4x (2x of level 1), half the objects

coarsen_level("scan.zarrvectors", source_level=0, target_level=1,
              coarsen_factor=2.0, sparsity_factor=1.0)
```

`sparsity_factor` is `>= 1` and means "keep `1/sparsity_factor` of the objects",
so `object_sparsity = 1 / sparsity_factor`.

### Validation

L1:

- `0/` must exist and contain `vertices/` (error) and `vertex_fragments/`
  (warning if missing).
- Each level should have a non-empty attribute block (warning otherwise).
- For each level, which of `vertex_attributes`, `fragment_attributes`,
  `object_index`, `object_attributes`, `groups`, `links` exist is listed.

L2 (per level):

- `vertex_count`, if present, is a non-negative integer.
- `object_sparsity` is in `(0.0, 1.0]`.
- **Only for a level that overrides `chunk_shape`:** the override is a positive
  integer multiple of the root's (nested grids), and, on each axis where the
  level's bin is no larger than its chunk, the bin divides the chunk.
  A coarse level's bin may legitimately be *larger* than its chunk (bin 200 in a
  chunk of 100), so no tiling is required there — and levels that inherit the
  root `chunk_shape` are not checked at all.

At construction time, `LevelMetadata.validate()` additionally enforces:
`level >= 0`; level 0 has no `bin_shape` or `parent_level`; level `> 0` has both;
`bin_ratio >= 1`; `preserves_object_ids` requires `parent_level` and
`inherited_num_objects`; the attribute-chunking keys are coherent.

L2 also reads `bin_shape` / `bin_ratio` from the level's raw attributes, where
current writers do not store them, so those particular checks only apply to
stores that still carry the old keys. L2 does not check that the `level` key
equals the directory name or that bin ratios are monotone.
