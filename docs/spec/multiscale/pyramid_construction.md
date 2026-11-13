# Pyramid construction

## Terms

**Metavertex**
: A vertex of a coarser level that replaces every source vertex in one
  bin of that level. Its position is the centroid of those vertices.

**Coarsen factor**
: The `coarsen_factor` passed to `coarsen_level` (or the first value of
  each `build_pyramid` factor pair). The target level's bin shape is the
  source level's bin shape times this factor, so factors compound down a
  pyramid: `[(2, 1), (2, 1), (2, 1)]` gives bins 2, 4 and 8 times the
  root's.

**Sparsity factor**
: See [Object sparsity](sparsity.md).

**Coarsening method**
: The function that builds a coarser level, named by `method=`. Core
  provides `per_object`.

**Cross-level link**
: A link from a vertex at one level to its metavertex at a coarser
  level, stored in the level's link family under a non-zero level delta
  (see [Links](../object_model/links.md)).

**Chunk scale factor**
: The `chunk_scale_factor` passed to `coarsen_level` (or each entry of
  `build_pyramid`'s `chunk_scale_factors`): a per-axis positive integer
  that multiplies the source level's `chunk_shape` to give the target
  level's.

---

## Introduction

A pyramid holds coarser copies of level 0 in levels `1`, `2`, ….
`build_pyramid` builds them in sequence; `coarsen_level` builds one.
Each coarser level merges the vertices in each of its bins into one
metavertex, may drop objects (see [Object sparsity](sparsity.md)), may
have a coarser chunk grid, and may be connected to the levels around it
by cross-level links. This page specifies what the core `per_object`
method writes. Building a pyramid step by step is covered in the
[pyramid tutorial](../../tutorials/multiscale/building_pyramids.md).

---

## Technical reference

### Methods and strategies

Core provides the `per_object` coarsening method and the `random`
selection strategy. Any other `method=` or `sparsity_strategy=` name is
dispatched to an implementation registered with
`zarr_vectors.building.register_coarsen_strategy` or
`register_selection_strategy`. The `zarr-vectors-tools` package
registers the coarsening methods `polyline` and `skeleton` and the
selection strategies listed in [Object sparsity](sparsity.md) when it is
imported. A name that is not registered raises `ValueError` naming the
package to install. `options=` passes extra keywords to a registered
method; `per_object` takes none and raises `CoarseningError` when given
any.

### What `per_object` writes

For a target level built from a source level:

- **Bins.** The target bin shape is the source level's bin shape
  (`base_bin_shape` from the root at level 0) times `coarsen_factor`.
  The vertices of every kept object are binned together, and each
  non-empty bin becomes one metavertex at the centroid of its vertices.
  A metavertex can therefore belong to more than one object.
- **Vertices and fragments.** Metavertices are assigned to the target
  level's chunks. Each metavertex is one fragment of one vertex in its
  chunk.
- **Object index.** Each kept object's manifest names, in the order of
  its source vertices, the fragments of the metavertices they fall in,
  with consecutive repeats removed. Every source object id has a row; a
  dropped object's manifest is empty. When the source has no object
  index, the level is coarsened as one object and the target has no
  object index.
- **Object attributes.** Each object attribute of the source is copied
  for the kept objects; dropped objects are absent (see
  [Object attributes](../object_model/object_attributes.md)).
- **Not written.** The target level has no `links/0/` family, no vertex
  attributes and no fragment attributes. Per-vertex attribute
  aggregation is left to registered methods, which name their
  aggregation with the tokens in `zarr_vectors.constants.VALID_AGGREGATIONS`
  (`mean`, `sum`, `mode`, `count`, `min`, `max`).

The target level's attributes record:

| Key | Value |
|-----|-------|
| `coarsening_method` | `"per_object"` |
| `parent_level` | The source level |
| `object_sparsity` | `1 / sparsity_factor` |
| `vertex_count` | The number of metavertices |
| `arrays_present` | `["vertices", "object_index"]`, or `["vertices"]` without an object index |
| `chunk_shape` | Only when it differs from the root's (see below) |
| `preserves_object_ids` | `true` when the level has an object index |
| `inherited_num_objects` | The source level's object count when it has an object index, else `0` |
| `shared_fragments` | `true` |

The level's bin shape and its `bin_ratio` (bin shape over level 0's,
per axis, rounded to an integer of at least 1) are recorded in the
root's [`multiscales` block](multiscale_metadata.md). The root's
`format_capabilities` gains `shared_fragments`, and
`preserved_object_ids` when the level has an object index.

The target level must not exist: `coarsen_level` raises
`CoarseningError` rather than write into one. If a step fails after it
starts writing, it removes the target level, any `links/+1` family it
added to the source level, and the root attributes it set. A
`build_pyramid` run keeps the levels it finished before a failing step.

### `build_pyramid` and `coarsen_level`

```python
from zarr_vectors.multiresolution.coarsen import build_pyramid

build_pyramid(
    "scan.zarrvectors",
    factors=[(2.0, 1.0), (2.0, 2.0)],  # (coarsen_factor, sparsity_factor) per level
)
```

```python
def build_pyramid(
    store_path,
    *,
    factors: list[tuple[float, float]],
    chunk_scale_factors: list[int | tuple[int, ...]] | None = None,
    sparsity_strategy: str = "random",
    sparsity_seed: int | None = None,
    cross_level_depth: int = 1,
    cross_level_storage: str = "explicit",
    method: str = "per_object",
    options: dict | None = None,
) -> dict

def coarsen_level(
    store_path,
    source_level: int,
    target_level: int,
    *,
    coarsen_factor: float = 1.0,
    sparsity_factor: float = 1.0,
    chunk_scale_factor: int | tuple[int, ...] = 1,
    sparsity_strategy: str = "random",
    sparsity_seed: int | None = None,
    cross_level_storage: str = "none",
    method: str = "per_object",
    options: dict | None = None,
) -> dict
```

Both are in `zarr_vectors.multiresolution.coarsen`;
`zarr_vectors.api.Dataset.build_pyramid` calls `build_pyramid` on an
open dataset. `factors[i]` builds level `i + 1` from level `i`. A factor of `1.0`
leaves that axis unchanged. `factors` is required.

### Per-level chunk grids (`chunk_scale_factor`)

With the default chunk scale factor `1`, every level uses the root's
`chunk_shape`. Otherwise the target level's chunk shape is

```
target_chunk_shape = source_chunk_shape × chunk_scale_factor   # per axis
```

A scalar applies to every axis; a tuple gives one integer per axis and
must have `sid_ndim` entries. Every value must be a positive integer, so
chunk grids nest. The result is written as the level's `chunk_shape`
attribute only when it differs from the root's.

Because each factor multiplies the source level's chunk shape, a
level's multiple of the root's chunk shape is read from metadata, not
from the factor list: `get_level_chunk_shape(root_meta, level_meta)`
gives a level's chunk shape and
`chunk_scale_from_root(root_meta, level_meta)` its per-axis integer
multiple of the root's (both in `zarr_vectors.core.metadata`). Links
between levels with different chunk grids are placed through the
[anchor](../object_model/links.md#cross-level-placement-the-anchor).

### Cross-level links

Two arguments of `build_pyramid` set which cross-level links are
written:

| `cross_level_depth` | Meaning |
|---------------------|---------|
| `0` | None, whatever `cross_level_storage` says |
| `N` (default `1`) | Deltas up to `N` |
| `-1` | Deltas up to the number of levels minus one |

| `cross_level_storage` | Written |
|-----------------------|---------|
| `"none"` | Nothing |
| `"implicit"` | `+N` at the finer level |
| `"explicit"` (default) | `+N` at the finer level and `-N` at the coarser level |

Each record links one vertex to the metavertex it was merged into, or
for `N ≥ 2` to the metavertex that contains that metavertex `N - 1`
levels further up. Every vertex with a metavertex at the target level
has one record. The families are written with `link_width: 2`,
`directed: true`; endpoint 0 is the vertex at the owning level (see
[Links](../object_model/links.md)).

`coarsen_level` writes the `±1` families for its own step when
`cross_level_storage` is not `"none"` (its default). Deltas of 2 or more
are written only by `build_pyramid`, after every level exists.

`build_pyramid` records `cross_level_depth` and `cross_level_storage` in
the root attributes, and adds `multiscale_links` to
`format_capabilities` when cross-level links are requested and the
store has at least two levels.

The [multiscale links notebook](../../../examples/07_multiscale_links.ipynb)
builds a three-level pyramid and reads its cross-level links.

### Validation

Before writing, `build_pyramid` raises `ValueError` when:

- `cross_level_storage` is not `"none"`, `"implicit"` or `"explicit"`;
- `cross_level_depth` is less than `-1`;
- `chunk_scale_factors` is given and its length differs from `factors`;
- an entry of `factors` is not a pair.

`coarsen_level` raises `CoarseningError` when a `chunk_scale_factor`
tuple does not have `sid_ndim` entries or a value is less than 1, and
when the target level exists.

On a finished pyramid, L5 validation checks that levels are numbered
from 0 without gaps, that each level's `vertex_count` is no larger than
the level before, and that `object_sparsity` is in `(0, 1]`.
