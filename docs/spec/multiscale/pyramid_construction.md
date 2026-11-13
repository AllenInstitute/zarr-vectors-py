# Pyramid construction

## Terms

**Metavertex**
: A vertex of a coarser level that replaces every source vertex in one
  of its bins, placed at their centroid.

**Coarsen factor**
: The `coarsen_factor` of `coarsen_level` (first value of each
  `build_pyramid` factor pair). The target bin shape is the source bin
  shape times this factor, so factors compound: `[(2, 1), (2, 1), (2, 1)]`
  gives bins 2, 4 and 8 times the root's.

**Coarsening method**
: The function that builds a coarser level, named by `method=`. Core
  provides `per_object`.

**Chunk scale factor**
: The `chunk_scale_factor` of `coarsen_level` (each entry of
  `build_pyramid`'s `chunk_scale_factors`): a positive integer per axis
  that multiplies the source level's `chunk_shape`.

The sparsity factor is defined in [Object sparsity](sparsity.md);
cross-level links in [Links](../object_model/links.md).

---

## Introduction

Levels `1`, `2`, … hold coarser copies of level 0. `build_pyramid`
builds them in sequence; `coarsen_level` builds one. This page specifies
what the core `per_object` method writes. For a walk-through, see the
[pyramid tutorial](../../tutorials/multiscale/building_pyramids.md).

---

## Technical reference

### Methods and strategies

Core provides the `per_object` method and the `random` selection
strategy. Other `method=` and `sparsity_strategy=` names are looked up
in implementations registered with
`zarr_vectors.building.register_coarsen_strategy` or
`register_selection_strategy`. Importing `zarr-vectors-tools` registers
the methods `polyline` and `skeleton` and the strategies in
[Object sparsity](sparsity.md). An unregistered name raises `ValueError`
naming the package to install. `options=` passes keywords to a
registered method; `per_object` raises `CoarseningError` if given any.

### What `per_object` writes

| Part | Target level |
|------|--------------|
| Bins | Source bin shape (root `base_bin_shape` at level 0) times `coarsen_factor`. The vertices of all kept objects are binned together; each non-empty bin becomes one metavertex, which may belong to several objects. |
| Vertices and fragments | Each metavertex is a one-vertex fragment in its target chunk. |
| Object index | A kept object's manifest names the fragments of the metavertices its source vertices fall in, in source order, with consecutive repeats removed. Every source id has a row; dropped objects are empty. A source with no object index is coarsened as one object and the target has no object index. |
| Object attributes | Copied for kept objects; absent for dropped ones. |
| Not written | `links/0/`, vertex attributes, fragment attributes. Registered methods that aggregate vertex attributes name the method with a token from `zarr_vectors.constants.VALID_AGGREGATIONS` (`mean`, `sum`, `mode`, `count`, `min`, `max`). |

Level attributes:

| Key | Value |
|-----|-------|
| `coarsening_method` | `"per_object"` |
| `parent_level` | The source level |
| `object_sparsity` | `1 / sparsity_factor` |
| `vertex_count` | Number of metavertices |
| `arrays_present` | `["vertices", "object_index"]`, or `["vertices"]` without an object index |
| `chunk_shape` | Only when it differs from the root's |
| `preserves_object_ids` | `true` when the level has an object index |
| `inherited_num_objects` | Source object count when it has an object index, else `0` |
| `shared_fragments` | `true` |

The bin shape and `bin_ratio` (bin shape over level 0's per axis,
rounded to an integer of at least 1) go in the root's
[`multiscales` block](multiscale_metadata.md). The root's
`format_capabilities` gains `shared_fragments`, and
`preserved_object_ids` when the level has an object index.

`coarsen_level` raises `CoarseningError` if the target level exists. If
a step fails part way, it removes the target level, any `links/+1`
family it added to the source level, and the root attributes it set.
`build_pyramid` keeps the levels finished before a failing step.

### `build_pyramid` and `coarsen_level`

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
open dataset. `factors[i]` is `(coarsen_factor, sparsity_factor)` for
level `i + 1`, built from level `i`; `1.0` leaves an axis unchanged.

### Per-level chunk grids (`chunk_scale_factor`)

```
target_chunk_shape = source_chunk_shape × chunk_scale_factor   # per axis
```

A scalar applies to every axis; a tuple has `sid_ndim` positive
integers, so chunk grids nest. The level's `chunk_shape` attribute is
written only when it differs from the root's. Read a level's chunk shape
with `get_level_chunk_shape(root_meta, level_meta)` and its integer
multiple of the root's with `chunk_scale_from_root(root_meta,
level_meta)` (both in `zarr_vectors.core.metadata`). Links between
levels with different grids use the
[anchor](../object_model/links.md#cross-level-placement-the-anchor).

### Cross-level links

| `cross_level_depth` | Deltas written |
|---------------------|----------------|
| `0` | none, whatever `cross_level_storage` says |
| `N` (default `1`) | up to `N` |
| `-1` | up to the number of levels minus one |

| `cross_level_storage` | Written |
|-----------------------|---------|
| `"none"` | nothing |
| `"implicit"` | `+N` at the finer level |
| `"explicit"` (default for `build_pyramid`) | `+N` at the finer level and `-N` at the coarser level |

Each record links a vertex to the metavertex it was merged into (for
`N ≥ 2`, the metavertex containing that one, `N - 1` levels up). Every
vertex with a metavertex at the target level has one record. Families
have `link_width: 2` and `directed: true`; endpoint 0 is the vertex at
the owning level.

`coarsen_level` writes only its own `±1` families, and only when
`cross_level_storage` is not `"none"` (its default). `build_pyramid`
writes deltas of 2 or more after every level exists. It records
`cross_level_depth` and `cross_level_storage` in the root attributes,
and adds `multiscale_links` to `format_capabilities` when links are
requested and the store has at least two levels. Example:
[`07_multiscale_links.ipynb`](../../../examples/07_multiscale_links.ipynb).

### Validation

| Raised by | Exception | When |
|-----------|-----------|------|
| `build_pyramid` | `ValueError` | `cross_level_storage` is not `"none"`, `"implicit"` or `"explicit"` |
| `build_pyramid` | `ValueError` | `cross_level_depth < -1` |
| `build_pyramid` | `ValueError` | `chunk_scale_factors` length differs from `factors` |
| `build_pyramid` | `ValueError` | an entry of `factors` is not a pair |
| `coarsen_level` | `CoarseningError` | a `chunk_scale_factor` tuple lacks `sid_ndim` entries, or a value is below 1 |
| `coarsen_level` | `CoarseningError` | the target level exists |

L5 checks that levels are numbered from 0 without gaps and that no
level has more vertices than the level before (see
[Validation overview](../validation/overview.md#level-5-pyramid)).
