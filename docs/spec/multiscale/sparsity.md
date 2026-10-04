# Object sparsity

## Terms

**Object sparsity**
: A float value in `(0.0, 1.0]` declaring the fraction of objects from the
  full-resolution level that are retained at a given coarser level. A value
  of `1.0` means all objects are retained (no thinning). A value of `0.5`
  means 50 % of objects are retained.

**Object thinning**
: The process of selecting a subset of discrete objects (streamlines,
  skeletons, meshes) for a coarser resolution level. Object thinning is
  independent of spatial coarsening (vertex binning).

**Sparsity strategy**
: The algorithm used to select which objects to retain. Four strategies are
  supported: `spatial_coverage`, `length`, `attribute`, and `random`.

**Balanced pyramid**
: A multi-resolution pyramid where the data volume at each level decreases
  by a roughly constant factor. Achieved by choosing `object_sparsity` and
  `bin_ratio` such that `total_reduction(level N) / total_reduction(level N-1)`
  is approximately constant across levels.

**Representative point**
: A single spatial position used to characterise the spatial location of an
  entire object for the `spatial_coverage` strategy. Typically the midpoint
  or centroid of the object's vertex sequence.

---

## Introduction

For geometry types with discrete objects — streamlines, graphs, skeletons,
and meshes — multi-resolution involves not just spatial coarsening of vertex
positions, but also the option to thin out objects at coarser levels.
This is the `object_sparsity` mechanism.

Spatial coarsening alone reduces data volume by merging nearby vertices into
metanodes, but does not reduce the number of objects. If a tractography
dataset contains one million streamlines, a pyramid level with
`bin_ratio = (2, 2, 2)` will have ~8× fewer vertices per streamline but
still one million streamlines. For overview rendering, one million even-
coarser streamlines may be far more than necessary.

Object sparsity solves this by explicitly limiting the number of objects at
coarser levels, using one of four selection strategies designed to preserve
the perceptual quality of the spatial representation.

---

## Technical reference

### Configuration

Object sparsity is specified in `level_configs` during pyramid construction:

```python
from zarr_vectors.multiresolution.coarsen import build_pyramid

build_pyramid(
    "tracts.zarrvectors",
    factors=[(2.0, 1.00), (4.0, 4.00)],
)
```

Or via `apply_sparsity` directly:

```python
from zarr_vectors.multiresolution.object_selection import apply_sparsity

kept_ids = apply_sparsity(
    n_objects=10000,
    sparsity=0.5,
    strategy="spatial_coverage",
    representative_points=midpoints,    # shape (n_objects, D)
    bin_shape=(100.0, 100.0, 100.0),
)
```

### Sparsity strategies

#### `spatial_coverage`

**Goal:** retain the objects that provide the best spatial coverage of the
dataset, so the thinned set looks spatially similar to the full set.

**Algorithm:**

1. Compute a representative point for each object (default: midpoint of
   the vertex sequence).
2. Assign representative points to spatial bins of size `bin_shape`.
3. For each bin, retain at most `ceil(sparsity × objects_in_bin)` objects.
4. Within each bin, select retained objects uniformly at random.

This strategy ensures that all spatial regions are represented at the
coarser level, proportionally to their object density at the base level.
It prevents the thinned set from clustering in dense regions while leaving
sparse regions empty.

```python
kept_ids = apply_sparsity(
    n_objects, sparsity=0.5, strategy="spatial_coverage",
    representative_points=midpoints,
    bin_shape=(100.0, 100.0, 100.0),
)
```

#### `length`

**Goal:** retain the longest objects (streamlines, skeletons). Useful when
long objects are more anatomically significant.

**Algorithm:** sort objects by total arc length (sum of inter-vertex
distances), descending. Retain the top `ceil(sparsity × n_objects)`.

```python
kept_ids = apply_sparsity(
    n_objects, sparsity=0.5, strategy="length",
    lengths=streamline_lengths,   # shape (n_objects,), precomputed arc lengths
)
```

When `lengths` is not provided, `zarr-vectors-py` computes arc lengths from
the `vertices/` array automatically (at the cost of reading all vertex data).

#### `attribute`

**Goal:** retain objects with the highest (or lowest) value of a named
per-object attribute.

**Algorithm:** sort objects by `attribute_values`, descending (or ascending
if `ascending=True`). Retain the top `ceil(sparsity × n_objects)`.

```python
kept_ids = apply_sparsity(
    n_objects, sparsity=0.5, strategy="attribute",
    attribute_values=fa_values,   # shape (n_objects,), e.g. mean FA
)
```

This strategy is useful for tractography datasets where a per-streamline
metric (mean FA, tract coherence, connectivity weight) indicates scientific
importance.

#### `random`

**Goal:** reproducible random selection. Simple baseline strategy.

**Algorithm:** sample `ceil(sparsity × n_objects)` object IDs uniformly
at random without replacement, using the declared `seed` for reproducibility.

```python
kept_ids = apply_sparsity(
    n_objects, sparsity=0.5, strategy="random",
    seed=42,
)
```

### Total volume reduction

The total data volume reduction from level 0 to level `N` is:

```
total_reduction = vertex_reduction × object_reduction
```

where:

```
vertex_reduction   = product(bin_ratio[N][d] for d in range(D))   (theoretical max)
object_reduction   = 1.0 / object_sparsity[N]
```

For a 3-D streamline store with `bin_ratio = (4, 4, 4)` and
`object_sparsity = 0.25` at level 2:

```
vertex_reduction  = 4 × 4 × 4 = 64×
object_reduction  = 1.0 / 0.25 = 4×
total_reduction   = 64 × 4 = 256×
```

In practice, vertex reduction is an upper bound: the actual reduction
depends on whether bins at the coarser level contain enough vertices to
form non-trivial metanodes. In sparse regions, bins may contain only one
vertex, giving no reduction from spatial coarsening.

### Designing a balanced pyramid

A common design goal is a balanced pyramid where each level is
roughly `k×` smaller than the previous. For a target reduction of `8×`
per level and `D = 3`:

- If spatial coarsening alone provides `8×` (`bin_ratio = (2,2,2)`),
  set `object_sparsity = 1.0`.
- If you want `16×` per level (aggressive thinning):
  `bin_ratio = (2,2,2)` (8× vertex) × `object_sparsity = 0.5` (2× object)
  = 16×.
- If spatial density is very uneven, `spatial_coverage` strategy maintains
  perceptual quality better than `random` for the same `object_sparsity`.

```python
build_pyramid(
    "tracts.zarrvectors",
    factors=[(2.0, 2.00), (4.0, 4.00), (8.0, 8.00)],
)
```

### Object sparsity and point clouds

Object sparsity is only defined for geometry types with discrete objects.
For point clouds (`GEOM_POINT_CLOUD`), there is no concept of a discrete
object; every vertex is independent. The per-level `.zattrs` for a point
cloud store always has `"object_sparsity": 1.0`, and the `sparsity_strategy`
key is absent.

Spatial thinning of point clouds is exposed via the
[`point_thinning` strategy](../../../zarr_vectors/multiresolution/object_selection.py):
`select_point_thinning(positions, bin_shape, seed=...)` keeps at most
one survivor per spatial bin, yielding a uniform-density thinned
cloud.  The same strategy is selectable through
`apply_sparsity(..., strategy="point_thinning",
representative_points=, bin_shape=)`.

### Storing each object once

A sparsity pyramid stores the objects kept at level `L + 1` again at every
finer level.  With [additive refinement](additive_refinement.md) each is
stored only at the coarsest level that keeps it, and a level's complete
content is read as the union over its chain -- about 1x the storage of
level 0 instead of 1.14-1.33x, and nothing already drawn is fetched again
when a viewer zooms in.

### Validation

L2 checks:

- `object_sparsity` is in `(0.0, 1.0]` at every level.
- Point cloud stores have `object_sparsity == 1.0` at every level.

L5 checks (pyramid consistency):

- `object_count` at level `N` ≤ `object_count` at level `N-1`.
- `object_count` at level `N` ≈ `object_count` at level 0 × `object_sparsity`
  (within a tolerance of 1 object, due to ceiling arithmetic).
- `vertex_count` at level `N` ≤ `vertex_count` at level `N-1`.
