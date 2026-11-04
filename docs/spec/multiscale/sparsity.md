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

**Sparsity factor**
: The value you pass to `build_pyramid` / `coarsen_level`: a number
  `>= 1` meaning "keep one object in N". `sparsity_factor=4.0` keeps a
  quarter of the objects. The stored `object_sparsity` is its reciprocal,
  `1 / sparsity_factor`.

**Sparsity strategy**
: The algorithm used to select which objects to retain. Core ships one,
  `random`. Others (for example spatial coverage, length-ranked,
  attribute-ranked and point-thinning) are registered by
  `zarr-vectors-tools` and are selected by name via
  `sparsity_strategy=`; requesting one without that package installed
  raises an error naming it.

**Balanced pyramid**
: A multi-resolution pyramid where the data volume at each level decreases
  by a roughly constant factor. Achieved by choosing `object_sparsity` and
  `bin_ratio` such that `total_reduction(level N) / total_reduction(level N-1)`
  is approximately constant across levels.

**Representative point**
: A single spatial position used to characterise the spatial location of an
  entire object, as used by spatial-coverage strategies registered by
  `zarr-vectors-tools`. Typically the midpoint
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
coarser levels.

Object ids are **preserved**: a dropped object keeps its id at the coarser
level as an empty manifest slot, so ids stay stable across levels
(format capability `preserved_object_ids`).

---

## Technical reference

### Configuration

Sparsity is the second element of each `factors` tuple passed to
`build_pyramid` (and `sparsity_factor=` on `coarsen_level`). It is a
*per-level ratio against the level below*, not a fraction of level 0:

```python
from zarr_vectors.multiresolution.coarsen import build_pyramid

build_pyramid(
    "tracts.zarrvectors",
    factors=[(2.0, 1.0), (4.0, 4.0)],   # (coarsen_factor, sparsity_factor)
    sparsity_strategy="random",         # the only core strategy
    sparsity_seed=42,                   # reproducible selection
)
```

Sparsity is only applied when `sparsity_factor > 1.0` **and** the source
level has more than one object. A store with no `object_index` (a plain
point cloud) is treated as a single implicit object, so sparsity has no
effect on it.

The selection step can also be called directly. Note that `apply_sparsity`
takes the *fraction to keep* (in `(0, 1]`), not a factor:

```python
from zarr_vectors.multiresolution.object_selection import apply_sparsity

kept = apply_sparsity(10000, 0.5, "random", seed=42)   # sorted int64 indices
```

### Sparsity strategies

#### `random` (core)

Samples `max(1, round(sparsity × n_objects))` object indices uniformly
without replacement using `numpy.random.default_rng(seed)`, and returns
them sorted. Reproducible for a fixed `seed`. This is the default and the
only strategy core implements.

#### Strategies from `zarr-vectors-tools`

Spatial-coverage, length-ranked, attribute-ranked and point-thinning
selection are not part of core. `apply_sparsity` accepts the optional
context arguments they need (`lengths`, `attribute_values`,
`attribute_mode`, `representative_points`, `bin_shape`) and forwards them
verbatim to a registered strategy; the built-in `random` ignores them.
A registered strategy has the signature
`fn(n_objects, target_count, **ctx) -> sorted int64 indices` (see
[`registry.py`](../../../zarr_vectors/multiresolution/registry.py)). Which
names are available depends on the installed version of that package; the
core coarsener itself passes `representative_points=None`, so a strategy
that needs object geometry must obtain it on its own.

### Total volume reduction

The total data volume reduction from level 0 to level `N` is:

```
total_reduction = vertex_reduction × object_reduction
```

where:

```
vertex_reduction   = product(bin_ratio[N][d] for d in range(D))   (theoretical max)
object_reduction   = 1.0 / object_sparsity[N]   (= sparsity_factor, compounded)
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
roughly `k×` smaller than the previous. Because both factors are
per-level ratios, they compound. For a target reduction of `8×` per level
and `D = 3`:

- If spatial coarsening alone provides `8×` (`coarsen_factor=2.0` in 3-D),
  use `sparsity_factor=1.0`.
- For `16×` per level: `coarsen_factor=2.0` (8× vertex) with
  `sparsity_factor=2.0` (2× object).

```python
build_pyramid(
    "tracts.zarrvectors",
    factors=[(2.0, 2.0), (2.0, 2.0), (2.0, 2.0)],   # 16x per level
)
```

### Object sparsity and point clouds

Object sparsity is only meaningful for geometry types with discrete objects.
A point cloud written without `object_ids` has no `object_index`, so the
core coarsener sees one implicit object and never thins it. Note however
that the `object_sparsity` written to the level metadata is simply
`1 / sparsity_factor`; the writer does not force it to `1.0` for point
clouds, so pass `sparsity_factor=1.0` for them.

### Validation

L2 checks `object_sparsity` is in `(0.0, 1.0]` at every level, and L5
repeats that check, but both read the value as a flat key on the level
group, which current writers do not produce (they nest it under
`zarr_vectors_level`). `LevelMetadata.validate()` enforces the same range
when a level is created. No validator checks that point clouds have
`object_sparsity == 1.0`, and none compares object counts across levels.
L5 checks that `vertex_count` does not increase from level to level. See
[Validation overview](../validation/overview.md).
