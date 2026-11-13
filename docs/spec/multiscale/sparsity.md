# Object sparsity

## Terms

**Object sparsity**
: The `object_sparsity` value in a level's attributes: the fraction of
  the source level's objects a coarser level keeps, in `(0.0, 1.0]`.
  `1.0` keeps every object.

**Sparsity factor**
: The `sparsity_factor` passed to `coarsen_level` (or the second value
  of each `build_pyramid` factor pair), at least `1.0`. A level built
  with sparsity factor `s` has `object_sparsity = 1 / s`.

**Selection strategy**
: The function that chooses which objects a level keeps, named by
  `sparsity_strategy`.

---

## Introduction

Spatial coarsening merges vertices into metavertices but keeps every
object. Object sparsity also drops objects at a coarser level, for
stores whose geometry has discrete objects (an object index). The kept
objects keep their object ids; a dropped object stays addressable at
that level with an empty manifest. How the coarsened level is built is
in [Pyramid construction](pyramid_construction.md).

---

## Technical reference

### What a sparsified level stores

For a level built with sparsity factor `s > 1` from a source level with
`n` objects:

- The number of kept objects is `max(1, round(n / s))`.
- The level's attributes record `object_sparsity: 1 / s`.
- `object_index` has one row per source object id. A dropped object's
  manifest is empty; `num_present` counts the kept ones (see
  [Object manifest](../object_model/object_manifest.md)).
- Each object attribute has a row per source object id. A dropped
  object's row holds the column's fill value and is absent in the
  attribute's presence mask (see
  [Object attributes](../object_model/object_attributes.md)).
- The root's `format_capabilities` lists `preserved_object_ids`.

A level read from a source with no object index is coarsened as one
object, and nothing is dropped.

### Selection strategies

Core provides one strategy, `random`: a uniform sample without
replacement, reproducible with `sparsity_seed`. Other names (the
`zarr-vectors-tools` package registers `spatial_coverage`, `length`,
`attribute` and `point_thinning`) are dispatched through
`zarr_vectors.building.register_selection_strategy`; asking for a name
that is not registered raises `ValueError` naming the package to
install.

`apply_sparsity` runs a strategy on its own and returns the sorted
indices of the kept objects:

```python
from zarr_vectors.multiresolution.object_selection import apply_sparsity

kept = apply_sparsity(10, 0.5, "random", seed=42)
print(kept)  # [0 3 4 5 7]
```

Its signature is
`apply_sparsity(n_objects, sparsity, strategy="random", *, seed=None,
lengths=None, attribute_values=None, attribute_mode="max",
representative_points=None, bin_shape=None)`. `sparsity` is the fraction
to keep; at `1.0` or more every index is returned. The keyword arguments
after `seed` are passed unchanged to a registered strategy; `random`
ignores them.

### Validation

- L2 and L5: `object_sparsity` is in `(0.0, 1.0]` at every level (absent
  reads as `1.0`).
- L5: each level's `vertex_count` is no larger than the level before.
