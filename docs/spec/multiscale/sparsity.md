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

Object sparsity drops objects at a coarser level of a store with an
object index. Kept objects keep their ids; a dropped object keeps its
row with an empty manifest. The rest of coarsening is in
[Pyramid construction](pyramid_construction.md).

---

## Technical reference

### What a sparsified level stores

For sparsity factor `s > 1` and a source level with `n` objects:

| Item | Value |
|------|-------|
| Kept objects | `max(1, round(n / s))` |
| Level attributes | `object_sparsity: 1 / s` |
| [`object_index`](../object_model/object_manifest.md) | one row per source id; dropped rows are empty; `num_present` counts kept rows |
| [Object attributes](../object_model/object_attributes.md) | one row per source id; dropped rows are absent |
| Root `format_capabilities` | includes `preserved_object_ids` |

A source with no object index is coarsened as one object; nothing is
dropped.

### Selection strategies

Core provides `random`: a uniform sample without replacement, seeded by
`sparsity_seed`. Other strategies are added with
`zarr_vectors.building.register_selection_strategy`; `zarr-vectors-tools`
adds `spatial_coverage`, `length`, `attribute` and `point_thinning`. An
unregistered name raises `ValueError` naming the package to install.

`apply_sparsity` returns the sorted indices of the kept objects:

```python
from zarr_vectors.multiresolution.object_selection import apply_sparsity

kept = apply_sparsity(10, 0.5, "random", seed=42)
print(kept)  # [0 3 4 5 7]
```

Signature: `apply_sparsity(n_objects, sparsity, strategy="random", *,
seed=None, lengths=None, attribute_values=None, attribute_mode="max",
representative_points=None, bin_shape=None)`. `sparsity` is the fraction
to keep; `1.0` or more keeps all. Keywords after `seed` go to registered
strategies; `random` ignores them.

### Validation

| Level | Check |
|-------|-------|
| L2, L5 | `object_sparsity` is in `(0.0, 1.0]` (absent reads as `1.0`). Does not run on stores this package writes ([L2](../validation/l2_metadata.md)) |
| L5 | Each level has no more vertices than the level before |
