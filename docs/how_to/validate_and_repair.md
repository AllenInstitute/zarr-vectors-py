# Validation and repair

This page shows how to validate a store and repair its bookkeeping (counts,
lists and manifests) with `zarr_vectors.building`. Repairs do not rebuild
damaged data; rewrite that from source. The
[Validation overview](../spec/validation/overview.md) lists what each level
checks.

## Check a store

The example store is a set of polylines with a two-level pyramid:

```python
import numpy as np
import zarr_vectors as zv

rng = np.random.default_rng(0)
starts = rng.uniform(100.0, 900.0, size=(500, 3))
paths = [
    np.clip(start + rng.normal(0.0, 5.0, size=(40, 3)).cumsum(axis=0), 0, 1000)
    .astype(np.float32)
    for start in starts
]

ds = zv.create("tracts.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    kind="polyline",
    layout=zv.Layout(cells=5),
))
ds.add_polylines(paths, streamlines=True)
ds.build_pyramid(factors=[(2.0, 1.0), (2.0, 4.0)], chunk_scale_factors=[2, 2])

result = ds.validate(level=5)
print(result.summary())
```

```text
Level 5 validation: PASS
  58 passed, 0 warnings, 0 errors
```

`validate(level=...)` runs levels 1 to 5 (default 3). Use level 1 when opening
a store, level 3 after writing one, and level 5 for a store with a pyramid.
Level 3 decodes every chunk, so it is slow on a large store.

`result.ok` is true when `result.errors` is empty; warnings do not fail. If
level 1 or 2 fails, the levels above it do not run, so the result shows the
first problem, not every problem. `zarr_vectors.validate.validate(store,
level=...)` does the same for a path, a URL or an open store group, and `zvtools validate STORE --level N` (from
`zarr-vectors-tools`) does it from the shell.

## Repair a store

| Problem | How it shows | Fix |
|---|---|---|
| The store was written by 0.9.x | `UnsupportedFormatVersionError` on open, naming the migration | `zarr_vectors.building.migrate_metadata_0_10(path)` rewrites the metadata in place; safe to rerun. See [Migrating from 0.9.x](../spec/layout/root_metadata.md#migrating-from-09x) |
| A required array is missing | `0/vertices/ missing` | Write the level again from source |
| A root metadata field is wrong | errors naming the field, such as `chunk_shape has 2 dims, expected 3` | `update_root_metadata(root, field=...)` rewrites only the fields you pass; `update_level_metadata` for a level |
| `arrays_present` misses arrays on disk | readers do not see the array; validation is silent | `refresh_arrays_present(level)` |
| The presence manifest misses cells | `obj <id> refs non-existent chunk ...` | `rebuild_presence(level)` |
| Link counts are stale after parallel writes | `num_physical_records=... != ... rows on disk` | `finalize_links(level, delta=0)` |
| The root `multiscales` layout names levels that are gone | `read_multiscale_metadata(root)` | `write_multiscale_metadata(root)` |
| A pyramid level is missing | `Levels [0, 2], expected [0, 1]` | Remove the levels above 0, coarsest first, and build the pyramid again |
| The object index points at missing data | `obj <id> refs non-existent chunk ...` | `write_object_index` from manifests you rebuild from source |

Run the fixes for parallel writes after every worker has finished; see
[HPC pipelines](hpc_pipelines.md) and [Cloud stores](cloud_stores.md#write-links-from-parallel-workers).

For example, to repair a pyramid that has lost a level.
`remove_resolution_level` refuses to leave a gap, so remove the coarsest level
first:

```python
import shutil

from zarr_vectors.building import (
    list_resolution_levels, open_store, remove_resolution_level,
)
from zarr_vectors.validate import validate

shutil.rmtree("tracts.zarrvectors/1")                  # damage: level 1 is lost
print(validate("tracts.zarrvectors", level=5).errors)

root = open_store("tracts.zarrvectors", mode="r+")
for index in reversed(list_resolution_levels(root)[1:]):   # coarsest first
    remove_resolution_level(root, index)
ds = zv.open("tracts.zarrvectors", mode="r+")
ds.build_pyramid(factors=[(2.0, 1.0), (2.0, 4.0)], chunk_scale_factors=[2, 2])
print(ds.levels, validate("tracts.zarrvectors", level=5).ok)
```

```text
['Levels [0, 2], expected [0, 1]']
(0, 1, 2) True
```

## Validate in a test suite

```python
import pytest
from zarr_vectors.validate import validate

STORES = ["tracts.zarrvectors"]


@pytest.mark.parametrize("store", STORES)
def test_store_is_valid(store):
    result = validate(store, level=5)
    assert result.ok, result.summary()
```
