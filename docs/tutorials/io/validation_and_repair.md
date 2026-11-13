# Validation and repair

This tutorial validates a store, breaks copies of it on purpose, and repairs
each one with the functions in `zarr_vectors.building`. Those functions
reconcile bookkeeping (counts, lists and manifests) with the data on disk;
they do not reconstruct damaged data, which is rewritten from source. What
each validation level checks is listed in the
[Validation overview](../../spec/validation/overview.md).

## Create a store

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
```

## Check a store

`Dataset.validate(level=...)` runs the checks up to the given level, from 1
to 5 (default 3):

```python
result = ds.validate(level=5)
print(result.summary())
print(result.ok, result.level, len(result.passed), result.warnings, result.errors)
```

```text
Level 5 validation: PASS
  51 passed, 0 warnings, 0 errors
True 5 51 [] []
```

| Level | Checks |
|-------|--------|
| 1 | structure: root metadata, levels, required arrays |
| 2 | root and level metadata |
| 3 | the data against the metadata, the object index and the links |
| 4 | what each geometry type requires |
| 5 | the pyramid: level numbering and decreasing vertex counts |

Use level 1 for a quick check when opening a store, level 3 after writing
one, and level 5 for a store with a pyramid. A level-3 run decodes every
chunk, so its cost grows with the store.

The result's `passed`, `warnings` and `errors` are lists of strings, and `ok`
is true when there are no errors. Warnings do not fail validation. The levels
run in order and stop early: if level 1 fails nothing else runs, and if level
2 fails levels 3 to 5 are skipped. A failing result therefore shows the first
problem, not every problem.

`zarr_vectors.validate.validate(store, level=...)` is the same check as a
function and takes a path, a URL or an open store; `validate_structure`,
`validate_metadata`, `validate_consistency`, `validate_conformance` and
`validate_multiresolution` run one level each. From the shell, the companion
package `zarr-vectors-tools` provides `zvtools validate STORE --level N`.

The sections below each break a copy of the store:

```python
import shutil

from zarr_vectors.validate import validate


def broken_copy(name):
    shutil.rmtree(name, ignore_errors=True)
    shutil.copytree("tracts.zarrvectors", name)
    return name
```

## Missing arrays

```python
store = broken_copy("no_vertices.zarrvectors")
shutil.rmtree(f"{store}/0/vertices")
print(validate(store, level=3).errors)
```

```text
['0/vertices/ missing']
```

Level-1 messages name the level directory (`0/`); later levels prefix
`resolution_<N>:`. A store missing data was written that way or truncated in
transit. There is no repair: write the level again from source.

## Wrong root metadata

When the root metadata is wrong but the data is right, correct the field with
`update_root_metadata`. It rewrites only the fields you pass. Read the
metadata first, and change only a field you know disagrees with the data:

```python
from zarr_vectors.building import open_store, read_root_metadata, update_root_metadata

store = broken_copy("bad_chunk_shape.zarrvectors")
root = open_store(store, mode="r+")
update_root_metadata(root, chunk_shape=[200.0, 200.0])
print(validate(store, level=3).errors)

update_root_metadata(root, chunk_shape=[200.0, 200.0, 200.0])
print(read_root_metadata(root).chunk_shape, validate(store, level=3).ok)
```

```text
['chunk_shape has 2 dims, expected 3', 'resolution_1: Level chunk_shape rank 3 != root chunk_shape rank 2', 'resolution_2: Level chunk_shape rank 3 != root chunk_shape rank 2']
(200.0, 200.0, 200.0) True
```

`update_level_metadata` does the same for one level's metadata, and
`read_level_metadata` reads it. If the data was written under the wrong
geometry, the metadata is right and the store must be written again.

## A level's `arrays_present` list

Each level's metadata lists the arrays it holds in `arrays_present`. The list
can miss arrays that are on disk, and a reader that consults it will not see
them. `refresh_arrays_present` lists the level's arrays and writes the list
back:

```python
from zarr_vectors.building import (
    get_resolution_level,
    read_level_metadata,
    refresh_arrays_present,
)

root = open_store("tracts.zarrvectors", mode="r+")
print(read_level_metadata(root, 0).arrays_present)
print(refresh_arrays_present(get_resolution_level(root, 0)))
```

```text
['vertices', 'object_index']
['links', 'object_index', 'vertex_fragments', 'vertices']
```

Validation does not report this; run it after a parallel write, or when an
array you wrote does not show up.

## The presence manifest

Each per-chunk array records which cells hold data in its `nonempty_chunks`
manifest. Parallel writers skip it, because every worker would update the same
field. A manifest that misses cells hides their data from the object index
check:

```python
from zarr_vectors.building import rebuild_presence

store = broken_copy("stale_presence.zarrvectors")
level = get_resolution_level(open_store(store, mode="r+"), 0)
meta = level.read_array_meta("vertices")
meta["nonempty_chunks"] = meta["nonempty_chunks"][:-5]   # forget five cells
level.write_array_meta("vertices", meta)

result = validate(store, level=3)
print(len(result.errors), result.errors[0])

print(len(rebuild_presence(level, "vertices")))
print(rebuild_presence(level))
print(validate(store, level=3).ok)
```

```text
337 resolution_0: obj 30 refs non-existent chunk (4, 4, 0)
122
['links/+1/0.0.0', 'links/0/+1.-1.0', 'links/0/+1.0.+1', 'links/0/+1.0.-1', 'links/0/+1.0.0', 'links/0/0.+1.+1', 'links/0/0.+1.0', 'links/0/0.0.+1', 'vertex_fragments', 'vertices']
True
```

Given an array path, `rebuild_presence` returns that array's cell keys; given
none, it rebuilds every per-chunk array of the level and returns their paths.
It rebuilds sharded arrays too (`on_sharded="derive"`, the default); pass
`"skip"` to leave them alone or `"raise"` to fail on one. Breaking the store
here goes through `read_array_meta` and `write_array_meta`, methods of the
store's internal group class.

## Link counts

A links family records `num_links` and `num_physical_records`.
`write_link_cells`, the per-cell writer for parallel workers, leaves them
alone, so the counts are wrong until `finalize_links` recounts:

```python
from zarr_vectors.building import finalize_links, read_links, write_link_cells

store = broken_copy("stale_links.zarrvectors")
level = get_resolution_level(open_store(store, mode="r+"), 0)
(source, _), (target, _) = read_links(level, delta=0)[0]
write_link_cells(level, [[(source, 0), (target, 0)]], 3, delta=0)
print(validate(store, level=3).errors)

part = finalize_links(level, delta=0)
print(part.num_links, part.num_physical_records, validate(store, level=3).ok)
```

```text
['resolution_0: links[delta=0] num_physical_records=1102 != 1103 rows on disk']
1103 1103 True
```

Run `finalize_links` after every worker has finished and before sharding. The
whole parallel sequence is in
[Cloud stores](cloud_stores.md#write-links-from-parallel-workers).

## Pyramid levels

Removing a level from the middle of a pyramid breaks the level numbering,
which level 5 reports:

```python
from zarr_vectors.building import list_resolution_levels, remove_resolution_level

store = broken_copy("gap.zarrvectors")
root = open_store(store, mode="r+")
remove_resolution_level(root, 1)
print(validate(store, level=5).errors)
```

```text
['Levels [0, 2], expected [0, 1]']
```

To repair, remove the levels above level 0 and build the pyramid again:

```python
for index in reversed(list_resolution_levels(root)[1:]):
    remove_resolution_level(root, index)

gap = zv.open(store, mode="r+")
gap.build_pyramid(factors=[(2.0, 1.0), (2.0, 4.0)], chunk_scale_factors=[2, 2])
print(gap.levels, validate(store, level=5).ok)
```

```text
(0, 1, 2) True
```

`remove_resolution_level` leaves the removed level's entry in the root
`multiscales` list; `build_pyramid` rewrites it. When only that list is out of
date, `write_multiscale_metadata` rewrites it from the levels on disk, and
`read_multiscale_metadata` reads it:

```python
from zarr_vectors.building import read_multiscale_metadata, write_multiscale_metadata

store = broken_copy("stale_multiscales.zarrvectors")
root = open_store(store, mode="r+")
remove_resolution_level(root, 2)
print([d["path"] for d in read_multiscale_metadata(root)[0]["datasets"]])

multiscales = write_multiscale_metadata(root)
print([d["path"] for d in multiscales[0]["datasets"]])
print(multiscales[0]["datasets"][1]["coordinateTransformations"])
```

```text
['0', '1', '2']
['0', '1']
[{'type': 'scale', 'scale': [2.0, 2.0, 2.0]}, {'type': 'translation', 'translation': [50.0, 50.0, 50.0]}]
```

To recompute the levels above a level whose data changed, keeping each
level's recorded bin ratio, sparsity and chunk shape, the only function is
the internal `rebuild_pyramid_from_level`, which may change between releases.
It trusts the source level, so validate that at level 3 first:

```python
from zarr_vectors.ops.refresh import rebuild_pyramid_from_level  # internal

root = open_store("tracts.zarrvectors", mode="r+")
print([s["vertex_count"] for s in rebuild_pyramid_from_level(root, source_level=0)])
print(validate("tracts.zarrvectors", level=5).ok)
```

```text
[551, 25]
True
```

## The object index

An object index that points at missing chunks or fragments is reported at
level 3 (`resolution_0: obj <id> refs non-existent chunk ...`).
`write_object_index` rewrites the index from manifests you supply; nothing
in the package derives the manifests for you, so this works only when you can
reconstruct them, for example from the source data.

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
