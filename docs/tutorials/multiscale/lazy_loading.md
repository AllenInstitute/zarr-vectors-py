# Deferred reads and out-of-core access

This tutorial reads a store without loading it whole: it picks a level from
metadata, reads one grid cell at a time, spreads cells across workers and
reads single objects.

Nothing is read until you ask:

- `zv.open()` reads metadata only.
- `ds.select(...)` returns a `Query`, a pending read. Building and inspecting
  it reads nothing.
- `read()`, `count()` and `object_ids()` read array data. `zv.aopen` and
  `Query.aread` are the asynchronous versions.

## Create the example stores

A point cloud with a three-level pyramid, and 500 streamlines:

```python
import numpy as np
import zarr_vectors as zv

rng = np.random.default_rng(0)

scan = zv.create("scan.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    kind="point_cloud",
    layout=zv.Layout(cells=5),
))
scan.add_points(rng.uniform(0, 1000, (100_000, 3)).astype(np.float32))
scan.build_pyramid(factors=[(2.0, 1.0), (2.0, 1.0)], chunk_scale_factors=[2, 2])

starts = rng.uniform(100.0, 900.0, size=(500, 3))
paths = [
    np.clip(start + rng.normal(0.0, 5.0, size=(40, 3)).cumsum(axis=0), 0, 1000)
    .astype(np.float32)
    for start in starts
]
tracts = zv.create("tracts.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    kind="polyline",
    layout=zv.Layout(cells=5),
))
tracts.add_polylines(paths, streamlines=True)
```

## Open a store and query it

```python
ds = zv.open("scan.zarrvectors")
for index in ds.levels:
    level = ds.level(index)
    print(index, level.vertex_count, level.scale, level.grid.shape)

q = ds.select(bbox=((100.0, 100.0, 100.0), (300.0, 300.0, 300.0)))
print(q.explain())
print(q.count())
```

```text
0 100000 (200.0, 200.0, 200.0) (6, 6, 6)
1 1000 (400.0, 400.0, 400.0) (3, 3, 3)
2 125 (800.0, 800.0, 800.0) (2, 2, 2)
point_cloud read: level 0, bbox [100.0, 100.0, 100.0]..[300.0, 300.0, 300.0]
  via read_points(bbox, level)
798
```

The level properties come from metadata. `explain()` names the reader that
will run, without reading. A remote store opens the same way; see
[Cloud stores](../../how_to/cloud_stores.md).

## Choose a level before reading

`Dataset.resolution(scale=...)` returns the level whose chunk shape is closest
to a physical size:

```python
print(ds.resolution(scale=800.0).index, ds.resolution(scale=50.0).index)
print(ds.level(2).read())
```

```text
2 0
ReadResult(kind='point_cloud', vertices=125)
```

Whole-level reads are exact at every level. A bounding-box query on level 1
or above can miss vertices inside the box unless the box covers whole chunks.
For a region on a coarse level, read the level whole and filter in memory.

## Read one cell at a time

`Level.cells()` lists the cells a level holds; `select(cells=[...])` reads
them. Only one cell is in memory at a time:

```python
level = ds.level(0)
total = 0
for ref in level.cells():
    total += level.select(cells=[ref]).read().vertex_count
print(len(level.cells()), total)
```

```text
125 100000
```

`Query.cells()` lists the cells a box query touches. A cell read returns
everything in the cell, and cuts polylines at cell boundaries.

## Split the work across workers

A cell reference is a plain value, so the list can feed a thread pool, a
process pool or a job array. Threads overlap request latency on a remote
store:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=8) as pool:
    counts = list(pool.map(
        lambda ref: level.select(cells=[ref]).read().vertex_count, level.cells()
    ))
print(len(counts), sum(counts))
```

```text
125 100000
```

## Read one object

`Level.objects` reads objects by id. A list of ids is gathered in one pass:

```python
objects = zv.open("tracts.zarrvectors").level(0).objects
print(objects[42])
print(objects[[42, 43, 44]])
```

```text
ReadResult(kind='polyline', vertices=40)
ReadResult(kind='polyline', vertices=120, parts=3)
```

`ids()` returns the ids that hold geometry. On a thinned pyramid level a
dropped id reads as `vertex_count == 0`.

## See also

- {doc}`../../api/api`: `Dataset`, `Query`, `ReadResult`, `Level`.
- [Building multi-resolution pyramids](building_pyramids.md): creating the
  levels.
