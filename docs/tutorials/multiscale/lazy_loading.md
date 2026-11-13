# Deferred reads and out-of-core access

This tutorial reads a store without loading it whole: choosing a resolution
level from metadata, reading one grid cell at a time, splitting cells across
workers, and reading single objects. Everything uses
{doc}`zarr_vectors.api <../../api/api>`, re-exported from the top-level
package, apart from one section on the builder surface.

Nothing is read until you ask for it:

- `zv.open()` reads metadata and no array data.
- `ds.select(...)` returns a `Query`, a description of a pending read. Building,
  narrowing and inspecting one perform no I/O.
- `Query.read()`, `.aread()`, `.count()` and `.object_ids()` are the terminals
  that read array data. `zv.aopen` and `Query.aread` are the asynchronous
  versions.

## Create the example stores

The examples use a point cloud with a three-level pyramid and a store of 500
streamlines in two groups:

```python
import numpy as np
import zarr_vectors as zv

rng = np.random.default_rng(0)

scan = zv.create("scan.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    kind="point_cloud",
    vertex_attributes={"intensity": zv.AttributeSpec(dtype="float32")},
    layout=zv.Layout(cells=5),
))
scan.add_points(
    rng.uniform(0, 1000, (100_000, 3)).astype(np.float32),
    attributes={"intensity": rng.uniform(0, 1, 100_000).astype(np.float32)},
)
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
tracts.add_polylines(
    paths, streamlines=True, groups={0: range(250), 1: range(250, 500)},
)
```

## Open a store

```python
import zarr_vectors as zv

ds = zv.open("scan.zarrvectors")

print(ds.levels, ds.kinds, ds.ndim, ds.format_version)
print(ds.bounds)

for index in ds.levels:
    level = ds.level(index)
    print(index, level.vertex_count, level.scale, level.resolution, level.grid.shape)
```

```text
(0, 1, 2) ('point_cloud',) 3 (0, 9, 4)
(array([0., 0., 0.]), array([1000., 1000., 1000.]))
0 100000 (200.0, 200.0, 200.0) (50.0, 50.0, 50.0) (6, 6, 6)
1 1000 (400.0, 400.0, 400.0) (100.0, 100.0, 100.0) (3, 3, 3)
2 125 (800.0, 800.0, 800.0) (200.0, 200.0, 200.0) (2, 2, 2)
```

All of these come from metadata. `Level.scale` is the physical size of one
grid cell (the chunk shape), `Level.resolution` the size of one bin, and
`Level.grid` the cell grid. A `Level` is a handle: you narrow it with
`select(...)` and finish with a terminal.

A remote store opens with the same call, given a URL and a
`zv.StorageOptions`; see [Cloud stores](../../how_to/cloud_stores.md).

## A query is the deferred read

```python
q = ds.select(bbox=((100.0, 100.0, 100.0), (300.0, 300.0, 300.0)))

print(q)
print(q.explain())
print(q.cells())
```

```text
Query(point_cloud read: level 0, bbox [100.0, 100.0, 100.0]..[300.0, 300.0, 300.0])
point_cloud read: level 0, bbox [100.0, 100.0, 100.0]..[300.0, 300.0, 300.0]
  via read_points(attribute_names, bbox, level)
CellSet(8 cell(s))
```

`explain()` names the reader that will run and its arguments; `plan()` returns
the cells and arrays the read will touch. Neither reads array data.

The terminals read:

```python
print(q.count())
result = q.read()
print(result, result.positions.shape)
```

```text
798
ReadResult(kind='point_cloud', vertices=798, attributes=['intensity']) (798, 3)
```

Narrowing returns a new query and reads nothing:

```python
lean = q.select(attributes=["intensity"])
print(lean.read().attributes.names())

first = q.limit(5).read()
print(first.vertex_count, first.truncated, first.complete)
```

```text
('intensity',)
5 True False
```

A limited result has `truncated` set and `complete` false. `select()` also
takes `objects=`, `groups=`, `near=` and `where=`; see {doc}`../../api/api`.

## Choose a level before reading

`Dataset.resolution(scale=...)` returns the level whose chunk shape is closest
to the size you ask for, so a viewer can pick a level by physical size:

```python
print(ds.resolution(scale=800.0).index)
print(ds.resolution(scale=400.0).index)
print(ds.resolution(scale=50.0).index)

print(ds.level(2).read())
```

```text
2
1
0
ReadResult(kind='point_cloud', vertices=125)
```

Whole-level reads are exact at every level. A bounding-box query on a
coarsened level (level 1 and above) can miss vertices that are inside the box,
unless the box covers whole chunks:

```python
box = ((400.0, 400.0, 400.0), (600.0, 600.0, 600.0))
for index in (1, 2):
    in_box = ds.select(level=index, bbox=box).read().vertex_count
    positions = ds.level(index).read().positions
    by_hand = int(((positions >= 400.0) & (positions <= 600.0)).all(axis=1).sum())
    print(index, in_box, by_hand)
```

```text
1 0 8
2 0 1
```

For a region on a coarse level, read the level whole, or a box aligned to its
chunks, and filter in memory as above.

## Read one grid cell at a time

`Query.cells()` names the cells a query touches, as `CellRef` values that
`select(cells=...)` accepts:

```python
cells = sorted(q.cells(), key=lambda ref: ref.key)
print(len(cells), cells[0], cells[0].coords)

part = ds.select(cells=[cells[0]]).read()
print(part, part.vertex_count)
```

```text
8 CellRef(0.0.0) (0, 0, 0)
ReadResult(kind='point_cloud', vertices=779, attributes=['intensity']) 779
```

A cell read returns everything in the cell, not only what is inside the box.
For geometry with objects the two answers differ in shape as well as size:

```python
tracts = zv.open("tracts.zarrvectors")

qp = tracts.select(bbox=((410.0, 410.0, 410.0), (590.0, 590.0, 590.0)))
print(qp.count(), qp.read().part_count)
print(sum(tracts.select(cells=[ref]).read().vertex_count for ref in qp.cells()))
```

```text
600 15
450
```

The bounding-box read returns 15 whole streamlines, including their parts
outside the box. The cell-by-cell reads return only the vertices in those
cells, with each streamline cut at cell boundaries.

`Query.cells()` is empty for a query without a box. To walk every cell of a
level, use `Level.cells()`, which lists the cells the level holds:

```python
print(ds.select().cells())

level = ds.level(0)
total = 0
for ref in level.cells():
    total += level.select(cells=[ref]).read().vertex_count
print(len(level.cells()), total, level.vertex_count)
```

```text
CellSet(0 cell(s))
125 100000 100000
```

Only one cell is in memory at a time. `Query.iter_cells()` raises
`NotImplementedError`; use this loop instead.

## Split the work across workers

Each worker takes its own slice of the sorted cell references. The split
reads nothing:

```python
def cells_for(query, *, worker, workers):
    ordered = sorted(query.cells(), key=lambda ref: ref.key)
    return ordered[worker::workers]


for worker in range(4):
    mine = cells_for(q, worker=worker, workers=4)
    total = sum(ds.select(cells=[ref]).read().vertex_count for ref in mine)
    print(worker, len(mine), total)
```

```text
0 2 1549
1 2 1630
2 2 1623
3 2 1618
```

The same references work with a thread pool, which overlaps request latency
on a remote store:

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

A `CellRef` is a plain value, so the list can feed a process pool, a job array
or `dask.delayed` in the same way.

## Read one object

For geometry with objects, `Level.objects` reads objects by ID:

```python
level = tracts.level(0)

print(level.objects, len(level.objects), level.objects.count, level.objects.slots)
print(level.objects.ids()[:5])

one = level.objects[42]
print(one, one.vertex_count, one.polylines[0].shape)

few = level.objects[[42, 43, 44]]
print(few, few.part_count)
```

```text
ObjectCatalog(level=0, count=500, slots=500) 500 500 500
[0 1 2 3 4]
ReadResult(kind='polyline', vertices=40) 40 (40, 3)
ReadResult(kind='polyline', vertices=120, parts=3) 3
```

A list of IDs is gathered in one pass rather than one read per object.
`ids()` returns only the IDs that hold geometry; `ids(present=False)` returns
every slot. On a thinned pyramid level a dropped ID is still a slot, and
reading it gives `vertex_count == 0`.

`manifests(ids)` gives where each object's geometry sits, as ordered
`(chunk_coords, fragment_index)` references:

```python
for object_id, manifest in level.objects.manifests([42, 43]).items():
    print(object_id, len(manifest), manifest[:2])
```

```text
42 1 [((2, 3, 2), 0)]
43 3 [((3, 2, 2), 3), ((3, 1, 2), 1)]
```

Groups are addressed by name or ID, and `.read()` is the terminal:

```python
print(level.groups, list(level.groups))
print(level.groups.by_id(0))
print(level.groups["group_0"].read())
```

```text
GroupCatalog(['group_0', 'group_1']) ['group_0', 'group_1']
ObjectGroup('group_0', id=0, members=250)
ReadResult(kind='polyline', vertices=10000, parts=250)
```

## Fragment indices

A cell's fragment index is part of the physical layout, so it is read through
the builder surface, {doc}`zarr_vectors.building <../../api/building>`:

```python
from zarr_vectors.building import (
    get_resolution_level,
    list_chunk_keys,
    open_store,
    read_vertex_fragment_index,
)

root = open_store("scan.zarrvectors", mode="r")
level_group = get_resolution_level(root, 0)

keys = list_chunk_keys(level_group)
print(len(keys), keys[:3])

fidx = read_vertex_fragment_index(level_group, keys[0])
print(fidx.num_fragments, fidx.num_range_fragments)
print(fidx.is_range(0), fidx.range(0))
print(fidx.indices(0)[:5])
```

```text
125 [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
64 64
True (0, 5)
[0 1 2 3 4]
```

`ChunkFragmentIndex` has `num_fragments`, `num_range_fragments`, `is_range(f)`,
`range(f)`, `indices(f)` and `indices_view(f)`; no fragment data is decoded
until one of the last three is called. The byte layout is in
[Fragment-index arrays](../../spec/layout/fragment_index_arrays.md).

## Reading from object stores

On a remote store every array fetch is a request, so aim for fewer, larger,
overlapping requests:

1. **Decide from metadata.** `ds.levels`, `Level.vertex_count` and
   `Level.grid` read no array data.
2. **Use a coarse level for overviews.** Read level 0 only for the region on
   show.
3. **Narrow before the terminal.** `select(bbox=...)`,
   `select(attributes=[...])` and `select(objects=[...])` reduce what is
   fetched; `explain()` shows whether a narrowing reaches the reader.
4. **Read cells concurrently** with a thread pool, as above.
5. **Gather objects.** `level.objects[[7, 9, 11]]` is one pass; three
   `objects[i]` reads are three.

## See also

- {doc}`../../api/api`: `Dataset`, `Query`, `ReadResult`, `Level`,
  `ObjectCatalog`.
- {doc}`../../api/building`: the builder surface.
- [Building multi-resolution pyramids](building_pyramids.md): creating the
  levels this page chooses between.
