# Deferred reads and out-of-core access

An eager read of a whole store is impractical the moment the store stops
fitting in memory, and expensive long before that on an object store where
every array fetch is a network request. The supported surface is built for
that case: **nothing is read until you ask for it.**

- `zv.open()` reads the store's metadata and no array data.
- `ds.select(...)` returns a `Query`, which is a value describing a pending
  read. Building one, narrowing it, and inspecting it perform no I/O.
- `Query.read()`, `.aread()`, `.count()` and `.object_ids()` are the
  terminals. They are the only calls on this page that touch array data.
  (`zv.aopen` and `Query.aread` are the non-blocking pair, for hosts such as
  Pyodide where zarr's synchronous bridge cannot work.)

This page covers choosing a resolution level before reading anything, reading
one grid cell at a time to stay under a memory budget, splitting that work
across workers, and reading a single object without decoding the whole store.

---

## `zarr_vectors.lazy` is not the way to do this

Earlier versions of this page called `zarr_vectors.lazy` "the lazy API" and
recommended `open_zv` for exactly the two cases above. The package now
disagrees, and says so the moment you call it:

```bash
python -W always::DeprecationWarning -c \
  "from zarr_vectors.lazy import open_zv; open_zv('scan.zarrvectors')"
```

```text
<string>:1: DeprecationWarning: open_zv() is superseded by zarr_vectors.open(),
which returns a Dataset. The lazy layer reads chunk-by-chunk in Python and opens
no batched-read block, so against an object store it is slower than the eager
path it was meant to improve on; Dataset drives the batching engine instead.
```

Read that carefully: the layer that was recommended *for* object stores is the
one that is slower *against* object stores. It issues a request per chunk from
Python and never opens a batched-read block, so the per-request latency the old
advice was trying to amortise is paid in full, once per chunk. `Dataset` drives
the batching engine instead.

`zarr_vectors.lazy.writer.ZVWriter` carries a matching warning:

```text
DeprecationWarning: ZVWriter is superseded by Dataset.add_* for creating
geometry and Dataset.editing() for per-element edits. Note that
add_node_attribute_sync / add_face_attribute_sync have NO replacement on
Dataset -- a bulk per-vertex attribute column is not an edit -- so use
zarr_vectors.building.write_vertex_attribute for those.
```

That warning names one genuine gap and one stale pointer. The gap is real: a
bulk per-vertex attribute column is not an edit, so `Dataset.editing()` does
not cover it. The pointer is stale — there is no `write_vertex_attribute` on
`zarr_vectors.building` today; the per-chunk writer
`building.write_chunk_attributes` is what exists. Treat the missing whole-array
spelling as a gap to report, not a reason to keep using `ZVWriter`.

The module tier is not a matter of opinion:

```python
import zarr_vectors as zv

print(zv.stability("zarr_vectors.lazy"))
```

```text
internal
```

Everything below uses `zarr_vectors.api` (re-exported from the top-level
package) and, for the one section that genuinely needs the physical layout,
`zarr_vectors.building`. Both carry a compatibility promise; see
{doc}`../../api/index`.

---

## Opening a store reads metadata, not data

```python
import zarr_vectors as zv

ds = zv.open("scan.zarrvectors")

print(ds)
print(ds.levels, ds.kinds, ds.ndim, ds.format_version)
print(ds.bounds)

for index in ds.levels:
    level = ds.level(index)
    print(index, level.vertex_count, level.scale, level.resolution, level.grid.shape)
```

```text
Dataset('file:///.../scan.zarrvectors', point_cloud, levels=[0, 1, 2])
(0, 1, 2) ('point_cloud',) 3 (0, 9, 0)
(array([0., 0., 0.]), array([1000., 1000., 1000.]))
0 100000 (200.0, 200.0, 200.0) (50.0, 50.0, 50.0) (5, 5, 5)
1 1000 (400.0, 400.0, 400.0) (100.0, 100.0, 100.0) (3, 3, 3)
2 125 (800.0, 800.0, 800.0) (200.0, 200.0, 200.0) (2, 2, 2)
```

Every number in that loop comes from metadata. `Level.vertex_count` is a
stamped field, `Level.scale` is the physical size of one grid cell,
`Level.resolution` is the physical size of one spatial-query cell (the finest
region a bounding-box read can isolate), and `Level.grid` is the cell grid as a
first-class object. No vertex data has been fetched.

`Level` is a handle, not a container. There is no `level.vertices` to hold and
no `.compute()` to call: you narrow with `select(...)` and finish with a
terminal.

Opening a remote store is the same call — a URL rather than a path, plus
credentials and backend choice in a `StorageOptions`:

```python
ds = zv.open(
    "s3://open-neuro/synchrotron.zarrvectors",
    storage=zv.StorageOptions(options={"anon": True}),
)
print(ds.levels)
```

`StorageOptions` is separate from `Schema` deliberately: where the bytes live
is not a property of the data, so the same dataset moves between a disk and a
bucket without its schema changing. The backend is resolved rather than asked
for — explicit `StorageOptions.backend` first, then `$ZARR_VECTORS_BACKEND`,
then the URL scheme.

---

## A query is the deferred handle

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

`explain()` names the reader that will run and the arguments it will get;
`plan()` returns the same answer as a comparable value listing the exact cells
and arrays the read will touch. Both are pure metadata arithmetic. This is the
part `open_zv` never offered: you can see the cost of a read before paying it.

Only now does anything get fetched:

```python
print(q.count())
result = q.read()
print(result, result.positions.shape)
```

```text
797
ReadResult(kind='point_cloud', vertices=797, attributes=['intensity']) (797, 3)
```

Narrowing returns a **new** query and still reads nothing, so filter before you
compute rather than after:

```python
lean = q.select(attributes=["intensity"])
print(lean.explain())
print(lean.read().attributes.names())

first = q.limit(5).read()
print(first.vertex_count, first.truncated, first.complete)
```

```text
point_cloud read: level 0, bbox [100.0, 100.0, 100.0]..[300.0, 300.0, 300.0]
  via read_points(attribute_names, bbox, level)
('intensity',)
5 True False
```

`limit()` sets `truncated`, and `complete` goes `False` to say the result is
not the whole answer. `select()` also accepts `objects=`, `groups=`, `near=`
and `where=`; see {doc}`../../api/api` for the full set and what each one
requires of the store.

---

## Pick a level before you read anything

A viewer wants a level by physical size, not by index — an answer that survives
someone adding a level to the pyramid. That is what `Dataset.resolution` is
for, and it replaces the hand-rolled bin-shape comparison this page used to
carry:

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

`resolution(scale=...)` returns the `Level` whose cell size is closest to the
one you asked for. Reading the coarsest level whole is 125 vertices against
level 0's 100 000 — which is the entire point of a pyramid for an overview.

Note that whole-level reads at coarse levels are exact, while bounding-box
queries against coarsened levels currently under-report. For a region at low
resolution, read the level whole and filter in memory.

---

## Reading one grid cell at a time

`Query.cells()` names the grid cells the query touches. It reads nothing — it
is the cell-shaped view of `plan()` — and it returns a `CellSet` of opaque
`CellRef`s that feed straight back into `select(cells=...)`:

```python
cells = sorted(q.cells(), key=lambda ref: ref.key)
print(len(cells), cells[0], cells[0].coords)

part = ds.select(cells=[cells[0]]).read()
print(part, part.vertex_count)
```

```text
8 CellRef(0.0.0) (0, 0, 0)
ReadResult(kind='point_cloud', vertices=815, attributes=['intensity']) 815
```

A `CellRef` is deliberately opaque: you get one from `cells()` or from
`Grid.cells_in(...)` and hand it back, but you never build one out of integers.
That is what keeps a change in how cells are addressed from becoming a
downstream break — and it is the replacement for passing raw `chunks=[(3, 1,
2), ...]` coordinates you had to derive yourself.

Two things to know before you build on this.

**A cell read is a cell read, not a filtered read.** The 815 vertices above are
everything in that cell, not the 797 inside the bounding box. Cells are the
unit of I/O; the box is applied afterwards. For a point cloud that is only a
question of over-reading, but for a geometry with objects it changes the shape
of the answer:

```python
tracts = zv.open("tracts.zarrvectors")

qp = tracts.select(bbox=((410.0, 410.0, 410.0), (590.0, 590.0, 590.0)))
print(qp.count(), qp.read().part_count)
print(sum(tracts.select(cells=[ref]).read().vertex_count for ref in qp.cells()))
```

```text
1800 45
827
```

A bounding-box read of a polyline store returns 45 **whole** streamlines,
1800 vertices, including the parts of them that lie outside the box. Reading
the same query cell by cell gives 827 vertices — the ones actually in those
cells, with each streamline cut at the cell boundary. Both are correct answers
to different questions.

**`cells()` only names cells for a spatially bounded selection.** An unbounded
query reads every cell of the array, which the plan expresses as an expansion
rather than a list, so `cells()` comes back empty. To walk a whole level, ask
the grid instead:

```python
print(ds.select().cells())

level = ds.level(0)
total = 0
for ref in sorted(level.grid.cells_in(ds.bounds), key=lambda ref: ref.key):
    if not level.grid.holds(ref):
        continue
    total += level.select(cells=[ref]).read().vertex_count
print(total, level.vertex_count)
```

```text
CellSet(0 cell(s))
100000 100000
```

`cells_in` is inclusive of the far edge, so it can name references just past
the allocation; `Grid.holds(ref)` is the guard. The loop is memory-bounded:
peak usage is one cell, not one level.

---

## Splitting the work across workers

`cells()` is what a caller shards on. Each worker takes a slice of the cell
references and reads only its own — no coordination, no shared cursor, and the
split itself costs no I/O:

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
0 2 1665
1 2 1582
2 2 1620
3 2 1598
```

The same references parallelise in-process, which is what to reach for on a
remote store where the cost is latency rather than CPU:

```python
from concurrent.futures import ThreadPoolExecutor

refs = [ref for ref in level.grid.cells_in(ds.bounds) if level.grid.holds(ref)]

with ThreadPoolExecutor(max_workers=8) as pool:
    counts = list(pool.map(
        lambda ref: level.select(cells=[ref]).read().vertex_count, refs
    ))
print(len(counts), sum(counts))
```

```text
125 100000
```

This is the supported replacement for the lazy layer's `to_delayed()`. A
`CellRef` is a plain value, so the same list feeds `dask.delayed`, a process
pool, or a job array just as well — the scheduler is your choice, not the
library's.

---

## Streaming: what is not available yet

`Query.iter_cells()` is the streaming terminal, and it raises today:

```python
try:
    for ref, part in q.iter_cells():
        ...
except NotImplementedError as exc:
    print(exc)
```

```text
Query.iter_cells() arrives with the resolver phase. The legacy readers
materialise the whole result, so a generator over one would use the same peak
memory while implying it does not. Until then: Query.cells() gives the cell
references, and select(cells=[ref]).read() reads one.
```

The docstring pins the contract it will honour, so that nobody codes to a
guess in the meantime: it will yield `(CellRef, ReadResult)` pairs, one per
cell that `cells()` names, in sorted key order; each result will cover exactly
that cell, cut into parts as `Selection.cells_split` asks for, so an object
spanning two cells appears as one part in each; per-cell failures will arrive
in that result's `errors` rather than as an exception, so one unreadable cell
does not end the stream; and
`sum(r.vertex_count for _, r in q.iter_cells()) == q.count()`.

Until it lands, the cell loop in the two sections above **is** the accurate
replacement — same references, same order, one cell in memory at a time. What
it does not yet give you is the per-cell error isolation, so a cell that fails
to read will raise out of your loop.

---

## Reading one object without decoding the whole store

For geometry with discrete objects, `Level.objects` addresses them by id and
reads only what was asked for:

```python
tracts = zv.open("tracts.zarrvectors")
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

`len()` and `slots` come from metadata. `count` reads the stamped
`num_present` where the store has it and decodes the manifests otherwise.
Passing a list gathers those objects in one pass rather than one read each,
which is the difference that matters on a bucket.

`ids(present=True)` — the default — returns only ids that actually hold
geometry. That distinction is load-bearing on a sparsified pyramid level, where
every dropped id is still an addressable slot: asking for one gives
`vertex_count == 0`, indistinguishable from an empty region. Pass
`present=False` for every slot.

If you are building your own gather plan and need the physical answer,
`manifests(ids)` gives it without reaching for a private name:

```python
for object_id, manifest in level.objects.manifests([42, 43]).items():
    print(object_id, len(manifest), manifest[:2])
```

```text
42 1 [((3, 1, 1), 2)]
43 1 [((1, 2, 2), 2)]
```

Each entry is that object's ordered `(chunk_coords, fragment_index)`
references — where its geometry actually sits — which is what a tool building
its own gather plan needs, and what it used to import three private names to
get.

Groups work the same way, addressed by name or id, and `by_id`/`__getitem__`
return a handle whose `.read()` is the terminal:

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

---

## Fragment indices

Inspecting a cell's fragment index is a physical question, so it belongs to the
builder surface rather than to `api`. Everything here is promised — do not
reach into `zarr_vectors.encoding` or `zarr_vectors.core` for it:

```python
from zarr_vectors.building import (
    open_store,
    get_resolution_level,
    list_chunk_keys,
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
True (0, 9)
[0 1 2 3 4]
```

`read_vertex_fragment_index` is the narrow helper for the `vertex_fragments/`
family; it replaces `encoding.fragments.read_fragment_index`, which took an
array name. `ChunkFragmentIndex` exposes `num_fragments`,
`num_range_fragments`, `is_range(f)`, `range(f)`, `indices(f)` and
`indices_view(f)`, and materialises no fragment payload until one of the last
three is called.

See [Fragment-index arrays](../../spec/layout/fragment_index_arrays.md) for the
byte layout.

---

## Performance on object stores

Remote stores have per-request latency (~50–200 ms for S3), so the goal is
fewer, larger, overlapping requests.

1. **Decide from metadata.** `ds.levels`, `Level.vertex_count` and
   `Level.grid` cost nothing. Choose the level before fetching anything.
2. **Prefer a coarse level for overviews.** `ds.resolution(scale=...)` picks
   one by physical size; drop to level 0 only for the region you are showing.
3. **Narrow before the terminal.** `select(bbox=...)`,
   `select(attributes=[...])` and `select(objects=[...])` all reduce what the
   read fetches. Check with `explain()`: it says whether a narrowing reaches
   the reader or is applied in memory afterwards.
4. **Read cells concurrently.** The cell loop plus a thread pool overlaps
   latency; a serial loop does not.
5. **Gather, don't iterate.** `level.objects[[7, 9, 11]]` is one pass;
   three separate `objects[i]` reads are three.

Caching and concurrency are properties of the backend and the executor you run
under, not options on `zv.open`.

---

## See also

- {doc}`../../api/api` — the full data surface: `Schema`, `Query`,
  `ReadResult`, `Level`, `ObjectCatalog`.
- {doc}`../../api/building` — the builder surface, for code that needs the
  physical layout.
- [Building pyramids](building_pyramids.md) — creating the coarse levels this
  page selects between.
