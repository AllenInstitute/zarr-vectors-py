# Polylines and streamlines

Polylines and streamlines store ordered vertex sequences — paths through
3-D space. Use `polyline` for general ordered paths (vascular centrelines,
cell migration tracks, traced axons). Use `streamline` when the paths come
from MRI or synchrotron tractography and you need to record that the store
holds tractography.

The two types are technically identical in their on-disk arrays; the
distinction is in the declared geometry kind. All write/read calls in this
tutorial apply equally to both.

Everything here is on the two supported surfaces: `zarr_vectors.api` —
re-exported from the top-level package, so `zv.create` and `zv.open` below are
`api` objects — and `zarr_vectors.building` for the two places the api has no
equivalent yet (object-attribute values and per-group attribute values). Both
are covered by the compatibility promise; `zv.stability("zarr_vectors.api")`
says so at runtime. TRK/TCK/TRX converters live in the companion package
**`zarr-vectors-tools`**.

The page is one continuous session — later blocks reuse the stores earlier
blocks create.

---

## Writing streamlines

### Minimal write

A store starts as a `Schema`: where the paths live in space, what the axes
mean, and roughly how much data is coming. `Layout(cells=...)` says how finely
to cut the volume up; chunk and bin shapes are derived from it rather than
passed by hand.

```python
import numpy as np
import zarr_vectors as zv

rng = np.random.default_rng(0)

# 500 streamlines, each a random walk of 40 steps through a 1 000³ µm volume
starts = rng.uniform(200.0, 800.0, size=(500, 3))
streamlines = [
    (s + rng.normal(0.0, 8.0, size=(40, 3)).cumsum(axis=0)).astype(np.float32)
    for s in starts
]

tracts_min = zv.create("tracts_min.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    axes=(zv.Axis("x", unit="micrometer"),
          zv.Axis("y", unit="micrometer"),
          zv.Axis("z", unit="micrometer")),
    kind="polyline",
    expected=zv.SizeHints(n_vertices=20_000, n_objects=500),
    layout=zv.Layout(cells=5),
))
print(tracts_min.add_polylines(streamlines, streamlines=True))
print(tracts_min.kinds, tracts_min.level(0).kind)
print(tracts_min.level(0).scale, tracts_min.level(0).resolution)
```

```text
{'polyline_count': 500, 'vertex_count': 20000, 'chunk_count': 92, 'cross_chunk_link_count': 1878, 'group_count': 0}
('polyline', 'streamline') polyline
(200.0, 200.0, 200.0) (50.0, 50.0, 50.0)
```

`streamlines=True` *adds* `"streamline"` to the store's declared kinds; it does
not replace `"polyline"`, and both `Level.kind` and `ReadResult.kind` stay
`"polyline"`, which is what the readers dispatch on. `Level.scale` is the chunk
shape and `Level.resolution` is the bin shape — five cells per axis over 1 000 µm
gives 200 µm chunks, and the default four bins per chunk axis gives 50 µm bins.
See [Choosing a layout](../../how_to/choose_chunk_and_bin.md) for how to pick
them.

### Write with groups and attributes

Groups tag streamlines as belonging to named bundles or experimental
conditions. Per-streamline values (mean FA, arc length) go in
`object_attributes=`; per-vertex values go in `attributes=`.

```python
rng = np.random.default_rng(0)
n = 1000

starts = rng.uniform(300.0, 700.0, size=(n, 3))
streamlines = [
    (s + rng.normal(0.0, 6.0, size=(int(k), 3)).cumsum(axis=0)).astype(np.float32)
    for s, k in zip(starts, rng.integers(20, 60, size=n))
]

# Per-streamline arc length, for use as an object attribute
lengths = np.array([np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1))
                    for s in streamlines], dtype=np.float32)

# Per-streamline FA (simulated)
mean_fa = rng.uniform(0.2, 0.8, n).astype(np.float32)

# Per-vertex FA (simulated) — ONE ARRAY PER STREAMLINE, not one flat array
per_vertex_fa = [rng.uniform(0.1, 0.9, len(s)).astype(np.float32)
                 for s in streamlines]

tracts = zv.create("tracts.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    axes=(zv.Axis("x", unit="micrometer"),
          zv.Axis("y", unit="micrometer"),
          zv.Axis("z", unit="micrometer")),
    kind="polyline",
    vertex_attributes={"fa": zv.AttributeSpec(dtype="float32")},
    object_attributes={"length": zv.AttributeSpec(dtype="float32"),
                       "mean_fa": zv.AttributeSpec(dtype="float32")},
    expected=zv.SizeHints(n_vertices=50_000, n_objects=n),
    layout=zv.Layout(cells=5),
))
report = tracts.add_polylines(
    streamlines,
    attributes={"fa": per_vertex_fa},
    object_attributes={"length": lengths, "mean_fa": mean_fa},
    groups={
        0: list(range(300)),            # corticospinal tract
        1: list(range(300, 600)),       # arcuate fasciculus
        2: list(range(600, 1000)),      # uncinate fasciculus
    },
    streamlines=True,
)
print(report)
print(type(per_vertex_fa).__name__, type(per_vertex_fa[0]).__name__,
      per_vertex_fa[0].shape, streamlines[0].shape)
```

```text
{'polyline_count': 1000, 'vertex_count': 39019, 'chunk_count': 28, 'cross_chunk_link_count': 2591, 'group_count': 3}
list ndarray (55,) (55, 3)
```

Three things to note:

- **Per-vertex attributes are a list of arrays, one per polyline**, aligned with
  the geometry — `per_vertex_fa[k]` has as many values as `streamlines[k]` has
  vertices. Handing over one flat concatenated array instead fails, with a
  message that names nothing useful:

  ```pycon
  >>> flat = np.concatenate(per_vertex_fa)     # (39019,) — wrong shape
  >>> tracts.add_polylines(streamlines, attributes={"fa": flat})
  IndexError: invalid index to scalar variable.
  ```

  If you see that `IndexError`, check the shape of your attributes before
  anything else. (`add_points` is the opposite: a point cloud has no parts, so
  its attributes *are* one flat array per name.)
- **Group keys are integers at write time**: `groups` is
  `{group_id: [polyline_indices]}`, and a string key such as `{"CST": [...]}`
  raises `TypeError`. Names are attached afterwards, and are resolved on read —
  see [Reading a named group](#reading-a-named-group) below. A group whose
  members are contiguous may be passed as a `range` object, which is stored
  compactly.
- The declarations in the `Schema` (`vertex_attributes=`, `object_attributes=`)
  document the store and size its arrays. They are not enforced at write time,
  and the values are always passed to the write call.

### Acquisition metadata

There is no `streamline_metadata=` argument. Free-form per-format metadata —
step size, seeding strategy, the reference image a tractogram was run against —
goes in the store's header registry, one opaque JSON-compatible dict per source
format, stored as the attributes of a group under `headers/<format>/`:

```python
tracts.headers.add("trk", {
    "step_size": 0.5,
    "seeding": "wm_mask",
    "reference": "b0.nii.gz",
})
print(tracts.headers.available_formats)
print(tracts.headers.get("trk")["step_size"])
```

```text
['trk']
0.5
```

`zarr_vectors.headers` is a supported module, and the package only round-trips
the dict: giving the keys meaning is the format package's job.

---

## Reading streamlines

### Read everything

Every read of every geometry returns a `ReadResult`. For a polyline store
`positions` is the flat `(N, D)` vertex array and `polylines` is the derived
per-streamline view.

```python
tracts = zv.open("tracts.zarrvectors")

r = tracts.read()
print(r)
print(r.part_count, r.vertex_count, r.positions.shape)
print(len(r.polylines), r.polylines[0].shape)
print(r.attributes.names(), r.attributes["fa"].shape, r.attributes_read)
print(r.part_objects.shape, r.part_objects[:5])
```

```text
ReadResult(kind='polyline', vertices=39019, parts=1000, attributes=['fa'])
1000 39019 (39019, 3)
1000 (55, 3)
('fa',) (39019,) True
(1000,) [0 1 2 3 4]
```

`polylines[k]` is a real `(N_k, D)` array, reassembled across chunk boundaries,
not a list of per-chunk segments. `part_objects` is one object id *per part*,
not per vertex. `attributes_read` is worth checking: an empty `r.attributes`
with `attributes_read == False` means nobody looked, not that there are none.

### Read one streamline by ID

Object IDs are stable integer identifiers assigned at write time (0-indexed):

```python
one = tracts.level(0).objects[42]
print(one, one.part_count, one.vertex_count, one.polylines[0].shape)
print(np.allclose(one.polylines[0], streamlines[42]))
print(tracts.level(0).objects)
print(tracts.level(0).objects[[0, 42, 99]].part_count)
```

```text
ReadResult(kind='polyline', vertices=51) 1 51 (51, 3)
True
ObjectCatalog(level=0, count=1000, slots=1000)
3
```

### Reading a named group

Groups are written as integer rows, but they do not have to be *read* as
integers. Naming the rows is purely additive — the rows and their members are
untouched, and a reader that knows nothing about names sees exactly what it saw
before — and it is what makes the store self-describing:

```python
level = tracts.level(0)
print(level.groups)
print(level.groups.by_id(0))

tracts = zv.open("tracts.zarrvectors", mode="r+")
tracts.level(0).groups.name_rows(["CST", "AF", "UF"])

tracts = zv.open("tracts.zarrvectors")
print(tracts.groups)
print(list(tracts.groups))
print(tracts.groups["CST"].read())
print(tracts.groups["AF"].members[:5])
```

```text
GroupCatalog(['group_0', 'group_1', 'group_2'])
ObjectGroup('group_0', id=0, members=300)
GroupCatalog(['CST', 'AF', 'UF'])
['CST', 'AF', 'UF']
ReadResult(kind='polyline', vertices=11641, parts=300)
[300 301 302 303 304]
```

`ds.groups` is shorthand for `ds.level(0).groups`. A row that was never named
answers to `"group_<id>"`, so a store written before names existed is still
addressable without migrating it, and `by_id` reaches a row directly. An
unknown name raises, and the error lists what the level actually has:

```pycon
>>> tracts.groups["cst"]
KeyError: "no group named 'cst'; this level has ['CST', 'AF', 'UF']"
```

Names live on the groups array's own attributes. Other per-group *values* — a
numeric tract id, a hemisphere code — are a separate array family under
`group_attributes/`, which `add_polylines` has no argument for; write them with
`zarr_vectors.building.write_polylines(group_attributes={...})` and read them
back with `building.read_groupings_attributes(level_group, name)`.

### Read object attributes without fetching geometry

For large stores where you only need the per-streamline values — to filter by
length or FA before loading any geometry — the api names them and `building`
reads them. `read_object_attributes` returns a dense array indexed by object
ID:

```python
from zarr_vectors import building

print(tracts.level(0).attribute_names("object"))

root = building.open_store("tracts.zarrvectors", mode="r")
level_group = building.get_resolution_level(root, 0)
lengths = building.read_object_attributes(level_group, "length")
mean_fa = building.read_object_attributes(level_group, "mean_fa")
print(lengths.shape, lengths.dtype)

# Select long, high-FA streamlines, then fetch only those
good = np.nonzero((lengths > 300) & (mean_fa > 0.5))[0]
print(good.size)
sub = tracts.select(objects=good.tolist()).read()
print(sub, sub.part_count)
```

```text
('length', 'mean_fa')
(1000,) float32
349
ReadResult(kind='polyline', vertices=15738, parts=349) 349
```

Object-attribute values are one of the two gaps this page reaches into
`building` for (per-group attribute values, above, are the other):
`Level.attribute_names("object")` names them, but nothing on the `api` surface
returns them. `building` is the supported place to get them — reaching into
`zarr_vectors.core` for the same function is what turned earlier layout
refactors into downstream breaks, and a missing name is a gap to report rather
than a reason to import from `core`.

---

## Spatial bounding-box queries

`select()` builds a `Query` and reads nothing; `count()`, `read()` and
`object_ids()` are the terminals that touch the store. A bbox read of a
polyline store returns **whole** streamlines — never clipped, never split at
the boundary — so read it as *"give me the streamlines passing through this
region"*:

```python
q = tracts.select(bbox=((400.0, 400.0, 400.0), (600.0, 600.0, 600.0)))
print(q.explain())
rq = q.read()
print(rq, rq.vertex_count, rq.part_count)
print(q.cells())

inside = ((rq.positions >= 400.0) & (rq.positions <= 600.0)).all(axis=1)
print(int(inside.sum()))
```

```text
polyline read: level 0, bbox [400.0, 400.0, 400.0]..[600.0, 600.0, 600.0]
  via read_polylines(bbox, level)
ReadResult(kind='polyline', vertices=20366, parts=521) 20366 521
CellSet(8 cell(s))
4521
```

`explain()` names the reader that will run and the arguments it will get, and
`cells()` reads nothing and returns the grid cells the query touches.

The selection is **chunk-granular** for polylines: of the 521 streamlines
returned, only 219 have a vertex inside the box — the rest merely pass through
a chunk it overlaps. Filter in memory, as above, when you need vertex-exact
geometry. Note also that `count()` counts *vertices* for every geometry; use
`part_count` or `len(r.polylines)` for a streamline count.

Object ids narrow a region query, and intersect with it:

```python
ids = q.object_ids()[:3].tolist()
print(ids)
narrow = q.select(objects=ids)
print(narrow.explain())
print(narrow.read())
```

```text
[2, 8, 9]
polyline read: level 0, bbox [400.0, 400.0, 400.0]..[600.0, 600.0, 600.0], 3 object id(s)
  via read_polylines(bbox, level, object_ids)
ReadResult(kind='polyline', vertices=87, parts=3)
```

Chaining returns a **new** query; the original is untouched. An `objects=` id
that does not intersect the bbox is dropped rather than raising.

---

## Multi-resolution pyramids

```python
tracts = zv.open("tracts.zarrvectors", mode="r+")
print(zv.coarsen_methods())
rep = tracts.build_pyramid(
    factors=[(2.0, 1.0), (2.0, 1.0)],
    chunk_scale_factors=[2, 2],
    method="per_object",
)
print(rep["levels_created"], [s["vertex_count"] for s in rep["level_specs"]])

tracts = zv.open("tracts.zarrvectors")
for i in tracts.levels:
    lv = tracts.level(i)
    print(i, lv.vertex_count, lv.scale, lv.resolution, lv.attribute_names("vertex"))
```

```text
('per_object',)
2 [191, 28]
0 39019 (200.0, 200.0, 200.0) (50.0, 50.0, 50.0) ('fa',)
1 191 (400.0, 400.0, 400.0) (100.0, 100.0, 100.0) ()
2 28 (800.0, 800.0, 800.0) (200.0, 200.0, 200.0) ()
```

Each entry in `factors` is `(coarsen_factor, sparsity_factor)` applied to the
level below, so they compound: `[(2.0, 1.0), (2.0, 1.0)]` bins at 2× and 4× the
root bin. The sparsity factor drops objects — `(2.0, 4.0)` would keep a quarter
of the streamlines at that level — and `1.0` opts out. Pass
`chunk_scale_factors=` alongside, or every level inherits the root chunk shape
and `resolution(scale=)` has nothing to tell them apart by. `zv.coarsen_methods()`
lists the installed coarseners; `"per_object"` is the one in core, and it
aggregates each surviving streamline's vertices into bin centroids while
preserving its object id.

Note what a coarsened level does **not** inherit: object attributes carry over,
per-vertex attributes do not (`attribute_names("vertex")` is empty above). Re-open
the dataset after building — the handle that built the pyramid keeps stale level
metadata.

There is no supported call for adding or rebuilding a *single* level:
`zarr_vectors.multiresolution.coarsen.coarsen_level` and
`zarr_vectors.ops.refresh.rebuild_pyramid_from_level` both reach past the
contract into internal modules, and using either is a gap to report rather than
a settled spelling. `Dataset.build_pyramid` covers the whole-pyramid case,
which is the common one.

---

## Ingesting and exporting tractography formats

Format converters for TRK, TCK, and TRX (and the `zarr-vectors` CLI)
live in the companion package **`zarr-vectors-tools`**.

---

## Common pitfalls

**Links do not contain every edge of a streamline.**
Streamlines are `implicit_sequential`: within a chunk, consecutive
vertices are connected by *vertex order alone*, and no link record is
written for them. Only a transition that crosses a chunk boundary becomes a
stored link, under a non-zero offsets segment such as `links/0/0.0.+1/`. The
all-zero (intra-chunk) segment `links/0/0.0.0/` is never even created for this
type — there is no row that could live in it, which is also why a polyline
store has no `link_fragments/`:

```python
import os
print(sorted(os.listdir("tracts.zarrvectors/0")))
print(sorted(p for p in os.listdir("tracts.zarrvectors/0/links/0") if p != "zarr.json"))
print(os.path.exists("tracts.zarrvectors/0/links/0/0.0.0"))
```

```text
['groups', 'links', 'object_attributes', 'object_index', 'vertex_attributes', 'vertex_fragments', 'vertices', 'zarr.json']
['+1.+1.0', '+1.-1.0', '+1.0.+1', '+1.0.-1', '+1.0.0', '0.+1.+1', '0.+1.-1', '0.+1.0', '0.0.+1']
False
```

Inspecting the links family therefore shows you the chunk-boundary bridges, not
the streamline. Use `level.objects[k]` to retrieve a complete vertex sequence.

**A bbox query can miss a streamline that crosses the box.**
Selection is chunk-granular: a streamline is returned when it has a vertex in
a chunk the box touches. One whose vertices are spaced farther apart than a
chunk can pass clean through without depositing a vertex in any of them, and
will not be returned. Reduce the tractography step size, or use a finer
`Layout(cells=...)`, relative to the vertex spacing.

**Bounding-box reads of coarsened levels under-report.**
A bbox query against level 1 or 2 can return nothing while the level genuinely
holds vertices in that box. Read a coarse level whole and filter in memory
instead; bbox queries belong at level 0.

**Group names are resolved — group *ids* are still what is stored.**
`groups={...}` takes integer keys at write time and `select(groups=[...])`
takes integers, but names go on afterwards with `level.groups.name_rows([...])`
and are then the natural way in: `tracts.groups["CST"].read()`. Nothing has to
keep a private name→ID mapping any more, and a store that carries its names is
readable without the writing application's source next to it. `by_id` remains
for stores written by other tools that carry no names.

**Hand-written attribute arrays lose their alignment.**
Per-vertex attribute rows align 1:1 with the vertex rows in the same cell, and
the writers own that order — they permute the attribute arrays with the
geometry. An array written directly into `vertex_attributes/` outside
`add_polylines` / `building.write_polylines` is not permuted with it and will
be silently misaligned. Always write attributes through the write API. For
rechunking along a non-spatial dimension, `building.rechunk` with
`building.RechunkSpec` (or `building.rechunk_by_attribute`) is the supported
entry point — `zarr_vectors.rechunk` itself is internal.
