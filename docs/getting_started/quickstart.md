# Quickstart

This page covers the most common operations in `zarr-vectors`: creating a
point-cloud store, writing to it, reading it back, running a spatial
bounding-box query, adding coarser resolution levels, and writing a set of
streamlines with group labels. All examples use synthetic data and require
only the base install.

The page is one continuous session — later blocks reuse the stores that
earlier blocks create.

---

## The two supported surfaces

`zarr-vectors` promises two public surfaces and reserves the right to change
everything else. The package will tell you which is which:

```python
import zarr_vectors as zv

for module in ("api", "building", "constants", "exceptions", "core", "lazy", "types"):
    print(module, "->", zv.stability(f"zarr_vectors.{module}"))
```

```text
api -> supported
building -> supported
constants -> supported
exceptions -> supported
core -> internal
lazy -> internal
types -> undecided
```

- **`zarr_vectors.api`** — the data-oriented surface used on this page. It is
  re-exported from the top-level package, so `zv.create`, `zv.open`,
  `zv.Schema` and everything else below are `api` objects.
- **`zarr_vectors.building`** — the surface for tools that *build* stores:
  converters, ingest pipelines, and anything that needs the writers and
  encoders underneath `add_points`.
- Everything reported `internal` (`core`, `lazy`, `spatial`, `encoding`, …) is
  an implementation detail and may change without notice. `undecided` means
  exactly that: still supported in practice, not yet promised.

`import zarr_vectors as zv` is the spelling used throughout. `zv.open_dataset`
and `zv.create_dataset` are aliases — the same function objects as `zv.open`
and `zv.create` — for code that imports the names bare and would rather not
shadow the builtin `open`.

### The older `zarr_vectors.types` functions

Earlier versions of this page used `zarr_vectors.types.write_points` /
`read_points` and their per-geometry siblings. They still work, but they are
superseded: they are storage-shaped rather than data-shaped — every call takes
the chunk and bin geometry — and each geometry's reader hands back its own
differently-shaped dict, so no two of them compose. The API below takes a
schema once and returns one `ReadResult` for every geometry.

---

## Point clouds

### Creating a store

A store starts as a `Schema`: where the data lives in space, what the axes
mean, which attributes to expect, and roughly how big it will be.

```python
import numpy as np
import zarr_vectors as zv

schema = zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    axes=(
        zv.Axis("x", unit="micrometer"),
        zv.Axis("y", unit="micrometer"),
        zv.Axis("z", unit="micrometer"),
    ),
    kind="point_cloud",
    vertex_attributes={
        "intensity": zv.AttributeSpec(dtype="float32"),
        "label": zv.AttributeSpec(dtype="int32", categorical=True),
    },
    expected=zv.SizeHints(n_vertices=100_000),
    layout=zv.Layout(cells=5),
)
ds = zv.create("scan.zarrvectors", schema=schema)

print(ds)
print(ds.bounds)
print(ds.kinds, ds.ndim, ds.format_version)
print(ds.level(0).scale, ds.level(0).resolution)
```

```text
Dataset('file:///.../scan.zarrvectors', point_cloud, levels=[0])
(array([0., 0., 0.]), array([1000., 1000., 1000.]))
('point_cloud',) 3 (0, 9, 0)
(200.0, 200.0, 200.0) (50.0, 50.0, 50.0)
```

The store is written as a directory tree called `scan.zarrvectors/`. A
relative path is normalised to an absolute `file://` URL, which is what the
repr shows.

`vertex_attributes` and `expected` are declarative: they document the store
and size its arrays, but nothing enforces them at write time.

### Where `chunk_shape` and `bin_shape` went

Nothing about the format changed. A **chunk** is still the I/O unit — one file
on disk — and a **bin** is still the spatial index unit inside it. They are
now layout concerns with working defaults instead of mandatory arguments to
every call.

`zv.Layout(cells=5)` divides the 1000 µm bounds into five cells per axis,
which is where the numbers printed above come from: `Level.scale` is the chunk
shape (200 µm) and `Level.resolution` is the bin shape (50 µm, i.e. four bins
per axis, 64 bins per chunk). Those are exactly the values earlier versions of
this page passed by hand as `chunk_shape=(200., 200., 200.)` and
`bin_shape=(50., 50., 50.)` — derived now instead of dictated. The resulting
grid is a first-class object:

```python
grid = ds.level(0).grid
print(grid)
print(grid.shape, grid.cell_shape)
```

```text
Grid(5x5x5 cells of (200.0, 200.0, 200.0))
(5, 5, 5) (200.0, 200.0, 200.0)
```

Storage geometry is no longer accepted anywhere on this surface:

```pycon
>>> zv.Schema(bounds=((0., 0., 0.), (1., 1., 1.)), chunk_shape=(1., 1., 1.))
TypeError: Schema.__init__() got an unexpected keyword argument 'chunk_shape'
>>> zv.create("nope.zarrvectors", schema=schema, chunk_shape=(1., 1., 1.))
TypeError: create() got an unexpected keyword argument 'chunk_shape'
```

`layout=` is itself optional. `bounds` is not — without it the layout cannot
be resolved:

```python
ds_min = zv.create(
    "minimal.zarrvectors",
    schema=zv.Schema(bounds=((0., 0., 0.), (1000., 1000., 1000.))),
)
print(ds_min.level(0).scale, ds_min.level(0).resolution)
```

```text
(1000.0, 1000.0, 1000.0) (250.0, 250.0, 250.0)
```

That is the honest default — `Layout()` means `cells="auto"`, one chunk for
the whole volume — and it is almost never what you want. Pass `cells=`.

For the reasoning behind a particular chunk and bin size, see
[Core concepts](concepts.md) and
[Choosing a layout](../how_to/choose_chunk_and_bin.md).

If the store may already exist, `zv.open_or_create` is the entry point that
checks the schema you passed against the one on disk:

```pycon
>>> zv.open_or_create("minimal.zarrvectors",
...                   schema=zv.Schema(bounds=((0., 0., 0.), (500., 500., 500.))))
SchemaConflict: file:///.../minimal.zarrvectors already exists and disagrees with the schema it was opened with:
  bounds max: [500.0, 500.0, 500.0] != [1000.0, 1000.0, 1000.0]
Pass on_conflict='keep' to use the store's own values.
```

### Writing points

```python
rng = np.random.default_rng(42)

# 100 000 points uniformly distributed in a 1 000³ µm volume
positions = rng.uniform(0, 1000, size=(100_000, 3)).astype(np.float32)
intensity = rng.uniform(0, 1, size=100_000).astype(np.float32)
label = rng.integers(0, 5, size=100_000).astype(np.int32)

report = ds.add_points(
    positions,
    attributes={"intensity": intensity, "label": label},
)
print(report)
```

```text
{'vertex_count': 100000, 'chunk_count': 125, 'object_count': 0, 'group_count': 0, 'bins_per_chunk': (4, 4, 4)}
```

**`add_points` is a one-shot write, not an append.** A second call into the
same store re-derives the grid from its own batch and overwrites what was
there. A batch that lands inside the first batch's extent silently replaces it.
A batch that lands outside may raise, as below — but only when the new grid is
smaller than the coordinates require; it is not a guarantee, so do not rely on
an error to catch this:

```pycon
>>> ds2.add_points(first_1000_points)       # 1 000 points
>>> ds2.add_points(another_500_points)      # 500 more, same region
>>> ds2.read().vertex_count                 # not 1 500
500
>>> ds2.add_points(points_somewhere_else)   # a batch outside the first extent
StoreError: Chunk coords (3, 3, 3) out of grid (2, 2, 2) for array 'vertices'
```

Assemble the full array first, or write each batch to its own store.

### Reading everything back

```python
r = ds.read()
print(r)
print(r.vertex_count)
print(r.positions.shape, r.positions.dtype)
print(r.attributes.names())
print(r.attributes["intensity"].shape, r.attributes["label"].dtype)
print(r.attributes_read, r.complete, r.errors)
print(r.part_count)
```

```text
ReadResult(kind='point_cloud', vertices=100000, attributes=['intensity', 'label'])
100000
(100000, 3) float32
('intensity', 'label')
(100000,) int32
True True ()
1
```

`attributes_read` is not decoration: an empty `r.attributes` with
`attributes_read == False` means *nobody looked*, not *there are none*. It
comes back `False` on narrowed reads whose reader could not carry the
attributes.

Counting does not have to read anything. `Level` answers from metadata:

```python
print(ds.level(0).vertex_count, ds.level(0).attribute_names("vertex"))
```

```text
100000 ('intensity', 'label')
```

### Querying a bounding box

`select()` builds a `Query` and reads nothing:

```python
q = ds.select(bbox=((100.0, 100.0, 100.0), (200.0, 200.0, 200.0)))
print(type(q).__name__)
print(q)
print(q.explain())
```

```text
Query
Query(point_cloud read: level 0, bbox [100.0, 100.0, 100.0]..[200.0, 200.0, 200.0])
point_cloud read: level 0, bbox [100.0, 100.0, 100.0]..[200.0, 200.0, 200.0]
  via read_points(attribute_names, bbox, level)
```

`explain()` names the reader that will run and the arguments it will get.
`count()`, `read()` and `object_ids()` are the terminals — they are what
actually touch the store:

```python
print(q.count())
r = q.read()
print(r)
print(r.positions.min(axis=0))
print(r.positions.max(axis=0))
print(r.attributes["intensity"][:3])
print(q.cells())
```

```text
109
ReadResult(kind='point_cloud', vertices=109, attributes=['intensity', 'label'])
[100.439674 100.72887  100.454704]
[199.90297 198.85411 197.56297]
[0.5707176 0.9738375 0.6510283]
CellSet(8 cell(s))
```

For a point cloud the box is vertex-exact: 109 is the true number of points in
it, not the contents of the chunks it overlaps. `q.cells()` reads nothing and
returns the grid cells the query touches — eight of them, because a 100–200 µm
box straddles the 200 µm chunk boundary on every axis.

### Narrowing a query

Chaining returns a **new** query; the original is untouched, and two boxes
intersect geometrically (note the max corner clamped to 200, not 400):

```python
inner = q.select(bbox=((150.0, 150.0, 150.0), (400.0, 400.0, 400.0)))
print(inner.selection.bbox)
print(inner.count())
print(q.count())
```

```text
(array([150., 150., 150.]), array([200., 200., 200.]))
14
109
```

Attributes, radius selections and limits narrow the same way:

```python
lean = q.select(attributes=["intensity"])
print(lean.read().attributes.names())

qn = ds.select(near=((500.0, 500.0, 500.0), 50.0))
print(qn.explain())
print(qn.count())
print(round(float(np.linalg.norm(qn.read().positions - 500.0, axis=1).max()), 3))

lim = ds.select(bbox=((0., 0., 0.), (500., 500., 500.))).limit(10).read()
print(lim.vertex_count, lim.truncated, lim.complete)
```

```text
('intensity',)
point_cloud read: level 0, within 50.0 of [500.0, 500.0, 500.0]
  via read_points(attribute_names, bbox, level) then filtered in memory
59
49.895
10 True False
```

The sphere really is enforced — the farthest of those 59 points is 49.895 µm
from the centre — but `explain()` is honest that it costs a bounding-box read
plus an in-memory filter. `limit()` sets `truncated`, and `complete` goes
`False` to say the result is not the whole answer.

Two narrowings need a store built for them. `select(objects=…)` needs object
ids in the store (see [Reading one object by ID](#reading-one-object-by-id));
against a plain point cloud written without them it fails with a bare
`KeyError: 'sid_ndim'`. `select(where=…)` needs a store written with
attribute chunking:

```pycon
>>> q.select(where={"label": 3}).count()
ArrayError: attribute_filter requires a store written with chunk_by_attribute (level metadata has no chunk_attribute_name)
```

### Adding coarser levels

`build_pyramid` adds decimated levels for level-of-detail rendering. Pass
`chunk_scale_factors=` alongside `factors=`, or every level inherits the root
chunk shape and `resolution(scale=)` has nothing to distinguish them by:

```python
ds = zv.open("scan.zarrvectors", mode="r+")
print(zv.coarsen_methods())
report = ds.build_pyramid(factors=[(2.0, 1.0), (2.0, 1.0)], chunk_scale_factors=[2, 2])
print(report["levels_created"], [s["vertex_count"] for s in report["level_specs"]])
print(ds.levels)
```

```text
('per_object',)
2 [1000, 125]
(0, 1, 2)
```

Re-open the dataset afterwards. The handle that built the pyramid keeps stale
level metadata — vertex counts are right, but `resolution` still reports the
base bin size until the store is opened again:

```python
ds = zv.open("scan.zarrvectors")
for index in ds.levels:
    level = ds.level(index)
    print(index, level.vertex_count, level.scale, level.resolution)
```

```text
0 100000 (200.0, 200.0, 200.0) (50.0, 50.0, 50.0)
1 1000 (400.0, 400.0, 400.0) (100.0, 100.0, 100.0)
2 125 (800.0, 800.0, 800.0) (200.0, 200.0, 200.0)
```

Read a level either through its `Level` handle or with `level=`:

```python
coarse = ds.level(1)
print(coarse)
print(coarse.read())
print(ds.read(level=1).vertex_count)
```

```text
Level(1, kind='point_cloud', vertices=1000)
ReadResult(kind='point_cloud', vertices=1000)
1000
```

A viewer usually wants a level by physical size rather than by index, which is
what `resolution(scale=)` is for:

```python
print(ds.resolution(scale=400.0).index)
print(ds.resolution(scale=800.0).index)
print(ds.resolution(scale=200.0).index)
```

```text
1
2
0
```

Whole-level reads at coarse levels are exact. Bounding-box queries against
coarsened levels currently under-report, so read the level whole and filter in
memory if you need a region at low resolution.

See [Building pyramids](../tutorials/multiscale/building_pyramids.md) for the
coarsening strategies and what they preserve.

---

## Streamlines

### Writing polylines

Polylines are written as a list of `(N_k, D)` arrays — one array per
streamline — and their vertex attributes are a **list of one array per
polyline**, aligned with the geometry:

```python
rng = np.random.default_rng(0)

# 500 streamlines, each with 40 vertices, walking through 3-D space
starts = rng.uniform(200.0, 800.0, size=(500, 3))
streamlines = [
    (start + rng.normal(0.0, 8.0, size=(40, 3)).cumsum(axis=0)).astype(np.float32)
    for start in starts
]
# one (40,) array per streamline — NOT one flat (20000,) array
fa = [rng.uniform(0.0, 1.0, size=len(s)).astype(np.float32) for s in streamlines]

tracts = zv.create("tracts.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    axes=(zv.Axis("x", unit="micrometer"),
          zv.Axis("y", unit="micrometer"),
          zv.Axis("z", unit="micrometer")),
    kind="polyline",
    vertex_attributes={"fa": zv.AttributeSpec(dtype="float32")},
    expected=zv.SizeHints(n_vertices=20_000, n_objects=500),
    layout=zv.Layout(cells=5),
))
report = tracts.add_polylines(
    streamlines,
    attributes={"fa": fa},
    # Optional: assign streamlines to groups
    groups={0: list(range(250)), 1: list(range(250, 500))},
)
print(len(streamlines), streamlines[0].shape)
print(type(fa[0]).__name__, fa[0].shape)
print(report)
```

```text
500 (40, 3)
ndarray (40,)
{'polyline_count': 500, 'vertex_count': 20000, 'chunk_count': 92, 'cross_chunk_link_count': 1878, 'group_count': 2}
```

Passing one flat per-vertex array instead of a list fails, and the message
names nothing useful — if you see this, check the shape of your attributes:

```pycon
>>> flat = rng.uniform(0.0, 1.0, size=20_000).astype(np.float32)   # (20000,) — wrong
>>> tracts.add_polylines(streamlines, attributes={"fa": flat})
IndexError: invalid index to scalar variable.
```

`add_polylines(..., streamlines=True)` adds `"streamline"` to the store's
declared geometry kinds — it does not replace `"polyline"`. A store created with
`kind="polyline"` and written with the flag reports
`ds.kinds == ('polyline', 'streamline')`, while `ReadResult.kind` and
`Level.kind` both stay `"polyline"`. Note also that `Schema(kind=...)` is
recorded, not validated: `zv.create` accepts any string, so a typo becomes a
store whose declared kind no reader dispatches on.

### Reading a group

Groups written as `{0: [...], 1: [...]}` come back as rows named `group_0`
and `group_1`:

```python
level = tracts.level(0)
print(level.groups)
group = level.groups.by_id(0)
print(group)
print(len(group), group.members[:5])
print(group.read())
```

```text
GroupCatalog(['group_0', 'group_1'])
ObjectGroup('group_0', id=0, members=250)
250 [0 1 2 3 4]
ReadResult(kind='polyline', vertices=10000, parts=250)
```

Naming the rows is additive, and makes the store self-describing:

```python
level.groups.name_rows(["left", "right"])
tracts = zv.open("tracts.zarrvectors")
print(tracts.groups)
print(list(tracts.groups))
print(tracts.groups["left"].read())
print(tracts.groups["right"].members[:5])
```

```text
GroupCatalog(['left', 'right'])
['left', 'right']
ReadResult(kind='polyline', vertices=10000, parts=250)
[250 251 252 253 254]
```

(`ds.groups` is shorthand for `ds.level(0).groups`.)

### Reading one object by ID

```python
one = tracts.level(0).objects[42]
print(one)
print(one.part_count, one.vertex_count, one.polylines[0].shape)
print(np.allclose(one.polylines[0], streamlines[42]))
print(tracts.level(0).objects)
print(tracts.level(0).objects[[42, 43]].part_count)
```

```text
ReadResult(kind='polyline', vertices=40)
1 40 (40, 3)
True
ObjectCatalog(level=0, count=500, slots=500)
2
```

`polylines[0]` is a real `(40, 3)` array that round-trips the input
streamline, not a list of per-chunk segments: the result object reassembles
each streamline before handing it back.

### One result type, two geometries

Every read of every geometry returns a `ReadResult`:

```python
all_tracts = tracts.read()
print(all_tracts)
print(all_tracts.kind, all_tracts.vertex_count, all_tracts.part_count, all_tracts.positions.shape)
print(len(all_tracts.polylines), all_tracts.polylines[0].shape)
print(all_tracts.attributes.names(), all_tracts.attributes["fa"].shape, all_tracts.attributes_read)
print(all_tracts.part_objects.shape, all_tracts.part_objects[:5])

points = zv.open("scan.zarrvectors").read()
for result in (points, all_tracts):
    print(result.kind, result.positions.shape, result.part_count, len(result.polylines))
```

```text
ReadResult(kind='polyline', vertices=20000, parts=500, attributes=['fa'])
polyline 20000 500 (20000, 3)
500 (40, 3)
('fa',) (20000,) True
(500,) [0 1 2 3 4]
point_cloud (100000, 3) 1 1
polyline (20000, 3) 500 500
```

That last pair of lines is the point: `.positions` is `(N, D)` for both kinds
and `.polylines` is the derived per-part view — one whole-store part for the
point cloud, 500 for the tracts. `.part_objects` is one object id *per part*,
not per vertex.

### Region queries on streamlines

A bounding-box read of a polyline store is chunk-granular and returns **whole**
streamlines — read it as *"give me the streamlines passing through this
region"*:

```python
q = tracts.select(bbox=((410.0, 410.0, 410.0), (590.0, 590.0, 590.0)))
print(q.explain())
rq = q.read()
print(rq)
print(rq.vertex_count, rq.part_count)

inside = ((rq.positions >= 410.0) & (rq.positions <= 590.0)).all(axis=1)
print(int(inside.sum()))
```

```text
polyline read: level 0, bbox [410.0, 410.0, 410.0]..[590.0, 590.0, 590.0]
  via read_polylines(bbox, level)
ReadResult(kind='polyline', vertices=1800, parts=45)
1800 45
631
```

45 streamlines × 40 vertices = 1800 vertices, of which 631 are actually inside
the box; filter in memory as above if you need vertex-exact geometry. Note
also that `count()` counts *vertices* for every geometry — use `.part_count`
or `len(r.polylines)` for a streamline count.

Object ids narrow a region query, and `object_ids()` is a terminal — it reads:

```python
ids = q.object_ids()[:3].tolist()
print(ids)
narrow = q.select(objects=ids)
print(narrow.explain())
rn = narrow.read()
print(rn)
print(np.unique(rn.part_objects).tolist())
print(q.count(), narrow.count())
```

```text
[2, 14, 21]
polyline read: level 0, bbox [410.0, 410.0, 410.0]..[590.0, 590.0, 590.0], 3 object id(s)
  via read_polylines(bbox, level, object_ids)
ReadResult(kind='polyline', vertices=120, parts=3)
[2, 14, 21]
1800 120
```

An `objects=` id that does not intersect the bbox is dropped rather than
raising — correct intersection semantics, occasionally surprising.

---

## Format converters

Ingesting from third-party formats (LAS, PLY, CSV, TRK, TCK, TRX, SWC,
GraphML, OBJ, STL) and exporting back to them lives in the companion
package **`zarr-vectors-tools`**, alongside the `zarr-vectors` CLI.

---

## Validation

Validation levels 1–5 check progressively deeper properties of the store. Call
the module function with a filesystem path:

```python
from zarr_vectors.validate import validate

result = validate("scan.zarrvectors", level=3)
print(result.summary())
print(result.ok, len(result.passed), len(result.warnings), len(result.errors))
```

```text
Level 3 validation: PASS
  36 passed, 0 warnings, 0 errors
True 36 0 0
```

```python
print(validate("scan.zarrvectors", level=5).summary())
print(validate("tracts.zarrvectors", level=5).summary())
```

```text
Level 5 validation: PASS
  40 passed, 1 warnings, 0 errors
  WARN:  Point cloud but links array exists
Level 5 validation: PASS
  26 passed, 0 warnings, 0 errors
```

(That warning appears because `build_pyramid` wrote a links array into the
point-cloud store; a freshly written point cloud has none.)

Use the module function rather than `Dataset.validate()`: the method hands the
dataset's `file://` URL to a path-based validator and reports `FAIL` on a
perfectly good local store. Note also that `zarr_vectors.validate` is one of
the `undecided` modules above. See
[Validation](../tutorials/io/validation_and_repair.md) for what each level
checks.

---

## Versions and capabilities

Three versions matter, and they move independently: the package, the API
surface, and the on-disk format.

```python
ds = zv.open("scan.zarrvectors")
print(zv.__api_version__)        # the API surface this page documents
print(ds.format_version)         # the on-disk format of this store
print(sorted(ds.capabilities), ds.supports("nonempty_chunks"))
```

```text
(1, 0)
(0, 9, 0)
['multiscale_links', 'shared_fragments'] False
```

The third is `zv.__version__`, the installed package version. It comes from
setuptools-scm and moves with every commit, so never assert against it.

`capabilities` describes what this particular store can do — the two above
were added by `build_pyramid`. Assert what you need rather than testing
version numbers by hand:

```pycon
>>> zv.require_format(ds, ">=0.9")     # returns None
>>> zv.require_format(ds, ">=1.0")
FormatError: file:///.../scan.zarrvectors is on-disk format 0.9.0, which does not satisfy '>=1.0'. There is no backward-compatible reader: an older store must be rewritten from source, and a newer one needs a newer zarr-vectors.
>>> zv.require_api(">=1.0", features=["surfaces", "query-cells"])   # returns None
>>> zv.require_api(features=["streaming-reads"])
ImportError: zarr-vectors 1.0 does not provide: streaming-reads. Known features: coarsen-strategy-options, presence-rebuild, query-cells, selection-level-optional, sharded-presence-guard, surfaces, vertex-attributes-on-read.
```

`require_format` raises a `FormatError` (a `zarr-vectors` exception — the data
is wrong), while `require_api` raises `ImportError` deliberately: the
*installation* is wrong, and that is what import machinery reports.
`zv.FEATURES` lists the feature names it knows.

---

## Next steps

- **[Concepts](concepts.md)** — understand chunks, bins, fragments, and
  the multiscale pyramid before working with larger datasets.
- **[Choosing a layout](../how_to/choose_chunk_and_bin.md)** —
  what to pass when the defaults behind `Layout` are not right for your data.
- **[Data type tutorials](../tutorials/data_types/point_clouds.md)** — deeper
  walkthroughs for each geometry type.
- **[Building pyramids](../tutorials/multiscale/building_pyramids.md)** — add
  multi-resolution levels for level-of-detail rendering.
- **[Neuroglancer integration](../tutorials/neuroglancer/overview.md)** — visualise
  your stores in Neuroglancer using `zv-ngtools`.
