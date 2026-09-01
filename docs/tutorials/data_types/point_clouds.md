# Point clouds

Point clouds are the simplest ZVF geometry type: a collection of spatial
positions with optional per-vertex scalar or vector attributes. They arise
in synchrotron absorption imaging (HiP-CT, micro-CT), single-molecule
localisation microscopy (STORM, PALM, MINFLUX), spatial transcriptomics
(Visium HD, Xenium, MERFISH), and lidar scanning.

This tutorial covers writing, reading, spatial querying, attribute handling,
and building multi-resolution pyramids. Everything here is on the supported
`zarr_vectors.api` surface — re-exported from the top-level package, so
`zv.create` and `zv.open` below are `api` objects — apart from two places that
deliberately reach into `zarr_vectors.building`, the other supported surface,
for physical detail the api does not carry. Format converters for
LAS/PLY/CSV/XYZ live in the companion package **`zarr-vectors-tools`**.

The page is one continuous session — later blocks reuse the stores earlier
blocks create.

---

## Writing a point cloud

### Minimal write

A store starts as a `Schema`: where the data lives in space, what the axes
mean, and roughly how much of it there is. Chunk and bin shapes are not
arguments — they are derived from `Layout`, which says how finely to cut the
volume up.

```python
import numpy as np
import zarr_vectors as zv

ds = zv.create("scan_min.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    axes=(zv.Axis("x", unit="micrometer"),
          zv.Axis("y", unit="micrometer"),
          zv.Axis("z", unit="micrometer")),
    kind="point_cloud",
    expected=zv.SizeHints(n_vertices=100_000),
    layout=zv.Layout(cells=5),          # ~5 chunks per axis
))

rng = np.random.default_rng(42)
positions = rng.uniform(0, 1000, (100_000, 3)).astype(np.float32)

print(ds.add_points(positions))
print(ds.kinds, ds.ndim, ds.format_version)
print(ds.level(0).scale, ds.level(0).resolution)
print(ds.level(0).grid, ds.level(0).grid.shape)
print(ds.level(0).vertex_count)
```

```text
{'vertex_count': 100000, 'chunk_count': 125, 'object_count': 0, 'group_count': 0, 'bins_per_chunk': (4, 4, 4)}
('point_cloud',) 3 (0, 9, 0)
(200.0, 200.0, 200.0) (50.0, 50.0, 50.0)
Grid(5x5x5 cells of (200.0, 200.0, 200.0)) (5, 5, 5)
100000
```

`Level.scale` is the chunk shape — the I/O unit, one file per occupied cell —
and `Level.resolution` is the bin shape, the spatial index unit inside it. Five
cells per axis across 1 000 µm gives 200 µm chunks, and the default four
subcells per axis gives 50 µm bins, which is where `bins_per_chunk: (4, 4, 4)`
comes from. Counting does not read anything: `Level.vertex_count` answers from
metadata.

A fresh store holds only what has been written:

```python
import os
print(sorted(os.listdir("scan_min.zarrvectors")))
print(sorted(os.listdir("scan_min.zarrvectors/0")))
```

```text
['0', 'zarr.json']
['vertex_fragments', 'vertices', 'zarr.json']
```

`zarr.json` is the Zarr v3 group document; store-level fields live under its
`attributes.zarr_vectors` and the per-level transforms under
`attributes.multiscales`.

### Write with per-vertex attributes

Any number of named float or integer attribute arrays can be attached. Each
array has one entry per vertex — the same length as `positions` — and a vector
attribute keeps its trailing width:

```python
rng = np.random.default_rng(42)
n = 100_000
positions  = rng.uniform(0, 1000, (n, 3)).astype(np.float32)
intensity  = rng.uniform(0, 1, n).astype(np.float32)         # absorption
label      = rng.integers(0, 8, n).astype(np.int32)          # class label
rgb        = rng.integers(0, 256, (n, 3)).astype(np.uint8)   # colour
confidence = rng.uniform(0.5, 1, n).astype(np.float32)

ds = zv.create("scan.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    axes=(zv.Axis("x", unit="micrometer"),
          zv.Axis("y", unit="micrometer"),
          zv.Axis("z", unit="micrometer")),
    kind="point_cloud",
    vertex_attributes={
        "intensity":  zv.AttributeSpec(dtype="float32", unit="absorbance"),
        "label":      zv.AttributeSpec(dtype="int32", categorical=True),
        "color":      zv.AttributeSpec(dtype="uint8", channels=3),
        "confidence": zv.AttributeSpec(dtype="float32"),
    },
    expected=zv.SizeHints(n_vertices=n),
    layout=zv.Layout(cells=5),
))
print(ds.add_points(positions, attributes={
    "intensity": intensity, "label": label,
    "color": rgb, "confidence": confidence,
}))
print(ds.level(0).attribute_names("vertex"))
print(sorted(os.listdir("scan.zarrvectors/0/vertex_attributes")))
```

```text
{'vertex_count': 100000, 'chunk_count': 125, 'object_count': 0, 'group_count': 0, 'bins_per_chunk': (4, 4, 4)}
('color', 'confidence', 'intensity', 'label')
['color', 'confidence', 'intensity', 'label', 'zarr.json']
```

The `Schema` *declares* the attributes — dtype, channel width, whether a label
is categorical, what a value is measured in — and sizes their arrays; the values
themselves are always passed to the write call, as `attributes=`. Declarations
are documentation and layout input, not validation: nothing checks a written
array against its spec.

Each name becomes one child array under `vertex_attributes/`, on the same chunk
grid as `vertices/`, with rows aligned 1:1 with it.

### Axes and units

`Axis` carries the axis name and the unit it is measured in, and they are
written as OME-NGFF axes so other tooling can read them:

```python
import json
meta = json.load(open("scan.zarrvectors/zarr.json"))
print(sorted(meta["attributes"]))
print(meta["attributes"]["multiscales"][0]["axes"])
```

```text
['multiscales', 'zarr_vectors']
[{'name': 'x', 'type': 'space', 'unit': 'micrometer'}, {'name': 'y', 'type': 'space', 'unit': 'micrometer'}, {'name': 'z', 'type': 'space', 'unit': 'micrometer'}]
```

There is no coordinate-reference-system argument on `Schema`: axis names and
units are as much as the schema says about what the coordinates mean.

### Choosing the grid

A practical starting point: chunks large enough that each holds roughly
10 000–100 000 vertices, and the default four bins per chunk axis. `cells` is
the data-shaped spelling — cut the volume into about this many pieces per axis:

```python
total_vertices = 10_000_000
bounds_extent  = 4000.0                 # µm per axis
target_cells   = 8                      # -> 500 µm chunks
print(bounds_extent / target_cells, total_vertices / target_cells ** 3)
```

```text
500.0 19531.25
```

When the grid is fixed from outside — a pipeline whose chunks must line up with
an image volume's — `cell_size` sets it directly, and `subcells` sets the
number of bins per chunk axis:

```python
fixed = zv.create("fixed.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    kind="point_cloud",
    layout=zv.Layout(cell_size=(250.0, 250.0, 250.0), subcells=5),
))
print(fixed.level(0).scale, fixed.level(0).resolution)
```

```text
(250.0, 250.0, 250.0) (50.0, 50.0, 50.0)
```

See [Choosing a layout](../../how_to/choose_chunk_and_bin.md)
for the reasoning behind a particular size.

---

## Reading a point cloud

### Read everything

```python
ds = zv.open("scan.zarrvectors")

r = ds.read()
print(r)
print(r.vertex_count, r.positions.shape, r.positions.dtype)
print(r.attributes.names())
print(r.attributes["intensity"].shape, r.attributes["label"].dtype,
      r.attributes["color"].shape)
print(r.attributes_read, r.complete, r.errors)
```

```text
ReadResult(kind='point_cloud', vertices=100000, attributes=['color', 'confidence', 'intensity', 'label'])
100000 (100000, 3) float32
('color', 'confidence', 'intensity', 'label')
(100000,) int32 (100000, 3)
True True ()
```

A whole-store read brings the attributes with it, and vector attributes keep
their `(N, C)` shape. `attributes_read` is not decoration: an empty
`r.attributes` with `attributes_read == False` means *nobody looked*, not
*there are none* — it comes back `False` on narrowed reads whose reader could
not carry the attributes.

### Reading less

`select()` builds a `Query` and reads nothing; `count()` and `read()` are the
terminals that touch the store. Naming attributes narrows what is loaded:

```python
lean = ds.select(attributes=["intensity"]).read()
print(lean.attributes.names(), lean.attributes_read)
print(lean.vertex_count)
```

```text
('intensity',) True
100000
```

### Read a specific level

If the store has a multi-resolution pyramid, `level=` (or the `Level` handle)
reads a coarser representation — see [Multi-resolution
pyramids](#multi-resolution-pyramids) below, which builds one into this store.

---

## Spatial bounding-box queries

ZVF queries target individual bins — not full chunks — so for a point cloud the
box is vertex-exact: the count is the true number of points inside it, not the
contents of the chunks it overlaps.

```python
q = ds.select(bbox=((100.0, 100.0, 100.0), (200.0, 200.0, 200.0)))
print(q.explain())
print(q.count())
r = q.read()
print(r)
print(r.positions.min(axis=0))
print(r.positions.max(axis=0))
print(q.cells())
```

```text
point_cloud read: level 0, bbox [100.0, 100.0, 100.0]..[200.0, 200.0, 200.0]
  via read_points(attribute_names, bbox, level)
109
ReadResult(kind='point_cloud', vertices=109, attributes=['color', 'confidence', 'intensity', 'label'])
[100.439674 100.72887  100.454704]
[199.90297 198.85411 197.56297]
CellSet(8 cell(s))
```

`explain()` names the reader that will run and the arguments it will get, and
`cells()` reads nothing — eight cells, because a 100–200 µm box straddles the
200 µm chunk boundary on every axis.

A radius selection and a row limit narrow the same way:

```python
qn = ds.select(near=((500.0, 500.0, 500.0), 50.0))
print(qn.explain())
print(qn.count())
print(round(float(np.linalg.norm(qn.read().positions - 500.0, axis=1).max()), 3))

lim = ds.select(bbox=((0.0, 0.0, 0.0), (500.0, 500.0, 500.0))).limit(10).read()
print(lim.vertex_count, lim.truncated, lim.complete)
```

```text
point_cloud read: level 0, within 50.0 of [500.0, 500.0, 500.0]
  via read_points(attribute_names, bbox, level) then filtered in memory
59
49.895
10 True False
```

The sphere really is enforced — the farthest point is inside 50 µm — but
`explain()` is honest that it costs a bounding-box read plus an in-memory
filter. `limit()` sets `truncated`, and `complete` goes `False` to say the
result is not the whole answer.

---

## Multi-resolution pyramids

### Building a pyramid

```python
ds = zv.open("scan.zarrvectors", mode="r+")
print(zv.coarsen_methods())
rep = ds.build_pyramid(
    factors=[(2.0, 1.0), (2.0, 1.0)],
    chunk_scale_factors=[2, 2],
    method="per_object",
)
print(rep["levels_created"], [s["vertex_count"] for s in rep["level_specs"]])

ds = zv.open("scan.zarrvectors")
print(ds.levels)
for i in ds.levels:
    lv = ds.level(i)
    print(i, lv.vertex_count, lv.scale, lv.resolution, lv.attribute_names("vertex"))
print(sorted(ds.capabilities))
```

```text
('per_object',)
2 [1000, 125]
(0, 1, 2)
0 100000 (200.0, 200.0, 200.0) (50.0, 50.0, 50.0) ('color', 'confidence', 'intensity', 'label')
1 1000 (400.0, 400.0, 400.0) (100.0, 100.0, 100.0) ()
2 125 (800.0, 800.0, 800.0) (200.0, 200.0, 200.0) ()
['multiscale_links', 'shared_fragments']
```

Each entry in `factors` is a `(coarsen_factor, sparsity_factor)` pair applied to
the level below, so they compound: `[(2.0, 1.0), (2.0, 1.0)]` bins at 2× and 4×
the root bin. Either factor at `1.0` opts out — sparsity drops whole objects,
which does nothing to a point cloud written without object ids. Vertices within
a bin collapse to their centroid; `zv.coarsen_methods()` lists the installed
coarseners, and `"per_object"` is the one in core.

Pass `chunk_scale_factors=` alongside `factors=`, or every level inherits the
root chunk shape and `resolution(scale=)` has nothing to tell the levels apart
by. Re-open the dataset after building: the handle that built the pyramid keeps
stale level metadata. Note that per-vertex attributes are **not** carried into a
coarsened level — `attribute_names("vertex")` is empty above level 0.

A viewer usually wants a level by physical size rather than by index, which is
what `resolution(scale=)` is for:

```python
print(ds.resolution(scale=400.0).index, ds.resolution(scale=800.0).index)
print(ds.read(level=1).vertex_count)
print(ds.level(1).read())
```

```text
1 2
1000
ReadResult(kind='point_cloud', vertices=1000)
```

There is no supported call for adding or rebuilding a *single* level:
`zarr_vectors.multiresolution.coarsen.coarsen_level` and
`zarr_vectors.ops.refresh.rebuild_pyramid_from_level` both reach past the
contract into internal modules, and using either is a gap to report rather than
a settled spelling. `Dataset.build_pyramid` covers the whole-pyramid case,
which is the common one.

### Overview first, then detail

Overview-first rendering is a whole read of a coarse level, then a bounding-box
read at level 0 for the region of interest:

```python
overview = ds.read(level=2)
detail = ds.select(
    bbox=((400.0, 400.0, 400.0), (600.0, 600.0, 600.0)),
    level=0,
).read()
print(overview.vertex_count, detail.vertex_count)
```

```text
125 765
```

Do **not** put the box on the coarse level instead. Bounding-box queries against
coarsened levels currently under-report, and they under-report silently:

```python
box = ((400.0, 400.0, 400.0), (600.0, 600.0, 600.0))
lo, hi = np.array(box[0]), np.array(box[1])
for lvl in ds.levels:
    queried = ds.select(bbox=box, level=lvl).count()
    whole = ds.read(level=lvl).positions
    in_box = int((((whole >= lo) & (whole <= hi)).all(axis=1)).sum())
    print(lvl, queried, in_box)
```

```text
0 765 765
1 0 8
2 0 1
```

Level 0 agrees exactly; the coarse levels return nothing while genuinely
holding vertices in the box. Read a coarse level whole and filter in memory.

### Anisotropic data

For data with anisotropic sampling (4×4×25 nm voxels, say), the anisotropy
belongs in the layout — `cells` takes one count per axis — and in
`chunk_scale_factors`, whose entries may be per-axis tuples. The coarsen factor
itself is a single number, so binning coarsens isotropically:

```python
aniso = zv.create("aniso.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    kind="point_cloud",
    expected=zv.SizeHints(n_vertices=50_000),
    layout=zv.Layout(cells=(8, 8, 2)),      # coarse in z, fine in x and y
))
rng = np.random.default_rng(7)
aniso.add_points(rng.uniform(0, 1000, (50_000, 3)).astype(np.float32))
print(aniso.level(0).scale, aniso.level(0).resolution)

aniso = zv.open("aniso.zarrvectors", mode="r+")
aniso.build_pyramid(factors=[(2.0, 1.0)], chunk_scale_factors=[(2, 2, 1)])

aniso = zv.open("aniso.zarrvectors")
for i in aniso.levels:
    print(i, aniso.level(i).vertex_count, aniso.level(i).scale, aniso.level(i).resolution)
```

```text
(125.0, 125.0, 500.0) (31.25, 31.25, 125.0)
0 50000 (125.0, 125.0, 500.0) (31.25, 31.25, 125.0)
1 1024 (250.0, 250.0, 500.0) (62.5, 62.5, 250.0)
```

---

## Ingesting and exporting external formats

Format converters for LAS, PLY, CSV, and XYZ (and the `zarr-vectors`
CLI) live in the companion package **`zarr-vectors-tools`**.

---

## Validation

Validation levels 1–5 check progressively deeper properties of the store. Call
the module function with a filesystem path rather than `Dataset.validate()`,
which hands the validator a `file://` URL and reports `FAIL` on a perfectly
good local store:

```python
from zarr_vectors.validate import validate

result = validate("scan.zarrvectors", level=5)
print(result.summary())
print(result.ok, len(result.passed), len(result.warnings), len(result.errors))
```

```text
Level 5 validation: PASS
  40 passed, 1 warnings, 0 errors
  WARN:  Point cloud but links array exists
True 40 1 0
```

That warning is `build_pyramid`'s doing — it writes the cross-level links array
into the point-cloud store. A point cloud that has never been coarsened has no
`links/` at all. Note that `zarr_vectors.validate` is *undecided*: stable in
practice, but its result objects carry no compatibility promise. See
[Validation](../io/validation_and_repair.md) for what each level checks.

---

## Common pitfalls

**`add_points` is a one-shot write, not an append.**
A second call re-derives the grid from its own batch and overwrites what was
there; a batch that lands inside the first batch's extent silently replaces it.
Assemble the full array first, or write each batch to its own store.

```pycon
>>> ds2.add_points(first_1000_points)      # 1 000 points
>>> ds2.add_points(another_500_points)     # 500 more, same region
>>> ds2.read().vertex_count                # not 1 500
500
```

**Bin shape not dividing chunk shape.**
On this surface it cannot happen: `Layout` derives the bin shape *by dividing*
the chunk shape into `subcells` per axis, so any `subcells` divides exactly
(`subcells=3` over a 200 µm chunk gives 66.667 µm bins, which is correct, not a
rounding bug). Code that sets `bin_shape` by hand through
`zarr_vectors.building.write_points` is on its own — the writer does not check
— so check it yourself:

```pycon
>>> from zarr_vectors import building
>>> building.validate_bin_shape_divides_chunk((200.0, 200.0, 200.0), (60.0, 60.0, 60.0))
MetadataError: chunk_shape[0]=200.0 is not an integer multiple of bin_shape[0]=60.0 (ratio=3.333333)
```

**float64 positions.**
`Schema(position_dtype="float64")` is recorded on the schema, but `add_points`
writes float32 today, so coordinates that need sub-nanometre precision at
kilometre scale come back rounded. Until the api carries the declaration
through, `building.write_points(..., dtype="float64")` is the way to actually
store float64 — verify with `ds.read().positions.dtype` rather than assuming.
Be aware that float64 doubles storage size and reduces Blosc compression ratio.

**Attribute array of the wrong length.**
An attribute array shorter than `positions` fails during binning, with an index
error that names neither the attribute nor the writer:

```pycon
>>> ds.add_points(positions_100, attributes={"intensity": intensity_99})
IndexError: index 99 is out of bounds for axis 0 with size 99
```

If you see that, check whether your data pipeline dropped or duplicated rows.

**Reading a large store without a bbox.**
`ds.read()` loads every vertex into memory. For stores with tens of millions of
vertices, either take a bounding box, cap the result with `.limit(n)`, or walk
the query's cells and read one at a time:

```python
big = zv.open("scan.zarrvectors")
q = big.select(bbox=((0.0, 0.0, 0.0), (400.0, 400.0, 400.0)))
cells = q.cells()
print(cells, len(cells))
print(sum(big.select(cells=[c]).read().vertex_count for c in cells), q.count())
```

```text
CellSet(27 cell(s)) 27
21537 6465
```

A per-cell read returns whole chunks, so it sees more vertices than the exact
box does — filter each part as it arrives. (`Query.iter_cells()` is the
generator form of that loop and raises `NotImplementedError` for now: the
readers under it materialise the whole result, so a generator would use the
same peak memory while implying it does not.)
