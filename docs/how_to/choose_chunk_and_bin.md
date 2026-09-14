# Choosing a layout: cells, cell size and subcells

The quickstart's
[Where `chunk_shape` and `bin_shape` went](../getting_started/quickstart.md#where-chunk_shape-and-bin_shape-went)
explains the change this page continues. Nothing about the format moved: a
**chunk** is still the I/O unit — one file on disk, one object in a bucket —
and a **bin** is still the spatial index unit inside it. What moved is who
computes them. You no longer pass `chunk_shape=` and `bin_shape=` to every
write; you attach a `zv.Layout` to the `Schema` once, and
`Layout.resolve` derives them.

So this page is about choosing the `Layout` — and, since the numbers are now
derived rather than dictated, about checking a choice with `zv.Grid` *before*
writing anything.

---

## The quick version

If you are unsure, start here and tune later. `cells=N` means "cut the volume
into about `N` pieces per axis", which is the spelling most callers want:

```python
import zarr_vectors as zv

# 10 million points in a 4 000³ µm volume
schema = zv.Schema(
    bounds=((0.0, 0.0, 0.0), (4000.0, 4000.0, 4000.0)),
    kind="point_cloud",
    expected=zv.SizeHints(n_vertices=10_000_000),
    layout=zv.Layout(cells=8),
)
resolved = schema.layout.resolve(schema)
print(resolved.chunk_shape, resolved.bin_shape)
```

```text
(500.0, 500.0, 500.0) (125.0, 125.0, 125.0)
```

Those are exactly the values an older version of this page told you to pass by
hand as `chunk_shape=(500., 500., 500.)`. The rule of thumb behind `cells=8`
is unchanged: aim for roughly 10 000–100 000 vertices in each cell, which here
is 10 000 000 / 8³ ≈ 19 500.

Check it before writing anything:

```python
grid = zv.Grid.plan(((0.0, 0.0, 0.0), (4000.0, 4000.0, 4000.0)), target_cells=8)
print(grid)
print(grid.capacity(n_vertices=10_000_000))
```

```text
Grid(8x8x8 cells of (500.0, 500.0, 500.0))
8x8x8 = 512 cells, ~0.2 MB/cell -- fits
```

---

## What `Layout` actually controls

`Layout` has one job: turn the data's `bounds` into the four physical
parameters the storage layer needs. Every field has a working `"auto"`.

| Field | What it says | Default |
|-------|--------------|---------|
| `cells` | How many cells per axis to cut the volume into | `"auto"` → 1, the whole volume in one cell |
| `cell_size` | The cell size in coordinate units, when the grid is fixed from outside | `None` |
| `subcells` | How many bins per axis inside each cell | `"auto"` → 4 |
| `pack` | Whether to pack cells into shards | `"auto"` → on for object stores, off for local |
| `compression` | Compressor name | `"auto"` → `$ZARR_VECTORS_COMPRESSION`, else none |

`Layout.resolve(schema)` computes `chunk_shape`, `bin_shape`, `shard_shape`
and `compressor` from those, and it is the only place in the package where
those four are produced. It needs `Schema.bounds` — the cell size is a
fraction of the extent, so with nothing to divide there is nothing to derive:

```pycon
>>> zv.Layout(cells=8).resolve(zv.Schema())
MetadataError: Layout.resolve needs Schema.bounds: the cell size is a fraction of the extent, so there is nothing to divide.
```

The object it returns is internal — the supported surface never names
`chunk_shape` or `bin_shape`. Once a store exists the same two numbers come
back as `Level.scale` (the cell size) and `Level.resolution` (the bin size),
and the grid itself as `Level.grid`.

### `cells` or `cell_size`

`cells` is the data-shaped spelling and should be your default. `cell_size`
is the escape hatch for a grid fixed from outside — a pipeline whose cells
must line up with an image volume's, or a store you are rewriting to match an
existing one. For an extent that divides exactly, the two are the same thing
said two ways:

```python
fixed = zv.Schema(
    bounds=((0.0, 0.0, 0.0), (4000.0, 4000.0, 4000.0)),
    layout=zv.Layout(cell_size=(500.0, 500.0, 500.0)),
)
print(fixed.layout.resolve(fixed).chunk_shape)
print(zv.Grid.plan(((0.0, 0.0, 0.0), (4000.0, 4000.0, 4000.0)),
                   cell_size=(500.0, 500.0, 500.0)))
```

```text
(500.0, 500.0, 500.0)
Grid(8x8x8 cells of (500.0, 500.0, 500.0))
```

When it does not divide exactly, `cell_size` keeps the size you asked for and
the grid overhangs the bounds, while `cells` keeps the count and adjusts the
size. Both are legitimate; pick the one whose invariant you actually need.

`cell_size` wins when both are set.

### `subcells`

`bin_shape` must divide `chunk_shape` exactly, which is why `Layout` takes a
count rather than a size — a size is a number a caller can get wrong:

```python
for subcells in ("auto", 1, 2, 4, 8):
    s = zv.Schema(
        bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
        layout=zv.Layout(cells=5, subcells=subcells),
    )
    r = s.layout.resolve(s)
    print(subcells, r.chunk_shape, r.bin_shape)
```

```text
auto (200.0, 200.0, 200.0) (50.0, 50.0, 50.0)
1 (200.0, 200.0, 200.0) (200.0, 200.0, 200.0)
2 (200.0, 200.0, 200.0) (100.0, 100.0, 100.0)
4 (200.0, 200.0, 200.0) (50.0, 50.0, 50.0)
8 (200.0, 200.0, 200.0) (25.0, 25.0, 25.0)
```

### `pack`

One storage object per cell is fine on a local filesystem, where an `open()`
is cheap. Against S3 or GCS each one is a request, so cells are packed into
shards until a shard is worth a round-trip — 16 MiB by default, tunable with
`Layout(target_object_bytes=...)`. `resolve` decides from the store's URL
scheme, so you normally leave it alone:

```python
s = zv.Schema(
    bounds=((0.0, 0.0, 0.0), (8000.0, 8000.0, 8000.0)),
    expected=zv.SizeHints(n_vertices=200_000_000),
    layout=zv.Layout(cells=16),
)
print(s.layout.resolve(s, store_kind="local").shard_shape)
print(s.layout.resolve(s, store_kind="object").shard_shape)
```

```text
None
(3, 3, 3)
```

3³ = 27 cells of ~0.6 MB each is ~16 MB per object, which is the target. The
size hint is what makes that arithmetic possible — with `expected` unset,
`resolve` falls back to a flat 4 cells per axis per shard. Wrong hints cost
performance, never correctness.

---

## Checking a choice before you write

`zv.Grid` answers two questions about a grid that does not exist yet, which is
the only moment the answers can change a decision. Discovering after the fact
that your cells are 100 MB each means throwing away a store you have already
written.

- `zv.Grid.plan(bounds, target_cells=...)` — or `cell_size=...` — builds the
  grid a store with those bounds would get.
- `grid.capacity(n_vertices=...)` — is each cell a sane size?
- `grid.cells_in((lo, hi))` — how many cells does a query touch?
- `grid.cell_of(point)` — which cell is this coordinate in?
- `grid.holds(ref)` — is that cell inside the allocation?

`cell_of` and `cells_in` hand back opaque `CellRef` values rather than integer
coordinates, so a change in how cells are addressed is not a downstream break.

`capacity` estimates bytes per cell from the vertex count and flags a grid
whose cells are too big:

```python
coarse = zv.Grid.plan(((0.0, 0.0, 0.0), (4000.0, 4000.0, 4000.0)), target_cells=1)
print(coarse)
print(coarse.capacity(n_vertices=10_000_000))
```

```text
Grid(1x1x1 cells of (4000.0, 4000.0, 4000.0))
1x1x1 = 1 cells, ~120.0 MB/cell -- does not fit: ~120 MB per cell exceeds the 67 MB target; use more cells
```

`fits` is an upper-bound check only. Nothing warns you about cells that are
too *small*, and that is the more common mistake — see the HiP-CT example
below. Read `est_bytes_per_cell` yourself and compare it against the floor in
the next section.

The estimate counts positions only: `n_vertices × ndim × 4` bytes, divided
across the cells. Per-vertex attributes add to that in proportion, so a store
with five `float32` attributes will be roughly 2.7× the figure `capacity`
reports.

---

## Cell-size guidance

### Primary consideration: I/O unit size

Each cell is one file on disk or one object in cloud storage. Choose `cells`
so that:

- **Typical queries load 1–8 cells.** If a query always hits exactly one
  cell, the cell size is well matched to the query size. `Grid.cells_in`
  measures this without reading anything.
- **Each cell holds 50 KB–50 MB compressed.** Below ~50 KB the per-request
  overhead dominates; above ~50 MB a small query drags in far more than it
  needs.

Those two pull in opposite directions, and the second is the one to honour
when they conflict: a query that touches 60 small cells is slow, but a store
of 4 KB objects is slow for *every* access pattern.

Target 10 000–100 000 vertices per cell for point clouds. For sparser
geometry types (streamlines, skeletons), 100–500 objects per cell is the
older rule of thumb, and it holds for stores of tens of thousands of objects.
At a million objects it cannot be met without cells so small that any real
query touches hundreds of them; there, size the cells by bytes and let the
object count fall where it does.

### Converting a target size to `cells`

The table below is in physical units because that is how a use case is
usually stated. `cells` is `extent / target`, rounded to something tidy — or
set `cell_size` directly and let `Grid.plan` report the resulting count.

| Use case | Target cell size | Rationale |
|----------|------------------|-----------|
| Interactive local viewer | 100–200 physical units | Small enough for fast partial loads |
| Cloud serving (S3/GCS) | 300–500 physical units | Fewer objects → lower request cost |
| HPC batch analysis | 500–2000 physical units | Large cells reduce file-count overhead |
| Neuroglancer (fine mesh) | 10–50 physical units | Meshes are dense; small regions needed |
| Synchrotron tractography | 50–100 mm | Typical white-matter query region |

### Anisotropic data

`cells` divides the *extent*, so a scalar `cells` on a non-cubic volume gives
non-cubic cells. Pass one count per axis to get back to roughly equal physical
extents:

```python
bounds = ((0.0, 0.0, 0.0), (2048.0, 2048.0, 512.0))
print(zv.Grid.plan(bounds, target_cells=8))
print(zv.Grid.plan(bounds, target_cells=(8, 8, 2)))
```

```text
Grid(8x8x8 cells of (256.0, 256.0, 64.0))
Grid(8x8x2 cells of (256.0, 256.0, 256.0))
```

`zv.Layout(cells=(8, 8, 2))` takes the same per-axis form. This matters for
data on anisotropic voxels — 1 µm × 1 µm × 4 µm, say — where the extent is
already lopsided in voxel terms but the physical volume is not.

---

## Subcell guidance

### Primary consideration: query granularity

`subcells` sets the finest spatial region the fragment index can name inside a
cell. A bounding-box query that overlaps part of a cell resolves to the bins
it touches rather than to the whole cell.

**Rule of thumb:** set `subcells` so that a typical query spans 2–8 bins per
axis.

```text
Typical query 50³ µm, cell size 200³ µm:
  bin ≤ 50/2 = 25 µm  and  bin ≥ 50/8 = 6.25 µm
  → subcells = 8 gives a 25 µm bin ✓
```

### Bins per cell and index overhead

The fragment index holds one 16-byte entry per bin, plus a small header and
bitmap. Measured on a 5×5×5-cell store of 100 000 points, one
`vertex_fragments` blob per cell:

| `subcells` | Bins per cell (3-D) | Index bytes per cell |
|-----------|---------------------|----------------------|
| 1 | 1 | 52 |
| 2 | 8 | 164 |
| 4 | 64 | 1 060 |
| 5 | 125 | 2 044 |
| 8 | 512 | 8 284 |
| 10 | 1 000 | 16 156 |
| 16 | 4 096 | 66 076 |

At the default `subcells=4` the index is ~1 KB per cell, negligible beside the
vertex data. Above 8 per axis it stops being negligible: at `subcells=16` the
index is 66 KB per cell, which for a store whose cells hold 20 000 vertices is
a quarter of the payload again. Go there only when the query granularity
genuinely needs it.

### One bin per cell

`subcells=1` makes the bin equal to the cell — no sub-cell indexing at all.
Reach for it when:

- All queries read entire cells.
- The store is written for sequential batch processing only.
- You need maximum compatibility with tools that do not understand bins.

```python
zv.Layout(cells=5, subcells=1)
```

`subcells` changes only how much of a cell a query has to touch. It never
changes the answer — the same bounding box returns the same 109 points at
every setting in the table above.

---

## Worked examples

### Synchrotron point cloud (HiP-CT)

```text
Dataset:     200M vertices in an 8 000³ µm HiP-CT scan
Query size:  ~100³ µm (interactive viewport at high zoom)
Platform:    S3, Neuroglancer serving
```

```python
bounds = ((0.0, 0.0, 0.0), (8000.0, 8000.0, 8000.0))
viewport = ((1000.0, 1000.0, 1000.0), (1100.0, 1100.0, 1100.0))

for cells in (8, 16, 40, 80):
    grid = zv.Grid.plan(bounds, target_cells=cells)
    cap = grid.capacity(n_vertices=200_000_000)
    print(f"cells={cells:<3} cell={grid.cell_shape[0]:>6.0f} um "
          f"{cap.cells:>7} cells {cap.est_bytes_per_cell / 1e3:>8.1f} kB/cell "
          f"{len(grid.cells_in(viewport))} read per viewport")
```

```text
cells=8   cell=  1000 um     512 cells   4687.5 kB/cell 1 read per viewport
cells=16  cell=   500 um    4096 cells    585.9 kB/cell 1 read per viewport
cells=40  cell=   200 um   64000 cells     37.5 kB/cell 1 read per viewport
cells=80  cell=   100 um  512000 cells      4.7 kB/cell 8 read per viewport
```

`cells=16` is the answer. It is the only row in the 50 KB–50 MB band:
`cells=40` looks appealing because the cell matches the viewport, but 37 kB
objects are below the floor, and `cells=80` puts half a million objects in the
bucket at 4.7 kB each. The viewport is served instead by the bins — a 500 µm
cell with `subcells=10` gives 50 µm bins, so a 100³ µm viewport resolves to a
2×2×2 corner of one cell:

```python
schema = zv.Schema(
    bounds=bounds,
    kind="point_cloud",
    expected=zv.SizeHints(n_vertices=200_000_000),
    layout=zv.Layout(cells=16, subcells=10),
)
r = schema.layout.resolve(schema, store_kind="object")
print(r.chunk_shape, r.bin_shape, r.shard_shape)
```

```text
(500.0, 500.0, 500.0) (50.0, 50.0, 50.0) (3, 3, 3)
```

`subcells=10` costs ~16 kB of index per cell against ~586 kB of vertices —
about 3%, bought deliberately to keep the viewport read small.

### DWI tractography (1M streamlines, 50-vertex average)

```text
Dataset:     1M streamlines × 50 vertices = 50M vertices
             Streamlines span a 180³ mm MRI volume
Query size:  Typically one white-matter bundle (~30 × 30 × 80 mm)
Platform:    Local analysis + Neuroglancer
```

```python
bounds = ((0.0, 0.0, 0.0), (180.0, 180.0, 180.0))
bundle = ((60.0, 60.0, 50.0), (90.0, 90.0, 130.0))

for cells in (4, 6, 12):
    grid = zv.Grid.plan(bounds, target_cells=cells)
    cap = grid.capacity(n_vertices=50_000_000)
    print(f"cells={cells:<3} cell={grid.cell_shape[0]:>5.0f} mm "
          f"{cap.est_bytes_per_cell / 1e6:>6.2f} MB/cell "
          f"{1_000_000 // cap.cells:>6} streamlines/cell "
          f"{len(grid.cells_in(bundle)):>3} read per bundle")
```

```text
cells=4   cell=   45 mm   9.38 MB/cell  15625 streamlines/cell   8 read per bundle
cells=6   cell=   30 mm   2.78 MB/cell   4629 streamlines/cell  16 read per bundle
cells=12  cell=   15 mm   0.35 MB/cell    578 streamlines/cell  54 read per bundle
```

This is the case where the two heuristics disagree. `cells=12` is the only row
that meets the 100–500-objects rule, and it makes a bundle read touch 54
cells. `cells=4` keeps the bundle to eight reads of 9 MB — comfortably inside
the 50 MB ceiling — at the cost of 15 000 streamlines per cell. Take
`cells=4`, and use `subcells` to get the granularity back:

```python
schema = zv.Schema(
    bounds=bounds,
    kind="polyline",
    expected=zv.SizeHints(n_vertices=50_000_000, n_objects=1_000_000),
    layout=zv.Layout(cells=4, subcells=5),
)
r = schema.layout.resolve(schema)
print(r.chunk_shape, r.bin_shape)
```

```text
(45.0, 45.0, 45.0) (9.0, 9.0, 9.0)
```

A 30 mm bundle width spans between 3 and 4 of those 9 mm bins per axis, inside
the 2–8 target. If the pipeline needs the round 50 mm grid rather than the
round cell count, `zv.Layout(cell_size=(50.0, 50.0, 50.0))` gives the same
4×4×4 allocation with the outermost cells overhanging the bounds:

```python
print(zv.Grid.plan(bounds, cell_size=(50.0, 50.0, 50.0)))
```

```text
Grid(4x4x4 cells of (50.0, 50.0, 50.0))
```

Note also that a bounding-box read of a polyline store returns *whole*
streamlines, so the bytes a bundle query moves are bounded by the cells it
touches, not by the box.

### EM connectome skeletons (10 000 neurons)

```text
Dataset:     10 000 skeletons, avg 5 000 nodes/neuron = 50M nodes
             Data in a 1 000³ µm EM volume
Query:       Single neuron retrieval by ID (object index)
             Spatial query by region (~100³ µm)
Platform:    Local analysis
```

```python
bounds = ((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0))
region = ((300.0, 300.0, 300.0), (400.0, 400.0, 400.0))

for cells in (5, 10, 20):
    grid = zv.Grid.plan(bounds, target_cells=cells)
    cap = grid.capacity(n_vertices=50_000_000)
    print(f"cells={cells:<3} cell={grid.cell_shape[0]:>5.0f} um "
          f"{50_000_000 // cap.cells:>7} nodes/cell "
          f"{cap.est_bytes_per_cell / 1e6:>5.2f} MB/cell "
          f"{len(grid.cells_in(region))} read per region")
```

```text
cells=5   cell=  200 um  400000 nodes/cell  4.80 MB/cell 8 read per region
cells=10  cell=  100 um   50000 nodes/cell  0.60 MB/cell 8 read per region
cells=20  cell=   50 um    6250 nodes/cell  0.07 MB/cell 27 read per region
```

`cells=10` lands squarely in the 10 000–100 000 nodes-per-cell band and keeps
the region read to eight cells; the default `subcells=4` gives a 25 µm bin,
four per axis across the query:

```python
schema = zv.Schema(
    bounds=bounds,
    kind="graph",
    expected=zv.SizeHints(n_vertices=50_000_000, n_objects=10_000),
    layout=zv.Layout(cells=10, subcells=4),
)
r = schema.layout.resolve(schema)
print(r.chunk_shape, r.bin_shape)
```

```text
(100.0, 100.0, 100.0) (25.0, 25.0, 25.0)
```

Retrieval by neuron ID does not go through the grid at all — it goes through
the object index, which is why the per-cell object count (~10 here) is not a
problem the way it would be for a region-only workload.

---

## Put the layout on the `Schema`, not on the write

`add_points` and its siblings accept a `layout=` argument, and it is a trap on
a store whose declared grid says something else. The write honours the layout
you pass; the store's metadata keeps the grid it was created with; and every
bounding-box query afterwards looks in the wrong place and finds nothing:

```python
import numpy as np

rng = np.random.default_rng(42)
positions = rng.uniform(0, 1000, size=(100_000, 3)).astype(np.float32)

ds = zv.create("bad.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),   # Layout() -> one cell
))
ds.add_points(positions, layout=zv.Layout(cells=5))       # writes a 5x5x5 grid
print(ds.level(0).grid)
print(ds.read().vertex_count)
print(ds.select(bbox=((100.0, 100.0, 100.0), (200.0, 200.0, 200.0))).count())
```

```text
Grid(1x1x1 cells of (1000.0, 1000.0, 1000.0))
100000
0
```

The data is all there — the whole-store read returns 100 000 — but the
declared grid contradicts the chunk keys, and nothing raises. Decide the
layout once, on the `Schema` you pass to `zv.create`. Use `layout=` on a write
only when it resolves to the grid the store already has.

---

## After writing: profile and tune

The distribution that matters is vertices per cell, and a uniform estimate is
only ever an estimate — real data clusters. `Level.grid` gives you the grid
the store actually has, in physical units, so the profile is a few lines:

```python
import numpy as np

level = zv.open("scan.zarrvectors").level(0)
grid = level.grid
xyz = level.read().positions

ijk = np.floor(
    (xyz - np.asarray(grid.origin)) / np.asarray(grid.cell_shape)
).astype(int)
_, counts = np.unique(ijk, axis=0, return_counts=True)

print(grid, f"{len(counts)}/{grid.cells} occupied")
print(f"min={counts.min()} median={int(np.median(counts))} "
      f"p95={int(np.percentile(counts, 95))} max={counts.max()}")
```

```text
Grid(5x5x5 cells of (200.0, 200.0, 200.0)) 125/125 occupied
min=746 median=799 p95=848 max=864
```

(That is the quickstart's 100 000 uniformly-distributed points, so the spread
is narrow by construction. Real data will not look like this.)

Read it as:

- `median` far below 10 000 — the cells are too small for the data density;
  lower `cells`.
- `p95` far above 100 000 — the cells are too large for interactive use;
  raise `cells`.
- Many cells unoccupied — the data does not fill its bounds. That is fine in
  itself; empty cells cost nothing on disk. But it means the uniform estimate
  from `Grid.capacity` was optimistic, and the occupied cells are proportionally
  fuller than it predicted.

The `zarr-vectors info` CLI in the companion package **`zarr-vectors-tools`**
prints this summary directly, without the reshaping above.

---

## See also

- [Quickstart](../getting_started/quickstart.md) — `Layout` in the context of
  a whole session.
- [Core concepts](../getting_started/concepts.md) — what chunks, bins and
  fragments are.
- [Memory-efficient writes](memory_efficient_writes.md) — why a large cell
  costs write-time RAM as well as read-time bandwidth.
