# Choosing a layout: cells, cell size and subcells

This page shows how to choose the `zv.Layout` for a store, check the choice
with `zv.Grid` before writing anything, and profile the result afterwards.
[Core concepts](../explanation/concepts.md) explains what cells (chunks),
bins and fragments are.

## What you need

- The bounds of the data.
- A rough vertex count.
- The size of a typical spatial query.

## Step 1: describe the data on the `Schema`

Attach the layout to the `Schema` you pass to `zv.create`. `cells=N` cuts the
volume into `N` cell widths per axis:

```python
import zarr_vectors as zv

# 10 million points in a 4 000 µm cube
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

`Layout.resolve` turns the layout and `Schema.bounds` into the cell size
(`chunk_shape`), bin size (`bin_shape`), shard shape and compressor. It raises
`MetadataError` if the schema has no `bounds`. Once a store exists, the cell
size is `Level.scale`, the bin size is `Level.resolution` and the grid is
`Level.grid`.

Every field has a working `"auto"`:

| Field | What it sets | `"auto"` (default) |
|-------|--------------|--------------------|
| `cells` | Cells per axis | From `Schema.expected.n_vertices`, aiming at 64 MiB of coordinates per cell; 1 per axis with no hint |
| `cell_size` | Cell size in coordinate units; wins over `cells` | `None` |
| `subcells` | Bins per axis inside each cell | 4 |
| `pack` | Whether cells are packed into shards | On for object stores, off for local paths |
| `compression` | Compressor name | `$ZARR_VECTORS_COMPRESSION`, else `zstd` |

With no `expected` hint and no `cells`, the store gets one cell per axis, so
set one of them for anything you will query spatially.

Use `cell_size` when the grid is fixed from outside, for example to match an
image volume's tiles. When the extent does not divide exactly, `cell_size`
keeps the size and the grid overhangs the bounds, while `cells` keeps the count
and adjusts the size.

## Step 2: check the cell size

`zv.Grid.plan` builds the grid a store with these bounds would get, without
creating one. `capacity()` estimates bytes per cell, and `cells_in()` counts
the cells a query box touches. Compare a few candidates against your typical
query:

```python
bounds = ((0.0, 0.0, 0.0), (8000.0, 8000.0, 8000.0))
query = ((1000.0, 1000.0, 1000.0), (1100.0, 1100.0, 1100.0))

for cells in (8, 16, 40, 80):
    grid = zv.Grid.plan(bounds, target_cells=cells)
    cap = grid.capacity(n_vertices=200_000_000)
    print(f"cells={cells:<3} cell={grid.cell_shape[0]:>6.0f} um "
          f"{cap.cells:>7} cells {cap.est_bytes_per_cell / 1e3:>8.1f} kB/cell "
          f"{len(grid.cells_in(query))} read per query")
```

```text
cells=8   cell=  1000 um     729 cells   3292.2 kB/cell 1 read per query
cells=16  cell=   500 um    4913 cells    488.5 kB/cell 1 read per query
cells=40  cell=   200 um   68921 cells     34.8 kB/cell 1 read per query
cells=80  cell=   100 um  531441 cells      4.5 kB/cell 8 read per query
```

Each cell is one storage object unless cells are packed into shards. Weigh two
costs against each other:

- Large cells make a small query read far more data than it needs.
  `capacity()` reports `fits=False` above its 64 MiB target (printed as 67 MB).
- Small cells mean many objects, and every object read or written has a fixed
  cost. Nothing warns about cells that are too small; read
  `est_bytes_per_cell` and the cell count yourself.

Pick the coarsest grid whose cells a typical query can still afford to read
whole, and let the bins (step 3) narrow reads inside a cell.

Two limits of the estimate:

- It counts coordinates only, `n_vertices × ndim × 4` bytes. Five float32
  attributes beside three float32 coordinates make the real figure 8/3 of it.
- It divides by every allocated cell. `target_cells=8` allocates 9 × 9 × 9,
  because a vertex on the upper bound needs a cell, so data inside the bounds
  fills about 8³ of the 9³ cells and each occupied cell holds more than the
  estimate. Clustered data is fuller still.

### Anisotropic data

`cells` divides the extent, so a scalar `cells` on a non-cubic volume gives
non-cubic cells. Pass one count per axis to keep cells roughly cubic:

```python
bounds = ((0.0, 0.0, 0.0), (2048.0, 2048.0, 512.0))
print(zv.Grid.plan(bounds, target_cells=8))
print(zv.Grid.plan(bounds, target_cells=(8, 8, 2)))
```

```text
Grid(9x9x9 cells of (256.0, 256.0, 64.0))
Grid(9x9x3 cells of (256.0, 256.0, 256.0))
```

`zv.Layout(cells=(8, 8, 2))` takes the same per-axis form.

## Step 3: choose the subcells

`subcells` sets how many bins per axis each cell is divided into, which is the
finest region the fragment index can name. A query that overlaps part of a cell
resolves to the bins it touches rather than to the whole cell. `bin_shape`
must divide `chunk_shape`, which is why `Layout` takes a count rather than a
size:

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

More bins cost index space. Each cell's fragment index has a 16-byte header, a
bitmap of one bit per bin padded to 8 bytes, and 16 bytes per bin, plus offsets
for any explicitly listed fragments. Measured on a store of 100 000 points in
125 cells, one `vertex_fragments` blob per cell, before compression:

| `subcells` | Bins per cell (3-D) | Index bytes per cell |
|-----------|---------------------|----------------------|
| 1 | 1 | 44 |
| 2 | 8 | 156 |
| 4 | 64 | 1 052 |
| 5 | 125 | 2 036 |
| 8 | 512 | 8 276 |
| 10 | 1 000 | 16 148 |
| 16 | 4 096 | 66 068 |

The index grows with the cube of `subcells` while the vertex data does not, so
choose bins about as small as the typical query, and no smaller.
`subcells=1` makes the bin equal to the cell, for stores that are always read
whole cells at a time.

`subcells` changes how much of a cell a query touches, never the answer: on
that store the same bounding box returned the same 109 points at every setting
in the table.

## Step 4: leave packing to `resolve`

On a local filesystem one object per cell is cheap. Against an object store
each object is a request, so `pack="auto"` packs cells into shards of about
16 MiB (`Layout(target_object_bytes=...)` changes the target). `zv.create`
passes the store kind from the URL, so you normally leave `pack` alone:

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

The shard size comes from the `expected` hint; without one, `resolve` uses 4
cells per axis per shard. A wrong hint costs performance, not correctness.

```{warning}
Set the layout on the `Schema`, not on a write. `add_points(layout=...)` on a
store whose declared grid differs writes cells on the passed grid while the
metadata keeps the declared one. Whole-store reads still return every vertex,
but bounding-box queries return nothing, and no error is raised.
```

## Step 5: profile the written store

Real data clusters, so check vertices per cell once the store exists.
`Level.grid` gives the grid the store has, in coordinate units:

```python
import numpy as np
import zarr_vectors as zv

rng = np.random.default_rng(42)
ds = zv.create("scan.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    kind="point_cloud",
    layout=zv.Layout(cells=5),
))
ds.add_points(rng.uniform(0, 1000, size=(100_000, 3)).astype(np.float32))

level = zv.open("scan.zarrvectors").level(0)
grid = level.grid
xyz = level.read().positions

# Cells are addressed absolutely: floor(position / cell size).
ijk = np.floor(xyz / np.asarray(grid.cell_shape)).astype(int)
_, counts = np.unique(ijk, axis=0, return_counts=True)

print(grid, f"{len(counts)}/{grid.cells} occupied")
print(f"min={counts.min()} median={int(np.median(counts))} "
      f"p95={int(np.percentile(counts, 95))} max={counts.max()}")
```

```text
Grid(6x6x6 cells of (200.0, 200.0, 200.0)) 125/216 occupied
min=746 median=799 p95=848 max=864
```

These points are uniform, so the spread is narrow. On real data, a high `p95`
or `max` points to cells that are too large where the data is dense; a low
median, to cells that are too small. Empty cells cost nothing on disk.

## See also

- [Quickstart](../getting_started/quickstart.md): `Layout` in a whole session.
- [Memory-efficient writes](memory_efficient_writes.md): sizing cells for a
  write that does not fit in memory.
