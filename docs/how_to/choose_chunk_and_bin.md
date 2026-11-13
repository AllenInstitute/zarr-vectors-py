# Choosing a layout: cells, cell size and subcells

This page shows how to choose the `zv.Layout` for a store and check it before
and after writing. [Core concepts](../explanation/concepts.md) explains cells
(chunks), bins and fragments.

## What you need

- The bounds of the data.
- A rough vertex count.
- The size of a typical spatial query.

## Step 1: set the layout on the `Schema`

`cells=N` cuts the volume into `N` cells per axis:

```python
import zarr_vectors as zv

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

Every field has a working `"auto"`:

| Field | What it sets | `"auto"` (default) |
|-------|--------------|--------------------|
| `cells` | Cells per axis; one number or one per axis | From `Schema.expected.n_vertices`, aiming at 64 MiB of coordinates per cell; 1 per axis with no hint |
| `cell_size` | Cell size in coordinate units; wins over `cells` | `None` |
| `subcells` | Bins per axis inside each cell | 4 |
| `pack` | Whether cells are packed into shards | On for object stores, off for local paths |
| `compression` | Compressor name | `$ZARR_VECTORS_COMPRESSION`, else `zstd` |

With no `expected` hint and no `cells`, the store gets one cell per axis, so
set one of them. Leave `pack` alone; see [Cloud stores](cloud_stores.md#packing).
Use `cell_size` when the grid must match something else,
such as an image volume's tiles. On a non-cubic volume, pass one count per
axis, such as `cells=(8, 8, 2)`, to keep cells roughly cubic.

Set the layout on the `Schema`, not on a write. If `add_points(layout=...)`
passes a different grid, bounding-box queries return nothing, with no error.

## Step 2: check the cell size

`zv.Grid.plan` builds the grid without creating a store. `capacity()`
estimates bytes per cell, and `cells_in()` counts the cells a query box
touches:

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

Large cells make a small query read more than it needs; `capacity()` reports
`fits=False` above 64 MiB. Small cells mean many objects, each with a fixed
cost. Pick the coarsest grid whose cells
a typical query can afford to read whole.

The estimate counts coordinates only (`n_vertices × ndim × 4` bytes). Add
your attributes: five float32 attributes beside three float32 coordinates
make the real figure 8/3 of it.

## Step 3: choose the subcells

A query that covers part of a cell reads only the bins it touches. The answer
is the same at every `subcells` setting; only the amount read changes.

The fragment index grows with the cube of `subcells`: about 1 kB per cell at
4, and 8 kB at 8. Choose bins about as small as a typical query, and no
smaller. `subcells=1` makes the bin equal to the cell.

## Step 4: profile the written store

Real data clusters, so check vertices per cell once the store exists:

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

A high `p95` or `max` means cells are too large where the data is dense. A
low median means they are too small. Empty cells cost nothing on disk.

## See also

- [Memory-efficient writes](memory_efficient_writes.md): sizing cells for a
  write that does not fit in memory.
