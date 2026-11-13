# Memory-efficient writes

This page shows how to write a point set that does not fit in memory: you size
the grid, allocate the store once, write it one partition of whole cells at a
time, then finalise it. Peak memory is one partition, whatever the size of the
dataset.

A cell is written whole: every vertex in it, with its attributes, must be in
memory when the cell is encoded. So the unit you stream is the cell, not the
vertex.

`Dataset.add_points` is not an append. A second call into the same store
replaces what the first wrote (or raises `StoreError` if its batch reaches
cells the first did not allocate), so looping it over slabs keeps only the last
slab. Use it when the whole batch fits in memory; use `zarr_vectors.building`,
as below, when it does not.

## What you need

- The bounds of the whole dataset before you write anything.
- A way to load one spatial slab of the input at a time. The examples use a
  synthetic `load_slab`:

```python
import numpy as np

rng = np.random.default_rng(0)

def load_slab(z_lo, z_hi, n=4000):
    """Synthetic stand-in for your reader: points with z_lo <= z < z_hi."""
    p = rng.uniform(0., 1000., size=(n, 3)).astype("float32")
    p[:, 2] = rng.uniform(z_lo, z_hi, size=n).astype("float32")
    p[:, 2] = np.minimum(p[:, 2], np.nextafter(np.float32(z_hi), np.float32(0)))
    return p, rng.uniform(size=n).astype("float32")
```

## Step 1: size the grid

`Grid.plan` computes the grid a store with these bounds would get, without
creating one, and `capacity()` turns a vertex count into bytes per cell. Do
this before writing: changing the cell size afterwards means rewriting the
store.

```python
import zarr_vectors as zv

grid = zv.Grid.plan(bounds=((0., 0., 0.), (10_000., 10_000., 10_000.)),
                    cell_size=(500., 500., 500.))
print(grid, grid.cells)
print(grid.capacity(n_vertices=500_000_000))

# The same 500M vertices in a coarse grid:
print(zv.Grid.plan(bounds=((0., 0., 0.), (10_000., 10_000., 10_000.)),
                   target_cells=2).capacity(n_vertices=500_000_000))
```

```text
Grid(21x21x21 cells of (500.0, 500.0, 500.0)) 9261
21x21x21 = 9261 cells, ~0.6 MB/cell -- fits
3x3x3 = 27 cells, ~222.2 MB/cell -- does not fit: ~222 MB per cell exceeds the 67 MB target; use more cells
```

The estimate counts coordinates only (`n_vertices / cells × ndim × 4`
bytes) and spreads them over every allocated cell, including the extra row of
cells a bound on a cell boundary adds. Add your per-vertex attributes by hand:
five float32 attributes beside three float32 coordinates make the real figure
8/3 of the estimate. [Choosing a layout](choose_chunk_and_bin.md) covers the choice of
cell size.

## Step 2: allocate the store once

```python
from zarr_vectors.building import (
    LevelMetadata, create_store, create_resolution_level,
    create_vertices_array, create_attribute_array, defer_presence,
    open_write_session,
)

STORE  = "large_scan.zarrvectors"
BOUNDS = ([0., 0., 0.], [1000., 1000., 1000.])
CHUNK  = (250., 250., 250.)
BIN    = (62.5, 62.5, 62.5)
BINS_PER_CHUNK = tuple(int(round(c / b)) for c, b in zip(CHUNK, BIN))
N_BINS = int(np.prod(BINS_PER_CHUNK))

root = create_store(STORE, bounds=BOUNDS, chunk_shape=CHUNK,
                    base_bin_shape=BIN, geometry_types=["point_cloud"])
level = create_resolution_level(root, 0, LevelMetadata(
    level=0, vertex_count=0, arrays_present=["vertices", "vertex_attributes"]))

with open_write_session(level, bounds=BOUNDS, chunk_shape=CHUNK):
    create_vertices_array(level, dtype="float32")
    create_attribute_array(level, "intensity", dtype="float32")

defer_presence(level)
```

`open_write_session` requires `bounds` and `chunk_shape`: they set the extent
and origin of the cell grid each array is sized to. Every partition must pass
the same two values.

`defer_presence` declares that no write on this level will update the presence
manifest, `nonempty_chunks`. The manifest is one attribute shared by every cell
of an array, so partitions that each update it race one another. While the
level is deferred, zarr-vectors readers find the cells by listing the store;
[Stamping under your own lock](hpc_pipelines.md#stamping-under-your-own-lock)
covers readers that cannot.

## Step 3: write one partition at a time

Each partition loads its slab, writes the cells it covers, and frees the slab:

```python
from zarr_vectors.building import (
    open_store, get_resolution_level, write_chunk_vertices,
    write_chunk_attributes,
)
# Internal module: building has no bin-level equivalent (see below).
from zarr_vectors.spatial.chunking import assign_bins, group_bins_by_chunk

written = 0
for slab in range(4):                      # 4 z-slabs, one cell deep each
    positions, intensity = load_slab(slab * 250., (slab + 1) * 250.)

    level = get_resolution_level(open_store(STORE, mode="r+"), 0)
    with open_write_session(level, bounds=BOUNDS, chunk_shape=CHUNK):
        per_chunk = group_bins_by_chunk(assign_bins(positions, BIN),
                                        BINS_PER_CHUNK)
        for cell, fragments in per_chunk.items():
            verts = [np.zeros((0, 3), "float32") for _ in range(N_BINS)]
            attrs = [np.zeros((0,), "float32") for _ in range(N_BINS)]
            for fragment_index, rows in fragments.items():
                verts[fragment_index] = positions[rows]
                attrs[fragment_index] = intensity[rows]

            write_chunk_vertices(level, cell, verts, dtype="float32",
                                 record_presence=False)
            write_chunk_attributes(level, "intensity", cell, attrs,
                                   dtype="float32", record_presence=False)
            written += sum(len(v) for v in verts)

    del positions, intensity
```

`record_presence=False` repeats the deferral per call. With `defer_presence` it
is redundant; without it, it is what keeps the partitions from racing.

Write one fragment per bin, as above, whenever the store declares
`base_bin_shape`. Readers use the fragment index to pick the bins a bounding
box touches, so a cell written as a single fragment returns wrong results:
on a test store with 4 × 4 × 4 bins per cell, a bounding-box query returned 73
of the 674 points inside the box, and `validate()` only warns. `assign_bins`
and `group_bins_by_chunk` come from `zarr_vectors.spatial.chunking`, which is
internal; `zarr_vectors.building.assign_chunks` maps vertices to cells only,
which is correct only when each cell is one bin.

Partitions do not have to be slabs. Any set of whole cells works, as long as no
two partitions share a cell. Peak memory is roughly

```text
cells_per_partition × vertices_per_cell × bytes_per_vertex
```

plus the encode buffers. For example, 8 cells of 50 000 vertices at 32 bytes
each (three float32 coordinates and five float32 attributes) is 12.8 MB of
payload. Smaller partitions cost round trips rather than correctness: each one
reopens the store and opens its own write session.

## Step 4: finalise

After every partition has finished, one process writes the manifests and the
level metadata:

```python
from zarr_vectors.building import (
    rebuild_presence, refresh_arrays_present, update_level_metadata,
    write_multiscale_metadata,
)

root = open_store(STORE, mode="r+")
level = get_resolution_level(root, 0)

rebuild_presence(level)                    # every manifest; ends the deferral
print(refresh_arrays_present(level))       # the arrays actually on disk
update_level_metadata(level, vertex_count=written)
write_multiscale_metadata(root)

ds = zv.open(STORE)
print(ds.level(0).vertex_count, ds.level(0).scale, ds.level(0).resolution)
print(ds.read())
```

```text
['vertex_attributes', 'vertex_fragments', 'vertices']
16000 (250.0, 250.0, 250.0) (62.5, 62.5, 62.5)
ReadResult(kind='point_cloud', vertices=16000, attributes=['intensity'])
```

`refresh_arrays_present` lists the arrays found on disk, so `vertex_fragments`
appears although `LevelMetadata` did not name it. Never run either call while
partitions are still writing. Writers with no finalising process use
`collect_presence` and `apply_presence` instead; see
[Stamping under your own lock](hpc_pipelines.md#stamping-under-your-own-lock).

Build coarser levels from the finished store:

```python
report = zv.open(STORE, mode="r+").build_pyramid(
    factors=[(2., 1.), (2., 1.)], chunk_scale_factors=[2, 2])
print(report["levels_created"], [s["vertex_count"] for s in report["level_specs"]])
```

```text
2 [512, 64]
```

## Change the cell size of an existing store

If the data already fits in memory and only the cell size is wrong, write it
with one cell and re-cut the grid with `rechunk`. This is the least
memory-efficient route: the first write holds the whole dataset.

```python
import numpy as np
import zarr_vectors as zv
from zarr_vectors.building import RechunkSpec, rechunk

rng = np.random.default_rng(0)
positions = rng.uniform(0., 1000., size=(20_000, 3)).astype("float32")
object_ids = np.repeat(np.arange(200), 100)

temp = zv.create("scan_temp.zarrvectors", schema=zv.Schema(
    bounds=((0., 0., 0.), (1000., 1000., 1000.)),
    kind="point_cloud",
    layout=zv.Layout(cells=1),
))
temp.add_points(positions, object_ids=object_ids)
print(zv.open("scan_temp.zarrvectors").level(0).grid)

summary = rechunk(
    "scan_temp.zarrvectors",
    RechunkSpec(by="spatial", spatial_chunk_shape=(250., 250., 250.)),
    output="scan.zarrvectors",
)
print(summary)
print(zv.open("scan.zarrvectors").level(0).grid)
```

```text
Grid(2x2x2 cells of (1000.0, 1000.0, 1000.0))
{'objects_rechunked': 200, 'bins_created': 1, 'total_vertices': 20000, 'rechunk_dims': ['spatial', 'x', 'y', 'z'], 'output_path': 'scan.zarrvectors'}
Grid(5x5x5 cells of (250.0, 250.0, 250.0))
```

`rechunk` copies level 0's vertices and object index only: links, attributes
and coarser levels are not carried over, so use it for point clouds. Chunk keys
in the output gain a leading axis (`c/0/1/3/3` rather than `c/1/3/3`), named in
`rechunk_dims`. With `output=None` it rewrites the store in place. `RechunkSpec`
also groups by a non-spatial value (`by="group"`, `by="object_id"` or
`by="attribute:<name>"`), and `rechunk_by_attribute(store, name)` gives every
distinct value its own bin. Rechunking rewrites the whole store: never run it
while anything else is writing to it.

## Measure memory use

`tracemalloc` measures the write path:

```python
import tracemalloc
import numpy as np
import zarr_vectors as zv

tracemalloc.start()

probe = zv.create("probe.zarrvectors", schema=zv.Schema(
    bounds=((0., 0., 0.), (1000., 1000., 1000.)),
    kind="point_cloud", layout=zv.Layout(cells=4)))
probe.add_points(np.random.default_rng(1)
                 .uniform(0., 1000., size=(100_000, 3)).astype("float32"))

current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory: {peak / 1e6:.1f} MB")
tracemalloc.stop()
```

```text
Peak memory: 7.3 MB
```

`tracemalloc` sees Python allocations, including NumPy buffers, but not a
compressor's native scratch space. Treat the figure as a lower bound.

## Next steps

- [HPC pipelines](hpc_pipelines.md): the same partitioned write across SLURM
  array tasks or MPI ranks.
- [Choosing a layout](choose_chunk_and_bin.md): how to pick the cell size.
- {doc}`Building stores <../api/building>`: reference for the surface used
  here.
- Readers for formats such as LAS, TRX and SWC, which feed the same per-cell
  writes, are in the companion package zarr-vectors-tools.
