# Memory-efficient writes

A chunk is written whole. Every vertex that lands in one chunk — positions
and per-vertex attributes together — must be in memory at the moment that
chunk is encoded, because its fragment index describes rows of one packed
buffer. For a dataset that does not fit in RAM, the question is therefore not
*"how do I stream vertices?"* but *"how do I write one chunk at a time, and
how big is a chunk?"*

Two facts shape everything below:

- **`Dataset.add_points` is a one-shot write, not an append.** A second call
  into the same store re-derives the grid from its own batch and overwrites
  what was there. Looping `add_points` over slabs does not accumulate — it
  leaves you with the last slab. See
  [Quickstart](../getting_started/quickstart.md#writing-points).
- **Chunk size is a decision you make before the store exists.** Once
  written, changing it means rewriting the store.

So a partitioned write uses `zarr_vectors.building` — the supported surface
for code that *constructs* stores, whose per-chunk verbs are exactly "the
per-chunk verbs an ingest worker runs in parallel". `zarr_vectors.api`
(`zv.create` / `add_points`) is the right surface when the batch fits in
memory; it does not have a partitioned writer.

---

## Step 0: size the grid before writing anything

`Grid.plan` computes the grid a store with those bounds *would* get, without
creating one, and `capacity()` turns a vertex count into bytes per cell. This
is the only moment the answer can still change a decision — a pipeline that
discovers its chunks are 750 MB after the fact has already written a store it
must throw away.

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
Grid(20x20x20 cells of (500.0, 500.0, 500.0)) 8000
20x20x20 = 8000 cells, ~0.8 MB/cell -- fits
2x2x2 = 8 cells, ~750.0 MB/cell -- does not fit: ~750 MB per cell exceeds the 67 MB target; use more cells
```

`est_bytes_per_cell` counts coordinates only (`n_vertices / cells × ndim ×
4`). Add your per-vertex attributes to it by hand: five float32 attributes
alongside three float32 coordinates makes the real figure ⅔ larger again.
Peak memory during the write is roughly one *partition* — which is one or
more whole cells — not one cell, so size partitions with that headroom.

See [Choosing a layout](choose_chunk_and_bin.md) for what
else pulls the number around.

---

## Strategy 1: write in spatial partitions

The general shape of a bounded-memory write: a coordinator allocates the
store's arrays once, then each partition is loaded, written cell by cell, and
freed. Peak memory is one partition, whatever the size of the dataset.

### The coordinator allocates

```python
import numpy as np
import zarr_vectors as zv
from zarr_vectors.building import (
    LevelMetadata, create_store, create_resolution_level,
    create_vertices_array, create_attribute_array, open_write_session,
    open_store, get_resolution_level, write_chunk_vertices,
    write_chunk_attributes, rebuild_presence, refresh_arrays_present,
    update_level_metadata, write_multiscale_metadata,
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
```

```{warning}
**Allocate exactly once, in the coordinator.** `create_vertices_array` and
`create_attribute_array` rewrite the array's `zarr.json`, and inside a write
session that drops every cell already on disk. Calling them again from each
partition looks harmless and silently reduces a four-slab write to its last
slab. Workers write cells; they never create arrays.
```

`bounds` and `chunk_shape` are required by `open_write_session` because the
chunk grid's extent — and its origin, for data with negative coordinates — is
the shape of each vlen array, and has to be sized up front. Every partition
must pass the same two values as the coordinator did, or their grids disagree.

### Each partition writes its own cells

```python
# NOTE: internal import -- see the admonition below.
from zarr_vectors.spatial.chunking import assign_bins, group_bins_by_chunk

rng = np.random.default_rng(0)
written = 0

for slab in range(4):                      # 4 z-slabs, one chunk deep each
    z_lo, z_hi = slab * 250., (slab + 1) * 250.
    positions, intensity = load_slab(z_lo, z_hi)   # only this slab in memory

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

`record_presence=False` is not an optimisation. `nonempty_chunks` is a single
attribute shared by every cell of an array, so a partition that stamps it
races every other partition. Partitions skip it; the coordinator rebuilds it
once, below.

### One fragment per bin, not one per cell

The `verts` list above has one entry per *bin*, most of them empty. That is
deliberate. A fragment is the unit a coarsener picks representatives from, so
collapsing a cell into a single fragment — `write_chunk_vertices(level, cell,
[positions_in_cell])`, which is the tempting one-liner — throws away the bin
structure that `build_pyramid` later needs. The same 16 000 vertices in the
same 4×4×4 grid:

| fragments per cell | level 1 | level 2 |
|--------------------|---------|---------|
| 64 (one per bin)   | 512     | 64      |
| 1 (whole cell)     | 8       | 1       |

```{note}
`assign_bins` and `group_bins_by_chunk` are imported from
`zarr_vectors.spatial.chunking`, which is **internal** — the import above
reaches past the contract, and may break between releases.
`zarr_vectors.building` exports `assign_chunks` (vertices → cells) but no
bin-level equivalent. That is a gap to report rather than a reason to treat
`spatial` as public. If you do not need per-bin fragments, use
`building.assign_chunks` and pass one fragment per cell, accepting the
coarsening cost above.
```

### The coordinator finalises

```python
root = open_store(STORE, mode="r+")
level = get_resolution_level(root, 0)

rebuild_presence(level)                    # the nonempty_chunks nobody stamped
print(refresh_arrays_present(level))       # what is actually on disk
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

`refresh_arrays_present` walks the level rather than trusting the list the
writers declared, which is why `vertex_fragments` appears even though the
`LevelMetadata` above never mentioned it. Both verbs are coordinator-only:
never run either while partitions are still writing.

Coarser levels come afterwards, from the finished store:

```python
report = zv.open(STORE, mode="r+").build_pyramid(
    factors=[(2., 1.), (2., 1.)], chunk_scale_factors=[2, 2])
print(report["levels_created"], [s["vertex_count"] for s in report["level_specs"]])
```

```text
2 [512, 64]
```

---

## Strategy 2: bound peak memory by partition size

There is no streaming-writer class. The bound on peak memory is the partition,
and you set it by choosing how much of the volume each iteration loads:

```text
peak ≈ cells_per_partition × vertices_per_cell × bytes_per_vertex
```

At 8 cells per partition, 50 000 vertices per cell and 32 bytes per vertex
(three float32 coordinates plus five float32 attributes) that is about 13 MB
of payload — call it double, for the encode buffer the write allocates
alongside it. Partitions do not have to be slabs; any set of whole cells
works, as long as no two partitions share one. Chunk-aligned partitions are
the cheapest, because no cell is ever written twice.

Smaller partitions cost round-trips rather than correctness: each one reopens
the store and opens its own write session. Against a high-latency object
store that matters, and a partition covering a few hundred cells amortises it.

---

## Strategy 3: generator-based ingest

For formats read by external libraries (LAS, TRX, SWC, …), the companion
package **`zarr-vectors-tools`** provides streaming converters that yield
vertex batches without loading the whole file. They drive the same
`zarr_vectors.building` per-chunk verbs shown in Strategy 1, so the memory
argument above applies to them unchanged.

---

## Strategy 4: rechunk after the initial write

If your data is already in a format that can be read block by block, the
smallest change to an existing pipeline is to write it once with a coarse
chunk shape and re-cut the grid afterwards. `rechunk` with
`RechunkSpec(by="spatial")` changes the spatial chunk shape and keeps the
store's topology, levels and object ids:

```python
import numpy as np
import zarr_vectors as zv
from zarr_vectors.building import RechunkSpec, rechunk

# Initial write: one chunk for the whole volume.
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
Grid(1x1x1 cells of (1000.0, 1000.0, 1000.0))
{'objects_rechunked': 200, 'bins_created': 1, 'total_vertices': 20000, 'rechunk_dims': ['spatial', 'x', 'y', 'z'], 'output_path': 'scan.zarrvectors'}
Grid(4x4x4 cells of (250.0, 250.0, 250.0))
```

This is the *least* memory-efficient route, and it is worth being explicit
about why: the initial single-chunk write holds the entire dataset in memory
at once, which is the problem the rest of this page exists to avoid. Reach
for it when the dataset already fits and the chunk shape is merely wrong, not
when it does not fit at all.

Two things to know about the result:

- **Rechunked stores gain a prefix dimension.** Chunk keys become
  `(bin, z, y, x)` — `c/0/2/0/0` rather than `c/2/0/0` — because `rechunk`'s
  output layout is shared with the non-spatial forms below. `rechunk_dims`
  in the summary names the axes.
- **`output=None` rechunks in place**, by writing a temporary store and
  replacing the source. Leave `output` set while you still want the original.

`RechunkSpec` also rechunks along a non-spatial dimension — `by="group"`,
`by="object_id"`, or `by="attribute:<name>"` — so that all objects sharing a
value land in one chunk. `building.rechunk_by_attribute(store, "cell_type")` is the
categorical shorthand: every distinct value gets its own bin, however many
there are.

```{warning}
Rechunking rewrites where every byte lives, so it is a coordinator
operation. Never run it while anything else is writing to the store.
```

---

## Monitoring memory usage

`tracemalloc` measures the write path directly:

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
Peak memory: 8.0 MB
```

`tracemalloc` sees Python allocations only. NumPy's own buffers are counted,
but a compressor's native scratch space is not, so treat the figure as a
lower bound and leave headroom.

---

## Next steps

- **[HPC pipelines](hpc_pipelines.md)** — the same partitioned write, run
  across SLURM array tasks or MPI ranks instead of a `for` loop.
- **[Choosing a layout](choose_chunk_and_bin.md)** — how to
  pick the numbers `Grid.plan` takes.
- **{doc}`Building stores <../api/building>`** — the full reference for the
  surface Strategy 1 uses.
