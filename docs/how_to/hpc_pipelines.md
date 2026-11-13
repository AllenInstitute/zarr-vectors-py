# HPC pipelines

This guide shows how to write one store from many processes, such as SLURM
array tasks or MPI ranks, and how to divide read work between ranks.

Every parallel write has three phases:

1. One process creates the store and allocates its arrays.
2. Workers write disjoint sets of grid cells without touching the shared
   presence manifest.
3. One process writes the manifest and the level metadata.

All three use `zarr_vectors.building`. `Dataset.add_points` is a one-shot
write, so calling it from each task keeps only one task's data.
[Memory-efficient writes](memory_efficient_writes.md) walks through the same
pattern in one process and explains each call.

## Plan the grid before you submit

The grid is fixed when the store is created and decides how many cells there
are to divide among tasks. `Grid.plan` computes it without creating anything:

```python
import zarr_vectors as zv

BOUNDS = ((0., 0., 0.), (10_000., 10_000., 10_000.))
grid = zv.Grid.plan(bounds=BOUNDS, cell_size=(500., 500., 500.))
print(grid, grid.cells)
print(grid.capacity(n_vertices=500_000_000))
```

```text
Grid(21x21x21 cells of (500.0, 500.0, 500.0)) 9261
21x21x21 = 9261 cells, ~0.6 MB/cell -- fits
```

The cell count sizes the task range; the bytes per cell, times the cells a
task holds, sizes its memory request.

### Partitions must be disjoint

Two tasks writing the same cell is a lost update: the second writer wins and
nothing detects it. Enumerate the grid and slice it, which is disjoint by
construction:

```python
task_id, n_tasks = 3, 16                          # e.g. from the scheduler
cells = sorted(grid, key=lambda c: c.coords)      # every CellRef, ordered
partition = zv.CellSet(cells[task_id::n_tasks])   # this task's cells
```

```{warning}
Do not cut partitions with `Grid.cells_in`. It includes cells on the upper
boundary of the box, so slabs cut on cell boundaries share a plane of cells.
For a 1 000 µm cube in 250 µm cells, four z-slabs return 50 cells each, 200 in
total, and adjacent slabs share 25. That is right for a read and wrong for a
parallel write.
```

If you split by coordinates instead, give each task a half-open range
(`z_lo <= z < z_hi`).

## SLURM job array

### `init_store.py`: runs once, before the array

```python
from zarr_vectors.building import (
    LevelMetadata, create_store, create_resolution_level,
    create_vertices_array, create_attribute_array, defer_presence,
    open_write_session,
)

STORE  = "/scratch/scan.zarrvectors"
BOUNDS = ([0., 0., 0.], [1000., 1000., 1000.])
CHUNK  = (250., 250., 250.)
BIN    = (125., 125., 125.)

root = create_store(STORE, bounds=BOUNDS, chunk_shape=CHUNK,
                    base_bin_shape=BIN, geometry_types=["point_cloud"])
level = create_resolution_level(root, 0, LevelMetadata(
    level=0, vertex_count=0,
    arrays_present=["vertices", "vertex_attributes"]))

with open_write_session(level, bounds=BOUNDS, chunk_shape=CHUNK):
    create_vertices_array(level, dtype="float32")
    create_attribute_array(level, "intensity", dtype="float32")

defer_presence(level)        # no task writes nonempty_chunks
```

### Submit script

```bash
#!/bin/bash
#SBATCH --array=0-19
#SBATCH --ntasks=1

python write_partition.py --slab-index ${SLURM_ARRAY_TASK_ID} --n-slabs 20 \
    --output /scratch/scan.zarrvectors
```

### `write_partition.py`

`load_slab` stands for your reader; it must return the points with
`z_lo <= z < z_hi` and their intensities.

```python
import argparse
import numpy as np
from zarr_vectors.building import (
    open_store, get_resolution_level, open_write_session,
    write_chunk_vertices, write_chunk_attributes,
)
# Internal module: building has no bin-level equivalent.
from zarr_vectors.spatial.chunking import assign_bins, group_bins_by_chunk

from my_reader import load_slab

BOUNDS = ([0., 0., 0.], [1000., 1000., 1000.])
CHUNK  = (250., 250., 250.)
BIN    = (125., 125., 125.)
BINS_PER_CHUNK = tuple(int(round(c / b)) for c, b in zip(CHUNK, BIN))
N_BINS = int(np.prod(BINS_PER_CHUNK))

parser = argparse.ArgumentParser()
parser.add_argument("--slab-index", type=int)
parser.add_argument("--n-slabs", type=int)
parser.add_argument("--output")
args = parser.parse_args()

dz = (BOUNDS[1][2] - BOUNDS[0][2]) / args.n_slabs
z_lo = BOUNDS[0][2] + args.slab_index * dz
positions, intensity = load_slab(z_lo, z_lo + dz)

level = get_resolution_level(open_store(args.output, mode="r+"), 0)
written = 0
with open_write_session(level, bounds=BOUNDS, chunk_shape=CHUNK):
    per_chunk = group_bins_by_chunk(assign_bins(positions, BIN), BINS_PER_CHUNK)
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

print(f"Slab {args.slab_index} done: {written} vertices")
```

`nonempty_chunks` is one attribute shared by every cell of an array, so tasks
that each update it race. On a level declared with `defer_presence`, no write
updates it, whatever `record_presence` says; on an undeclared level,
`record_presence=False` does the same per call.

A cell in a deferred level is visible as soon as its payload lands:
`list_chunks` and `chunk_exists` find it by listing the store, so a stage that
reads what a task just wrote sees it before the finalisation step. Readers
that take `nonempty_chunks` straight from `zarr.json` do not; for those, see
[Stamping under your own lock](#stamping-under-your-own-lock). On an
undeclared level, cells written with `record_presence=False` stay invisible to
`list_chunks` until the manifest is rebuilt.

Write one fragment per bin, as above: a cell written as a single fragment on a
store with `base_bin_shape` gives wrong bounding-box results (see
[Memory-efficient writes](memory_efficient_writes.md)).

### Finalisation, after all array tasks

```bash
#!/bin/bash
#SBATCH --dependency=afterok:${ARRAY_JOB_ID}
#SBATCH --ntasks=1

python finalise_store.py --output /scratch/scan.zarrvectors --vertices 80000000
```

```python
# finalise_store.py
import argparse
import zarr_vectors as zv
from zarr_vectors.building import (
    open_store, get_resolution_level, rebuild_presence,
    refresh_arrays_present, update_level_metadata, write_multiscale_metadata,
)

parser = argparse.ArgumentParser()
parser.add_argument("--output")
parser.add_argument("--vertices", type=int)
args = parser.parse_args()

root = open_store(args.output, mode="r+")
level = get_resolution_level(root, 0)

rebuild_presence(level)                      # every manifest; ends the deferral
print(refresh_arrays_present(level))         # the arrays actually on disk
update_level_metadata(level, vertex_count=args.vertices)
write_multiscale_metadata(root)

zv.open(args.output, mode="r+").build_pyramid(
    factors=[(2., 1.), (2., 1.)], chunk_scale_factors=[2, 2])
print("Store finalised.")
```

```text
['vertex_attributes', 'vertex_fragments', 'vertices']
Store finalised.
```

Never run this while tasks are still writing. `build_pyramid` needs the whole
store, so it belongs here rather than in a task.

`rebuild_presence` finds the cells by listing and reading the store. If your
tasks report the cells they wrote, record those instead:

```python
set_presence(level, {"vertices": cells, "vertex_fragments": cells,
                     "vertex_attributes/intensity": cells}, end_deferral=True)
```

Each named array's manifest becomes exactly the cells given, so name every
array the tasks wrote and pass the union of their cells.
`end_presence_deferral(level)` ends the declaration on its own; it raises
`StoreError` while any array has no manifest, unless you pass
`missing="derive"`, which rebuilds those from the store.

### Stamping under your own lock

If a consumer that reads `nonempty_chunks` directly must run before the
finalisation step, and your workers already serialise something (a file lock,
a queue, a database row), write the payloads unlocked and apply only the
manifest update under the lock. On a level without `defer_presence`,
`collect_presence` holds the stamps back and `apply_presence` writes them, one
attribute write per array:

```python
with level.collect_presence() as pending:
    for cell, groups in my_cells:                # payloads, unlocked
        write_chunk_vertices(level, cell, groups, dtype="float32")
with store_write_lock(path):                     # your lock
    level.apply_presence(pending)
```

The payloads land as they are written; only the stamps wait. When
`apply_presence` returns, `list_chunks` reports every cell. `collect_presence`
cannot be combined with `batched_writes` in either order, and
`open_write_session` opens a `batched_writes` block, so write outside a session
here; nesting them raises `StoreError`. A `pending` token belongs to the thread
that collected it.

| Situation | Use |
|---|---|
| A finalisation step follows, and early readers go through zarr-vectors | `defer_presence`, then `rebuild_presence` or `set_presence` |
| Readers that take `nonempty_chunks` from `zarr.json` run before finalisation, and you hold a lock | `collect_presence` and `apply_presence` |
| One writer | The default: each write updates the manifest |

## MPI

The same three phases, with `Barrier` where the job dependency was:

```python
from mpi4py import MPI
import numpy as np
from zarr_vectors.building import (
    LevelMetadata, create_store, create_resolution_level,
    create_vertices_array, create_attribute_array, defer_presence,
    open_write_session, open_store, get_resolution_level,
    write_chunk_vertices, write_chunk_attributes, rebuild_presence,
    refresh_arrays_present, update_level_metadata, write_multiscale_metadata,
)
from zarr_vectors.spatial.chunking import assign_bins, group_bins_by_chunk

from my_reader import load_x_range   # points with x_lo <= x < x_hi, intensities

STORE  = "/scratch/scan.zarrvectors"
BOUNDS = ([0., 0., 0.], [10_000., 10_000., 10_000.])
CHUNK  = (500., 500., 500.)
BIN    = (125., 125., 125.)
BINS_PER_CHUNK = tuple(int(round(c / b)) for c, b in zip(CHUNK, BIN))
N_BINS = int(np.prod(BINS_PER_CHUNK))

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

if rank == 0:                                     # phase 1
    root = create_store(STORE, bounds=BOUNDS, chunk_shape=CHUNK,
                        base_bin_shape=BIN, geometry_types=["point_cloud"])
    level = create_resolution_level(root, 0, LevelMetadata(
        level=0, vertex_count=0,
        arrays_present=["vertices", "vertex_attributes"]))
    with open_write_session(level, bounds=BOUNDS, chunk_shape=CHUNK):
        create_vertices_array(level, dtype="float32")
        create_attribute_array(level, "intensity", dtype="float32")
    defer_presence(level)
comm.Barrier()

dx = (BOUNDS[1][0] - BOUNDS[0][0]) / size        # phase 2
positions, intensity = load_x_range(rank * dx, (rank + 1) * dx)
level = get_resolution_level(open_store(STORE, mode="r+"), 0)
written = 0
with open_write_session(level, bounds=BOUNDS, chunk_shape=CHUNK):
    for cell, fragments in group_bins_by_chunk(
            assign_bins(positions, BIN), BINS_PER_CHUNK).items():
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

total = comm.reduce(written, op=MPI.SUM, root=0)
comm.Barrier()

if rank == 0:                                     # phase 3
    root = open_store(STORE, mode="r+")
    level = get_resolution_level(root, 0)
    rebuild_presence(level)
    refresh_arrays_present(level)
    update_level_metadata(level, vertex_count=total)
    write_multiscale_metadata(root)
```

Run it with `mpirun -n 16 python mpi_write.py`. Ranks need neither MPI-IO nor
parallel HDF5: each writes ordinary files under its own cell keys.

## Workers that append objects

A pipeline that builds objects cell by cell (skeleton paths with seam links
between neighbouring cells, for example) writes four things per batch, in this
order. Each call takes whole arrays:

```python
from zarr_vectors import building as zb

n0 = committed_objects                      # the coordinator's count
start, k = zb.write_chunk_fragments(        # 1. the new objects' fragments
    lg, cc, csr=(indices, offsets), mode="append")
zb.write_object_manifests(                  # 2. one manifest row per object
    lg, chunk_coords=np.tile(cc, (k, 1)), fragment_idx=np.arange(start, start + k),
    mode="append", at=n0)
zb.write_object_attribute_columns(          # 3. their attribute columns
    lg, {"length": lengths, "n_nodes": counts}, at=n0)
zb.write_link_cells(                        # 4. seams and their attributes
    lg, chunks=seam_chunks, vids=seam_vids, attributes={"w": weights})
# the coordinator then commits num_objects = n0 + k
```

On a store created with `manifest_layout="dense"`, step 2 stores the arrays as
given; see [Dense object indexes](gpu.md#dense-object-indexes).

When every worker has finished, the coordinator runs
`zb.finalize_links(lg, delta=0)` once. It rebuilds presence for the link,
link-attribute and link-fragment arrays; in a deferred level it leaves presence
to `rebuild_presence` or `set_presence`.

Two things are shared between workers here:

- The object index and object attribute columns. An `at=` append resizes the
  arrays and truncates or pads to `at`, so only one process may append. To fill
  them in parallel, reserve the rows and place them (below).
- Seam link cells. A seam is stored in the cell of its anchor chunk, which may
  belong to another worker. Route each seam to the worker that owns its anchor,
  or serialise these writes.

### One shard per task, published at the end

On a sharded store, `zb.shard_transaction(level, shard)` stages every cell
write inside the block. Each write must fall inside the shard, and reads of
the task's cells see the staged data. Under the default `mode="replace"` the
shard is rebuilt from the cells written in the block alone; `mode="merge"`
keeps the cells already there. On exit each changed shard object is written as
a `.partial` file, fsynced and renamed into place; an exception publishes
nothing. Presence must be deferred.

```python
shard = zb.shard_of(lg, task_cells[0])             # the shard's grid index
with zb.shard_transaction(lg, shard) as tx:
    for cc in task_cells:
        zb.write_chunk_vertices(lg, cc, groups[cc])
        zb.write_chunk_attributes(lg, "intensity", cc, values[cc])
report(tx.written)                                 # {array: ["i.j.k", ...]}
```

The coordinator allocates every array beforehand and afterwards records
presence from the tasks' reports with `zb.set_presence(..., end_deferral=True)`.
A crash during the renames leaves some of the task's shards new and some old;
re-running the task converges. `io_threads=` encodes, writes and fsyncs the
shard objects from a thread pool. On entry the transaction removes `.partial`
files a failed attempt left; `sweep=False` skips that listing.

### Placing object rows in parallel

`zv.concurrency_contract(level)` states what may be written at once: cells or
shards owned by one process each, and object rows placed in disjoint, aligned
ranges after one reservation.

```python
# Coordinator, before the workers:
contract = zv.concurrency_contract(lg)       # at_alignment: 65,536 unsharded
zb.reserve_object_rows(lg, n_objects, sid_ndim=3,
                       columns={"length": "float32"})

# Each worker, rows [lo, hi); lo and hi are multiples of at_alignment,
# except that the last range ends at n_objects:
zb.write_object_manifests(lg, chunk_coords=cc, fragment_idx=fi,
                          manifest_offsets=off, mode="place", at=lo)
zb.write_object_attribute_columns(lg, {"length": lengths}, mode="place", at=lo)

# Coordinator, once every worker has finished:
zb.commit_object_index(lg, n_objects)
```

`mode="place"` writes only rows `[at, at + n)`, with no resize, padding or
metadata. A dense index also needs each worker's fragments in its own aligned
block region: `zb.aligned_regions(counts, contract["block_alignment"])` turns
per-worker fragment counts into region starts, the last start is the
`n_blocks=` to reserve, and each worker passes its start as `block_at=`. Every
reserved row must be written, an object with no fragments as an empty
manifest.

A worker that needs a halo around its cell reads it in one prefetch with
`zb.read_neighbourhood(lg, cc, arrays, halo=1)`; see
[GPU arrays and array-form I/O](gpu.md).

## Divide read work across ranks

`Query.cells()` lists the grid cells a query touches, and `select(cells=...)`
restricts the query to some of them. Every rank can compute the same split
independently:

```python
import zarr_vectors as zv

rank, size = 0, 4                          # e.g. from MPI
ds = zv.open("/scratch/scan.zarrvectors")
q  = ds.select(bbox=((0., 0., 0.), (1000., 1000., 1000.)))

cells = sorted(q.cells(), key=lambda c: c.coords)
print(len(cells))

mine = zv.CellSet(cells[rank::size])       # this rank's share
print(len(mine), q.select(cells=mine).count())
```

```text
64
16 4000
```

On the 16 000-point store written by the SLURM example, the four ranks' counts
sum to the query's 16 000. To check that an install has this feature, call
`zv.require_api(features=["query-cells"])`.

## Lustre and GPFS

Stripe the output directory before writing (Lustre shown; adjust the count for
your system):

```bash
mkdir /scratch/scan.zarrvectors
lfs setstripe -c 8 /scratch/scan.zarrvectors
```

Cell files live at `<level>/<array>/c/i/j/k`; `0/vertices/c/` and
`0/vertex_attributes/<name>/c/` carry most of the bytes.

- Declare `shard_shape` on `create_store`, not on each writer: arrays a task
  creates on first write take the store's declaration. With sharding, many
  cells share one object, so two tasks writing disjoint cells may write
  the same file; give each task whole shards, as in
  [One shard per task](#one-shard-per-task-published-at-the-end).
- Use few, large write sessions. Each `open_write_session` block flushes its
  cells together, so one session covering many cells costs fewer round trips
  than one session per cell.
- Memory per task is roughly `cells_per_task × vertices_per_cell ×
  bytes_per_vertex`, plus encode buffers.

If a whole store is written by one task, you can write it to node-local
storage and copy it to the shared file system at the end:

```bash
python write_whole_store.py --output /local/scratch/scan_${SLURM_JOB_ID}.zarrvectors
rsync -a /local/scratch/scan_${SLURM_JOB_ID}.zarrvectors/ /lustre/project/scan.zarrvectors/
```

## Write to object storage

Install the cloud extra:

```bash
pip install "zarr-vectors[cloud]"
```

The functions above take a URL wherever they take a path; `s3://`, `gs://`
and `az://` URLs go to obstore if it is installed, else fsspec. Pass
credentials and endpoints as `storage_options=` to `create_store` and
`open_store`, or as `zv.StorageOptions(backend=..., options=...)` to
`zv.create` and `zv.open`.

Each cell written to object storage is a request, so large write sessions
matter more there. `building.shard_store` and `building.reshard` pack the
cells of a finished store into shard objects.

Without outbound network access from compute nodes, write to the shared file
system and copy afterwards:

```bash
aws s3 sync /lustre/project/scan.zarrvectors/ s3://my-bucket/scan.zarrvectors/
```

## Next steps

- [Memory-efficient writes](memory_efficient_writes.md): the same partitioned
  write in one process.
- [Choosing a layout](choose_chunk_and_bin.md): how to pick the cell size.
- {doc}`Building stores <../api/building>`: reference for every call used
  here.
