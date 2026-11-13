# HPC pipelines

This guide shows how to write one store from many processes, such as SLURM
array tasks or MPI ranks, and how to split read work between ranks.

Every parallel write has three phases:

1. One process creates the store and allocates its arrays.
2. Workers write disjoint sets of cells, without updating the shared presence
   manifest.
3. One process writes the manifests and the level metadata.

[Memory-efficient writes](memory_efficient_writes.md) shows each phase in one
process, with runnable code. This page shows how to spread it over a cluster.
`Dataset.add_points` is a one-shot write, so it cannot be called from each
task.

## Partition the grid

Two tasks that write the same cell lose data, with no error. Enumerate the
grid and slice it, which gives disjoint partitions:

```python
import zarr_vectors as zv

grid = zv.Grid.plan(bounds=((0., 0., 0.), (1000., 1000., 1000.)),
                    cell_size=(250., 250., 250.))
task_id, n_tasks = 3, 16                          # e.g. from the scheduler
cells = sorted(grid, key=lambda c: c.coords)      # every CellRef, ordered
partition = zv.CellSet(cells[task_id::n_tasks])   # this task's cells
```

Do not cut partitions with `Grid.cells_in`: it includes cells on the upper
boundary of the box, so adjacent slabs share cells. If you split by
coordinates, give each task a half-open range (`z_lo <= z < z_hi`).

## SLURM job array

1. Run Step 2 of [Memory-efficient writes](memory_efficient_writes.md) once,
   before the array. It creates the store, allocates the arrays and calls
   `defer_presence`.
2. Submit the array. Each task runs Step 3 for its own slab only:

   ```bash
   #!/bin/bash
   #SBATCH --array=0-19
   #SBATCH --ntasks=1

   python write_partition.py --slab-index ${SLURM_ARRAY_TASK_ID} --n-slabs 20 \
       --output /scratch/scan.zarrvectors
   ```

3. Submit the finalisation, which runs Step 4, to start after the array:

   ```bash
   #!/bin/bash
   #SBATCH --dependency=afterok:${ARRAY_JOB_ID}
   #SBATCH --ntasks=1

   python finalise_store.py --output /scratch/scan.zarrvectors
   ```

Build the pyramid in the finalisation step: `build_pyramid` needs the whole
store. Cells a task has written are visible to `list_chunks` before
finalisation, because a deferred level is read by listing the store.

If your tasks report the cells they wrote, record them instead of
`rebuild_presence`, which lists the store. Name every array the tasks wrote:

```python
set_presence(level, {"vertices": cells, "vertex_fragments": cells,
                     "vertex_attributes/intensity": cells}, end_deferral=True)
```

## MPI

The same three phases, with `Barrier` where the job dependency was:

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

if rank == 0:
    allocate_store()                 # phase 1: Step 2 of Memory-efficient writes
comm.Barrier()

written = write_cells(rank, size)    # phase 2: Step 3, this rank's cells only
total = comm.reduce(written, op=MPI.SUM, root=0)
comm.Barrier()

if rank == 0:
    finalise_store(total)            # phase 3: Step 4
```

Run it with `mpirun -n 16 python mpi_write.py`. Ranks need neither MPI-IO nor
parallel HDF5: each writes ordinary files under its own cell keys.

## Stamping under your own lock

Use this when a reader that takes `nonempty_chunks` straight from `zarr.json`
must run before finalisation, and your workers already share a lock. Write
the payloads unlocked and apply only the manifest update under the lock. This
works on a level without `defer_presence`:

```python
with level.collect_presence() as pending:
    for cell, groups in my_cells:                # payloads, unlocked
        write_chunk_vertices(level, cell, groups, dtype="float32")
with store_write_lock(path):                     # your lock
    level.apply_presence(pending)
```

When `apply_presence` returns, `list_chunks` reports every cell. Write outside
`open_write_session`: `collect_presence` cannot be nested with it, and raises
`StoreError` if you try.

## Workers that append objects

A pipeline that builds objects cell by cell writes four things per batch, in
this order:

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

When every worker has finished, the coordinator runs
`zb.finalize_links(lg, delta=0)` once.

Only one process may append to the object index (`at=` resizes the arrays). A
seam link is stored in the cell of its anchor chunk, so route each seam to the
worker that owns that cell. To fill the object index in parallel, see
[Placing object rows in parallel](#placing-object-rows-in-parallel).

## One shard per task

On a sharded store, `zb.shard_transaction(level, shard)` stages every write
in the block and publishes the shard atomically on exit. An exception
publishes nothing. Presence must be deferred.

```python
shard = zb.shard_of(lg, task_cells[0])             # the shard's grid index
with zb.shard_transaction(lg, shard) as tx:
    for cc in task_cells:
        zb.write_chunk_vertices(lg, cc, groups[cc])
        zb.write_chunk_attributes(lg, "intensity", cc, values[cc])
report(tx.written)                                 # {array: ["i.j.k", ...]}
```

Every write must fall inside the shard. The coordinator then records presence
from the reports with `zb.set_presence(..., end_deferral=True)`. Re-running a
failed task is safe.

## Placing object rows in parallel

`zv.concurrency_contract(level)` states what may be written at once. Object
rows go in disjoint, aligned ranges after one reservation:

```python
# Coordinator, before the workers:
contract = zv.concurrency_contract(lg)
zb.reserve_object_rows(lg, n_objects, sid_ndim=3,
                       columns={"length": "float32"})

# Each worker, rows [lo, hi); lo and hi are multiples of
# contract["at_alignment"], except that the last range ends at n_objects:
zb.write_object_manifests(lg, chunk_coords=cc, fragment_idx=fi,
                          manifest_offsets=off, mode="place", at=lo)
zb.write_object_attribute_columns(lg, {"length": lengths}, mode="place", at=lo)

# Coordinator, once every worker has finished:
zb.commit_object_index(lg, n_objects)
```

Every reserved row must be written; an object with no fragments gets an empty
manifest. A [dense index](gpu.md#dense-object-indexes) also needs each
worker's fragments in its own region, from `zb.aligned_regions`.

## Divide read work across ranks

`Query.cells()` lists the cells a query touches, and `select(cells=...)`
restricts the query to some of them. Every rank can compute the same split:

```python
import zarr_vectors as zv

rank, size = 0, 4                          # e.g. from MPI
ds = zv.open("/scratch/scan.zarrvectors")
q  = ds.select(bbox=((0., 0., 0.), (1000., 1000., 1000.)))

cells = sorted(q.cells(), key=lambda c: c.coords)
mine = zv.CellSet(cells[rank::size])       # this rank's share
print(len(cells), len(mine), q.select(cells=mine).count())
```

```text
64 16 4000
```

## Lustre and GPFS

- Stripe the output directory before writing, for example
  `lfs setstripe -c 8 /scratch/scan.zarrvectors` on Lustre.
- Declare `shard_shape` on `create_store`. With sharding, many cells share one
  file, so give each task whole shards.
- Use few, large write sessions: each `open_write_session` block flushes its
  cells together.
- A store written by one task can go to node-local storage and be copied to
  the shared file system at the end.

To write to object storage, pass a URL wherever these functions take a path;
see [Cloud stores](cloud_stores.md).

## Next steps

- {doc}`Building stores <../api/building>`: reference for every call used
  here.
