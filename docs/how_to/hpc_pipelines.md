# HPC pipelines

This guide covers writing ZVF stores from HPC (High Performance Computing)
environments: SLURM job arrays, MPI parallel writes, Lustre/GPFS file system
optimisation, and sharding read work across ranks.

Parallel writing has a fixed shape in `zarr-vectors`, and every pattern below
is the same three phases:

1. **A coordinator creates the store and allocates its arrays — once.**
2. **Workers write disjoint sets of grid cells**, each skipping the shared
   presence manifest.
3. **A coordinator rebuilds that manifest** and writes the multiscale
   metadata.

All three use `zarr_vectors.building`, the supported surface for code that
constructs stores. `zarr_vectors.api` (`zv.create` / `add_points`) has no
partitioned writer: `add_points` is a one-shot write that re-derives the grid
from its own batch, so calling it once per task leaves you with one task's
data. See [Memory-efficient writes](memory_efficient_writes.md) for the
single-process version of the same pattern, and for what each per-chunk verb
does.

---

## Plan the grid before you submit

The chunk grid decides how many cells there are to divide among tasks, and it
is fixed at store creation. `Grid.plan` answers both questions without
creating anything — which is the only moment either answer can still change
the job script.

```python
import zarr_vectors as zv

BOUNDS = ((0., 0., 0.), (10_000., 10_000., 10_000.))
grid = zv.Grid.plan(bounds=BOUNDS, cell_size=(500., 500., 500.))
print(grid, grid.cells)
print(grid.capacity(n_vertices=500_000_000))
```

```text
Grid(20x20x20 cells of (500.0, 500.0, 500.0)) 8000
20x20x20 = 8000 cells, ~0.8 MB/cell -- fits
```

8000 cells and ~0.8 MB each: that sizes the `--array` range and tells you a
task holding a few hundred cells stays well inside a modest `--mem`.

### Partitions must be disjoint

Two tasks writing the same cell is a lost-update race, not a merge. The
reliable way to divide the work is to enumerate the grid and slice it —
disjoint by construction:

```python
cells = sorted(grid, key=lambda c: c.coords)      # every CellRef, ordered
n_tasks = 16
partition = zv.CellSet(cells[task_id::n_tasks])   # this task's cells
```

```{warning}
Do **not** cut partitions with `Grid.cells_in`. It is inclusive on the upper
boundary, so slabs cut on chunk boundaries share a whole plane of cells:
four 250 µm z-slabs of a 4×4×4 grid return 50 cells each — 200 in total for
a 64-cell grid — with 25 cells shared between each adjacent pair. That is
correct for a *read* (a box touching a boundary really does touch both
cells) and silently wrong for a parallel write.
```

Slicing coordinates rather than cells has the same trap: give each task a
half-open range (`z_lo <= z < z_hi`), never a closed one.

---

## SLURM job array pattern

### `init_store.py` — runs once, before the array

```python
import numpy as np
from zarr_vectors.building import (
    LevelMetadata, create_store, create_resolution_level,
    create_vertices_array, create_attribute_array, open_write_session,
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
```

```{warning}
This step is not optional and must not be folded into the array tasks. The
`create_*_array` calls rewrite each array's `zarr.json`, and inside a write
session that drops every cell already on disk — so a task that re-creates
the arrays erases whatever its siblings have written. Tasks write cells;
they never create arrays.
```

### Submit script

```bash
#!/bin/bash
#SBATCH --job-name=zvf_write
#SBATCH --array=0-19          # 20 z-slabs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

python write_partition.py \
    --slab-index ${SLURM_ARRAY_TASK_ID} \
    --n-slabs 20 \
    --output /scratch/scan.zarrvectors
```

### `write_partition.py`

```python
import argparse
import numpy as np
from zarr_vectors.building import (
    open_store, get_resolution_level, open_write_session,
    write_chunk_vertices, write_chunk_attributes,
)
# Internal import -- see the note below.
from zarr_vectors.spatial.chunking import assign_bins, group_bins_by_chunk

BOUNDS = ([0., 0., 0.], [1000., 1000., 1000.])
CHUNK  = (250., 250., 250.)
BIN    = (125., 125., 125.)
BINS_PER_CHUNK = tuple(int(round(c / b)) for c, b in zip(CHUNK, BIN))
N_BINS = int(np.prod(BINS_PER_CHUNK))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slab-index", type=int)
    parser.add_argument("--n-slabs", type=int)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    # Half-open slab: [z_lo, z_hi).  Adjacent slabs must not share a cell.
    dz = (BOUNDS[1][2] - BOUNDS[0][2]) / args.n_slabs
    z_lo = BOUNDS[0][2] + args.slab_index * dz

    positions = load_slab_positions(z_lo, z_lo + dz)
    intensity = load_slab_intensity(z_lo, z_lo + dz)

    level = get_resolution_level(open_store(args.output, mode="r+"), 0)
    written = 0
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

    print(f"Slab {args.slab_index} done: {written} vertices")

if __name__ == "__main__":
    main()
```

`record_presence=False` is what makes this safe to run in parallel.
`nonempty_chunks` is one attribute shared by every cell of an array, so a task
that stamps it races every other task; tasks skip it and the finalisation step
rebuilds it from disk.

Write the vertices of a cell as one fragment **per bin**, as above, rather
than one fragment for the whole cell. Fragments are what a coarsener draws
representatives from, so a single-fragment cell produces a nearly empty
pyramid — see
[Memory-efficient writes](memory_efficient_writes.md#one-fragment-per-bin-not-one-per-cell).

```{note}
`assign_bins` and `group_bins_by_chunk` come from
`zarr_vectors.spatial.chunking`, which is **internal**: this import reaches
past the contract and may break between releases.
`zarr_vectors.building` exports `assign_chunks` (vertices → cells) but no
bin-level equivalent. That is a gap to report rather than a reason to treat
`spatial` as public.
```

### Finalisation step (after all array jobs complete)

```bash
#!/bin/bash
#SBATCH --dependency=afterok:${ARRAY_JOB_ID}
#SBATCH --ntasks=1
#SBATCH --mem=8G

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

rebuild_presence(level)                      # the nonempty_chunks nobody stamped
print(refresh_arrays_present(level))         # what is actually on disk
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

`refresh_arrays_present` walks the level rather than trusting what the writers
declared, which is why `vertex_fragments` shows up even though `init_store.py`
never listed it. Both it and `rebuild_presence` are coordinator verbs — never
run either while tasks are still writing.

`build_pyramid` needs the whole store, so it belongs here rather than in a
task. Coarsening a single level in isolation is not on the supported surface;
if that is what you need, report it rather than importing
`zarr_vectors.multiresolution` or `zarr_vectors.ops`, which are internal.

---

## MPI parallel writes

The same three phases, with `Barrier` where the job dependency was. Rank 0
allocates, every rank writes its own cells, rank 0 finalises:

```python
from mpi4py import MPI
import numpy as np
import zarr_vectors as zv
from zarr_vectors.building import (
    LevelMetadata, create_store, create_resolution_level,
    create_vertices_array, create_attribute_array, open_write_session,
    open_store, get_resolution_level, write_chunk_vertices,
    write_chunk_attributes, rebuild_presence, refresh_arrays_present,
    update_level_metadata, write_multiscale_metadata,
)
# Internal import -- bin-level assignment has no supported equivalent; see the
# note under `write_partition.py` above.
from zarr_vectors.spatial.chunking import assign_bins, group_bins_by_chunk

STORE  = "/scratch/scan.zarrvectors"
BOUNDS = ([0., 0., 0.], [10_000., 10_000., 10_000.])
CHUNK  = (500., 500., 500.)
BIN    = (125., 125., 125.)
BINS_PER_CHUNK = tuple(int(round(c / b)) for c, b in zip(CHUNK, BIN))
N_BINS = int(np.prod(BINS_PER_CHUNK))

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --- phase 1: rank 0 allocates -----------------------------------------
if rank == 0:
    root = create_store(STORE, bounds=BOUNDS, chunk_shape=CHUNK,
                        base_bin_shape=BIN, geometry_types=["point_cloud"])
    level = create_resolution_level(root, 0, LevelMetadata(
        level=0, vertex_count=0,
        arrays_present=["vertices", "vertex_attributes"]))
    with open_write_session(level, bounds=BOUNDS, chunk_shape=CHUNK):
        create_vertices_array(level, dtype="float32")
        create_attribute_array(level, "intensity", dtype="float32")
comm.Barrier()

# --- phase 2: every rank writes a half-open x-slab ---------------------
dx = (BOUNDS[1][0] - BOUNDS[0][0]) / size
x_lo = BOUNDS[0][0] + rank * dx
positions = load_positions_x_range(x_lo, x_lo + dx)   # [x_lo, x_lo + dx)
intensity = load_intensity_x_range(x_lo, x_lo + dx)

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

# --- phase 3: rank 0 finalises -----------------------------------------
if rank == 0:
    root = open_store(STORE, mode="r+")
    level = get_resolution_level(root, 0)
    rebuild_presence(level)
    refresh_arrays_present(level)
    update_level_metadata(level, vertex_count=total)
    write_multiscale_metadata(root)
    print(f"Wrote {total} vertices from {size} ranks.")
```

Run with:

```bash
mpirun -n 16 python mpi_write.py
```

Ranks do not need MPI-IO or a parallel HDF5 build: each writes ordinary files
under its own cell keys, and the file system does the rest. The only shared
mutable state is the presence manifest, which is exactly what
`record_presence=False` keeps them out of.

---

## Sharding read work across ranks

Reading back is easier than writing, because `Query.cells()` computes which
grid cells a query touches from metadata alone — it reads nothing — and the
result feeds straight back into `select(cells=...)`:

```python
import zarr_vectors as zv

ds = zv.open("/scratch/scan.zarrvectors")
q  = ds.select(bbox=((0., 0., 0.), (1000., 1000., 1000.)))

cells = sorted(q.cells(), key=lambda c: c.coords)
print(len(cells))

mine = zv.CellSet(cells[rank::size])       # this rank's share
print(len(mine), q.select(cells=mine).count())
```

```text
125
32 3929
```

Each rank reads only its own cells, and the per-rank counts sum exactly to the
whole query — 16 000 across four ranks in the store above. Because `cells()`
is metadata arithmetic, every rank can compute the same partition
independently, with no communication and no coordinator.

`select(cells=...)` is the supported replacement for the old
`chunks=[(3, 1, 2), ...]` argument: a `CellRef` is opaque, so a change in how
cells are addressed on disk is not a downstream break. Guard on the feature if
you need to:

```python
zv.require_api(features=["query-cells"])
```

---

## Lustre / GPFS optimisation

### Striping

Stripe the output directory across multiple OSTs before writing:

```bash
# Lustre: stripe across 8 OSTs (adjust for your system)
mkdir /scratch/scan.zarrvectors
lfs setstripe -c 8 /scratch/scan.zarrvectors
```

For stores with many cells per directory, keep the metadata documents on a
narrow stripe and the cell files on a wide one:

```bash
lfs setstripe -c 1 -S 1m /scratch/scan.zarrvectors        # metadata stripe
lfs setstripe -c 8 -S 4m /scratch/scan.zarrvectors/0/vertices/c/
```

Cell files live at `<array>/c/i/j/k` — one file per grid cell, per array
family. `0/vertices/c/` and `0/vertex_attributes/<name>/c/` are the two that
carry the bulk of the bytes.

### Parallel I/O considerations

- **Never write the same cell from two tasks.** Partition the grid, not the
  coordinate space, or use half-open coordinate ranges — see
  [Partitions must be disjoint](#partitions-must-be-disjoint). Nothing in the
  format detects the collision; the second writer simply wins.
- **Skip the presence manifest in tasks.** Pass `record_presence=False` to
  every per-chunk write and rebuild once, afterwards. This is the one piece of
  genuinely shared state.
- **Use a large `chunk_shape` on Lustre.** Stripe granularity is typically
  1–4 MB; cells smaller than that see no parallelism benefit.
  `Grid.plan(...).capacity(n_vertices=...)` gives the per-cell figure before
  you commit to a shape.
- **Fewer, larger write sessions.** Each `open_write_session` block flushes
  its cells in one concurrent gather, so a task covering many cells in one
  session costs far fewer round-trips than one session per cell.
- **Avoid `O_SYNC` writes.** Zarr v3 does not use synchronous writes by
  default. If your Lustre mount forces `O_SYNC`, contact your sysadmin.

### Temporary local SSD → copy to Lustre

On clusters with local NVMe scratch (e.g. `/local/scratch`), write to local
storage first and copy to Lustre at the end. This works only when a whole
store is written by one task — tasks on different nodes cannot share a local
scratch directory, so each needs its own store, and there is no merge step.

```bash
# SLURM: write to local NVMe, copy to Lustre at job end
python write_whole_store.py --output /local/scratch/scan_${SLURM_JOB_ID}.zarrvectors

# After job
rsync -a /local/scratch/scan_${SLURM_JOB_ID}.zarrvectors/ \
         /lustre/project/scan.zarrvectors/
```

---

## Writing to S3 from HPC

For cloud-destined datasets, write directly to S3 from the compute nodes (if
outbound internet is available):

```bash
# Install cloud extras in your conda environment
pip install "zarr-vectors[cloud]"
```

Every function above takes a URL wherever it takes a path; the backend layer
routes `s3://`, `gs://` and `az://` to obstore or fsspec by scheme. Nothing
else about the pattern changes:

```python
import zarr_vectors as zv
from zarr_vectors.building import create_store, open_store

print(zv.detect_scheme("s3://my-bucket/scan.zarrvectors"))

STORE = "s3://my-bucket/scan.zarrvectors"
root  = create_store(STORE, bounds=BOUNDS, chunk_shape=CHUNK,
                     base_bin_shape=BIN, geometry_types=["point_cloud"])
# ... tasks: open_store(STORE, mode="r+") exactly as above
```

```text
s3
```

Credentials and endpoint overrides go through `storage_options` on
`create_store` / `open_store`, or `zv.StorageOptions(backend=..., options=...)`
on the `zv.create` / `zv.open` side.

Cell-per-cell writes to object storage are latency-bound, so the "fewer,
larger write sessions" advice above matters much more here than on Lustre.
Consider sharding the finished store — `zv.sharding` is internal, but
`building.shard_store` and `building.reshard` are the supported entry points —
so many cells pack into one storage object. Sharding does not change the key
layout: keys stay `c/i/j/k`, each file just covers several cells.

On clusters without outbound internet, write to Lustre first, then copy to S3
as a post-processing step:

```bash
# After all SLURM partitions complete
aws s3 sync /lustre/project/scan.zarrvectors/ \
            s3://my-bucket/scan.zarrvectors/ \
            --no-progress
```

---

## Recommended job sizing

| Dataset size | Strategy | SLURM resources |
|-------------|----------|-----------------|
| < 10M vertices | Single job, `zv.create` + `add_points` | 1 node, 32 GB RAM, 4 CPU |
| 10M–1B vertices | Job array over disjoint grid partitions | 16–64 tasks, 16–32 GB each |
| > 1B vertices | MPI over disjoint grid partitions, smaller partitions per rank | 64–256 ranks, 8 GB each |
| Any size, S3 destination | As above; add the `[cloud]` extra, fewer and larger write sessions | As above |

Per-task memory is set by the partition, not by the dataset: roughly
`cells_per_task × vertices_per_cell × bytes_per_vertex`, doubled for the
encode buffer. Shrinking partitions lowers memory at the cost of more round
trips, so tune the two together rather than raising `--mem`.

---

## Next steps

- **[Memory-efficient writes](memory_efficient_writes.md)** — the same
  partitioned write in one process, and what each per-chunk verb does.
- **[Choosing a layout](choose_chunk_and_bin.md)** — how to
  pick the numbers `Grid.plan` takes.
- **{doc}`Building stores <../api/building>`** — the full reference for the
  surface every example here uses.
