# Memory-efficient writes

This page shows how to write a point set that does not fit in memory. You
allocate the store once, write it one partition of whole cells at a time, then
finalise it. Peak memory is one partition.

The unit you stream is the cell: every vertex in a cell must be in memory when
the cell is written. `Dataset.add_points` is not an append; a second call
replaces the first. Use it when the data fits in memory, and
`zarr_vectors.building`, as below, when it does not.

## What you need

- The bounds of the whole dataset.
- A way to load one spatial slab of the input at a time. The example uses a
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

`Grid.plan` computes the grid without creating a store, and `capacity()`
estimates bytes per cell. Changing the cell size later means rewriting the
store, so do this first:

```python
import zarr_vectors as zv

grid = zv.Grid.plan(bounds=((0., 0., 0.), (10_000., 10_000., 10_000.)),
                    cell_size=(500., 500., 500.))
print(grid.capacity(n_vertices=500_000_000))
```

```text
21x21x21 = 9261 cells, ~0.6 MB/cell -- fits
```

The estimate counts coordinates only. [Choosing a
layout](choose_chunk_and_bin.md#step-2-check-the-cell-size) covers the choice.

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

Every write session must pass the same `bounds` and `chunk_shape`.
`defer_presence` stops writes from updating the presence manifest,
`nonempty_chunks`, which every cell of an array shares. Until the store is
finalised, zarr-vectors readers find cells by listing the store.

## Step 3: write one partition at a time

Each partition loads its slab, writes its cells, and frees the slab:

```python
from zarr_vectors.building import (
    open_store, get_resolution_level, write_chunk_vertices,
    write_chunk_attributes,
)
# Internal module: building has no bin-level equivalent.
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

Write one fragment per bin, as above. A cell written as a single fragment on a
store with `base_bin_shape` gives wrong bounding-box results, and `validate()`
only warns.

A partition can be any set of whole cells, as long as no two partitions share
a cell. Peak memory is about `cells_per_partition × vertices_per_cell ×
bytes_per_vertex`, plus encode buffers.

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
refresh_arrays_present(level)              # the arrays actually on disk
update_level_metadata(level, vertex_count=written)
write_multiscale_metadata(root)

ds = zv.open(STORE, mode="r+")
print(ds.read())
report = ds.build_pyramid(factors=[(2., 1.), (2., 1.)], chunk_scale_factors=[2, 2])
print(report["levels_created"])
```

```text
ReadResult(kind='point_cloud', vertices=16000, attributes=['intensity'])
2
```

Never finalise while partitions are still writing.

## Result

A store with all 16 000 points at level 0 and two coarser levels. To change
the cell size of an existing store, see
[Rechunking](../spec/chunking/rechunking.md).

## Next steps

- [HPC pipelines](hpc_pipelines.md): the same write across SLURM array tasks
  or MPI ranks.
- {doc}`Building stores <../api/building>`: reference for every call used
  here.
