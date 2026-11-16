# Chunk shape

## Terms

**`chunk_shape`**
: A D-tuple of positive floats declaring the spatial extent of each chunk
  in the coordinate units of the store (e.g. micrometres, nanometres,
  voxels). A chunk at grid coordinate `(i_0, i_1, …, i_{D-1})` occupies
  the spatial region
  `[i_d × chunk_shape[d], (i_d + 1) × chunk_shape[d])` for each axis `d`.

**Chunk grid**
: The regular partition of space induced by `chunk_shape`. The grid a level
  allocates is anchored at `origin[d] = floor(bounds_min[d] / chunk_shape[d])`
  and spans `floor(bounds_max[d] / chunk_shape[d]) - origin[d] + 1` cells along
  axis `d` (the bounds are *inclusive*: a vertex exactly on `bounds_max` is
  storable). This is what `zarr_vectors.spatial.chunking.grid_layout` computes;
  `compute_grid_shape` (`ceil(extent / chunk_shape)`) is **not** the allocation
  and can be one cell short.

**Chunk coordinate**
: The D-tuple of integers identifying a chunk. It is **absolute** and may be
  negative: the chunk coordinate of a vertex with position `p` is
  `floor(p[d] / chunk_shape[d])` for each axis `d`. The chunk at coordinate `c`
  is stored in array cell `c - origin`.

**Storage key**
: The string path at which a chunk's data is stored in the Zarr store. For a
  `vertices/` chunk at *cell index* `(i, j, k)` (= absolute coordinate minus
  origin) the key is `0/vertices/c/i/j/k` (with the `c/` prefix of Zarr v3's
  default key encoding). The dotted form `"i.j.k"` used in `nonempty_chunks`
  and in the APIs is the **absolute** coordinate.

**Boundary chunk**
: A chunk whose spatial extent extends beyond the actual data bounding box.
  Boundary chunks are valid; Zarr stores them normally. Empty boundary
  chunks (all fill values) need not be written to disk.

**Read amplification**
: The ratio of data loaded from disk to data actually used by a query.
  A bbox query that spans N bins but loads M chunks is said to have
  amplification M / (N × average_bin_size / chunk_size_D).

---

## Introduction

`chunk_shape` is the single most consequential parameter in a Zarr Vectors store.
It controls both the physical layout of data on disk (or in object storage)
and the unit of I/O: every read operation fetches at least one full chunk,
and every write operation produces at least one full chunk file.

Choosing `chunk_shape` well means matching the granularity of your
I/O operations to the granularity of your access patterns. A chunk that
is too large wastes bandwidth on queries that need only a small spatial
region. A chunk that is too small creates excessive file-system metadata
overhead (millions of tiny files) and high per-request latency in cloud
storage.

Unlike `bin_shape`, which controls only the spatial index granularity,
`chunk_shape` is baked into the physical file layout and cannot be changed
without rewriting the store (see [Rechunking](rechunking.md)). Set
it deliberately before writing any data.

The unit of cost is the **cell**, not the vertex: every non-empty cell of every
array is one object in the store, and the per-cell cost of writing it is fixed.
A `chunk_shape` that allocates more than 100 000 cells makes the writer emit a
`RuntimeWarning` (suggesting a coarser `chunk_shape`, or `shard_store` to pack
cells into shard objects). The automatic layout used by the high-level API
targets a cell size rather than a cell count.

---

## Technical reference

### Definition and units

`chunk_shape` is declared as a D-tuple of positive floats in the root
attributes' `zarr_vectors` block:

```json
{ "zarr_vectors": { "chunk_shape": [200.0, 200.0, 200.0], "...": "..." } }
```

The unit is the same as the vertex coordinate unit declared by the axes'
`unit` fields in `multiscales[0].axes`. If the unit is `"micrometer"`, then each
chunk covers 200 µm per axis. If coordinates are in voxel indices,
`chunk_shape` values are still floats but are typically integers or simple
fractions.

If `chunk_shape` is omitted at creation, `create_store` uses one chunk covering
the bounds, and `write_points` one chunk covering the data.

There is no requirement that `chunk_shape` be isotropic. For anisotropic
data (e.g. 4 µm × 4 µm × 25 µm voxels), an anisotropic `chunk_shape`
that is roughly equal in physical extent per axis often performs better:

```json
{ "chunk_shape": [200.0, 200.0, 175.0] }
```

#### Per-level override

`chunk_shape` in the root attributes is the **level-0 default**. A coarser
level may carry its own `chunk_shape` in its `zarr_vectors_level` block, which
must be a positive integer multiple of the root's along every axis (nested
grids; `coarsen_level(..., chunk_scale_factor=2)` doubles the parent's). A
level without the key inherits the root's. See [Level groups](../layout/level_groups.md).

### Relationship to the Zarr array shape

Every per-spatial-chunk array (`vertices`, `vertex_fragments`, …) is a vlen-bytes
array whose logical shape **is** the chunk grid of its level —
`chunk_grid_shape`, e.g. `(5, 6, 4)` — and whose Zarr chunk shape is
`(1, 1, …, 1)`: one Zarr chunk (one file, `c/i/j/k`) per spatial chunk. There is
no `N_max` or `D` dimension; the number of vertices in a chunk is whatever the
variable-length blob holds. See [Chunk arrays](../layout/chunk_arrays.md).

`chunk_shape` (physical units) therefore determines the array *shape*, and the
vertex density of the data determines how big each cell's blob is.

Worked example: `chunk_shape = (100, 100, 100)` and
`bounds = ([-150, 0, 0], [250, 99.9, 99.9])`:

```python
import numpy as np
from zarr_vectors.spatial.chunking import grid_layout

origin, shape = grid_layout(([-150, 0, 0], [250, 99.9, 99.9]), (100, 100, 100))
# origin == (-2, 0, 0)   shape == (5, 1, 1)
# a vertex at x = -150 has absolute chunk coord -2 -> cell 0
# a vertex at x =  250 has absolute chunk coord  2 -> cell 4
```

### How chunk coordinates are computed

For a vertex at position `p = [p_0, p_1, p_2]` in a 3-D store:

```python
chunk_coord = tuple(int(math.floor(p[d] / chunk_shape[d])) for d in range(D))
```

The write functions in `zarr-vectors-py` compute it vectorised with
`numpy.floor(positions / chunk_shape)` in `float64`, so a coordinate that
rounds onto a boundary in floating point lands on the upper chunk. Negative
positions give negative chunk coordinates (handled by the grid origin).

Vertices with a coordinate exactly equal to `N × chunk_shape[d]` (a chunk
boundary) are placed in chunk `N`, not chunk `N-1`. This is the
conventional half-open interval `[N × C, (N+1) × C)`.

### Chunk size and data volume

The number of vertices per chunk depends on the vertex spatial density of
the data. For uniformly distributed point clouds:

```
expected_vertices_per_chunk ≈ total_vertices × (chunk_volume / total_volume)
```

where `chunk_volume = product(chunk_shape)` and `total_volume =
product(bounding_box_extent)`.

A practical target for interactive spatial queries is **10 000–100 000
vertices per chunk**. Fewer vertices per chunk means more chunks (more
files, more requests) but faster individual reads. More vertices per chunk
means fewer files but slower individual reads.

For streamlines and other extended objects, the relevant metric is not
vertex density but *object density per chunk* — the expected number of
streamlines that pass through a given chunk. A chunk should contain enough
streamlines to be worth fetching (typically 50–500 for a tractography
dataset).

### I/O implications

#### Local file system

Each Zarr Vectors spatial chunk corresponds to one file on disk. File-system
performance is sensitive to:

- **Inode overhead.** Every file incurs metadata overhead. On Linux ext4,
  this is negligible up to ~10 million files; on NFS, overhead per file
  can be significant. If the total expected chunk count exceeds ~1 million,
  consider larger `chunk_shape` or the sharding codec.
- **Directory entry lookup.** Zarr v3 uses `c/i/j/k` paths; intermediate
  directories are created as needed. Very deep chunk grids (many chunks)
  mean many nested directories, which is handled well by modern
  file systems but can slow down recursive directory listings.
- **Write bandwidth.** Large chunks saturate disk bandwidth more efficiently
  because per-write overhead is amortised over more data. For HPC parallel
  writes (many ranks writing simultaneously), smaller chunks reduce lock
  contention.

#### Cloud object storage

Each chunk corresponds to one object. Cloud object stores (S3, GCS) have:

- **Per-request cost.** S3 charges per PUT (write) and GET (read). Minimise
  chunk count by using larger `chunk_shape` or the sharding codec.
- **Minimum effective chunk size.** Requests smaller than ~64–256 KB are
  wasteful because the per-request overhead dominates. Ensure the typical
  compressed chunk size exceeds 128 KB.
- **Parallelism ceiling.** Cloud stores handle thousands of concurrent
  requests well. There is no benefit to reducing chunk count below the
  parallelism ceiling of your read pipeline.

#### Recommended chunk sizes by scenario

| Scenario | Recommended `chunk_shape` (3-D, isotropic) |
|----------|--------------------------------------------|
| Interactive visualisation (local) | 100–200 physical units |
| Interactive visualisation (cloud, S3) | 200–500 physical units |
| Batch analysis (HPC, Lustre) | 500–1000 physical units |
| Full-resolution archive (no queries) | 1000–2000 physical units |
| Neuroglancer serving | 64–128 nm or equivalent for sub-micrometre data |

These are starting points. Profile your actual access patterns and adjust.

### Chunk shape is immutable

`chunk_shape` cannot be changed after a store is written without rewriting
all chunk data. The chunk grid defines the physical file layout; changing
`chunk_shape` would require moving data between files.

If you need to change `chunk_shape` on an existing store, use
`zarr_vectors.rechunk.rechunk_spatial` (links-aware, all geometry types), as
described in [Rechunking](rechunking.md); `suggest_chunk_shape` proposes a
shape from the busiest cell's link count.

### Validation

L1: the root `zarr_vectors` block exists.

L2:
- `len(chunk_shape) == D` (the number of axes).
- All elements are strictly positive.
- If `base_bin_shape` is set, `chunk_shape[d] / base_bin_shape[d]` is an
  integer on every axis (tolerance `1e-9` on the ratio). This constraint is
  documented in detail in [Chunk vs bin](chunk_vs_bin.md).
- A per-level `chunk_shape` is a positive integer multiple of the root's.

The validators do not cross-check each array's declared grid shape against
`chunk_shape` and the bounds.
