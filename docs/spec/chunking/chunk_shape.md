# Chunk shape

## Terms

**`chunk_shape`**
: A D-tuple of positive floats declaring the spatial extent of each chunk
  in the coordinate units of the store (e.g. micrometres, nanometres,
  voxels). A chunk at grid coordinate `(i_0, i_1, …, i_{D-1})` occupies
  the spatial region
  `[i_d × chunk_shape[d], (i_d + 1) × chunk_shape[d])` for each axis `d`.

**Chunk grid**
: The regular partition of space induced by `chunk_shape`. The number of
  chunks along axis `d` is `ceil(extent[d] / chunk_shape[d])`, where
  `extent[d]` is the maximum coordinate observed across all vertices.

**Chunk coordinate**
: The D-tuple of non-negative integers identifying a chunk within the
  chunk grid. The chunk coordinate of a vertex with position `p` is
  `floor(p[d] / chunk_shape[d])` for each axis `d`.

**Storage key**
: The string path at which a chunk's compressed data is stored in the
  Zarr store. For a `vertices/` chunk at coordinate `(i, j, k)` the key
  is `0/vertices/c/i/j/k` (with `c/` prefix per Zarr v3
  default encoding).

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
without rewriting the entire store (see [Rechunking](rechunking.md)). Set
it deliberately before writing any data.

---

## Technical reference

### Definition and units

`chunk_shape` is declared as a D-tuple of positive floats in root `.zattrs`:

```json
{ "chunk_shape": [200.0, 200.0, 200.0] }
```

The unit is the same as the vertex coordinate unit declared in
`"axis_units"`. If `axis_units` is `"micrometer"`, then each chunk
covers 200 µm per axis. If coordinates are in voxel indices (integer),
`chunk_shape` values are still floats but are typically integers or
simple fractions.

There is no requirement that `chunk_shape` be isotropic. For anisotropic
data (e.g. 4 µm × 4 µm × 25 µm voxels), an anisotropic `chunk_shape`
that is roughly equal in physical extent per axis often performs better:

```json
{ "chunk_shape": [200.0, 200.0, 175.0] }
```

### Relationship to the Zarr array shape

The `vertices/` Zarr array has logical shape `(*chunk_grid_shape, N_max, D)`
where each element of `chunk_grid_shape` is the number of chunks along
that axis. The Zarr chunk shape (the unit of storage for the Zarr array
itself) is `(1, 1, …, 1, N_max, D)` — one Zarr chunk per Zarr Vectors spatial
chunk.

The Zarr Vectors `chunk_shape` (physical units) and the Zarr chunk shape (array
elements) are related only through the vertex density of the data. There
is no direct mathematical mapping between them; the Zarr array's chunk
dimensions are set by `N_max` (maximum expected vertices per Zarr Vectors chunk),
not by `chunk_shape`.

### How chunk coordinates are computed

For a vertex at position `p = [p_0, p_1, p_2]` in a 3-D store:

```python
chunk_coord = tuple(int(math.floor(p[d] / chunk_shape[d])) for d in range(D))
```

This is an exact integer division; floating-point coordinates very close
to a chunk boundary (within floating-point epsilon) are assigned to the
chunk whose lower bound is nearest. The write functions in
`zarr-vectors-py` use `numpy.floor` for consistency.

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

If you need to change `chunk_shape` on an existing store, use the
rechunking workflow described in [Rechunking](rechunking.md).

### Validation

L1: `chunk_shape` is present in root `.zattrs` and is a list of D positive
numbers.

L2:
- `len(chunk_shape) == spatial_dims`.
- All elements are strictly positive.
- `chunk_shape[d] % bin_shape[d] == 0` for all `d` (within floating-point
  tolerance `1e-6 × chunk_shape[d]`). This constraint is documented in
  detail in [Chunk vs bin](chunk_vs_bin.md).
- The `chunk_grid` shape declared in each Zarr array's `zarr.json` is
  consistent with the `chunk_shape` and the data bounding box.
