# Bin shape

## Terms

**`bin_shape`**
: A D-tuple of positive floats declaring the spatial extent of each
  supervoxel bin within a chunk. `bin_shape` must evenly divide
  `chunk_shape` in every dimension. The number of bins per chunk along
  axis `d` is `chunk_shape[d] / bin_shape[d]`.

**Supervoxel bin**
: The finest spatial subdivision in Zarr Vectors. Each bin is a rectangular region
  of physical space whose size is `bin_shape`. Bins tile the interior of
  each chunk exactly, with no gaps or overlaps. All vertices whose position
  falls within a bin are stored together as a fragment (fragment).

**Bins per chunk (`B_per_chunk`)**
: The total number of bins within a single chunk:
  `product(chunk_shape[d] / bin_shape[d] for d in range(D))`. For
  `chunk_shape = (200, 200, 200)` and `bin_shape = (50, 50, 50)`:
  `B_per_chunk = 4 × 4 × 4 = 64`.

**Effective bin shape**
: The bin shape at a given resolution level, after applying the level's
  `bin_ratio`. `effective_bin_shape[d] = base_bin_shape[d] × bin_ratio[d]`.
  At level 0, effective bin shape equals `base_bin_shape`.

**`base_bin_shape`**
: The bin shape at the full-resolution level (level 0). Stored in root
  `.zattrs`. Serves as the reference from which all coarser-level bin
  shapes are derived.

**Divisibility constraint**
: The requirement that `chunk_shape[d] % bin_shape[d] == 0` for all `d`.
  Ensures bins tile chunks exactly, which is necessary for the fragment index
  to be well-defined. Validated at write time and by the L2 validator.

---

## Introduction

Bins are the unit of spatial querying in Zarr Vectors. When you issue a bounding-box
query, the query engine does not load entire chunks — it identifies which
bins overlap the query region and reads only those bins' vertices. This
sub-chunk spatial resolution is what makes Zarr Vectors efficient for small targeted
queries on large datasets.

`bin_shape` is the parameter that controls this query granularity. A smaller
`bin_shape` means finer spatial queries (less read amplification for small
bboxes) but more bins per chunk (larger fragment index, more index bookkeeping). A
larger `bin_shape` means coarser queries but lower index overhead.

Unlike `chunk_shape`, `bin_shape` does not affect the physical file layout.
Changing `bin_shape` requires only recomputing the fragment index and re-sorting
vertices within each chunk — the chunk files themselves stay the same size
and location. Re-binning is exposed for point-cloud stores via
[`zarr_vectors.rechunk.rebin_level(store_path, target_bin_shape, level=0)`](../../../zarr_vectors/rechunk/rebin.py);
for other geometry types (polyline, streamline, graph, skeleton, mesh)
the fragment structure carries per-object semantics so a full
[rechunk](rechunking.md) is required instead.

---

## Technical reference

### Declaration and storage

`base_bin_shape` is declared in root `.zattrs`:

```json
{
  "chunk_shape":    [200.0, 200.0, 200.0],
  "base_bin_shape": [50.0,  50.0,  50.0]
}
```

The effective bin shape at each resolution level is stored in the per-level
`.zattrs` for convenience:

```json
{
  "level":     1,
  "bin_ratio": [2, 2, 2],
  "bin_shape": [100.0, 100.0, 100.0]
}
```

Readers should compute `bin_shape` at level `N` as:
```
bin_shape[N][d] = base_bin_shape[d] × bin_ratio[N][d]
```

and cross-check against the stored `bin_shape` value. A mismatch is a
validation error (L2).

### Divisibility constraint

For bins to tile chunks exactly, `bin_shape` must divide `chunk_shape`
evenly in every dimension:

```
chunk_shape[d] / bin_shape[d]  must be a positive integer,  for all d
```

This is enforced at write time. The write functions raise `ValueError` if
the constraint is violated.

**Example of a valid configuration:**

```
chunk_shape = (200, 200, 200)
bin_shape   = (50, 50, 50)    → 4.0 bins per axis ✓
```

**Example of an invalid configuration:**

```
chunk_shape = (200, 200, 200)
bin_shape   = (60, 60, 60)    → 3.333… bins per axis ✗
```

The constraint applies at every resolution level. Because the effective
bin shape at level `N` is `base_bin_shape × bin_ratio[N]`, and
`base_bin_shape` already satisfies the constraint at level 0, coarser
levels are automatically valid provided `bin_ratio` components are
positive integers and the products `base_bin_shape[d] × bin_ratio[N][d]`
still divide `chunk_shape[d]`.

For example, with `chunk_shape = (200,)`, `base_bin_shape = (50,)`,
and `bin_ratio = [4]` at level 2: `effective_bin_shape = 200.0`. This is
valid (1 bin per chunk at level 2). `bin_ratio = [5]` would give
`effective_bin_shape = 250.0 > chunk_shape[0]`, which is invalid.

**Rule:** `bin_ratio[N][d]` must satisfy
`base_bin_shape[d] × bin_ratio[N][d] ≤ chunk_shape[d]`, and the ratio
must be a divisor.

### Bin coordinate arithmetic

For a vertex at position `p` within a chunk at coordinate `chunk_coord`:

```python
# Local position within the chunk (physical units)
local = p - np.array(chunk_coord) * np.array(chunk_shape)  # shape (D,)

# Bin coordinate (integer, 0-indexed within chunk)
bin_coord = np.floor(local / effective_bin_shape).astype(int)  # shape (D,)

# Flat bin index (C-order ravel)
bpc_shape  = tuple(int(c / b) for c, b in zip(chunk_shape, effective_bin_shape))
bin_flat   = int(np.ravel_multi_index(bin_coord, bpc_shape))
```

### Default behaviour when `bin_shape` is omitted

If `bin_shape` is not specified at write time, it defaults to
`chunk_shape`. This results in exactly one bin per chunk
(`B_per_chunk = 1`), which is equivalent to a purely chunk-indexed store
without sub-chunk spatial indexing.

This default preserves backward compatibility with readers and tools that
do not understand the bin concept. A bbox query on a one-bin-per-chunk
store degrades gracefully to chunk-granularity queries.

### Choosing `bin_shape`

The optimal `bin_shape` depends on the typical bounding-box query size
relative to `chunk_shape`.

**Heuristic:** set `bin_shape` so that a typical query bbox spans 2–8 bins
per axis. This ensures the query reads only the relevant data while keeping
`B_per_chunk` manageable (8–512 bins per chunk for a 3-D store).

| `chunk_shape` | Typical query bbox | Recommended `bin_shape` | `B_per_chunk` |
|--------------|-------------------|------------------------|---------------|
| (200, 200, 200) | ~50³ region | (50, 50, 50) | 64 |
| (500, 500, 500) | ~100³ region | (100, 100, 100) | 125 |
| (500, 500, 500) | ~250³ region | (250, 250, 250) | 8 |
| (1000, 1000, 1000) | ~200³ region | (200, 200, 200) | 125 |

For point cloud stores that will be queried at many scales (e.g. by a
visualiser that pans and zooms), a smaller `bin_shape` (more bins) is
preferable. For batch analysis that reads entire volumes sequentially, a
larger `bin_shape` (fewer bins) reduces fragment index overhead.

### Effect on multiscale pyramids

At coarser resolution levels, the effective bin shape grows proportionally
with `bin_ratio`. This is the mechanism by which Zarr Vectors achieves spatial
downsampling: the bin grid at level 1 is half as fine as at level 0 (for
`bin_ratio = (2, 2, 2)`), so each bin at level 1 covers 8× the volume and
contains up to 8× as many merged vertices.

The `chunk_shape` never changes across levels. Only the bin grid changes.
This means the file layout (one file per chunk) is identical at all levels;
only the fragment index and vertex positions differ.

### Validation

L1: `base_bin_shape` is present in root `.zattrs` and is a list of D
positive numbers.

L2:
- `len(base_bin_shape) == spatial_dims`.
- All elements are strictly positive.
- `chunk_shape[d] % base_bin_shape[d] == 0` for all `d`.
- For each resolution level `N`: `effective_bin_shape[N][d] ≤ chunk_shape[d]`
  and `chunk_shape[d] % effective_bin_shape[N][d] == 0` for all `d`.
- `bin_ratio` values at every level are positive integers.

L3:
- At level 0 with the default writer (i.e. when
  `LevelMetadata.shared_fragments=False`), the range fragment count
  `R` in each chunk's `vertex_fragments/<chunk_coords>` blob equals
  that chunk's non-empty-bin count. `B_per_chunk =
  product(chunk_shape[d] / effective_bin_shape[N][d])` remains the
  upper bound on the total fragment count `F`.
- For every range fragment, `start + count ≤ vertex_count_in_chunk`.
  For every explicit fragment, all entries of the index list lie in
  `[0, vertex_count_in_chunk)`.
- At coarsened levels with `shared_fragments=True`, `F` is bounded
  by the number of distinct metavertices in the chunk and has no
  fixed relation to `B_per_chunk`.
