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
  each chunk exactly, with no gaps or overlaps. In an undifferentiated point
  cloud, all vertices whose position falls within a bin are stored together as
  one fragment. (Other geometry types do not fragment by bin; see
  [Where bins are used](#where-bins-are-used).)

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
: The bin shape at the full-resolution level (level 0). Stored in the root
  attributes under `zarr_vectors.base_bin_shape`; absent means "equal to
  `chunk_shape`". Serves as the reference from which all coarser-level bin
  shapes are derived.

**Divisibility constraint**
: The requirement that `chunk_shape[d] / base_bin_shape[d]` is an integer for
  all `d`. Ensures level-0 bins tile chunks exactly, which is necessary for
  bin ↔ fragment indexing to be well-defined. Checked by the L2 validator, by
  `rebin_level` and by `add_resolution_level`; the point-cloud writer itself
  rounds `chunk_shape / bin_shape` and does not check.

---

## Introduction

Bins are the unit of sub-chunk spatial querying for point clouds. When you issue
a bounding-box query against a point-cloud store with several bins per chunk,
the reader still fetches each overlapping chunk, but it decodes only the
fragments of the bins that overlap the query region. This sub-chunk resolution
is what keeps small queries on large chunks cheap.

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

### Where bins are used

Only **undifferentiated point clouds** (`write_points` without `object_ids`,
`object_attributes` or `groups`) fragment by bin; there, fragment `f` of a chunk
is the bin with row-major index `f` (see [Fragments](../object_model/fragments.md)).
Point clouds written with objects, polylines, lines, graphs, skeletons and
meshes accept `bin_shape` and record it as `base_bin_shape`, but their writers
split at chunk boundaries only and emit one fragment per object per chunk. For
those types `bin_shape` still sets the **coarsening grid** of a pyramid (bin
centroids become metavertices) and the NGFF scale of each level. It does not
subdivide chunks on disk, and `rebin_level` refuses them.

### Declaration and storage

`base_bin_shape` is declared in the root attributes' `zarr_vectors` block:

```json
{
  "zarr_vectors": {
    "chunk_shape":    [200.0, 200.0, 200.0],
    "base_bin_shape": [50.0,  50.0,  50.0]
  }
}
```

The effective bin shape of each level is **not** stored in the level's own
metadata. It is the level's NGFF transform in the root `multiscales` block:

```json
{"path": "1", "coordinateTransformations": [
  {"type": "scale",       "scale":       [2.0, 2.0, 2.0]},
  {"type": "translation", "translation": [50.0, 50.0, 50.0]}
]}
```

so that `bin_ratio = scale` and `bin_shape = 2 × translation`, and by
construction

```
bin_shape[N][d] = base_bin_shape[d] × bin_ratio[N][d]
```

(`read_level_metadata` returns both; level 0 reports `None` and uses
`base_bin_shape`.) A level's scale is relative to level 0; the `coarsen_factor`
you pass to `coarsen_level` is relative to the *parent* level and compounds.

### Divisibility constraint

For level-0 bins to tile chunks exactly, `base_bin_shape` must divide
`chunk_shape` evenly in every dimension:

```
chunk_shape[d] / base_bin_shape[d]  must be a positive integer,  for all d
```

Tolerance is `1e-9` on the ratio. The violation is reported as a
`MetadataError` by `validate_bin_shape_divides_chunk` (called by
`rebin_level` and `add_resolution_level`) and as an L2 error by the validator;
`write_points` does not itself check it, so an invalid pair surfaces later.

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

The constraint is **not** required of coarser levels. A coarse level's bin is
routinely *larger* than the chunk it sits in (bin 200 in a chunk of 100: one
bin spans two chunks), and the library's own coarsener produces such levels.
The check is applied to a coarser level only if it overrides `chunk_shape`, and
then only on axes where the bin is no larger than the chunk.

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

If `bin_shape` is not specified at write time, the effective bin shape is
`chunk_shape` and **no `base_bin_shape` key is written**. This results in
exactly one bin per chunk (`B_per_chunk = 1`), which is equivalent to a purely
chunk-indexed store without sub-chunk spatial indexing: each chunk has a single
fragment `(0, N)`. A bbox query on such a store degrades gracefully to
chunk-granularity queries.

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

At coarser resolution levels, the effective bin shape grows with the level's
`bin_ratio`. This is the mechanism by which Zarr Vectors achieves spatial
downsampling: with `bin_ratio = (2, 2, 2)` each bin at level 1 covers 8× the
volume, and the coarsener replaces the vertices of an object that fall in one
bin by a single centroid (a metavertex).

The chunk grid does not have to change with the bin grid. By default every
level inherits the root `chunk_shape`, so the file layout is identical at all
levels and only the fragment index and vertex positions differ. A level may
instead scale its chunks by an integer factor (`chunk_scale_factor`), keeping
the grids nested; its bins are then measured against that larger chunk.

### Re-binning

`zarr_vectors.rechunk.rebin_level(store, target_bin_shape, level=0)` changes the
bin grid of a **point-cloud** store in place without moving chunk files. See
[Rechunking](rechunking.md#rechunking-only-bin_shape) for its side effects
(attributes are not rewritten; fragments become occupied-bins-only).

### Validation

L1: the root `zarr_vectors` block exists.

L2:
- If `base_bin_shape` is present: length `D`, all elements strictly positive,
  and `chunk_shape[d] / base_bin_shape[d]` an integer on every axis.
- Per level (when the level carries `bin_shape` / `bin_ratio` in its raw
  attributes, which current writers do not): `bin_ratio >= 1` and the bin
  divides the effective chunk shape.
- A level that overrides `chunk_shape` has nested chunks and, where the bin is
  no larger than the chunk, bins that tile it.

L3 (undifferentiated point clouds only, and only when `base_bin_shape` yields
more than one bin per chunk):
- a chunk has no more fragments than `B_per_chunk`;
- for the first three chunks checked, points of fragment `f` lie inside the bin
  `fragment_index_to_bin(f, …)` (a warning, with a `1e-4` tolerance).

Writer invariants that the validator does not check individually: a range
fragment satisfies `start + count <= rows in the chunk` and an explicit
fragment's indices lie in `[0, rows)`. When `shared_fragments=True`, `F` is
bounded by the number of distinct metavertices and has no fixed relation to
`B_per_chunk`.
