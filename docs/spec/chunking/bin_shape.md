# Bin shape

## Terms

**Bin**
: A box of space inside a chunk. Bins form a regular grid aligned to the
  coordinate origin, like chunks, but finer.

**`base_bin_shape`**
: The bin edge lengths at level 0, a tuple of D positive floats in the
  root attributes. Optional; when absent, the bin shape is the chunk
  shape.

**Bin shape of a level**
: The bin edge lengths at a resolution level. Level 0 uses
  `base_bin_shape`. A coarser level has its own, recorded through its
  `bin_ratio`.

**`bin_ratio`**
: The per-axis factor by which a level's bin shape exceeds level 0's.
  Each value is at least 1.

**Bins per chunk**
: `chunk_shape[d] / bin_shape[d]` on each axis. With `chunk_shape = 200`
  and `bin_shape = 50`, a chunk holds 4 × 4 × 4 = 64 bins.

---

## Introduction

A chunk is the unit a reader fetches; a bin is the unit a reader selects
within it. At level 0 of a point cloud, the vertices of each bin are
stored together as one [fragment](../object_model/fragments.md), so a
bounding-box read decodes only the rows of the bins it touches.

`bin_shape` therefore sets the spatial granularity of reads inside a
chunk. At coarser levels it also sets the resolution: the coarsener
merges vertices bin by bin, and larger bins mean fewer, coarser
vertices. How bins and chunks interact is described in
[Chunk vs bin](chunk_vs_bin.md).

---

## Technical reference

### Declaration

Pass `bin_shape` to the writer:

```python
import numpy as np
from zarr_vectors.types.points import write_points

rng = np.random.default_rng(0)
pos = rng.uniform(0, 1000, (5000, 3)).astype("float32")
write_points("points.zarrvectors", pos,
             chunk_shape=(200., 200., 200.), bin_shape=(50., 50., 50.))
# {'vertex_count': 5000, 'chunk_count': 125, 'object_count': 0,
#  'group_count': 0, 'bins_per_chunk': (4, 4, 4)}
```

The root attributes then hold:

```json
{
  "zarr_vectors": {
    "chunk_shape": [200.0, 200.0, 200.0],
    "base_bin_shape": [50.0, 50.0, 50.0]
  }
}
```

When `bin_shape` is omitted, `base_bin_shape` is absent and each chunk is
one bin.

Level 0 has no bin shape of its own; it uses `base_bin_shape`. A coarser
level's bin shape and `bin_ratio` are recorded in the NGFF
`multiscales[0].datasets` entry for that level, not in the level's
attributes: the `scale` transform is the `bin_ratio`, and the
`translation` is half the bin shape. After
`build_pyramid(store, factors=[(2.0, 1.0)] * 3)` on the store above:

| Level | `scale` (`bin_ratio`) | `translation` | Bin shape |
|-------|-----------------------|---------------|-----------|
| 0 | `[1.0, 1.0, 1.0]` | `[25.0, 25.0, 25.0]` | 50 |
| 1 | `[2.0, 2.0, 2.0]` | `[50.0, 50.0, 50.0]` | 100 |
| 2 | `[4.0, 4.0, 4.0]` | `[100.0, 100.0, 100.0]` | 200 |
| 3 | `[8.0, 8.0, 8.0]` | `[200.0, 200.0, 200.0]` | 400 |

`zarr_vectors.core.store.read_level_metadata` returns these as
`bin_shape` and `bin_ratio`. See
[Multiscale metadata](../multiscale/multiscale_metadata.md) for the
transforms.

### Divisibility

`base_bin_shape` must divide `chunk_shape` on every axis, so bins tile
each chunk exactly:

```
chunk_shape[d] / base_bin_shape[d]   is a positive integer, for every d
```

`chunk_shape = 200` with `bin_shape = 50` is valid (4 bins per axis);
`bin_shape = 60` is not (3.33 bins per axis).

A coarser level's bin may be larger than its chunk: level 3 above has
400-unit bins in 200-unit chunks, and one bin then spans several chunks.
Where a level's bin is no larger than its chunk shape, it must divide
that chunk shape.

### Bin coordinates and fragments

Bins are aligned to the coordinate origin, not to the chunk. For a vertex
at `p` in the chunk with coordinate `c`:

```python
bins_per_chunk = (chunk_shape / bin_shape).round().astype(int)
bin_coord   = np.floor(p / bin_shape).astype(int)    # absolute
local       = bin_coord - c * bins_per_chunk         # 0 .. bins_per_chunk-1
fragment_ix = int(np.ravel_multi_index(local, bins_per_chunk))  # C order
```

At level 0 of a point cloud written without object ids, fragment
`fragment_ix` of a chunk holds the vertices of that bin, and every chunk
has `prod(bins_per_chunk)` fragments, empty bins included. Written with
object ids, a point cloud has one fragment per object per chunk instead,
and the bin grid does not set the fragments. Other geometry types use
fragments for objects or segments; see
[Fragments](../object_model/fragments.md).

### Changing the bin shape

`zarr_vectors.rechunk.rebin_level` re-bins one level of a point-cloud
store in place; see [Rechunking](rechunking.md#re-binning-a-level).

### Validation

[`zarr_vectors.validate`](../validation/overview.md) checks bins at these
levels:

| Level | Check | Result |
|-------|-------|--------|
| 2 (metadata) | `base_bin_shape`, when present, has D entries, each `> 0`, and divides `chunk_shape` on every axis (tolerance 1e-9 on the ratio) | error |
| 2 (metadata) | On a level with its own `chunk_shape`, the level's bin, where not larger, divides that chunk shape | error |
| 3 (consistency) | On a point-cloud-only level without `object_index`, with more than one bin per chunk: no chunk has more fragments than bins | error |
| 3 (consistency) | Same levels, first three chunks: every vertex of fragment `k` lies inside bin `k` (tolerance 1e-4) | warning |

These level 3 checks take the bin shape of level 0 from `base_bin_shape`;
coarser levels are not bin-checked.
