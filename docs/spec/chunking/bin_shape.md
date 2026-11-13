# Bin shape

## Terms

**Bin**
: A box of space inside a chunk. Bins form a regular grid aligned to the
  coordinate origin, like chunks, but finer.

**`base_bin_shape`**
: The bin edge lengths at level 0, a tuple of D positive floats in the
  root attributes. Optional; when absent, the bin shape is the chunk
  shape.

**`bin_ratio`**
: The per-axis factor, at least 1, by which a level's bin shape exceeds
  level 0's.

**Bins per chunk**
: `chunk_shape[d] / bin_shape[d]` on each axis. With `chunk_shape = 200`
  and `bin_shape = 50`, a chunk holds 4 × 4 × 4 = 64 bins.

---

## Introduction

A [chunk](chunk_shape.md) is the unit a reader fetches; a bin is the unit
a reader selects within it. At level 0 of a point cloud, each bin's
vertices are stored together as one
[fragment](../object_model/fragments.md). A bounding-box read fetches the
chunks the box touches, decodes only the fragments of the bins it
covers, then keeps the vertices inside the box. With one bin per chunk,
or on an [attribute-chunked](attribute_chunking.md) level, it decodes
every fragment of each chunk.

At coarser levels the bin shape also sets resolution: the coarsener
writes one vertex per object per bin.

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
# {'vertex_count': 5000, 'chunk_count': 125, ..., 'bins_per_chunk': (4, 4, 4)}
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

Without `bin_shape`, `base_bin_shape` is absent and each chunk is one bin.

A coarser level's bin shape is recorded in the NGFF
`multiscales[0].datasets` entry for that level, not in the level's
attributes: the `scale` transform is the `bin_ratio`, and the
`translation` is half the bin shape. After
`build_pyramid(store, factors=[(2.0, 1.0)] * 3)` on the store above:

| Level | `scale` (`bin_ratio`) | `translation` | Bin shape | Bins per chunk |
|-------|-----------------------|---------------|-----------|----------------|
| 0 | `[1.0, 1.0, 1.0]` | `[25.0, 25.0, 25.0]` | 50 | 4 × 4 × 4 |
| 1 | `[2.0, 2.0, 2.0]` | `[50.0, 50.0, 50.0]` | 100 | 2 × 2 × 2 |
| 2 | `[4.0, 4.0, 4.0]` | `[100.0, 100.0, 100.0]` | 200 | 1 |
| 3 | `[8.0, 8.0, 8.0]` | `[200.0, 200.0, 200.0]` | 400 | one bin spans 2 × 2 × 2 chunks |

`zarr_vectors.core.store.read_level_metadata` returns `bin_shape` and
`bin_ratio`; see [Multiscale metadata](../multiscale/multiscale_metadata.md).

### Divisibility

`base_bin_shape` must divide `chunk_shape` on every axis, so bins tile
each chunk exactly. `chunk_shape = 200` with `bin_shape = 50` is valid;
`bin_shape = 60` is not.

A coarser level's bin may be larger than its chunk (level 3 above).
Where it is not, it must divide the chunk shape.

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
`fragment_ix` holds that bin's vertices, and every chunk has
`prod(bins_per_chunk)` fragments, empty bins included. With object ids, a
point cloud has one fragment per object per chunk instead. Other geometry
types use fragments for objects or segments; see
[Fragments](../object_model/fragments.md).

`zarr_vectors.rechunk.rebin_level` changes one level's bin shape in
place; see [Rechunking](rechunking.md#re-binning-a-level).

### Validation

[`zarr_vectors.validate`](../validation/overview.md) checks:

| Level | Check | Result |
|-------|-------|--------|
| 2 | `base_bin_shape`, when present, has D entries, each `> 0`, and divides `chunk_shape` on every axis (tolerance 1e-9 on the ratio) | error |
| 2 | On a level with its own `chunk_shape`, the level's bin, where not larger, divides that chunk shape | error |
| 3 | On a point-cloud-only level without `object_index`, with more than one bin per chunk: no chunk has more fragments than bins | error |
| 3 | Same levels, first three chunks: every vertex of fragment `k` lies inside bin `k` (tolerance 1e-4) | warning |

The level 3 checks use level 0's bin shape; coarser levels are not
bin-checked.
