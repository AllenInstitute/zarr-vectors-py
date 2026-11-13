# Chunk shape

## Terms

**`chunk_shape`**
: A tuple of D positive floats, in the store's coordinate units, giving
  the edge lengths of one chunk. The chunk at absolute coordinate
  `(i_0, …, i_{D-1})` covers `[i_d × chunk_shape[d], (i_d + 1) × chunk_shape[d])`
  on each axis `d`.

**Chunk coordinate**
: The absolute integer coordinate of a chunk: `floor(p[d] / chunk_shape[d])`
  for a vertex at position `p`. It can be negative.

**Chunk grid**
: The block of chunk coordinates a level allocates, set by the store's
  `bounds`. Its first coordinate is the **grid origin**.

**Storage key**
: The path of one chunk's cell in a per-chunk array, for example
  `0/vertices/c/1/0/3`. Its indices are the chunk coordinate minus the
  grid origin.

---

## Introduction

A store cuts space into a regular grid of chunks. Each vertex is stored
in the chunk that contains it, and reads and writes work on whole
chunks. Large chunks mean fewer files and requests; small chunks suit
small queries. [Bins](bin_shape.md) subdivide a chunk.
[Choose chunk and bin sizes](../../how_to/choose_chunk_and_bin.md)
covers picking values.

---

## Technical reference

### Declaration

`chunk_shape` is a required key of the root `zarr_vectors` block, for
example `"chunk_shape": [200.0, 200.0, 200.0]`. Its unit is the
vertices' coordinate unit and its length is the spatial dimensionality
D. It need not be isotropic.

A level other than 0 may record its own `zarr_vectors_level.chunk_shape`,
a positive integer multiple of the root value on each axis, so coarse
chunks nest over fine ones. `build_pyramid(..., chunk_scale_factors=...)`
writes these; see [Pyramid construction](../multiscale/pyramid_construction.md).
Other levels use the root value.

### Chunk grid and storage keys

The writer derives the grid from the root `bounds`, which are inclusive:

```
origin[d] = floor(lo[d] / chunk_shape[d])
shape[d]  = floor(hi[d] / chunk_shape[d]) - origin[d] + 1
```

For bounds of about −150 to 450 on each axis and `chunk_shape = 200`:
`origin = -1` and `shape = 2 - (-1) + 1 = 4`, so the grid covers chunk
coordinates −1 to 2.

Each per-chunk array (for example `vertices`) is one Zarr array whose
shape is the chunk grid and whose Zarr chunks are single cells. Its
`zarr.json` includes:

```javascript
{
  "shape": [4, 4, 4],
  "data_type": "variable_length_bytes",
  "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1, 1, 1]}},
  "attributes": {
    "nonempty_chunks": ["-1.-1.-1", "-1.-1.0", "-1.-1.1", "..."],
    "chunk_grid_origin": [-1, -1, -1],
    "zv_array": "vertices"
  }
}
```

`nonempty_chunks` lists the absolute coordinates of chunks that hold
data. `chunk_grid_origin` is present only when the origin is not all
zeros. The Zarr chunk shape `[1, 1, 1]` counts array cells, unlike
`chunk_shape`. The chunk at absolute coordinate `(-1, -1, 2)` is
stored at `0/vertices/c/0/0/3`. Empty chunks have no file. A level
chunked by attribute has extra leading axes; see
[Attribute chunking](attribute_chunking.md#array-shape).

`chunk_shape` is fixed for a written store. To change it, write a new
store with `rechunk_spatial`; see [Rechunking](rechunking.md).

### Validation

[`zarr_vectors.validate`](../validation/overview.md) checks:

| Level | Check | Result |
|-------|-------|--------|
| 1 | The root has a `zarr_vectors` block; a level's per-chunk arrays have the same rank | error |
| 2 | `chunk_shape` has D entries, each `> 0` | error |
| 2 | A per-level `chunk_shape` is a positive integer multiple of the root value on every axis | error |
| 3 | Every chunk of a level decodes, and each fragment has D columns | error |

Bin checks are in [Bin shape](bin_shape.md#validation).
