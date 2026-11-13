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
: The block of chunk coordinates a level allocates, from its **grid
  origin** `floor(lo / chunk_shape)` to `floor(hi / chunk_shape)`
  inclusive, where `lo` and `hi` are the store's `bounds`.

**Storage key**
: The path of one chunk's cell in a per-chunk array, for example
  `0/vertices/c/1/0/3`. Its indices are the chunk coordinate minus the
  grid origin.

---

## Introduction

A Zarr Vectors store cuts space into a regular grid of boxes called
chunks. Every vertex is stored in the chunk that contains it, and a chunk
is the unit of I/O: a read fetches whole chunks, and a write produces
whole chunks. `chunk_shape` sets the size of those boxes.

Large chunks mean fewer files and fewer requests, but a small query
fetches data it then discards. Small chunks fit small queries but
multiply files and requests. Inside a chunk, a finer grid of
[bins](bin_shape.md) lets a reader pick out the vertices a query needs;
[Chunk vs bin](chunk_vs_bin.md) compares the two.
[Choose chunk and bin sizes](../../how_to/choose_chunk_and_bin.md) covers
picking values.

---

## Technical reference

### Declaration

`chunk_shape` is a required key of the `zarr_vectors` block in the root
attributes:

```json
{
  "zarr_vectors": {
    "chunk_shape": [200.0, 200.0, 200.0],
    "bounds": [[-149.9, -149.8, -149.9], [449.9, 449.8, 449.9]]
  }
}
```

The unit is the coordinate unit of the vertices. `chunk_shape` need not
be isotropic. Its length is the spatial dimensionality D of the store.

A level other than 0 may carry its own `chunk_shape` in the level's
attributes (`zarr_vectors_level.chunk_shape`). Each axis must be a
positive integer multiple of the root `chunk_shape`, so coarse chunks
nest exactly over fine ones. `build_pyramid(..., chunk_scale_factors=[2, 1])`
writes such levels: with a root `chunk_shape` of 200, levels 1 and 2
record `"chunk_shape": [400.0, 400.0, 400.0]`. A level without the key
uses the root value. See
[Pyramid construction](../multiscale/pyramid_construction.md).

### Chunk grid and storage keys

The writer derives the grid from the root `bounds`, which are inclusive:

```
origin[d] = floor(lo[d] / chunk_shape[d])
shape[d]  = floor(hi[d] / chunk_shape[d]) - origin[d] + 1
```

Worked example: 2,000 points uniform in `[-150, 450)³` with
`chunk_shape = 200`. Then `origin = floor(-0.75) = -1` and
`shape = floor(2.25) - (-1) + 1 = 4`, so the grid is 4 × 4 × 4 chunks
covering chunk coordinates −1 to 2 on each axis.

Each per-chunk array (for example `vertices`) is one Zarr array whose
shape is the chunk grid and whose Zarr chunks are single cells. Each
cell holds one chunk's bytes. The array's `zarr.json` for the example:

```json
{
  "shape": [4, 4, 4],
  "data_type": "variable_length_bytes",
  "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1, 1, 1]}},
  "codecs": [{"name": "vlen-bytes", "configuration": {}}],
  "fill_value": "",
  "attributes": {
    "nonempty_chunks": ["-1.-1.-1", "-1.-1.0", "-1.-1.1", "..."],
    "chunk_grid_origin": [-1, -1, -1],
    "zv_array": "vertices",
    "dtype": "float32",
    "encoding": "raw"
  }
}
```

- `chunk_grid_origin` is present only when the origin is not all zeros.
- `nonempty_chunks` lists the absolute chunk coordinates that hold data,
  dot-separated.
- The Zarr chunk shape `[1, 1, 1]` is not `chunk_shape`: the first counts
  array cells, the second is a length in world units.

The chunk at absolute coordinate `(-1, -1, 2)` is stored at
`0/vertices/c/0/0/3`. Empty chunks have no file.

A level chunked by attribute has leading array axes before the spatial
ones; see [Attribute chunking](attribute_chunking.md#array-shape).

### Changing the chunk shape

`chunk_shape` is fixed for a written store. To change it, write a new
store with `rechunk_spatial`; see [Rechunking](rechunking.md).

### Validation

[`zarr_vectors.validate`](../validation/overview.md) checks the chunk
shape at these levels:

| Level | Check | Result |
|-------|-------|--------|
| 1 (structure) | The root has a `zarr_vectors` block; the per-chunk arrays of a level have the same rank | error |
| 2 (metadata) | `chunk_shape` has D entries, each `> 0` | error |
| 2 (metadata) | A per-level `chunk_shape` is a positive integer multiple of the root `chunk_shape` on every axis, and the level's bin shape, where not larger than that chunk shape, divides it | error |
| 3 (consistency) | Every chunk of a level decodes, and each fragment has D columns | error |

Divisibility of `chunk_shape` by `base_bin_shape` is covered in
[Bin shape](bin_shape.md#validation).
