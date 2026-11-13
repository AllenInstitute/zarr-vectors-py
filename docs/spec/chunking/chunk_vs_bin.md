# Chunk vs bin

## Terms

**I/O unit**
: What a reader fetches from storage in one request: one chunk's cell of
  a per-chunk array. Set by [`chunk_shape`](chunk_shape.md).

**Selection unit**
: What a reader can pick out of a fetched chunk without testing every
  vertex: one bin's fragment. Set by [`bin_shape`](bin_shape.md).

---

## Introduction

A Zarr Vectors store has two nested grids. Chunks decide how data is
split into files and requests. Bins, inside each chunk, decide how
finely a reader can select vertices once the chunk is fetched, and how
coarsely the pyramid merges vertices at lower resolutions.

The two are set separately because they answer different questions.
Chunk size is about storage: file counts, request counts and request
sizes. Bin size is about queries and resolution. This page shows how
the two combine. Picking values is covered in
[Choose chunk and bin sizes](../../how_to/choose_chunk_and_bin.md).

---

## Technical reference

### Two nested grids

```
Chunk grid (chunk_shape)             one cell per chunk in each per-chunk array
 └─ chunk (i, j, k)
     ├─ vertices                      every vertex of the chunk, bin by bin
     ├─ vertex_fragments              fragment index: rows of each fragment
     └─ bin grid (bin_shape)          fragment k = vertices of local bin k
```

A bounding-box read of points at level 0:

1. Finds the non-empty chunks the box touches and fetches their
   `vertices` and `vertex_fragments` cells.
2. In each chunk, computes the local bin range the box covers and decodes
   only those fragments.
3. Keeps the vertices that lie inside the box.

Step 2 is skipped, and every fragment of a touched chunk is decoded,
when the store has one bin per chunk or the level is
[chunked by attribute](attribute_chunking.md).

### Worked example

A point cloud of 5,000 points in `[0, 1000)³`, `chunk_shape = 200`,
`bin_shape = 50`: 125 chunks of 64 bins each.

| Query box | Chunks fetched | Bins decoded per chunk |
|-----------|----------------|------------------------|
| `(50, 50, 50)–(100, 100, 100)` | 1 | up to 8 (the box's bounds touch bins 1 and 2 on each axis) |
| `(0, 0, 0)–(200, 200, 200)` | up to 8 (the upper bound 200 touches the next chunks) | all 64 in chunk (0, 0, 0) |
| `(150, 150, 150)–(250, 250, 250)` | 8 | up to 8 per chunk, of 64 |

In the third query each chunk is fetched whole but only its bins near the
box are decoded and tested. With `bin_shape = chunk_shape` the same
query would decode and test all vertices of the eight chunks.

### Across pyramid levels

Each coarser level has a larger bin shape (its `bin_ratio` times
`base_bin_shape`). By default every level keeps the root
`chunk_shape`; `chunk_scale_factors` in `build_pyramid` gives a level
larger chunks. With `chunk_shape = 200`, `bin_shape = 50` and three
levels coarsened by 2:

| Level | `chunk_shape` | Bin shape | Bins per chunk |
|-------|---------------|-----------|----------------|
| 0 | 200 | 50 | 4 × 4 × 4 |
| 1 | 200 | 100 | 2 × 2 × 2 |
| 2 | 200 | 200 | 1 |
| 3 | 200 | 400 | one bin spans 2 × 2 × 2 chunks |

At coarser levels the coarsener writes one vertex per object per bin, so
the bin shape sets the level's resolution. See
[Pyramid construction](../multiscale/pyramid_construction.md).

### Summary

| Property | `chunk_shape` | `bin_shape` |
|----------|---------------|-------------|
| Controls | Files, requests, the read unit | Selection inside a chunk; resolution of coarse levels |
| Stored in | Root `zarr_vectors.chunk_shape`; optional per-level override | Root `zarr_vectors.base_bin_shape`; per-level NGFF transforms |
| Default | Required | Equal to `chunk_shape` |
| Across levels | Root value, or an integer multiple of it per level | Grows with `bin_ratio` |
| Changed after writing by | `rechunk_spatial` (new store) | `rebin_level` (point clouds, in place) |
