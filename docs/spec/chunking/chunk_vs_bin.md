# Chunk vs bin

## Terms

**I/O unit**
: The minimum amount of data that can be read from or written to storage in
  a single operation. In Zarr Vectors, the I/O unit is one chunk.

**Spatial query unit**
: The minimum spatial region that can be retrieved with a single index
  lookup. In Zarr Vectors, the spatial query unit is one bin.

**Read amplification**
: The ratio of data loaded from storage to data actually needed by a query.
  A query that intersects 6 bins but must load 2 whole chunks to access
  them has amplification proportional to `2 × chunk_volume / 6 × bin_volume`.

**Separation of concerns**
: The design principle behind Zarr Vectors' two-level spatial hierarchy. `chunk_shape`
  is chosen to optimise file I/O efficiency; `bin_shape` is chosen to
  optimise spatial query granularity. The two choices are made independently.

---

## Introduction

Earlier Zarr Vectors implementations (and many competing formats) used a single
spatial parameter — the chunk size — for both I/O and spatial indexing.
This forced an uncomfortable trade-off: small chunks gave fine query
granularity but poor I/O efficiency; large chunks gave efficient I/O but
coarse query granularity.

Zarr Vectors solves this by separating the two concerns. `chunk_shape` controls the
physical file layout and can be set large for I/O efficiency. `bin_shape`
controls the spatial index granularity and can be set small for query
precision. The two are independently configurable, subject only to the
divisibility constraint.

This separation was a deliberate architectural decision in the development
of `zarr-vectors-py` and represents one of the most important differences
between Zarr Vectors and simpler chunked formats.

---

## Technical reference

### Two-level spatial hierarchy

```
Space
 │
 ├─ Chunk grid (chunk_shape)          ← file layout, I/O unit
 │   ├─ Chunk (0,0,0) ─────────────────────────────────────────┐
 │   │   ├─ Bin grid (bin_shape)      ← spatial index, query unit │
 │   │   │   ├─ Bin (0,0,0) ──── fragment ──── vertex slice             │
 │   │   │   ├─ Bin (0,0,1) ──── fragment ──── vertex slice             │
 │   │   │   ├─ Bin (0,0,2) ──── fragment ──── vertex slice             │
 │   │   │   ├─ Bin (0,0,3) ──── fragment ──── vertex slice             │
 │   │   │   └─ … 60 more bins                                     │
 │   │   └── vertex_fragments ─────── fragment index               │
 │   └─────────────────────────────────────────────────────────────┘
 │   ├─ Chunk (0,0,1)
 │   └─ …
```

A chunk is fetched as a single I/O operation. Within the chunk, the
fragment index is used to locate exactly which vertex rows belong to
each fragment (and therefore to the query bbox). No client-side filtering
of all vertices in the chunk is required. See
[Fragment-index arrays](../layout/fragment_index_arrays.md) for the on-disk
layout.

### Worked example: query amplification

Suppose a 3-D point cloud store with:

```
total_volume  = 1000³ µm = 10⁹ µm³
chunk_shape   = (200, 200, 200)   → 5 × 5 × 5 = 125 chunks
bin_shape     = (50, 50, 50)      → 4 × 4 × 4 = 64 bins/chunk
total_bins    = 125 × 64 = 8000 bins
```

**Query A:** bbox of (50, 50, 50) → (100, 100, 100), a 50³ µm box.

- Number of overlapping bins: 1 (the query exactly fits one bin).
- Number of overlapping chunks: 1.
- Vertices loaded: ~1/8000 of total.
- Vertices used: ~1/8000 of total.
- Amplification: **1×** (ideal).

**Query B:** bbox of (0, 0, 0) → (200, 200, 200), an entire chunk.

- Number of overlapping bins: 64.
- Number of overlapping chunks: 1.
- Vertices loaded: ~1/125 of total.
- Vertices used: ~1/125 of total.
- Amplification: **1×** (ideal — query aligns with chunk boundary).

**Query C:** bbox of (150, 150, 150) → (250, 250, 250), straddling 8 chunks.

- Number of overlapping bins: 8 (one bin from each of the 8 straddled chunks).
- Number of overlapping chunks: 8.
- Vertices loaded: ~8/125 of total.
- Vertices used: ~8/8000 of total (one bin per chunk).
- Amplification: **64×** — we load 64 bins worth of data but use only 8.

This shows why `bin_shape` matters most at chunk boundaries. When a query
straddles many chunks but intersects only a small number of bins per chunk,
the fragment index prevents loading the entire contents of each chunk.

### The old design and why it was changed

In the pre-separation design, `chunk_shape` served as both the I/O unit and
the spatial query unit. A user choosing `chunk_shape` faced a direct
conflict: they wanted large chunks for I/O efficiency but small chunks for
query precision.

The separator change introduced `bin_shape` as an independent parameter,
with the fragment index providing the sub-chunk spatial resolution. The
`chunk_shape` could then be set based solely on I/O considerations (file
count, cloud request cost, bandwidth utilisation), while `bin_shape` was
set based solely on query requirements.

A secondary motivation was multi-resolution correctness. In the pre-
separation design, adding a coarser level required changing `chunk_shape`
at the coarser level — which would have required a different file layout
per level, complicating readers and cache strategies. With the separation,
`chunk_shape` is fixed across all levels and only `bin_shape` changes via
`bin_ratio`. The file layout is identical at every level; only the spatial
index granularity changes.

### When to use one bin per chunk

Setting `bin_shape = chunk_shape` (one bin per chunk, `B_per_chunk = 1`)
is appropriate when:

- All queries read entire chunks (no spatial sub-selection needed).
- The store will be processed sequentially without bbox queries.
- The data is so sparse that there is at most one vertex per chunk (the fragment
  index provides no benefit).
- You are writing a store for compatibility with a reader that does not
  understand the fragment index.

One-bin-per-chunk mode is the default when `bin_shape` is omitted. It is
fully supported and incurs no overhead beyond storing a trivial fragment index
(one entry per chunk with `offset=0, count=N_chunk`).

### When to use many bins per chunk

Fine binning (`B_per_chunk` of 64–512) is appropriate when:

- Queries are significantly smaller than one chunk volume.
- An interactive visualiser fetches data in viewport-sized bboxes.
- The data is dense and not all vertices in a chunk are relevant to a
  given query.

The overhead of a fine fragment index is modest: `B_per_chunk × 2 × 8 bytes` per
chunk (two int64 values per bin). For `B_per_chunk = 64`, this is
1 024 bytes per chunk — negligible compared to the vertex data itself.

### Relationship to multiscale pyramids

The chunk/bin separation is especially important for multi-resolution stores.
At coarser levels, `bin_ratio` increases `bin_shape` while `chunk_shape`
stays constant:

```
Level 0: chunk_shape=(200,200,200), bin_shape=(50,50,50)  → 64 bins/chunk
Level 1: chunk_shape=(200,200,200), bin_shape=(100,100,100) → 8 bins/chunk
Level 2: chunk_shape=(200,200,200), bin_shape=(200,200,200) → 1 bin/chunk
```

At level 2, each chunk has exactly one bin covering the full chunk volume.
All vertices in the chunk are merged into one coarse metanode. The chunk
file layout is identical to level 0; only the bin grid has changed.

This uniformity means a Neuroglancer-style viewer can use exactly the same
chunk-fetching logic at all levels, with no special cases for coarse vs
fine resolution.

### Summary comparison

| Property | `chunk_shape` | `bin_shape` |
|----------|--------------|-------------|
| Controls | File layout, I/O unit | Spatial index granularity, query unit |
| Changes across levels | Never | Yes (scales with `bin_ratio`) |
| Affects file count | Yes | No |
| Affects query precision | Indirectly | Directly |
| Can be changed post-write | Only by rechunking | In principle (re-index only) |
| Typical value (3-D, 200µm chunk) | (200, 200, 200) | (25, 25, 25) – (100, 100, 100) |
