# Sharding

## Terms

**Sharding codec** (`sharding_indexed`)
: A Zarr v3 built-in codec that packs multiple logical *inner chunks* into a
  single storage key called a *shard*. The shard stores an index at the end
  of the file that maps each inner chunk's coordinate to its byte offset and
  length within the shard.

**Shard**
: The storage unit when using the sharding codec. One shard corresponds to
  one key in the underlying store (one file on disk, one object in S3).
  A shard contains multiple inner chunks.

**Inner chunk**
: The logical unit of data within a shard, equivalent to a normal Zarr v3
  chunk. In the Zarr Vectors context, the inner chunk is one array **cell**,
  i.e. one Zarr Vectors spatial chunk: the inner chunk shape is always
  `(1, 1, …, 1)` because the array's shape already *is* the chunk grid (see
  [Chunk arrays](../layout/chunk_arrays.md)).

**Outer chunk** (shard shape)
: The number of inner chunks grouped into one shard, expressed as a tuple.
  An outer chunk of `(4, 4, 4)` means each shard contains `4 × 4 × 4 = 64`
  inner chunks (Zarr Vectors spatial chunks).

**Shard index**
: A fixed-size lookup table appended to the end of each shard file that
  maps each inner chunk coordinate to its byte range within the shard.
  Enables retrieving a single inner chunk without reading the whole shard.

---

## Introduction

The sharding codec addresses a common problem in cloud-native workflows:
a fine `chunk_shape` chosen for query performance results in a very large
number of small files (or S3 objects), which is expensive to manage and
slow to list.

With sharding, you keep the logical Zarr Vectors `chunk_shape` small (good for
queries), but multiple Zarr Vectors chunks are packed into a single shard file. A
reader fetching one Zarr Vectors chunk makes one HTTP request to the shard file (or
uses HTTP range requests to fetch only the relevant byte range), exactly
as if each chunk were a separate file — but the object count is reduced by
the shard factor.

Sharding is available natively in Zarr v3 and does not require changes to
the Zarr Vectors data model. It is purely a storage-layer optimisation. The fragment index,
object model, and multiscale metadata are identical with or without sharding.

---

## Technical reference

### When to use sharding

| Scenario | Use sharding? |
|----------|--------------|
| Local file system, modest chunk count (<1M chunks) | No — unnecessary overhead |
| Local file system, very many small chunks (>1M) | Consider — reduces inode pressure |
| Cloud object store (S3/GCS), small chunk_shape | Yes — significantly reduces request cost |
| Neuroglancer serving from S3 | Yes — reduces GET request cost |
| Read-heavy workflows with many parallel readers | Yes — shard index enables range requests |
| Write-heavy workflows with many parallel writers | No — shard contention hurts write throughput |

### Configuration

Declare it on the store. `create_store(shard_shape=N)` records the shape
in the root `zarr_vectors` block, and **every array allocated afterwards
honours it** — including the ones created lazily, on first write, inside
workers that never saw the argument:

```python
from zarr_vectors.building import create_store

create_store(
    "scan.zarrvectors",
    bounds=[[0, 0, 0], [1000, 1000, 1000]],
    chunk_shape=(100.0, 100.0, 100.0),
    geometry_types=["point_cloud"],
    shard_shape=4,          # 4 cells per shard edge, every axis
)
```

The units are grid **cells**, not coordinates, so one declaration stays
meaningful at every level of a pyramid even though their grids differ. An
`int` broadcasts to every axis; pass a per-axis list when they differ.

Without the declaration, sharding reaches only the arrays a given writer
call creates. That was the whole story before 0.9.3, and it produced
stores that were sharded in whatever the coordinator made and flat in
everything a worker added — a half-sharded store whose object count was
nowhere near what was asked for.

The per-call argument still works and still wins, for a writer that wants
a different packing than the store's. Every `write_*` function
(`write_points`, `write_polylines`, `write_lines`, `write_graph`,
`write_mesh`, …) takes **`shard_shape=`**, and `shard_store` applies it
afterwards to an existing store. It is an `int` (broadcast to every axis)
or a per-axis tuple, in units of cells:

```python
from zarr_vectors.types.points import write_points

write_points(
    "scan.zarrvectors",
    positions,
    chunk_shape=(100.0, 100.0, 100.0),   # the logical chunk — one inner chunk
    bin_shape=(25.0, 25.0, 25.0),
    shard_shape=(4, 4, 4),               # 4×4×4 = 64 inner chunks per shard
)
```

This produces one shard object per `(4, 4, 4)` block of the chunk grid, each
holding up to 64 Zarr Vectors chunks. The number of shard objects per array is
about `ceil(grid_shape[d] / 4)` per axis — a factor of up to 64 fewer objects
than without sharding. `shard_shape` must have the same rank as the grid
(which has one extra leading axis in an attribute-chunked store). The default,
`shard_shape=None`, is unsharded: one storage object per non-empty cell.

A separate `compressor=` keyword (`"zstd"`, `"blosc"` or a codec list) controls
compression of the cells; it is independent of sharding and defaults to none.

### Codec pipeline with sharding

When a shard shape is given, each per-chunk array's `zarr.json` has the
`sharding_indexed` codec as its only codec; the `vlen-bytes` serializer (and a
compressor, if requested) move *inside* it. For a 3-D array with
`shard_shape=(4, 4, 4)` (as zarr-python 3 writes it):

```json
{
  "shape": [12, 12, 8],
  "data_type": "variable_length_bytes",
  "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4, 4, 4]}},
  "codecs": [
    {
      "name": "sharding_indexed",
      "configuration": {
        "chunk_shape": [1, 1, 1],
        "codecs": [ {"name": "vlen-bytes", "configuration": {}} ],
        "index_codecs": [
          {"name": "bytes", "configuration": {"endian": "little"}},
          {"name": "crc32c"}
        ],
        "index_location": "end"
      }
    }
  ]
}
```

The outer `chunk_grid` is the **shard** shape; the `chunk_shape` inside the
sharding configuration is the **inner** shape — always one cell.

### Relationship between shard shape and Zarr Vectors chunk shape

The Zarr Vectors `chunk_shape` (physical units) maps to the *inner* chunk of the
sharding codec. The shard shape (outer chunk) is expressed in inner-chunk
units (integers), not in physical units:

```
Zarr Vectors chunk_shape = (100, 100, 100) µm
shard_shape     = (4, 4, 4) inner chunks
→ each shard covers (400, 400, 400) µm of physical space
```

Readers that understand the shard index can fetch a single Zarr Vectors chunk with
one HTTP range request; readers that do not understand sharding must fetch
the entire shard file. `zarr-vectors-py` always uses shard-indexed reads
when the sharding codec is present.

### Read behaviour with sharding

When reading a single Zarr Vectors chunk from a sharded store:

1. Identify the shard containing the requested inner chunk.
2. Fetch only the shard index (last `n_inner_chunks × 16 bytes` of the
   shard file, via an HTTP range request for the tail of the object).
3. Look up the byte offset and length of the requested inner chunk.
4. Fetch the inner chunk data (second HTTP range request).

Total: **2 HTTP requests** for one Zarr Vectors chunk, regardless of shard size.
Without sharding, it is **1 HTTP request** per chunk but many more objects.

### Write behaviour with sharding

The shard is the unit of storage, so updating one cell of a sharded array
rewrites its shard object (a read-modify-write of up to `product(shard_shape)`
cells). Two workers that write different cells of the **same** shard can
therefore lose each other's updates. For parallel writes (HPC, multi-process),
assign non-overlapping shard ranges to different workers — or write unsharded
and run `shard_store` once at the end (see
[Ordering](#ordering-shard-the-links-family-last)).

### Choosing shard shape

A good shard shape balances:

- **Shard file size:** target 10–100 MB per shard. Smaller shards offer
  finer parallelism; larger shards reduce object count more aggressively.
- **Read efficiency:** if most queries read entire shards, a large shard
  shape saves requests. If most queries read only 1–2 inner chunks per
  shard, a small shard shape minimises wasted range-request bytes.
- **Write parallelism:** shards are the unit of write contention. If you need
  exactly one object per Zarr Vectors chunk, leave sharding off
  (`shard_shape=None`) rather than using `(1, 1, 1)`, which is a degenerate
  sharded layout with an index per cell.

A practical default for Neuroglancer-style serving:

```python
shard_shape = (8, 8, 8)   # 512 inner chunks per shard, ~25–200 MB per shard
```

### Sharding and the fragment index

The `vertex_fragments/` and `link_fragments/` arrays benefit especially
from sharding. Unlike `vertices/`, which may have large chunks, fragment-
index chunks are small (typically tens to a few hundred bytes per Zarr Vectors
spatial chunk — header plus a short range table plus a tiny CSR when
fragments are explicit). Without sharding, a large store creates millions
of tiny files for the fragment indices alone. With sharding, many
fragment-index chunks are packed into a single shard, and a viewer can
fetch the entire spatial index for a region in a handful of requests.

### Sharding the object layer

The object layer -- `object_index/`'s arrays (`manifests`, or the dense
`manifest_spans` / `manifest_blocks`, and `object_ids`) and every
`object_attributes/<name>` column -- has no spatial grid: each array is 1-D
along objects, chunked in fixed row buckets of 16,384 (index: `manifests`,
`manifest_spans`, `object_ids`) or 65,536 (a dense index's `manifest_blocks`
as created now, and attribute columns) rows, one storage object per bucket.
At 10^9 objects that is about 61,000 objects per index array and 15,000 per
attribute column.

These arrays may be sharded along rows with the same `sharding_indexed`
codec, the shard a whole number of buckets. zarr-vectors-py requires the
shard to hold a multiple of 65,536 rows, so both buckets tile it and every
array in the layer can share one shard size. This is not a format change:
the sharding is recorded only in each array's own `zarr.json`, which any
Zarr v3 reader already understands, and no zarr-vectors metadata key names
it. A reader reads a sharded object layer exactly as an unsharded one.

zarr-vectors-py keeps a level's object layer in one layout: a writer that
creates or rewrites an object-layer array (a new attribute column, an index
rewritten in full) takes the shard size the level's existing object arrays
have.

```python
from zarr_vectors.building import shard_object_layer, shard_store

shard_object_layer(level, 1_048_576)        # 16 attribute buckets per shard
shard_object_layer(level, None)             # back to one object per bucket
shard_store("scan.zv", shard_shape=8, object_shard_rows=1_048_576)
```

A write into part of a shard rewrites that shard, so a writer appending in
small batches pays for the whole shard each time: batch appends in shard-
sized blocks. For the same reason, the shard (not the bucket) is the unit of
concurrency for writers that fill disjoint row ranges in parallel.

### Compatibility

`sharding_indexed` is a core Zarr v3 codec, so any Zarr v3 implementation that
supports it (zarr-python 3.x, zarrs, tensorstore, …) can open a sharded store.
`zarr-vectors-py` itself creates sharded arrays through zarr-python's `shards=`
argument, which needs zarr-python 3.2 or later. Consumers on older
implementations will fail to open sharded arrays; record the requirement for
them in your own metadata namespace (see
[Root metadata](../layout/root_metadata.md#application-metadata)). The
library's own reader does not need to be told: `is_sharded` and
`get_shard_info` inspect the arrays.

### Implementation notes (zarr-vectors-py)

* Each Zarr Vectors logical array (`vertices`, `vertex_fragments`,
  `links/<delta>/<offsets>`, `link_fragments`,
  `link_attributes/<name>/<delta>/<offsets>`, attribute arrays, ...) maps
  to a single Zarr v3 vlen-bytes array whose shape is the level's chunk
  grid. One cell of that array holds one Zarr Vectors spatial chunk's payload
  bytes; absent chunks are vlen-bytes `b""` (the codec's fill value).

  ```{note}
  Before 0.9.0 this page claimed the same of `cross_chunk_links/<delta>/`,
  which was **false**: that family's cells were keyed by endpoint-chunk
  tuples, not a spatial grid, so it had no chunk-grid shape to shard and
  was silently excluded from `shard_store`. The 0.9.0 merge makes the
  claim true. `links/<delta>/<offsets>/` is an ordinary rank-D grid — cell
  = the record's source chunk — so the whole connectivity family, its
  intra-chunk array included, now shards like any other array. The
  predicate deciding this is `_is_per_chunk_array`, which
  `zarr_vectors.sharding.io` imports so there is one definition of what
  may be migrated.
  ```
* The sharding codec is configured with `chunk_shape = (1,)*ndim`
  (one Zarr Vectors chunk per inner Zarr chunk) and an outer chunk shape equal
  to `shard_shape`. zarr-python ≥ 3.2 exposes this directly via the
  `shards=` kwarg on `create_array`.
* A per-array `nonempty_chunks` attribute (a sorted list of chunk
  keys) is maintained as a side-channel manifest so `list_chunks` and
  `chunk_exists` stay O(1) — without it, those calls would have to
  fetch every shard index to find non-empty cells.
* No custom space-filling-curve (Morton/Hilbert) mapping is used.
  The native codec already clusters spatially-adjacent inner chunks
  into the same shard via the C-order outer grid, which delivers the
  same read-locality benefit without a ZV-specific indirection.

### Conversion API

```python
from zarr_vectors.sharding import (
    shard_store, unshard_store, reshard, is_sharded, get_shard_info,
)

shard_store("scan.zv", shard_shape=8)          # default 8; 8x8x8 = 512 chunks/shard
shard_store("scan.zv", shard_shape=(4, 4, 16)) # anisotropic
shard_store("scan.zv", arrays=["vertices", "vertex_fragments"])   # only some arrays
unshard_store("scan.zv")                       # back to per-chunk objects
reshard("scan.zv", 4)                          # re-layout; shard_shape=None unshards
is_sharded("scan.zv")                          # True if any array uses sharding_indexed
get_shard_info("scan.zv")                      # {"sharded", "arrays": [...], "shard_count"}
```

`shard_store` returns `{"arrays_sharded", "chunks_packed", "shard_shape"}`,
`unshard_store` returns `{"arrays_unsharded", "chunks_extracted"}`. Each
walks every level and every array for which `_is_per_chunk_array` holds
(`vertices`, `vertex_fragments`, `link_fragments`, `links/<delta>/<offsets>`,
`vertex_attributes/*`, `fragment_attributes/*`,
`link_attributes/*/<delta>/<offsets>`); `object_index`, `object_attributes`,
`groups` and `group_attributes` have no spatial grid and are never sharded.
`arrays=` takes **array** paths: naming a family group such as `links/0` is
silently skipped — name its `<offsets>` children. Arrays already in the
requested layout are skipped, and unsharding a flat store is a no-op. Both
read an array's payloads into memory before rewriting it, one array at a time.

### Ordering: rebuild presence before repacking

```{important}
`shard_store` MUST run **after** `finalize_links`, never before.
```

This constrains `shard_store` — the pass that repacks an existing store.
It does **not** mean a store has to be built flat: a store declaring
`shard_shape` is born sharded and needs no repacking pass at all, which
is cheaper and sidesteps this ordering constraint entirely.

`shard_store` selects each array's cells through `list_chunks`, which
reads the `nonempty_chunks` manifest. The decentralized write protocol
leaves that manifest empty on purpose: it is array-wide state, so
stamping it is a read-modify-write that two workers race on even when
their *cells* are disjoint — the loser's key vanishes while its payload
sits on disk. Workers therefore pass `record_presence=False` and leave
one coordinator to reconstruct it. Repacking before that reconstruction
strands the payloads: the cells are on disk, but nothing enumerates them.

A level declared with `defer_presence` is the exception. Its arrays carry
no manifest at all rather than an empty one, `list_chunks` answers for
them from the store, and so `shard_store` finds every cell whether or not
the rebuild has run.

```{note}
Until 0.9.3 this section stated a second, stronger constraint: that a
sharded array's presence could not be rebuilt at all, because
`derive_nonempty_chunks` listed one object per cell and a sharded array
has none. It now reads a sharded array's cells out of the shard objects
themselves — zarr builds a sharded array's chunk grid from the *shard*
shape, so a listed key names a shard, and the cells it covers are read in
one slice. That is what makes born-sharded stores repairable, and it is
why sharding is no longer forced to be the last pass.
```

The safe order for a scale-out ingest is:

```python
# 1. Workers, in parallel, each owning a disjoint set of source chunks:
write_link_cells(lg, batch, sid_ndim, delta=0)      # record_presence=False

# 2. Coordinator, once every worker has finished:
finalize_links(lg, delta=0)     # rebuilds nonempty_chunks + counts

# 3. Coordinator, last:
shard_store("scan.zv", shard_shape=8)
```

```{note}
Step 1's `record_presence=False` leaves the cells invisible to
`list_chunks` until step 2 runs. Where a writer has no step 2 to wait for
— consumers reading as soon as it returns — `Group.collect_presence` and
`Group.apply_presence` hold the stamps back instead, and apply them
coalesced under a lock the caller already holds. The payloads still go
out unlocked, so the race this section describes is avoided the same way;
only the moment of the stamp moves. Sharding remains a coordinator pass
that runs after all of it.
```

Reversing steps 2 and 3 fails **silently, not loudly**. `shard_store`
selects each array's cells via `list_chunks`, which reads the very
manifest the workers skipped; finding it empty, it takes its
"nothing to migrate" branch and moves on. The links family is simply left
unsharded — no error, no warning, and a later `finalize_links` will still
recover the counts, so the only symptom is that the sharding you asked for
never happened.
