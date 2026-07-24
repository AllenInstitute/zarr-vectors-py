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
  chunk. In the ZVF context, the inner chunk corresponds to one ZVF spatial
  chunk. The inner chunk shape must equal the ZVF `chunk_shape` expressed
  in array coordinates.

**Outer chunk** (shard shape)
: The number of inner chunks grouped into one shard, expressed as a tuple.
  An outer chunk of `(4, 4, 4)` means each shard contains `4 × 4 × 4 = 64`
  inner chunks (ZVF spatial chunks).

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

With sharding, you keep the logical ZVF `chunk_shape` small (good for
queries), but multiple ZVF chunks are packed into a single shard file. A
reader fetching one ZVF chunk makes one HTTP request to the shard file (or
uses HTTP range requests to fetch only the relevant byte range), exactly
as if each chunk were a separate file — but the object count is reduced by
the shard factor.

Sharding is available natively in Zarr v3 and does not require changes to
the ZVF data model. It is purely a storage-layer optimisation. The fragment index,
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

Sharding is configured via the `codec_config` argument to write functions,
or by passing a pre-configured Zarr array spec. A minimal sharding
configuration for a 3-D ZVF store:

```python
from zarr_vectors.types.points import write_points

write_points(
    "scan.zarrvectors",
    positions,
    chunk_shape=(100.0, 100.0, 100.0),   # ZVF logical chunk — the inner chunk
    bin_shape=(25.0, 25.0, 25.0),
    shard_shape=(4, 4, 4),               # 4×4×4 = 64 inner chunks per shard
)
```

This produces one shard file per `(4, 4, 4)` block of the ZVF chunk grid.
Each shard contains up to 64 ZVF chunks. The total number of shard files is
`ceil(grid_shape[d] / 4) for d in [0,1,2]` — a factor of 64 fewer files
than without sharding.

### Codec pipeline with sharding

When `shard_shape` is specified, the `zarr.json` for each array uses the
`sharding_indexed` codec as the outermost byte-to-byte codec:

```json
{
  "codecs": [
    {
      "name": "sharding_indexed",
      "configuration": {
        "chunk_shape": [1, 1, 1, 65536, 3],
        "codecs": [
          {"name": "bytes",  "configuration": {"endian": "little"}},
          {"name": "blosc",  "configuration": {"cname": "zstd", "clevel": 5}}
        ],
        "index_codecs": [
          {"name": "bytes",  "configuration": {"endian": "little"}},
          {"name": "crc32c"}
        ],
        "index_location": "end"
      }
    }
  ]
}
```

The `chunk_shape` inside the sharding configuration is the *inner* chunk
shape (one ZVF spatial chunk). The shard shape is inferred from the outer
array chunk grid.

### Relationship between shard shape and ZVF chunk shape

The ZVF `chunk_shape` (physical units) maps to the *inner* chunk of the
sharding codec. The shard shape (outer chunk) is expressed in inner-chunk
units (integers), not in physical units:

```
ZVF chunk_shape = (100, 100, 100) µm
shard_shape     = (4, 4, 4) inner chunks
→ each shard covers (400, 400, 400) µm of physical space
```

Readers that understand the shard index can fetch a single ZVF chunk with
one HTTP range request; readers that do not understand sharding must fetch
the entire shard file. `zarr-vectors-py` always uses shard-indexed reads
when the sharding codec is present.

### Read behaviour with sharding

When reading a single ZVF chunk from a sharded store:

1. Identify the shard containing the requested inner chunk.
2. Fetch only the shard index (last `n_inner_chunks × 16 bytes` of the
   shard file, via an HTTP range request for the tail of the object).
3. Look up the byte offset and length of the requested inner chunk.
4. Fetch the inner chunk data (second HTTP range request).

Total: **2 HTTP requests** for one ZVF chunk, regardless of shard size.
Without sharding, it is **1 HTTP request** per chunk but many more objects.

### Write behaviour with sharding

Writing to a sharded store serialises writes at the shard level: multiple
inner chunks that share a shard must be written atomically. `zarr-vectors-py`
buffers all inner chunks for a shard before writing the shard file. This
increases peak memory usage compared to non-sharded writes.

For parallel writes (HPC, multi-process), assign non-overlapping shard
ranges to different workers to avoid shard-level write contention.

### Choosing shard shape

A good shard shape balances:

- **Shard file size:** target 10–100 MB per shard. Smaller shards offer
  finer parallelism; larger shards reduce object count more aggressively.
- **Read efficiency:** if most queries read entire shards, a large shard
  shape saves requests. If most queries read only 1–2 inner chunks per
  shard, a small shard shape minimises wasted range-request bytes.
- **Write parallelism:** shards are the unit of write locking. A shard
  shape of `(1, 1, 1)` disables sharding (each shard is one inner chunk);
  this is useful if you need exactly one file per ZVF chunk.

A practical default for Neuroglancer-style serving:

```python
shard_shape = (8, 8, 8)   # 512 inner chunks per shard, ~25–200 MB per shard
```

### Sharding and the fragment index

The `vertex_fragments/` and `link_fragments/` arrays benefit especially
from sharding. Unlike `vertices/`, which may have large chunks, fragment-
index chunks are small (typically tens to a few hundred bytes per ZVF
spatial chunk — header plus a short range table plus a tiny CSR when
fragments are explicit). Without sharding, a large store creates millions
of tiny files for the fragment indices alone. With sharding, many
fragment-index chunks are packed into a single shard, and a viewer can
fetch the entire spatial index for a region in a handful of requests.

### Compatibility

Sharding requires Zarr v3 ≥ 2.18 (where `sharding_indexed` is built in).
Readers that use an older Zarr version will fail to open sharded arrays.
Document the use of sharding in the store's root `.zattrs` `"notes"` field
to ensure consumers are aware of the requirement.

### Implementation notes (zarr-vectors-py)

* Each ZVF logical array (`vertices`, `vertex_fragments`,
  `links/<delta>/<offsets>`, `link_fragments`,
  `link_attributes/<name>/<delta>/<offsets>`, attribute arrays, ...) maps
  to a single Zarr v3 vlen-bytes array whose shape is the level's chunk
  grid. One cell of that array holds one ZVF spatial chunk's payload
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
  (one ZVF chunk per inner Zarr chunk) and an outer chunk shape equal
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
from zarr_vectors.sharding import shard_store, unshard_store, reshard

shard_store("scan.zv", shard_shape=8)          # 8x8x8 = 512 chunks/shard
shard_store("scan.zv", shard_shape=(4, 4, 16)) # anisotropic
unshard_store("scan.zv")                       # back to per-chunk objects
reshard("scan.zv", 4)                          # change shard shape in place
reshard("scan.zv", None)                       # equivalent to unshard_store
```

Both conversion functions are idempotent: shards already in the
requested layout are skipped, and unsharding a flat store is a no-op.

### Ordering: shard the links family last

```{important}
`shard_store` MUST run **after** `finalize_links`, never before.
```

`finalize_links` rebuilds each link array's `nonempty_chunks` manifest
from the store listing, via `Group.derive_nonempty_chunks`. That rebuild
is **unsharded-only**: it works by listing chunk objects at
`<array>/c/i/j/k` and reading the cell coordinate back out of the key
name. A shard packs many cells into a single object whose inner index is
**not derivable from key names**, so once an array is sharded the listing
no longer enumerates its cells and the rebuild finds nothing.

The constraint is a consequence of how the decentralized write protocol
splits work. `nonempty_chunks` is array-wide state, so stamping it is a
read-modify-write that two workers race on even when their *cells* are
disjoint — the loser's key vanishes while its payload sits on disk.
Workers therefore write with `record_presence=False` and skip the
manifest entirely, leaving one coordinator to reconstruct it. Sharding
before that reconstruction strands the payloads: the cells are on disk,
but nothing enumerates them.

The safe order for a scale-out ingest is:

```python
# 1. Workers, in parallel, each owning a disjoint set of source chunks:
write_link_cells(lg, batch, sid_ndim, delta=0)      # record_presence=False

# 2. Coordinator, once every worker has finished:
finalize_links(lg, delta=0)     # rebuilds nonempty_chunks + counts

# 3. Coordinator, last:
shard_store("scan.zv", shard_shape=8)
```

Reversing steps 2 and 3 fails **silently, not loudly**. `shard_store`
selects each array's cells via `list_chunks`, which reads the very
manifest the workers skipped; finding it empty, it takes its
"nothing to migrate" branch and moves on. The links family is simply left
unsharded — no error, no warning, and a later `finalize_links` will still
recover the counts, so the only symptom is that the sharding you asked for
never happened.
