# Cloud stores

ZV stores on Amazon S3, Google Cloud Storage, Azure Blob Storage, and
public HTTP are accessed through the **backend layer** described in the
[store types spec page](../../spec/foundations/store_types.md). The
read/write API is identical to local stores — only the URL changes.

---

## Installation

The library ships with two cloud-capable backends; install whichever
matches your needs. `obstore` is preferred (Rust-based, faster parallel
reads), and the library auto-prefers it when both are installed.

```bash
pip install "zarr-vectors[obstore]"    # preferred
# OR
pip install "zarr-vectors[cloud]"      # fsspec + s3fs/gcsfs/adlfs (fallback)
```

You can also install both — the library will pick `obstore` and fall back
to `fsspec` for any URL scheme it can't handle.

---

## Backend resolution at a glance

When you pass a cloud URL to any `read_*` / `write_*` / `open_store` /
`open_zv` call, the backend is chosen in this order:

1. **Explicit `backend=` kwarg** — e.g. `backend="fsspec"` forces fsspec
   even if obstore is installed.
2. **`ZARR_VECTORS_BACKEND` environment variable** — e.g.
   `export ZARR_VECTORS_BACKEND=obstore`.
3. **URL-scheme auto-detect** — `s3://`, `gs://`, `gcs://`, `az://`,
   `azure://`, `abfs://`, `http(s)://` → `obstore` if installed else
   `fsspec`.

If neither cloud backend is installed for a cloud URL, the call raises a
`StoreError` with an install hint.

See [`zarr_vectors/core/backends/__init__.py`](../../../zarr_vectors/core/backends/__init__.py)
for the canonical scheme table and
[`tests/test_backends.py`](../../../tests/test_backends.py) for the
test matrix.

---

## Amazon S3

### Anonymous (public) read access

Many open neuroscience datasets on S3 allow anonymous access. The
backend layer handles it transparently — there's no `anon=True` kwarg
to pass:

```python
from zarr_vectors.types.points import read_points

result = read_points(
    "s3://open-neuro-data/datasets/synchrotron.zarrvectors",
    level=2,                                # coarse level — fast
)
print(result["vertex_count"])
```

Authentication is opt-in: if the bucket allows anonymous reads, the
default backend config will use it.

### Where credentials go

**The typed `read_*` / `write_*` functions take only `backend=` — a
backend *name*.** They have no `storage_options` parameter and no
`**backend_kwargs`, so passing `skip_signature=`, `aws_access_key_id=`,
`region=`, or `token=` to `read_points`/`write_points`/`read_polylines`
raises `TypeError`.

Backend options belong to the store-opening functions, which do accept
them:

| Function | Accepts |
|----------|---------|
| `open_store(path, mode=...)` | `backend=`, `storage_options=`, `**backend_kwargs` |
| `open_zv(path)` | `backend=`, `storage_options=`, `**backend_kwargs` |
| `ZVStore.set_backend(name)` | `storage_options=`, `**backend_kwargs` |
| `read_points` / `write_points` / `read_polylines` / `read_graph` / `read_mesh` | `backend=` only |

Ambient credentials work without configuration — `obstore` and `fsspec`
both read `~/.aws/credentials`, environment variables, and IAM roles — so
for the typed readers this is the supported path:

```python
result = read_points("s3://my-bucket/scan.zarrvectors")
```

To pass credentials explicitly, open the store yourself and supply
`storage_options` (loose `**backend_kwargs` are merged into it):

```python
import os
from zarr_vectors.core.store import open_store

root = open_store(
    "s3://my-bucket/scan.zarrvectors",
    mode="r",
    backend="obstore",
    storage_options={
        "aws_access_key_id":     os.environ["AWS_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "region":                "us-east-1",
    },
)
```

The same works for the lazy handle, which can also swap credentials on an
already-open store:

```python
from zarr_vectors.lazy import open_zv

store = open_zv(
    "s3://my-bucket/scan.zarrvectors",
    backend="obstore",
    storage_options={"skip_signature": True},   # force anonymous
)

store.set_backend("fsspec", storage_options={"anon": True})
```

Option names match the active backend (`obstore` uses `aws_*` /
`skip_signature`; `fsspec`/`s3fs` uses `key` / `secret` / `anon`). Prefer
ambient credentials when possible.

### Writing to S3

```python
import numpy as np
from zarr_vectors.types.points import write_points

rng = np.random.default_rng(0)
positions = rng.uniform(0, 1000, (100_000, 3)).astype(np.float32)

write_points(
    "s3://my-bucket/datasets/scan.zarrvectors",
    positions,
    chunk_shape=(500., 500., 500.),       # larger chunks = fewer S3 objects
    bin_shape=(100., 100., 100.),
    backend="obstore",                    # a backend *name* — no options here
)
```

`write_points` takes `backend=` only. Region and credentials come from
the ambient environment (`AWS_REGION`, `~/.aws/config`, IAM role); there
is no `region=` argument on the typed writers.

**Chunk size guidance for S3.** Each ZV spatial chunk becomes one S3
object. S3 charges per PUT (write) and GET (read) request. To minimise
cost and request count, use `chunk_shape` values that produce chunks of
at least 100 KB compressed. For typical synchrotron point clouds at
~100 000 vertices per chunk (float32, Blosc-compressed), this is
roughly 200–500 µm per axis.

### S3 bucket configuration for Neuroglancer serving

To serve a ZV store from S3 to `zv-ngtools` or the Neuroglancer web
app, configure CORS on the bucket:

```json
[
  {
    "AllowedHeaders": ["*"],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedOrigins": ["*"],
    "ExposeHeaders":  ["ETag", "Content-Length"],
    "MaxAgeSeconds":  3600
  }
]
```

Apply with the AWS CLI:

```bash
aws s3api put-bucket-cors \
    --bucket my-bucket \
    --cors-configuration file://cors.json
```

---

## Google Cloud Storage

```python
from zarr_vectors.types.polylines import read_polylines, write_polylines

# Read — uses Application Default Credentials
result = read_polylines("gs://my-bucket/tracts.zarrvectors", level=1)
print(result["polyline_count"])

# Write
write_polylines(
    "gs://my-bucket/tracts.zarrvectors",
    streamlines,
    chunk_shape=(100., 100., 100.),
    bin_shape=(25., 25., 25.),
    geometry_type="streamline",
)
```

To pass GCS credentials explicitly, open the store rather than calling
the typed reader — `read_polylines` accepts `backend=` but no `token=`:

```python
from zarr_vectors.lazy import open_zv

store = open_zv(
    "gs://my-bucket/tracts.zarrvectors",
    backend="fsspec",                       # gcsfs route
    storage_options={"token": "/path/to/service-account.json"},
)
```

GCS CORS:

```bash
gsutil cors set cors.json gs://my-bucket
```

---

## Azure Blob Storage

```python
from zarr_vectors.types.points import read_points

result = read_points("az://account/container/scan.zarrvectors")
# or:   "abfs://container@account.dfs.core.windows.net/scan.zarrvectors"
```

Ambient credentials follow the standard `DefaultAzureCredential` chain
(env vars, managed identity, Azure CLI session).

---

## Building a pyramid on a remote store

The pyramid builder takes a path or URL the same way the writers do:

```python
from zarr_vectors.multiresolution.coarsen import build_pyramid

build_pyramid(
    "s3://my-bucket/scan.zarrvectors",
    factors=[(2.0, 1.0), (2.0, 1.0)],     # coarsen 2× per level, no sparsity
    cross_level_depth=1,                  # ±1 cross-level edges per pair
    cross_level_storage="explicit",       # write both +1 and -1
)
```

For very large datasets on cloud, the pyramid build is I/O bound. Run
on a cloud VM in the same region as the bucket — building a pyramid
from within AWS `us-east-1` against a bucket in the same region is
~10× faster than from a laptop.

See [`docs/tutorials/multiscale/building_pyramids.md`](../multiscale/building_pyramids.md)
for the full pyramid API, and [`examples/07_multiscale_links.ipynb`](../../../examples/07_multiscale_links.ipynb)
for the cross-level link layout that `build_pyramid` produces.

---

## Consolidated metadata

On stores with many resolution levels and attribute arrays, opening
the store requires one metadata request per Zarr group and array. On
S3 with ~50 ms per request, this adds noticeable latency.

Consolidated metadata packs all `.zattrs` and `zarr.json` files into a
single `.zmetadata` key, reducing store-open latency to one request:

```python
import zarr
from zarr_vectors.core.store import open_store

root = open_store("s3://my-bucket/scan.zarrvectors", mode="r+")
zarr.consolidate_metadata(root.zarr_group.store)
```

After consolidation, subsequent opens are dramatically faster.
Regenerate consolidated metadata after any structural change (adding
a resolution level, writing new attributes).

---

## Reading from cloud with the lazy API

```python
import numpy as np
from zarr_vectors.lazy import open_zv

store = open_zv("s3://open-neuro/scan.zarrvectors")

print(store.levels)                          # metadata only — no chunk I/O
print(store[2].vertex_count)                 # one metadata request

# Coarse overview — a handful of chunk requests
coarse = store[store.levels[-1]].vertices.compute()

# Detail in a small region — N chunk requests
from zarr_vectors.types.points import read_points
detail = read_points(
    "s3://open-neuro/scan.zarrvectors",
    bbox=(np.array([500., 500., 500.]),
          np.array([700., 700., 700.])),
)
```

`open_zv` accepts the same `backend=` / `**backend_kwargs` as
`open_store`.

---

## Forcing a specific backend

Pass `backend="obstore"` or `backend="fsspec"` to override
auto-detection:

```python
# Force fsspec even though obstore is installed
read_points("s3://my-bucket/scan.zarrvectors", backend="fsspec")

# Use fsspec for a non-cloud URL (e.g. SFTP)
read_points("sftp://host/path/scan.zarrvectors", backend="fsspec")
```

Or set it globally for the process:

```bash
export ZARR_VECTORS_BACKEND=fsspec
```

---

## Estimating cloud storage cost

A quick estimate for an S3-hosted point cloud store:

```python
import zarr
from zarr_vectors.core.store import open_store

root = open_store("s3://my-bucket/scan.zarrvectors", mode="r")
zg   = root.zarr_group

# Walk every array and sum stored bytes / chunk counts.
total_bytes  = sum(a.nbytes_stored        for _, a in zg.arrays(recurse=True))
total_chunks = sum(a.nchunks_initialized  for _, a in zg.arrays(recurse=True))

print(f"Total compressed size: {total_bytes / 1e9:.2f} GB")
print(f"Total S3 objects:      {total_chunks:,}")
print(f"Monthly S3 storage:    ~${total_bytes / 1e9 * 0.023:.2f}"
      f"  (us-east-1 standard)")
print(f"Cost per 1M GETs:      ~${total_chunks / 1e6 * 0.40:.4f}")
```

This counts every chunk across all resolution levels and every array
family — including the `links/<delta>/<offsets>/` and
`link_attributes/<name>/<delta>/<offsets>/` arrays. Connectivity is a
single family: intra-chunk links are the all-zero offsets segment
(`links/0/0.0.0/`), cross-chunk links are the non-zero ones
(`links/0/0.0.+1/`), and each `<offsets>` segment is its own array, so a
store with many distinct offset directions has proportionally more
objects.

---

## Reducing object count with sharding

Each ZV spatial chunk is one cloud object, and per-request cost and
latency scale with object count. `shard_store` repacks every per-chunk
array with Zarr v3's standard `sharding_indexed` codec, so many inner
chunks travel as one object:

```python
from zarr_vectors.sharding import shard_store

stats = shard_store(
    "s3://my-bucket/scan.zarrvectors",
    shard_shape=8,        # outer chunk = 8 inner chunks per axis (~512 in 3-D)
)
print(stats["arrays_sharded"], stats["chunks_packed"], stats["shard_shape"])
```

`shard_shape` is expressed in *inner-chunk* units — one inner chunk is
one ZVF spatial chunk. An `int` broadcasts to every axis; a tuple sets
each axis explicitly. Pass `arrays=[...]` to convert only selected
logical arrays. The result is plain Zarr v3, readable by any conformant
implementation; no ZV-specific metadata is involved. See the
[sharding spec](../../spec/chunking/sharding.md).

---

## Decentralized link writes

When many workers write links in parallel — the common shape of a
distributed cloud ingest — they must not race on the family-wide
bookkeeping. The pattern is worker-writes, coordinator-finalizes,
coordinator-shards, **in that order**:

```python
# --- on each worker: write only the cells this worker owns ---
from zarr_vectors.core.arrays import write_link_cells, write_link_attribute_cells

partition = write_link_cells(level_group, batch, sid_ndim, delta=0)

# Per-link attributes follow the same partition, so they land beside
# the records they describe.
write_link_attribute_cells(
    level_group, "weight", weights, partition=partition, delta=0,
)
```

The returned `LinkPartition` describes **this batch** — its `num_links`
is the batch's logical record count, not the family's — and it is what
you pass to `write_link_attribute_cells`.

`write_link_cells` touches only the cells its batch lands in, leaves
every other cell alone, and deliberately does **not** maintain the
family's `num_links` / `num_physical_records` counts. It routes placement
through the same choke point `write_links` uses, so disjoint per-cell
writes plus one finalize are equivalent to a single whole-family
`write_links` over the union of the batches.

It is safe only while workers own **disjoint source chunks** — the
per-cell update is read-modify-write, so two workers appending to one
cell lose rows. A coordinator may pre-create the family with
`create_links_array` so workers agree on `directed` / `store` /
`sid_ndim` and don't race to create it.

```python
# --- on the coordinator, once every worker has finished ---
from zarr_vectors.core.arrays import finalize_links
from zarr_vectors.sharding import shard_store

part = finalize_links(level_group, delta=0)      # 1. reconcile counts
shard_store("s3://my-bucket/scan.zarrvectors")   # 2. then shard
```

Order matters: `finalize_links` rescans every offsets array and every
cell to recompute the counts, so it must run after all cells are on disk
and **before** sharding. If the counts are left unreconciled, L3
validation reports
`links[delta=0] num_physical_records=<N> != <M> rows on disk`.

### How link rows and attribute rows stay aligned

`read_links` and `read_link_attributes` both enumerate in **(offsets
segment, cell) sorted order** — segments sorted, then cells within each
segment. That shared enumeration is the *only* thing tying attribute row
`i` to link record `i`; nothing on disk records the association. So
`read_link_attributes(...)[i]` describes `read_links(...)[i]`, and the
two must never drift.

One consequence worth knowing when reading raw families: segment names
sort lexicographically, and `+` / `-` precede `0` in ASCII — so the
intra-chunk segment `0.0.0` sorts **last**, after every cross-chunk
direction. The order is deterministic, just not the one you might guess.

Under `store="duplicate"` a logical record is filed in several cells and
so comes back once per copy; dedupe, or query one location with
`read_links_for_tuple`.


