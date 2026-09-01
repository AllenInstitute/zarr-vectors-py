# Cloud stores

ZV stores on Amazon S3, Google Cloud Storage, Azure Blob Storage, and
public HTTP are reached through the **backend layer** described in the
[store types spec page](../../spec/foundations/store_types.md). The
read/write API is identical to local stores — only the URL changes, and
`zarr_vectors.api` is the surface either way.

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

## The backend is resolved, never asked for

There is no `backend=` argument anywhere on the data API. Storage is a
separate value, `zv.StorageOptions`, and the backend is *derived* from it:

1. **Explicit** — `zv.StorageOptions(backend="fsspec")` wins over everything.
2. **`ZARR_VECTORS_BACKEND`** — e.g. `export ZARR_VECTORS_BACKEND=obstore`.
3. **URL scheme** — when neither of the above is set,
   `StorageOptions.resolve_backend` returns `None`, which is the signal to
   let the store layer auto-detect from the scheme.

```python
import os
import zarr_vectors as zv

print(zv.StorageOptions().resolve_backend("s3://bucket/scan.zarrvectors"))
print(zv.StorageOptions(backend="fsspec").resolve_backend("s3://bucket/scan.zarrvectors"))

os.environ["ZARR_VECTORS_BACKEND"] = "obstore"
print(zv.StorageOptions().resolve_backend("s3://bucket/scan.zarrvectors"))
print(zv.StorageOptions(backend="fsspec").resolve_backend("s3://bucket/scan.zarrvectors"))
```

```text
None
fsspec
obstore
fsspec
```

`None` is not "no backend": it is "nothing was forced, use the scheme".
The scheme table it falls through to:

| URL scheme | Backend |
|------------|---------|
| none, `file` | `local` |
| `s3`, `gs`, `gcs`, `az`, `azure`, `abfs`, `http`, `https` | `obstore` if installed, else `fsspec` |
| anything else (`sftp`, …) | `local` — force a backend explicitly if that is wrong |

If a cloud scheme is given and neither cloud backend is installed, the
call raises a `StoreError` naming the install command. Note the last row:
`sftp://` is *not* in the cloud table, so reaching an SFTP host means
saying `StorageOptions(backend="fsspec")` rather than relying on the scheme.

See [`tests/test_backends.py`](../../../tests/test_backends.py) for the
test matrix.

---

## Amazon S3

### Anonymous (public) read access

Many open neuroscience datasets on S3 allow anonymous access. The backend
layer handles it transparently — there is no `anon=True` argument to pass:

```python
import zarr_vectors as zv

ds = zv.open("s3://open-neuro-data/datasets/synchrotron.zarrvectors")
print(ds.levels)
print(ds.level(2).vertex_count)          # coarse level — metadata only
```

Authentication is opt-in: if the bucket allows anonymous reads, the
default backend config will use it.

### Where credentials go

**Credentials are backend options, and backend options live in
`StorageOptions.options`.** That is the only channel: the data API takes
`storage=`, and `storage` carries both the backend name and its options,
so nothing on `open` / `create` / `add_points` / `read` has to grow a
storage argument.

| Call | Storage argument |
|------|------------------|
| `zv.open(url, mode=..., storage=...)` | `storage=zv.StorageOptions(...)` |
| `zv.create(url, schema=..., storage=...)` | `storage=zv.StorageOptions(...)` |
| `zv.open_or_create(url, schema=..., storage=...)` | `storage=zv.StorageOptions(...)` |
| `building.open_store(url, mode, ...)` | `backend=`, `storage_options=`, `**backend_kwargs` |

The `building` row is the store-construction surface, which is
deliberately more physical; everything else goes through `StorageOptions`.

Ambient credentials work without configuration — `obstore` and `fsspec`
both read `~/.aws/credentials`, environment variables, and IAM roles — so
this is the supported path and the one to prefer:

```python
ds = zv.open("s3://my-bucket/scan.zarrvectors")
```

To pass credentials explicitly, put them in `options`:

```python
import os
import zarr_vectors as zv

ds = zv.open(
    "s3://my-bucket/scan.zarrvectors",
    mode="r",
    storage=zv.StorageOptions(
        backend="obstore",
        options={
            "aws_access_key_id":     os.environ["AWS_ACCESS_KEY_ID"],
            "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
            "region":                "us-east-1",
        },
    ),
)
```

Option names match the *active* backend (`obstore` uses `aws_*` /
`skip_signature`; `fsspec`/`s3fs` uses `key` / `secret` / `anon`), so the
same intent is spelled two ways:

```python
anon_obstore = zv.StorageOptions(backend="obstore", options={"skip_signature": True})
anon_fsspec  = zv.StorageOptions(backend="fsspec",  options={"anon": True})

ds = zv.open("s3://open-neuro-data/scan.zarrvectors", storage=anon_obstore)
```

There is no "swap the credentials on an open handle" call. A `Dataset`
holds the `StorageOptions` it was opened with; to change them, open the
URL again with different ones. That is one metadata request, and it makes
the credentials a property of the handle rather than a mutable global.

### Writing to S3

```python
import numpy as np
import zarr_vectors as zv

rng = np.random.default_rng(0)
positions = rng.uniform(0, 1000, (100_000, 3)).astype(np.float32)

schema = zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    kind="point_cloud",
    layout=zv.Layout(cells=2),      # 2 cells/axis = 500 µm chunks = fewer S3 objects
)

ds = zv.create(
    "s3://my-bucket/datasets/scan.zarrvectors",
    schema=schema,
    storage=zv.StorageOptions(backend="obstore"),
)
ds.add_points(positions)
```

Region and credentials come from the ambient environment (`AWS_REGION`,
`~/.aws/config`, IAM role) unless you put them in `StorageOptions.options`.

### Chunk size guidance for S3

Each ZV spatial chunk becomes one S3 object per array family, and S3
charges per PUT and per GET. `zv.Layout` is where that trade-off is
expressed — in cells per axis, not in micrometres:

```python
schema = zv.Schema(bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)))
print(zv.Layout(cells=5).resolve(schema).chunk_shape)
print(zv.Layout(cells=2).resolve(schema).chunk_shape)
print(zv.Layout(cell_size=(500.0, 500.0, 500.0)).resolve(schema).chunk_shape)
```

```text
(200.0, 200.0, 200.0)
(500.0, 500.0, 500.0)
(500.0, 500.0, 500.0)
```

Aim for chunks of at least 100 KB compressed. For typical synchrotron
point clouds at ~100 000 vertices per chunk (float32, Blosc-compressed)
that is roughly 200–500 µm per axis, so `cells=2`…`cells=5` over a
1 000 µm volume. `Layout.cell_size` is the escape hatch when the grid is
fixed from outside — a pipeline whose chunks must line up with an image
volume, say.

`Layout.compression` defaults to `"auto"`, which means "whatever
`$ZARR_VECTORS_COMPRESSION` says, and no compressor if it says nothing".

### Packing is already on for object stores

`Layout.pack` decides whether many cells travel as one storage object.
Its default is not a constant — it is resolved against the kind of store
being written, because one object per cell is cheap on a local filesystem
and expensive on a bucket:

```python
schema = zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    layout=zv.Layout(cells=8),
)
print(schema.layout.resolve(schema, store_kind="local").shard_shape)
print(schema.layout.resolve(schema, store_kind="object").shard_shape)
```

```text
None
(4, 4, 4)
```

A dataset whose URL carries a cloud scheme *is* an object store as far as
this decision goes, so writing to `s3://…` packs without being asked and
writing to a local path does not. Forcing it either way is
`zv.Layout(..., pack=True)` / `pack=False`.

Through the `Dataset` writers the automatic shard always works out to at
most four cells per axis, whatever you passed at create time. The shard
shape is derived from the store's *own declared* schema, and a store does
not record the `expected=zv.SizeHints(...)` hint — so the size-driven path
(`Layout.target_object_bytes`) never engages on a write into an existing
store, and the four-cell fallback is what you get:

```python
schema = zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    expected=zv.SizeHints(n_vertices=100_000),
    layout=zv.Layout(cells=8),
)
ds = zv.create("packed.zarrvectors", schema=schema)
ds.add_points(positions, layout=zv.Layout(cells=8, pack=True))

from zarr_vectors.building import get_shard_info
info = get_shard_info("packed.zarrvectors")
print([(a["name"], a["grid_shape"], a["shard_shape"]) for a in info["arrays"]])
```

```text
[('0/vertices', [8, 8, 8], [4, 4, 4]), ('0/vertex_fragments', [8, 8, 8], [4, 4, 4])]
```

That is a safe middle, not a tuned answer: if you want a different one,
write the store and then choose the shard shape explicitly with
`building.reshard` (see
[Reducing object count with sharding](#reducing-object-count-with-sharding)
below). The point is that the *hand-tuning step is no longer mandatory* —
an unadorned `zv.create(...)` + `add_points(...)` against a bucket already
produces a packed store.

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
import zarr_vectors as zv

# Read — uses Application Default Credentials
ds = zv.open("gs://my-bucket/tracts.zarrvectors")
print(ds.level(1).objects.count)

# Write — `streamlines` is a list of (N_k, 3) arrays, one per tract
tracts = zv.create(
    "gs://my-bucket/tracts.zarrvectors",
    schema=zv.Schema(
        bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
        kind="polyline",
        layout=zv.Layout(cells=10),
    ),
)
tracts.add_polylines(streamlines, streamlines=True)
```

To pass GCS credentials explicitly, put the token in `StorageOptions`:

```python
ds = zv.open(
    "gs://my-bucket/tracts.zarrvectors",
    storage=zv.StorageOptions(
        backend="fsspec",                                   # gcsfs route
        options={"token": "/path/to/service-account.json"},
    ),
)
```

GCS CORS:

```bash
gsutil cors set cors.json gs://my-bucket
```

---

## Azure Blob Storage

```python
import zarr_vectors as zv

ds = zv.open("az://account/container/scan.zarrvectors")
# or:   "abfs://container@account.dfs.core.windows.net/scan.zarrvectors"
```

Ambient credentials follow the standard `DefaultAzureCredential` chain
(env vars, managed identity, Azure CLI session).

---

## Building a pyramid on a remote store

Pyramids are built through the dataset handle, so the URL and the
credentials are already settled by the time you ask:

```python
ds = zv.open("s3://my-bucket/scan.zarrvectors", mode="r+")

print(zv.coarsen_methods())               # what this install can coarsen with

report = ds.build_pyramid(
    factors=[(2.0, 1.0), (2.0, 1.0)],     # coarsen 2× per level, no sparsity
    chunk_scale_factors=[2, 2],           # and grow the chunk with the level
    method="per_object",
    cross_level_depth=1,                  # ±1 cross-level edges per pair
    cross_level_storage="explicit",       # write both +1 and -1
)
print(report["levels_created"])
```

```text
('per_object',)
2
```

`method` selects the coarsener; `zv.coarsen_methods()` lists the ones this
installation has, including any a strategy package registered on import.
Pass a strategy's own knobs as `options={...}`.

Pass `chunk_scale_factors=` alongside `factors=` — without it every level
inherits the root chunk shape, which on a bucket means the coarse levels
have as many objects as the fine one, for a fraction of the data.

For very large datasets on cloud, the pyramid build is I/O bound. Run
on a cloud VM in the same region as the bucket — building a pyramid
from within AWS `us-east-1` against a bucket in the same region is
~10× faster than from a laptop.

See [Building pyramids](../multiscale/building_pyramids.md) for the full
pyramid API and for the cross-level link layout that `build_pyramid` produces.

---

## Consolidated metadata

On stores with many resolution levels and attribute arrays, opening
the store requires one metadata request per Zarr group and array. On
S3 with ~50 ms per request, this adds noticeable latency.

A ZV store is Zarr v3, so there are **no `.zattrs` files anywhere** —
that is the v2 spelling. Every group and array carries a `zarr.json`, and
consolidating folds all of the descendant ones into the root `zarr.json`
under a `consolidated_metadata` key. There is no separate `.zmetadata`
object either; that too is v2.

```python
import warnings

import zarr
import zarr_vectors as zv

ds = zv.open("s3://my-bucket/scan.zarrvectors", mode="r+")
with warnings.catch_warnings():                  # v3 consolidation warns; see below
    warnings.simplefilter("ignore")
    zarr.consolidate_metadata(ds.store.zarr_group.store)
```

Two things to know before reaching for this:

- `zarr` warns that consolidated metadata is not part of the Zarr v3
  specification and may not be understood by other implementations. It is
  a performance option, not a portable one. `zarr-vectors` itself reads it
  when it is there — a store opened after consolidation resolves every
  level's metadata from the one root document.
- The store fields survive it: `attributes.zarr_vectors` and
  `attributes.multiscales` are still on the root `zarr.json` afterwards,
  with `consolidated_metadata` added beside them.

Regenerate it after any structural change (adding a resolution level,
writing new attributes) — a stale consolidated document describes a store
that no longer exists.

`ds.store` is the documented escape hatch on `Dataset`, and
`.zarr_group` is *not* among the promised `Group` methods, so this snippet
reaches past the contract. `zarr-vectors` has no supported wrapper for
consolidation; that is a gap to report rather than a reason to make a
habit of `.zarr_group`.

---

## Reading from cloud with the data API

The API is the same one the [quickstart](../../getting_started/quickstart.md)
uses. What is worth knowing on a bucket is *which calls cost requests*:

```python
import zarr_vectors as zv

ds = zv.open("s3://open-neuro/scan.zarrvectors")

print(ds.levels)                     # metadata only — no chunk I/O
print(ds.level(2).vertex_count)      # metadata only — the count is recorded

# Coarse overview — a handful of chunk requests
coarse = ds.read(level=ds.levels[-1])

# Detail in a small region
q = ds.select(bbox=((500.0, 500.0, 500.0), (700.0, 700.0, 700.0)))
print(q.explain())                   # names the reader and its arguments
print(len(q.cells()))                # how many grid cells it will touch — reads nothing
detail = q.read()                    # only now does anything leave the machine
```

`Level.vertex_count` and `Level.objects.count` answer from the level's
recorded metadata, so counting a remote store is free. `Query.explain()`
and `Query.cells()` are the two calls that let you predict a query's cost
before paying it: the cell count is the number of stored objects the read
will fetch *per array family it needs*, so a bbox that straddles a chunk
boundary on every axis costs eight, not one.

`ds.resolution(scale=...)` picks a level by physical size rather than by
index, which is usually what a viewer wants:

```python
print(ds.resolution(scale=400.0).index)
```

---

## Forcing a specific backend

```python
# Force fsspec even though obstore is installed
ds = zv.open(
    "s3://my-bucket/scan.zarrvectors",
    storage=zv.StorageOptions(backend="fsspec"),
)

# Use fsspec for a scheme the auto-detect table does not cover
ds = zv.open(
    "sftp://host/path/scan.zarrvectors",
    storage=zv.StorageOptions(backend="fsspec"),
)
```

Or set it globally for the process, which every `StorageOptions` with no
explicit `backend` will then pick up:

```bash
export ZARR_VECTORS_BACKEND=fsspec
```

---

## Estimating cloud storage cost

Request cost is driven by the number of stored objects, and
`zarr_vectors.building` can count them without decoding anything:

```python
from zarr_vectors.building import (
    array_is_sharded,
    get_resolution_level,
    list_resolution_levels,
    open_store,
    per_chunk_array_paths,
)

root = open_store("s3://my-bucket/scan.zarrvectors", mode="r")

objects = 0
for index in list_resolution_levels(root):
    level = get_resolution_level(root, index)
    for path in per_chunk_array_paths(level):
        cells = len(level.list_chunks(path))
        packed = array_is_sharded(level, path)
        objects += cells
        print(f"{index}/{path:<30} {cells:>6} cells{'  (sharded)' if packed else ''}")

print(f"Stored objects:    {objects:,}")
print(f"Cost per 1M GETs:  ~${objects / 1e6 * 0.40:.4f}")
```

Run against a small local two-level point cloud, that prints:

```text
0/links/+1/0.0.0                    125 cells
0/vertex_attributes/intensity       125 cells
0/vertex_fragments                  125 cells
0/vertices                          125 cells
1/links/-1/+1.+1.+1                   8 cells
1/links/-1/+1.+1.0                   12 cells
1/links/-1/+1.0.+1                   12 cells
1/links/-1/+1.0.0                    18 cells
1/links/-1/0.+1.+1                   12 cells
1/links/-1/0.+1.0                    18 cells
1/links/-1/0.0.+1                    18 cells
1/links/-1/0.0.0                     27 cells
1/vertex_fragments                   27 cells
1/vertices                           27 cells
Stored objects:    679
Cost per 1M GETs:  ~$0.0003
```

`per_chunk_array_paths` walks a level recursively, so the listing includes
every array family: `links/<delta>/<offsets>/` and
`link_attributes/<name>/<delta>/<offsets>/` as well as `vertices/`. That is
the shape of the cost. Connectivity is a *single* family — intra-chunk
links are the all-zero offsets segment (`links/0/0.0.0/`), cross-chunk
links are the non-zero ones (`links/0/0.0.+1/`), and each `<offsets>`
segment is its own array — so a store with many distinct offset directions
has proportionally more objects. Above, one pyramid level's eight
cross-level directions cost more objects than its vertices do.

Note what the cell count is *not*: when an array is sharded, many cells
travel in one storage object, so `array_is_sharded` is the column that
tells you whether the number is an upper bound. Stored *bytes* are best
asked of the bucket, which knows the compressed truth:

```bash
aws s3 ls --recursive --summarize s3://my-bucket/scan.zarrvectors
```

---

## Reducing object count with sharding

Each ZV spatial chunk is one cloud object, and per-request cost and
latency scale with object count. A store written to a bucket is packed
already (see [above](#packing-is-already-on-for-object-stores)); a store
written locally and then uploaded is not, and `shard_store` repacks it
after the fact with Zarr v3's standard `sharding_indexed` codec:

```python
from zarr_vectors.building import get_shard_info, reshard, shard_store, unshard_store

stats = shard_store(
    "s3://my-bucket/scan.zarrvectors",
    shard_shape=4,        # outer chunk = 4 inner chunks per axis (64 in 3-D)
)
print(stats)

info = get_shard_info("s3://my-bucket/scan.zarrvectors")
print(info["sharded"], info["shard_count"])
print(info["arrays"][0])
```

```text
{'arrays_sharded': 14, 'chunks_packed': 679, 'shard_shape': [4, 4, 4]}
True 42
{'name': '0/vertex_fragments', 'grid_shape': [5, 5, 5], 'shard_shape': [4, 4, 4], 'shard_count': 8}
```

`shard_shape` is expressed in *inner-chunk* units — one inner chunk is
one Zarr Vectors spatial chunk. An `int` broadcasts to every axis; a tuple sets
each axis explicitly. Pass `arrays=[...]` to convert only selected
logical arrays. The result is plain Zarr v3, readable by any conformant
implementation; no ZV-specific metadata is involved. See the
[sharding spec](../../spec/chunking/sharding.md).

Two companions, both in `building`, make it reversible:

```python
reshard("s3://my-bucket/scan.zarrvectors", 2)   # change the shard shape in place
unshard_store("s3://my-bucket/scan.zarrvectors")  # back to one object per cell
```

```text
{'action': 'shard', 'arrays_sharded': 14, 'chunks_packed': 679, 'shard_shape': [2, 2, 2]}
{'arrays_unsharded': 14, 'chunks_extracted': 679}
```

`reshard(path, None)` also unshards, reporting
`{'action': 'unshard', ...}`; `get_shard_info` on an unsharded store
reports `{'sharded': False, 'arrays': [], 'shard_count': 0}`.

---

## Decentralized link writes

When many workers write links in parallel — the common shape of a
distributed cloud ingest — they must not race on the family-wide
bookkeeping. The pattern is coordinator-creates, worker-writes,
coordinator-finalizes, coordinator-shards, **in that order**. Every name
below is in `zarr_vectors.building`:

```python
from zarr_vectors.building import (
    create_links_array,
    create_store,
    get_resolution_level,
    write_link_attribute_cells,
    write_link_cells,
)

# --- on the coordinator, before any worker starts ---
root = create_store(
    "s3://my-bucket/edges.zarrvectors",
    bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
    chunk_shape=(100.0, 100.0, 100.0),
    geometry_types=["graph"],
    ndim=3,
)
level_group = get_resolution_level(root, 0)
create_links_array(level_group, link_width=2, delta=0, sid_ndim=3)
```

Pre-creating the family is what makes the workers agree on `directed` /
`store` / `sid_ndim` instead of racing to create it with different answers.

```python
import numpy as np

# --- on each worker: write only the cells this worker owns ---
# Each record is a list of (chunk_coords, vertex_index) endpoints.
batch = [
    [((0, 0, 0), 1), ((1, 0, 0), 2)],
    [((0, 0, 0), 3), ((1, 0, 0), 4)],
]
partition = write_link_cells(level_group, batch, 3, delta=0)   # 3 = sid_ndim

# Per-link attributes follow the same partition, so they land beside
# the records they describe.  One row per record, in input order.
weights = np.array([0.5, 0.25], dtype="float32")
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
cell lose rows.

```python
# --- on the coordinator, once every worker has finished ---
from zarr_vectors.building import finalize_links, shard_store

part = finalize_links(level_group, delta=0)           # 1. reconcile counts
print(part.num_links, part.num_physical_records)
shard_store("s3://my-bucket/edges.zarrvectors")       # 2. then shard
```

Order matters: `finalize_links` rescans every offsets array and every
cell to recompute the counts, so it must run after all cells are on disk
and **before** sharding. If the counts are left unreconciled, L3
validation reports
`links[delta=0] num_physical_records=<N> != <M> rows on disk` — see
[Validation and repair](validation_and_repair.md#l3--consistency).

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
