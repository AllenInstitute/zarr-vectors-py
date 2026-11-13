# Cloud stores

A store on Amazon S3, Google Cloud Storage, Azure Blob Storage or an HTTP
server is opened and written with the same `zarr_vectors` calls as a local
store; only the URL changes, plus a `zv.StorageOptions` where the backend
needs settings. This tutorial installs a cloud backend, shows how the backend
is chosen, writes and packs a store for an object store, counts the objects it
will create, and writes links from parallel workers. The backend layer is
specified in [Store types](../../spec/foundations/store_types.md).

## Install a cloud backend

```bash
pip install "zarr-vectors[obstore]"   # obstore
pip install "zarr-vectors[fsspec]"    # fsspec with s3fs, gcsfs and adlfs
```

`zarr-vectors[cloud]` installs the same as `zarr-vectors[obstore]`. When both
backends are installed, obstore is used for cloud URLs unless you choose
otherwise.

## How the backend is chosen

The data API (`zv.open`, `zv.create`, `zv.open_or_create`) takes one storage
argument, `storage=zv.StorageOptions(backend=..., options=...)`. The backend
is resolved in this order:

1. `StorageOptions.backend`, if set.
2. The `ZARR_VECTORS_BACKEND` environment variable.
3. The URL scheme.

`StorageOptions.resolve_backend` returns the first of the two settings, or
`None` when the scheme decides:

```python
import os

import zarr_vectors as zv

url = "s3://my-bucket/scan.zarrvectors"
print(zv.StorageOptions().resolve_backend(url))
print(zv.StorageOptions(backend="fsspec").resolve_backend(url))

os.environ["ZARR_VECTORS_BACKEND"] = "obstore"
print(zv.StorageOptions().resolve_backend(url))
print(zv.StorageOptions(backend="fsspec").resolve_backend(url))
del os.environ["ZARR_VECTORS_BACKEND"]
```

```text
None
fsspec
obstore
fsspec
```

From the scheme:

| URL scheme | Backend |
|------------|---------|
| none, `file` | local |
| `s3`, `gs`, `gcs`, `az`, `azure`, `abfs`, `http`, `https` | obstore if installed, otherwise fsspec |
| any other, such as `sftp` | local |

For a scheme outside the table, such as `sftp://`, set the backend
explicitly:

```python
ds = zv.open(
    "sftp://host/path/scan.zarrvectors",
    storage=zv.StorageOptions(backend="fsspec"),
)
```

A cloud URL with neither backend installed raises `StoreError`:

```python
from zarr_vectors.exceptions import StoreError

try:
    zv.open("s3://my-bucket/scan.zarrvectors")
except StoreError as error:
    print(error)
```

```text
URL 's3://my-bucket/scan.zarrvectors' has scheme 's3' which requires a cloud backend, but neither 'obstore' nor 'fsspec' is installed. Install with: pip install zarr-vectors[obstore]
```

## Credentials and other backend settings

`StorageOptions.options` is passed to the backend unchanged: to
`obstore.store.from_url(url, **options)` for obstore, and as `storage_options`
to Zarr's `FsspecStore.from_url` for fsspec. The option names are therefore
the backend library's own, and the same setting is spelt differently for each:

```python
import os

import zarr_vectors as zv

s3_keys_obstore = zv.StorageOptions(backend="obstore", options={
    "access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
    "secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
    "region": "us-east-1",
})
s3_keys_fsspec = zv.StorageOptions(backend="fsspec", options={
    "key": os.environ["AWS_ACCESS_KEY_ID"],
    "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
})
public_obstore = zv.StorageOptions(backend="obstore", options={"skip_signature": True})
public_fsspec = zv.StorageOptions(backend="fsspec", options={"anon": True})
gcs_fsspec = zv.StorageOptions(backend="fsspec", options={"token": "service-account.json"})

ds = zv.open("s3://my-bucket/scan.zarrvectors", storage=s3_keys_obstore)
```

Settings you leave out are up to the backend library, including whether it
reads credentials from the environment. See the obstore and fsspec (s3fs,
gcsfs, adlfs) documentation for the names each accepts.

A `Dataset` keeps the storage options it was opened with. To use different
credentials, open the URL again.

`zarr_vectors.building.open_store` and `create_store`, the builder surface,
take `backend=`, `storage_options=` and backend keyword arguments instead of a
`StorageOptions`.

## Write a store to a bucket

```python
import numpy as np
import zarr_vectors as zv

rng = np.random.default_rng(0)
positions = rng.uniform(0, 1000, (100_000, 3)).astype(np.float32)

ds = zv.create(
    "s3://my-bucket/datasets/scan.zarrvectors",
    schema=zv.Schema(
        bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
        kind="point_cloud",
        layout=zv.Layout(cells=2),
    ),
)
ds.add_points(positions)
```

### Chunk size

Each spatial chunk of each array is a separate stored object unless the
store is packed, and object stores charge per request. `zv.Layout` sets the
chunk size as cells per axis or as a physical size:

```python
schema = zv.Schema(bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)))
print(zv.Layout(cells=5).resolve(schema).chunk_shape)
print(zv.Layout(cells=2).resolve(schema).chunk_shape)
print(zv.Layout(cell_size=(500.0, 500.0, 500.0)).resolve(schema).chunk_shape)
print(zv.Layout().resolve(schema).compressor)
```

```text
(200.0, 200.0, 200.0)
(500.0, 500.0, 500.0)
(500.0, 500.0, 500.0)
zstd
```

Use `cell_size` when the grid must line up with something else, such as an
image volume. `Layout.compression="auto"`, the default, uses
`$ZARR_VECTORS_COMPRESSION` if it is set and `zstd` otherwise.

### Packing

`Layout.pack` decides whether many cells are stored as one object (a Zarr v3
shard). Its default, `"auto"`, packs on an object store and not on a local
path; a URL with a cloud scheme counts as an object store:

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

Pass `pack=True` or `pack=False` to choose. The shard shape chosen when
writing through a `Dataset` is at most four cells per axis:

```python
from zarr_vectors.building import get_shard_info

ds = zv.create("packed.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    layout=zv.Layout(cells=8),
))
ds.add_points(positions, layout=zv.Layout(cells=8, pack=True))

info = get_shard_info("packed.zarrvectors")
print([(a["name"], a["grid_shape"], a["shard_shape"]) for a in info["arrays"]])
```

```text
[('0/vertices', [8, 8, 8], [4, 4, 4]), ('0/vertex_fragments', [8, 8, 8], [4, 4, 4])]
```

For another shard shape, use `reshard` after writing (see
[Shard an existing store](#shard-an-existing-store)).

## Count the stored objects

The number of stored objects drives request cost. The builder surface counts
them per array without decoding anything. The example builds a small local
store with one pyramid level; a URL works the same way:

```python
import numpy as np
import zarr_vectors as zv
from zarr_vectors.building import (
    array_is_sharded,
    get_resolution_level,
    list_resolution_levels,
    open_store,
    per_chunk_array_paths,
)

rng = np.random.default_rng(0)
scan = zv.create("scan.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    kind="point_cloud",
    vertex_attributes={"intensity": zv.AttributeSpec(dtype="float32")},
    layout=zv.Layout(cells=5),
))
scan.add_points(
    rng.uniform(0, 1000, (100_000, 3)).astype(np.float32),
    attributes={"intensity": rng.uniform(0, 1, 100_000).astype(np.float32)},
)
scan.build_pyramid(factors=[(2.0, 1.0)], chunk_scale_factors=[2])

root = open_store("scan.zarrvectors", mode="r")
objects = 0
for index in list_resolution_levels(root):
    level = get_resolution_level(root, index)
    for path in per_chunk_array_paths(level):
        cells = len(level.list_chunks(path))
        packed = array_is_sharded(level, path)
        objects += cells
        print(f"{index}/{path:<30} {cells:>6} cells{'  (sharded)' if packed else ''}")
print(f"Stored objects: {objects:,}")
```

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
Stored objects: 679
```

Every link direction (`links/<delta>/<offsets>/`) is its own array, so the
downward links of level 1 take more objects than its vertices. On a sharded
array many cells share one object, and the cell count is an upper bound.

## Shard an existing store

A store written to a local path and then uploaded is not packed.
`shard_store` packs it with Zarr v3's `sharding_indexed` codec:

```python
from zarr_vectors.building import get_shard_info, reshard, shard_store, unshard_store

print(shard_store("scan.zarrvectors", shard_shape=4))

info = get_shard_info("scan.zarrvectors")
print(info["sharded"], info["shard_count"])
print(sorted(info["arrays"], key=lambda a: a["name"])[0])

print(reshard("scan.zarrvectors", 2))
print(unshard_store("scan.zarrvectors"))
print(get_shard_info("scan.zarrvectors"))
```

```text
{'arrays_sharded': 14, 'chunks_packed': 679, 'shard_shape': [4, 4, 4]}
True 42
{'name': '0/links/+1/0.0.0', 'grid_shape': [6, 6, 6], 'shard_shape': [4, 4, 4], 'shard_count': 8}
{'action': 'shard', 'arrays_sharded': 14, 'chunks_packed': 679, 'shard_shape': [2, 2, 2]}
{'arrays_unsharded': 14, 'chunks_extracted': 679}
{'sharded': False, 'arrays': [], 'shard_count': 0}
```

`shard_shape` counts spatial chunks: an integer applies to every axis, a
tuple sets each axis. `arrays=[...]` limits the change to some arrays.
`reshard(path, None)` unshards. The result is plain Zarr v3; see
[Sharding](../../spec/chunking/sharding.md).

## Build a pyramid on a remote store

`Dataset.build_pyramid` works the same on a URL:

```python
ds = zv.open("s3://my-bucket/scan.zarrvectors", mode="r+")
ds.build_pyramid(factors=[(2.0, 1.0), (2.0, 1.0)], chunk_scale_factors=[2, 2])
```

Pass `chunk_scale_factors`: without it every level keeps the level-0 chunk
shape, so coarse levels are stored in as many objects as level 0. See
[Building multi-resolution pyramids](../multiscale/building_pyramids.md).

## Read from a remote store

`ds.levels`, `Level.vertex_count` and `Level.objects.count` come from metadata.
`Query.explain()` and `Query.cells()` show what a read will fetch before it
runs. [Deferred reads and out-of-core access](../multiscale/lazy_loading.md)
covers choosing a level and reading cell by cell.

## Consolidated metadata

Opening a store reads one `zarr.json` per group and array. Consolidating
copies the descendant metadata into the root `zarr.json` under
`consolidated_metadata`, so it can be read in one request. There is no
wrapper for this in `zarr_vectors`; call Zarr on the store's underlying group:

```python
import warnings

import zarr
import zarr_vectors as zv

ds = zv.open("scan.zarrvectors", mode="r+")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    zarr.consolidate_metadata(ds.store.zarr_group.store)
print(zv.open("scan.zarrvectors").levels)
```

```text
(0, 1)
```

Zarr warns that consolidated metadata is not part of the Zarr v3
specification, so other implementations may ignore it. The root attributes
(`zarr_vectors`, `multiscales`, `ome`) are unchanged. Consolidate again after
adding a level or an attribute array, or the consolidated copy describes the
old store.

## Write links from parallel workers

When workers write links in parallel, a coordinator creates the link family,
each worker writes only its own cells, and the coordinator then reconciles
the counts and, last, shards the store. Every name below is in
`zarr_vectors.building`.

On the coordinator, before any worker starts:

```python
from zarr_vectors.building import (
    create_links_array,
    create_store,
    get_resolution_level,
)

root = create_store(
    "edges.zarrvectors",
    bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
    chunk_shape=(100.0, 100.0, 100.0),
    geometry_types=["graph"],
    ndim=3,
)
level_group = get_resolution_level(root, 0)
create_links_array(level_group, link_width=2, delta=0, sid_ndim=3)
```

Creating the family first fixes `directed`, `store` and `sid_ndim` for every
worker.

On each worker:

```python
import numpy as np
from zarr_vectors.building import write_link_attribute_cells, write_link_cells

# each record is a list of (chunk_coords, vertex_index) endpoints
batch = [
    [((0, 0, 0), 1), ((1, 0, 0), 2)],
    [((0, 0, 0), 3), ((1, 0, 0), 4)],
]
partition = write_link_cells(level_group, batch, 3, delta=0)   # 3 = sid_ndim
print(partition.num_links)

weights = np.array([0.5, 0.25], dtype="float32")   # one row per record
write_link_attribute_cells(level_group, "weight", weights, partition=partition, delta=0)
```

```text
2
```

`write_link_cells` changes only the cells its batch lands in and returns a
`LinkPartition` for that batch, which `write_link_attribute_cells` uses to
place the attribute rows beside their records. It does not update the
family's `num_links` and `num_physical_records`. Each cell update is a
read-modify-write, so workers must own disjoint source chunks; two workers
writing to one cell lose rows.

On the coordinator, after every worker has finished:

```python
from zarr_vectors.building import finalize_links, read_link_attributes, read_links, shard_store

part = finalize_links(level_group, delta=0)
print(part.num_links, part.num_physical_records)
print(read_links(level_group, delta=0))
print(read_link_attributes(level_group, "weight", delta=0))
print(shard_store("edges.zarrvectors"))
```

```text
2 2
[(((0, 0, 0), 1), ((1, 0, 0), 2)), (((0, 0, 0), 3), ((1, 0, 0), 4))]
[0.5  0.25]
{'arrays_sharded': 2, 'chunks_packed': 2, 'shard_shape': [8, 8, 8]}
```

`finalize_links` recounts every cell, so run it after all cells are written
and before sharding. Unreconciled counts fail
[consistency validation](validation_and_repair.md#check-a-store).

### How link rows and attribute rows line up

`read_links` and `read_link_attributes` both return rows in the same order:
offset segments sorted by name, then cells within each segment. Attribute row
`i` describes link record `i`; nothing else on disk records the pairing. In
that sort `+` and `-` come before `0`, so the intra-chunk segment `0.0.0`
comes after every cross-chunk segment.

With `store="duplicate"` a record is filed in several cells and is returned
once per copy; deduplicate, or read one location with `read_links_for_tuple`.
