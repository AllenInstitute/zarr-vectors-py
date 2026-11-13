# Cloud stores

This page shows how to read and write a store on Amazon S3, Google Cloud
Storage, Azure Blob Storage or an HTTP server. The calls are the same as for a
local store; only the URL changes, plus a `zv.StorageOptions` where the
backend needs settings. The backend layer is specified in
[Store types](../spec/foundations/store_types.md).

## Install a cloud backend

```bash
pip install "zarr-vectors[obstore]"   # obstore; [cloud] installs the same
pip install "zarr-vectors[fsspec]"    # fsspec with s3fs, gcsfs and adlfs
```

A cloud URL with neither backend installed raises `StoreError`.

## Choose the backend

`zv.open`, `zv.create` and `zv.open_or_create` take
`storage=zv.StorageOptions(backend=..., options=...)`. The first of these
that is set picks the backend:

1. `StorageOptions.backend`.
2. The `ZARR_VECTORS_BACKEND` environment variable.
3. The URL scheme:

| URL scheme | Backend |
|------------|---------|
| none, `file` | local |
| `s3`, `gs`, `gcs`, `az`, `azure`, `abfs`, `http`, `https` | obstore if installed, otherwise fsspec |
| any other, such as `sftp` | local; set `backend="fsspec"` instead |

## Pass credentials

`StorageOptions.options` goes to the backend unchanged, so the option names
are the backend library's own:

```python
import os

import zarr_vectors as zv

s3_keys = zv.StorageOptions(backend="obstore", options={
    "access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
    "secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
    "region": "us-east-1",
})
ds = zv.open("s3://my-bucket/scan.zarrvectors", storage=s3_keys)
```

| Setting | obstore | fsspec |
|---|---|---|
| S3 keys | `access_key_id`, `secret_access_key` | `key`, `secret` |
| Public bucket | `skip_signature=True` | `anon=True` |

Settings you leave out, such as credentials from the environment, are up to
the backend library. A `Dataset` keeps the options it was opened with; to
change them, open the URL again. The builder functions `open_store` and
`create_store` take `backend=` and `storage_options=` instead.

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
ds.build_pyramid(factors=[(2.0, 1.0)], chunk_scale_factors=[2])
```

Object stores charge per request, and each cell of each array is a separate
object unless cells are packed. Choose the cell size as in
[Choosing a layout](choose_chunk_and_bin.md). Pass `chunk_scale_factors` to
`build_pyramid`: without it, coarse levels keep the level-0 chunk shape and
take as many objects as level 0.

### Packing

`Layout.pack="auto"`, the default, packs cells into Zarr v3 shards on an
object store and not on a local path. A `Dataset` write uses at most four
cells per axis per shard. Pass `pack=True` or `pack=False` to choose.

A store written locally and then uploaded is not packed. Pack it before
uploading with `shard_store`:

```python
import numpy as np
import zarr_vectors as zv
from zarr_vectors.building import get_shard_info, shard_store

rng = np.random.default_rng(0)
local = zv.create("scan.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    kind="point_cloud",
    layout=zv.Layout(cells=8),
))
local.add_points(rng.uniform(0, 1000, (100_000, 3)).astype(np.float32))

print(shard_store("scan.zarrvectors", shard_shape=4))
print(get_shard_info("scan.zarrvectors")["shard_count"])
```

```text
{'arrays_sharded': 2, 'chunks_packed': 1024, 'shard_shape': [4, 4, 4]}
16
```

`shard_shape` counts cells: an integer for every axis, or a tuple.
`reshard(path, shape)` changes the shape and `unshard_store(path)` undoes
it. The result is plain Zarr v3; see
[Sharding](../spec/chunking/sharding.md).

## Read from a remote store

`ds.levels`, `Level.vertex_count` and `Level.objects.count` come from
metadata. `Query.explain()` and `Query.cells()` show what a read will fetch
before it runs. [Deferred reads and out-of-core
access](../tutorials/multiscale/lazy_loading.md) covers reading cell by cell.

## Consolidate the metadata

Opening a store reads one `zarr.json` per group and array. Consolidating
copies them into the root `zarr.json`, so they are read in one request. Call
Zarr on the store's group:

```python
import warnings

import zarr
import zarr_vectors as zv

ds = zv.open("scan.zarrvectors", mode="r+")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")      # not part of the Zarr v3 spec
    zarr.consolidate_metadata(ds.store.zarr_group.store)
```

Other Zarr implementations may ignore consolidated metadata. Consolidate
again after adding a level or an attribute array.

## Write links from parallel workers

A coordinator creates the link family, each worker writes its own cells, and
the coordinator then reconciles the counts and shards the store.

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
weights = np.array([0.5, 0.25], dtype="float32")              # one per record
write_link_attribute_cells(level_group, "weight", weights, partition=partition, delta=0)
```

Workers must own disjoint source chunks: two workers writing one cell lose
rows.

On the coordinator, after every worker has finished:

```python
from zarr_vectors.building import finalize_links, read_links, shard_store

part = finalize_links(level_group, delta=0)
print(part.num_links, read_links(level_group, delta=0))
shard_store("edges.zarrvectors")
```

```text
2 [(((0, 0, 0), 1), ((1, 0, 0), 2)), (((0, 0, 0), 3), ((1, 0, 0), 4))]
```

`finalize_links` recounts every cell, so run it after all writes and before
sharding. `read_links` and `read_link_attributes` return rows in the same
order, so attribute row `i` belongs to link record `i`.
