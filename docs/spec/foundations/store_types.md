# Store types and backends

This page describes where a store's bytes can live, how
`zarr-vectors-py` picks a storage backend for a path or URL, and which
optional packages each backend needs.

## Terms

**Storage backend**
: The Zarr v3 store (`zarr.abc.store.Store`) that carries a store's
  bytes, built with the storage library's own constructor. Backend
  selection is in `zarr_vectors/core/backends/__init__.py`; the stores
  are built in `zarr_vectors/core/store.py`.

**Backend name**
: The string that selects a backend: `"local"`, `"obstore"`,
  `"fsspec"` or `"icechunk"`.

**URL scheme**
: The leading `scheme://` of a store URL, such as `s3://`. Used to pick
  a backend when none is named. A bare path and `file://` are local.

---

## Introduction

The data model and every operation are the same on every backend; the
backend affects only latency, throughput and cost per request. Each
backend is a plain Zarr v3 store:

| Backend name | Zarr store | Built with | Extra |
|--------------|------------|------------|-------|
| `local` | `zarr.storage.LocalStore` | `LocalStore(path)` | none |
| `obstore` | `zarr.storage.ObjectStore` | `ObjectStore(obstore.store.from_url(url, **options))` | `zarr-vectors[obstore]` (also installed by `zarr-vectors[cloud]`) |
| `fsspec` | `zarr.storage.FsspecStore` | `FsspecStore.from_url(url, storage_options=options)` | `zarr-vectors[fsspec]` (`fsspec`, `s3fs`, `gcsfs`, `adlfs`) |
| `icechunk` | the `icechunk` session store | an Icechunk repository session | `zarr-vectors[icechunk]` (Python 3.12 or later) |

A pre-built Zarr store, such as `zarr.storage.MemoryStore`, can also be
passed wherever a path is accepted.

---

## Technical reference

### Entry points

The supported entry points take a `StorageOptions`:

```python
zarr_vectors.open(source, *, mode="r", storage=None)
zarr_vectors.create(target, *, schema, storage=None)
zarr_vectors.open_or_create(target, *, schema, storage=None, on_conflict="raise")
```

`StorageOptions(backend=None, options={})` names the backend and holds
the options passed to the store constructor (`obstore` `from_url`
keyword arguments, `fsspec` `storage_options`, or Icechunk storage
settings). `source` and `target` may be a path, a URL or a pre-built
Zarr store.

The lower-level functions in `zarr_vectors.core.store` take the same
choices as keyword arguments:

```python
create_store(path, *, ..., backend=None, storage_options=None, **backend_kwargs)
open_store(path, mode="r", *, backend=None, storage_options=None,
           require_zv=True, **backend_kwargs)
```

Loose `**backend_kwargs` are merged into `storage_options`. The local
backend and pre-built stores take no options: passing any raises
`TypeError` rather than ignoring them. The type writers and readers
(`write_points`, `read_points` and the others) take `backend=` but no
store options; for credentials, rely on the environment or open the
store first and pass the handle.

### Backend selection

The backend is chosen in this order (`resolve_backend_name`):

1. the explicit backend (`StorageOptions.backend` or `backend=`);
2. the `ZARR_VECTORS_BACKEND` environment variable;
3. the URL scheme:

| Scheme | Backend |
|--------|---------|
| none, `file://` | `local` |
| `s3://`, `gs://`, `gcs://`, `az://`, `azure://`, `abfs://`, `http://`, `https://` | `obstore` if installed, else `fsspec`; else `StoreError` with an install hint |
| any other | `local` |

`SCHEMES_LOCAL` and `SCHEMES_OBJECT_STORE` in
`zarr_vectors/core/backends/__init__.py` hold the scheme lists. An
explicit backend applies to any scheme, so
`backend="fsspec"` reaches any filesystem `fsspec` has a driver for.
Icechunk is used only when named.

### Local file system

The local backend writes one file per stored cell, with directories
for groups and arrays (see
[Directory structure](../layout/directory_structure.md)). Stores with
many small cells produce many files; the
[sharding codec](../chunking/sharding.md) packs cells into fewer files.

### Object stores

With `obstore` or `fsspec` installed, an object-store URL works like a
path. Credentials come from the store options or from the
environment, as the storage library reads them (for example
`~/.aws/credentials` or `gcloud auth application-default login`):

```python
import zarr_vectors as zv

ds = zv.open(
    "s3://my-bucket/scan.zarrvectors",
    storage=zv.StorageOptions(backend="obstore",
                              options={"region": "us-west-2"}),
)
```

Each stored cell is one object, so larger chunks, or sharding, mean
fewer requests. See [Chunk shape](../chunking/chunk_shape.md).

### Icechunk

`backend="icechunk"` opens an Icechunk repository at a local path or a
`memory://`, `s3://`, `gs://`/`gcs://` or `az://`/`azure://`/`abfs://`
URL and works through a session on it, giving transactional, versioned
writes.

### In-memory and ZIP stores

A `zarr.storage.MemoryStore` holds a store in memory until the process
exits. When a writer is given a pre-built store, it does not pass its
bounds or chunk shape on to the new store, so create the store first:

```python
import numpy as np
import zarr
from zarr_vectors.core.store import create_store
from zarr_vectors.types.points import read_points, write_points

rng = np.random.default_rng(0)
positions = rng.uniform(0, 400, (1000, 3)).astype(np.float32)

store = zarr.storage.MemoryStore()
create_store(store, chunk_shape=(200.0, 200.0, 200.0),
             bounds=([0.0, 0.0, 0.0], [400.0, 400.0, 400.0]))
write_points(store, positions, chunk_shape=(200.0, 200.0, 200.0))
result = read_points(store)
print(result["vertex_count"])
```

```
1000
```

A store archived as ZIP, with the store's contents at the top of the
archive, can be read through `zarr.storage.ZipStore(path, mode="r")`
passed in the same way. The writers do not produce ZIP stores.

### Validation

The backend does not change validation: `zarr_vectors.validate.validate`
accepts a path, a URL or an open store handle, and level 1 reports an
error when the store cannot be opened. See
[Validation](../validation/overview.md).
