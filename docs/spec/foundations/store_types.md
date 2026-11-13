# Store types and backends

Where a store's bytes can live, how `zarr-vectors-py` picks a storage
backend, and which extras each backend needs.

## Terms

**Backend**
: The Zarr v3 store (`zarr.abc.store.Store`) that carries a store's
  bytes. It is named by a string: `"local"`, `"obstore"`, `"fsspec"` or
  `"icechunk"`. Selection is in `zarr_vectors/core/backends/__init__.py`;
  construction is in `zarr_vectors/core/store.py`.

## Introduction

Every operation works the same on every backend. The backend affects
only latency, throughput and cost per request.

| Backend | Zarr store | Built with | Extra |
|---------|------------|------------|-------|
| `local` | `zarr.storage.LocalStore` | `LocalStore(path)` | none |
| `obstore` | `zarr.storage.ObjectStore` | `ObjectStore(obstore.store.from_url(url, **options))` | `obstore` (also in `cloud`) |
| `fsspec` | `zarr.storage.FsspecStore` | `FsspecStore.from_url(url, storage_options=options)` | `fsspec` (`fsspec`, `s3fs`, `gcsfs`, `adlfs`) |
| `icechunk` | Icechunk session store | an Icechunk repository session | `icechunk` (Python 3.12 or later) |

A pre-built Zarr store, such as `zarr.storage.MemoryStore`, can be
passed wherever a path is accepted. A writer given a pre-built store
does not pass its bounds or chunk shape to it, so call `create_store`
on the store first.

## Technical reference

### Entry points

```python
zarr_vectors.open(source, *, mode="r", storage=None)
zarr_vectors.create(target, *, schema, storage=None)
zarr_vectors.open_or_create(target, *, schema, storage=None, on_conflict="raise")
```

`source` and `target` may be a path, a URL or a pre-built store.
`StorageOptions(backend=None, options={})` names the backend and holds
the options for its constructor: `obstore` `from_url` keyword
arguments, `fsspec` `storage_options`, or Icechunk storage settings.

The lower-level `create_store` and `open_store` in
`zarr_vectors.core.store` take `backend=`, `storage_options=` and loose
`**backend_kwargs`, which are merged into `storage_options`. The local
backend and pre-built stores take no options; passing any raises
`TypeError`. The type writers and readers (`write_points`,
`read_points` and the others) take `backend=` but no options. For
credentials, rely on the environment or pass an open store.

### Backend selection

`resolve_backend_name` uses the first of:

1. the explicit backend (`StorageOptions.backend` or `backend=`);
2. the `ZARR_VECTORS_BACKEND` environment variable;
3. the URL scheme:

| Scheme | Backend |
|--------|---------|
| none, `file://`, or any scheme not listed below | `local` |
| `s3://`, `gs://`, `gcs://`, `az://`, `azure://`, `abfs://`, `http://`, `https://` | `obstore` if installed, else `fsspec`, else `StoreError` with an install hint |

An explicit backend applies to any scheme, so `backend="fsspec"`
reaches any filesystem `fsspec` has a driver for. Icechunk is used only
when named.

### Backend notes

| Backend | Notes |
|---|---|
| Local | One file per stored cell. [Sharding](../chunking/sharding.md) packs cells into fewer files. |
| Object stores | One object per stored cell, so larger chunks or sharding mean fewer requests (see [Chunk shape](../chunking/chunk_shape.md)). Credentials come from the options or the environment (for example `~/.aws/credentials`). |
| Icechunk | Opens a repository at a local path or a `memory://`, `s3://`, `gs://`/`gcs://` or `az://`/`azure://`/`abfs://` URL, with transactional, versioned writes. |
| ZIP | `zarr.storage.ZipStore(path, mode="r")` reads a store archived as ZIP, with its contents at the top of the archive. The writers do not produce ZIP stores. |

```python
import zarr_vectors as zv

ds = zv.open(
    "s3://my-bucket/scan.zarrvectors",
    storage=zv.StorageOptions(backend="obstore",
                              options={"region": "us-west-2"}),
)
```

### Validation

The backend does not change validation.
`zarr_vectors.validate.validate` accepts a path, a URL or an open
store, and level 1 reports an error when the store cannot be opened.
See [Validation](../validation/overview.md).
