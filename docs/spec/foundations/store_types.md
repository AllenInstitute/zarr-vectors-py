# Store types and the backend layer

## Terms

**Storage backend**
: A concrete Zarr v3 `zarr.abc.store.Store` for a storage system (local
  file system, S3, GCS, Azure, HTTP, in-memory) plus the transactional
  `icechunk` store. zarr-vectors builds these using zarr's / obstore's /
  icechunk's own native constructors — there is no custom byte-level
  abstraction layer. Backend *selection* (URL scheme → driver name)
  lives in
  [`zarr_vectors/core/backends/__init__.py`](../../../zarr_vectors/core/backends/__init__.py);
  the store objects are built in
  [`zarr_vectors/core/store.py`](../../../zarr_vectors/core/store.py).

**Backend name**
: A short string that selects a backend: `"local"`, `"obstore"`,
  `"fsspec"`, or `"icechunk"`. Passed as the `backend=` kwarg to public
  store entry points or set globally via the `ZARR_VECTORS_BACKEND`
  environment variable (icechunk is explicit-only, never auto-detected).

**URL scheme**
: The leading `scheme://` of the store URL (e.g. `s3://bucket/foo`).
  Used for backend auto-detection when no explicit `backend=` is given.
  Bare paths (no scheme) and `file://` resolve to the local backend.

**ZIP store**
: A Zarr store that packages all chunk files into a single ZIP archive.
  Read-only via `zarr.storage.ZipStore`; useful for archiving or
  transferring a complete store as a single file. Not produced by
  `zarr-vectors` writers directly.

**In-memory store**
: Backed by a Python dict. Useful for unit tests and intermediate
  processing pipelines that do not need disk persistence.

**Consolidated metadata**
: An optional `.zmetadata` file at the store root that caches every
  `zarr.json` / `.zattrs` file in the hierarchy. Avoids many small
  metadata reads on high-latency stores (S3/GCS).

---

## Introduction

ZV stores are backend-agnostic: the same `create_store` / `open_store` /
`open_zv` calls work whether the data lives on a local SSD, a ZIP
archive, an in-memory dict, or a cloud object store. The backing store
type affects performance characteristics (latency, throughput, cost per
request) but not the data model or the semantics of any operation.

Each backend is a plain Zarr v3 store built by its own native
constructor — no custom adapter classes:

| Backend name | Zarr store | Built via | Optional dep |
|--------------|------------|-----------|--------------|
| `local`      | `zarr.storage.LocalStore` | `LocalStore(path)` | none (always available) |
| `obstore`    | `zarr.storage.ObjectStore` | `ObjectStore(obstore.store.from_url(url, **opts))` | `obstore` (Rust object-store bindings; preferred for cloud) |
| `fsspec`     | `zarr.storage.FsspecStore` | `FsspecStore.from_url(url, storage_options=…)` | `fsspec` + scheme driver (`s3fs`, `gcsfs`, `adlfs`) |
| `icechunk`   | `icechunk` session store | `Repository.open(...).session().store` | `icechunk` (transactional/versioned) |

For most users the relevant choice is between **local** (development,
HPC analysis) and **cloud** (sharing, Neuroglancer serving). The two
cloud byte backends are interchangeable from the caller's perspective —
the library picks `obstore` automatically when it's installed and falls
back to `fsspec` otherwise. Use `backend="icechunk"` for transactional,
commit-versioned storage on top of any of these.

---

## Technical reference

### Public entry points

All three entry points accept `backend=`, `storage_options=`, and
loose `**backend_kwargs`:

```python
from zarr_vectors.core.store import create_store, open_store
from zarr_vectors.lazy import open_zv

create_store(path, *, ..., backend=None, storage_options=None, **backend_kwargs) -> Group
open_store(path, mode="r", *, backend=None, storage_options=None, **backend_kwargs) -> Group
open_zv(path, *, backend=None, storage_options=None, **backend_kwargs) -> ZVStore
```

`backend` is one of `"local"` / `"obstore"` / `"fsspec"` / `"icechunk"`
or `None` for auto-detect. `storage_options` is a dict of credentials /
options forwarded verbatim to the store constructor (fsspec
`storage_options`, obstore `from_url` kwargs, or icechunk `*_storage`
kwargs). Loose `**backend_kwargs` are merged into `storage_options` for
back-compat, so `key=...`, `region=...`, `anon=True` also work as bare
kwargs.

### URL scheme dispatch

When `backend=None`, the resolution order in
[`backends.resolve_backend_name`](../../../zarr_vectors/core/backends/__init__.py)
is:

1. **Explicit `backend=` kwarg** — always wins.
2. **`ZARR_VECTORS_BACKEND` environment variable** — second.
3. **URL-scheme auto-detect**:

| Scheme(s) | Resolved backend |
|-----------|------------------|
| (no scheme), `file://` | `local` |
| `s3://`, `gs://`, `gcs://`, `az://`, `azure://`, `abfs://`, `http://`, `https://` | `obstore` if installed; else `fsspec`; else `StoreError` with an install hint |

`SCHEMES_LOCAL` and `SCHEMES_OBJECT_STORE` in
[`backends/__init__.py`](../../../zarr_vectors/core/backends/__init__.py)
are the canonical tables.

### Local file system store

The default. Pass any local path as a string or `pathlib.Path`:

```python
from zarr_vectors.types.points import write_points, read_points

write_points("scan.zarrvectors", positions, chunk_shape=(200., 200., 200.))
result = read_points("scan.zarrvectors")
```

Internally the local backend uses `zarr.storage.LocalStore`. Each chunk
is one file; directories correspond to group and array paths. A typical
tree:

```
scan.zarrvectors/
├── zarr.json
├── .zattrs
├── 0/
│   ├── zarr.json
│   ├── .zattrs
│   ├── vertices/
│   │   ├── zarr.json
│   │   └── c/
│   │       ├── 0/0/0
│   │       └── …
│   └── …
└── …
```

**Performance notes:**
- File creation overhead dominates write performance on file systems
  with high inode metadata costs (many NFS mounts, HDD-backed systems).
  Use larger `chunk_shape` values to reduce file count.
- Linux `ext4` and `xfs` handle large directories well; `FAT32` and
  some network shares have per-directory file count limits. For more
  than ~100 000 chunks per array, use the
  [sharding codec](../chunking/sharding.md).
- On HPC systems with Lustre or GPFS, stripe the `.zarrvectors`
  directory for write parallelism:
  `lfs setstripe -c 8 scan.zarrvectors`.

### Cloud object stores

The preferred cloud backend is `obstore` (Rust bindings, parallel
range reads). It's enabled automatically when installed:

```bash
pip install "zarr-vectors[obstore]"     # preferred
# OR
pip install "zarr-vectors[cloud]"       # fsspec + scheme drivers (fallback)
```

**S3 (read):**

```python
from zarr_vectors.types.points import read_points

result = read_points(
    "s3://open-neuro-data/datasets/synchrotron.zarrvectors",
    level=2,
)
```

If `obstore` is installed, the URL scheme auto-routes to it; otherwise
`fsspec` is used. Force a specific backend with `backend=`:

```python
read_points(
    "s3://my-bucket/scan.zarrvectors",
    backend="obstore",
    region="us-west-2",
)
```

**GCS / Azure (read):**

```python
read_points("gs://my-bucket/scan.zarrvectors")        # obstore or fsspec
read_points("az://account/container/scan.zarrvectors")
```

**Authenticated writes:** pass credentials via `storage_options=`
(or loose `**backend_kwargs`, which merge into it) — they are forwarded
to the store constructor — or rely on standard ambient credentials
(`~/.aws/credentials`, `gcloud auth application-default`, environment
variables).

```python
from zarr_vectors.types.points import write_points

write_points(
    "s3://my-bucket/scan.zarrvectors",
    positions,
    chunk_shape=(500., 500., 500.),   # larger chunks = fewer S3 objects
    backend="obstore",
    region="us-east-1",
)
```

**Performance notes for object stores:**
- Each ZV chunk is one object on cloud. Minimise the per-chunk
  count by increasing `chunk_shape` relative to your typical query
  bbox (target ≥ 100 KB compressed per chunk).
- Enable **consolidated metadata** (below) to collapse store-open
  latency on stores with many resolution levels / attribute arrays.
- For Neuroglancer serving, use `zv-ngtools` which implements its
  own request batching on top of the same backend layer.

### Selecting the backend explicitly

Use the `backend=` kwarg when you need to:

- Override the auto-detect (e.g. force `fsspec` even though `obstore`
  is installed, for compatibility with a custom filesystem):

  ```python
  open_store("s3://bucket/scan.zarrvectors", backend="fsspec")
  ```

- Use a backend with a non-cloud URL (e.g. an `fsspec` driver pointed
  at SFTP):

  ```python
  open_store("sftp://host/path/scan.zarrvectors", backend="fsspec")
  ```

Or set `ZARR_VECTORS_BACKEND=fsspec` once in the environment to apply
the override for the whole process.

### ZIP store

ZIP stores are read-only at the Zarr layer and not produced by
`zarr-vectors` writers directly. To distribute a store as a single
file, write to a local store and then archive:

```bash
zip -r scan.zarrvectors.zip scan.zarrvectors/
```

Open the archive via `zarr.storage.ZipStore` directly (not through
`open_store`):

```python
import zarr
store = zarr.storage.ZipStore("scan.zarrvectors.zip", mode="r")
root  = zarr.open_group(store, mode="r")
```

### In-memory store

Useful for tests and one-off pipelines. Pass a `zarr.storage.MemoryStore`
through to the Zarr layer directly:

```python
import zarr
from zarr_vectors.types.points import write_points, read_points

store = zarr.storage.MemoryStore()
write_points(store, positions, chunk_shape=(200., 200., 200.))
result = read_points(store)
```

In-memory stores are discarded when the Python process exits.

### Consolidated metadata

On stores with many groups (large pyramids, many attribute arrays),
opening the store issues one metadata request per Zarr group and array.
On S3 with ~50 ms per request, this adds noticeable latency.

Consolidate after writing:

```python
import zarr
from zarr_vectors.core.store import open_store

root = open_store("scan.zarrvectors", mode="r+")
zarr.consolidate_metadata(root.zarr_group.store)
```

This writes a single `.zmetadata` file at the store root; subsequent
`open_store()` calls read from it instead of issuing individual
metadata requests. Regenerate after any structural change (e.g. adding
a resolution level).

### Choosing a store type

| Scenario | Recommended setup |
|----------|-------------------|
| Local analysis / development | Local backend (default) |
| HPC batch processing         | Local backend on Lustre/GPFS with striping |
| Archiving a completed dataset | Local write, then `zip -r` |
| Cloud sharing / Neuroglancer serving | `obstore` (or `fsspec` fallback) + consolidated metadata |
| Unit tests | In-memory store via `zarr.storage.MemoryStore` |
| Very large stores with many small chunks | Local or cloud + the [sharding codec](../chunking/sharding.md) |

### Capability tokens

The backend layer is independent of the
[format capability tokens](../layout/root_metadata.md) stamped on
`RootMetadata.format_capabilities` — backends carry data bytes, not
format semantics. See the capability list for `CAP_MULTISCALE_LINKS`,
`CAP_PRESERVED_OBJECT_IDS`, `CAP_SHARED_VERTEX_GROUPS`.
