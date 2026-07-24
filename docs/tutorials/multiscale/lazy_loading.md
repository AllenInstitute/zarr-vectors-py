# Lazy loading

The ZVF read functions (`read_points`, `read_polylines`, etc.) are eager:
they fetch and return all requested data immediately. For large stores or
remote datasets, an eager read of the full store is impractical.

The **lazy API** provides `open_zv`, which opens the store metadata
without reading any array data. Vertex and attribute data are fetched only
when you call `.compute()`. This is the recommended access pattern for:

- Stores too large to fit in memory.
- Remote stores (S3, GCS) where each array fetch is a network request.
- Interactive viewers that need the coarsest level first and finer levels
  on demand.
- Analysis pipelines that filter by metadata before deciding which data
  to load.

---

## Opening a store lazily

```python
from zarr_vectors.lazy import open_zv

# Opens metadata only — no vertex data fetched
store = open_zv("synchrotron.zarrvectors")

print(store.geometry_types)   # ['point_cloud']
print(store.ndim)             # 3
print(store.chunk_shape)      # (200.0, 200.0, 200.0)
print(store.bin_shape)        # (200.0, 200.0, 200.0)
print(store.levels)           # [0, 1, 2, 3]
print(store.bounds)           # ([0.07, 1.52, 0.12], [399.2, 398.1, 399.8])
print(store.zv_version)       # '0.9.0'
```

`ZVStore` is a **metadata handle**. Its full public surface is:

| Member | Kind | Meaning |
|--------|------|---------|
| `levels` | property | `list[int]` of resolution levels present |
| `ndim` | property | spatial dimensionality |
| `chunk_shape` | property | chunk shape in world units |
| `bin_shape` | property | level-0 bin shape |
| `base_bin_shape` | property | base bin shape from root metadata |
| `bins_per_chunk` | property | bins per chunk per axis |
| `bounds` | property | `(lo, hi)` store bounds |
| `geometry_types` | property | `list[str]` |
| `zv_version` | property | format version string |
| `headers` | property | parsed store headers |
| `path` / `url` | property | store location |
| `object_levels(oid)` | method | levels at which `oid` is present |
| `set_backend(...)` | method | swap the storage backend |
| `store[level]` | `__getitem__` | the `ZVLevel` for that level |

There is no `store.read(...)`, no `store.iter_chunks(...)`, and no
`raw_array(...)`. `ZVStore` is also **not** a context manager and has no
`close()` — there is nothing to release, since it holds no open handles
or cache of its own.

Opening a remote store is identical — pass a cloud URL:

```python
from zarr_vectors.lazy import open_zv

# The backend layer auto-routes cloud URLs via obstore (or fsspec).
store = open_zv("s3://open-neuro/synchrotron.zarrvectors")
print(store.levels)   # metadata only
```

`open_zv` accepts `backend=` and `storage_options=` for explicit backend
selection and credentials; any extra keyword arguments are forwarded to
the backend. It does **not** take `cache_size`, `n_workers`, or
`prefetch`.

---

## Working with a level

Index the store to get a `ZVLevel`. This is where the data lives.

```python
level = store[0]

print(level.level_index)     # 0
print(level.vertex_count)    # 500000 — from metadata, no data read
print(level.chunk_count)     # 8
print(level.chunk_keys)      # [(0,0,0), (0,0,1), (0,1,0), ...]
print(level.bin_ratio)       # (1, 1, 1) — or None
```

Note that `ZVLevel.bin_shape` and `ZVLevel.bin_ratio` are **optional**:
they read the per-level metadata and return `None` when the level does
not declare them (a single-level store written without a pyramid
typically does not). `ZVStore.bin_shape`, by contrast, always resolves —
it falls back to `base_bin_shape` or `chunk_shape`. Guard accordingly.

### Level-of-detail selection

There is no automatic level-selection helper. Pick a level from the
metadata yourself, comparing each level's bin shape against a target
resolution and falling back to the store-wide value when a level does not
declare one:

```python
def level_for_resolution(store, target):
    """Coarsest level whose bin shape is still finer than `target`."""
    best = store.levels[0]
    for lv in store.levels:
        shape = store[lv].bin_shape or store.bin_shape
        if max(shape) <= target:
            best = lv
    return best

lv = level_for_resolution(store, 200.0)
print(lv, store[lv].vertex_count)
```

---

## Fetching data

Vertices and attributes are lazy collections. Nothing is read until
`.compute()`.

```python
level = store[0]

# Lazy handle — no I/O yet
verts = level.vertices

# Now fetch
positions = verts.compute()      # (N, D) numpy array
print(positions.shape)
```

### Attributes

`level.attributes` is a **dict-like proxy**, not a dict: it supports
`acc[name]` and `name in acc` only. It is not iterable — do not call
`list()` or `for ... in` on it, as it has no `__iter__` and will not
terminate.

```python
attrs = level.attributes

if "intensity" in attrs:
    values = attrs["intensity"].compute()   # (N,) numpy array
    print(values.shape, values.dtype)
```

### Filtering before fetching

`level.filter(...)` returns a `ZVView` — still lazy. This is the way to
restrict a read spatially or by object:

```python
import numpy as np

view = level.filter(
    bbox=(np.array([0., 0., 0.]), np.array([200., 200., 200.])),
)

# Still no data read. Materialise either the vertices...
positions = view.vertices.compute()
print(positions.shape)

# ...or the whole filtered result
result = view.compute()
print(result["vertex_count"])
print(result["positions"].shape)
```

`filter` accepts `bbox`, `object_ids`, and `group_ids`, and views can be
chained with a further `.filter(...)`. `view.compute()` returns a dict
with `positions`, `vertex_count`, and — where the type has them —
`object_ids`.

---

## Parallel and out-of-core access with Dask

`to_delayed()` is the supported way to process a level chunk by chunk:
it returns one delayed object per chunk, each yielding that chunk's
`(M, D)` vertex array when computed.

```python
delayeds = store[0].vertices.to_delayed()
print(len(delayeds))          # one per chunk

# With Dask installed these are dask.delayed objects; without it, a
# synchronous fallback with the same .compute() interface.
first_chunk = delayeds[0].compute()
print(first_chunk.shape)
```

This gives you memory-bounded streaming: process one delayed at a time
rather than materialising the level.

```python
total = 0
for d in store[0].vertices.to_delayed():
    total += len(d.compute())
print(total)
```

---

## Per-object level lookup

For discrete-object types (polylines, graphs, skeletons, meshes) in an
ID-preserving pyramid, `object_levels` reports the levels at which an
object is present — useful for a viewer choosing an LOD per object.

```python
print(store.object_levels(0))    # e.g. [0, 1, 2]
print(store[0].has_object(0))    # True
print(store[0].has_object(999))  # False
```

`object_levels` returns `[]` for an object that is present nowhere.

Both are meaningful only where an `object_index/` exists. An
undifferentiated point cloud has no object index, and `has_object` raises
`KeyError` rather than returning `False` on such a store — check
`geometry_types` first if the type is not known ahead of time.

---

## Fragment indices

To inspect a chunk's fragment index directly, use
`read_fragment_index` from `zarr_vectors.encoding.fragments`, passing the
level's underlying group:

```python
from zarr_vectors.encoding.fragments import read_fragment_index
from zarr_vectors.core.store import open_store, get_resolution_level

root = open_store("scan.zarrvectors", mode="r")
level_group = get_resolution_level(root, 0)

fidx = read_fragment_index(level_group, "vertex_fragments", (2, 3, 1))
print(fidx.num_fragments)          # F: fragments in chunk (2,3,1)
print(fidx.num_range_fragments)    # R: how many are range fragments
print(fidx.is_range(0))            # bool — cheap, single bit lookup
start, count = fidx.range(0)       # row range, if fragment 0 is a range
rows = fidx.indices(0)             # explicit row indices otherwise
```

`ChunkFragmentIndex` exposes `num_fragments`, `num_range_fragments`,
`is_range(f)`, `range(f)`, `indices(f)`, and `indices_view(f)`. No
fragment payload is materialised until `range`/`indices`/`indices_view`
is called.

For the vertex fragment index specifically, `core.arrays` offers a
narrower helper that needs no array name:

```python
from zarr_vectors.core.arrays import read_vertex_fragment_index

fidx = read_vertex_fragment_index(level_group, (2, 3, 1))
```

See [Fragment-index arrays](../../spec/layout/fragment_index_arrays.md) for the
byte layout and the full `ChunkFragmentIndex` API.

---

## Performance on object stores

Remote stores have per-request latency (~50–200 ms for S3). To keep
request counts down:

1. Read metadata first (`store.levels`, `store[lv].vertex_count`) and
   decide *which* level you need before fetching any vertices.
2. Filter before computing — `level.filter(bbox=...)` restricts what
   `.compute()` will fetch.
3. Prefer a coarse level for overviews; only drop to level 0 for detail.
4. Use `to_delayed()` with a Dask scheduler to overlap fetches.

Caching and parallelism are properties of the backend and the Dask
scheduler you run under, not options on `open_zv`.
