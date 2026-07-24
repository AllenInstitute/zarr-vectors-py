# Rechunking

## Terms

**Rechunking**
: The process of rewriting an existing ZVF store with a different
  `chunk_shape` (and optionally a different `bin_shape`). Rechunking
  produces a new store; the original is not modified in-place unless
  explicitly requested.

**In-place rechunking**
: Rechunking that overwrites the source store.  Supported by
  `zarr_vectors.rechunk.rechunk(path, spec)` when ``output`` is left
  as ``None``: the engine writes to a sibling ``<name>.rechunked``
  directory, renames the source to ``<name>.backup``, moves the new
  store into place, and deletes the backup.  The source path
  identifier is preserved across the swap.

**Out-of-place rechunking**
: Rechunking that writes to a new destination path. The original store
  is preserved. This is the default behaviour.

**Chunk boundary crossing**
: When rechunking from a fine `chunk_shape` to a coarser one, vertices
  that were in separate source chunks may end up in the same destination
  chunk. When rechunking from a coarse `chunk_shape` to a finer one,
  vertices that were in the same source chunk may end up in different
  destination chunks.

**Link invalidation**
: When rechunking, every link must be recomputed, because the chunk
  boundaries change. A record's `<offsets>` array is determined by the
  gap between its endpoints' chunks, so both the offsets naming its
  array and the chunk-local `vi_k` within it change.

---

## Introduction

Because `chunk_shape` controls the physical file layout, it cannot be
modified without rewriting the underlying data. Rechunking is the operation
that achieves this. It reads all vertices from the source store (chunk by
chunk), re-assigns each vertex to its chunk in the destination layout, and
writes the result.

Rechunking is relatively rare in practice: most users choose `chunk_shape`
once before writing and do not change it. The most common scenario is
discovering, after writing a large dataset, that the chosen `chunk_shape`
is poorly matched to the actual query access pattern.

Rechunking is also necessary when importing data from a different format
(e.g. TRK or LAS) that has its own chunking scheme, and when building a
ZVF store that will be served from cloud storage where chunk size
requirements differ from those for local access.

---

## Technical reference

### Rechunking API

```python
from zarr_vectors.core.rechunk import rechunk_store

rechunk_store(
    source="scan.zarrvectors",
    dest="scan_rechunked.zarrvectors",
    chunk_shape=(500.0, 500.0, 500.0),   # new chunk shape
    bin_shape=(100.0, 100.0, 100.0),     # new bin shape (optional)
)
```

If `bin_shape` is omitted, the new bin shape defaults to `chunk_shape / 4`
per axis (isotropic), or the closest valid divisor.

### What rechunking does

Rechunking proceeds in three phases:

**Phase 1 — Vertex repartitioning.** For each vertex in the source store,
compute its destination chunk coordinate in the new chunk grid:

```python
new_chunk_coord = tuple(
    int(math.floor(p[d] / new_chunk_shape[d])) for d in range(D)
)
```

Vertices are buffered per destination chunk. Once all vertices for a
destination chunk have been collected, the chunk is sorted into fragment order
(by new bin coordinate) and written.

**Phase 2 — Attribute and link re-assignment.** Per-vertex attributes are
re-ordered to match the new vertex ordering. For polyline and streamline
stores, the `links/<delta>/` family is also recomputed: each record is
re-partitioned against the new chunk grid and re-filed under the offsets
naming the new gap between its endpoints. Records that were intra-chunk
may become cross-chunk and vice versa — they simply move between offsets
arrays within the same family.

**Phase 3 — Object index rebuild.** For stores with an `object_index`, the
index is rebuilt from scratch against the new chunk layout.

### Rechunking and resolution levels

Rechunking operates on a single resolution level. To rechunk a multi-
resolution store, rechunk each level separately:

```python
from zarr_vectors.core.rechunk import rechunk_store

for level in [0, 1, 2]:
    rechunk_store(
        source=f"scan.zarrvectors/resolution_{level}",
        dest=f"scan_rechunked.zarrvectors/resolution_{level}",
        chunk_shape=(500.0, 500.0, 500.0),
        bin_shape=(100.0, 100.0, 100.0),
        level=level,
        source_root="scan.zarrvectors",       # for metadata
        dest_root="scan_rechunked.zarrvectors",
    )
```

Alternatively, rechunk only the base level and rebuild coarser levels:

```python
from zarr_vectors.core.rechunk import rechunk_store
from zarr_vectors.multiresolution.coarsen import build_pyramid

rechunk_store("scan.zarrvectors", "scan_rechunked.zarrvectors",
              chunk_shape=(500.0, 500.0, 500.0), levels=[0])

build_pyramid("scan_rechunked.zarrvectors", factors=[(2.0, 1.00), (4.0, 2.00)])
```

### Memory usage

Rechunking requires buffering all vertices that will land in a single
destination chunk. For a store with uniform vertex density and isotropic
rechunking from `chunk_shape=(200,…)` to `chunk_shape=(500,…)`, each
destination chunk receives approximately `(500/200)³ ≈ 15.6×` as many
vertices as the average source chunk. Peak memory usage scales with
`new_chunk_volume / old_chunk_volume × avg_vertices_per_source_chunk`.

For very large rechunking ratios (e.g. 10× per axis), use the streaming
rechunker which writes destination chunks as they fill rather than
buffering all source data:

```python
rechunk_store(
    source="scan.zarrvectors",
    dest="scan_rechunked.zarrvectors",
    chunk_shape=(2000.0, 2000.0, 2000.0),
    streaming=True,          # lower memory; slower due to multiple source passes
)
```

### CLI usage

The `zarr-vectors rechunk` CLI subcommand lives in the companion package
**`zarr-vectors-tools`**.

### What is preserved vs recomputed

| Array | Preserved? | Notes |
|-------|-----------|-------|
| `vertices/` values | Yes | Same positions, different chunk assignment |
| `vertex_fragments/` | Recomputed | Fragment partition changes with new bin grid |
| `link_fragments/` | Recomputed | Parallel to `vertex_fragments/` for `links/0/` |
| `links/<delta>/<offsets>/` | Recomputed | Both the offsets naming each record's array and its chunk-local vertex indices change |
| `link_attributes/<name>/<delta>/<offsets>/` | Recomputed | Mirrors the offsets arrays cell-for-cell |
| `attributes/` values | Yes | Reordered to match new vertex ordering |
| `object_index/` | Recomputed | Chunk coordinates *and* fragment indices change |
| `object_attributes/` | Preserved | Per-object, not per-chunk |
| `groupings/` | Preserved | Per-group, not per-chunk |
| Root `.zattrs` | Updated | `chunk_shape` and `base_bin_shape` updated |
| Per-level `.zattrs` | Updated | `bin_shape` updated; `bin_ratio` unchanged |

### Rechunking only `bin_shape`

If only `bin_shape` changes (i.e. `chunk_shape` is the same), the work
is cheaper because vertex positions do not change chunk assignment —
only the fragment partition within each chunk does. `rebin_level`
handles this case:

```python
from zarr_vectors.rechunk.rebin import rebin_level

rebin_level(
    "scan.zarrvectors",
    target_bin_shape=(25.0, 25.0, 25.0),   # finer bins within the same chunks
    level=0,
)
```

**`rebin_level` is point-cloud-only.** It raises `StoreError` unless
`geometry_types == ["point_cloud"]` exactly. Re-binning re-sorts
vertices *within* a chunk, which is safe only when the fragment layout
is derivable from positions alone; every other type carries per-object
or per-bundle fragment semantics that the re-sort would flatten
irreversibly. For those, use `zarr_vectors.rechunk.rechunk()`, which
rebuilds the object manifests too.

The links question therefore does not arise: a store `rebin_level`
accepts has no links, no `object_index/`, and no manifests to rewrite.

`rebin_level` is in-place: it rewrites `vertex_fragments/` and re-sorts
vertices within each chunk, and updates root `base_bin_shape` (when
`level == 0`) plus the level's `bin_shape` / `bin_ratio`. Chunk file
paths are unaffected.

> **Per-vertex attribute arrays are NOT rewritten.** `rebin_level`
> leaves `attributes/` untouched while shuffling vertex order within
> each chunk, so any post-write per-vertex attribute array is left
> **misaligned** with its vertices and must be regenerated by the
> caller. This is a documented property of the function, not a bug —
> but it means `rebin_level` is unsafe to run on a store with
> per-vertex attributes unless you regenerate them afterwards.

`target_bin_shape` must evenly divide `chunk_shape` or the call raises.

### Validation after rechunking

Run the full L5 validator after rechunking to confirm the new store is
consistent:

```python
from zarr_vectors.validate import validate
result = validate("scan_rechunked.zarrvectors", level=5)
```

Pay particular attention to L3 checks (fragment offset consistency) and L4 checks
(cross-chunk link validity), as these are most likely to expose bugs in
the rechunking implementation.
