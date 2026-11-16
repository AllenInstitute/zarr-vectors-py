# Rechunking

## Terms

**Rechunking**
: Rewriting an existing Zarr Vectors store so that its chunk layout differs
  from the source's. Zarr Vectors has three distinct operations that change a
  layout, because "layout" means three different things:

  | Operation | Changes | In place? | Function |
  |-----------|---------|-----------|----------|
  | **Spatial rechunk** | the spatial `chunk_shape` | no — writes `output` | `rechunk_spatial` |
  | **Dimension rechunk** | prefixes every chunk key with a bin of a non-spatial dimension (group / object id / attribute value) | yes by default, or to `output` | `rechunk`, `rechunk_by_attribute` |
  | **Re-bin** | the bin grid inside each chunk (point clouds only) | yes | `rebin_level` |

**In-place rechunking**
: `zarr_vectors.rechunk.rechunk(path, spec)` with `output=None`: the engine
  writes a sibling `<name>.rechunked` directory, renames the source to
  `<name>.backup`, moves the new store into place, and deletes the backup. The
  source path is preserved across the swap, but **the original data is gone
  afterwards**. Pass `output=` to keep it. (`rechunk_spatial` has no in-place
  mode.)

**Chunk boundary crossing**
: When rechunking from a fine `chunk_shape` to a coarser one, vertices
  that were in separate source chunks may end up in the same destination
  chunk. When rechunking from a coarse `chunk_shape` to a finer one,
  vertices that were in the same source chunk may end up in different
  destination chunks.

**Link invalidation**
: When the spatial chunk grid changes, every link must be recomputed. A
  record's `<offsets>` array is determined by the gap between its endpoints'
  chunks, so both the offsets naming its array and the chunk-local `vi_k` within
  it change.

**Prefix dimension**
: The extra leading chunk-key axis a dimension rechunk adds, so that all objects
  in one group / attribute value are contiguous on disk
  (`(prefix_bin, x, y, z)`). Recorded in the level's `chunk_dims`; see
  [Attribute-based chunking](attribute_chunking.md).

---

## Introduction

Because `chunk_shape` controls the physical file layout, it cannot be modified
without rewriting the underlying data. `rechunk_spatial` is the operation that
does so, for every geometry kind and every resolution level. It exists because
chunk size is chosen at ingest, but the right size depends on the *density* of
what was ingested: the same grid that suits a coarse level can leave a fine
level with chunks holding tens of millions of links.

Rechunking is relatively rare in practice: most users choose `chunk_shape`
once before writing. The common scenarios are discovering after a large write
that the chunk size is poorly matched to the access pattern, or needing
physically contiguous groups for fast categorical reads (the dimension
rechunk).

---

## Technical reference

### Spatial rechunk: `rechunk_spatial`

```python
from zarr_vectors.rechunk import rechunk_spatial, suggest_chunk_shape

rechunk_spatial(
    "scan.zarrvectors",
    "scan_rechunked.zarrvectors",          # output: must not exist (or pass force=True)
    chunk_shape=(500.0, 500.0, 500.0),     # new root chunk shape; None = suggest_chunk_shape()
)
```

Full signature: `rechunk_spatial(store_path, output, chunk_shape=None, *,
level_chunk_shapes=None, target_links_per_chunk=8_000_000, levels=None,
force=False)`.

- `chunk_shape` — the new root chunk shape. If `None`, one is chosen by
  `suggest_chunk_shape`, which halves the chunk per axis until the busiest
  cell's projected link count is under `target_links_per_chunk` (default 8M,
  well under the `2**32`-byte cell ceiling at which the vlen-bytes codec would
  silently truncate).
- `level_chunk_shapes={level: shape}` — give coarse levels their own, coarser
  grid; each shape must be an integer multiple of the root's. Without it the
  whole pyramid is written on one grid.
- `levels` — the levels to carry across (default all).

Bin shape: the output keeps the source's `base_bin_shape`, or `chunk_shape / 4`
per axis if the source had none.

#### What it does

1. **Vertices** of each object are re-assigned to chunks by
   `floor(position / chunk_shape)` and written one fragment per object per chunk.
2. **Links** of any `link_width` are re-split onto the new grid: the intra-chunk
   majority is written directly, the cross-chunk remainder through `write_links`,
   so endpoint canonicalisation and the `perm_idx` that restores winding are
   handled by the library. Records move between offsets arrays of the same
   family.
3. **Object index** is rebuilt, with **object ids preserved exactly**.
4. **Object attributes**, **groups** and **group attributes** are carried across
   per level.
5. Every level is rechunked, not just level 0.

#### What is preserved vs recomputed

| Array | After `rechunk_spatial` | Notes |
|-------|-------------------------|-------|
| `vertices/` values | Preserved | Same positions, different chunk assignment |
| `vertex_fragments/` | Recomputed | One fragment per object per chunk (no bin fragments) |
| `links/<delta>/<offsets>/`, `link_fragments/` | Recomputed | Offsets and chunk-local vertex indices change |
| `object_index/` | Recomputed | Chunk coordinates and fragment indices change; ids preserved |
| `object_attributes/` | Preserved | Per-object, not per-chunk |
| `groups`, `group_attributes/` | Preserved | Per-group |
| `vertex_attributes/` | **Not carried** | Not read or written by this function |
| `fragment_attributes/` | **Not carried** | Derived data; regenerate (`write_fragment_owner_column`) |
| `link_attributes/` | **Not carried** | |
| Root `zarr_vectors` block | New | `chunk_shape` updated; `bounds`, `geometry_types` and conventions copied |
| Per-level metadata | Rebuilt | `parent_level`, `coarsening_method`, a per-level `chunk_shape` override; levels `> 0` are stamped `preserves_object_ids` |

If you need per-vertex or per-link attributes on the output, write them again
afterwards (see [Attributes](../object_model/object_attributes.md#adding-attributes-to-an-existing-store)).

### Dimension rechunk: `rechunk` and `rechunk_by_attribute`

```python
from zarr_vectors.rechunk import RechunkSpec, rechunk, rechunk_by_attribute

rechunk("tracts.zarrvectors", RechunkSpec(by="group"), output="tracts_by_group.zarrvectors")
rechunk_by_attribute("cells.zarrvectors", "gene")           # in place; one chunk-key prefix per value
```

`RechunkSpec.by` is `"group"`, `"object_id"`, `"attribute:<name>"` (a
**per-object** attribute) or `"spatial"`; `bins`, `categorical`,
`spatial_chunk_shape` and `prefix_dim_name` refine the mapping. With
`output=None` the call is in place (see the Terms).

This engine is deliberately narrow, and its docstring in
`zarr_vectors.rechunk.spatial` says why a separate spatial rechunk exists. It
rewrites **level 0 only**, copies only `vertices`, `object_index` and (for
`by="group"`) the group memberships, never reads or writes the `links` family,
and **renumbers objects** with a running counter — so ids no longer join back to
the source and no mapping is stored. For a point cloud with no links that is
lossless; for any type whose topology lives in links, use it only on stores you
can afford to lose links from. Each destination chunk holds one fragment per
object piece.

### Rechunking and resolution levels

`rechunk_spatial` rechunks all levels in one call (or those in `levels=`), so
there is no per-level loop to write. To rechunk only the base level and rebuild
coarser levels, pass `levels=[0]` and build a pyramid on the output:

```python
from zarr_vectors.rechunk import rechunk_spatial
from zarr_vectors.multiresolution.coarsen import build_pyramid

rechunk_spatial("scan.zarrvectors", "scan_rechunked.zarrvectors",
                chunk_shape=(500.0, 500.0, 500.0), levels=[0])
build_pyramid("scan_rechunked.zarrvectors", factors=[(2.0, 1.0), (2.0, 2.0)])
```

### Memory usage

Both engines hold a whole level's per-object geometry in memory while they
re-bucket it: `rechunk_spatial` reads each level into `{object_id:
(vertices, links)}` before writing. Peak memory scales with the size of a
level, not with the ratio of old to new chunk volume, and there is no
streaming option. For very large stores, rechunk level by level
(`levels=[n]`, one output store per call) or re-ingest from source.

### CLI usage

A `zarr-vectors rechunk` CLI subcommand, if present, lives in a companion
package and is not part of this repository.

### Rechunking only `bin_shape`

If only `bin_shape` changes (i.e. `chunk_shape` is the same), the work
is cheaper because vertex positions do not change chunk assignment —
only the fragment partition within each chunk does. `rebin_level`
handles this case:

```python
from zarr_vectors.rechunk import rebin_level

rebin_level(
    "scan.zarrvectors",
    target_bin_shape=(25.0, 25.0, 25.0),   # finer bins within the same chunks
    level=0,
)
# -> {"chunks_processed", "vertices_processed", "bins_created",
#     "old_bin_shape", "new_bin_shape", "level"}
```

**`rebin_level` is point-cloud-only.** It raises `StoreError` unless
`geometry_types == ["point_cloud"]` exactly. Re-binning re-sorts
vertices *within* a chunk, which is safe only when the fragment layout
is derivable from positions alone; every other type carries per-object
fragment semantics that the re-sort would flatten irreversibly. For those,
use `rechunk_spatial`, which rebuilds the object manifests too.

The links question therefore does not arise: a store `rebin_level`
accepts has no links to rewrite.

`rebin_level` is in-place. It re-sorts the vertices of each chunk into the
new bins and rewrites `vertex_fragments/` with **one fragment per *occupied*
bin**, sorted by bin coordinate — unlike `write_points`, which emits a
fragment for every bin. After a rebin, fragment `f` is therefore *not* the bin
with row-major index `f`. It updates the root `base_bin_shape` (when
`level == 0`) and rewrites the level's metadata block. Chunk file paths are
unaffected.

> **Per-vertex attribute arrays are NOT rewritten.** `rebin_level`
> leaves `vertex_attributes/` untouched while shuffling vertex order within
> each chunk, so any per-vertex attribute array is left **misaligned** with its
> vertices and must be regenerated by the caller. This is a documented property
> of the function, not a bug — but it means `rebin_level` is unsafe to run on a
> store with per-vertex attributes unless you regenerate them afterwards. It
> also re-encodes vertices as `float32`.

`target_bin_shape` must evenly divide `chunk_shape` or the call raises
`MetadataError`.

### Validation after rechunking

Run the validator after rechunking to confirm the new store is consistent:

```python
from zarr_vectors.validate import validate

result = validate("scan_rechunked.zarrvectors", level=3)
print(result)
```

`validate(path, level=N)` accepts levels 1–5; the shipped checks are
described in [Validation](../validation/overview.md). Pay particular attention
to the L3 manifest and link checks (every object's blocks name an existing chunk
and fragment, and every link's endpoint chunks exist), as these are the most
likely to expose a rechunking bug.
