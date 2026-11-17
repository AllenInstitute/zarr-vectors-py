# L2: Metadata validation

## Terms

**Divisibility constraint**
: Each axis's chunk size must be an integer multiple of its bin size, so
  bins tile chunks. L2 requires
  `abs(chunk_shape[d] / bin_shape[d] - round(...)) <= 1e-9`.

## Introduction

L2 reads the root and level attributes and checks their values. It reads
no array data, no array `zarr.json` and no link family (links are checked
at [L3](l3_consistency.md)).

## Technical reference

L2 is `validate_metadata` in `zarr_vectors/validate/metadata.py`. Each
check adds a message to the `ValidationResult` (see
[Validation overview](overview.md#validationresult-api)). All findings
below are errors unless marked.

### Root

| Check | Message |
|-------|---------|
| The store opens; L2 stops if not | `Cannot open store: ...` |
| The root metadata parses (required keys and types); L2 stops if not | `Cannot read root metadata: ...` |
| `RootMetadata.validate()` passes: `zv_version` at least 0.9.0, valid conventions and cross-level settings | `Root metadata invalid: ...` |
| `sid_ndim >= 1` | `SID dimensionality is <n>, must be >= 1` |
| `len(chunk_shape) == sid_ndim` | `chunk_shape has <n> dims, expected <d>` |
| Every `chunk_shape[i] > 0` | `chunk_shape[i] = <v>, must be > 0` |
| `bounds` min and max have `sid_ndim` entries | `Bounds dim mismatch: ...` |
| `links_convention` is empty or one of `explicit`, `implicit_sequential`, `implicit_sequential_with_branches` | `Unknown links_convention: '<v>'` |
| `object_index_convention` is empty or `standard` or `identity` | `Unknown object_index_convention: '<v>'` |
| `cross_chunk_strategy` is empty or one of `explicit_links`, `boundary_deduplication`, `both` | `Unknown cross_chunk_strategy: '<v>'` |
| `geometry_types` is not empty (**warning**) | `No geometry_types specified` |
| `base_bin_shape`, if set, has `sid_ndim` entries | `base_bin_shape has <n> dims, expected <d>` |
| Every `base_bin_shape[i] > 0` | `base_bin_shape[i]=<v>, must be > 0` |
| `chunk_shape[i]` is an integer multiple of `base_bin_shape[i]` | `chunk_shape[i]=<c> not integer multiple of base_bin_shape[i]=<b>` |

### Each level

Level fields are read from the level's `zarr_vectors_level` block, with
legacy flat keys as a fallback. `bin_shape` and `bin_ratio` come from the
level's NGFF transform when it has one: at level 0 they are not checked
(level 0 uses the root `base_bin_shape`), and at coarser levels the
transform on stores this package writes is level 0's, so the checks see
`bin_ratio = 1` and the base bin shape, not the level's own bins.

| Check | Message |
|-------|---------|
| The level's metadata can be read | `resolution_N: cannot read metadata: ...` |
| Presence is not deferred: the arrays carry `nonempty_chunks` (**warning**) | `resolution_N: presence is deferred ...` |
| `vertex_count`, if present, is a non-negative int | `resolution_N: vertex_count=<v> invalid` |
| A per-level `chunk_shape` is a positive integer multiple of the root's on every axis | `resolution_N: <MetadataError>` |
| `bin_shape` has `sid_ndim` entries | `resolution_N: bin_shape has <n> dims` |
| Every `bin_shape[i] > 0` | `resolution_N: bin_shape[i]=<v> not > 0` |
| `bin_shape[i]` divides the level's effective chunk shape (its override, else the root's). Skipped on an axis where the bin is larger than the chunk, and on a coarse level without a `chunk_shape` override | `resolution_N: bin_shape[i]=<b> does not divide chunk_shape[i]=<c>` |
| `bin_ratio` has `sid_ndim` entries, each `>= 1` | `resolution_N: bin_ratio ...` |
| `object_sparsity` (default `1.0`) is a number in `(0, 1]` | `resolution_N: object_sparsity=<v> not in (0, 1]` |

### Octree capabilities

Once per store, for each of the root's `format_capabilities` and
`required_capabilities`. Both are warnings and suggest
`zarr_vectors.octree_chunks.refresh_capabilities(store)`.

| Check | Message |
|-------|---------|
| A level with octree chunks means the list has `octree_chunks` | `level(s) [...] have octree chunks but the root's <list> lacks 'octree_chunks' ...` |
| `octree_chunks` in the list means some level has octree chunks | `the root's <list> lists 'octree_chunks' but no level has octree chunks; ...` |

### Not checked at L2

- Link families and link attributes.
- Array metadata: `vertices/` dtype, `vertex_fragments/` encoding.
- The `multiscales` block and its coordinate transforms.
- Rules per geometry type (L4).

### Example

A point-cloud store with empty `geometry_types` and a `base_bin_shape`
that does not divide `chunk_shape` (the pass count includes L1).
`RootMetadata.validate()` and the explicit divisibility check both report
the bad `base_bin_shape`:

```
Level 2 validation: FAIL
  19 passed, 1 warnings, 2 errors
  ERROR: Root metadata invalid: chunk_shape[0]=200.0 is not an integer multiple of base_bin_shape[0]=30.0 (ratio=6.666667)
  ERROR: chunk_shape[0]=200.0 not integer multiple of base_bin_shape[0]=30.0
  WARN:  No geometry_types specified
```
