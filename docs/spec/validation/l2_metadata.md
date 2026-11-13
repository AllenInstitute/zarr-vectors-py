# L2: Metadata validation

## Terms

**Schema check**
: A check that reads the root or a level's attributes and verifies their
  values: ranges, vocabulary tokens, and agreement with `sid_ndim`.

**Divisibility constraint**
: Each axis's chunk size must be an integer multiple of its bin size, so
  bins tile chunks. L2 computes `ratio = chunk_shape[d] / bin_shape[d]` and
  requires `abs(ratio - round(ratio)) <= 1e-9`; the tolerance applies to the
  ratio, not to the sizes.

## Introduction

L2 reads the root attributes and each level's attributes and checks that
their values are consistent. It reads no array data, no array `zarr.json`
and no link family; link families are first checked at
[L3](l3_consistency.md). L2 catches unrecognised vocabulary tokens and
dimensionality mismatches between `sid_ndim`, `chunk_shape` and the bin
shape.

## Technical reference

L2 is `validate_metadata` in `zarr_vectors/validate/metadata.py`. Each check
adds a message to the `ValidationResult` (see
[Validation overview](overview.md#validationresult-api)).

L2 parses the root metadata with `RootMetadata`, so required keys and
parseable types are enforced by the parser. A store that does not open gives
`Cannot open store`, and root metadata that does not parse gives
`Cannot read root metadata`; in either case L2 stops. The checks below apply
after a successful parse.

### Root metadata checks

| Rule | Failure |
|------|---------|
| `sid_ndim >= 1` | Error |
| `len(chunk_shape) == sid_ndim` | Error |
| Every `chunk_shape[i] > 0` | Error |
| `bounds` has `len(min) == len(max) == sid_ndim` | Error |
| `links_convention` ∈ `{explicit, implicit_sequential, implicit_sequential_with_branches}` | Error |
| `object_index_convention` ∈ `{standard, identity}` | Error |
| `cross_chunk_strategy` ∈ `{explicit_links, boundary_deduplication, both}` | Error |
| `geometry_types` is not empty | Warning |

An empty convention value passes; only a non-empty unrecognised token is an
error.

When `base_bin_shape` is set:

| Rule | Failure |
|------|---------|
| `len(base_bin_shape) == sid_ndim` | Error |
| Every `base_bin_shape[i] > 0` | Error |
| `chunk_shape[i]` is an integer multiple of `base_bin_shape[i]` | Error |

### Per-level checks

For every level:

| Rule | Failure |
|------|---------|
| The level's metadata can be read | Error (`resolution_N: cannot read metadata`) |
| A per-level `chunk_shape` is a positive integer multiple of the root `chunk_shape` on every axis | Error |
| The level's presence is not deferred (its arrays carry `nonempty_chunks`) | Warning |

L2 also looks for `vertex_count`, `bin_shape`, `bin_ratio` and
`object_sparsity` among the level's attributes, but reads them from the top
level of the attributes, where a store keeps only the `zarr_vectors_level`
block. These checks therefore do not run on stores written by this package:
a level with `vertex_count: -3` or `object_sparsity: 1.5` passes L2.

### Octree capability checks

Once per store, for each of `format_capabilities` and
`required_capabilities` in the root metadata:

| Rule | Failure |
|------|---------|
| A level with octree chunks implies the list contains `octree_chunks` | Warning |
| `octree_chunks` in the list implies some level has octree chunks | Warning |

Both warnings suggest `zarr_vectors.octree_chunks.refresh_capabilities(store)`.

### Not checked at L2

- Link families: `link_width`, `directed`, `store`, `num_physical_records`,
  and the row alignment of `link_attributes/` with `links/`.
- Array metadata: `vertices/` dtype, `vertex_fragments/` encoding.
- `multiscales` transforms, axis counts, and level-to-group correspondence.
- Rules per geometry type (those are L4).

### Example L2 report

A point-cloud store with an emptied `geometry_types` and a `base_bin_shape`
that does not divide `chunk_shape`:

```
Level 2 validation: FAIL
  19 passed, 1 warnings, 1 errors
  ERROR: chunk_shape[0]=200.0 not integer multiple of base_bin_shape[0]=30.0
  WARN:  No geometry_types specified
```

The pass count includes the L1 passes. L2 pass messages read like
`Root metadata parsed`, `SID dimensionality: 3`,
`links_convention: 'implicit_sequential'`, `bins_per_chunk: (7, 4, 4)` and
`resolution_0 metadata parsed`.
