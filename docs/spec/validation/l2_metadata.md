# L2: Metadata validation

## Terms

**Schema check**
: A validation check that reads a group's attributes and verifies that
  its values conform to the ZVF specification — valid ranges, valid
  vocabulary tokens, and dimensional agreement with `sid_ndim`.

**Divisibility constraint**
: The requirement that each axis's `chunk_shape[d]` be an integer
  multiple of `bin_shape[d]`, so bins tile chunks cleanly. Checked at
  L2 with a floating-point tolerance.

**Floating-point tolerance**
: The permissible difference when checking a float for integrality. L2
  computes `ratio = chunk_shape[d] / bin_shape[d]` and requires
  `abs(ratio - round(ratio)) <= 1e-9`. Note the tolerance applies to
  the **ratio**, not to the shape values.

---

## Introduction

L2 validation reads the store's root and per-level group attributes and
checks that their contents are internally consistent and conform to the
ZVF specification. No array data is read.

L2 catches the most common class of write-time bugs: invalid ranges,
unrecognised vocabulary tokens, and dimensionality mismatches between
`sid_ndim`, `chunk_shape`, and `bin_shape`.

L2's scope is narrower than the name suggests: it inspects **group
attributes only**, never an array's `zarr.json`, and never the link
families. See *What L2 does not check* below before relying on it.

---

## Technical reference

L2 is implemented by
[`validate_metadata`](../../../zarr_vectors/validate/metadata.py).
Checks are reported as free-text messages on a `ValidationResult`; they
do not carry stable machine-readable check IDs (see
[Validation overview](overview.md#validationresult-api)).

L2 reads the root metadata **through `RootMetadata`**, not as raw JSON.
Structural well-formedness of the document (required keys, parseable
types) is therefore enforced by the parser: a document that cannot be
parsed produces a single `Cannot read root metadata` error and L2
returns immediately. The checks below are those applied *on top of* a
successful parse.

### Root metadata checks

| Rule | Failure type |
|------|--------------|
| `sid_ndim >= 1` | Error |
| `len(chunk_shape) == sid_ndim` | Error |
| Every `chunk_shape[i] > 0` | Error |
| `bounds`, if present, has `len(min) == len(max) == sid_ndim` | Error |
| `links_convention` ∈ `{explicit, implicit_sequential, implicit_sequential_with_branches}` | Error |
| `object_index_convention` ∈ `{standard, identity}` | Error |
| `cross_chunk_strategy` ∈ `{explicit_links, boundary_deduplication, both}` | Error |
| `geometry_types` is non-empty | Warning |

A convention field that is empty / falsy **passes** — the check is
`if value and value not in VALID: error`. Only a non-empty unrecognised
token is an error.

> `cross_chunk_strategy` survives the merge of `cross_chunk_links/` into
> `links/`. It is **semantic, not physical**: it says how a writer
> handled geometry at chunk boundaries (duplicate the boundary record,
> or link across), a question the layout merge does not answer. It does
> not name a directory.

### Bin-shape checks (root)

Applied only when `base_bin_shape` is not `None`:

| Rule | Failure type |
|------|--------------|
| `len(base_bin_shape) == sid_ndim` | Error |
| Every `base_bin_shape[i] > 0` | Error |
| `chunk_shape[i] / base_bin_shape[i]` is an integer (tolerance `1e-9` on the ratio) | Error |

### Per-level checks

For every resolution level:

| Rule | Failure type |
|------|--------------|
| The level's attributes can be read | Error |
| `vertex_count`, if present, is an `int >= 0` | Error |
| `bin_shape` (or legacy `bin_size`), if present, has length `sid_ndim` | Error |
| Every `bin_shape[i] > 0` | Error |
| `chunk_shape[i] / bin_shape[i]` is an integer (tolerance `1e-9`) | Error |
| `bin_ratio`, if present, has length `sid_ndim` | Error |
| Every `bin_ratio[i] >= 1` | Error |
| `object_sparsity` (default `1.0`) is in `(0, 1]` | Error |

### What L2 does *not* check

These are called out because earlier revisions of this page specified
them and no shipped code enforces them. They are not ZVF requirements
at L2:

- **Nothing about link families.** L2 does not open
  `links/<delta>/`, does not check `link_width`, `sid_ndim`,
  `directed`, `store`, or `num_physical_records`, and does not verify
  that a `link_attributes/<name>/<delta>/<offsets>/` array is
  row-aligned with the offsets array it mirrors. Link families are
  reached first at [L3](l3_consistency.md).
- **No array `zarr.json` inspection at all.** L2 reads group attributes
  only. `vertices/` dtype and trailing dimension, `vertex_fragments/`
  encoding tokens, and blob magic are not checked at L2.
- **No `multiscales` / `coordinateTransformations` checks.** Scale and
  translation values, axis counts, and level-to-group correspondence
  are not validated.
- **No per-level `chunk_shape` override validation.** A level may carry
  a `chunk_shape` override (see
  [Pyramid construction](../multiscale/pyramid_construction.md#per-level-chunk-grids-chunk_scale_factor)).
  `zarr_vectors.core.metadata` provides
  `validate_level_chunk_shape_against_root` to check that it nests
  inside root, but **the L2 validator never calls it**. Note also that
  L2's per-level `bin_shape` divisibility check above divides the
  **root** `chunk_shape`, not the level's effective one — so on a store
  built with `chunk_scale_factor > 1` that check is testing the wrong
  quantity.
- **No cross-level `level` / ordering checks**, no `level_key_matches_name`,
  no monotone `bin_ratio` across levels, and no geometry-type-specific
  metadata rules (those are L4).

### Example L2 report

```
Level 2 validation: FAIL
  8 passed, 1 warnings, 1 errors
  ERROR: resolution_1: bin_shape[0]=30.0 does not divide chunk_shape[0]=200.0
  WARN:  No geometry_types specified
```

Passing messages take the form `Root metadata parsed`,
`SID dimensionality: 3`, `chunk_shape dimensionality matches SID`,
`links_convention: 'implicit_sequential'`, `bins_per_chunk: (4, 4, 4)`,
and `resolution_0: vertex_count=100000`.
