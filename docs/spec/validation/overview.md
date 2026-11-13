# Validation overview

## Terms

**Conformance level**
: An integer from 1 to 5 setting how deep a validation run goes. A run at
  level `N` runs the checks of every level up to `N`.

**Check**
: One assertion made during validation. It adds a message to the passed,
  warnings or errors list. Checks have no ids.

**Warning**
: A finding that does not make the store invalid but may matter to some
  readers, such as a missing RFC 8 `ome` node.

**Error**
: A finding that makes the store non-conforming at the level run, such as
  a `base_bin_shape` that does not divide `chunk_shape`.

## Introduction

`zarr_vectors.validate.validate()` checks a store at one of five levels.
Run it after writing, converting or rechunking a store. Writers check
their arguments but not the data they write, so run at least level 3
after writing a new kind of store.

## Technical reference

### Conformance levels

| Level | Name | Checks | Reads | Page |
|-------|------|--------|-------|------|
| 1 | Structural | Required nodes exist; per-chunk arrays share one grid rank | metadata | [L1](l1_structural.md) |
| 2 | Metadata | Root and level attribute values | metadata | [L2](l2_metadata.md) |
| 3 | Consistency | Chunks decode; manifests and links name chunks and fragments that exist | every chunk | [L3](l3_consistency.md) |
| 4 | Geometry conventions | `links_convention` per geometry type; mesh `link_width`; links in point clouds | metadata | [below](#level-4-geometry-conventions) |
| 5 | Pyramid | Level numbering; vertex counts per level | every chunk | [below](#level-5-pyramid) |

There is no sampling option; to bound the cost, run at level 2.

### Level 4: geometry conventions

`validate_conformance` in `zarr_vectors/validate/conformance.py`. It
checks metadata, not topology or geometry.

| Check | Severity | Message |
|-------|----------|---------|
| Each type in `geometry_types` is known | warning | `Unknown geometry type: '<t>'` |
| `links_convention` is allowed for the type (table below); skipped when level 0 has a `geometry_index` | error | `'<t>' requires links in {...}, got '<c>'` |
| Level 0 exists; if not, the remaining checks are skipped | warning | `No level 0 for geometry checks` |
| Mesh: `links/0/` has `link_width >= 3` | error | `Mesh link_width=<w>, must be >= 3` |
| Mesh: link metadata exists | warning | `Mesh geometry but no links metadata` |
| Point cloud without graph, skeleton or mesh: no `links/0/`. Fires on every such store, with or without links | warning | `Point cloud but links array exists` |

| Geometry type | Allowed `links_convention` |
|---------------|----------------------------|
| `point_cloud` | any |
| `line`, `polyline`, `streamline` | `implicit_sequential` |
| `graph`, `mesh` | `explicit` |
| `skeleton` | `implicit_sequential_with_branches` or `explicit` |

### Level 5: pyramid

`validate_multiresolution` in `zarr_vectors/validate/conformance.py`. A
store with one level passes. All findings are errors.

| Check | Message |
|-------|---------|
| Levels are numbered `0, 1, …, n-1` | `Levels [0, 2], expected [0, 1]` |
| No level has more vertices than the level before (counted by decoding chunks) | `resolution_N: <n> > resolution_M (<m>)` |
| `bin_ratio` volume does not decrease. Does not run on stores this package writes | `resolution_N: bin_ratio volume ...` |
| `object_sparsity` is in `(0, 1]`. Does not run on stores this package writes | `resolution_N: object_sparsity=... out of range` |
| Each level can be read | `resolution_N: <exception>` |

L5 does not compare object counts, bin shapes or vertex correspondence
between levels.

### Python API

```python
from zarr_vectors.validate import validate

result = validate("contacts.zarrvectors", level=5)
print(result.summary())
# Level 5 validation: PASS
#   25 passed, 0 warnings, 0 errors
```

`validate(store, *, level=3)` takes a path, URL or open `Group`. A
`level` outside 1–5 raises `ValueError`. `Dataset.validate(level=...)`
does the same on an open dataset. The `zvtools validate` command is in
[zarr-vectors-tools](https://zarr-vectors-tools.readthedocs.io/en/latest/quickstart.html).

Levels run in order. An L1 error ends the run after L1; an L2 error ends
it after L2 when `level >= 3`. L3, L4 and L5 always run in full. So a
failing store may have more errors than reported; re-run after each fix.

### `ValidationResult` API

| Member | Type | Meaning |
|--------|------|---------|
| `level` | `int` | The level `validate()` was called with |
| `ok` | `bool` | `True` when there are no errors |
| `passed` | `list[str]` | Messages of checks that passed |
| `warnings` | `list[str]` | Warning messages |
| `errors` | `list[str]` | Error messages |
| `summary()` | `str` | Status line, counts, then every error and warning |
| `merge(other)` | `None` | Appends another result's three lists |

Messages are plain strings and are not a stable interface.
