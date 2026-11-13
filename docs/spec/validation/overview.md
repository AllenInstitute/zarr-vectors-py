# Validation overview

## Terms

**Conformance level**
: An integer from 1 to 5 choosing how deep a validation run goes. A run at
  level `N` runs the checks of every level up to `N`.

**`ValidationResult`**
: The object `validate()` returns: the run's `level`, three lists of message
  strings (`passed`, `warnings`, `errors`), an `ok` property and a
  `summary()` method.

**Check**
: One assertion made during validation. It adds a message to one of the
  three lists. Checks have no ids; validation does not stop at the first
  failed check.

**Warning**
: A finding that does not make the store invalid but may matter to some
  readers. Example: no RFC 8 `ome` node in the root attributes.

**Error**
: A finding that makes the store non-conforming at the level run. Example: a
  `base_bin_shape` that does not divide `chunk_shape`.

## Introduction

`zarr-vectors-py` includes a validator with five levels. Levels 1 and 2
read metadata and are fast. Level 3 reads every chunk. Levels 4 and 5 read
metadata again, except that level 5 decodes chunks to count each level's
vertices.

Uses:

- After writing or converting a store, before sharing it.
- After rechunking, to confirm the result is consistent.
- In CI, at level 2 on large stores and level 3 or above on small ones.
- When debugging a writer, to find which check fails.

## Technical reference

### Conformance levels

| Level | Name | Checks | Page |
|-------|------|--------|------|
| 1 | Structural | Required nodes exist; per-chunk arrays share one grid rank | [L1](l1_structural.md) |
| 2 | Metadata | Root and level attribute values: vocabulary tokens, dimensionality, bin and chunk divisibility | [L2](l2_metadata.md) |
| 3 | Consistency | Chunks decode; manifests and link records name chunks and fragments that exist; canonical offsets | [L3](l3_consistency.md) |
| 4 | Geometry conventions | Each geometry type's `links_convention`; mesh `link_width`; links in point clouds | [below](#level-4-geometry-conventions) |
| 5 | Pyramid | Levels numbered from 0 without gaps; vertex count does not grow from one level to the next | [below](#level-5-pyramid) |

### Level 4: geometry conventions

L4 is `validate_conformance` in `zarr_vectors/validate/conformance.py`. It
checks metadata conventions, not topology or geometry.

| Rule | Failure |
|------|---------|
| Each type in `geometry_types` is known | Warning (`Unknown geometry type`) |
| `links_convention` is one the type allows (table below) | Error, unless level 0 has a `geometry_index` (a composite store) |
| Level 0 exists | Warning, and the remaining checks are skipped |
| Mesh: `links/0/` has `link_width >= 3` | Error when it is 1 or 2 |
| Mesh: link metadata exists | Warning (`Mesh geometry but no links metadata`) |
| Point cloud without graph, skeleton or mesh: no `links/0/` | Warning (`Point cloud but links array exists`) |

| Geometry type | Allowed `links_convention` |
|---------------|----------------------------|
| `point_cloud` | any |
| `line`, `polyline`, `streamline` | `implicit_sequential` |
| `graph`, `mesh` | `explicit` |
| `skeleton` | `implicit_sequential_with_branches` or `explicit` |

The point-cloud check looks for `links/0/` by reading its attributes, and
that read returns an empty result rather than failing when the node is
absent, so every point-cloud store gets the warning, with or without links.

### Level 5: pyramid

L5 is `validate_multiresolution` in `zarr_vectors/validate/conformance.py`.
A store with one level passes.

| Rule | Failure |
|------|---------|
| Level numbers are `0, 1, …, n-1` | Error (`Levels [0, 2], expected [0, 1]`) |
| No level has more vertices than the level before it | Error |
| Each level can be read | Error (`resolution_N: …`) |

L5 counts each level's vertices by decoding its chunks. It also looks for
`vertex_count`, `bin_ratio` and `object_sparsity` among the level's
attributes, but reads them from the top level of the attributes, where a
store keeps only the `zarr_vectors_level` block. So the `bin_ratio` and
`object_sparsity` range checks do not run on stores written by this package.

L5 does not check object counts, bin shapes or vertex correspondence between
levels.

### Python API

```python
from zarr_vectors.validate import validate

result = validate("contacts.zarrvectors", level=5)
print(result.summary())
# Level 5 validation: PASS
#   25 passed, 0 warnings, 0 errors

if not result.ok:
    for msg in result.errors:
        print("ERROR:", msg)

result = validate("contacts.zarrvectors", level=2)   # metadata only
```

`validate(store, *, level=3)` takes a path, URL or open `Group`. A `level`
outside 1–5 raises `ValueError`. An open dataset has the same check as
`Dataset.validate(level=...)`.

Levels run in order, and a failure stops the run early:

- If L1 has an error, `validate()` returns after L1.
- If L2 has an error and `level >= 3`, it returns after L2.
- L3, L4 and L5 all run.

So `result.errors` on a failing store may not list everything wrong with it;
re-run after each fix.

### `ValidationResult` API

```python
result.level              # int: the level validate() was called with
result.ok                 # bool: True when there are no errors
result.passed             # list[str]: messages for checks that passed
result.warnings           # list[str]: warning messages
result.errors             # list[str]: error messages
result.summary()          # str: status line, counts, then every error and warning
result.merge(other)       # None: absorb another result's three lists
```

The lists hold plain strings from every level run. Code that needs to tell
findings apart has to match message text, which is not a stable interface.

### Command line

The `zvtools validate` command is in the companion package
[zarr-vectors-tools](https://zarr-vectors-tools.readthedocs.io/en/latest/quickstart.html).

### Validation and writers

Writers check their arguments before writing, but not the data they write.
Run at least level 3 after writing a new kind of store.

### Cost

Levels 1, 2 and 4 read metadata only. Level 3 reads every chunk of every
level, and level 5 decodes every chunk to count vertices. `validate()` has no
sampling option; to bound the cost, run at level 2. The only sampling is
internal: the L3 point-cloud bin-bounds check examines the first three chunks
of each level.
