# Validation overview

## Terms

**Conformance level**
: An integer from 1 to 5 declaring the thoroughness of a validation run.
  Each level is a strict superset of the previous: a store that passes
  level `N` also passes all levels below `N`. Running level 5 runs all
  checks.

**`ValidationResult`**
: The object returned by `validate()`. Carries the run's `level` plus
  three lists of message strings — `passed`, `warnings`, `errors` — an
  `ok` property, and a `summary()` method.

**Check**
: A single assertion evaluated during validation. A check either passes,
  emits a warning (non-fatal), or records an error (fatal). Errors
  accumulate within a level; validation does not stop at the first
  failure. A check is **not** a reified object — it appends a
  human-readable string to one of the three lists. There are no stable
  check IDs to match on.

**Warning**
: A non-fatal issue that indicates the store may behave unexpectedly in
  some tools but is not technically invalid. Example: a store without
  consolidated metadata will be slow to open on object stores.

**Error**
: A fatal issue that indicates the store does not conform to the ZVF spec
  at the declared conformance level. Example: a `bin_shape` value that
  does not evenly divide `chunk_shape`.

---

## Introduction

`zarr-vectors-py` ships a multi-level validator that checks ZVF stores for
correctness and conformance. Validation is organised into five progressively
deeper levels. Shallow levels (1–2) are fast and check structural and
metadata properties. Deeper levels (3–5) are more expensive because they
require reading array data, but they catch logical inconsistencies that
metadata checks alone cannot detect.

The validator is designed to be useful in several contexts:

- **After writing:** confirm that a newly written store is valid before
  sharing it.
- **After ingest:** confirm that a third-party format was correctly
  translated.
- **After rechunking:** confirm that the rechunked store is consistent.
- **CI pipelines:** run level 1–2 checks quickly; reserve level 5 for
  nightly runs.
- **Debugging:** identify the specific check that fails to pinpoint bugs
  in writer or converter code.

---

## Technical reference

### Conformance levels

| Level | Name | What it checks | Typical runtime |
|-------|------|----------------|-----------------|
| 1 | Structural | Required paths exist ([`structure.py`](../../../zarr_vectors/validate/structure.py)). Filesystem presence only — **no** Zarr node-type inspection | < 1 s |
| 2 | Metadata | Group-attribute validity ([`metadata.py`](../../../zarr_vectors/validate/metadata.py)): `sid_ndim` agreement, vocabulary tokens, bin/chunk divisibility. **No** array `zarr.json`, **no** link families | 1–5 s |
| 3 | Consistency | Chunk decode, manifest integrity, link offsets-segment and record validity ([`consistency.py`](../../../zarr_vectors/validate/consistency.py)) | 10 s – 10 min (reads all chunks) |
| 4 | Geometry | `links_convention` valid for each declared geometry type; mesh `link_width >= 3`; point clouds carry no links ([`conformance.py`](../../../zarr_vectors/validate/conformance.py)) | < 1 s |
| 5 | Pyramid | Levels contiguous from 0; `vertex_count` non-increasing across levels; `bin_ratio` volume non-decreasing; `object_sparsity` in `(0, 1]` | adds per-level cost |

The level names are historical. **L4 does not verify topology or
geometry**: it checks metadata conventions, not tree structure,
watertightness, or polyline gaps. **L5 does not check object counts**,
`bin_shape` consistency, or cross-level vertex correspondence.

L5 falls back to decoding every chunk to count vertices when a level
omits its `vertex_count` attribute, so its cost is only per-level-
metadata-sized on stores that record that attribute.

### Python API

```python
from zarr_vectors.validate import validate

# Run full validation (level 5)
result = validate("scan.zarrvectors", level=5)

# Print summary
print(result.summary())
# Level 5 validation: PASS
#   54 passed, 2 warnings, 0 errors

# Check programmatically
if not result.ok:
    for msg in result.errors:
        print(f"ERROR: {msg}")

# Run only fast checks (CI use)
result = validate("scan.zarrvectors", level=2)
```

`validate()` takes the store path and `level=` (default `3`) — nothing
else. A `level` outside `1–5` raises `ValueError`.

**Short-circuiting.** Levels run cumulatively but abort early on a
fatal result, so a failing store reports the *first* level that failed
rather than every level's findings:

- If L1 fails, `validate()` returns immediately — no L2+ checks run.
- If L2 fails and `level >= 3`, it returns after L2 — no L3+ checks run.
- L3, L4, and L5 do not short-circuit each other.

Consequently `result.errors` on a failing store is not an exhaustive
list of everything wrong with it. Re-run after each fix.

### `ValidationResult` API

```python
result.level              # int: the level this run was invoked at
result.ok                 # bool: True if there are no errors
result.passed             # list[str]: messages for checks that passed
result.warnings           # list[str]: non-fatal warning messages
result.errors             # list[str]: fatal error messages
result.summary()          # str: multi-line report — status line, counts,
                          #      then every error and warning
result.merge(other)       # None: absorb another result's three lists
```

The three lists hold **plain strings**, not objects. There is no
`is_valid`, `report()`, `as_dict()`, or `all_checks`; there is no
`Check` class, and messages carry no structured level, ID, status, or
store path. Programmatic consumers that need to distinguish specific
findings must match on message text, which is **not** a stable
interface.

`result.level` records the level `validate()` was *called* with, not
the level that produced any given message — messages from all executed
levels are merged into one flat set of lists.

### CLI usage

The `zarr-vectors validate` CLI lives in the companion package
**`zarr-vectors-tools`**.

### Validation and writers

The write functions in `zarr-vectors-py` perform inline validation of
arguments before writing. However, this does not substitute for post-write
validation: inline checks guard against obviously invalid parameters but do
not verify the correctness of the written data (e.g. fragment offset arithmetic,
cross-chunk link completeness). Always run at least level 3 after writing
a new store.

### Validation and performance

Levels 1 and 2 touch only metadata. Level 3 reads every chunk at every
level; for a large store (> 100 GB) it may take several minutes.

There is **no sampling or chunk-subsetting option**: `validate()`
accepts only `level=`. To bound cost, either run at `level=2` or point
the validator at a smaller store. The one place L3 samples is internal
and not configurable — its point-cloud bin-bounds check examines at
most the first 3 chunks per level (see [L3](l3_consistency.md)).
