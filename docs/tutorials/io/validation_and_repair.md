# Validation and repair

The `zarr-vectors` validator checks ZVF stores for conformance at five
levels of increasing thoroughness. This tutorial covers running validation
and interpreting results.

A note on scope: **there is no repair module.** `zarr-vectors` ships no
general-purpose repair API — no function that rebuilds an object index, a
fragment index, or a links family from damaged data. The validator tells
you what is wrong; fixing it almost always means rewriting the affected
level from source through the normal write API. The few genuine
in-place remedies that do exist are shown below, and each one is a
general-purpose writer being used deliberately, not a repair tool.

For the complete check catalogue by level, see
[Validation overview](../../spec/validation/overview.md),
[L1 structural](../../spec/validation/l1_structural.md),
[L2 metadata](../../spec/validation/l2_metadata.md), and
[L3 consistency](../../spec/validation/l3_consistency.md).

---

## Running validation

The `zarr-vectors` CLI (with `validate` and `info` subcommands) lives in
the companion package **`zarr-vectors-tools`**. The Python API shown
below is part of this core package.

### Python API

```python
from zarr_vectors.validate import validate

result = validate("scan.zarrvectors", level=5)

# One-line status plus every error and warning
print(result.summary())
# Level 5 validation: PASS
#   54 passed, 0 warnings, 0 errors

# Programmatic access
print(result.ok)                # bool — True when there are no errors
print(result.level)             # int — the level that was requested
print(len(result.errors))       # int
print(len(result.warnings))     # int
print(len(result.passed))       # int

# passed / warnings / errors are plain lists of strings.
for msg in result.errors:
    print(msg)
```

`ValidationResult` is a dataclass with exactly four fields — `level`,
`passed`, `warnings`, `errors` — plus the `ok` property and the
`summary()`, `add_pass()`, `add_warning()`, `add_error()` and `merge()`
methods. Each message is a **string**, not a structured record: there is
no per-error `check`, `path`, or `level` attribute to read, so filter by
substring if you need to triage programmatically.

Levels run cumulatively, and the run short-circuits: if L1 fails,
nothing further runs; if L2 fails and `level >= 3`, the deeper passes are
skipped. A failing `summary()` therefore shows the *first* thing that
broke, not everything that is wrong.

The individual passes are also importable directly, each returning its
own `ValidationResult`:

```python
from zarr_vectors.validate import (
    validate_structure,        # L1
    validate_metadata,         # L2
    validate_consistency,      # L3
    validate_conformance,      # L4
    validate_multiresolution,  # L5
)
```

---

## Choosing a validation level

| Situation | Recommended level |
|-----------|-----------------|
| Quick structural check (CI, file open) | 1 |
| After writing a new store | 3 |
| After ingest from external format | 3 |
| After rechunking | 3 |
| Before publishing / sharing a dataset | 5 |
| Nightly CI on reference fixtures | 5 |

Level 3 reads all array data and is the minimum recommended for any store
that will be shared or used in analysis. Level 5 additionally checks
multi-resolution pyramid correctness.

`validate()` takes `store_path` and `level` only. There is no sampling
option — every level is all-or-nothing over the store, so budget L3+ runs
on very large stores accordingly.

---

## Interpreting common errors

Message text below is quoted from the validator source. `resolution_{N}`
is the level prefix the deeper passes stamp on every message.

### L1 — structure

L1 checks that the store exists, has root metadata, has at least one
resolution level, and that each level has the directories it needs.

```
Store path does not exist: scan.zarrvectors
No root metadata found (expected .zattrs, zarr.json, or metadata.json)
No resolution level directories found
resolution_0/vertices/ missing
```

Those are errors. L1 also emits warnings, which do **not** fail
validation:

```
resolution_0/vertex_fragments/ missing
resolution_0/ has no metadata file
resolution_0/links/ exists but has no <delta> subdirs
```

The last one is why a links family with no `<delta>` segments is a
warning rather than an error: a connectivity type with no stored links —
a fully implicit-sequential polyline, say — is legal.

*Remedy:* none of these are repairable in place. A store missing its
vertices or its root metadata was written that way, or was truncated in
transit; rewrite the level from source.

---

### L2 — metadata

L2 validates the root metadata and each level's attributes.

```
SID dimensionality is 0, must be >= 1
chunk_shape has 2 dims, expected 3
chunk_shape[2] = 0, must be > 0
Unknown links_convention: 'implicit_seq'
Unknown object_index_convention: 'sparse'
Unknown cross_chunk_strategy: 'lazy'
resolution_0: bin_shape[2]=60.0 does not divide chunk_shape[2]=200.0
resolution_0: bin_ratio[0]=0 < 1
resolution_0: object_sparsity=1.5 not in (0, 1]
```

The `bin_shape` divisibility error means the declared `bin_shape` does not
evenly divide `chunk_shape` on that axis. It is a metadata-level check —
the validator compares the two declared tuples and does not look at the
data.

*Remedy:* these are all `.zattrs` / root-metadata faults, and the honest
fix depends on which is true:

- If the **metadata** is wrong and the data is fine (usually a hand-edit),
  correct the attribute. Any Zarr attribute writer will do; `open_store`
  in `r+` mode gives you the group.
- If the **data** was actually written under the bad geometry, the
  metadata is telling the truth and the store must be rewritten from
  source. There is no rebinning API in this package.

```python
from zarr_vectors.core.store import open_store

root = open_store("scan.zarrvectors", mode="r+")
# Inspect before changing anything — only edit attrs you are certain
# disagree with the data on disk.
print(root.attrs.to_dict())
```

`.attrs` is a dict-*like* wrapper, not a dict: it supports `attrs[k]`,
`attrs.get(k, default)`, `k in attrs`, `attrs.update(d)`, and
`attrs.to_dict()`. It is not iterable, so `dict(root.attrs)` and
`for k in root.attrs` do not work — use `to_dict()`.

---

### L3 — consistency

L3 decodes every chunk and cross-checks the data against the metadata.

**Vertex / fragment errors**

```
resolution_0: chunk (2,3,1) decode failed: <exception>
resolution_0: chunk (2,3,1) fragment[7] shape (4200,)
resolution_0: chunk (2,3,1) has 9 fragments, exceeds bins_per_chunk product 8
resolution_0: metadata vertex_count=4092, actual=4200
```

These indicate a writer bug or a corrupted chunk. *Remedy:* none in
place — re-ingest the level.

**Object index errors**

```
resolution_0: obj 1042 refs non-existent chunk (8,8,4)
resolution_0: obj 1042 refs fragment_idx=12 >= 8
```

The object index points at a chunk or fragment that is not there, usually
after something moved vertices without rewriting the index. *Remedy:* the
index can be rewritten with `write_object_index` if — and only if — you
can reconstruct the correct manifests yourself; the package will not
derive them for you.

**Links errors**

```
resolution_0: links[delta=0] offsets segment '0.0.+' malformed: <reason>
resolution_0: links[delta=0] num_physical_records=1500 != 1499 rows on disk
resolution_0: links[delta=0] refs non-existent source chunk (8,8,4)
resolution_0: links[delta=0] refs non-existent chunk (8,8,5)
```

A `num_physical_records` mismatch is the one links error with a real
remedy: it is exactly what `finalize_links` reconciles after
decentralized per-cell writes. If workers wrote cells with
`write_link_cells` and no coordinator ever finalized, run it now:

```python
from zarr_vectors.core.arrays import finalize_links
from zarr_vectors.core.store import open_store, get_resolution_level

root = open_store("tracts.zarrvectors", mode="r+")
level_group = get_resolution_level(root, 0)
partition = finalize_links(level_group, delta=0)
print(partition.num_links, partition.num_physical_records)
```

See [Cloud stores](cloud_stores.md) for the full decentralized
write-then-finalize sequence.

**Canonical-form errors**

The L3 validator also enforces invariants on the *directory names* of a
links family — the offsets segments themselves, before any data is read:

```
resolution_0: links[delta=0] segment '0.0.-1' offset 1 is lexicographically
negative; a canonical family stores each record once, under the positive offset
resolution_0: links[delta=0] segment '0.+1.0_0.0.+1' offsets are not
non-decreasing; violates the canonical-sort invariant
```

These two checks are **gated**: they run only when the family is
`directed=False`, `store="canonical"`, and `delta == 0`. Directed families
key on input endpoint order, `duplicate` families deliberately lead with
each incident chunk, and cross-level (`delta != 0`) records are never
sorted — all three legitimately carry lex-negative offsets, so enforcing
canonical form on them would be wrong.

An all-zero segment (`0.0.0`) is always legal — it is the intra-chunk
array, not a violation.

*Remedy:* none in place. A family in non-canonical form was written by
something that bypassed `write_links`; rewrite it through the real writer.

---

### L4 — conformance

L4 checks that each declared geometry type has the metadata it requires.

```
'mesh' requires links in ('explicit',), got 'implicit_sequential'
Mesh link_width=2, must be >= 3
```

and warnings:

```
Unknown geometry type: 'polygon'
Point cloud but links array exists
```

---

### L5 — multiresolution

L5 checks pyramid shape and monotonicity.

```
Levels [0, 1, 3], expected [0, 1, 2]
resolution_2: 5000 > resolution_1 (4000)
```

The second says a coarser level has *more* vertices than the level below
it, which means the coarsening did not actually coarsen.

*Remedy:* this one has a genuine rebuild path — the levels above a known-good
source level can be re-coarsened from scratch, reusing each target level's
own recorded `bin_ratio` / `object_sparsity` / `chunk_shape`:

```python
from zarr_vectors.ops.refresh import rebuild_pyramid_from_level
from zarr_vectors.core.store import open_store

root = open_store("scan.zarrvectors", mode="r+")
summaries = rebuild_pyramid_from_level(root, source_level=0)
```

This replaces the old level data in place. It trusts `source_level`
completely — validate that level at L3 first.

If instead the `multiscales` metadata itself is stale, regenerate it:

```python
from zarr_vectors.core.multiscale import write_multiscale_metadata
from zarr_vectors.core.store import open_store

root = open_store("scan.zarrvectors", mode="r+")
write_multiscale_metadata(root)
```

---

## Validation after any fix

Always re-run the validator at the same or higher level after changing
anything:

```python
from zarr_vectors.validate import validate

result = validate("tracts.zarrvectors", level=3)
assert result.ok, result.summary()
print("Store is valid.")
```

---

## Automated validation in CI

Use the Python API in a small driver script if you want a CI step, or
invoke `zarr-vectors validate` from **`zarr-vectors-tools`** if it is
already installed in the CI environment.

In a pytest fixture:

```python
import pytest
from zarr_vectors.validate import validate
from pathlib import Path

FIXTURES = list((Path("tests") / "fixtures").glob("*/store.zarrvectors"))

@pytest.mark.parametrize("store_path", FIXTURES, ids=lambda p: p.parent.name)
@pytest.mark.slow
def test_fixture_passes_l5(store_path):
    result = validate(str(store_path), level=5)
    assert result.ok, result.summary()
```
