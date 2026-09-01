# Validation and repair

The `zarr-vectors` validator checks Zarr Vectors stores for conformance at five
levels of increasing thoroughness. This tutorial covers running validation
and interpreting results.

A note on scope: **there is no general-purpose repair module.**
`zarr-vectors` ships no function that rebuilds an object index, a fragment
index, or a links family from damaged data. The validator tells you what is
wrong; fixing it almost always means rewriting the affected level from
source through the normal write API.

Three genuine in-place repair operations do exist, and all three are
supported names in `zarr_vectors.building`:

| Function | Repairs |
|----------|---------|
| `finalize_links` | a links family's `num_links` / `num_physical_records` counts |
| `refresh_arrays_present` | a level's `arrays_present` list |
| `rebuild_presence` | one array's (or a level's) `nonempty_chunks` manifest |

None of them is a fixer for corrupt data. Each is the *coordinator half of
a decentralised write* — bookkeeping that parallel workers deliberately
skip because it is a single shared field they would race on — being used
after the fact. They are shown below where the errors they answer are.

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

`zarr_vectors.validate` is **undecided** on the stability manifest —
neither promised nor disowned. It is stable in practice and widely used,
but its result objects have never been given a compatibility promise:

```python
import zarr_vectors as zv

print(zv.stability("zarr_vectors.validate"))
```

```text
undecided
```

### Python API

```python
from zarr_vectors.validate import validate

result = validate("scan.zarrvectors", level=5)

# One-line status plus every error and warning
print(result.summary())
# Level 5 validation: PASS
#   35 passed, 1 warnings, 0 errors

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

### `validate()` takes a path, not a URL

L1 walks the store as a filesystem tree, so `store_path` must be a
filesystem path. A `file://` URL is *not* accepted — it is treated as a
relative directory name and fails the very first check:

```pycon
>>> validate("scan.zarrvectors", level=1).ok
True
>>> print(validate("file:///data/scan.zarrvectors", level=1).summary())
Level 1 validation: FAIL
  0 passed, 0 warnings, 1 errors
  ERROR: Store path does not exist: file:/data/scan.zarrvectors
```

This is worth knowing because `Dataset.url` is always a URL — a locally
opened dataset reports `file:///…` — so `Dataset.validate(level=...)`,
which forwards that URL, fails in exactly that way on a perfectly good
local store. Pass the path yourself until that is fixed.

---

## Choosing a validation level

| Situation | Recommended level |
|-----------|-----------------|
| Quick structural check (CI, file open) | 1 |
| After writing a new store | 3 |
| After ingest from external format | 3 |
| After rechunking (`building.rechunk`) | 3 |
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

Message text below is quoted from the validator source. L2 and deeper
stamp a `resolution_{N}` prefix on every message; **L1 does not** — it
names the level directory instead, which under the current layout is a
bare integer (`0/`, `1/`), not `resolution_0/`.

### L1 — structure

L1 checks that the store exists, has root metadata, has at least one
resolution level, and that each level has the directories it needs.

```
Store path does not exist: scan.zarrvectors
No root metadata found (expected .zattrs, zarr.json, or metadata.json)
No resolution level directories found
0/vertices/ missing
```

Those are errors. L1 also emits warnings, which do **not** fail
validation:

```
0/vertex_fragments/ missing
1/ has no metadata file
0/links/ exists but has no <delta> subdirs
```

Read the root-metadata message as a list of spellings the check will
*accept*, not as a description of the format. A ZV store is Zarr v3, and
its root document is `zarr.json` — the store fields under
`attributes.zarr_vectors`, the per-level transforms under
`attributes.multiscales`. `.zattrs` is the Zarr **v2** spelling and
`metadata.json` is older still; neither is written by anything in this
package. The check tolerates them so that a store from an older writer
still reaches L2, where its metadata will be read properly or rejected.

The links warning is why a links family with no `<delta>` segments is a
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

*Remedy:* these are all faults in the `zarr_vectors` / `zarr_vectors_level`
blocks of a `zarr.json`, and the honest fix depends on which is true:

- If the **metadata** is wrong and the data is fine (usually a hand-edit),
  correct the attribute with `building.update_root_metadata` or
  `building.update_level_metadata`. Both are read-modify-write over the
  one block they own, so they will not disturb anything else on the
  document.
- If the **data** was actually written under the bad geometry, the
  metadata is telling the truth and the store must be rewritten from
  source. There is no rebinning API in this package.

Read before you write — only edit fields you are certain disagree with the
data on disk:

```python
from zarr_vectors.building import (
    open_store, read_level_metadata, read_root_metadata, update_root_metadata,
)

root = open_store("scan.zarrvectors", mode="r+")

meta = read_root_metadata(root)
print(meta.sid_ndim, meta.chunk_shape, meta.base_bin_shape)
print(read_level_metadata(root, 0).vertex_count)

# ... and only then, if the declaration is the thing that is wrong:
update_root_metadata(root, base_bin_shape=[50.0, 50.0, 50.0])
```

```text
3 (200.0, 200.0, 200.0) (50.0, 50.0, 50.0)
100000
```

The raw attributes are reachable too, as `root.attrs`. It is a
dict-*like* wrapper, not a dict: it supports `attrs[k]`,
`attrs.get(k, default)`, `k in attrs`, `attrs.update(d)`, and
`attrs.to_dict()`. It is not iterable, so `dict(root.attrs)` and
`for k in root.attrs` do not work — use `to_dict()`. On a ZV root it holds
two keys, `zarr_vectors` and `multiscales`.

#### When the level's `arrays_present` is wrong

`arrays_present` is hand-listed at every write site, so it drifts: an
array can sit on disk undeclared, and a reader that gates on the list will
not see it. `building.refresh_arrays_present` re-derives it by walking the
level and writes the answer back:

```python
from zarr_vectors.building import (
    get_resolution_level, open_store, read_level_metadata, refresh_arrays_present,
)

root = open_store("scan.zarrvectors", mode="r+")
print("declared:", read_level_metadata(root, 0).arrays_present)
print("on disk :", refresh_arrays_present(get_resolution_level(root, 0)))
```

```text
declared: ['vertices', 'vertex_attributes', 'object_index']
on disk : ['links', 'object_index', 'vertex_attributes', 'vertex_fragments', 'vertices']
```

That gap is real and ordinary — the store above was written by
`add_points` and then given a pyramid, and neither step declared
`vertex_fragments` or `links`. The function is the single owner of the
field; a coordinator calls it once after a parallel phase.

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

One near neighbour of these *does* have a remedy. An array's
`nonempty_chunks` attribute is the manifest of which cells hold data, and
it is a single shared field that parallel writers skip on purpose. If it
disagrees with the cells actually on disk, `building.rebuild_presence`
re-derives it:

```python
from zarr_vectors.building import get_resolution_level, open_store, rebuild_presence

root = open_store("scan.zarrvectors", mode="r+")
level = get_resolution_level(root, 0)

print(len(level.read_array_meta("vertices")["nonempty_chunks"]))
print(len(rebuild_presence(level, "vertices")))    # one array: the cell keys
print(rebuild_presence(level))                     # the level: the arrays rebuilt
```

```text
125
125
['links/+1/0.0.0', 'vertex_attributes/intensity', 'vertex_fragments', 'vertices']
```

On a healthy store the two counts agree — that is the point of running it.
They diverge only when the manifest has actually drifted from the cells on
disk, which is the fault this repairs.

Passing an array path returns that array's cell keys; passing nothing
walks every per-chunk array in the level and returns the paths it
rebuilt. Sharded arrays are skipped (`on_sharded="skip"` is the default
here) because sharding runs after the rebuild by contract, so a sharded
array's manifest is already correct; pass `on_sharded="raise"` to assert
that ordering instead.

**Object index errors**

```
resolution_0: obj 1042 refs non-existent chunk (8,8,4)
resolution_0: obj 1042 refs fragment_idx=12 >= 8
```

The object index points at a chunk or fragment that is not there, usually
after something moved vertices without rewriting the index. *Remedy:* the
index can be rewritten with `building.write_object_index` if — and only
if — you can reconstruct the correct manifests yourself; the package will
not derive them for you.

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
from zarr_vectors.building import finalize_links, get_resolution_level, open_store

root = open_store("tracts.zarrvectors", mode="r+")
level_group = get_resolution_level(root, 0)
partition = finalize_links(level_group, delta=0)
print(partition.num_links, partition.num_physical_records)
```

`finalize_links` rescans every offsets array and every cell to recompute
the counts, so it must run after all cells are on disk and **before**
sharding. See [Cloud stores](cloud_stores.md#decentralized-link-writes)
for the full decentralized write-then-finalize sequence.

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

The second warning is routine on any point cloud that has been given a
pyramid: `build_pyramid` writes the cross-level `links/+1` and `links/-1`
families that tie a metavertex to the vertices it stands for, so a
perfectly good multiscale point cloud passes L5 with this one warning.

---

### L5 — multiresolution

L5 checks pyramid shape and monotonicity.

```
Levels [0, 1, 3], expected [0, 1, 2]
resolution_2: 5000 > resolution_1 (4000)
```

The second says a coarser level has *more* vertices than the level below
it, which means the coarsening did not actually coarsen.

*Remedy:* rebuild the pyramid. On the supported surface that is
`Dataset.build_pyramid`, which covers the common case — every level above
level 0, rebuilt from scratch:

```python
import zarr_vectors as zv

ds = zv.open("scan.zarrvectors", mode="r+")
report = ds.build_pyramid(factors=[(2.0, 1.0), (2.0, 1.0)], chunk_scale_factors=[2, 2])
print(report["levels_created"])
```

```text
2
```

That re-derives the levels from the schema you give it. What it does *not*
do is re-coarsen from an arbitrary source level while reusing each
existing target level's own recorded `bin_ratio` / `object_sparsity` /
`chunk_shape` — which is what you want when the pyramid's *shape* is right
and only its contents are stale. That operation exists, but only as an
internal name:

```python
# Reaching past the contract: zarr_vectors.ops is internal and may change
# between releases.  There is no `building` or `api` equivalent yet — a gap
# to report rather than a reason to make a habit of importing from ops.
from zarr_vectors.building import open_store
from zarr_vectors.ops.refresh import rebuild_pyramid_from_level

root = open_store("scan.zarrvectors", mode="r+")
summaries = rebuild_pyramid_from_level(root, source_level=0)
print([s["vertex_count"] for s in summaries])
```

```text
[1000, 27]
```

This replaces the old level data in place. It trusts `source_level`
completely — validate that level at L3 first. Prefer
`Dataset.build_pyramid` unless you specifically need the existing levels'
recorded parameters preserved.

If instead the `multiscales` metadata itself is stale — the scale and
translation transforms that make the pyramid readable as OME-NGFF —
regenerate it. This one *is* supported:

```python
from zarr_vectors.building import open_store, write_multiscale_metadata

root = open_store("scan.zarrvectors", mode="r+")
multiscales = write_multiscale_metadata(root)
print([d["path"] for d in multiscales[0]["datasets"]])
print(multiscales[0]["datasets"][1]["coordinateTransformations"])
```

```text
['0', '1']
[{'type': 'scale', 'scale': [2.0, 2.0, 2.0]}, {'type': 'translation', 'translation': [50.0, 50.0, 50.0]}]
```

`building.read_multiscale_metadata` reads back the same document without
rewriting it, which is the safer call when you only want to look.

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

`str(store_path)` on a `Path` is the right spelling here — the validator
wants the filesystem path, not a URL.
