# L1: Structural validation

## Terms

**Structural check**
: A validation check that examines only the presence of paths in the
  store, without reading array data or interpreting metadata values.

**Required path**
: A store path that must exist for a valid Zarr Vectors store. Missing required
  paths are L1 errors. L1's required set is **geometry-type
  independent** — see *Scope* below.

**Level directory**
: A directory under the store root whose name parses as an integer
  (`0/`, `1/`, …). Directories whose names do not parse as integers
  (e.g. `parametric/`) are not level directories and are skipped by the
  level walk.

---

## Introduction

L1 validation answers the question: "does this store have the right shape?"
It checks that the store root, its metadata file, and each resolution
level's required paths are present. It does not read any array data and
does not interpret metadata values.

L1 is the fastest validation level and is appropriate as a first triage
step when opening an unfamiliar store. An L1 failure means the store is
structurally incomplete and cannot be read by any Zarr Vectors reader.

### Scope

L1 is deliberately **narrow**, and narrower than a reader's needs:

- It checks **path presence only**, via ordinary filesystem directory
  tests. It does **not** open `zarr.json` and does **not** verify Zarr
  node types. A path that exists as the wrong node type passes L1 and
  fails later.
- It is **not** parameterised by geometry type. L1 applies one required
  set to every store, so it cannot express "streamlines must have
  links". Type-specific connectivity requirements are checked at **L4**
  by [`validate_conformance`](../../../zarr_vectors/validate/conformance.py),
  which is the level that reads `geometry_types` and
  `links_convention`.

Anything stated as a per-type requirement in this page's history was
never enforced at L1; it has been removed rather than left as an
aspiration.

---

## Technical reference

### Checks performed

L1 is implemented by
[`validate_structure`](../../../zarr_vectors/validate/structure.py).
Checks are reported as free-text messages on a `ValidationResult`; they
do not carry stable machine-readable check IDs (see
[Validation overview](overview.md#validationresult-api)).

#### Root level

| Rule | Failure type |
|------|--------------|
| The store path exists | Error (returns immediately) |
| The store path is a directory | Error (returns immediately) |
| At least one of `.zattrs`, `zarr.json`, or `metadata.json` exists at the root | Error |
| At least one level directory exists | Error (returns immediately) |

Any of `.zattrs`, `zarr.json`, or `metadata.json` satisfies the root
metadata check — L1 does not require a specific one, and does not parse
whichever it finds. `metadata.json` is **not** separately recommended
or warned about.

The first three failures are **fatal to the walk**: `validate_structure`
returns as soon as one trips, so no per-level results follow.

#### Per level directory

Repeated for every integer-named directory under the root, in ascending
numeric order:

| Rule | Failure type |
|------|--------------|
| `N/vertices/` exists and is a directory | **Error** |
| `N/vertex_fragments/` exists and is a directory | Warning |
| `N/.zattrs` or `N/zarr.json` exists | Warning |

`vertices/` is the **only** per-level path whose absence is an L1 error.

#### Optional paths (presence recorded, never required)

Each of these emits a *pass* when present and nothing at all when
absent:

`vertex_attributes/`, `fragment_attributes/`, `object_index/`,
`object_attributes/`, `groups/`

At the root, `parametric/` is recorded the same way.

#### Link families

L1 scans the `links/` directory and lists its `<delta>` subdirectories:

| Rule | Failure type |
|------|--------------|
| `N/links/` absent | *(nothing — not required at L1)* |
| `N/links/` present with at least one `<delta>` subdirectory | Pass, listing the deltas found |
| `N/links/` present but containing no `<delta>` subdirectory | Warning |

L1 stops there. It does **not** descend into `<delta>/<offsets>/`, does
not parse offsets segments, and does not require `links/` to exist for
any geometry type. Offsets-segment grammar is checked at
[L3](l3_consistency.md); type-specific connectivity requirements at L4.

> **Legacy tolerance.** `structure.py` applies the same scan to a
> `cross_chunk_links/` directory if one is present. That family was
> merged into `links/` and is never written by a current writer; the
> branch only keeps L1 from being silent about a pre-merge store. It
> imposes no requirement, and a store MUST NOT rely on it.

### Example L1 report

```
Level 1 validation: FAIL
  6 passed, 2 warnings, 1 errors
  ERROR: 1/vertices/ missing
  WARN:  1/vertex_fragments/ missing
  WARN:  0/links/ exists but has no <delta> subdirs
```

Passing messages take the form `Store root exists and is a directory`,
`Root metadata file found`, `Found 2 resolution level(s)`,
`0/vertices/ exists`, and `0/links/ exists (deltas: 0,+1)`.

### Implementation notes for contributors

L1 works by walking the store directory with ordinary filesystem tests
and checking for the presence of required paths:

```python
from zarr_vectors.validate.structure import validate_structure

result = validate_structure(store_root)
print(result.summary())
```

`validate_structure` takes the store path alone — there is no
`geometry_type` parameter and no `REQUIRED_ARRAYS` table to extend. A
new geometry type that needs its own required paths (see
[Adding geometry types](../contributing/adding_geometry_types.md))
belongs in `zarr_vectors/validate/conformance.py` at L4, which is where
geometry types are dispatched on.
