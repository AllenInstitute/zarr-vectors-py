# L1: Structural validation

## Terms

**Structural check**
: A validation check that examines only the presence of paths and
  root attribute blocks in the store, without reading array data or
  interpreting metadata values.

**Required path**
: A store path that must exist for a valid Zarr Vectors store. Missing required
  paths are L1 errors. L1's required set is **geometry-type
  independent** — see *Scope* below.

**Level directory**
: A group under the store root whose name parses as an integer
  (`0/`, `1/`, …). Directories whose names do not parse as integers
  (e.g. `parametric/`) are not level directories and are skipped by the
  level walk.

---

## Introduction

L1 validation answers the question: "does this store have the right shape?"
It checks that the store opens, the root carries a `zarr_vectors`
metadata block, and each resolution
level's required paths are present. It does not read any array data and
does not interpret metadata values.

L1 is the fastest validation level and is appropriate as a first triage
step when opening an unfamiliar store. An L1 failure means the store is
structurally incomplete and cannot be read by any Zarr Vectors reader.

### Scope

L1 is deliberately **narrow**, and narrower than a reader's needs:

- It checks **presence only**. It opens the store through the same
  `open_store` machinery the readers use (so it works on local paths,
  URLs, pre-opened `Group` handles and any supported backend, not just
  directories), but it does not read array data, does not parse
  `zarr_vectors` values, and does not verify Zarr node types. A path
  that exists as the wrong node type passes L1 and fails later.
- It is **not** parameterised by geometry type. L1 applies one required
  set to every store, so it cannot express "streamlines must have
  links". Type-specific connectivity requirements are checked at **L4**
  by [`validate_conformance`](../../../zarr_vectors/validate/conformance.py),
  which is the level that reads `geometry_types` and
  `links_convention`.

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
| The store can be opened (`open_store(..., require_zv=False)`) | Error (returns immediately) |
| The root attributes contain a `zarr_vectors` block | Error |
| The root attributes contain an RFC 8 `ome` node | Warning |
| At least one level group exists | Error (returns immediately) |

The `ome` node is optional (format 0.9.2): a store without one is valid
and fully readable, but cannot be resolved as a member of an OME
collection. The warning's remedy is
`zarr_vectors.building.stamp_ome_node(store)`, which adds the node
without rewriting data.

The first and last failures are **fatal to the walk**: `validate_structure`
returns as soon as one trips, so no per-level results follow. A missing
`zarr_vectors` block is recorded and the walk continues.

##### Per level directory

Repeated for every integer-named directory under the root, in ascending
numeric order:

| Rule | Failure type |
|------|--------------|
| `N/vertices/` exists | **Error** |
| `N/vertex_fragments/` exists | Warning |
| The level group has non-empty attributes | Warning |

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

### Example L1 report

```
Level 1 validation: FAIL
  6 passed, 3 warnings, 1 errors
  ERROR: 1/vertices/ missing
  WARN:  No RFC 8 'ome' node in the root attributes; ...
  WARN:  1/vertex_fragments/ missing
  WARN:  0/links/ exists but has no <delta> subdirs
```

Passing messages take the form `Store root opened`,
`Root metadata file found`, `RFC 8 'ome' node found`, `Found 2 resolution level(s)`,
`0/vertices/ exists`, and `0/links/ exists (deltas: 0,+1)`.

### Implementation notes for contributors

L1 opens the store as a `Group` and checks for the presence of required
paths:

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
