# L1: Structural validation

## Terms

**Structural check**
: A validation check that examines the store's node layout: which nodes
  exist, and the shape of the per-chunk arrays' cell grids. It reads no
  cell data.

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
It checks that the store root, its metadata, and each resolution
level's required nodes are present, and that each level's per-chunk
arrays agree on the rank of the chunk grid they share. It reads no cell
data, and it interprets metadata only as far as the rank check needs.

L1 is the fastest validation level and is appropriate as a first triage
step when opening an unfamiliar store. An L1 failure means the store is
structurally incomplete and cannot be read by any Zarr Vectors reader.

### Scope

L1 is deliberately **narrow**, and narrower than a reader's needs:

- It asks the store, not a filesystem, so it works on any backend a
  reader can open. It resolves nodes through the same lookup readers
  use.
- It opens each per-chunk array's `zarr.json` for one purpose: its
  **rank**, and the arity of the keys in its `nonempty_chunks`
  manifest (see *Chunk grid rank* below). A rank skew is structural:
  the array cannot address the cells its level's keys name, so no
  later level can be trusted over it.
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
| The store opens (path, URL or open `Group`) | Error (returns immediately) |
| The root attributes carry a `zarr_vectors` block | Error |
| The root attributes carry an RFC 8 `ome` node | Warning |
| At least one resolution level exists | Error (returns immediately) |

The two immediate returns are **fatal to the walk**: no per-level
results follow them.

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

#### Chunk grid rank

A level's per-chunk arrays (`vertices`, `vertex_fragments`,
`vertex_attributes/<n>`, `fragment_attributes/<n>`, `links/<d>/<off>`,
`link_attributes/…`) are cells of one chunk grid, so they share one
rank: `sid_ndim`, plus one when the level is chunked by an attribute.
`object_index/`, `object_attributes/` and `groups/` have no chunk grid
and are not checked.

| Rule | Failure type |
|------|--------------|
| Every key in an array's `nonempty_chunks` has as many components as the array has dimensions | Error |
| Every per-chunk array has the same rank as `N/vertices` | Error |
| On a level with `chunk_attribute_values`, `N/vertices` has rank `sid_ndim + 1` | Error |
| On such a level, the leading axis of `N/vertices` has `len(chunk_attribute_values)` entries | Error |

An array with no `nonempty_chunks` attribute lists no keys. That is not
an error, because a `record_presence=False` write leaves it that way until
its rebuild.

Both of the first two rules are needed.
`derive_nonempty_chunks` rebuilds a manifest at its array's own rank, so
after a rebuild a mis-ranked array agrees with its own keys, and only
the comparison with `vertices` catches it. The last rule holds because a
reader resolves an attribute value to a bin by its position in
`chunk_attribute_values`. A leading axis with more entries than that list
means some bin is unreachable, or readable under another bin's label.
`chunk_attribute_values` is read leniently: a level whose metadata
does not parse is reported at L2, and this rule is skipped for it.

### Example L1 report

```
Level 1 validation: FAIL
  6 passed, 2 warnings, 1 errors
  ERROR: 1/vertices/ missing
  WARN:  1/vertex_fragments/ missing
  WARN:  0/links/ exists but has no <delta> subdirs
```

Passing messages take the form `Store root opened`,
`Root metadata file found`, `Found 2 resolution level(s)`,
`0/vertices/ exists`, `0/links/ exists (deltas: 0,+1)`, and
`0/ per-chunk arrays share rank 4 (3 arrays)`.

### Implementation notes for contributors

L1 walks the store through `open_store`, checks for the required
nodes, and reads each per-chunk array's shape and presence manifest:

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
