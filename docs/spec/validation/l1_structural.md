# L1: Structural validation

## Terms

**Structural check**
: A check on the store's node layout: which nodes exist, and the rank of
  the per-chunk arrays' cell grids. It reads no cell data.

**Level group**
: A group under the store root whose name is an integer (`0/`, `1/`, …).
  Other root members, such as `parametric/`, are not levels.

## Introduction

L1 answers the question "does this store have the right shape?". It checks
that the store opens, that the root carries Zarr Vectors metadata, that each
level has its required nodes, and that each level's per-chunk arrays agree on
the rank of the chunk grid they share. It is the fastest level and a good
first step with an unfamiliar store.

L1 works through the store API, so it runs on any backend a reader can open.
It applies one required set to every store, whatever its geometry type;
type-specific requirements are checked at L4 (see
[Validation overview](overview.md#conformance-levels)).

## Technical reference

### Checks performed

L1 is `validate_structure` in `zarr_vectors/validate/structure.py`. Each
check adds a message to the `ValidationResult` (see
[Validation overview](overview.md#validationresult-api)).

#### Root

| Rule | Failure |
|------|---------|
| The store opens (path, URL or open `Group`) | Error, and L1 stops |
| The root attributes carry a `zarr_vectors` block | Error |
| The root attributes carry an RFC 8 `ome` node | Warning |
| At least one level group exists | Error, and L1 stops |

The `ome` warning suggests `zarr_vectors.building.stamp_ome_node(store)`,
which adds the node without rewriting data.

#### Each level

For every level group, in ascending order:

| Rule | Failure |
|------|---------|
| The level group opens | Error, and the level is skipped |
| `N/vertices/` exists | Error |
| `N/vertex_fragments/` exists | Warning |
| The level's attributes are not empty | Warning (`N/ has no metadata file`) |
| `N/links/`, when present, holds at least one `<delta>` group | Warning |
| The level's per-chunk arrays can be listed | Warning |

`vertices/` is the only per-level node whose absence is an error. L1 does not
look inside `links/<delta>/`; offsets names are checked at
[L3](l3_consistency.md).

These nodes add a pass message when present and nothing when absent:
`vertex_attributes/`, `fragment_attributes/`, `object_index/`,
`object_attributes/`, `groups/`, and `parametric/` at the root.

#### Chunk grid rank

A level's per-chunk arrays (`vertices`, `vertex_fragments`,
`vertex_attributes/<n>`, `fragment_attributes/<n>`, `links/<d>/<off>`,
`link_attributes/…`) are cells of one chunk grid, so they share one rank:
`sid_ndim`, plus one when the level is chunked by an attribute.
`object_index/`, `object_attributes/` and `groups/` have no chunk grid and
are not checked.

| Rule | Failure |
|------|---------|
| Every key in an array's `nonempty_chunks` has as many components as the array has dimensions | Error |
| Every per-chunk array has the same rank as `N/vertices` | Error |
| On a level with `chunk_attribute_values`, `N/vertices` has rank `sid_ndim + 1` | Error |
| On such a level, the leading axis of `N/vertices` has `len(chunk_attribute_values)` entries | Error |

An array with no `nonempty_chunks` attribute lists no keys; that is not an
error, because a write with `record_presence=False` leaves it so until
presence is rebuilt.

Both of the first two rules are needed: `derive_nonempty_chunks` rebuilds a
manifest at its array's own rank, so a mis-ranked array agrees with its own
keys, and only the comparison with `vertices` catches it. The last rule holds
because a reader resolves an attribute value to a bin by its position in
`chunk_attribute_values`. A level whose `chunk_attribute_values` does not
parse is reported at L2 and skipped here.

### Example L1 report

A two-level line store whose level 1 lost its `vertices/` and
`vertex_fragments/` arrays:

```
Level 1 validation: FAIL
  15 passed, 1 warnings, 1 errors
  ERROR: 1/vertices/ missing
  WARN:  1/vertex_fragments/ missing
```

Pass messages read like `Store root opened`, `Root metadata file found`,
`Found 2 resolution level(s)`, `0/vertices/ exists`,
`0/links/ exists (deltas: 0)` and
`0/ per-chunk arrays share rank 3 (13 arrays)`.

### Running L1 alone

```python
from zarr_vectors.validate.structure import validate_structure

result = validate_structure("contacts.zarrvectors")
print(result.summary())
```

`validate_structure` takes only the store. A new geometry type that needs its
own required nodes (see
[Adding geometry types](../contributing/adding_geometry_types.md)) belongs at
L4, in `zarr_vectors/validate/conformance.py`.
