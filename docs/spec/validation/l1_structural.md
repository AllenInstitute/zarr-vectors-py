# L1: Structural validation

## Terms

**Structural check**
: A check on the store's node layout: which nodes exist, and the rank of
  the per-chunk arrays' cell grids. It reads no cell data.

**Level group**
: A group under the store root whose name is an integer (`0/`, `1/`, …).
  Other root members, such as `parametric/`, are not levels.

## Introduction

L1 checks that the store opens, that the root has Zarr Vectors metadata,
that each level has its required nodes, and that each level's per-chunk
arrays share one chunk-grid rank. It works through the store API, so it
runs on any backend. Rules per geometry type are at L4 (see
[Validation overview](overview.md#conformance-levels)).

## Technical reference

L1 is `validate_structure` in `zarr_vectors/validate/structure.py`. Each
check adds a message to the `ValidationResult` (see
[Validation overview](overview.md#validationresult-api)). `N` is the level.

### Root

| Check | Severity | Message |
|-------|----------|---------|
| The store opens (path, URL or open `Group`); L1 stops if not | error | `Cannot open store: ...` |
| The root attributes have a `zarr_vectors` block | error | `No root metadata found ...` |
| The root attributes have an RFC 8 `ome` node; the message suggests `zarr_vectors.building.stamp_ome_node(store)` | warning | `No RFC 8 'ome' node in the root attributes; ...` |
| At least one level group exists; L1 stops if not | error | `No resolution level directories found` |

### Each level

| Check | Severity | Message |
|-------|----------|---------|
| The level group opens; the level is skipped if not | error | `N/ cannot be opened: ...` |
| `N/vertices/` exists | error | `N/vertices/ missing` |
| `N/vertex_fragments/` exists | warning | `N/vertex_fragments/ missing` |
| The level's attributes are not empty | warning | `N/ has no metadata file` |
| `N/links/`, when present, holds at least one `<delta>` group | warning | `N/links/ exists but has no <delta> subdirs` |
| The per-chunk arrays can be listed | warning | `N/ per-chunk arrays cannot be listed: ...` |

`vertex_attributes/`, `fragment_attributes/`, `object_index/`,
`object_attributes/`, `groups/` and the root's `parametric/` add a pass
message when present and nothing when absent. Offsets names under
`links/<delta>/` are checked at [L3](l3_consistency.md).

### Chunk grid rank

A level's per-chunk arrays (`vertices`, `vertex_fragments`,
`vertex_attributes/<n>`, `fragment_attributes/<n>`, `links/<d>/<off>`,
`link_attributes/…`) share one rank: `sid_ndim`, plus one when the level
is chunked by an attribute. `object_index/`, `object_attributes/` and
`groups/` are not checked. All findings are errors.

| Check | Message |
|-------|---------|
| Every key in an array's `nonempty_chunks` has as many components as the array has dimensions | `N/<path> has rank <r> but <k> of its presence keys do not ...` |
| Every per-chunk array has the rank of `N/vertices` | `N/<path> has rank <r> but N/vertices has rank <v>; ...` |
| With `chunk_attribute_values`, `N/vertices` has rank `sid_ndim + 1` | `N/ is chunked by attribute, so its arrays need rank ...` |
| With `chunk_attribute_values`, the leading axis of `N/vertices` has `len(chunk_attribute_values)` entries | `N/vertices has <b> bins on its leading axis but ...` |

An array with no `nonempty_chunks` lists no keys, which is not an error
(a `record_presence=False` write leaves it so until presence is rebuilt).
A level whose `chunk_attribute_values` does not parse is reported at L2
and skipped here.

### Example

A two-level line store whose level 1 lost `vertices/` and
`vertex_fragments/`:

```
Level 1 validation: FAIL
  15 passed, 1 warnings, 1 errors
  ERROR: 1/vertices/ missing
  WARN:  1/vertex_fragments/ missing
```

To run L1 alone:

```python
from zarr_vectors.validate.structure import validate_structure

print(validate_structure("contacts.zarrvectors").summary())
```
