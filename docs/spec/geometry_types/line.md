# Line (`line`)

## Terms

**Line segment**
: A pair of vertices `(A, B)` joined by a straight edge. A `line` store holds
  a collection of independent segments; each segment is one object.

**`GEOM_LINE`**
: The geometry type constant `"line"`, stored in the `geometry_types` list
  of the root attributes.

**Split segment**
: A segment whose two endpoints fall in different chunks. It is stored as
  two single-vertex fragments joined by one link record.

## Introduction

The `line` type stores independent two-point segments: contact sites (the two
ends of a synapse), connectors, pairwise distance annotations. Each segment is
an object with its own manifest, so segments can be read by id.

Connectivity follows `links_convention: implicit_sequential`: within a
fragment, vertex order carries the segment. A segment inside one chunk needs
no link record; only split segments are stored in `links/`. For ordered paths
of any length, use [`polyline`](polyline.md).

## Technical reference

### Arrays present

At level 0 of a store written by `write_lines`:

| Path | Present | Description |
|------|---------|-------------|
| `vertices/` | Always | Endpoint positions |
| `vertex_fragments/` | Always | Fragment index over `vertices/` rows |
| `object_index/` | Always | One manifest per segment |
| `links/0/<offsets>/` | When a segment is split | One record per split segment, in a non-zero-offset array |
| `vertex_attributes/<name>/` | With `vertex_attributes=` | One row per endpoint |
| `object_attributes/<name>/` | With `object_attributes=` | One row per segment |

The two cases are written as follows:

| Case | Fragments | Manifest | Link |
|------|-----------|----------|------|
| Both endpoints in one chunk | One fragment holding both vertices | `[(c, f)]` | None |
| Endpoints in chunks `c_a` and `c_b` | One single-vertex fragment in each chunk | `[(c_a, f_a), (c_b, f_b)]` | One record in `links/0/<offsets>/` |

Within each chunk, fragments are assigned in segment order. A store whose
segments all sit in one chunk has no `links/` group. The link family is
undirected with `store: "canonical"`, so each split segment is stored once,
under the lexicographically positive offset; see
[Links](../object_model/links.md).

### Root attributes

A line store has no type-specific root keys. The writer sets these values in
the `zarr_vectors` block (see [Root metadata](../layout/root_metadata.md)):

```json
{
  "geometry_types": ["line"],
  "links_convention": "implicit_sequential",
  "object_index_convention": "standard",
  "cross_chunk_strategy": "explicit_links"
}
```

### Write and read API

```python
import numpy as np
from zarr_vectors.building import write_lines
from zarr_vectors.types.lines import read_lines

rng = np.random.default_rng(0)
starts = rng.uniform(0, 500, (1000, 3))
endpoints = np.stack([starts, starts + rng.normal(0, 20, (1000, 3))], axis=1).astype(np.float32)
strength = rng.uniform(0, 1, 1000).astype(np.float32)

summary = write_lines(
    "contacts.zarrvectors",
    endpoints,                     # (N, 2, D): one row per segment
    chunk_shape=(100.0, 100.0, 100.0),
    object_attributes={"strength": strength},
)
print(summary)
# {'line_count': 1000, 'chunk_count': 192, 'cross_chunk_count': 388}

result = read_lines("contacts.zarrvectors")
print(result["line_count"], result["endpoints"].shape)   # 1000 (1000, 2, 3)

result = read_lines("contacts.zarrvectors", object_ids=[0, 5, 42])
print(result["line_count"])   # 3

result = read_lines(
    "contacts.zarrvectors",
    bbox=(np.array([0.0, 0.0, 0.0]), np.array([100.0, 100.0, 100.0])),
)
print(result["line_count"])   # 6
```

`cross_chunk_count` is the number of split segments. `read_lines` takes
`level=`, `object_ids=`, `bbox=` and `attribute_filter=`, and returns
`endpoints` `(M, 2, D)`, `vertex_attributes` and `line_count`. A bounding-box
read keeps a segment when either endpoint is inside the box.

### Validation

- L1: `vertices/` missing at a level is an error. `links/` is not required.
- L3: each `links/0/` offsets name parses and, because the family is
  undirected and canonical, no offset is lexicographically negative and the
  offsets are non-decreasing. Each record's chunks exist at the level.
- L4: `links_convention` other than `implicit_sequential` is an error.

See [Validation overview](../validation/overview.md).
