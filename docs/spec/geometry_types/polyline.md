# Polyline (`polyline`)

## Terms

**Polyline**
: An ordered sequence of vertices joined by consecutive edges,
  `V_0 — V_1 — … — V_{n-1}`. A polyline store holds a collection of such
  sequences; each is one object, identified by its index in the input list.

**`GEOM_POLYLINE`**
: The geometry type constant `"polyline"`, stored in the `geometry_types`
  list of the root attributes.

**Object manifest**
: The list of `(chunk, fragment_index)` pairs that together hold one
  object's vertices, stored in `object_index/`. See
  [Object manifest](../object_model/object_manifest.md).

**Cross-chunk link**
: A link record joining the last vertex of a polyline's run in one chunk to
  the first vertex of its next run in another chunk.

## Introduction

The `polyline` type stores ordered paths of any length. Each path is an
object that can be read by id, assigned to groups and given per-path
attributes (length, label, confidence).

Connectivity follows `links_convention: implicit_sequential`: within a
fragment, consecutive vertices are joined by definition, so no link record
is stored for them. The writer splits each polyline where it crosses a chunk
boundary; each run of consecutive vertices in one chunk becomes one fragment,
and consecutive runs are joined by a cross-chunk link. A polyline that crosses
chunk boundaries *n* times has *n + 1* fragments, listed in path order in its
manifest, and *n* cross-chunk links.

[`streamline`](streamline.md) stores the same layout under a different
geometry type.

## Technical reference

### Arrays present

At level 0 of a store written by `write_polylines`:

| Path | Present | Description |
|------|---------|-------------|
| `vertices/` | Always | Vertex positions, each fragment in path order |
| `vertex_fragments/` | Always | Fragment index over `vertices/` rows |
| `object_index/` | Always | One manifest per polyline |
| `links/0/<offsets>/` | When a polyline crosses a chunk | Cross-chunk links, in non-zero-offset arrays only |
| `vertex_attributes/<name>/` | With `vertex_attributes=` | One row per vertex |
| `object_attributes/<name>/` | With `object_attributes=` | One row per polyline |
| `fragment_attributes/<name>/` | With `fragment_attributes=` | One row per fragment |
| `groups/` | With `groups=` | Group id → object ids |
| `group_attributes/<name>/` | With `group_attributes=` | One row per group |

A store whose polylines each sit in one chunk has no `links/` group. The link
family is undirected with `store: "canonical"`, so each cross-chunk link is
stored once, under the lexicographically positive offset; see
[Links](../object_model/links.md).

### Root attributes

A polyline store has no type-specific root keys. The writer sets these values
in the `zarr_vectors` block (see [Root metadata](../layout/root_metadata.md)):

```json
{
  "geometry_types": ["polyline"],
  "links_convention": "implicit_sequential",
  "object_index_convention": "standard",
  "cross_chunk_strategy": "explicit_links"
}
```

### Write and read API

```python
import numpy as np
from zarr_vectors.building import write_polylines
from zarr_vectors.types.polylines import read_polylines

rng = np.random.default_rng(0)
polylines = [
    (rng.uniform(100, 400, 3) + rng.normal(0, 5, (rng.integers(10, 60), 3)).cumsum(0)).astype(np.float32)
    for _ in range(100)
]
lengths = np.array(
    [np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in polylines], dtype=np.float32,
)

summary = write_polylines(
    "paths.zarrvectors",
    polylines,
    chunk_shape=(100.0, 100.0, 100.0),
    geometry_type="polyline",          # the default is "streamline"
    groups={0: list(range(50)), 1: list(range(50, 100))},
    group_attributes={"label": np.array([7, 9], dtype=np.int32)},
    object_attributes={"length": lengths},
)
print(summary)
# {'polyline_count': 100, 'vertex_count': 3180, 'chunk_count': 63,
#  'cross_chunk_link_count': 344, 'group_count': 2}

result = read_polylines("paths.zarrvectors")
print(result["polyline_count"], result["vertex_count"])   # 100 3180
first = np.concatenate(result["polylines"][0])     # one array per run
print(len(result["polylines"][0]), np.array_equal(first, polylines[0]))   # 3 True

print(read_polylines("paths.zarrvectors", object_ids=[0, 5, 42])["polyline_count"])   # 3
print(read_polylines("paths.zarrvectors", group_ids=[0])["polyline_count"])           # 50
print(read_polylines(
    "paths.zarrvectors",
    bbox=(np.array([0.0, 0.0, 0.0]), np.array([200.0, 200.0, 200.0])),
)["polyline_count"])   # 38
```

`read_polylines` takes `level=`, `object_ids=`, `group_ids=`, `bbox=`,
`chunks=` and `attribute_filter=`, and returns:

| Key | Type | Description |
|-----|------|-------------|
| `polylines` | `list[list[ndarray]]` | Per polyline, its runs in path order; concatenate them for the whole path |
| `object_ids` | `ndarray` | Object id of each returned polyline |
| `vertex_attributes` | `dict[str, ndarray]` | Per-vertex attributes, concatenated over the result |
| `ndim` | `int` | Spatial dimensionality |
| `polyline_count` | `int` | Polylines returned |
| `vertex_count` | `int` | Vertices returned |

A `bbox=` read returns the whole of every polyline with a run in a chunk
the box touches, so paths are not cut at the box. A `chunks=` read returns
only the runs stored in the named chunks, and each unbroken stretch of them
comes back as its own polyline, so `polyline_count` can exceed the number of
objects.

### Validation

- L1: `vertices/` missing at a level is an error. `object_index/` and
  `links/` are not required.
- L3: each manifest entry names a chunk present at the level and a fragment
  index below that chunk's fragment count. Each `links/0/` offsets name
  parses, no offset is lexicographically negative and the offsets are
  non-decreasing. Each record's chunks exist at the level.
- L4: `links_convention` other than `implicit_sequential` is an error.

No level checks that a link's vertex index exists within its chunk, or that
a polyline's runs join up. See [Validation overview](../validation/overview.md).
