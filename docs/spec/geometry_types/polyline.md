# Polyline (`polyline`)

## Terms

**Polyline**
: An ordered sequence of vertices connected by consecutive edges:
  `V_0 — V_1 — V_2 — … — V_{n-1}`. A polyline store holds a collection
  of such sequences, each identified by an integer object ID.

**`GEOM_POLYLINE`**
: The geometry type constant `"polyline"`, recorded in the root
  `zarr_vectors` block as `"geometry_types": ["polyline"]`. Note that
  `write_polylines` defaults to `geometry_type="streamline"`; pass
  `geometry_type="polyline"` (or use `Dataset.add_polylines`, which does
  so unless `streamlines=True`) to get a `polyline` store.

**Object ID**
: A non-negative integer uniquely identifying a polyline within the store.
  Object IDs are assigned at write time and are stable across reads.

**Object manifest**
: The set of `(chunk_coord, fragment_indices)` pairs that together contain all
  vertices of a given object. Encoded in the `object_index/` array. See
  [Object manifest](../object_model/object_manifest.md).

**Bridge**
: A stored connection between the last vertex of a polyline segment in
  chunk A and the first vertex of the continuation segment in chunk B.
  Written when a polyline spans multiple chunks. A bridge is an ordinary
  record in the one link family, distinguished only by having non-zero
  offsets — see [Links](../object_model/links.md).

---

## Introduction

The `polyline` type extends `line` with an ordered path of arbitrary
length: each polyline is a named, addressable entity that can be
retrieved by ID. Polylines may span multiple spatial chunks; links with
non-zero offsets preserve inter-chunk connectivity.

Use `polyline` over `line` whenever you need to:
- Retrieve individual paths by ID.
- Assign paths to named groups.
- Preserve path topology across chunk boundaries.
- Store per-path metadata (length, label, confidence).

Use `streamline` instead of `polyline` when the paths are MRI tractography
streamlines and you need to store tractography-specific metadata (step size,
seeding strategy, propagation algorithm).

---

## Technical reference

### Arrays present

| Array path | Required | Description |
|-----------|----------|-------------|
| `vertices/` | Yes | Vertex positions (`float32` by default) |
| `vertex_fragments/` | Yes | Fragment index over `vertices/` rows |
| `object_index/` | Yes | Per-object manifest blobs naming fragments |
| `links/0/<offsets>/` | Only when a polyline spans chunks | Bridges joining consecutive segments; offsets always non-zero |
| `vertex_attributes/<name>/` | No | Per-vertex attributes |
| `object_attributes/<name>/` | No | Per-polyline attributes |
| `groups/` | No | Group ID → [object IDs] |

There is no `cross_chunk_links/` array — that family was merged into
`links/`. A store whose polylines each sit within one chunk has **no
`links/` group at all**; see *Why polylines have no intra-chunk links*
below.

### Vertex ordering within a chunk

For a polyline that contributes vertices to multiple bins within a chunk,
its vertices appear in multiple fragments. Within each fragment, vertices are stored in
traversal order (from the start of the polyline toward the end).

Across fragments within the same chunk, vertices from different
polylines may be interleaved. A fragment holds one contiguous run of a
single polyline's vertices, in traversal order.

### Why polylines have no intra-chunk links

`polyline` uses `links_convention: implicit_sequential`. Within a
fragment, **vertex order is the topology**: consecutive vertices are
connected by definition, so a consecutive pair needs no link record.

The all-zero-offsets (intra-chunk) array would therefore never hold a
row, and `write_polylines` does not create a links array up front.
`write_links` creates exactly the non-zero-offset arrays the bridges
land in. **Every link record in a `polyline` store has non-zero
offsets.**

### Bridges in `links/0/<offsets>/`

A bridge is written when a polyline's traversal leaves one chunk and
enters another. Each record is two `(chunk, vertex_index)` endpoints:
the last vertex of the segment in chunk A and the first vertex of the
continuation in chunk B. `vi_k` is local to chunk `src + o_k`, so the
second endpoint's index is local to the chunk the offsets name — not to
the source.

The family is written with the default policy — undirected,
`store="canonical"` — so each bridge is stored once, under the
lexicographically positive offset, with `perm_idx` recovering the
original traversal direction.

A polyline crossing chunks *n* times yields *n* bridges; its vertices
are distributed across *n + 1* fragments, which `object_index/` lists in
traversal order.

### Write API

```python
import numpy as np
from zarr_vectors.types.polylines import write_polylines

rng = np.random.default_rng(0)
polylines = [
    rng.normal(0, 50, (rng.integers(10, 60), 3)).cumsum(0).astype(np.float32)
    for _ in range(1000)
]

summary = write_polylines(
    "paths.zarrvectors",
    polylines,
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
    geometry_type="polyline",        # default is "streamline"
    # Optional group assignment: {int group id: [polyline indices]}
    groups={
        0: list(range(500)),
        1: list(range(500, 1000)),
    },
    # Optional per-polyline attributes, shape (O,) or (O, C)
    object_attributes={
        "length": np.array([np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
                            for p in polylines], dtype=np.float32),
    },
)
# summary: {"polyline_count", "vertex_count", "chunk_count",
#           "cross_chunk_link_count", "group_count"}
```

`chunk_shape` is required. Other keywords: `bin_shape`, `bounds`,
`vertex_attributes` (`{name: [array_per_polyline, ...]}`, each `(N_k,)` or
`(N_k, C)`), `fragment_attributes`, `group_attributes`, `dtype`,
`backend`, `chunk_by_attribute`, `out_of_bounds`, `compressor`,
`shard_shape`. `out_of_bounds="ignore"` is rejected (dropping vertices
would break path connectivity); use `"raise"` (default) or `"expand"`.
An empty list raises `ArrayError`.

Through the dataset API the equivalent call is
`ds.add_polylines(polylines, attributes=..., object_attributes=..., groups=...)`.

### Read API

```python
import numpy as np
from zarr_vectors.types.polylines import read_polylines

# All polylines
result = read_polylines("paths.zarrvectors")
print(result["polyline_count"])        # 1000
print(result["vertex_count"])          # total vertices
print(result["object_ids"][:3])        # source object id per returned polyline

# Each entry of result["polylines"] is a LIST of fragment arrays (one per
# stored run); concatenate to get the full path.
path0 = np.concatenate(result["polylines"][0], axis=0)   # (N_0, D)

# By object ID
result = read_polylines("paths.zarrvectors", object_ids=[0, 5, 42])
print(result["polyline_count"])        # 3

# By group (integer group ids)
result = read_polylines("paths.zarrvectors", group_ids=[0])
print(result["polyline_count"])        # 500

# Spatial bbox
result = read_polylines(
    "paths.zarrvectors",
    bbox=(np.array([0., 0., 0.]), np.array([100., 100., 100.])),
)
```

Return keys: `polylines`, `object_ids`, `vertex_attributes` (concatenated
across the returned fragments, in the same order; a column that cannot be
aligned is omitted), `ndim`, `polyline_count`, `vertex_count`. Other
keywords: `level`, `chunks`, `attribute_filter`, `backend`.

### Spatial query semantics

A `bbox` query on a polyline store resolves the box to the chunks it
touches and returns the **complete** geometry of every polyline that has
at least one fragment in one of those chunks. The match is per chunk, so
a polyline whose fragment sits in a touched chunk but outside the box
itself is still returned in full. There is no `clip=` option.

To crop at the *segment* level, pass `chunks=[...]` (a whitelist of
chunk-coordinate tuples): only fragments stored in the listed chunks are
returned, and each surviving contiguous run becomes its own output
polyline, so `polyline_count` can exceed the number of objects and
`object_ids` can repeat. When both `bbox` and `chunks` are given the
effective whitelist is their intersection.

### Groupings

Groups are collections of object IDs, `{group_id: [object_id, ...]}`
with integer group ids, stored in the `groups/` array (per-group
attributes in `group_attributes/<name>/`). `read_polylines(group_ids=[...])`
expands a group to its member objects.

### Validation

L1: `vertices/` exists at every level. `object_index/` and `links/` are
recorded when present but are **not** required at L1.

L3:
- Every manifest entry names a chunk present at the level and a
  `fragment_index` below that chunk's fragment count.
- Every `links/0/` offsets segment parses, and — the family being
  undirected, canonical, and intra-level — each offset is
  lexicographically non-negative and offsets are non-decreasing.
- Every record's endpoints name chunks present at the level.

L4: `links_convention` MUST be `implicit_sequential` for `polyline`.

Gap detection (a vertex with no outgoing edge that is not a polyline's
last vertex) is **not** implemented at any level. Neither is a check
that bridges reference valid vertex indices *within* the endpoint
chunk — only the chunk's existence is checked. See
[Validation overview](../validation/overview.md).
