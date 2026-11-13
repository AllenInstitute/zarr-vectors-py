# Skeleton (`skeleton`)

## Terms

**Skeleton**
: A rooted tree describing the branching shape of a neuron, vessel or other
  tubular structure. Each vertex has one parent, except a root, which has
  none.

**`GEOM_SKELETON`**
: The geometry type constant `"skeleton"`, stored in the `geometry_types`
  list of the root attributes.

**Parent edge**
: An edge written `[child, parent]` as vertex indices.

**Branch record**
: A link record `[child, parent]` stored for an edge that the implicit
  rule (below) does not give.

**SWC**
: The text format for neuronal morphology, one vertex per row:
  `id, type, x, y, z, radius, parent_id`. Radius and type are stored as
  ordinary per-vertex attributes; the format reserves no attribute names
  or type codes for them.

## Introduction

The `skeleton` type stores rooted trees with
`links_convention: implicit_sequential_with_branches`. Most parent edges are
not stored: within a fragment, a vertex's parent is the row before it. Only
the edges this rule does not give are stored as link records. Each tree, or
each piece of one, is an object.

The read rule, which every reader applies:

- A vertex with no record naming it as child, and that does not open a
  fragment, has the row before it as parent. This is the only implicit edge.
- Every record `[child, parent]` is an edge, and stops its child taking an
  implicit parent.
- A record whose parent is outside the read (another object, or outside the
  `bbox` or `chunks` scope) gives no edge, but still stops the implicit one;
  that child is a root of the read.
- A vertex with several records keeps all of them.

Two writers produce skeleton stores: `write_graph(kind="skeleton")` takes a
whole dataset in one call, and a streaming writer writes one chunk at a time.
SWC import and export are in the companion package
[zarr-vectors-tools](https://zarr-vectors-tools.readthedocs.io/en/latest/convert/skeletons.html).

## Technical reference

### Arrays present

At level 0 of a store written by `write_graph(kind="skeleton")`:

| Path | Present | Description |
|------|---------|-------------|
| `vertices/` | Always | Vertex positions, one fragment per object per chunk, depth-first order |
| `vertex_fragments/` | Always | Fragment index over `vertices/` rows |
| `links/0/0.0.0/` | Always, possibly empty | Branch records within one chunk, chunk-local indices |
| `links/0/<offsets>/` | When an edge crosses chunks | Every cross-chunk parent edge |
| `link_fragments/` | Always | Fragment index over the link rows |
| `object_index/` | Always | One manifest per object |
| `vertex_attributes/<name>/` | With `vertex_attributes=` | One row per vertex, for example `radius` |
| `link_attributes/<name>/0/<offsets>/` | With `link_attributes=` | One row per stored record |
| `object_attributes/<name>/`, `fragment_attributes/<name>/` | With the matching argument | One row per object or fragment |

The link family has `link_width: 2`, `directed: false` and
`store: "canonical"`; see [Links](../object_model/links.md).

The streaming writer's level 0 has the same `vertices/`, `vertex_fragments/`,
`links/0/`, `link_fragments/` and `object_index/` arrays, plus
`fragment_attributes/segment_id/` (one `uint64` source id per fragment) and,
once the object index is written, `object_attributes/segment_id/`. Its link
family has `directed: true`.

### Root attributes

A skeleton store has no type-specific root keys. The writers set these
values in the `zarr_vectors` block (see
[Root metadata](../layout/root_metadata.md)):

```json
{
  "geometry_types": ["skeleton"],
  "links_convention": "implicit_sequential_with_branches",
  "object_index_convention": "standard",
  "cross_chunk_strategy": "explicit_links"
}
```

The streaming writer does not set `cross_chunk_strategy`.

### Whole-dataset writer

```python
import numpy as np
from zarr_vectors.building import write_graph
from zarr_vectors.types.graphs import read_graph

rng = np.random.default_rng(0)
n = 200
parent = np.array([-1] + [int(rng.integers(max(0, i - 3), i)) for i in range(1, n)])
positions = np.zeros((n, 3))
positions[0] = (500.0, 500.0, 500.0)
for i in range(1, n):
    positions[i] = positions[parent[i]] + rng.normal(0, 30, 3)
positions = positions.astype(np.float32)
edges = np.stack([np.arange(n), parent], axis=1)    # [child, parent]; [0, -1] marks the root
radius = rng.uniform(0.2, 3.0, n).astype(np.float32)

summary = write_graph(
    "neuron.zarrvectors",
    positions,
    edges,
    chunk_shape=(200.0, 200.0, 200.0),
    kind="skeleton",
    vertex_attributes={"radius": radius},
)
print(summary)
# {'node_count': 200, 'edge_count': 199, 'chunk_count': 11, 'intra_edge_count': 29,
#  'cross_edge_count': 78, 'object_count': 1, 'kind': 'skeleton'}

result = read_graph("neuron.zarrvectors")
print(result["node_count"], result["edge_count"])   # 200 199
```

`write_graph(kind="skeleton")`:

- takes edge rows `[child, parent]` in any row order and vertices in any
  order. A row `[v, -1]` declares `v` a root and is not an edge; any other
  negative index raises `ValueError`. An object with more than one root
  raises `ArrayError`; an object may hold pieces of several trees.
- reorders vertices depth-first from each root, and puts each fragment's
  parentless vertex first.
- stores a record for every edge the read rule does not give: every
  cross-chunk edge, every edge whose child is not the row after its parent
  or opens a fragment, and every parent edge of a vertex that has more than
  one.
- stores `link_attributes` rows only for the stored records; values for
  edges given by the read rule are not kept.

`read_graph` returns `positions`, `edges` as `[child, parent]` rows,
`node_count` and `edge_count`, and takes the arguments described under
[Graph](graph.md#write-and-read-api). A scoped read returns the induced
subgraph.

### Streaming writer

For data too large for one call, `init_skeleton_store` creates the store and
its arrays, `write_skeleton_chunk` writes one chunk's pieces,
`write_skeleton_cross_chunk_links` writes the edges between chunks, and
`finalize_skeleton_store` closes the write. The object index is written by
the caller.

```python
import numpy as np
from zarr_vectors.building import (
    finalize_skeleton_store, init_skeleton_store, read_skeleton_by_segment_id,
    write_object_attributes, write_object_index, write_skeleton_chunk,
    write_skeleton_cross_chunk_links,
)

root, level0 = init_skeleton_store(
    "stream.zarrvectors",
    chunk_shape=(100.0, 100.0, 100.0),
    bounds=([0.0, 0.0, 0.0], [200.0, 100.0, 100.0]),
    ndim=3,
    attribute_dtypes={"radius": "float32"},
)
# Segment 77: a branched piece in chunk (1, 0, 0), a two-vertex piece in (0, 0, 0)
# whose first vertex hangs off vertex 2 of the first piece.
piece_a = {
    "segment_id": 77,
    "positions": np.array([[110, 10, 10], [120, 10, 10], [130, 10, 10], [120, 20, 10]], np.float32),
    "edges": np.array([[1, 0], [2, 1], [3, 1]]),           # [child, parent], root omitted
    "attributes": {"radius": np.ones(4, np.float32)},
    "anchors": {"parent": 2},
}
piece_b = {
    "segment_id": 77,
    "positions": np.array([[90, 10, 10], [80, 10, 10]], np.float32),
    "edges": np.array([[1, 0]]),
    "attributes": {"radius": np.ones(2, np.float32)},
    "anchors": {"child": 0},
}
records_a, anchors_a = write_skeleton_chunk(level0, (1, 0, 0), [piece_a])
records_b, anchors_b = write_skeleton_chunk(level0, (0, 0, 0), [piece_b])
write_skeleton_cross_chunk_links(
    level0, [(anchors_a["parent"], anchors_b["child"])], ndim=3,   # (parent, child)
)
# The object-index reduce: one object (id 0) for segment 77.
write_object_index(
    level0, {0: [(cc, f) for _, cc, f in records_a + records_b]}, sid_ndim=3, total_objects=1,
)
write_object_attributes(level0, "segment_id", np.array([77], dtype=np.uint64))
finalize_skeleton_store(root)

skeleton = read_skeleton_by_segment_id("stream.zarrvectors", 77)
print(skeleton["fragment_count"], skeleton["positions"].shape, skeleton["edges"].tolist())
# 3 (6, 3) [[1, 0], [2, 1], [5, 4], [3, 1]]
```

`write_skeleton_chunk` splits each piece into linear paths, one fragment per
path, and stores one branch record `[child_local, parent_local]` in
`links/0/0.0.0/` for each path that leaves another path. It returns the
`(segment_id, chunk, fragment_index)` records for the object index and the
chunk-local positions of the requested `anchors`.

`write_skeleton_cross_chunk_links` takes `((chunk, vertex), (chunk, vertex))`
pairs with endpoint 0 the parent and endpoint 1 the child, and stores them in
that order, filed under the parent's chunk.

`read_skeleton_by_segment_id` returns `positions`, `edges` (`[child,
parent]`), `attributes` and `fragment_count` for one segment id, or `None`
when the id is absent. Its edges are the within-path edges and the branch
records; the cross-chunk records are not read, so the example above returns
no edge between vertex 2 and vertex 4.

### Validation

- L1 and L2: as for any store.
- L3: each `links/0/` offsets name parses, `num_physical_records` matches
  the rows on disk, and each record's chunks exist at the level. The
  lexicographic-sign and ordering checks apply to the `write_graph` family,
  which is undirected, and not to the streaming writer's directed family.
- L4: `links_convention` other than `implicit_sequential_with_branches` or
  `explicit` is an error.

No level checks that a skeleton is a tree, that link vertex indices exist
within their chunk, or attribute values such as radius. See
[Validation overview](../validation/overview.md).
