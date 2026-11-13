# Skeleton (`skeleton`)

## Terms

**Skeleton**
: A rooted tree describing the branching shape of a neuron, vessel or other
  tubular structure. Each vertex has one parent, except a root. The type
  constant is `GEOM_SKELETON`, `"skeleton"`.

**Branch record**
: A link record `[child, parent]` for a parent edge that the read rule
  below does not give.

**SWC**
: The text format for neuronal morphology: `id, type, x, y, z, radius,
  parent_id` per vertex. Radius and type are ordinary vertex attributes;
  the format reserves no names or codes for them.

## Introduction

A skeleton uses `links_convention: implicit_sequential_with_branches`.
Within a fragment, a vertex's parent is the row before it, so most parent
edges are not stored. Each tree, or piece of one, is an object. What every
type shares is on [Geometry types](index.md#common-to-every-type).

Every reader applies this rule:

- A vertex that opens no fragment and is the child of no record has the
  row before it as parent. This is the only implicit edge.
- Every record `[child, parent]` is an edge, and its child takes no
  implicit parent.
- A record whose parent is outside the read (another object, or outside
  the `bbox` or `chunks` scope) gives no edge but still blocks the implicit
  one; its child is a root of the read.
- A vertex with several records keeps all of them.

Two writers produce skeleton stores. They differ in their link family:

| Writer | Family | Cross-chunk records |
|--------|--------|---------------------|
| `write_graph(kind="skeleton")`, whole dataset | `directed: false`, `canonical` | Sorted, with `perm_idx`; read back as `[child, parent]` |
| Streaming (`write_skeleton_chunk`, one chunk at a time) | `directed: true`, `canonical` | `[parent, child]`, no `perm_idx` |

Both store intra-chunk records as `[child, parent]`. See
[Links](../object_model/links.md#endpoint-order-of-skeleton-links).
SWC import and export are in
[zarr-vectors-tools](https://zarr-vectors-tools.readthedocs.io/en/latest/convert/skeletons.html).

## Technical reference

### Arrays present

At level 0 of a store written by `write_graph(kind="skeleton")`:

| Path | Present | Description |
|------|---------|-------------|
| `vertices/` | Always | One fragment per object per chunk, depth-first order |
| `links/0/0.0.0/` | Always, possibly empty | Branch records within one chunk, chunk-local indices |
| `links/0/<offsets>/` | When an edge crosses chunks | Every cross-chunk parent edge |
| `link_fragments/` | Always | Fragment index over the intra-chunk link rows |
| `object_index/` | Always | One manifest per object |
| `vertex_attributes/<name>/` | With `vertex_attributes=` | For example `radius` |
| `link_attributes/<name>/0/<offsets>/` | With `link_attributes=` | One row per stored record |
| `object_attributes/<name>/`, `fragment_attributes/<name>/` | With the matching argument | |

The streaming writer writes the same arrays, plus
`fragment_attributes/segment_id/` (one `uint64` source ID per fragment)
and, once the object index is written, `object_attributes/segment_id/`.

The root attributes (the streaming writer does not set
`cross_chunk_strategy`):

```javascript
{"geometry_types": ["skeleton"], "links_convention": "implicit_sequential_with_branches", ...}
```

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

summary = write_graph(
    "neuron.zarrvectors",
    positions,
    edges,
    chunk_shape=(200.0, 200.0, 200.0),
    kind="skeleton",
    vertex_attributes={"radius": rng.uniform(0.2, 3.0, n).astype(np.float32)},
)
print(summary)
# {'node_count': 200, 'edge_count': 199, 'chunk_count': 11, 'intra_edge_count': 29,
#  'cross_edge_count': 78, 'object_count': 1, 'kind': 'skeleton'}

result = read_graph("neuron.zarrvectors")
print(result["node_count"], result["edge_count"])   # 200 199
```

`write_graph(kind="skeleton")`:

- takes `[child, parent]` edge rows and vertices in any order. A row
  `[v, -1]` makes `v` a root and is not an edge; any other negative index
  raises `ValueError`. An object with more than one root raises
  `ArrayError`, but an object may hold pieces of several trees.
- reorders vertices depth-first from each root, with each fragment's
  parentless vertex first.
- stores a record for every edge the read rule does not give.
- keeps `link_attributes` rows only for stored records.

`read_graph` returns `edges` as `[child, parent]` rows; see
[Graph](graph.md#write-and-read-api).

### Streaming writer

`init_skeleton_store` creates the store, `write_skeleton_chunk` writes one
chunk's pieces, `write_skeleton_cross_chunk_links` writes the edges between
chunks, and `finalize_skeleton_store` closes the write. The caller writes
the object index.

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
# Segment 77: a branched piece in chunk (1, 0, 0), and a two-vertex piece in
# (0, 0, 0) whose first vertex hangs off vertex 2 of the first piece.
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
write_object_index(
    level0, {0: [(cc, f) for _, cc, f in records_a + records_b]}, sid_ndim=3, total_objects=1,
)
write_object_attributes(level0, "segment_id", np.array([77], dtype=np.uint64))
finalize_skeleton_store(root)

skeleton = read_skeleton_by_segment_id("stream.zarrvectors", 77)
print(skeleton["fragment_count"], skeleton["positions"].shape, skeleton["edges"].tolist())
# 3 (6, 3) [[1, 0], [2, 1], [5, 4], [3, 1]]
```

| Function | Behaviour |
|----------|-----------|
| `write_skeleton_chunk` | Splits each piece into linear paths, one fragment each, and stores a branch record `[child_local, parent_local]` in `links/0/0.0.0/` for each path that leaves another. Returns the `(segment_id, chunk, fragment_index)` records for the object index and the chunk-local positions of the requested `anchors` |
| `write_skeleton_cross_chunk_links` | Takes `((chunk, vertex), (chunk, vertex))` pairs, parent first, and stores them in that order under the parent's chunk |
| `read_skeleton_by_segment_id` | Returns `positions`, `edges` (`[child, parent]`), `attributes` and `fragment_count`, or `None` for an absent ID. It does not read cross-chunk records, so above there is no edge between vertex 2 and vertex 4 |

### Validation

The [common checks](index.md#validation) apply. The lexicographic-sign and
ordering checks apply to the `write_graph` family only, since the streaming
family is directed. L4 also accepts `links_convention` `explicit`. No level
checks that a skeleton is a tree, or attribute values such as radius.
