# Graph (`graph`)

## Terms

**Graph**
: A set of vertices (nodes) and edges joining pairs of them. It may be
  disconnected and may have cycles. The type constant is `GEOM_GRAPH`,
  `"graph"`.

**`kind`**
: The `write_graph` argument choosing the geometry type: `"graph"` (the
  default) or [`"skeleton"`](skeleton.md). It is not stored.

## Introduction

The `graph` type stores any vertex–edge graph, such as a vascular network
with loops. For trees, [`skeleton`](skeleton.md) stores fewer records.

Every edge is a link record (`links_convention: explicit`). An edge whose
ends share a chunk goes in `links/0/0.0.0/`; any other edge goes in the
array named by the offset between its chunks. A stored permutation lets a
reader return each edge's endpoints in written order.

Objects come from `object_ids=`, one ID per node; without it, every node
is in object 0. Each chunk has one fragment per object. What every type
shares is on [Geometry types](index.md#common-to-every-type).

## Technical reference

### Arrays present

At level 0 of a store written by `write_graph(kind="graph")`:

| Path | Present | Description |
|------|---------|-------------|
| `vertices/` | Always | One fragment per object per chunk |
| `links/0/0.0.0/` | Always, possibly empty | Intra-chunk edges, chunk-local indices |
| `links/0/<offsets>/` | When an edge crosses chunks | Cross-chunk edges |
| `link_fragments/` | Always | Fragment index over the intra-chunk link rows |
| `object_index/` | Always | One manifest per object |
| `vertex_attributes/<name>/` | With `vertex_attributes=` | |
| `link_attributes/<name>/0/<offsets>/` | With `link_attributes=` | One row per edge, mirroring `links/0/` |
| `object_attributes/<name>/`, `fragment_attributes/<name>/` | With the matching argument | |

The root attributes:

```javascript
{"geometry_types": ["graph"], "links_convention": "explicit", ...}
```

### Write and read API

```python
import numpy as np
from zarr_vectors.building import write_graph
from zarr_vectors.types.graphs import read_graph

rng = np.random.default_rng(0)
n_nodes = 500
positions = rng.uniform(0, 1000, (n_nodes, 3)).astype(np.float32)
edges = rng.integers(0, n_nodes, (1200, 2)).astype(np.int64)
edges = edges[edges[:, 0] != edges[:, 1]]
cell = (np.arange(n_nodes) % 4).astype(np.int64)        # owning object of each node

summary = write_graph(
    "network.zarrvectors",
    positions,
    edges,
    chunk_shape=(500.0, 500.0, 500.0),
    object_ids=cell,
    link_attributes={"weight": rng.uniform(0, 1, len(edges)).astype(np.float32)},
)
print(summary)
# {'node_count': 500, 'edge_count': 1198, 'chunk_count': 8, 'intra_edge_count': 152,
#  'cross_edge_count': 1046, 'object_count': 4, 'kind': 'graph'}

result = read_graph("network.zarrvectors", object_ids=[2])
print(result["node_count"], result["edge_count"])   # 125 71
```

`edges` is `(M, 2)` of node indices; any other shape raises `ArrayError`.
`read_graph` returns:

| Key | Type | Description |
|-----|------|-------------|
| `positions` | `ndarray (N, D)` | Node positions |
| `edges` | `ndarray (M, 2)` | Edges as indices into `positions`, endpoints in written order |
| `node_count`, `edge_count` | `int` | Sizes of the result |
| `object_ids` | `ndarray (N,)` | Owning object of each node; only when `object_ids=` is given |

A scoped read returns the induced subgraph: the nodes the scope selects
and the edges with both ends among them. So an edge between two objects
comes back only when both are asked for.

### Validation

Only the [common checks](index.md#validation) apply. No level checks
self-loops, duplicate edges or connectivity.
