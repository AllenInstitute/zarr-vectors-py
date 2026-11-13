# Graph (`graph`)

## Terms

**Graph**
: A set of vertices (nodes) and edges joining pairs of them. The graph may
  be disconnected and may contain cycles.

**`GEOM_GRAPH`**
: The geometry type constant `"graph"`, stored in the `geometry_types` list
  of the root attributes.

**`kind`**
: The `write_graph` argument choosing the geometry type: `"graph"` (the
  default) or `"skeleton"`. It is not stored; it decides `geometry_types`
  and `links_convention`. Skeletons are described in [Skeleton](skeleton.md).

## Introduction

The `graph` type stores an arbitrary vertex–edge graph: vascular networks
with anastomoses, connectivity embedded in 3-D space, any graph with cycles.
For trees, [`skeleton`](skeleton.md) stores fewer records.

A graph uses `links_convention: explicit`: every edge is a link record. An
edge whose ends share a chunk goes in the all-zero offsets array
`links/0/0.0.0/`; an edge whose ends are in different chunks goes in the
array named by the offset between them. The family is undirected with
`store: "canonical"`, so each edge is stored once, and a stored permutation
lets a reader return each edge's endpoints in the order they were written.
See [Links](../object_model/links.md).

Objects come from the `object_ids=` argument, one id per node. Without it,
every node belongs to object 0. Within each chunk, the writer stores one
fragment per object.

## Technical reference

### Arrays present

At level 0 of a store written by `write_graph(kind="graph")`:

| Path | Present | Description |
|------|---------|-------------|
| `vertices/` | Always | Node positions, one fragment per object per chunk |
| `vertex_fragments/` | Always | Fragment index over `vertices/` rows |
| `links/0/0.0.0/` | Always, possibly empty | Intra-chunk edges, chunk-local indices |
| `links/0/<offsets>/` | When an edge crosses chunks | Cross-chunk edges |
| `link_fragments/` | Always | Fragment index over the link rows |
| `object_index/` | Always | One manifest per object |
| `vertex_attributes/<name>/` | With `vertex_attributes=` | One row per node |
| `link_attributes/<name>/0/<offsets>/` | With `link_attributes=` | One row per edge, mirroring `links/0/` array for array |
| `object_attributes/<name>/` | With `object_attributes=` | One row per object |
| `fragment_attributes/<name>/` | With `fragment_attributes=` | One row per fragment |

### Root attributes

A graph store has no type-specific root keys. The writer sets these values
in the `zarr_vectors` block (see [Root metadata](../layout/root_metadata.md)):

```json
{
  "geometry_types": ["graph"],
  "links_convention": "explicit",
  "object_index_convention": "standard",
  "cross_chunk_strategy": "explicit_links"
}
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
    vertex_attributes={"degree": np.bincount(edges.ravel(), minlength=n_nodes).astype(np.float32)},
    link_attributes={"weight": rng.uniform(0, 1, len(edges)).astype(np.float32)},
)
print(summary)
# {'node_count': 500, 'edge_count': 1198, 'chunk_count': 8, 'intra_edge_count': 152,
#  'cross_edge_count': 1046, 'object_count': 4, 'kind': 'graph'}

result = read_graph("network.zarrvectors")
print(result["node_count"], result["edge_count"])   # 500 1198

result = read_graph("network.zarrvectors", object_ids=[2])
print(result["node_count"], result["edge_count"], np.unique(result["object_ids"]))   # 125 71 [2]

result = read_graph(
    "network.zarrvectors",
    bbox=(np.array([0.0, 0.0, 0.0]), np.array([500.0, 500.0, 500.0])),
)
print(result["node_count"], result["edge_count"])   # 49 12
```

`edges` is `(M, 2)` of node indices; any other shape raises `ArrayError`.
`read_graph` takes `level=`, `object_ids=`, `bbox=`, `chunks=` and
`attribute_filter=`, and returns:

| Key | Type | Description |
|-----|------|-------------|
| `positions` | `ndarray (N, D)` | Node positions |
| `edges` | `ndarray (M, 2)` | Edges as indices into `positions`, endpoints in written order |
| `node_count`, `edge_count` | `int` | Sizes of the result |
| `object_ids` | `ndarray (N,)` | Owning object of each node; only when `object_ids=` is given |

Every scoped read returns the induced subgraph: the nodes the scope selects
(by object, by chunk, or inside the box) and the edges whose two ends are
both among them. An edge between two objects comes back only when both are
asked for.

### Validation

- L1: `vertices/` missing at a level is an error. `links/` and
  `object_index/` are not required.
- L3: each `links/0/` offsets name parses under the family's `sid_ndim` and
  `link_width`; no offset is lexicographically negative and the offsets are
  non-decreasing; `num_physical_records` matches the rows on disk; each
  record's chunks exist at the level.
- L4: `links_convention` other than `explicit` is an error.

No level checks the vertex index in a link record, self-loops, duplicate
edges or connectivity. See [Validation overview](../validation/overview.md).
