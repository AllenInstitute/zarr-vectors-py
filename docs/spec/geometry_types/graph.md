# Graph (`graph`)

## Terms

**Graph**
: A collection of vertices (nodes) and edges connecting pairs of vertices.
  `write_graph` stores edges undirected (input order is preserved on
  read). The graph may be disconnected. Cycles are permitted for `graph`.

**`GEOM_GRAPH`**
: The geometry type constant `"graph"`, recorded as
  `"geometry_types": ["graph"]` in the root `zarr_vectors` block.

**`kind`**
: The `write_graph` argument selecting the store's geometry type:
  `"graph"` (default) or `"skeleton"`. It is not itself persisted —
  it determines `geometry_types` and `links_convention`.

**Link family policy**
: The `directed` / `store` / `link_width` / `sid_ndim` values recorded
  on the `links/<delta>/` **group**. For `graph` these are always
  `directed=false`, `store="canonical"`, `link_width=2`. See
  [Links](../object_model/links.md).

**Tree store**
: A store written with `kind="skeleton"`. It carries
  `geometry_types: ["skeleton"]` and
  `links_convention: "implicit_sequential_with_branches"`, letting it
  omit edges inferable from the parent-child sequence. See
  [Skeleton](skeleton.md).

---

## Introduction

The `graph` type stores an arbitrary vertex–edge graph, spatially chunked
like all other Zarr Vectors types. It is appropriate for connectivity data that does
not fit the stricter topology of `skeleton` (which requires a tree): vascular
networks with anastomoses, synaptic connectivity matrices embedded in 3-D
space, or any general graph with cycles.

`graph` and `skeleton` share the same underlying array schema and the same
writer (`write_graph(kind=...)`). They differ in `geometry_types` and
`links_convention`: `skeleton` assumes one rooted tree per object and
stores only the edges that depth-first order does not imply; `graph`
stores every edge explicitly and is unconstrained.

---

## Technical reference

### Arrays present

| Array path | Required | Description |
|-----------|----------|-------------|
| `vertices/` | Yes | Node positions |
| `vertex_fragments/` | Yes | Fragment index over `vertices/` rows |
| `links/0/0.0.0/` | Yes* | Edges whose endpoints share a chunk |
| `links/0/<offsets>/` | Yes* | Edges whose endpoints straddle chunks; offsets name the far chunk |
| `link_fragments/` | Yes (with `links/0/0.0.0/`) | Fragment index over the intra array's rows |
| `object_index/` | Yes | Per-object manifest blobs naming fragments |
| `vertex_attributes/<name>/` | No | Per-vertex attributes |
| `link_attributes/<name>/0/<offsets>/` | No | Per-edge attributes (`link_attributes=`), mirroring each offsets array cell-for-cell |
| `object_attributes/<name>/` | No | Per-object attributes |

*Which offsets arrays exist depends on the data: an edge lands in the
array named by the offsets between its endpoints, and the all-zero
array is simply where intra-chunk edges land. There is no separate
`cross_chunk_links/` family. Unlike `polyline`, `graph` uses
`links_convention: explicit`, so intra-chunk edges **are** materialised
— the all-zero array is normally populated.

`write_graph` always creates `links/0/` even when every edge is
implied, so the family is advertised in `arrays_present`.

### Root metadata

```json
{
  "zarr_vectors": {
    "geometry_types":   ["graph"],
    "links_convention": "explicit"
  }
}
```

(Alongside the usual `zarr_vectors` keys; see
[Root metadata](../layout/root_metadata.md).)

`graph` declares no type-specific root keys of its own. Tree-ness is
carried by `geometry_types` and `links_convention`, not by a flag:

| `write_graph(kind=…)` | `geometry_types` | `links_convention` |
|-----------------------|------------------|--------------------|
| `"graph"` (default) | `["graph"]` | `explicit` |
| `"skeleton"` | `["skeleton"]` | `implicit_sequential_with_branches` |

> **`is_directed` and `is_tree` are not root metadata keys.** Neither
> appears anywhere in the shipped code. `is_tree` survives only as a
> **deprecated** `write_graph` argument that maps to `kind=` and emits a
> `DeprecationWarning` (`is_tree=True` → `kind="skeleton"`); passing
> both raises `TypeError`. It is not persisted under that name. There is
> no `is_directed` argument, key, or validation check at any level.

### Edge encoding

`graph` writes one link family: `link_width = 2`, **undirected**,
`store="canonical"`. Every edge — intra-chunk or not — goes through a
single `write_links` call, which routes it to the offsets array naming
where its far endpoint sits relative to its source chunk.

Because the family is undirected and canonical, each edge is stored
**exactly once**. `write_links` canonical-sorts the endpoints by
`(chunk_coords, vertex_index)`, so:

- the source is the lex-smallest endpoint, and the stored offset is
  therefore lexicographically non-negative — `[i,j]` and `[j,i]` cannot
  both appear;
- `perm_idx` records the permutation applied, so a reader recovers the
  original input order.

An edge whose endpoints share a chunk lands in the all-zero-offsets
array (`links/0/0.0.0/`) with both indices local to that chunk. An edge
straddling chunks lands in the array named by the offsets between them,
with `vi_k` local to chunk `src + o_k`. See
[Links](../object_model/links.md).

> **Directed graphs are not supported by `write_graph`.** The writer
> takes no `is_directed` argument and always writes the family
> undirected, **including for `kind="skeleton"`**. The underlying
> `write_links` does accept `directed=True` (which suppresses the
> canonical sort so `A→B` and `B→A` file under opposite offsets); only
> the streaming skeleton writers use it — see [Skeleton](skeleton.md).
> The input order of every edge is still recovered on read through
> `perm_idx`.

### Object model for graphs

Objects are whatever you say they are: `write_graph(object_ids=...)` takes
an `(N,)` array assigning each **node** to an object ID. When omitted,
every node belongs to object 0 and the whole graph is a single object —
the writer does **not** discover connected components. A node's
object is recorded in `object_index/`, and per-object data goes in
`object_attributes/`.

### Write API

```python
import numpy as np
from zarr_vectors.types.graphs import write_graph

rng = np.random.default_rng(0)
n_nodes = 5000
positions = rng.uniform(0, 1000, (n_nodes, 3)).astype(np.float32)

# Random sparse graph: ~3 edges per node
src = rng.integers(0, n_nodes, 7500)
dst = rng.integers(0, n_nodes, 7500)
edges = np.column_stack([src, dst]).astype(np.int32)

summary = write_graph(
    "network.zarrvectors",
    positions=positions,
    edges=edges,
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
    kind="graph",          # default; "skeleton" for trees
)
# summary: {"node_count", "edge_count", "chunk_count", "intra_edge_count",
#           "cross_edge_count", "object_count", "kind"}
```

`chunk_shape` is required. Other keywords: `bounds`, `vertex_attributes`
(`{name: (N,)}` / `(N, C)`), `link_attributes` (`{name: (M,)}` / `(M, C)`),
`object_attributes`, `fragment_attributes`, `object_ids`, `dtype`,
`backend`, `chunk_by_attribute`, `out_of_bounds` (`"ignore"` is rejected,
since edges would dangle), `compressor`, `shard_shape`. The deprecated
`is_tree`, `node_attributes` and `edge_attributes` still work but warn.
Through the dataset API: `ds.add_graph(positions, edges, tree=False, ...)`.

### Write API — tree mode

```python
write_graph(
    "tree.zarrvectors",
    positions=positions,
    edges=edges,         # (n-1, 2) parent→child pairs
    chunk_shape=(200., 200., 200.),
    kind="skeleton",     # reorders depth-first; stores as geometry type "skeleton"
)
```

`kind="skeleton"` changes three things:

- nodes are **reordered depth-first from the root** (`_reorder_tree`),
  and `object_ids` / attributes are permuted to match;
- `geometry_types` is written as `["skeleton"]`, not `["graph"]`;
- `links_convention` is written as `implicit_sequential_with_branches`,
  so edges inferable from the depth-first sequence may be omitted.

> **`kind="skeleton"` checks one root per object, and nothing else.**
> `edges` are `[child, parent]` pairs. `write_graph` raises `ArrayError`
> if any single object has more than one root (a forest inside one object
> ID — give each tree its own `object_ids` entry, or write it with
> `kind="graph"`). It does not check connectivity or acyclicity, so a
> cyclic input is not rejected, and no validation level catches it.

`kind` must be `"graph"` or `"skeleton"`; anything else raises
`ValueError`.

### Read API

```python
import numpy as np
from zarr_vectors.types.graphs import read_graph

result = read_graph("network.zarrvectors")
print(result["node_count"])           # int
print(result["edge_count"])           # int
print(result["positions"].shape)      # (N, D)
print(result["edges"].shape)          # (E, 2), indices into positions

# Spatial bbox
result = read_graph(
    "network.zarrvectors",
    bbox=(np.array([0., 0., 0.]), np.array([200., 200., 200.])),
)
# Returns the nodes inside the bbox and the edges whose endpoints are
# BOTH inside it; edges crossing the bbox boundary are dropped.
```

`read_graph` returns exactly `positions`, `edges`, `node_count` and
`edge_count`: it does not return vertex, edge or object attributes. Other
keywords: `level`, `chunks` (whitelist of chunk coordinates; edges
spanning a listed and an unlisted chunk are dropped), `attribute_filter`,
`backend`. **`object_ids=` raises `NotImplementedError`** (the filter is
not implemented, and the reader refuses rather than silently returning
the whole level).

### Multi-graph stores

A single `graph` store may hold many disconnected graphs; give each its
own entry in `object_ids` when writing, so that they are recorded as
separate objects in `object_index/`. There is currently no
object-scoped read for graphs: read the level and filter the arrays.

### Validation

L1: `vertices/` exists at every level. `links/` and `object_index/` are
recorded when present but are **not** required at L1.

L3:
- Every `links/0/` offsets segment parses under the family's `sid_ndim`
  and `link_width`.
- The family being undirected, canonical, and intra-level, each offset
  is lexicographically non-negative and offsets are non-decreasing —
  this is what enforces "each edge stored once". The all-zero (intra)
  segment is legal and exempt.
- `num_physical_records`, if recorded, matches the rows on disk.
- Every record's endpoint chunks exist at the level.

L4: `links_convention` MUST be `explicit` for `graph`
([`GEOMETRY_LINK_REQ`](../../../zarr_vectors/validate/conformance.py)).

**Not checked at any level:** edge vertex indices within a chunk,
self-loops, duplicate edges *within* one offsets array, connectivity,
acyclicity, root count, or the edge-count identity. The offset-sign
rule above catches an `[i,j]`/`[j,i]` pair only when the two land in
*different* offsets arrays. See
[Validation overview](../validation/overview.md).
