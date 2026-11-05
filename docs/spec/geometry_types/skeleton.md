# Skeleton (`skeleton`)

## Terms

**Skeleton**
: A tree-structured graph representing the branching morphology of a
  biological structure — typically a neuron, blood vessel, or other
  tubular object. Skeletons are acyclic and connected; each vertex has
  exactly one parent except the root, which has none.

**`GEOM_SKELETON`**
: The geometry type constant `"skeleton"`, recorded as
  `"geometry_types": ["skeleton"]` in the root `zarr_vectors` block.

**SWC convention**
: The de facto standard file format for neuronal morphology, defined by
  Cannon et al. (1998). An SWC file stores one vertex per row with columns:
  `id, type, x, y, z, radius, parent_id`. Zarr Vectors does not require SWC
  semantics: `radius` and `type` are ordinary per-vertex attributes whose
  names and values you choose (this page uses `radius` and `swc_type`).

**Vertex type**
: An integer label from the SWC taxonomy identifying the anatomical role
  of a vertex (table below). A producer convention only; nothing in
  `zarr-vectors-py` reads or enforces it.

**Radius**
: The estimated radius of the tubular structure at a given vertex, as a
  per-vertex attribute. Its unit is the unit of the store's axes
  (`multiscales[0].axes`).

**Root vertex**
: The vertex with no parent, representing the origin of the tree (typically
  the soma centre for a neuron). With `write_graph` the root simply has no
  `[child, parent]` edge; the streaming writer records an explicit
  `[root_local, -1]` row per fragment.

---

## Introduction

The `skeleton` type is a specialisation of `graph` for rooted trees,
aligned to the SWC morphology convention. It is the appropriate type for
neuronal morphologies, vascular trees, and any other branching tree-
structured shape that will be read by or compared to SWC-compatible tools.

Zarr Vectors skeletons store all the information present in an SWC file (position,
radius, type, parent relationship) plus the spatial indexing and multi-
resolution features of Zarr Vectors. SWC converters live in the
companion package `zarr-vectors-tools`.

---

## Technical reference

### Arrays present

Identical to `graph`. Two writers exist, with one difference in the link
family (see below).

| Array path | Required | Description |
|-----------|----------|-------------|
| `vertices/` | Yes | Node positions |
| `vertex_fragments/` | Yes | Fragment index over `vertices/` rows |
| `links/0/<offsets>/` | Yes | The edges the reader cannot imply: branch points and fragment starts (intra-chunk, `0.0.0`), and **every** edge crossing a chunk boundary (non-zero offsets) |
| `link_fragments/` | With `links/0/0.0.0/` | Fragment index over the intra array's rows |
| `object_index/` | Yes | One manifest per object (`object_ids=` entry; default: the whole store is object 0) |
| `vertex_attributes/<name>/` | No | e.g. `radius` (float32), `swc_type` (int32); names are yours to choose |
| `object_attributes/<name>/` | No | Per-skeleton attributes |

### Root metadata

```json
{
  "zarr_vectors": {
    "geometry_types":   ["skeleton"],
    "links_convention": "implicit_sequential_with_branches"
  }
}
```

(Alongside the usual `zarr_vectors` keys.) Skeleton-ness is carried by
`geometry_types` and `links_convention`. L4 accepts either
`implicit_sequential_with_branches` or `explicit` for a skeleton.

> **`is_tree` and `swc_compatible` are not root metadata keys.** Neither
> is written or read by any shipped code. `is_tree` survives only as a
> deprecated `write_graph` argument (see [Graph](graph.md)); there is no
> `swc_compatible` key, argument, or check anywhere. SWC round-tripping
> is a property of which `vertex_attributes/` a store happens to carry,
> not of a declared flag.

### Implicit edges

With `implicit_sequential_with_branches`, a reader infers
`parent[i] = i - 1` over the concatenated read order (chunks in order,
fragments within a chunk), except at the start of a fragment, which has
no implied parent. Edges are `[child, parent]`. `write_graph(kind="skeleton")`
reorders nodes depth-first and stores in `links/` only the edges this rule
does not already give: branch points, fragment starts, and every edge whose
endpoints are in different chunks. `read_graph` rebuilds the full edge list.

### The link family for skeletons

Skeleton stores have one `links/0/` family (`link_width=2`). Whether it is
**directed** depends on the writer:

| Writer | Family `directed` | Endpoint order |
|--------|-------------------|----------------|
| `write_graph(kind="skeleton")` | `false` (canonical) | `[child, parent]`; input order recovered via `perm_idx` |
| Streaming: `init_skeleton_store`, `write_skeleton_chunk`, `write_skeleton_cross_chunk_links` | `true` | intra array `[child_local, parent_local]`; cross-chunk records lead with the **parent** |

The streaming writers (module `zarr_vectors.types.skeletons`) are for
scale-out ingestion where many workers each write some chunks. They set
`directed=True` because parent→child order is data: it suppresses the
canonical sort, which would otherwise swap parent and child whenever the
child's chunk sorts first. A directed family stores `A→B` and `B→A` under
**opposite offsets** (`0.0.+1` vs `0.0.-1`), so offsets are not constrained
to be lex-non-negative, and [L3](../validation/l3_consistency.md) skips the
canonical-offset check for them (`enforce_canonical` requires
`not directed`). `perm_idx` is `0` throughout a directed family.

Intra-chunk branch links from `write_skeleton_chunk` are written per cell
(`write_chunk_links`) as **one group per fragment** so
`read_chunk_link_fragment` can slice them back out, with an explicit
`[root_local, -1]` row per fragment.

> **Known inconsistency.** Within the streaming writers' directed family,
> the intra array is `[child, parent]` while the cross-chunk records lead
> with the parent. Because `directed=True` preserves input order verbatim,
> nothing on disk distinguishes them; a reader must know which writer
> produced a record. Treat the endpoint order of a streamed skeleton link
> as writer-dependent.

For cross-chunk records of the streaming writer the source is the
**parent's** chunk, so the offsets name where the child's chunk sits
relative to the parent's.

Streaming-writer stores also carry a `segment_id` fragment attribute and
a `coordinate_offset` key in the root `zarr_vectors` block (stored vertices
are `world - coordinate_offset`). `read_skeleton_by_segment_id(path, id)`
reads one skeleton back from such a store and returns `positions`,
`edges` (`[child, parent]`), `attributes` and `fragment_count`, or `None`
for an unknown id.

### SWC type taxonomy

| Code | Anatomical meaning |
|------|--------------------|
| 0 | Undefined |
| 1 | Soma |
| 2 | Axon |
| 3 | Basal dendrite |
| 4 | Apical dendrite |
| 5+ | Custom |

This is the standard SWC convention, given for producers. It is not
stored in or enforced by the format.

### Ingest and export

SWC converters (single-file and directory ingest, SWC export) live in
the companion package **`zarr-vectors-tools`**.

### Write API

```python
import numpy as np
from zarr_vectors.types.graphs import write_graph

# A 5-node tree: 0 is the root; edges are [child, parent] and the root
# has no edge.
node_positions = np.array(
    [[0, 0, 0], [10, 0, 0], [20, 0, 0], [20, 10, 0], [30, 0, 0]],
    dtype=np.float32,
)
parent_child_pairs = np.array(
    [[1, 0], [2, 1], [3, 2], [4, 2]], dtype=np.int64
)

write_graph(
    "neuron.zarrvectors",
    positions=node_positions,
    edges=parent_child_pairs,
    chunk_shape=(200., 200., 200.),
    kind="skeleton",
    vertex_attributes={
        "radius":   np.full(5, 1.5, dtype=np.float32),
        "swc_type": np.full(5, 3, dtype=np.int32),
    },
)
```

Through the dataset API: `ds.add_graph(positions, edges, tree=True, ...)`.

### Multi-skeleton stores

For many neurons in a shared coordinate space, pass `object_ids` (one entry
per node) so each tree is its own object; `write_graph` rejects an object
with more than one root. `read_graph(object_ids=...)` is **not
implemented** (it raises `NotImplementedError`), so there is no
by-object read for `write_graph` stores; read a region with `bbox=` or
`chunks=` instead:

```python
import numpy as np
from zarr_vectors.types.graphs import read_graph

result = read_graph(
    "connectome.zarrvectors",
    bbox=(np.array([1000., 2000., 500.]),
          np.array([1200., 2200., 700.])),
)
print(result["node_count"], result["edge_count"])
```

`read_graph` returns no attributes. For stores written with the streaming
writers, `read_skeleton_by_segment_id` pulls one skeleton with its
attributes.

### Validation

L1, L2: same as `graph`.

L3: offsets segments parse; `num_physical_records` matches the rows on
disk; every record's endpoint chunks exist at the level. For a **directed**
(streaming-writer) family the lex-non-negative and non-decreasing offset
checks do not apply, since `enforce_canonical` requires `not directed`; a
`write_graph` skeleton family is undirected and canonical, so they do.

L4: `links_convention` MUST be `implicit_sequential_with_branches` or
`explicit` for `skeleton`
([`GEOMETRY_LINK_REQ`](../../../zarr_vectors/validate/conformance.py)).

`write_graph(kind="skeleton")` itself rejects an object with more than one
root. **Not checked at any validation level:** that each skeleton is a valid
tree (connected, acyclic, single root), that link endpoints reference valid
vertex indices within their chunk, that `[child, parent]` direction is
consistent with tree depth, or anything about `swc_type` / `radius` value
ranges. See [Validation overview](../validation/overview.md).
