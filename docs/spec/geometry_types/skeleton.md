# Skeleton (`skeleton`)

## Terms

**Skeleton**
: A tree-structured graph representing the branching morphology of a
  biological structure — typically a neuron, blood vessel, or other
  tubular object. Skeletons are acyclic and connected; each vertex has
  exactly one parent except the root, which has none.

**`GEOM_SKELETON`**
: The geometry type constant `"skeleton"`.

**SWC convention**
: The de facto standard file format for neuronal morphology, defined by
  Cannon et al. (1998). An SWC file stores one vertex per row with columns:
  `id, type, x, y, z, radius, parent_id`. ZVF skeletons follow the SWC
  vertex type taxonomy and store `radius` and `type` as per-vertex
  attributes.

**Vertex type**
: An integer label from the SWC taxonomy identifying the anatomical role
  of a vertex: 1 = undefined, 2 = soma, 3 = axon, 4 = basal dendrite,
  5 = apical dendrite, 6 = custom, 7 = custom. Stored as per-vertex
  attribute `"swc_type"` (int32).

**Radius**
: The estimated radius of the tubular structure at a given vertex.
  Stored as per-vertex attribute `"radius"` (float32). Unit follows
  `axis_units` from root `.zattrs`.

**Root vertex**
: The vertex with no parent, representing the origin of the tree (typically
  the soma centre for a neuron). Identified by a parent edge index of `-1`.

---

## Introduction

The `skeleton` type is a specialisation of `graph` for rooted trees,
aligned to the SWC morphology convention. It is the appropriate type for
neuronal morphologies, vascular trees, and any other branching tree-
structured shape that will be read by or compared to SWC-compatible tools.

ZVF skeletons store all the information present in an SWC file (position,
radius, type, parent relationship) plus the spatial indexing and multi-
resolution features of ZVF. The `ingest_swc` and `export_swc` functions
provide lossless round-tripping between SWC and ZVF for standard SWC files.

---

## Technical reference

### Arrays present

Identical to `graph`, except that the link family is written
`directed=True`.

| Array path | Required | Description |
|-----------|----------|-------------|
| `vertices/` | Yes | Node positions (x, y, z) |
| `vertex_fragments/` | Yes | Fragment index over `vertices/` rows |
| `links/0/0.0.0/` | Yes | Intra-chunk branch links; one group per fragment |
| `links/0/<offsets>/` | Yes* | Parent–child edges crossing a chunk boundary |
| `link_fragments/` | Yes (with `links/0/0.0.0/`) | Fragment index over the intra array's rows |
| `object_index/` | Yes | One manifest blob per skeleton (one component = one object) |
| `attributes/swc_type/` | Recommended | int32 per vertex: SWC compartment type |
| `attributes/radius/` | Recommended | float32 per vertex: estimated radius |
| `attributes/swc_id/` | Optional | int64 per vertex: original SWC row ID (for round-trips) |
| `object_attributes/<name>/` | No | Per-skeleton attributes |

### Root `.zattrs` type-specific keys

```json
{
  "geometry_type":    "skeleton",
  "links_convention": "implicit_sequential_with_branches"
}
```

Skeleton-ness is carried by `geometry_type` and `links_convention`.

> **`is_tree` and `swc_compatible` are not root metadata keys.** Neither
> is written or read by any shipped code. `is_tree` survives only as a
> deprecated `write_graph` argument (see [Graph](graph.md)); there is no
> `swc_compatible` key, argument, or check anywhere. SWC round-tripping
> is a property of which `attributes/` a store happens to carry, not of
> a declared flag.

### The link family for skeletons

A skeleton store has **one** `links/0/` family, written with
`link_width=2` and — uniquely among the core geometry types —
**`directed=True`**.

`directed=True` is family policy set at creation
(`create_links_array(..., directed=True)`). It suppresses the canonical
sort, which matters because parent→child order is *data*: sorting
endpoints by `(chunk_coords, vertex_index)` would silently swap parent
and child whenever the child's chunk sorts before the parent's.

The consequence for the layout: a directed family stores `A→B` and
`B→A` under **opposite offsets** (`0.0.+1` vs `0.0.-1`) in different
cells, rather than deduplicating them into one lexicographically
positive offset. Offsets in a directed family are therefore **not**
constrained to be lex-non-negative, and [L3](../validation/l3_consistency.md)
skips that check for them (`enforce_canonical` requires `not directed`).
`perm_idx` is `0` throughout, since no permutation is applied.

Two writers populate this one family:

| Writer | Lands in | Endpoint order |
|--------|----------|----------------|
| `write_skeleton_chunk` | all-zero offsets (`0.0.0`) | `[child_local, parent_local]` |
| `write_skeleton_cross_chunk_links` | non-zero offsets | endpoint 0 = **parent**, endpoint 1 = child |

Intra-chunk branch links are written per-cell (`write_chunk_links`),
not through `write_links`, because they are already chunk-local and
all-zero-offset and must stay **one group per fragment** for
`read_chunk_link_fragment` to slice them back out — `write_links` files
a cell's records as a single group. Intra-chunk records are stored in
input order regardless of the `directed` policy, so branch links are
unaffected by it.

> **Known inconsistency.** The two writers use *opposite* endpoint
> conventions within the same family: the intra array is
> `[child, parent]` while the cross-chunk records lead with the parent.
> Because `directed=True` preserves input order verbatim, nothing on
> disk distinguishes them — a reader must know which writer produced a
> record. Treat the endpoint order of a skeleton link as
> writer-dependent until this is reconciled in the code; do not rely on
> a single rule across both arrays.

For cross-chunk records the source is the **parent's** chunk, so the
offsets name where the child's chunk sits relative to the parent's.

This is a key distinction from `polyline` links, where direction encodes
traversal order and the family is undirected. For skeletons, direction
encodes the tree's child↔parent relationship and is preserved
explicitly.

### SWC type taxonomy

| Code | Anatomical meaning |
|------|--------------------|
| 0 | Undefined |
| 1 | Soma |
| 2 | Axon |
| 3 | Basal dendrite |
| 4 | Apical dendrite |
| 5 | Custom (fork point) |
| 6 | Custom (end point) |
| 7 | Custom (unspecified) |

Note: some SWC conventions use `1` for soma and `2` for axon. The mapping
above follows the NeuroMorpho / SWC+ convention. The `ingest_swc` function
detects the convention from the input file header.

### Ingest and export

SWC converters (single-file and directory ingest, SWC export) live in
the companion package **`zarr-vectors-tools`**.

### Write API

```python
import numpy as np
from zarr_vectors.types.graphs import write_graph

# positions: (n_nodes, 3), edges: (n_nodes-1, 2) [child, parent]; root has parent=-1
write_graph(
    "neuron.zarrvectors",
    positions=node_positions,
    edges=parent_child_pairs,
    chunk_shape=(200., 200., 200.),
    kind="skeleton",
    vertex_attributes={
        "radius":   radii,       # float32
        "swc_type": swc_types,   # int32
    },
)
```

### Multi-skeleton stores

For connectome-scale datasets with thousands of neurons in a shared
coordinate space, a single `skeleton` store is more efficient than
separate per-neuron files:

```python
from zarr_vectors.types.graphs import read_graph

# Read one skeleton by ID
result = read_graph("connectome.zarrvectors", object_ids=[1042])
print(result["node_count"])    # nodes in skeleton 1042
print(result["attributes"]["radius"])

# Spatial query — returns all skeletons with nodes in the region
result = read_graph(
    "connectome.zarrvectors",
    bbox=(np.array([1000., 2000., 500.]),
          np.array([1200., 2200., 700.])),
)
```

### Validation

L1, L2: same as `graph`.

L3: offsets segments parse; `num_physical_records` matches the rows on
disk; every record's endpoint chunks exist at the level. **The
lex-non-negative and non-decreasing offset checks do not apply** —
`enforce_canonical` requires `not directed`, and skeleton families are
`directed=True`. This is correct: a directed family legitimately carries
`0.0.-1` alongside `0.0.+1`.

L4: `links_convention` MUST be `implicit_sequential_with_branches` or
`explicit` for `skeleton`
([`GEOMETRY_LINK_REQ`](../../../zarr_vectors/validate/conformance.py)).

**Not checked at any level:** that each skeleton is a valid tree
(connected, acyclic, single root), that link endpoints reference valid
vertex indices within their chunk, that `[child, parent]` direction is
consistent with tree depth, or anything about `swc_type` / `radius`
value ranges. The SWC taxonomy below is a convention for producers, not
an enforced constraint. See
[Validation overview](../validation/overview.md).
