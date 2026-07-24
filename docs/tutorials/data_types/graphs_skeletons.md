# Graphs and skeletons

ZVF provides two graph-structured geometry types. Use `skeleton` for
neuronal morphologies, vascular trees, and any other branching structure
that must align to the SWC convention. Use `graph` for arbitrary
connectivity — vascular networks with anastomoses, synaptic connectivity
graphs embedded in 3-D space, or any structure where cycles are valid.

Both types use the same on-disk array schema; the distinction is the
`kind` argument to `write_graph` and the additional SWC-compatible
attributes that `skeleton` stores.

All examples on this page use only the core `zarr-vectors` API.
SWC/GraphML converters live in the companion package
**`zarr-vectors-tools`**.

---

## Skeletons (SWC-aligned)

### Write a skeleton programmatically

```python
import numpy as np
from zarr_vectors.types.graphs import write_graph

rng = np.random.default_rng(0)
n_nodes = 800

# Simulate a branching skeleton: soma at origin, random walk branches
positions = np.zeros((n_nodes, 3), dtype=np.float32)
positions[1:] = rng.normal(0, 5, (n_nodes - 1, 3)).cumsum(0)

# Parent array: node 0 is root (parent = -1), others have sequential parents
# In a real skeleton, parent relationships encode the tree topology
parents = np.arange(-1, n_nodes - 1, dtype=np.int32)
edges   = np.column_stack([
    np.arange(1, n_nodes, dtype=np.int32),   # child indices
    parents[1:],                              # parent indices
])

radii    = rng.uniform(0.2, 2.0, n_nodes).astype(np.float32)
swc_type = np.ones(n_nodes, dtype=np.int32)
swc_type[0] = 1   # soma

write_graph(
    "neuron.zarrvectors",
    positions=positions,
    edges=edges,            # [child, parent] pairs; root has parent = -1
    chunk_shape=(200.0, 200.0, 200.0),
    kind="skeleton",        # "skeleton" for trees, "graph" (default) otherwise
    vertex_attributes={
        "radius":   radii,
        "swc_type": swc_type,
    },
)
```

`kind="skeleton"` reorders nodes depth-first from the root for storage.
It does **not** validate that your data is actually a tree — see
[Common pitfalls](#common-pitfalls).

Three older keyword names are still accepted but deprecated, and each
emits a `DeprecationWarning`:

| Deprecated | Use instead |
|------------|-------------|
| `is_tree=True` / `is_tree=False` | `kind="skeleton"` / `kind="graph"` |
| `node_attributes=` | `vertex_attributes=` |
| `edge_attributes=` | `link_attributes=` |

Passing both names of a pair raises `TypeError`.

### Spatial query on a skeleton

```python
from zarr_vectors.types.graphs import read_graph

# All nodes within a 100³ µm region
result = read_graph(
    "neuron.zarrvectors",
    bbox=(np.array([0., 0., 0.]), np.array([100., 100., 100.])),
)
print(result["node_count"])
print(result["edge_count"])   # edges where both endpoints are in bbox
```

An edge is returned only when **both** endpoints survive the filter, so
an edge leaving the bbox is dropped. There is no option to include
half-outside edges — widen the `bbox` if you need them.

### What `read_graph` returns

`read_graph` returns exactly four keys:

```python
result = read_graph("neuron.zarrvectors")
print(sorted(result))
# ['edge_count', 'edges', 'node_count', 'positions']
```

| Key | Meaning |
|-----|---------|
| `positions` | `(N, D)` node positions |
| `edges` | `(M, 2)` edge list, remapped to output indices |
| `node_count` | `int` |
| `edge_count` | `int` |

It does **not** return per-vertex attributes and does not return object
IDs. To read attributes, use the lazy API, which exposes them per level:

```python
from zarr_vectors.lazy import open_zv

level = open_zv("neuron.zarrvectors")[0]

if "radius" in level.attributes:
    radii = level.attributes["radius"].compute()    # (N,) array
    types = level.attributes["swc_type"].compute()
```

`level.attributes` is a dict-like proxy supporting `acc[name]` and
`name in acc` only — it is not iterable.

Note that the lazy attribute arrays are in **stored** order for the whole
level, which for `kind="skeleton"` is the depth-first reordering applied
at write time. They do not line up row-for-row with the output of a
*filtered* `read_graph` call.

### Filtering by attribute

`read_graph(attribute_filter=...)` only works on a store that was written
with `chunk_by_attribute`. On any other store it raises:

```
ArrayError: attribute_filter requires a store written with chunk_by_attribute
```

---

## Multi-skeleton stores

For connectome-scale datasets with thousands of neurons, a single ZVF
store is far more efficient than per-neuron files. SWC-directory ingest
and SWC ID-mapping helpers live in **`zarr-vectors-tools`**.

### Read a specific neuron

> **Known limitation.** `read_graph` accepts an `object_ids=` argument,
> but in the current version it has **no effect** — the parameter is
> never applied, and you get the whole level back regardless. Do not rely
> on it to isolate one neuron. Use `chunks=` or `bbox=` to restrict a
> read spatially, which does work.

```python
from zarr_vectors.types.graphs import read_graph

# Restrict spatially — this filter is applied.
result = read_graph(
    "connectome.zarrvectors",
    bbox=(np.array([2000., 3000., 1500.]),
          np.array([2500., 3500., 2000.])),
)
print(result["node_count"])
print(result["edges"].shape)        # (E, 2) [child, parent]
```

### Which objects are present in a region?

`read_graph` has no `return_object_ids` option and never returns an
`object_ids` key. To find which objects have nodes in a region, test
candidate IDs against the level index:

```python
from zarr_vectors.lazy import open_zv

store = open_zv("connectome.zarrvectors")
level = store[0]

print(level.present_oids)          # object IDs present at this level
print(level.has_object(42))        # True / False
print(store.object_levels(42))     # levels where object 42 exists
```

---

## General graphs

### Writing a graph

```python
from zarr_vectors.types.graphs import write_graph

rng = np.random.default_rng(0)
n_nodes   = 2000
positions = rng.uniform(0, 500, (n_nodes, 3)).astype(np.float32)

# Simulate a vascular network: ~3 edges per node
src = rng.integers(0, n_nodes, 3000)
dst = rng.integers(0, n_nodes, 3000)
# Remove self-loops
mask  = src != dst
edges = np.column_stack([src[mask], dst[mask]]).astype(np.int32)

write_graph(
    "vessels.zarrvectors",
    positions=positions,
    edges=edges,
    chunk_shape=(100.0, 100.0, 100.0),
    bin_shape=(25.0, 25.0, 25.0),
    kind="graph",
    vertex_attributes={
        "diameter": rng.uniform(1, 20, n_nodes).astype(np.float32),
        "flow":     rng.uniform(0, 1,  n_nodes).astype(np.float32),
    },
)
```

### Edge direction

`write_graph` takes no direction argument — there is no `is_directed`
parameter. Direction is a property of the geometry type, decided by
`kind`:

- `kind="graph"` writes its links family **undirected**. `A→B` and `B→A`
  are the same edge and are stored once, under a single canonical
  offsets segment.
- `kind="skeleton"` writes its links family **`directed=True`**, because
  parent→child order is data. `A→B` and `B→A` are distinct records and
  file under *opposite* offsets segments.

If you need a genuinely directed graph, `kind="skeleton"` is the type
that preserves endpoint order.

### Reading a graph

```python
from zarr_vectors.types.graphs import read_graph

result = read_graph("vessels.zarrvectors")
print(result["node_count"])              # 2000
print(result["edge_count"])
print(result["positions"].shape)         # (2000, 3)
print(result["edges"].shape)             # (E, 2)
```

Per-vertex attributes such as `diameter` are not in this dict — read them
lazily as shown above.

---

## Where edges live on disk

Every edge — intra-chunk and cross-chunk alike — lives in the single
`links/<delta>/` family. Both types use `link_width=2`.

- An edge whose endpoints share a chunk files under the all-zero offsets
  segment, `links/0/0.0.0/`.
- An edge crossing a chunk boundary files under the segment naming that
  displacement, e.g. `links/0/0.0.+1/`.

There is no separate cross-chunk array. See
[Links](../../spec/object_model/links.md) for the on-disk layout.

---

## GraphML ingest

GraphML conversion lives in the companion package
**`zarr-vectors-tools`**.

---

## Validation

```python
from zarr_vectors.validate import validate

result = validate("neuron.zarrvectors", level=4)
print(result.summary())
# Level 4 validation: PASS
#   38 passed, 0 warnings, 0 errors
```

Level 4 checks that the declared geometry type has a compatible
`links_convention`. It does **not** perform tree-topology checks: there
is no connectivity, acyclicity, or single-root validation at any level.
A cyclic graph stored as `kind="skeleton"` passes L4 with no errors.

---

## Multi-resolution pyramids

Graph pyramids coarsen vertex positions and deduplicate edges:

```python
from zarr_vectors.multiresolution.coarsen import build_pyramid

build_pyramid(
    "vessels.zarrvectors",
    factors=[(2.0, 1.00)],
)
```

Bin aggregation is fixed: source vertices collapse to their centroid.
There is no aggregation-mode parameter.

For skeleton stores with `object_sparsity < 1.0`, individual neurons are
thinned at coarser levels using the declared sparsity strategy.

---

## Common pitfalls

**Tree topology is never validated.**
`kind="skeleton"` reorders nodes depth-first but does not check that your
data is a tree. Writing a cyclic edge list as `kind="skeleton"` succeeds
silently, and the resulting store passes validation at every level. If
tree-ness matters to you, check it yourself before writing:

```python
import networkx as nx

G = nx.from_edgelist(edges.tolist())
assert nx.is_forest(G), "edges contain a cycle"
```

**SWC parent ID −1 vs 0.**
Some SWC tools use parent ID `0` (1-indexed) for the root; others use
`-1` (ZVF convention). SWC ingest lives in **`zarr-vectors-tools`**, so
consult that package for how it detects the root convention. Writing
through the core `write_graph` API, you supply `edges` directly and the
convention is whatever you encode.

**Object IDs change after rechunking.**
Object IDs are assigned at write time and are stable across reads on the
same store. However, rechunking rebuilds the `object_index/` and may
reassign IDs. If you need stable long-term IDs (e.g. for a connectome
database), store the canonical ID as a per-object attribute:

```python
write_graph(..., object_attributes={"neuron_id": canonical_ids})
```
