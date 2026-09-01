# Graphs and skeletons

ZVF provides two graph-structured geometry types. Use `skeleton` for
neuronal morphologies, vascular trees, and any other branching structure
that must align to the SWC convention. Use `graph` for arbitrary
connectivity — vascular networks with anastomoses, synaptic connectivity
graphs embedded in 3-D space, or any structure where cycles are valid.

Both types use the same on-disk array schema; the distinction is the
`kind` argument to `write_graph`. `kind="skeleton"` reorders nodes
depth-first and declares the store's geometry type as `skeleton`. The
SWC-compatible per-vertex attributes (`radius`, `swc_type`) are ones you
write yourself — a skeleton store written without them has none.

All examples on this page use only the core `zarr-vectors` package.
SWC/GraphML converters live in the companion package
**`zarr-vectors-tools`**.

`write_graph` and `read_graph` come from `zarr_vectors.types`, which is
**undecided** — neither promised nor disowned. The writer is also
exported from `zarr_vectors.building` and supported there; the reader
cannot be retired until the data api can carry per-vertex attributes for
every geometry. Everything else on this page uses the two supported
surfaces: `zarr_vectors` itself for reading data, and
`zarr_vectors.building` for the physical layout. Ask at runtime with
`zv.stability("zarr_vectors.types")`.

---

## Skeletons (SWC-aligned)

### Write a skeleton programmatically

```python
import numpy as np
from zarr_vectors.building import write_graph

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
IDs. Read attributes through the data API, which carries them on the
result:

```python
import zarr_vectors as zv

level = zv.open("neuron.zarrvectors").level(0)
print(level.attribute_names("vertex"))   # ('radius', 'swc_type') — reads no data

result = level.read()
radii  = result.attributes["radius"]     # (N,) float32
types  = result.attributes["swc_type"]   # (N,) int32
```

Earlier versions of this page used `zarr_vectors.lazy.open_zv` here. That
module is internal, and `open_zv` now warns why on every call: the lazy
layer reads chunk by chunk in Python and opens no batched-read block, so
against an object store it is slower than the eager path it was meant to
improve on. `zv.open` returns a `Dataset` that drives the batching engine
instead.

`result.attributes` is a real mapping — `names()`, `attributes[name]`,
`name in attributes`, iteration, `len()` — not the write-only proxy the
lazy layer handed back.

The columns are in **stored** order for the whole level, which for
`kind="skeleton"` is the depth-first reordering applied at write time.
A *narrowed* read does not line them up row-for-row; it declines to guess
instead, coming back with `attributes_read == False` and an empty
`attributes`.

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

> **Known limitation.** `read_graph` accepts an `object_ids=` argument
> but does not implement it. It refuses rather than pretending:
>
> ```pycon
> >>> read_graph("connectome.zarrvectors", object_ids=[42])
> NotImplementedError: read_graph(object_ids=...) is not implemented: the filter
> would be silently ignored and you would get the whole level back. Filter the
> returned arrays yourself, or use read_polylines, which does implement object_ids.
> ```
>
> `level.objects[42]` on the data API dispatches to the same reader and
> raises the same error, so one neuron cannot be read by id from either
> supported surface. Use `chunks=` or `bbox=` to restrict a read
> spatially, which does work, or gather one object's nodes through
> `zarr_vectors.building`:
>
> ```python
> import numpy as np
> from zarr_vectors.building import (
>     get_resolution_level, open_store, read_object_vertices,
> )
>
> level_group = get_resolution_level(open_store("connectome.zarrvectors", mode="r"), 0)
> neuron_42 = np.concatenate(read_object_vertices(level_group, 42, ndim=3))
> ```
>
> That follows the object's manifest and gives you its node positions —
> not its edges, which still means reading the level and filtering. It is
> the gap to report rather than a reason to import from `core`.

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

### Which objects does the store hold?

`read_graph` has no `return_object_ids` option and never returns an
`object_ids` key. Ask the level's object catalogue instead:

```python
import zarr_vectors as zv

ds = zv.open("connectome.zarrvectors")
level = ds.level(0)

print(level.objects)                       # ObjectCatalog(level=0, count=50, slots=50)
print(level.objects.ids())                 # ids that actually hold geometry
print(42 in level.objects)                 # True / False
print(len(level.objects))                  # slot count, from metadata; reads nothing

# levels where object 42 exists (replaces the lazy object_levels(42))
print([i for i in ds.levels if 42 in ds.level(i).objects])
```

`ids()` returns only ids that hold geometry, which is the distinction
that matters on a sparsified pyramid level: a dropped object keeps its
slot so ids stay stable across levels, so `count` and `slots` diverge
(25 present out of 50 slots, in the sparsified example below) and the
`ids()` list is the shorter one. Pass `present=False` for every
addressable slot.

The catalogue answers for a whole level, not for a region — and for
graphs and skeletons it is the only answer available. A query's
`object_ids()` terminal comes back empty on these stores, because the
graph reader carries no per-vertex object ids.

---

## General graphs

### Writing a graph

```python
from zarr_vectors.building import write_graph

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
parameter, and `kind` does not add one behind your back. Both kinds write
the same delta-0 links family policy, and you can read it back:

```python
from zarr_vectors.building import (
    get_resolution_level, link_family_policy, open_store,
)

level_group = get_resolution_level(open_store("vessels.zarrvectors", mode="r"), 0)
print(link_family_policy(level_group, 0))
# (2, 3, False, 'canonical')    # link_width, sid_ndim, directed, store
```

A `neuron.zarrvectors` written with `kind="skeleton"` prints the same
tuple. Earlier versions of this page said skeletons were written
`directed=True` and that `A→B` and `B→A` filed under *opposite* offsets
segments. Neither is true of the shipped writer. What does hold is
weaker and simpler:

- **Endpoint order is preserved.** Write a chain as `[child, parent]`
  pairs and `read_graph` hands them back as `[child, parent]` — for both
  kinds, including for edges that cross a chunk boundary. Parent→child
  direction rides on the column order you wrote, not on a flag in the
  store.
- **Nothing collapses a reversed duplicate.** Writing both `[0, 1]` and
  `[1, 0]` stores two records and reads two edges back, and both file
  under the *same* canonical offsets segment. If you want an undirected
  edge stored once, deduplicate before writing.

So neither kind produces a store a reader can tell is directed. `kind`
decides node ordering (depth-first for `skeleton`) and what the store
declares as its geometry type — not edge semantics.

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
off a `ReadResult` as shown above.

---

## Where edges live on disk

Every edge — intra-chunk and cross-chunk alike — lives in the single
`links/<delta>/<offsets>/` family. Both types use `link_width=2`.

- `<delta>` is how many pyramid levels the record spans: `0` for the
  edges you wrote, `+1` / `-1` for the parent/child links a pyramid adds
  (see [Multi-resolution pyramids](#multi-resolution-pyramids)). A store
  with no pyramid has only `links/0/`.
- `<offsets>` is where the record's other endpoint sits relative to the
  chunk holding it. An edge whose endpoints share a chunk files under the
  all-zero segment, `links/0/0.0.0/`; one crossing a chunk boundary files
  under the segment naming that displacement, e.g. `links/0/0.0.+1/`.

A cross-chunk edge is simply one with non-zero offsets. There is no
separate cross-chunk array — no `cross_chunk_links/` — and there has not
been one since format 0.9.0. See
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
#   22 passed, 0 warnings, 0 errors
```

The count is how many assertions ran, not a score: it moves with what the
store contains. Level 4 checks that the declared geometry type has a
compatible `links_convention`. It does **not** perform tree-topology checks: there
is no connectivity, acyclicity, or single-root validation at any level.
A cyclic graph stored as `kind="skeleton"` passes L4 with no errors.

---

## Multi-resolution pyramids

A graph pyramid replaces each bin of source vertices with one
metavertex at their centroid. Build it from the dataset:

```python
import zarr_vectors as zv

ds = zv.open("vessels.zarrvectors", mode="r+")
print(zv.coarsen_methods())          # ('per_object',)

report = ds.build_pyramid(factors=[(2.0, 1.00)], method="per_object")
print(report["levels_created"])      # 1
print(report["level_specs"][0]["vertex_count"])
```

`zarr_vectors.multiresolution.coarsen.build_pyramid` is the same routine
one layer down. That module is internal, so reach it through the
`Dataset` method.

Each `factors` entry is a `(coarsen_factor, sparsity_factor)` pair:

- **`coarsen_factor`** multiplies the bin shape. Aggregation itself is
  fixed — source vertices in a bin collapse to their centroid — and there
  is no aggregation-mode parameter. `method=` chooses a coarsening
  *strategy*, not an aggregation; `zv.coarsen_methods()` lists what is
  installed, which in the core package is `per_object` alone.
- **`sparsity_factor`** is an inverse. `1.0` keeps every object, `2.0`
  keeps half. It is stored as the level's `object_sparsity`, computed as
  `1 / sparsity_factor`, so a value below 1 is rejected — and the message
  names the derived value, not the one you passed: `factors=[(2.0, 0.5)]`
  raises `MetadataError: object_sparsity must be in (0, 1], got 2.0`.

Sparsity drops **whole objects**, chosen by `sparsity_strategy=`
(`"random"` in the core package). It applies to graphs and skeletons
alike, and a neuron is either present at a coarse level or absent from
it — never thinned. Dropped objects keep their slot, so ids stay stable
across levels: on this store, `factors=[(2.0, 2.00)]` gives a level 1
with 25 objects present out of 50 slots.

**A coarse level carries no edges.** It gets `vertices`,
`vertex_fragments`, `object_index` and a `links/-1/` family — the
cross-level links back to its parent's vertices — and nothing under
`links/0/`. Level 0 gains the matching `links/+1/` family in the same
pass. Reading the two levels back shows it:

```python
ds = zv.open("vessels.zarrvectors")
for i in ds.levels:
    level = ds.level(i)
    print(i, level.vertex_count, level.objects.count, level.read().edges.shape)
```

```text
0 2000 50 (3000, 2)
1 866 50 (0, 2)
```

If you need connectivity at a coarse level, derive it from the level-0
edges and the cross-level links yourself.

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

**Rechunking rebuilds the object index.**
Object IDs are assigned at write time and are stable across reads on the
same store, and across pyramid levels. Rechunking —
`zarr_vectors.building.rechunk` and `rechunk_by_attribute` — is the
exception: it writes a *new* store whose chunk keys gain a leading bin
dimension, and rebuilds `object_index/manifests/` against that new grid.
The shipped code carries the id set over, but nothing in the format
promises the mapping. If you need stable long-term IDs (e.g. for a
connectome database), do not lean on the slot number — store the
canonical ID as a per-object attribute:

```python
write_graph(..., object_attributes={"neuron_id": canonical_ids})
```
