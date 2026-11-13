# Graphs and skeletons

In this tutorial you write branching neuron skeletons, read them back, query a
region, write a general graph, add coarser levels and validate the store. A
**skeleton** is one rooted tree per object; a **graph** is any set of nodes
and edges. Run the steps in order as one Python session.

## 1. Create the store

```python
import numpy as np
import zarr_vectors as zv

neurons = zv.create("neurons.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    kind="skeleton",
    vertex_attributes={"radius": zv.AttributeSpec(dtype="float32")},
    layout=zv.Layout(cells=5),
))
```

[Choosing a layout](../../how_to/choose_chunk_and_bin.md) explains how to pick
the chunk and bin shapes.

## 2. Write the skeletons

`add_graph` takes every node's position, the edges as an `(E, 2)` array of
`[child, parent]` node indices, and one object id per node. `tree=True` writes
a skeleton. Each neuron here is a 40-node random walk with one side branch:

```python
rng = np.random.default_rng(0)
n_neurons, n_nodes = 20, 40
parent = np.arange(-1, n_nodes - 1)
parent[25] = 10                       # node 25 branches off node 10

positions, edges, object_ids = [], [], []
for k in range(n_neurons):
    pos = np.zeros((n_nodes, 3))
    pos[0] = rng.uniform(300.0, 700.0, 3)
    for i in range(1, n_nodes):
        pos[i] = pos[parent[i]] + rng.normal(0.0, 12.0, 3)
    positions.append(pos)
    edges.append(np.column_stack([np.arange(1, n_nodes), parent[1:]]) + k * n_nodes)
    object_ids.append(np.full(n_nodes, k))
positions = np.concatenate(positions).astype(np.float32)
edges = np.concatenate(edges)
object_ids = np.concatenate(object_ids)
radius = rng.uniform(0.2, 2.0, len(positions)).astype(np.float32)

neurons.add_graph(positions, edges, attributes={"radius": radius},
                  object_ids=object_ids, tree=True)
```

Each object must be one tree with one root; a second root raises
`ArrayError`. The writer does not check for cycles.

## 3. Read it back

`read()` returns every node and every edge as `[child, parent]` indices into
`positions`. Nodes come back in stored order, not the order you wrote, so use
`r.edges` with `r.positions`.

```python
r = neurons.read()
print(r)
print(neurons.level(0).objects[3])
```

```text
ReadResult(kind='skeleton', vertices=800, edges=780, attributes=['radius'])
ReadResult(kind='skeleton', vertices=40, edges=39)
```

## 4. Query a region

A bounding-box query returns the nodes inside the box and the edges with both
ends inside it. It returns no attributes.

```python
box = ((400.0, 400.0, 400.0), (600.0, 600.0, 600.0))
print(neurons.select(bbox=box).read())
```

```text
ReadResult(kind='skeleton', vertices=186, edges=162)
```

## 5. Write a general graph

Leave out `tree=True` to write a graph. Every edge is stored, and per-edge
values go in `edge_attributes=`:

```python
vessels = zv.create("vessels.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    kind="graph",
    layout=zv.Layout(cells=5),
))
nodes = rng.uniform(100.0, 900.0, (2000, 3)).astype(np.float32)
links = rng.integers(0, 2000, (6000, 2))
links = links[links[:, 0] != links[:, 1]]
weight = rng.uniform(0.0, 1.0, len(links)).astype(np.float32)

vessels.add_graph(nodes, links, edge_attributes={"weight": weight})
print(vessels.read())
```

```text
ReadResult(kind='graph', vertices=2000, edges=5994)
```

Edges are directed: `[b, a]` is a separate edge from `[a, b]`.

## 6. Add coarser levels

Each `factors` entry is a `(coarsen_factor, sparsity_factor)` pair. A sparsity
factor of 2.0 keeps half of the neurons, whole; `sparsity_seed=` makes the
choice repeatable.

```python
neurons.build_pyramid(factors=[(2.0, 1.0), (2.0, 2.0)], sparsity_seed=0)
for i in neurons.levels:
    print(i, neurons.level(i).objects.count, neurons.read(level=i))
```

```text
0 20 ReadResult(kind='skeleton', vertices=800, edges=780, attributes=['radius'])
1 20 ReadResult(kind='skeleton', vertices=67, edges=0)
2 10 ReadResult(kind='skeleton', vertices=18, edges=0)
```

A coarse level holds the centroids of the occupied bins and no edges. Object
ids are the same at every level.
[Building pyramids](../multiscale/building_pyramids.md) covers the options.

## 7. Validate

```python
print(neurons.validate(level=4).summary())
```

```text
Level 4 validation: PASS
  48 passed, 0 warnings, 0 errors
```

[Validation and repair](../../how_to/validate_and_repair.md) lists the checks.
To convert SWC or GraphML files, use the companion package
`zarr-vectors-tools`.
