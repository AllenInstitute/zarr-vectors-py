# Graphs and skeletons

In this tutorial you write a set of branching neuron skeletons to a new store,
read them back whole, by neuron and by region, add coarser levels and validate
the result; the last section writes a general graph, such as a vascular
network. A **skeleton** is one rooted tree per object; a **graph** is any set
of nodes and edges.

The steps run in order as one Python session. The
[quickstart](../../getting_started/quickstart.md) introduces the shared calls
on point clouds and polylines, and [Concepts](../../getting_started/concepts.md)
explains chunks, bins and objects.

## 1. Create the store

```python
import numpy as np
import zarr_vectors as zv

neurons = zv.create("neurons.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    axes=(zv.Axis("x", unit="micrometer"),
          zv.Axis("y", unit="micrometer"),
          zv.Axis("z", unit="micrometer")),
    kind="skeleton",
    vertex_attributes={"radius": zv.AttributeSpec(dtype="float32", unit="micrometer")},
    object_attributes={"soma_radius": zv.AttributeSpec(dtype="float32")},
    expected=zv.SizeHints(n_vertices=800, n_objects=20),
    layout=zv.Layout(cells=5),
))
print(neurons.level(0).scale, neurons.level(0).resolution)
```

```text
(200.0, 200.0, 200.0) (50.0, 50.0, 50.0)
```

[Choosing a layout](../../how_to/choose_chunk_and_bin.md) explains how to pick
the chunk and bin shapes.

## 2. Write the skeletons

`add_graph` takes every node's position, the edges as an `(E, 2)` array of
`[child, parent]` node indices, and one object id per node. `tree=True` writes
the store as a skeleton. Each neuron here is a 40-node random walk with one
side branch:

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
soma_radius = rng.uniform(5.0, 10.0, n_neurons).astype(np.float32)

print(neurons.add_graph(
    positions, edges,
    attributes={"radius": radius},
    object_attributes={"soma_radius": soma_radius},
    object_ids=object_ids,
    tree=True,
))
```

```text
{'node_count': 800, 'edge_count': 780, 'chunk_count': 26, 'intra_edge_count': 20, 'cross_edge_count': 145, 'object_count': 20, 'kind': 'skeleton'}
```

A skeleton stores nodes in depth-first order and keeps an edge on disk only
where the order cannot imply it: at a branch, or where the edge crosses a
chunk boundary. That is why the report counts 780 edges but far fewer stored
ones. The [skeleton specification](../../spec/geometry_types/skeleton.md)
describes the layout.

## 3. Read it back

`read()` returns every node, with its attributes, and every edge as
`[child, parent]` node indices into `positions`:

```python
r = neurons.read()
print(r)
print(r.positions.shape, r.edges.shape, r.attributes["radius"].shape)
```

```text
ReadResult(kind='skeleton', vertices=800, edges=780, attributes=['radius'])
(800, 3) (780, 2) (800,)
```

Nodes come back in the stored order, which need not match the order you wrote.
Use `r.edges` with `r.positions`, not your original indices.

To read one neuron, or a few, index the level's objects:

```python
one = neurons.level(0).objects[3]
print(one, one.edges.max() < one.vertex_count)
print(neurons.select(objects=[3, 5]).read())
```

```text
ReadResult(kind='skeleton', vertices=40, edges=39) True
ReadResult(kind='skeleton', vertices=80, edges=78)
```

## 4. Query a region

A bounding-box query returns the nodes inside the box and the edges whose
both ends are inside it:

```python
box = ((400.0, 400.0, 400.0), (600.0, 600.0, 600.0))
q = neurons.select(bbox=box)
print(q.explain())
in_region = q.read()
print(in_region, in_region.attributes_read)
print(bool(((in_region.positions >= 400.0) & (in_region.positions <= 600.0)).all()))
```

```text
skeleton read: level 0, bbox [400.0, 400.0, 400.0]..[600.0, 600.0, 600.0]
  via read_graph(bbox, level)
ReadResult(kind='skeleton', vertices=186, edges=162) False
True
```

A region read of a skeleton or graph returns no attributes:
`attributes_read` is `False`. Read the whole store, or the objects you need,
when you need them.

## 5. Per-neuron values

`Level.attribute_names("object")` lists the per-object attributes, and
`zarr_vectors.building.read_object_attributes` returns one value per object
id without reading any geometry:

```python
from zarr_vectors import building

print(neurons.level(0).attribute_names("object"))
level_group = building.get_resolution_level(
    building.open_store("neurons.zarrvectors", mode="r"), 0)
soma = building.read_object_attributes(level_group, "soma_radius")
large = np.nonzero(soma > 8.0)[0].tolist()
print(len(large), neurons.select(objects=large).read().vertex_count)
```

```text
('soma_radius',)
8 320
```

Skeleton and graph stores do not support `select(groups=...)`; select a group's
neurons with `objects=` instead.

## 6. Add coarser levels

`build_pyramid` writes coarser levels after level 0. Each `factors` entry is a
`(coarsen_factor, sparsity_factor)` pair applied to the level before it; a
sparsity factor of 2.0 keeps half of the neurons, whole, and drops the rest.
`sparsity_seed=` makes the random choice repeatable.

```python
rep = neurons.build_pyramid(factors=[(2.0, 1.0), (2.0, 2.0)], sparsity_seed=0)
print(rep["levels_created"])
for i in neurons.levels:
    lv = neurons.level(i)
    print(i, lv.resolution, lv.vertex_count, lv.objects.count, lv.read().edges.shape)
```

```text
2
0 (50.0, 50.0, 50.0) 800 20 (780, 2)
1 (100.0, 100.0, 100.0) 67 20 (0, 2)
2 (200.0, 200.0, 200.0) 18 10 (0, 2)
```

A coarse level holds the centroids of the occupied bins and no edges.
Object ids are the same at every level. [Building
pyramids](../multiscale/building_pyramids.md) covers the other options.

## 7. Validate

```python
print(neurons.validate(level=4).summary())
```

```text
Level 4 validation: PASS
  51 passed, 0 warnings, 0 errors
```

[Validation and repair](../io/validation_and_repair.md) lists what each level
checks.

To convert SWC or GraphML files to and from a store, use the companion package
`zarr-vectors-tools`.

## General graphs

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

print(vessels.add_graph(nodes, links, edge_attributes={"weight": weight}))
g = vessels.read()
print(g)
print(vessels.level(0).attribute_names("link"))

# Map the stored node order back to the input order to compare the edges.
written_index = {tuple(p): i for i, p in enumerate(nodes)}
to_input = np.array([written_index[tuple(p)] for p in g.positions])
print(set(map(tuple, to_input[g.edges].tolist())) == set(map(tuple, links.tolist())))
```

```text
{'node_count': 2000, 'edge_count': 5994, 'chunk_count': 125, 'intra_edge_count': 51, 'cross_edge_count': 5943, 'object_count': 1, 'kind': 'graph'}
ReadResult(kind='graph', vertices=2000, edges=5994)
('weight',)
True
```

As with a skeleton, nodes come back in the stored order. Each edge keeps the
column order you wrote, and the reversed pair `[b, a]` is a separate edge from
`[a, b]`.

## Common mistakes

**Two roots in one skeleton.** Each skeleton object must be one tree with one
root. A second root in the same object raises `ArrayError`; give each tree its
own object id, or write the data as a graph.

**Cycles in a skeleton.** The writer does not check for cycles. Check that
each object is a tree before writing it with `tree=True`.

**Per-edge values on a skeleton.** A skeleton stores only the edges it cannot
imply, so `edge_attributes=` keeps values for those edges alone. Store a
per-edge value on the child node of the edge, as a vertex attribute.
