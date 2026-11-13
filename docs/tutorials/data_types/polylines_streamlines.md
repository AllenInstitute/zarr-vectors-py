# Polylines and streamlines

In this tutorial you write tractography streamlines with per-vertex and
per-streamline values, read them back, query a region, name bundles, add
coarser levels and validate the store. The same calls write any ordered path,
such as a vessel centreline. Run the steps in order as one Python session.

## 1. Create the store

Each polyline is one object. Declare per-vertex and per-object attributes in
the `Schema`:

```python
import numpy as np
import zarr_vectors as zv

tracts = zv.create("tracts.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    kind="polyline",
    vertex_attributes={"fa": zv.AttributeSpec(dtype="float32")},
    object_attributes={"mean_fa": zv.AttributeSpec(dtype="float32")},
    layout=zv.Layout(cells=5),
))
```

[Choosing a layout](../../how_to/choose_chunk_and_bin.md) explains how to pick
the chunk and bin shapes.

## 2. Write the streamlines

Pass a list of `(N_k, 3)` arrays. A per-vertex attribute is also a list, with
one array per streamline. A per-object attribute has one value per
streamline. `groups=` maps an integer group id to streamline indices.

```python
rng = np.random.default_rng(0)
n = 1000
starts = rng.uniform(300.0, 700.0, size=(n, 3))
streamlines = [
    (s + rng.normal(0.0, 6.0, size=(int(k), 3)).cumsum(axis=0)).astype(np.float32)
    for s, k in zip(starts, rng.integers(20, 60, size=n))
]
fa = [rng.uniform(0.1, 0.9, len(s)).astype(np.float32) for s in streamlines]
mean_fa = np.array([f.mean() for f in fa], dtype=np.float32)

tracts.add_polylines(
    streamlines,
    attributes={"fa": fa},
    object_attributes={"mean_fa": mean_fa},
    groups={0: range(0, 300), 1: range(300, 600), 2: range(600, 1000)},
    streamlines=True,
)
print(tracts.kinds)
```

```text
('polyline', 'streamline')
```

`streamlines=True` adds `"streamline"` to the store's kinds.

## 3. Read it back

`positions` holds every vertex, and `polylines` splits them into one array per
streamline:

```python
r = tracts.read()
print(r)
print(np.allclose(r.polylines[0], streamlines[0]))
print(tracts.level(0).objects[42].vertex_count)
```

```text
ReadResult(kind='polyline', vertices=39019, parts=1000, attributes=['fa'])
True
51
```

`objects[k]` reads one streamline; `objects[[i, j]]` reads several.

## 4. Query a region

A bounding-box query returns whole, unclipped streamlines: every one with a
vertex in a chunk the box touches.

```python
box = ((400.0, 400.0, 400.0), (600.0, 600.0, 600.0))
in_region = tracts.select(bbox=box).read()
inside = [((p >= 400.0) & (p <= 600.0)).all(axis=1).any() for p in in_region.polylines]
print(in_region.part_count, sum(inside))
```

```text
521 219
```

Some returned streamlines have no vertex in the box. Filter in memory, as
above, when you need only those that do.

## 5. Named groups

`groups=` takes integer keys. Name the groups afterwards, with the store open
for writing:

```python
tracts = zv.open("tracts.zarrvectors", mode="r+")
tracts.level(0).groups.name_rows(["CST", "AF", "UF"])
print(tracts.groups)
print(tracts.groups["CST"].read().part_count)
```

```text
GroupCatalog(['CST', 'AF', 'UF'])
300
```

## 6. Add coarser levels

Each `factors` entry is a `(coarsen_factor, sparsity_factor)` pair. A sparsity
factor of 2.0 keeps a random half of the streamlines; `sparsity_seed=` makes
the choice repeatable.

```python
tracts.build_pyramid(factors=[(2.0, 1.0), (2.0, 2.0)], sparsity_seed=0)
for i in tracts.levels:
    lv = tracts.level(i)
    print(i, lv.resolution, lv.objects.count, lv.attribute_names("object"))
```

```text
0 (50.0, 50.0, 50.0) 1000 ('mean_fa',)
1 (100.0, 100.0, 100.0) 1000 ('mean_fa',)
2 (200.0, 200.0, 200.0) 500 ('mean_fa',)
```

Each kept streamline keeps its object id and has one vertex per bin it passes
through. Per-object attributes reach every level; per-vertex attributes stay
on level 0. [Building pyramids](../multiscale/building_pyramids.md) covers the
options.

## 7. Validate

```python
print(tracts.validate(level=4).summary())
```

```text
Level 4 validation: PASS
  53 passed, 0 warnings, 0 errors
```

[Validation and repair](../../how_to/validate_and_repair.md) lists the checks.
To convert TRK, TCK or TRX files, use the companion package
`zarr-vectors-tools`.
