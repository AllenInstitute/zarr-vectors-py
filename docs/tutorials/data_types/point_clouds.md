# Point clouds

In this tutorial you write a point cloud with per-point attributes, read it
back, query a region, add coarser levels and validate the store. Run the steps
in order as one Python session. [Concepts](../../explanation/concepts.md)
defines chunks, bins and levels.

## 1. Create the store

A `Schema` sets the bounds, axes, geometry kind and attributes.
`Layout(cells=5)` cuts each axis into five chunks.

```python
import numpy as np
import zarr_vectors as zv

ds = zv.create("scan.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    axes=(zv.Axis("x", unit="micrometer"),
          zv.Axis("y", unit="micrometer"),
          zv.Axis("z", unit="micrometer")),
    kind="point_cloud",
    vertex_attributes={
        "intensity": zv.AttributeSpec(dtype="float32"),
        "label": zv.AttributeSpec(dtype="int32", categorical=True),
        "color": zv.AttributeSpec(dtype="uint8", channels=3),
    },
    layout=zv.Layout(cells=5),
))
print(ds.level(0).scale, ds.level(0).resolution)
```

```text
(200.0, 200.0, 200.0) (50.0, 50.0, 50.0)
```

`scale` is the chunk shape and `resolution` the bin shape.
[Choosing a layout](../../how_to/choose_chunk_and_bin.md) explains how to pick
them.

## 2. Write the points

Pass the positions as an `(N, 3)` array and each attribute with one row per
point. Write all points in one call: a second `add_points` call replaces the
first.

```python
rng = np.random.default_rng(42)
n = 100_000
positions = rng.uniform(0, 1000, (n, 3)).astype(np.float32)
ds.add_points(positions, attributes={
    "intensity": rng.uniform(0, 1, n).astype(np.float32),
    "label": rng.integers(0, 8, n).astype(np.int32),
    "color": rng.integers(0, 256, (n, 3)).astype(np.uint8),
})
print(ds.level(0).vertex_count)
```

```text
100000
```

A point outside `bounds` raises `MetadataError`. Pass
`on_out_of_bounds="ignore"` to drop such points, or `"expand"` to grow the
bounds.

## 3. Read it back

```python
r = ds.read()
print(r)
print(r.positions.shape, r.attributes["color"].shape)
```

```text
ReadResult(kind='point_cloud', vertices=100000, attributes=['color', 'intensity', 'label'])
(100000, 3) (100000, 3)
```

## 4. Query a region

`select()` builds a query without reading; `count()` and `read()` run it. For
a point cloud the box is exact.

```python
q = ds.select(bbox=((100.0, 100.0, 100.0), (200.0, 200.0, 200.0)))
print(q.count())
box = q.read()
print(box.positions.min(axis=0), box.positions.max(axis=0))
```

```text
109
[100.439674 100.72887  100.454704] [199.90297 198.85411 197.56297]
```

`select()` also takes `near=` and `attributes=`, and `.limit()` caps the rows;
see [Deferred reads](../multiscale/lazy_loading.md).

## 5. Objects and groups

Points can belong to objects, and objects to groups. Pass one object id per
point, and `groups=` as a mapping from group id to object ids:

```python
spots_ds = zv.create("spots.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    kind="point_cloud",
    layout=zv.Layout(cells=5),
))
centres = rng.uniform(100, 900, (10, 3))
spots = np.concatenate([c + rng.normal(0, 20, (100, 3)) for c in centres]).astype(np.float32)
spots_ds.add_points(spots, object_ids=np.repeat(np.arange(10), 100),
                    groups={0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9]})

print(spots_ds.level(0).objects[3].vertex_count)
print(spots_ds.select(groups=[1]).read().vertex_count)
```

```text
100
500
```

## 6. Add coarser levels

Each `factors` entry is a `(coarsen_factor, sparsity_factor)` pair. Points
that share a bin become their centroid. Attributes stay on level 0.

```python
ds.build_pyramid(factors=[(2.0, 1.0), (2.0, 1.0)])
for i in ds.levels:
    print(i, ds.level(i).vertex_count, ds.level(i).resolution)
```

```text
0 100000 (50.0, 50.0, 50.0)
1 1000 (100.0, 100.0, 100.0)
2 125 (200.0, 200.0, 200.0)
```

[Building pyramids](../multiscale/building_pyramids.md) covers the options.

## 7. Validate

```python
print(ds.validate(level=4).summary())
```

```text
Level 4 validation: PASS
  41 passed, 1 warnings, 0 errors
  WARN:  Point cloud but links array exists
```

The warning refers to the links between levels that `build_pyramid` writes.
The store reads correctly.
[Validation and repair](../../how_to/validate_and_repair.md) lists the checks.

To convert LAS, PLY, CSV or XYZ files, use the companion package
`zarr-vectors-tools`.
