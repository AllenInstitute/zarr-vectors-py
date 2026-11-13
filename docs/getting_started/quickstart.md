# Quickstart

In this tutorial you create a point-cloud store, write points to it, read a
region, build coarser levels and validate the store. It uses synthetic data
and needs only the base install.

Install the package first with `pip install zarr-vectors`; see
[Installation](installation.md). Run the blocks in order in one Python
session.

## 1. Create a store

A `Schema` describes the data: its bounds, axes, geometry kind and
attributes. Its `Layout` sets the size of the chunks and bins; see
[Core concepts](../explanation/concepts.md).

```python
import numpy as np
import zarr_vectors as zv

schema = zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    axes=(
        zv.Axis("x", unit="micrometer"),
        zv.Axis("y", unit="micrometer"),
        zv.Axis("z", unit="micrometer"),
    ),
    kind="point_cloud",
    vertex_attributes={"intensity": zv.AttributeSpec(dtype="float32")},
    layout=zv.Layout(cells=5),
)
ds = zv.create("scan.zarrvectors", schema=schema)

print(ds)
print(ds.level(0).scale, ds.level(0).resolution)
```

```text
Dataset('file:///.../scan.zarrvectors', point_cloud, levels=[0])
(200.0, 200.0, 200.0) (50.0, 50.0, 50.0)
```

`cells=5` cuts the 1000 µm bounds into five chunks per axis, so each chunk is
200 µm. Each chunk holds four bins per axis, so each bin is 50 µm.
`bounds` is required; `layout=` is optional. The store is a directory called
`scan.zarrvectors/`.

## 2. Write points

```python
rng = np.random.default_rng(42)
positions = rng.uniform(0, 1000, size=(100_000, 3)).astype(np.float32)
intensity = rng.uniform(0, 1, size=100_000).astype(np.float32)

report = ds.add_points(positions, attributes={"intensity": intensity})
print(report)
```

```text
{'vertex_count': 100000, 'chunk_count': 125, 'object_count': 0, 'group_count': 0, 'bins_per_chunk': (4, 4, 4)}
```

:::{warning}
`add_points` writes once; it does not append. A second call replaces the
chunks its points fall in. Assemble the full array first.
:::

Points outside the bounds raise `MetadataError`.

## 3. Read a region

`ds.read()` reads the whole store. `select()` builds a query and reads
nothing; `count()` and `read()` run it:

```python
q = ds.select(bbox=((100.0, 100.0, 100.0), (200.0, 200.0, 200.0)))
print(q.count())

r = q.read()
print(r)
print(r.positions.min(axis=0), r.positions.max(axis=0))
```

```text
109
ReadResult(kind='point_cloud', vertices=109, attributes=['intensity'])
[100.439674 100.72887  100.454704] [199.90297 198.85411 197.56297]
```

Every read returns a `ReadResult`, whatever the geometry. `positions` is an
`(N, D)` array, and `attributes` holds one array per attribute name. For a
point cloud the box is exact. For polylines it returns whole streamlines that
pass through the box; see
[Polylines and streamlines](../tutorials/data_types/polylines_streamlines.md).

## 4. Build a pyramid

`build_pyramid` adds coarser levels. Each `(2.0, 1.0)` pair doubles the bin
size and keeps every object. `chunk_scale_factors` doubles the chunk size too:

```python
ds.build_pyramid(factors=[(2.0, 1.0), (2.0, 1.0)], chunk_scale_factors=[2, 2])

for index in ds.levels:
    level = ds.level(index)
    print(index, level.vertex_count, level.scale, level.resolution)

print(ds.read(level=1))
print(ds.resolution(scale=400.0).index)
```

```text
0 100000 (200.0, 200.0, 200.0) (50.0, 50.0, 50.0)
1 1000 (400.0, 400.0, 400.0) (100.0, 100.0, 100.0)
2 125 (800.0, 800.0, 800.0) (200.0, 200.0, 200.0)
ReadResult(kind='point_cloud', vertices=1000)
1
```

`resolution(scale=)` picks a level by chunk size, which is what a viewer
needs.

:::{warning}
A bounding-box query on a coarser level can miss points inside the box.
To get a region at low resolution, read the whole level and filter in memory.
:::

## 5. Validate

Validation levels 1 to 5 check more of the store at each level:

```python
print(ds.validate(level=3).summary())
```

```text
Level 3 validation: PASS
  40 passed, 0 warnings, 0 errors
```

## Next steps

- [Core concepts](../explanation/concepts.md): chunks, bins, fragments,
  objects and levels.
- [Point clouds](../tutorials/data_types/point_clouds.md) and the other
  data-type tutorials: each geometry in depth, including objects and groups.
- [Choosing a layout](../how_to/choose_chunk_and_bin.md): chunk and bin
  sizes for your data.
- [Building pyramids](../tutorials/multiscale/building_pyramids.md): the
  coarsening options.
- [Validation and repair](../how_to/validate_and_repair.md): what each
  validation level checks.
- [Ingesting other formats](../how_to/convert_files.md): LAS, PLY, TRK, SWC,
  OBJ and other files.
- [View stores in Neuroglancer](../how_to/view_in_neuroglancer.md).
- {doc}`API reference <../api/index>`: which modules are supported.
