# Point clouds

In this tutorial you write a point cloud with per-point attributes to a new
store, read it back, query a region, add coarser levels and validate the
result. A point cloud is a set of positions with optional per-point values,
such as localisation-microscopy detections, spatial-transcriptomics spots or
lidar returns.

The steps run in order as one Python session. The
[quickstart](../../getting_started/quickstart.md) introduces the same calls
more briefly, and [Concepts](../../getting_started/concepts.md) explains
chunks, bins and levels.

## 1. Create the store

A store starts from a `Schema`: the spatial bounds, the axes and their units,
the geometry kind and the attributes you will write. `Layout(cells=5)` cuts
each axis into five chunks; the chunk and bin shapes are derived from it.

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
        "intensity": zv.AttributeSpec(dtype="float32", unit="absorbance"),
        "label": zv.AttributeSpec(dtype="int32", categorical=True),
        "color": zv.AttributeSpec(dtype="uint8", channels=3),
    },
    expected=zv.SizeHints(n_vertices=100_000),
    layout=zv.Layout(cells=5),
))
print(ds.level(0).scale, ds.level(0).resolution)
```

```text
(200.0, 200.0, 200.0) (50.0, 50.0, 50.0)
```

`Level.scale` is the chunk shape and `Level.resolution` is the bin shape: five
cells across 1,000 µm give 200 µm chunks, and the default four bins per chunk
axis give 50 µm bins. [Choosing a layout](../../how_to/choose_chunk_and_bin.md)
explains how to pick `cells`, `cell_size` and `subcells` for your data.

## 2. Write the points

Pass the positions as an `(N, 3)` array and each attribute as an array with
one row per point. A vector attribute, such as a colour, keeps its trailing
width.

```python
rng = np.random.default_rng(42)
n = 100_000
positions = rng.uniform(0, 1000, (n, 3)).astype(np.float32)
intensity = rng.uniform(0, 1, n).astype(np.float32)
label = rng.integers(0, 8, n).astype(np.int32)
color = rng.integers(0, 256, (n, 3)).astype(np.uint8)

print(ds.add_points(positions, attributes={
    "intensity": intensity, "label": label, "color": color,
}))
print(ds.level(0).vertex_count, ds.level(0).attribute_names("vertex"))
```

```text
{'vertex_count': 100000, 'chunk_count': 125, 'object_count': 0, 'group_count': 0, 'bins_per_chunk': (4, 4, 4)}
100000 ('color', 'intensity', 'label')
```

The write checks each attribute against its declaration: an array whose dtype
or channel count differs from its `AttributeSpec` raises `SchemaConflict`.
Attributes you did not declare are written as given.

## 3. Read it back

`read()` returns a `ReadResult` holding the positions and, for a whole read,
every attribute:

```python
r = ds.read()
print(r)
print(r.positions.shape, r.positions.dtype)
print(r.attributes["intensity"].shape, r.attributes["color"].shape)
print(r.attributes_read, r.complete)
```

```text
ReadResult(kind='point_cloud', vertices=100000, attributes=['color', 'intensity', 'label'])
(100000, 3) float32
(100000,) (100000, 3)
True True
```

To load fewer attributes, name them:

```python
lean = ds.select(attributes=["intensity"]).read()
print(lean.attributes.names())
```

```text
('intensity',)
```

## 4. Query a region

`select()` builds a query without reading anything; `count()` and `read()` run
it. For a point cloud a bounding box is exact: the result holds the points
inside the box and nothing else.

```python
q = ds.select(bbox=((100.0, 100.0, 100.0), (200.0, 200.0, 200.0)))
print(q.explain())
print(q.count())
box = q.read()
print(box.positions.min(axis=0), box.positions.max(axis=0))
print(q.cells())
```

```text
point_cloud read: level 0, bbox [100.0, 100.0, 100.0]..[200.0, 200.0, 200.0]
  via read_points(attribute_names, bbox, level)
109
[100.439674 100.72887  100.454704] [199.90297 198.85411 197.56297]
CellSet(8 cell(s))
```

`explain()` names the reader that will run, and `cells()` lists the chunks the
query touches: eight, because the box meets a chunk boundary on every axis.

A sphere and a row limit narrow a query the same way:

```python
near = ds.select(near=((500.0, 500.0, 500.0), 50.0))
print(near.count())
print(float(np.linalg.norm(near.read().positions - 500.0, axis=1).max()) <= 50.0)

capped = ds.select(bbox=((0.0, 0.0, 0.0), (500.0, 500.0, 500.0))).limit(10).read()
print(capped.vertex_count, capped.truncated, capped.complete)
```

```text
59
True
10 True False
```

A limited read sets `truncated`, and `complete` is `False` to say the result
is not the whole answer.

## 5. Objects and groups

Points can belong to objects, such as the detections of one cell, and objects
to groups. Pass `object_ids=` with one id per point, and `groups=` as a mapping
from an integer group id to a list of object ids:

```python
spots_ds = zv.create("spots.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    kind="point_cloud",
    layout=zv.Layout(cells=5),
))
centres = rng.uniform(100, 900, (10, 3))
spots = np.concatenate([c + rng.normal(0, 20, (100, 3)) for c in centres]).astype(np.float32)
owner = np.repeat(np.arange(10), 100)
spots_ds.add_points(spots, object_ids=owner,
                    groups={0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9]})

print(spots_ds.level(0).objects)
print(spots_ds.level(0).objects[3].vertex_count)
print(spots_ds.groups)
print(spots_ds.select(groups=[1]).read().vertex_count)
```

```text
ObjectCatalog(level=0, count=10, slots=10)
100
GroupCatalog(['group_0', 'group_1'])
500
```

`objects[k]` reads one object's points and `select(groups=[...])` reads the
points of every object in the listed groups. A group without a name answers to
`group_<id>`; the [polylines tutorial](polylines_streamlines.md) shows how to
name groups.

## 6. Add coarser levels

`build_pyramid` writes coarser levels after level 0. Each `factors` entry is a
`(coarsen_factor, sparsity_factor)` pair applied to the level before it, so
`[(2.0, 1.0), (2.0, 1.0)]` gives bins two and four times the size of the
level-0 bins. Points that share a bin are replaced by their centroid.

```python
print(ds.build_pyramid(factors=[(2.0, 1.0), (2.0, 1.0)])["levels_created"])
for i in ds.levels:
    lv = ds.level(i)
    print(i, lv.vertex_count, lv.resolution, lv.attribute_names("vertex"))
```

```text
2
0 100000 (50.0, 50.0, 50.0) ('color', 'intensity', 'label')
1 1000 (100.0, 100.0, 100.0) ()
2 125 (200.0, 200.0, 200.0) ()
```

Per-point attributes stay on level 0. A viewer typically reads a coarse level
whole for an overview, then a level-0 region for detail:

```python
overview = ds.read(level=2)
detail = ds.select(bbox=((400.0, 400.0, 400.0), (600.0, 600.0, 600.0))).read()
print(overview.vertex_count, detail.vertex_count)
```

```text
125 765
```

[Building pyramids](../multiscale/building_pyramids.md) covers the other
options, including larger chunks at coarser levels.

## 7. Validate

`validate` checks the store's structure; a higher level checks more.

```python
print(ds.validate(level=4).summary())
```

```text
Level 4 validation: PASS
  41 passed, 1 warnings, 0 errors
  WARN:  Point cloud but links array exists
```

The warning refers to the `links/` arrays that `build_pyramid` writes to
connect each level's points to the level before and after it; the store reads
correctly.
[Validation and repair](../io/validation_and_repair.md) lists what each level
checks.

To convert LAS, PLY, CSV or XYZ files to and from a store, use the companion
package `zarr-vectors-tools`.

## Common mistakes

**Writing in batches.** Each `add_points` call writes level 0 afresh rather
than appending to it, so a second call replaces points from the first.
Assemble all the points, then write them in one call.

**Points outside the bounds.** A point outside the schema's `bounds` raises
`MetadataError` and nothing is written. Pass `on_out_of_bounds="ignore"` to
drop such points, or `on_out_of_bounds="expand"` to grow the bounds to fit
them.

**An attribute of the wrong length.** Each attribute needs exactly one row per
point. A shorter array fails during the write with an `IndexError` such as
`index 99 is out of bounds for axis 0 with size 99`; check the pipeline that
built the arrays for dropped or duplicated rows.
