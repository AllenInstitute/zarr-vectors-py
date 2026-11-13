# Building multi-resolution pyramids

A pyramid stores the same data at coarser and coarser resolutions, so a
reader loads only the detail it needs. This tutorial builds pyramids for a
point cloud and a set of streamlines with `Dataset.build_pyramid`. The
on-disk result is specified in
[Pyramid construction](../../spec/multiscale/pyramid_construction.md).

## The arguments

| Argument | What it sets |
|----------|--------------|
| `factors` | One `(coarsen, sparsity)` pair per new level. `coarsen` multiplies the bin shape of the level below. `sparsity` thins whole objects, measured against the full object set. |
| `chunk_scale_factors` | One entry per level, a scalar or per-axis tuple, multiplying the chunk shape of the level below. |
| `sparsity_strategy`, `sparsity_seed` | Which objects survive thinning. |
| `method`, `options` | The coarsening method and its settings. |
| `cross_level_storage`, `cross_level_depth` | Which links between levels are written. |

Coarsening replaces the vertices in each occupied bin with their centroid, a
**metavertex**. The only method in this package is `per_object`, the
default, which keeps object ids stable across levels. The companion package
`zarr-vectors-tools` registers more methods and strategies when imported.

## A point cloud pyramid

```python
import numpy as np
import zarr_vectors as zv

rng = np.random.default_rng(0)
positions = rng.uniform(0, 2000, (500_000, 3)).astype(np.float32)
intensity = rng.uniform(0, 1, 500_000).astype(np.float32)

ds = zv.create("synchrotron.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (2000.0, 2000.0, 2000.0)),
    kind="point_cloud",
    vertex_attributes={"intensity": zv.AttributeSpec(dtype="float32")},
    layout=zv.Layout(cells=10),
))
ds.add_points(positions, attributes={"intensity": intensity})

ds.build_pyramid(
    factors=[(2.0, 1.0), (2.0, 1.0), (2.0, 1.0)],
    chunk_scale_factors=[2, 2, 2],
)
for index in ds.levels:
    level = ds.level(index)
    print(index, level.vertex_count, level.scale, level.resolution,
          level.attribute_names("vertex"))
```

```text
0 500000 (200.0, 200.0, 200.0) (50.0, 50.0, 50.0) ('intensity',)
1 8000 (400.0, 400.0, 400.0) (100.0, 100.0, 100.0) ()
2 1000 (800.0, 800.0, 800.0) (200.0, 200.0, 200.0) ()
3 125 (1600.0, 1600.0, 1600.0) (400.0, 400.0, 400.0) ()
```

`scale` is the chunk shape and `resolution` the bin shape. A coarsen factor
`c` gives at most `1/c³` as many vertices as the level below in 3-D. Here
every bin is occupied, so each level holds one vertex per bin. In sparse data
the count is lower.

Coarse levels carry no vertex attributes. Where the data has objects, they
keep the object index and the object attributes.

Set `chunk_scale_factors` to match the coarsen factors. Without it every level
keeps the level-0 chunk shape. `Dataset.resolution(scale=...)` picks the level
whose chunk shape is closest, so it would then always return level 0:

```python
print([ds.resolution(scale=s).index for s in (200.0, 400.0, 800.0, 1600.0)])
```

```text
[0, 1, 2, 3]
```

## A streamline pyramid

Streamlines are coarsened and also thinned: whole streamlines are dropped at
coarser levels.

```python
starts = rng.uniform(200.0, 800.0, size=(2000, 3))
streamlines = [
    (start + rng.normal(0.0, 6.0, size=(60, 3)).cumsum(axis=0)).astype(np.float32)
    for start in starts
]

bundle = zv.create("bundle.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    kind="polyline",
    layout=zv.Layout(cells=10),
))
bundle.add_polylines(streamlines, streamlines=True)

bundle.build_pyramid(
    factors=[(2.0, 1.0), (2.0, 4.0), (2.0, 16.0)],
    chunk_scale_factors=[2, 2, 2],
    sparsity_seed=42,
)
for index in bundle.levels:
    print(index, bundle.level(index).objects.count)
```

```text
0 2000
1 2000
2 500
3 38
```

The sparsity factor is measured against the full object set, not the level
below. So escalate it from level to level: `4.0` then `16.0` select 500 and
125 objects. An object has geometry at a level only if it also survived at
the level below, so level 3 holds fewer than 125. A kept streamline has the
same object id at every level. The selection is `"random"`, the only strategy
in this package; `sparsity_seed` makes it repeatable.
[Sparsity](../../spec/multiscale/sparsity.md) describes the model.

## Links between levels

`build_pyramid` links each vertex to its metavertex on the level above (the
`+1` family, stored at the finer level) and back (the `-1` family, at the
coarser level). `cross_level_storage` sets which directions are written:

| Mode | `+N` at the finer level | `-N` at the coarser level |
|------|-------------------------|---------------------------|
| `"explicit"` (default) | yes | yes |
| `"implicit"` | yes | no; readers reverse `+N` |
| `"none"` | no | no |

`cross_level_depth` sets how many levels a link may span: `1` (default) links
adjacent levels, `2` also links two levels up, `-1` links every pair, and `0`
writes none. [Links](../../spec/object_model/links.md) gives the array layout.

## Removing levels

`build_pyramid` writes only levels that do not exist yet. Remove levels from
the top down with `remove_resolution_level`, then rewrite the root
`multiscales` list from the levels on disk:

```python
from zarr_vectors.building import (
    open_store, remove_resolution_level, write_multiscale_metadata,
)

root = open_store("synchrotron.zarrvectors", mode="r+")
remove_resolution_level(root, level_index=3)
write_multiscale_metadata(root)
print(zv.open("synchrotron.zarrvectors").levels)
```

```text
(0, 1, 2)
```

Removing a level from the middle of a pyramid leaves a store that fails
validation.

## See also

- [Deferred reads and out-of-core access](lazy_loading.md): choosing and
  reading the levels.
- {doc}`../../api/building`: the builder functions.
