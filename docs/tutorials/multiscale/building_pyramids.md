# Building multi-resolution pyramids

A multi-resolution pyramid stores the same dataset at progressively coarser
resolutions, so a viewer or pipeline can read only the density it needs. This
tutorial builds pyramids for a point cloud and a streamline bundle with
`Dataset.build_pyramid`, chooses the cross-level links, and manages the levels
afterwards. The on-disk result is specified in
[Pyramid construction](../../spec/multiscale/pyramid_construction.md).

## The choices `build_pyramid` takes

| Argument | What you choose |
|----------|-----------------|
| `factors` | One `(coarsen, sparsity)` tuple per new level. `coarsen` multiplies the bin shape of the level below, so factors compound. `sparsity` thins whole objects and is measured against the full object set, so it does not compound. |
| `chunk_scale_factors` | One entry per level, a scalar or a per-axis tuple, multiplying the chunk shape of the level below. |
| `sparsity_strategy`, `sparsity_seed` | Which objects survive thinning. |
| `method`, `options` | The coarsening implementation and its own settings. |
| `cross_level_storage`, `cross_level_depth` | Which links between levels are written. |

Coarsening replaces the vertices in each occupied bin with their centroid, a
**metavertex**. There is no choice of aggregation.

The available methods come from a registry:

```python
import zarr_vectors as zv

print(zv.coarsen_methods())
```

```text
('per_object',)
```

`per_object` is the default and keeps object IDs stable across levels. The
companion package `zarr-vectors-tools` registers further methods and selection
strategies when it is imported.

## A point cloud pyramid

```python
import numpy as np
import zarr_vectors as zv

rng = np.random.default_rng(0)
positions = rng.uniform(0, 2000, (500_000, 3)).astype(np.float32)
intensity = rng.uniform(0, 1, 500_000).astype(np.float32)
label = rng.integers(0, 16, 500_000).astype(np.int32)

ds = zv.create("synchrotron.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (2000.0, 2000.0, 2000.0)),
    kind="point_cloud",
    vertex_attributes={
        "intensity": zv.AttributeSpec(dtype="float32"),
        "label": zv.AttributeSpec(dtype="int32", categorical=True),
    },
    layout=zv.Layout(cells=10),
))
ds.add_points(positions, attributes={"intensity": intensity, "label": label})

report = ds.build_pyramid(
    factors=[(2.0, 1.0), (2.0, 1.0), (2.0, 1.0)],   # bins 2x, 4x, 8x level 0
    chunk_scale_factors=[2, 2, 2],
)
print(report["levels_created"], [s["vertex_count"] for s in report["level_specs"]])

for index in ds.levels:
    level = ds.level(index)
    print(index, level.vertex_count, level.scale, level.resolution, level.grid.shape)
```

```text
3 [8000, 1000, 125]
0 500000 (200.0, 200.0, 200.0) (50.0, 50.0, 50.0) (11, 11, 11)
1 8000 (400.0, 400.0, 400.0) (100.0, 100.0, 100.0) (6, 6, 6)
2 1000 (800.0, 800.0, 800.0) (200.0, 200.0, 200.0) (3, 3, 3)
3 125 (1600.0, 1600.0, 1600.0) (400.0, 400.0, 400.0) (2, 2, 2)
```

`Level.scale` is the chunk shape and `Level.resolution` the bin shape. With
`cells=10` the level-0 chunk shape is a tenth of the 2000 µm extent; the grid
has 11 cells per axis because the bounds are inclusive and the upper bound
falls on a cell boundary. The handle that built the pyramid sees the new
levels straight away.

### Reading the vertex counts

Coarsening emits one metavertex per occupied bin, so a level holds at most as
many vertices as it has occupied bins:

| Level | Bin shape | Bins across 2000 µm | Vertices |
|-------|-----------|---------------------|----------|
| 0 | 50 µm  | — | 500 000 |
| 1 | 100 µm | 20³ = 8000 | 8 000 |
| 2 | 200 µm | 10³ = 1000 | 1 000 |
| 3 | 400 µm | 5³ = 125 | 125 |

Here every bin is occupied, so each level count is the bin count. In sparse
data most bins are empty and the count follows the input instead.

### Choosing coarsen factors

A coarsen factor `c` divides the number of bins along each axis by `c`, so in
3D a level holds at most `1/c³` as many vertices as the level below: `2.0`
gives at most 8× fewer, `3.0` 27× fewer, `4.0` 64× fewer. The factor applies
to every axis alike.

### Matching `chunk_scale_factors` to `factors`

Without `chunk_scale_factors` every level keeps the level-0 chunk shape. The
bin shapes still coarsen, but `Level.scale` is the same everywhere, so
`Dataset.resolution(scale=...)`, which picks the level whose chunk shape is
closest, returns level 0 for every scale:

```python
flat = zv.create("no_chunk_scale.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (2000.0, 2000.0, 2000.0)),
    kind="point_cloud",
    layout=zv.Layout(cells=10),
))
flat.add_points(positions)
flat.build_pyramid(factors=[(2.0, 1.0), (2.0, 1.0), (2.0, 1.0)])

for index in flat.levels:
    level = flat.level(index)
    print(index, level.vertex_count, level.scale, level.resolution)
print([flat.resolution(scale=s).index for s in (200.0, 400.0, 800.0, 1600.0)])
```

```text
0 500000 (200.0, 200.0, 200.0) (50.0, 50.0, 50.0)
1 8000 (200.0, 200.0, 200.0) (100.0, 100.0, 100.0)
2 1000 (200.0, 200.0, 200.0) (200.0, 200.0, 200.0)
3 125 (200.0, 200.0, 200.0) (400.0, 400.0, 400.0)
[0, 0, 0, 0]
```

An entry can be a per-axis tuple, such as `(2, 2, 1)`, for anisotropic
chunking.

### Attributes on coarse levels

Coarse levels do not carry vertex attributes. They hold vertices, vertex
fragments and cross-level links; where the data has objects, they also hold the
object index and the object attributes of level 0.

```python
print(ds.level(0).attribute_names("vertex"))
print(ds.level(1).attribute_names("vertex"))
```

```text
('intensity', 'label')
()
```

## A streamline pyramid

Streamlines are coarsened into metavertices and also thinned: whole
streamlines are dropped at coarser levels.

```python
import numpy as np
import zarr_vectors as zv

rng = np.random.default_rng(0)
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

report = bundle.build_pyramid(
    factors=[
        (2.0, 1.0),     # level 1: keep every streamline
        (2.0, 4.0),     # level 2: keep 1 in 4
        (2.0, 16.0),    # level 3: keep 1 in 16
    ],
    chunk_scale_factors=[2, 2, 2],
    sparsity_strategy="random",
    sparsity_seed=42,
)
for index, spec in enumerate(report["level_specs"], start=1):
    print(index, spec["vertex_count"], spec["objects_kept"], spec["source_objects"])
```

```text
1 2721 2000 2000
2 397 500 2000
3 37 125 2000
```

A streamline kept at level 3 has the same object ID it had at level 0, so the
same object can be followed across resolutions.

### Choosing thinning factors

`source_objects` is 2000 at every level: the sparsity factor is measured
against the full object set, not against the level below. Escalate it from
level to level (`1.0`, `4.0`, `16.0` selects 2000, 500 and 125 objects).
Passing `4.0` twice with the same seed selects the same 500 objects twice.

`objects_kept` is what the strategy selected. An object only has geometry at a
level if it also survived at the level below, so fewer can be present:

```python
for index in bundle.levels:
    level = bundle.level(index)
    print(index, level.vertex_count, level.objects.count, level.objects.slots)
```

```text
0 120000 2000 2000
1 2721 2000 2000
2 397 500 2000
3 37 38 2000
```

`objects.slots` stays at 2000 because dropped objects keep empty slots. Use
`objects.count`, or `objects.ids()`, for the objects a level holds. The
selection model is described in [Sparsity](../../spec/multiscale/sparsity.md).

### Selection strategies and methods

`sparsity_strategy="random"` is the only strategy in this package; it is the
default, and `sparsity_seed` makes it reproducible. Naming a strategy or
method that is not registered raises a `ValueError` that says where to get it:

```python
small = zv.create("errors.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    kind="polyline",
))
small.add_polylines(streamlines[:10], streamlines=True)

for kwargs in ({"factors": [(2.0, 4.0)], "sparsity_strategy": "spatial_coverage"},
               {"factors": [(2.0, 1.0)], "method": "graph_aware"}):
    try:
        small.build_pyramid(**kwargs)
    except ValueError as error:
        print(error)
```

```text
object-selection strategy 'spatial_coverage' is not available in core (core provides only ['random']). It is provided by zarr-vectors-tools; install it with `pip install zarr-vectors-tools` (it registers its strategies on import). Registered: [].
coarsen method 'graph_aware' is not available in core (core provides only ['per_object']). It is provided by zarr-vectors-tools; install it with `pip install zarr-vectors-tools` (it registers its strategies on import). Registered: [].
```

The strategy name is only checked when a level thins objects, that is when
its sparsity factor is above 1 and the store holds more than one object. On a
point cloud a misspelt `sparsity_strategy` is ignored.

`options={...}` passes settings to a registered method; `per_object` takes
none and raises if given any. A package of your own registers methods and
strategies with `zarr_vectors.building.register_coarsen_strategy` and
`register_selection_strategy`.

## Links between levels

`build_pyramid` writes links from each vertex to its metavertex on the next
level up (the `+1` family at the finer level) and back down (the `-1` family at
the coarser level). The layout of the link arrays is in
[Links](../../spec/object_model/links.md).

This helper builds a small four-level store with the given options and lists
the link families written at each level:

```python
import os
import shutil

import numpy as np
import zarr_vectors as zv


def rebuild(name, **options):
    """Build a fresh four-level store and list its link families per level."""
    shutil.rmtree(name, ignore_errors=True)
    rng = np.random.default_rng(1)
    ds = zv.create(name, schema=zv.Schema(
        bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
        kind="point_cloud",
        layout=zv.Layout(cells=5),
    ))
    ds.add_points(rng.uniform(0, 1000, (20_000, 3)).astype(np.float32))
    ds.build_pyramid(
        factors=[(2.0, 1.0), (2.0, 1.0), (2.0, 1.0)],
        chunk_scale_factors=[2, 2, 2],
        **options,
    )
    families = {}
    for level in sorted(entry for entry in os.listdir(name) if entry.isdigit()):
        links = os.path.join(name, level, "links")
        families[int(level)] = (
            sorted(entry for entry in os.listdir(links) if entry != "zarr.json")
            if os.path.isdir(links) else []
        )
    return families, sorted(zv.open(name).capabilities)
```

### Storage modes

`cross_level_storage` decides which directions are written:

| Mode | `+N` at the finer level | `-N` at the coarser level |
|------|-------------------------|---------------------------|
| `"explicit"` (default) | yes | yes |
| `"implicit"` | yes | no; readers reverse `+N` |
| `"none"` | no | no |

```python
for mode in ("explicit", "implicit", "none"):
    families, capabilities = rebuild(f"xl_{mode}.zarrvectors", cross_level_storage=mode)
    print(f"{mode:9} {families} {capabilities}")
```

```text
explicit  {0: ['+1'], 1: ['+1', '-1'], 2: ['+1', '-1'], 3: ['-1']} ['multiscale_links', 'shared_fragments']
implicit  {0: ['+1'], 1: ['+1'], 2: ['+1'], 3: []} ['multiscale_links', 'shared_fragments']
none      {0: [], 1: [], 2: [], 3: []} ['shared_fragments']
```

### Depth

`cross_level_depth` sets how many levels a link may span. The default `1`
links adjacent levels only; `2` also links each vertex straight to its
metavertex two levels up; `-1` links every pair of levels; `0` writes no
links between levels.

```python
print(rebuild("xl_depth2.zarrvectors", cross_level_depth=2)[0])
print(rebuild("xl_all.zarrvectors", cross_level_depth=-1)[0])
print(rebuild("xl_depth0.zarrvectors", cross_level_depth=0))
```

```text
{0: ['+1', '+2'], 1: ['+1', '+2', '-1'], 2: ['+1', '-1', '-2'], 3: ['-1', '-2']}
{0: ['+1', '+2', '+3'], 1: ['+1', '+2', '-1'], 2: ['+1', '-1', '-2'], 3: ['-1', '-2', '-3']}
({0: [], 1: [], 2: [], 3: []}, ['shared_fragments'])
```

### Reading the links back

Link arrays are read through the builder surface,
{doc}`zarr_vectors.building <../../api/building>`:

```python
from zarr_vectors.building import (
    get_resolution_level,
    list_link_deltas,
    list_link_offsets,
    open_store,
    read_links,
)

root = open_store("synchrotron.zarrvectors", mode="r")

for index in (0, 1):
    level_group = get_resolution_level(root, index)
    deltas = list_link_deltas(level_group)
    print(index, deltas, {d: list_link_offsets(level_group, d)[:3] for d in deltas})

up = read_links(get_resolution_level(root, 0), delta=1)
print(len(up), up[0])
```

```text
0 [1] {1: ['0.0.0']}
1 [-1, 1] {-1: ['+1.+1.+1', '+1.+1.0', '+1.0.+1'], 1: ['0.0.0']}
500000 (((0, 0, 0), 0), ((0, 0, 0), 0))
```

Each record is a pair of `(chunk_coords, fragment_index)` endpoints. Upward
links stay in the all-zero offset because a metavertex lies in the coarse cell
that covers its source vertices; downward links spread over several offsets.

## Adding one level at a time

`build_pyramid` builds every level in one call, and the target level must not
exist yet. To add a single level on top of an existing pyramid, the only
function is the internal `coarsen_level`, which may change between releases:

```python
from zarr_vectors.multiresolution.coarsen import coarsen_level  # internal

print(coarsen_level(
    "synchrotron.zarrvectors",
    source_level=3,
    target_level=4,
    coarsen_factor=2.0,
    chunk_scale_factor=(2, 2, 1),
))
```

```text
{'vertex_count': 27, 'object_count': 1, 'objects_kept': 1, 'source_objects': 1, 'method': 'per_object', 'preserves_object_ids': True, 'shared_fragments': True}
```

`coarsen_level` writes no links between levels. To recompute every level
above a changed one, see
[Validation and repair](../io/validation_and_repair.md).

## Listing and removing levels

```python
from zarr_vectors.building import (
    list_resolution_levels,
    open_store,
    read_multiscale_metadata,
    remove_resolution_level,
    write_multiscale_metadata,
)

root = open_store("synchrotron.zarrvectors", mode="r+")
print(list_resolution_levels(root))

remove_resolution_level(root, level_index=4)
print(list_resolution_levels(root))
print([d["path"] for d in read_multiscale_metadata(root)[0]["datasets"]])

write_multiscale_metadata(root)
print([d["path"] for d in read_multiscale_metadata(root)[0]["datasets"]])
print(zv.open("synchrotron.zarrvectors").levels)
```

```text
[0, 1, 2, 3, 4]
[0, 1, 2, 3]
['0', '1', '2', '3', '4']
['0', '1', '2', '3']
(0, 1, 2, 3)
```

`remove_resolution_level` deletes the level's arrays but leaves its entry in
the root `multiscales` list; `write_multiscale_metadata` rewrites that list
from the levels on disk. Remove levels from the top down: removing a level
from the middle of a pyramid leaves a store that fails validation.

## Reading the levels

```python
ds = zv.open("synchrotron.zarrvectors")

for index in ds.levels:
    print(index, ds.read(level=index).vertex_count)

print(ds.resolution(scale=1600.0).index, ds.resolution(scale=200.0).index)

detail = ds.select(level=0, bbox=((400.0, 400.0, 400.0), (600.0, 600.0, 600.0))).read()
print(detail, detail.vertex_count)
```

```text
0 500000
1 8000
2 1000
3 125
3 0
ReadResult(kind='point_cloud', vertices=533, attributes=['intensity', 'label']) 533
```

`ds.read(level=i)` and `ds.level(i).read()` are the same read.
`ds.resolution(scale=...)` picks a level by chunk size. Bounding-box queries
on coarse levels have a limitation described in
[Deferred reads and out-of-core access](lazy_loading.md#choose-a-level-before-reading).

## See also

- [Deferred reads and out-of-core access](lazy_loading.md): choosing and reading
  the levels built here.
- {doc}`../../api/api`: `Dataset.build_pyramid`, `Level`, `Query`.
- {doc}`../../api/building`: the builder functions used above.
- [Pyramid construction](../../spec/multiscale/pyramid_construction.md) and
  [Sparsity](../../spec/multiscale/sparsity.md): the format side.
