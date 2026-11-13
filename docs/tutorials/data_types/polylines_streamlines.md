# Polylines and streamlines

In this tutorial you write a set of tractography streamlines, with per-vertex
and per-streamline values and named bundles, to a new store; read them back
whole, by bundle and by region; add coarser levels and validate the result.
The same calls write any ordered paths, such as vessel centrelines or cell
tracks, as polylines.

The steps run in order as one Python session. The
[quickstart](../../getting_started/quickstart.md) introduces the same calls
more briefly, and [Concepts](../../explanation/concepts.md) explains
chunks, bins, objects and groups.

## 1. Create the store

Each polyline is one object. Declare the per-vertex attributes and the
per-object attributes in the `Schema`:

```python
import numpy as np
import zarr_vectors as zv

tracts = zv.create("tracts.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    axes=(zv.Axis("x", unit="micrometer"),
          zv.Axis("y", unit="micrometer"),
          zv.Axis("z", unit="micrometer")),
    kind="polyline",
    vertex_attributes={"fa": zv.AttributeSpec(dtype="float32")},
    object_attributes={"length": zv.AttributeSpec(dtype="float32"),
                       "mean_fa": zv.AttributeSpec(dtype="float32")},
    expected=zv.SizeHints(n_vertices=40_000, n_objects=1000),
    layout=zv.Layout(cells=5),
))
print(tracts.level(0).scale, tracts.level(0).resolution)
```

```text
(200.0, 200.0, 200.0) (50.0, 50.0, 50.0)
```

[Choosing a layout](../../how_to/choose_chunk_and_bin.md) explains how to pick
the chunk and bin shapes.

## 2. Write the streamlines

Pass the streamlines as a list of `(N_k, 3)` arrays. A per-vertex attribute is
also a list, with one array per streamline and as many values as that
streamline has vertices. A per-object attribute has one value per streamline.
`groups=` maps an integer group id to the indices of its streamlines.

```python
rng = np.random.default_rng(0)
n = 1000
starts = rng.uniform(300.0, 700.0, size=(n, 3))
streamlines = [
    (s + rng.normal(0.0, 6.0, size=(int(k), 3)).cumsum(axis=0)).astype(np.float32)
    for s, k in zip(starts, rng.integers(20, 60, size=n))
]
fa = [rng.uniform(0.1, 0.9, len(s)).astype(np.float32) for s in streamlines]
lengths = np.array([np.linalg.norm(np.diff(s, axis=0), axis=1).sum()
                    for s in streamlines], dtype=np.float32)
mean_fa = np.array([f.mean() for f in fa], dtype=np.float32)

print(tracts.add_polylines(
    streamlines,
    attributes={"fa": fa},
    object_attributes={"length": lengths, "mean_fa": mean_fa},
    groups={0: range(0, 300), 1: range(300, 600), 2: range(600, 1000)},
    streamlines=True,
))
print(tracts.kinds)
```

```text
{'polyline_count': 1000, 'vertex_count': 39019, 'chunk_count': 28, 'cross_chunk_link_count': 2591, 'group_count': 3}
('polyline', 'streamline')
```

`streamlines=True` records that the store holds tractography by adding
`"streamline"` to the store's kinds; the level and the read results keep the
kind `"polyline"`. To make `"streamline"` the level's kind as well, create the
store with `kind="streamline"`.

Free-form acquisition details, such as the step size or seeding strategy, go
in the store's header registry, one dictionary per source format:

```python
tracts.headers.add("trk", {"step_size": 0.5, "seeding": "wm_mask"})
print(tracts.headers.available_formats, tracts.headers.get("trk")["step_size"])
```

```text
['trk'] 0.5
```

## 3. Read it back

`read()` returns a `ReadResult`. For polylines, `positions` holds every vertex
and `polylines` splits them into one array per streamline:

```python
r = tracts.read()
print(r)
print(r.part_count, r.positions.shape)
print(r.polylines[0].shape, r.attributes["fa"].shape)
print(np.allclose(r.polylines[0], streamlines[0]))
```

```text
ReadResult(kind='polyline', vertices=39019, parts=1000, attributes=['fa'])
1000 (39019, 3)
(55, 3) (39019,)
True
```

`part_objects` gives the object id of each streamline in the result. To read
one streamline, or a few, index the level's objects:

```python
one = tracts.level(0).objects[42]
print(one, np.allclose(one.polylines[0], streamlines[42]))
print(tracts.level(0).objects[[0, 42, 99]].part_count)
```

```text
ReadResult(kind='polyline', vertices=51, attributes=['fa']) True
3
```

## 4. Query a region

A bounding-box query returns whole streamlines: every streamline with a vertex
in a chunk the box touches, unclipped.

```python
box = ((400.0, 400.0, 400.0), (600.0, 600.0, 600.0))
q = tracts.select(bbox=box)
print(q.explain())
in_region = q.read()
print(in_region.part_count, q.cells())

inside = [((p >= 400.0) & (p <= 600.0)).all(axis=1).any() for p in in_region.polylines]
print(sum(inside))
```

```text
polyline read: level 0, bbox [400.0, 400.0, 400.0]..[600.0, 600.0, 600.0]
  via read_polylines(bbox, level)
521 CellSet(8 cell(s))
219
```

Because selection works by chunk, some returned streamlines have no vertex in
the box itself. Filter the result in memory, as above, when you need only
those that do. `count()` counts vertices; use `part_count` for the number of
streamlines.

Object ids narrow a region query further:

```python
ids = in_region.part_objects[:3].tolist()
print(q.select(objects=ids).read())
```

```text
ReadResult(kind='polyline', vertices=87, parts=3, attributes=['fa'])
```

## 5. Groups and per-streamline values

Name the groups so they can be read by name. Naming needs the store open for
writing:

```python
tracts = zv.open("tracts.zarrvectors", mode="r+")
tracts.level(0).groups.name_rows(["CST", "AF", "UF"])
print(tracts.groups)
print(tracts.groups["CST"].read())
print(tracts.select(groups=[1]).read().part_count)
```

```text
GroupCatalog(['CST', 'AF', 'UF'])
ReadResult(kind='polyline', vertices=11641, parts=300, attributes=['fa'])
300
```

A group without a name answers to `group_<id>`, and `groups.by_id(i)` reaches
a group by its integer id.

Per-streamline values can be read without loading any geometry, which lets
you choose streamlines before reading them. `Level.attribute_names("object")`
lists them, and `zarr_vectors.building.read_object_attributes` returns one
value per object id:

```python
from zarr_vectors import building

print(tracts.level(0).attribute_names("object"))
level_group = building.get_resolution_level(
    building.open_store("tracts.zarrvectors", mode="r"), 0)
length = building.read_object_attributes(level_group, "length")
fa_mean = building.read_object_attributes(level_group, "mean_fa")

long_and_bright = np.nonzero((length > 300) & (fa_mean > 0.5))[0]
print(long_and_bright.size, tracts.select(objects=long_and_bright.tolist()).read().part_count)
```

```text
('length', 'mean_fa')
323 323
```

## 6. Add coarser levels

`build_pyramid` writes coarser levels after level 0. Each `factors` entry is a
`(coarsen_factor, sparsity_factor)` pair applied to the level before it. The
coarsen factor doubles the bin size here; a sparsity factor of 2.0 keeps a
random half of the streamlines, and `sparsity_seed=` makes the choice
repeatable.

```python
rep = tracts.build_pyramid(factors=[(2.0, 1.0), (2.0, 2.0)], sparsity_seed=0)
print(rep["levels_created"])
for i in tracts.levels:
    lv = tracts.level(i)
    print(i, lv.resolution, lv.objects.count,
          lv.attribute_names("vertex"), lv.attribute_names("object"))
```

```text
2
0 (50.0, 50.0, 50.0) 1000 ('fa',) ('length', 'mean_fa')
1 (100.0, 100.0, 100.0) 1000 () ('length', 'mean_fa')
2 (200.0, 200.0, 200.0) 500 () ('length', 'mean_fa')
```

Each surviving streamline keeps its object id and is reduced to one vertex
per bin it passes through. Per-object attributes are carried to every
level; per-vertex attributes stay on level 0.
[Building pyramids](../multiscale/building_pyramids.md) covers the other
options.

## 7. Validate

```python
print(tracts.validate(level=4).summary())
```

```text
Level 4 validation: PASS
  53 passed, 0 warnings, 0 errors
```

[Validation and repair](../../how_to/validate_and_repair.md) lists what each level
checks.

To convert TRK, TCK or TRX files to and from a store, use the companion
package `zarr-vectors-tools`.

## Common mistakes

**A flat per-vertex attribute.** A per-vertex attribute passed as one
concatenated array, rather than a list with one array per polyline, fails with
`IndexError: invalid index to scalar variable.`

**Group names at write time.** `groups=` takes integer keys; a string key such
as `{"CST": [...]}` raises `TypeError`. Write integer ids, then name them with
`groups.name_rows`.

**Vertices far apart.** A streamline is found by a region query only through
the chunks that hold its vertices. A streamline whose step is longer than a
chunk can cross a chunk without a vertex in it, and a box inside that chunk
does not return it. Keep the step well below the chunk size.

**Inspecting links for the path.** Within a chunk, consecutive vertices are
connected by their order alone; only connections that cross a chunk boundary
are stored as links. Read a whole streamline with `level.objects[k]`. The
[polyline specification](../../spec/geometry_types/polyline.md) describes the
layout.
