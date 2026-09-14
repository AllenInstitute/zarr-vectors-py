# Building multi-resolution pyramids

A multi-resolution pyramid stores the same dataset at progressively coarser
spatial resolutions. Viewers and analysis pipelines select the appropriate
level based on viewport size, memory budget, or query scale — loading only the
data density they need.

`Dataset.build_pyramid` is the supported entry point, on the
{doc}`api surface <../../api/api>` re-exported from the top-level package.
The coarsening itself lives in `zarr_vectors.multiresolution`, which is
internal and changes without notice; nothing on this page imports from it
except the one section that documents a gap.

This tutorial covers pyramid construction for point clouds, streamlines and
graphs; cross-pyramid-level link materialisation; and level bookkeeping. See
the [Pyramid construction](../../spec/multiscale/pyramid_construction.md)
spec page for the algorithm and on-disk layout.

---

## Concepts recap

**Coarsen factor** scales the supervoxel bin size. It is a **per-level ratio
against the level below**, so factors compound: `[(2, 1), (2, 1), (2, 1)]`
bins at 2×, 4× and 8× the root bin shape.

**Sparsity factor** thins discrete objects (streamlines, skeletons, meshes) at
coarser levels. Unlike the coarsen factor it is **absolute, not compounding**:
it is measured against the store's full object set, so `4.0` at two successive
levels selects the same objects twice. See
[Choosing thinning factors](#choosing-thinning-factors) below.

**Aggregation** is not configurable. Coarsening aggregates a bin's source
vertices into their **centroid** (a metavertex); there is no `agg_mode`-style
choice of `mean` / `sum` / `mode` / `min` / `max`.

**Coarsening method** (`method`) selects the coarsening implementation. Ask
the package rather than guessing, because the registry knows and an exception
is a poor way to find out:

```python
import zarr_vectors as zv

print(zv.coarsen_methods())
```

```text
('per_object',)
```

Core ships exactly one: `per_object` (the default) — OID-stable, with
metavertices shared between objects. Anything else in that tuple was
registered on import by **`zarr-vectors-tools`**.

**Cross-level links** are edges from a fine-level vertex to its coarse-level
parent metanode, materialised at each adjacent level pair. Controlled by
`cross_level_storage` and `cross_level_depth`.

---

## Point cloud pyramids

Point clouds use spatial coarsening only; `sparsity_factor` is ignored,
because there are no objects to thin.

### A three-level pyramid

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
    expected=zv.SizeHints(n_vertices=500_000),
    layout=zv.Layout(cells=10),
))
ds.add_points(positions, attributes={"intensity": intensity, "label": label})
print(ds.level(0).scale, ds.level(0).resolution)

report = ds.build_pyramid(
    factors=[
        (2.0, 1.0),     # level 1: bins 2× the root bin
        (2.0, 1.0),     # level 2: 4×
        (2.0, 1.0),     # level 3: 8×
    ],
    chunk_scale_factors=[2, 2, 2],
)
print(report["levels_created"], [s["vertex_count"] for s in report["level_specs"]])
```

```text
(200.0, 200.0, 200.0) (50.0, 50.0, 50.0)
3 [8000, 1000, 125]
```

`zv.Layout(cells=10)` divides the 2000 µm bounds into ten cells per axis, so
level 0 has a 200 µm chunk shape and a 50 µm bin shape. Re-open the dataset
after building — the handle that built the pyramid keeps stale level metadata:

```python
ds = zv.open("synchrotron.zarrvectors")
for index in ds.levels:
    level = ds.level(index)
    print(index, level.vertex_count, level.scale, level.resolution, level.grid.shape)
```

```text
0 500000 (200.0, 200.0, 200.0) (50.0, 50.0, 50.0) (10, 10, 10)
1 8000 (400.0, 400.0, 400.0) (100.0, 100.0, 100.0) (5, 5, 5)
2 1000 (800.0, 800.0, 800.0) (200.0, 200.0, 200.0) (3, 3, 3)
3 125 (1600.0, 1600.0, 1600.0) (400.0, 400.0, 400.0) (2, 2, 2)
```

### Reading those numbers

The reduction is not `500000 / 8^N`. Coarsening emits **one metavertex per
occupied bin**, so once the bins are saturated the count is set by the bin
grid, not by the input:

| Level | Bin shape | Bins across the 2000 µm volume | Vertices |
|-------|-----------|--------------------------------|----------|
| 0 | 50 µm  | —      | 500 000 |
| 1 | 100 µm | 20³ = 8000 | 8 000 |
| 2 | 200 µm | 10³ = 1000 | 1 000 |
| 3 | 400 µm |  5³ =  125 |   125 |

With 500 000 points spread over 8000 level-1 bins, every bin holds ~62 points
and every one of them collapses to a single centroid — so level 1 is exactly
the bin count. Sparse data behaves the other way: where bins are mostly empty
the count tracks the input and the reduction really is ~`coarsen ** ndim`.
Either way the level count is `min(source vertices, occupied bins)`, and you
can predict the ceiling from `Level.resolution` before building.

### Choosing coarsen factors

Each `(coarsen, sparsity)` tuple produces one level from the one below. For
3D data the per-level reduction is at most `coarsen ** 3`:

| Per-level target reduction | `coarsen_factor` | 3D effect |
|----------------------------|------------------|-----------|
| 8×    | 2.0 | each axis halved |
| 27×   | 3.0 | each axis thirded |
| 64×   | 4.0 | each axis quartered |

`factors` is isotropic: each entry scales every axis by the same
`coarsen_factor`.

### Always pass `chunk_scale_factors`

`chunk_scale_factors` takes one entry per level — a scalar, or a per-axis
tuple — multiplying the source level's chunk shape to get the target's. Leave
it out and coarsening still works, but every level inherits the root chunk
shape, so `Level.scale` is identical everywhere and the level-picking helper a
viewer relies on has nothing to distinguish levels by:

```python
# the same 500 000 points, built with factors= but no chunk_scale_factors
flat = zv.open("no_chunk_scale.zarrvectors")
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

The vertex counts are right and `resolution` coarsens correctly, but every
`resolution(scale=...)` query collapses to level 0. Match
`chunk_scale_factors` to `factors` unless you have a reason not to.

### A note on attributes

There is no per-attribute aggregation setting and no `agg_mode` parameter.
Every bin's vertices collapse to a centroid metavertex under the one built-in
`per_object` method. A coarsened level also does not inherit
`vertex_attributes`: it gets `vertices`, `vertex_fragments`, its cross-level
`links/`, and — where the geometry has objects — `object_index` plus the
inherited `object_attributes`. Not `vertex_attributes`, `groups`,
`group_attributes`, `link_attributes` or `link_fragments`.

```python
print(ds.level(0).attribute_names("vertex"))
print(ds.level(1).attribute_names("vertex"))
```

```text
('intensity', 'label')
()
```

If you need categorical labels or counts carried up the pyramid, that
behaviour is not in core.

---

## Streamline and polyline pyramids

Polylines and streamlines use both spatial coarsening (vertex metanodes) and
object thinning (dropping whole streamlines at coarser levels).

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
    expected=zv.SizeHints(n_vertices=120_000, n_objects=2000),
    layout=zv.Layout(cells=10),
))
bundle.add_polylines(streamlines, streamlines=True)

report = bundle.build_pyramid(
    factors=[
        (2.0,  1.0),     # L1: coarser vertices, all streamlines kept
        (2.0,  4.0),     # L2: coarser again, keep 1 streamline in 4
        (2.0, 16.0),     # L3: coarser again, keep 1 in 16
    ],
    chunk_scale_factors=[2, 2, 2],
    method="per_object",
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

`per_object` preserves OIDs: a streamline kept at level 3 has the same OID it
had at level 0, so "the same object" is trackable across resolutions — which
is what Neuroglancer drill-down and ID-preserving analytics need.

### Choosing thinning factors

Note `source_objects` in that output: it is 2000 at every level, not the
previous level's count. **The sparsity factor is absolute** — measured against
the store's full object set — where the coarsen factor is relative to the
level below. Passing `4.0` at two consecutive levels with the same seed
therefore selects the same 500 objects twice and thins nothing the second
time. Escalate the factor instead: `1.0`, `4.0`, `16.0` gives 2000, 500 and
125 selected objects.

`objects_kept` is what the selection strategy chose. What actually carries
geometry can be smaller, because an object can only survive at a level if it
survived at the level below:

```python
bundle = zv.open("bundle.zarrvectors")
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

Level 3 selected 125 ids out of the 2000-id space, but only the 38 of them
that were also present at level 2 have anything to hold. `slots` stays at 2000
at every level — dropped objects leave addressable but empty slots — so use
`Level.objects.count`, or `ids(present=True)`, when you need what is really
there. See [Sparsity](../../spec/multiscale/sparsity.md) for the model.

### Sparsity strategies

`sparsity_strategy` picks which objects survive. **Core ships exactly one:
`"random"`** (the default), with `sparsity_seed` for reproducibility.

Non-random strategies — spatial-coverage, length-ranked, attribute-ranked,
point-thinning — live in **`zarr-vectors-tools`**, which registers them on
import. Requesting one without that package raises rather than silently
falling back:

```pycon
>>> bundle.build_pyramid(factors=[(2.0, 4.0)], sparsity_strategy="spatial_coverage")
ValueError: object-selection strategy 'spatial_coverage' is not available in core
(core provides only ['random']). It is provided by zarr-vectors-tools; install it
with `pip install zarr-vectors-tools` (it registers its strategies on import).
Registered: [].
```

`method=` behaves the same way, and its message names the coarseners instead:

```pycon
>>> bundle.build_pyramid(factors=[(2.0, 1.0)], method="graph_aware")
ValueError: coarsen method 'graph_aware' is not available in core (core provides
only ['per_object']). It is provided by zarr-vectors-tools; install it with
`pip install zarr-vectors-tools` (it registers its strategies on import).
Registered: [].
```

Pass a registered strategy's own knobs as `options={...}`. A strategy package
registers itself through `zarr_vectors.building.register_coarsen_strategy` and
`register_selection_strategy`, both of which are supported — so a third-party
coarsener does not need an internal import either.

Neither error fires on a store with no objects: a point cloud ignores the
sparsity path entirely, so a misspelled `sparsity_strategy` goes unnoticed
there. Check the name against a store that has objects.

---

## Cross-pyramid-level links

`build_pyramid` materialises edges between fine vertices and their coarse-level
parent metanodes, under `links/<delta>/<offsets>/`. `<delta>` is how many
pyramid levels the record spans — `+1` at the finer level for drill-up, `-1` at
the coarser level for drill-down — and `<offsets>` says where the other
endpoint sits relative to the source chunk. There is no separate cross-chunk
array: a cross-chunk link is simply one with non-zero offsets. See
[Links](../../spec/object_model/links.md) for the on-disk layout.

### Watching what gets written

The families are directories, so the easiest way to see what a set of options
produced is to look. A small rebuild-and-inspect helper makes the variants
below comparable:

```python
import os
import shutil

import numpy as np
import zarr_vectors as zv

def rebuild(name, **options):
    """Build a fresh four-level store and report its <delta> families."""
    shutil.rmtree(name, ignore_errors=True)
    rng = np.random.default_rng(1)
    ds = zv.create(name, schema=zv.Schema(
        bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
        kind="point_cloud",
        expected=zv.SizeHints(n_vertices=20_000),
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

### Default: `±1`, explicit

```python
print(rebuild(
    "xl_explicit.zarrvectors",
    cross_level_depth=1,               # default
    cross_level_storage="explicit",    # default
))
```

```text
({0: ['+1'], 1: ['+1', '-1'], 2: ['+1', '-1'], 3: ['-1']}, ['multiscale_links', 'shared_fragments'])
```

The `+1` family at the fine level always lands in the all-zero offsets segment
(`links/+1/0.0.0/`), because a parent metanode sits in the cell that covers its
children. The `-1` family at the coarse level spreads across offsets — one
coarse cell's children live in several fine cells — so it has segments like
`0.0.+1` and `+1.+1.+1` alongside `0.0.0`.

### Storage modes

`cross_level_storage` decides which directions are written.

| Mode | `+N` at fine level | `-N` at coarse level |
|------|--------------------|----------------------|
| `explicit` (default) | yes | yes |
| `implicit` | yes | no — readers reconstruct by flipping `+N` |
| `none`     | no  | no  |

```python
for mode in ("explicit", "implicit", "none"):
    families, _ = rebuild(f"xl_{mode}.zarrvectors", cross_level_storage=mode)
    print(f"{mode:9} {families}")
```

```text
explicit  {0: ['+1'], 1: ['+1', '-1'], 2: ['+1', '-1'], 3: ['-1']}
implicit  {0: ['+1'], 1: ['+1'], 2: ['+1'], 3: []}
none      {0: [], 1: [], 2: [], 3: []}
```

Use `"implicit"` to roughly halve the link footprint; readers then find the
`+N` array at the target level and flip endpoints to drill down.

### Multi-step deltas

`cross_level_depth` governs the deltas of magnitude 2 and above. Adjacent `±1`
arrays are emitted inline while each level is coarsened; a post-build pass then
composes the parent maps to produce the deeper ones.

```python
print(rebuild("xl_depth2.zarrvectors", cross_level_depth=2)[0])
print(rebuild("xl_all.zarrvectors", cross_level_depth=-1)[0])
```

```text
{0: ['+1', '+2'], 1: ['+1', '+2', '-1'], 2: ['+1', '-1', '-2'], 3: ['-1', '-2']}
{0: ['+1', '+2', '+3'], 1: ['+1', '+2', '-1'], 2: ['+1', '-1', '-2'], 3: ['-1', '-2', '-3']}
```

`depth=2` composes parent maps across two coarsening steps, so a level-0 vertex
points straight at its level-2 metanode in a single hop, with no chained
lookup. `cross_level_depth=-1` walks every available level pair.

### Opting out

**`cross_level_depth=0` does not opt out.** Depth only governs the composed
`|N| ≥ 2` arrays; the adjacent `±1` families are written inline by the
coarsener and are controlled by `cross_level_storage`. Passing `depth=0` leaves
every `±1` array on disk and merely skips the finalize pass — which also means
the store never gets the `multiscale_links` capability stamped, so a reader
that checks `ds.supports("multiscale_links")` decides the links are absent
while they are sitting right there:

```python
print(rebuild("xl_depth0.zarrvectors", cross_level_depth=0))
print(rebuild("xl_off.zarrvectors", cross_level_storage="none"))
```

```text
({0: ['+1'], 1: ['+1', '-1'], 2: ['+1', '-1'], 3: ['-1']}, ['shared_fragments'])
({0: [], 1: [], 2: [], 3: []}, ['shared_fragments'])
```

`cross_level_storage="none"` is what actually skips cross-level navigation —
it saves the arrays and the post-build pass both.

### Reading the link families back

Inspecting `links/` is a physical question, so it belongs to the
{doc}`builder surface <../../api/building>`:

```python
from zarr_vectors.building import (
    open_store,
    get_resolution_level,
    list_link_deltas,
    list_link_offsets,
    read_links,
)

root = open_store("synchrotron.zarrvectors", mode="r")

for index in (0, 1):
    level_group = get_resolution_level(root, index)
    deltas = list_link_deltas(level_group)
    print(index, deltas, {d: list_link_offsets(level_group, d)[:3] for d in deltas})

level_group = get_resolution_level(root, 0)
up = read_links(level_group, delta=1)
print(len(up), up[0])
```

```text
0 [1] {1: ['0.0.0']}
1 [-1, 1] {-1: ['+1.+1.+1', '+1.+1.0', '+1.0.+1'], 1: ['0.0.0']}
500000 (((0, 0, 0), 0), ((0, 0, 0), 0))
```

Each record is a tuple of `(chunk_coords, fragment_index)` endpoints.
See [Links](../../spec/object_model/links.md) and
[Pyramid construction](../../spec/multiscale/pyramid_construction.md) for the
layout these records use.

```{note}
`examples/07_multiscale_links.ipynb` predates the merged links layout and no
longer imports — it reaches for `constants.CROSS_CHUNK_LINKS` and
`core.arrays.read_cross_chunk_links`, neither of which survived FORMAT 0.9.0.
Use the code on this page instead until the notebook is rewritten.
```

---

## Per-level control: a gap in the supported surface

`build_pyramid` is isotropic and whole-pyramid. Two things it cannot express
have **no replacement on `api` or `building` today**:

- coarsening a *single* level, with an explicit `source_level` /
  `target_level` pair;
- re-coarsening every level above a given one in place after level 0 changed.

Both live on internal modules. Reaching for them is reaching past the
compatibility contract — they may move or change signature in any release, and
an import of one is a name nobody knows is load-bearing, which is exactly how
past layout refactors turned into downstream breaks. If you need either, the
right move is to ask for it to be promoted into `building` rather than to
depend on the internal spelling. It is a gap to report, not a reason to import
from `multiresolution` or `ops`.

For the common case — build the whole pyramid — `Dataset.build_pyramid` covers
it, and `chunk_scale_factors` already accepts a per-axis tuple per level, so
anisotropic *chunking* needs no internal import:

```python
ds.build_pyramid(
    factors=[(2.0, 1.0), (2.0, 1.0)],
    chunk_scale_factors=[(2, 2, 1), (2, 2, 1)],   # per-axis, per level
)
```

If you genuinely need one level at a time, this is the internal spelling, so
pin your `zarr-vectors` version if you use it:

```python
# INTERNAL — zv.stability("zarr_vectors.multiresolution") == "internal"
from zarr_vectors.multiresolution.coarsen import coarsen_level

coarsen_level(
    "synchrotron.zarrvectors",
    source_level=3,
    target_level=4,
    coarsen_factor=2.0,
    chunk_scale_factor=(2, 2, 1),
)
```

```text
{'vertex_count': 27, 'object_count': 1, 'objects_kept': 1, 'source_objects': 1, 'method': 'per_object', 'preserves_object_ids': True, 'shared_fragments': True}
```

`source_level` does not have to be 0, so calls chain to build a pyramid one
level at a time. Note that `coarsen_level` emits **no** cross-level link arrays
on its own — its `cross_level_storage` defaults to `"none"` for standalone
callers, and the composing pass that produces `|N| ≥ 2` deltas and stamps the
`multiscale_links` capability only runs inside `build_pyramid`. A pyramid
assembled from `coarsen_level` calls has no `links/<delta>/` at all.

The in-place re-coarsen (`zarr_vectors.ops.refresh.rebuild_pyramid_from_level`)
is internal on the same terms; it re-runs `coarsen_level` for every level above
its `source_level` using each target's own recorded parameters. Rebuilding with
`Dataset.build_pyramid` after removing the stale levels is the supported way to
get the same result.

---

## Listing and removing levels

Level bookkeeping is builder work, and all of it is on
`zarr_vectors.building` — including undoing the level the previous section
added:

```python
from zarr_vectors.building import (
    open_store, list_resolution_levels, remove_resolution_level,
)

root = open_store("synchrotron.zarrvectors", mode="r")
print(list_resolution_levels(root))

root = open_store("synchrotron.zarrvectors", mode="r+")
remove_resolution_level(root, level_index=4)
print(list_resolution_levels(open_store("synchrotron.zarrvectors", mode="r")))
```

```text
[0, 1, 2, 3, 4]
[0, 1, 2, 3]
```

Removing a level updates the root's multiscale metadata too, so the api
surface agrees immediately:

```python
print(zv.open("synchrotron.zarrvectors").levels)
```

```text
(0, 1, 2, 3)
```

For a read-only listing, `Dataset.levels` needs no builder import at all.

---

## Reading pyramid levels

```python
ds = zv.open("synchrotron.zarrvectors")

for index in ds.levels:
    print(index, ds.level(index).read().vertex_count, ds.read(level=index).vertex_count)
```

```text
0 500000 500000
1 8000 8000
2 1000 1000
3 125 125
```

`ds.level(i).read()` and `ds.read(level=i)` are the same read. A viewer
usually wants a level by physical size rather than by index:

```python
print(ds.resolution(scale=1600.0).index)
print(ds.resolution(scale=200.0).index)
```

```text
3
0
```

Drilling down to a region at full resolution works as expected:

```python
detail = ds.select(
    level=0, bbox=((400.0, 400.0, 400.0), (600.0, 600.0, 600.0)),
).read()
print(detail, detail.vertex_count)
```

```text
ReadResult(kind='point_cloud', vertices=533, attributes=['intensity', 'label']) 533
```

**Bounding-box queries against coarsened levels currently under-report** —
badly, not marginally:

```python
print(ds.select(level=2, bbox=((400.0, 400.0, 400.0), (600.0, 600.0, 600.0))).read().vertex_count)
print(ds.level(2).vertex_count)
```

```text
0
1000
```

For a region at low resolution, read the coarse level whole and filter in
memory. Whole-level reads at every level are exact.

---

## Performance tips

**Build levels from finest to coarsest.** `build_pyramid` does this
automatically — each level coarsens from the previous one, not from level 0,
so per-level work decreases as the pyramid grows.

**Skip cross-level emission when you do not need it.** If downstream consumers
never navigate between levels, `cross_level_storage="none"` saves the arrays
and the post-build pass. `cross_level_depth=0` does not.

**Size the pyramid to the bin grid, not the vertex count.** A level cannot
have more vertices than it has occupied bins, so a factor that takes the bin
grid below the number of cells you want on screen buys nothing but a smaller
file.

**`method="per_object"` is the only core method, and it is OID-stable.** It is
what you want when you need to track the same object across resolution levels,
and it is the default. Alternative methods appear in `zv.coarsen_methods()`
only when a strategy package such as **`zarr-vectors-tools`** is installed.

**Build pyramids near the store.** For cloud stores (S3 / GCS) run
`build_pyramid` from a VM in the same region as the bucket — pyramid building
is I/O-bound on cloud, and same-region latency is roughly an order of
magnitude lower than from a laptop.

---

## See also

- [Deferred reads and out-of-core access](lazy_loading.md) — choosing between
  the levels this page builds, and reading them without loading everything.
- {doc}`../../api/api` — `Dataset.build_pyramid`, `Level`, `Query`.
- {doc}`../../api/building` — the builder surface used above.
- [Pyramid construction](../../spec/multiscale/pyramid_construction.md) and
  [Sparsity](../../spec/multiscale/sparsity.md) — the format side.
