# Core concepts

This page explains the key ideas behind the Zarr Vectors and
`zarr-vectors-py`. It is intended as a mental-model introduction — enough
to make informed decisions about chunk sizes, bin sizes, and resolution
pyramids before writing your first large dataset. The
[Specification](../spec/index.md) gives the full technical treatment of
each concept.

The [Quickstart](quickstart.md) shows the calls; this page explains what
they are doing to the store.

---

## What is the Zarr Vectors?

Zarr Vectors is a chunked, cloud-native storage format for *spatial vector geometry*
data: point clouds, streamlines, graphs, skeletons, and meshes. It is built
on **Zarr v3** and follows a directory-tree layout in which every array
(vertex positions, connectivity, attributes) is stored as a spatially
chunked Zarr array.

The format was originally specified by Forest Collman at the Allen Institute
for Brain Sciences. `zarr-vectors-py` is a Python implementation of that
specification, extended with separated chunk/bin sizes, per-level object
sparsity, and OME-Zarr-compatible multiscale metadata.

Because every array in a Zarr Vectors store is an ordinary Zarr array and the
resolution pyramid is declared with OME-Zarr metadata, the format inherits
the existing Zarr ecosystem rather than sitting beside it. The specification,
this library, and the wider tooling are deliberately separate packages:

```{figure} ../_static/figures/zarr-vectors-ecosystem.png
:alt: Zarr Vectors within the Zarr ecosystem, backed by Zarr arrays and OME-Zarr metadata, split into the zarr-vectors schema, the zarr-vectors-py core Python library and the zarr-vectors-tools wider methods, alongside Neuroglancer and its Python filtering tools.
:width: 100%
:figclass: zv-figure

**Zarr Vectors in the Zarr ecosystem.** Zarr Vectors handles the non-array
data — geometry — on top of Zarr arrays and OME-Zarr metadata, so anything
that already speaks Zarr can read a store. The schema (`zarr-vectors`), this
core Python library (`zarr-vectors-py`), and the converters and wider methods
(`zarr-vectors-tools`) are separate packages; Neuroglancer reads the stores
directly, with filtering driven from its Python API.
```

---

## The store is a directory

A Zarr Vectors store is an ordinary directory on disk (or a prefix in cloud object
storage). Its name conventionally ends in `.zarrvectors`. It is a plain
Zarr v3 group, so its root document is `zarr.json` — not a Zarr v2 `.zattrs`,
and not a format-specific sidecar:

```
scan.zarrvectors/
├── zarr.json                 [always]  Zarr v3 group metadata.  Store-level fields live
│                                       under attributes.zarr_vectors (zv_version, bounds,
│                                       chunk_shape, base_bin_shape, geometry_types, …);
│                                       per-level scale and translation live under
│                                       attributes.multiscales (OME-NGFF 0.4).
├── 0/                        [always]  Full resolution.  Levels are bare integers, no prefix.
│   ├── zarr.json             [always]  Level metadata under attributes.zarr_vectors_level
│   │                                   (vertex_count, object_sparsity, coarsening_method,
│   │                                   parent_level, arrays_present, …).
│   ├── vertices/             [always]  One cell per occupied chunk, at c/<i>/<j>/<k>, holding
│   │   └── c/0/0/0 …                   every vertex in that chunk.  One cell per chunk, not
│   │                                   per bin.
│   ├── vertex_fragments/     [always]  Fragment index into the sibling vertices cell.
│   ├── vertex_attributes/    [written] One child array per name, same grid, rows aligned 1:1
│   │   └── intensity/                  with the vertices cell.
│   ├── links/                [written] Connectivity.  A group two levels deep, never an array:
│   │   ├── 0/0.0.0/                    links/<delta>/<offsets>/.  <delta> is how many pyramid
│   │   ├── 0/0.0.+1/                   levels the record spans (0 within a level, ±1 to the
│   │   └── +1/0.0.0/                   parent/child level).  <offsets> is where the record's
│   │                                   other endpoints sit relative to the chunk holding it,
│   │                                   so 0.0.0 is intra-chunk and 0.0.+1 reaches one chunk
│   │                                   along +z.
│   ├── link_fragments/       [written] Fragment index over the intra-chunk link array only.
│   ├── link_attributes/      [written] Mirrors links/ exactly, row for row.
│   │   └── weight/0/0.0.0/
│   ├── fragment_attributes/  [builder] One row per fragment in the chunk.
│   ├── object_index/         [written] manifests/ is a ragged array with one row per object
│   │   └── manifests/                  slot — that object's ordered (chunk, fragment) references.
│   ├── object_attributes/    [written] One array per name, one row per object.
│   ├── groups/               [written] One ragged array: row g is that group's object ids.
│   └── group_attributes/     [builder] One array per name, one row per group.
└── 1/                        [written] A coarser level from build_pyramid.  Same layout, but
    └── …                               fewer array families — see the pyramid section below.
```

- `[always]` — present the moment `zv.create(...)` returns. That is all a
  fresh store is: a root `zarr.json`, a `0/` group with its own `zarr.json`,
  and empty `vertices/` and `vertex_fragments/` arrays.
- `[written]` — appears only once the matching data is written, so which of
  these a store has depends on its geometry kind and on what the caller
  passed.
- `[builder]` — written only through `zarr_vectors.building`; the
  `zarr_vectors.api` surface reads them but does not produce them.

Which array families a level-0 write actually creates follows from the
geometry:

| Kind | Level-0 children |
|------|------------------|
| `point_cloud` | `vertices`, `vertex_fragments` |
| `line` | `vertices`, `vertex_fragments`, `links`, `object_index` |
| `mesh`, `graph` | `vertices`, `vertex_fragments`, `links`, `link_fragments`, `object_index` |

Polylines and streamlines get the `line` set: their intra-chunk edges are
implicit in vertex order, so they never write `link_fragments/`. Any kind
grows an `object_index/` as soon as object ids are written, and
`vertex_attributes/`, `object_attributes/` and `groups/` appear only when
that data is passed. A point cloud grows a `links/` only if you build a
pyramid.

Two more groups can sit at the root rather than inside a level:
`parametric/` for analytic shapes and `headers/` for preserved source-format
headers. Both are created lazily on first write. See
[Directory structure](../spec/layout/directory_structure.md) for the
authoritative tree per geometry type, and for a diagram of how vertices,
links, fragments and objects map onto these arrays across a multi-level
store.

Each sub-directory is a Zarr group. Arrays within a group are themselves
directories containing one file per chunk (or shard). Nothing is binary-
proprietary; the entire store can be inspected with `zarr` or a plain file
browser. Sharding does not change the tree — the keys stay `c/i/j/k`, and
each file just becomes a shard covering several cells.

---

## Chunks

A **chunk** is the unit of I/O. When `zarr-vectors` reads a spatial region
it determines which chunks overlap that region, issues one read per chunk
(or one HTTP range request in cloud storage), and returns the merged result.

The chunk size is given in the same units as your coordinate data (e.g.
micrometres, voxels) and applies uniformly across a level. You do not pass it
per call: you set it once, on the `Layout` attached to a `Schema`. The
data-shaped spelling says how many chunks to cut the volume into per axis:

```python
import zarr_vectors as zv

schema = zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    kind="point_cloud",
    layout=zv.Layout(cells=5),      # five chunks per axis over 1000 µm
)
ds = zv.create("scan.zarrvectors", schema=schema)

print(ds.level(0).scale)            # the chunk shape, in physical units
print(ds.level(0).grid)
```

```text
(200.0, 200.0, 200.0)
Grid(5x5x5 cells of (200.0, 200.0, 200.0))
```

When the grid is fixed from outside — a pipeline whose chunks must line up
with an image volume, say — name the size instead:

```python
fixed = zv.create("fixed.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    layout=zv.Layout(cell_size=(250.0, 250.0, 250.0)),
))
print(fixed.level(0).scale, fixed.level(0).grid.shape)
```

```text
(250.0, 250.0, 250.0) (4, 4, 4)
```

Choosing a chunk size:

- Larger chunks → fewer files, lower open/seek overhead, larger individual
  reads. Good for sequential access patterns and network file systems.
- Smaller chunks → faster targeted queries for small spatial regions. Good
  for interactive visualisation.
- A common starting point for 3-D biological data: **200–500 physical units**
  per axis, adjusted so each chunk contains ~10 000–100 000 vertices.

The default is `Layout()`, which means `cells="auto"` — one chunk for the
whole volume. That is the honest default and almost never the right one;
pass `cells=`. See
[Where `chunk_shape` and `bin_shape` went](quickstart.md#where-chunk_shape-and-bin_shape-went)
for how these replaced the old per-call `chunk_shape=` / `bin_shape=`
arguments, and
[Choosing a layout](../how_to/choose_chunk_and_bin.md) for
the sizing arithmetic.

---

## Supervoxel bins

A **bin** is a finer spatial subdivision *within* a chunk. Bins are the
unit of the spatial index: a bounding-box query resolves to a set of bins,
not a set of chunks. This means you can retrieve a small spatial region
without loading an entire chunk from disk.

Bins subdivide chunks evenly, so the bin size is not given in physical units
either — `subcells` says how many bins to cut each chunk into per axis, and
the physical bin shape falls out of that:

```python
print(ds.level(0).resolution)       # bin shape: 200 / 4, the default subcells

fine = zv.create("fine.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    layout=zv.Layout(cells=5, subcells=8),   # 8×8×8 = 512 bins per chunk
))
print(fine.level(0).scale, fine.level(0).resolution)
```

```text
(50.0, 50.0, 50.0)
(200.0, 200.0, 200.0) (25.0, 25.0, 25.0)
```

The bin grid within a chunk looks like this (2-D cross-section of the
`cells=5` store above, whose four bins per axis give 64 bins per chunk):

```
┌──────────────────────────────────────┐
│ chunk (200 × 200)                    │
│ ┌────────┬────────┬────────┬────────┐│
│ │ bin    │ bin    │ bin    │ bin    ││
│ │(50×50) │(50×50) │(50×50) │(50×50) ││
│ ├────────┼────────┼────────┼────────┤│
│ │ bin    │ bin    │ bin    │ bin    ││
│ │        │        │        │        ││
│ ├────────┼────────┼────────┼────────┤│
│ │  …                               ││
└──────────────────────────────────────┘
```

`subcells=1` gives one bin per chunk, which is backward-compatible behaviour
equivalent to a pure chunk-indexed store.

**Bin shape vs chunk shape — the key difference:**
the chunk shape controls the on-disk file layout. The bin shape controls
spatial query granularity, and it grows with the coarsening factor at each
coarser resolution level, so coarser levels have fewer, larger bins per
chunk. This is what keeps per-chunk data volume roughly constant across the
pyramid.

---

## Fragments

A **fragment** is Zarr Vectors' addressing primitive: one entry in a chunk's
fragment index, naming either a contiguous range of rows in that chunk's
`vertices/` cell or an explicit list of row indices. At level 0 fragments
line up with bins one-to-one — each non-empty bin emits exactly one range
fragment — so a bbox query resolves to a set of `(chunk, fragment)` pairs
and `zarr-vectors` retrieves each one without reading the rest of the chunk.

On disk the index lives in `vertex_fragments/`, one small blob per chunk,
addressed by the same `c/<i>/<j>/<k>` key as the vertices it describes. The
addresses it yields are row indices, not byte offsets, which is what lets two
fragments share a vertex. At coarsened pyramid levels the one-to-one
correspondence with bins is relaxed: a fragment there may represent a
metavertex shared between several objects' manifests. See
[Fragment-index arrays](../spec/layout/fragment_index_arrays.md) for the
byte layout and [Fragments](../spec/object_model/fragments.md) for the
read/write model.

---

## Object model

For geometry types that have discrete *objects* — lines, polylines,
streamlines, graphs, skeletons, meshes, and point clouds written with object
ids — Zarr Vectors stores an additional `object_index` group whose
`manifests/` array holds a per-object **manifest**: one row per object,
enumerating every chunk the object touches and the fragments within each
chunk. Reading one object by id is a manifest decode plus the fragment reads
it names:

```python
tracts = zv.open("tracts.zarrvectors")
one = tracts.level(0).objects[42]
print(one.part_count, one.vertex_count)
```

The manifest is self-contained: there is no chain of dependent reads.
At coarsened pyramid levels, two objects' manifests may name the same
fragment (a **shared fragment**) — the underlying vertex row is stored
once. See [Object manifest](../spec/object_model/object_manifest.md).

Edges that span a chunk boundary are still stored, and still matter for
geometry — you need them to draw a line segment that crosses a chunk. They
are not a separate array family, though. A **cross-chunk link** is simply a
record in `links/0/<offsets>/` whose offsets are non-zero: `links/0/0.0.0/`
holds the intra-chunk edges, `links/0/0.0.+1/` the ones reaching one chunk
along `+z`, and so on. (Format 0.9.0 merged what used to be a separate
`cross_chunk_links/` group into this scheme.) Pre-0.6 those edges were also
used during object retrieval, to discover an object's continuation chunks;
the manifest made that unnecessary.

---

## Multi-resolution pyramids

Zarr Vectors stores can contain multiple resolution levels under `0/`,
`1/`, etc. Each level is generated by spatial coarsening
(*binning*) and, for discrete-object types, optional *object thinning*.

**Coarsening** controls vertex reduction. A coarsening factor of 2 at level 1
means the bin shape at that level is twice the level-0 bin shape on every
axis, so each bin is 8× larger and its vertices are merged into one
metavertex. With the chunk grid held fixed at 200 µm, that means fewer and
larger bins per chunk at every step:

```
Level 0: bin shape = (50, 50, 50)    → 64 bins/chunk  (full resolution)
Level 1: bin shape = (100, 100, 100) → 8 bins/chunk   (2× coarser)
Level 2: bin shape = (200, 200, 200) → 1 bin/chunk    (4× coarser)
```

**Object sparsity** controls object reduction for discrete-object types.
A sparsity factor of `2.0` at level 1 keeps half the objects, recorded in the
level's metadata as `object_sparsity: 0.5`. Core selects them uniformly at
random, reproducibly with a seed; the ranked strategies — spatial coverage,
object length, per-object attribute value, point-thinning — live in
`zarr-vectors-tools`, which registers them on import.

Total data volume reduction at a level = vertex reduction × object
reduction. A coarsening factor of 2 gives 8× vertex reduction; with a
sparsity factor of 2 as well, total reduction = 8× × 2× = **16×**.

`Dataset.build_pyramid` builds the whole pyramid. `factors` is one
`(coarsen_factor, sparsity_factor)` pair per level to add, each measured
against the level below, so they compound. `1.0` opts out of that axis:

```python
tracts = zv.open("tracts.zarrvectors", mode="r+")
print(zv.coarsen_methods())
report = tracts.build_pyramid(
    factors=[(2.0, 1.0), (2.0, 1.0)],
    chunk_scale_factors=[2, 2],
    method="per_object",
)
print(report["levels_created"])
```

```text
('per_object',)
2
```

`coarsen_methods()` lists the methods this installation can dispatch to:
`"per_object"` is the one core ships, and an installed strategy package adds
its own. Pass `chunk_scale_factors=` alongside `factors=` if you want coarser
levels to carry larger chunks as well as larger bins; without it every level
inherits the root chunk shape.

A coarsened level is not a copy of level 0. It gets `vertices`,
`vertex_fragments`, `object_index` and the `links/-1/` back-references to its
parent, and it inherits `object_attributes`. It does not get
`vertex_attributes`, `groups`, `group_attributes`, `link_attributes` or
`link_fragments`.

Re-open the dataset after building: the handle that built the pyramid keeps
stale level metadata. See
[Building pyramids](../tutorials/multiscale/building_pyramids.md).

```{note}
Two pyramid operations have no supported spelling yet: coarsening a *single*
level (`zarr_vectors.multiresolution.coarsen.coarsen_level`) and rebuilding
the levels above an edited one
(`zarr_vectors.ops.refresh.rebuild_pyramid_from_level`). Both modules are
internal, so importing them reaches past the compatibility contract — that is
a gap to report rather than a reason to import from `core`.
`Dataset.build_pyramid` covers the common case of building the pyramid as a
whole.
```

---

## OME-Zarr multiscale metadata

Zarr Vectors borrows the `multiscales` JSON block from the
[OME-Zarr NGFF specification](https://ngff.openmicroscopy.org/). This
means any OME-Zarr-aware viewer can discover the resolution pyramid and
read coordinate transforms from a Zarr Vectors store without modification. The block
sits in the root `zarr.json` under `attributes.multiscales`, declares NGFF
version `0.4`, and carries one `datasets` entry per level. Each entry's
`scale` transform is that level's coarsening ratio against level 0 and its
`translation` is the bin-centroid offset (half the level's bin shape), so
viewers that understand OME-Zarr can perform correct physical-space alignment
at each level. The Zarr Vectors discriminator is
`multiscales[0].metadata.format == "zarr_vectors"`.

Zarr Vectors' own fields are kept out of that block rather than mixed into it, so a
viewer that only understands NGFF sees nothing unexpected: `base_bin_shape`,
`chunk_shape` and `bounds` sit beside it in `attributes.zarr_vectors`, and
`bin_ratio`, `object_sparsity` and the rest sit in each level's own
`attributes.zarr_vectors_level`.

---

## Geometry types

Zarr Vectors supports seven geometry types, each identified by a string constant in
`zarr_vectors.constants`:

| Constant | Value | Description |
|----------|-------|-------------|
| `GEOM_POINT_CLOUD` | `"point_cloud"` | Unconnected vertices with optional per-vertex attributes |
| `GEOM_LINE` | `"line"` | Pairs of vertices (line segments) |
| `GEOM_POLYLINE` | `"polyline"` | Ordered vertex sequences |
| `GEOM_STREAMLINE` | `"streamline"` | Polylines with tractography-specific metadata (step size, seeding) |
| `GEOM_GRAPH` | `"graph"` | Arbitrary vertex–edge graph; directed or undirected |
| `GEOM_SKELETON` | `"skeleton"` | Tree-structured graph aligned to the SWC convention |
| `GEOM_MESH` | `"mesh"` | Triangulated surface mesh with face arrays |

All types share the same chunked spatial layout. Types with discrete objects
(line, polyline, streamline, graph, skeleton, mesh) additionally have an
`object_index` and a `links/` group; a point cloud gets neither unless you
write object ids or build a pyramid.

---

## Coordinate systems and physical units

Coordinates are stored exactly as given: no implicit conversion is performed,
and the chunk and bin shapes are in the same units as the coordinates. What
the axes *mean* is declared on the schema, and is what reaches the NGFF
`multiscales` block:

```python
schema = zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    axes=(
        zv.Axis("x", unit="micrometer"),
        zv.Axis("y", unit="micrometer"),
        zv.Axis("z", unit="micrometer"),
    ),
    kind="point_cloud",
    layout=zv.Layout(cells=5),
)
```

Downstream tools that understand NGFF read those names and units and display
the data with correct physical scaling.

A full coordinate reference system — a `crs` dict under
`attributes.zarr_vectors` — is optional and is written through the
store-building surface (`zarr_vectors.building.create_store(..., crs=...)`)
rather than through `Schema`.

---

## Summary

| Concept | What it is | Configured by |
|---------|-----------|---------------|
| Store | A `.zarrvectors` directory; a Zarr v3 group | the target passed to `zv.create` |
| Chunk | I/O unit; one file per occupied chunk | `Layout(cells=…)` or `Layout(cell_size=…)` |
| Bin | Spatial query unit within a chunk | `Layout(subcells=…)` |
| Fragment | One entry in a chunk's fragment index; the rows of one bin at level 0 | Computed automatically |
| Object index | Per-object manifests naming `(chunk, fragment)` pairs | Written automatically for applicable types |
| Link | An edge, filed by how far it reaches: `links/<delta>/<offsets>/` | Written automatically for applicable types |
| Resolution level | Coarsened copy of the data | `Dataset.build_pyramid()` |
| Coarsening factor | Bin growth per level | first element of each `factors` pair |
| Object sparsity | Object thinning per level | second element of each `factors` pair |
