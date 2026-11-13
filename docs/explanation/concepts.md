# Core concepts

This page defines the terms the rest of these docs use and explains how a
store is organised. The [Specification](../spec/index.md) treats each concept
in full.

## What is Zarr Vectors?

Zarr Vectors is a chunked storage format for spatial vector geometry: point
clouds, streamlines, graphs, skeletons and meshes. It is built on Zarr v3.
Every array in a store is an ordinary Zarr array, and the resolution pyramid
is declared with OME-Zarr metadata, so existing Zarr tools can read a store.

Forrest Collman at the Allen Institute for Brain Science specified the
format. `zarr-vectors-py` is its Python implementation.

```{figure} ../_static/figures/zarr-vectors-ecosystem.png
:alt: Zarr Vectors within the Zarr ecosystem, backed by Zarr arrays and OME-Zarr metadata, split into the zarr-vectors schema, the zarr-vectors-py core Python library and the zarr-vectors-tools wider methods, alongside Neuroglancer and its Python filtering tools.
:width: 100%
:figclass: zv-figure

**Zarr Vectors in the Zarr ecosystem.** The schema (`zarr-vectors`), this
library (`zarr-vectors-py`) and the converters and wider methods
(`zarr-vectors-tools`) are separate packages. Neuroglancer reads the stores
directly.
```

## The store

A **store** is a Zarr v3 group: a directory on disk or a prefix in cloud
object storage. By convention its name ends in `.zarrvectors`. The root
`zarr.json` holds the format's metadata in its attributes:
`attributes.zarr_vectors` for store-level fields and `attributes.multiscales`
for the pyramid.

```
scan.zarrvectors/
├── zarr.json              root attributes: zarr_vectors, multiscales
├── 0/                     level 0, full resolution
│   ├── zarr.json          level attributes: zarr_vectors_level
│   ├── vertices/          vertex positions, one cell per occupied chunk
│   ├── vertex_fragments/  fragment index into vertices/
│   ├── vertex_attributes/ one array per attribute name
│   ├── links/             edges, as links/<delta>/<offsets>/
│   ├── link_fragments/    fragment index over intra-chunk links
│   ├── link_attributes/   mirrors links/
│   ├── fragment_attributes/
│   ├── object_index/      manifests/ and object_ids/
│   ├── object_attributes/ one row per object
│   ├── groups/            row g lists group g's object ids
│   └── group_attributes/
└── 1/                     a coarser level
```

A new store has only the root group, a `0/` group, and empty `vertices/`
and `vertex_fragments/` arrays. The other arrays appear when the matching data is
written:

| Kind | Level-0 arrays |
|------|----------------|
| `point_cloud` | `vertices`, `vertex_fragments` |
| `line` (also polylines and streamlines) | `vertices`, `vertex_fragments`, `links`, `object_index` |
| `mesh`, `graph` | `vertices`, `vertex_fragments`, `links`, `link_fragments`, `object_index` |

Any kind gets `object_index/` once object ids are written. Attribute arrays
and `groups/` appear only when you pass that data. `fragment_attributes/` and
`group_attributes/` are written only through `zarr_vectors.building`. The root
can also hold `parametric/` (analytic shapes) and `headers/` (preserved
source-format headers). [Directory structure](../spec/layout/directory_structure.md)
gives the full tree for each geometry.

## Chunks

A **chunk** is the unit of I/O: one file per occupied chunk in each array. On
the `zarr_vectors.api` surface a chunk is called a **cell**. A spatial read
finds the chunks that overlap the region and reads only those.

The chunk size is in the units of your coordinates and is the same across a
level. You set it on the schema's `Layout`: `cells=` gives the number of
chunks per axis, and `cell_size=` gives the size directly. Bounds are
inclusive, so 1000 µm bounds with `cells=5` give a grid of 6 × 6 × 6 chunks of
200 µm.

Larger chunks mean fewer files and larger reads; they suit batch processing
and network file systems. Smaller chunks make small-region queries faster;
they suit interactive viewing. With no `cells=`, the layout makes one chunk
for the whole volume, or chunks of about 64 MB when the schema has
`expected=zv.SizeHints(n_vertices=...)`. See
[Choosing a layout](../how_to/choose_chunk_and_bin.md).

## Bins

A **bin** is a subdivision of a chunk and the unit of the spatial index. A
bounding-box query resolves to bins, so it can read part of a chunk.
`Layout(subcells=n)` puts `n` bins per chunk on each axis (default 4, so 64
bins per chunk in 3-D). `subcells=1` makes the bin the same as the chunk.

`Level.scale` reports the chunk shape and `Level.resolution` the bin shape.

## Fragments

A **fragment** is one entry in a chunk's fragment index. It names a
contiguous range of rows in that chunk's `vertices/` cell, or a list of row
indices. At level 0 each non-empty bin has one fragment. So a bounding-box
query becomes a set of `(chunk, fragment)` pairs, each read without the rest
of the chunk. Fragments address rows, not bytes, so two fragments can share a
vertex. See [Fragments](../spec/object_model/fragments.md).

## Objects, manifests and groups

An **object** is one discrete item: a streamline, a skeleton, a mesh, or a
labelled set of points. Each object has an integer id. Its **manifest** is
one row of `object_index/manifests/`, listing every chunk the object touches
and the fragments in each. Reading one object decodes its manifest and reads
the fragments it names, with no chain of dependent reads.

At coarser levels two manifests can name the same fragment, a **shared
fragment**; the vertex row is stored once. A **group** is a set of objects,
stored as one row of `groups/`. See
[Object manifest](../spec/object_model/object_manifest.md).

## Links

A **link** is an edge between vertices, stored under
`links/<delta>/<offsets>/`. `<delta>` is how many levels the link spans: `0`
within a level, `+1` or `-1` to the next level. `<offsets>` is where the other
end sits relative to the chunk holding the record: `0.0.0` is within the
chunk, `0.0.+1` one chunk along `+z`. A link with non-zero offsets is a
**cross-chunk link**. Polylines write no `link_fragments/`, because vertex
order gives their edges within a chunk. Object reads do not use links: the manifest already
names every chunk.

## Levels and the pyramid

A **level** is one resolution of the data, stored as a numbered group (`0/`,
`1/`, …). Level 0 is full resolution. `Dataset.build_pyramid` adds coarser
levels. It takes one `(coarsen_factor, sparsity_factor)` pair per new level,
each relative to the level below, so the factors compound.

The **coarsening factor** scales the bin shape on every axis. The vertices in
each bin merge into one **metavertex**, so a factor of 2 cuts the vertex count
by up to 8 in 3-D. `chunk_scale_factors=` grows the chunk shape too; without
it every level keeps the level-0 chunk shape.

The **object sparsity** factor thins objects. A factor of 2 keeps half of
them, chosen at random with a seed, and the level records
`object_sparsity: 0.5`. Kept objects keep their ids. `zarr-vectors-tools`
adds ranked selection strategies.

A coarser level has `vertices`, `vertex_fragments`, `object_index`,
`links/-1/` back to its parent, and the parent's `object_attributes`. It has
no `vertex_attributes`, `groups`, `group_attributes`, `link_attributes` or
`link_fragments`. See
[Building pyramids](../tutorials/multiscale/building_pyramids.md).

## OME-Zarr multiscale metadata

The root `attributes.multiscales` block follows the
[OME-Zarr NGFF specification](https://ngff.openmicroscopy.org/), version 0.4,
so OME-Zarr viewers can find the levels. It has one `datasets` entry per
level. Each entry's `scale` is the level's coarsening against level 0, and its
`translation` is half the level's bin shape. Zarr Vectors marks the block with
`multiscales[0].metadata.format == "zarr_vectors"`.

Format-specific fields stay outside this block: `bounds`, `chunk_shape` and
`base_bin_shape` in `attributes.zarr_vectors`, and `vertex_count`,
`object_sparsity` and others in each level's `attributes.zarr_vectors_level`.

## Geometry types

| Constant | Value | Description |
|----------|-------|-------------|
| `GEOM_POINT_CLOUD` | `"point_cloud"` | Unconnected vertices |
| `GEOM_LINE` | `"line"` | Pairs of vertices (line segments) |
| `GEOM_POLYLINE` | `"polyline"` | Ordered vertex sequences |
| `GEOM_STREAMLINE` | `"streamline"` | Polylines from tractography |
| `GEOM_GRAPH` | `"graph"` | Vertices and edges, directed or undirected |
| `GEOM_SKELETON` | `"skeleton"` | Tree-structured graph, as in SWC |
| `GEOM_MESH` | `"mesh"` | Triangulated surface with faces |

The constants are in `zarr_vectors.constants`. All types share the same
chunked layout. Types other than point clouds also have `object_index/` and
`links/`.

## Coordinates and units

Coordinates are stored as given, and chunk and bin shapes use the same
units. The schema's `axes` (for example `zv.Axis("x", unit="micrometer")`)
name the axes and units in the `multiscales` block, so NGFF tools show the
data at the right physical scale. An optional coordinate reference system,
`crs` in `attributes.zarr_vectors`, is written with
`zarr_vectors.building.create_store(..., crs=...)`; `Schema` does not take
one.

## Summary

| Term | What it is | Set by |
|------|------------|--------|
| Store | A Zarr v3 group, named `*.zarrvectors` | the path passed to `zv.create` |
| Chunk (cell) | Unit of I/O; one file per occupied chunk | `Layout(cells=…)` or `Layout(cell_size=…)` |
| Bin | Unit of the spatial index within a chunk | `Layout(subcells=…)` |
| Fragment | One entry in a chunk's fragment index | computed |
| Object | One discrete item with an integer id | object ids at write time |
| Manifest | An object's `(chunk, fragment)` list | computed |
| Group | A set of objects | `groups=` at write time |
| Link | An edge, stored as `links/<delta>/<offsets>/` | computed |
| Level | One resolution of the data | `Dataset.build_pyramid()` |
| Coarsening factor | Bin growth per level | first element of each `factors` pair |
| Object sparsity | Object thinning per level | second element of each `factors` pair |
