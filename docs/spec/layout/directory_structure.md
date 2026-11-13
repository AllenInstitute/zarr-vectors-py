# Directory structure

This page shows the on-disk tree of a Zarr Vectors store for each geometry
type, and says which paths each writer creates.

## Terms

**Store root**
: The directory (or object-store prefix) holding the root `zarr.json`, the
  level groups and, optionally, `parametric/`.

  **File extension.** `.zarrvectors` is the extension to use in
  documentation, examples and generated output. `.zv` is an accepted short
  form, for places where path length matters. Nothing in the format reads
  the extension, and a store without one is still a store.

**Level group**
: The group `<N>/` holding one resolution level; `0/` is full resolution.
  See [Level groups](level_groups.md).

**Per-chunk array**
: An array with one cell per spatial chunk, whose cells are files under
  `c/` (`c/i/j/k` in 3-D). See [Chunk arrays](chunk_arrays.md).

**Link family**
: The group `links/<delta>/` (or `link_attributes/<name>/<delta>/`) whose
  children are one per-chunk array per offsets segment. See
  [Links](../object_model/links.md).

## Introduction

A store is a Zarr v3 hierarchy: the root group, one group per resolution
level, and in each level the arrays for that level. All Zarr Vectors
metadata is in the `attributes` of the `zarr.json` files; there are no other
metadata files. Every array and group has its own `zarr.json`, omitted from
the trees below except where it matters.

```{figure} ../../_static/figures/zarr-vectors-store-structure.png
:alt: Panel a, the directory tree of a three-level Zarr Vectors store with the base level expanded into vertices, vertex_attributes, links by level and chunk offset, vertex_fragments, link_fragments, object_index and groups. Panel b, vertices and links drawn across levels 0, 1 and 2, showing links that stay within a chunk, links carrying a chunk offset, links carrying a level offset, and objects split into fragments across chunks.
:width: 100%
:name: fig-store-structure
:figclass: zv-figure

**Zarr Vectors data structure.** How vector items are divided across the Zarr
arrays of a store. **a** — Directory structure of an example store containing
three levels, with the array structure of the base level expanded. **b** —
Items a store can represent: vertices, links and their offsets between chunks
and between resolution levels, and objects and object fragments, across the
same three-level store.
```

## Technical reference

The trees below come from stores written with the writers in
`zarr_vectors.types`, with `chunk_shape=(200, 200, 200)` over bounds
`[0, 400]` on each axis (a 3 × 3 × 3 chunk grid).

### Point cloud

`write_points(..., vertex_attributes={"intensity": ..., "label": ...})`:

```text
points.zarrvectors/
├── zarr.json                 # root attributes: see root_metadata.md
└── 0/
    ├── zarr.json             # level attributes: see level_groups.md
    ├── vertices/
    │   ├── zarr.json
    │   └── c/0/0/0, c/0/0/1, …   # one file per non-empty chunk
    ├── vertex_fragments/     # fragment index over vertices/ rows
    └── vertex_attributes/
        ├── intensity/
        └── label/
```

`fragment_attributes={"<name>": {chunk: values}}` adds
`fragment_attributes/<name>/`. Passing `object_ids` adds the object arrays
shown in the next tree.

### Streamline, polyline and line

`write_polylines(..., object_attributes={"length": ...}, groups=...,
group_attributes={"score": ...})`. `write_lines` writes the same tree:

```text
streamline.zarrvectors/
├── zarr.json
└── 0/
    ├── zarr.json
    ├── vertices/
    ├── vertex_fragments/
    ├── links/
    │   └── 0/                # level delta 0; group attributes hold the family policy
    │       ├── +1.0.0/       # segments whose next vertex is one chunk along +x
    │       ├── 0.+1.0/
    │       └── 0.0.+1/
    ├── object_index/
    │   ├── manifests/
    │   └── object_ids/
    ├── object_attributes/
    │   └── length/
    ├── groups/
    └── group_attributes/
        └── score/
```

Consecutive vertices within a chunk are linked by their order
(`links_convention` `implicit_sequential`), so these stores write only the
link arrays for segments that cross a chunk boundary: no
`links/0/0.0.0/` and no `link_fragments/`.

### Graph and skeleton

`write_graph(..., link_attributes={"weight": ...})`:

```text
graph.zarrvectors/
├── zarr.json
└── 0/
    ├── zarr.json
    ├── vertices/
    ├── vertex_fragments/
    ├── links/
    │   └── 0/
    │       ├── 0.0.0/        # edges with both ends in the cell
    │       ├── 0.0.+1/       # edges to the neighbour along +z
    │       ├── +1.-1.0/
    │       └── …             # one array per offset that occurs
    ├── link_fragments/       # fragment index over links/0/0.0.0/ rows only
    ├── link_attributes/
    │   └── weight/
    │       └── 0/
    │           ├── 0.0.0/    # same segments as links/0/
    │           └── …
    └── object_index/
        ├── manifests/
        └── object_ids/
```

`write_graph(..., kind="skeleton")` writes the same tree with
`link_width` 2; `vertex_attributes={"radius": ...}` adds
`vertex_attributes/radius/`.

### Mesh

`write_mesh`:

```text
mesh.zarrvectors/
├── zarr.json
└── 0/
    ├── zarr.json
    ├── vertices/
    ├── vertex_fragments/
    ├── links/
    │   └── 0/                # link_width 3: one row per triangle
    │       ├── 0.0.0_0.0.0/  # triangle wholly inside the cell
    │       ├── 0.0.0_0.0.+1/ # third vertex one chunk along +z
    │       ├── 0.0.+1_0.+1.0/
    │       └── …
    ├── link_fragments/       # fragment index over links/0/0.0.0_0.0.0/ rows
    └── object_index/
        ├── manifests/
        └── object_ids/
```

A triangle's segment has two offsets, one per non-source vertex, joined by
`_`.

### Pyramid

`build_pyramid(..., factors=[(2, 2), (2, 2)])` on the streamline store adds
levels `1/` and `2/`, each with `vertices/`, `vertex_fragments/` and
`object_index/`, and cross-level link families:

```text
streamline_pyr.zarrvectors/
├── zarr.json                 # multiscales and ome list all three levels
├── 0/
│   └── links/
│       ├── 0/                # as in the single-level store
│       └── +1/0.0.0/         # each level-0 vertex to its level-1 vertex
├── 1/
│   └── links/
│       ├── +1/0.0.0/         # to level 2
│       └── -1/0.0.0/         # to level 0
└── 2/
    └── links/
        └── -1/0.0.0/
```

Coarser streamline levels have no `links/0/`. Cross-level offsets are
measured from the source chunk re-anchored in the target level's grid; see
[Links](../object_model/links.md#cross-level-placement-the-anchor) and the
[multiscale links example](../../../examples/07_multiscale_links.ipynb).

### Sharded store

With a root `shard_shape` (for example `write_points(..., shard_shape=2)`)
the tree is the same, but each file under an array's `c/` is a shard
holding a block of cells; see [Sharding](../chunking/sharding.md).

### Parametric objects

`write_parametric_objects` adds a `parametric/` group at the store root,
outside the levels and not chunked:

```text
param.zarrvectors/
├── zarr.json
├── 0/ …
└── parametric/
    ├── zarr.json             # attributes: parametric_types (the type registry)
    ├── objects/              # float64, [num_objects, max_row_length]: type ID, then coefficients
    ├── names/                # variable_length_bytes, one name per object
    ├── object_attributes/
    │   └── score/
    ├── groups/
    └── group_attributes/
        └── w/
```

The `parametric_types` attribute maps each type ID to its name and
coefficient names; the built-in types are `0` plane, `1` line and
`2` sphere.

### Names and presence

- Level groups are named `0`, `1`, `2`, … . Root children whose names are
  not integers, other than `parametric/`, are not part of the format.
  [L5 validation](../validation/overview.md) requires the levels to run
  from `0` without gaps.
- Array and group names within a level are fixed. Named attributes go under
  `vertex_attributes/`, `fragment_attributes/`, `link_attributes/`,
  `object_attributes/` and `group_attributes/`.

| Path | Present in |
|------|------------|
| `zarr.json` (root) | Every store |
| `0/` | Every store |
| `vertices/`, `vertex_fragments/` | Every level |
| `links/0/<offsets>/` | Lines, polylines and streamlines (cross-chunk segments only); graphs, skeletons and meshes |
| `links/0/<all-zero offsets>/`, `link_fragments/` | Graphs, skeletons and meshes |
| `links/+1/`, `links/-1/` | Pyramids with `cross_level_storage` `"explicit"` |
| `object_index/` | Every type except a point cloud written without `object_ids` |
| `vertex_attributes/`, `fragment_attributes/`, `link_attributes/`, `object_attributes/` | When the writer is given that kind of attribute |
| `groups/`, `group_attributes/` | When the writer is given `groups` |
| `parametric/` | When parametric objects are written |
