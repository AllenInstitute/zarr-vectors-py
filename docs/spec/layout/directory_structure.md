# Directory structure

This page shows the on-disk tree of a Zarr Vectors store for each geometry
type, and which paths each writer creates.

## Terms

**Store root**
: The directory (or object-store prefix) holding the root `zarr.json`, the
  level groups and, optionally, `parametric/`. Use the extension
  `.zarrvectors`; `.zv` is an accepted short form. Nothing reads the
  extension.

Level groups, per-chunk arrays and link families are defined on
[Level groups](level_groups.md), [Chunk arrays](chunk_arrays.md) and
[Links](../object_model/links.md).

## Introduction

A store is a Zarr v3 hierarchy: a root group, one group per level, and the
arrays of each level. All metadata is in the `attributes` of the
`zarr.json` files; the trees below omit most of them.

```{figure} ../../_static/figures/zarr-vectors-store-structure.png
:alt: Panel a, the directory tree of a three-level store with level 0 expanded. Panel b, vertices, links with chunk and level offsets, and objects split into fragments across levels 0, 1 and 2.
:width: 100%
:name: fig-store-structure
:figclass: zv-figure

**Zarr Vectors data structure.** **a** — A three-level store, with level 0
expanded. **b** — Vertices, links with chunk and level offsets, and objects
split into fragments, across the same three levels.
```

## Technical reference

The trees come from the writers in `zarr_vectors.types`, with
`chunk_shape=(200, 200, 200)` over bounds `[0, 400]` on each axis.

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
    ├── vertex_fragments/
    └── vertex_attributes/
        ├── intensity/
        └── label/
```

### Streamline, polyline and line

`write_polylines(..., object_attributes=..., groups=...,
group_attributes=...)`. `write_lines` writes the same tree:

```text
streamline.zarrvectors/
├── zarr.json
└── 0/
    ├── zarr.json
    ├── vertices/
    ├── vertex_fragments/
    ├── links/
    │   └── 0/                # level delta 0
    │       ├── +1.0.0/       # segments whose next vertex is one chunk along +x
    │       ├── 0.+1.0/
    │       └── 0.0.+1/
    ├── object_index/
    ├── object_attributes/
    ├── groups/
    └── group_attributes/
```

Vertex order links consecutive vertices within a chunk, so these stores
have only cross-chunk link arrays: no `links/0/0.0.0/` and no
`link_fragments/`.

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
    │       └── …             # one array per offset that occurs
    ├── link_fragments/       # fragment index over links/0/0.0.0/ only
    ├── link_attributes/
    │   └── weight/0/…        # same arrays as links/0/
    └── object_index/
```

`write_graph(..., kind="skeleton")` writes the same tree.

### Mesh

`write_mesh` writes the graph tree without `link_attributes/`. Its link
arrays have `link_width` 3, one row per triangle. Each segment name joins
two offsets with `_`, one per non-source vertex:

```text
mesh.zarrvectors/0/
├── links/
│   └── 0/
│       ├── 0.0.0_0.0.0/      # triangle wholly inside the cell
│       ├── 0.0.0_0.0.+1/     # third vertex one chunk along +z
│       └── …
└── link_fragments/           # fragment index over links/0/0.0.0_0.0.0/
```

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

Coarser streamline levels have no `links/0/`. For how cross-level offsets
are measured, see
[Links](../object_model/links.md#cross-level-placement-the-anchor).

### Parametric objects

`write_parametric_objects` adds an unchunked `parametric/` group at the
store root:

```text
param.zarrvectors/
├── zarr.json
├── 0/ …
└── parametric/
    ├── zarr.json             # attributes: parametric_types
    ├── objects/              # float64, [num_objects, max_row_length]: type ID, then coefficients
    ├── names/                # variable_length_bytes, one name per object
    ├── object_attributes/
    ├── groups/
    └── group_attributes/
```

`parametric_types` maps each type ID to its name and coefficient names:
`0` plane, `1` line and `2` sphere are built in.

### Names and presence

Level groups are named `0`, `1`, `2`, …;
[L5 validation](../validation/overview.md) requires them to run from `0`
without gaps. Other root children, except `parametric/`, are not part of
the format. Array names within a level are fixed. In a sharded store the
tree is the same, but each file under `c/` is a shard holding a block of
cells; see [Sharding](../chunking/sharding.md).

| Path | Present in |
|------|------------|
| `zarr.json` (root), `0/` | Every store |
| `vertices/`, `vertex_fragments/` | Every level |
| `links/0/<offsets>/` | Lines, polylines, streamlines (cross-chunk only); graphs, skeletons, meshes |
| `links/0/<all-zero offsets>/`, `link_fragments/` | Graphs, skeletons, meshes |
| `links/+1/`, `links/-1/` | Pyramids with `cross_level_storage` `"explicit"` |
| `object_index/` | Every type except a point cloud written without `object_ids` |
| `vertex_attributes/`, `fragment_attributes/`, `link_attributes/`, `object_attributes/` | When the writer gets that kind of attribute |
| `groups/`, `group_attributes/` | When the writer gets `groups` |
| `parametric/` | When parametric objects are written |
