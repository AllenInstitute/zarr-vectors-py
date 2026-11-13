# Geometry types

Zarr Vectors defines seven geometry types. A store lists its types in
`geometry_types` in the [root attributes](../layout/root_metadata.md). Each
writer on these pages creates a store with one geometry type.

## Type constants

```python
from zarr_vectors.constants import (
    GEOM_POINT_CLOUD,   # "point_cloud"
    GEOM_LINE,          # "line"
    GEOM_POLYLINE,      # "polyline"
    GEOM_STREAMLINE,    # "streamline"
    GEOM_GRAPH,         # "graph"
    GEOM_SKELETON,      # "skeleton"
    GEOM_MESH,          # "mesh"
)
```

## Comparison matrix

What each type's writer stores at level 0. "Optional" means the writer
stores it when the caller passes the matching argument.

| Property | `point_cloud` | `line` | `polyline` | `streamline` | `graph` | `skeleton` | `mesh` |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `links_convention` | `implicit_sequential` | `implicit_sequential` | `implicit_sequential` | `implicit_sequential` | `explicit` | `implicit_sequential_with_branches` | `explicit` |
| `links/0/` family | — | when a line crosses a chunk | when a polyline crosses a chunk | when a streamline crosses a chunk | ✓ | ✓ | ✓ |
| `link_width` | — | 2 | 2 | 2 | 2 | 2 | face size, ≥ 3 |
| Intra-chunk links (`0.0.0`) | — | — | — | — | ✓ | branch records | ✓ (raw encoding) |
| Family `directed` | — | false | false | false | false | see [Skeleton](skeleton.md) | false |
| `object_index/` | optional | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `object_attributes/` | optional | optional | optional | optional | optional | optional | optional |
| `groups/` | optional | — | optional | optional | — | — | — |
| `vertex_attributes/` | optional | optional | optional | optional | optional | optional | optional |
| `link_attributes/` | — | — | — | — | optional | optional | — |
| Draco encoding | — | — | — | — | — | — | optional |

In the `implicit_sequential` types, vertex order within a fragment gives
the connectivity, so only links that cross a chunk boundary are stored. A
store whose objects each sit in one chunk has no `links/` group. See
[Links](../object_model/links.md).

## Common to every type

The type pages state only what differs from this section.

- **Arrays.** Every writer creates `vertices/` and `vertex_fragments/` at
  level 0 (see [Chunk arrays](../layout/chunk_arrays.md)).
  `vertex_attributes/<name>/` holds one row per vertex,
  `object_attributes/<name>/` one row per object, and
  `fragment_attributes/<name>/` one row per fragment.
- **Root attributes.** No type has type-specific root keys. Besides
  `geometry_types` and `links_convention`, the writers set
  `object_index_convention` `"standard"` and `cross_chunk_strategy`
  `"explicit_links"` (the streaming skeleton writer omits
  `cross_chunk_strategy`). The other keys are on
  [Root metadata](../layout/root_metadata.md).
- **Links.** Every type with links, except the streaming skeleton writer,
  writes one `links/0/` family that is undirected with
  `store: "canonical"`: each link is stored once, under the
  lexicographically positive offset. See [Links](../object_model/links.md).
- **Reading.** Each reader takes `level=` and filters that combine with
  AND. `bbox=` is a `(min_corner, max_corner)` pair of arrays.

  | Reader | Module | Filters |
  |--------|--------|---------|
  | `read_points` | `zarr_vectors.types.points` | `bbox`, `object_ids`, `group_ids`, `chunks`, `attribute_filter` |
  | `read_lines` | `zarr_vectors.types.lines` | `bbox`, `object_ids`, `attribute_filter` |
  | `read_polylines` | `zarr_vectors.types.polylines` | `bbox`, `object_ids`, `group_ids`, `chunks`, `attribute_filter` |
  | `read_graph` | `zarr_vectors.types.graphs` | `bbox`, `object_ids`, `chunks`, `attribute_filter` |
  | `read_mesh` | `zarr_vectors.types.meshes` | `bbox`, `object_ids`, `chunks`, `attribute_filter` |

- **Coarser levels.** See
  [Pyramid construction](../multiscale/pyramid_construction.md).

### Validation

These checks apply to every type; the type pages list only their own. See
[Validation overview](../validation/overview.md).

| Level | Check |
|-------|-------|
| L1 | `vertices/` missing at a level is an error; `vertex_fragments/` missing is a warning. `links/` and `object_index/` are not required |
| L3 | Each manifest entry names a chunk present at the level and a fragment below that chunk's fragment count |
| L3 | Each `links/0/` offsets name parses under the family's `sid_ndim` and `link_width`; `num_physical_records` matches the rows on disk; each record's source chunk exists at the level, and at delta 0 every endpoint's chunk |
| L3 | In an undirected canonical family, no offset is lexicographically negative and the offsets are non-decreasing |
| L4 | `links_convention` is one the type allows; see [Level 4](../validation/overview.md#level-4-geometry-conventions) |

No level checks that a link's vertex index exists within its chunk.

## Type selection guide

| Data | Type |
|------|------|
| Lidar, synchrotron point scan, single-molecule localisation | `point_cloud` |
| Pairs of connected points (short segments, contact sites) | `line` |
| Ordered vertex sequences | `polyline` |
| Tractography streamlines | `streamline` |
| General graph, possibly with cycles | `graph` |
| Neuronal morphology, vascular tree (SWC-compatible) | `skeleton` |
| 3-D surface mesh (cell boundary, brain surface, organelle hull) | `mesh` |

## Per-type documentation

```{toctree}
:maxdepth: 1

point_cloud
line
polyline
streamline
graph
skeleton
mesh
```
