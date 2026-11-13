# Geometry types

Zarr Vectors defines seven geometry types. A store names its types in the
`geometry_types` list of the `zarr_vectors` block in the root attributes; see
[Root metadata](../layout/root_metadata.md). Each writer described in these
pages creates a store with one geometry type.

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

What each type's writer stores at level 0. "Optional" means the writer stores
it when the caller passes the matching argument.

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

In the `implicit_sequential` types, vertex order within a fragment carries
the connectivity, so the writers store only the links that cross a chunk
boundary; a store whose objects all sit in one chunk has no `links/` group.
How links are stored is described in [Links](../object_model/links.md).

## Type selection guide

| Data description | Recommended type |
|-----------------|-----------------|
| Lidar, synchrotron point scan, single-molecule localisation | `point_cloud` |
| Pairs of connected points (short segments, contact sites) | `line` |
| Ordered vertex sequences | `polyline` |
| Tractography streamlines | `streamline` |
| General connectivity graph (not necessarily a tree) | `graph` |
| Neuronal morphology, vascular tree (tree topology, SWC-compatible) | `skeleton` |
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
