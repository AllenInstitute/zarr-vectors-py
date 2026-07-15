# Geometry types

ZVF defines seven geometry types. Each type is identified by a string
constant stored in the root `.zattrs` under `"geometry_type"`. A single
ZVF store holds exactly one geometry type; multi-type datasets require
multiple stores.

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

| Property | `point_cloud` | `line` | `polyline` | `streamline` | `graph` | `skeleton` | `mesh` |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Discrete objects | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `links/` family | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `link_width` | — | 2 | 2 | 2 | 2 | 2 | ≥ 3 |
| Intra-chunk links (`0.0.0`) | — | — | — | — | ✓ | ✓ | ✓* |
| Cross-chunk links (non-zero offsets) | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Family `directed` | — | false | false | false | false | **true** | false |
| `object_index/` | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `object_attributes/` | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `groupings/` | — | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| Per-vertex attributes | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Object sparsity | — | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| Multiscale pyramid | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Draco compression | — | — | — | — | — | — | ✓ |

\* Mesh intra-chunk faces move into the Draco bitstream under
`encoding="draco"`, leaving only boundary faces in `links/`.

There is no `cross_chunk_links/` family: connectivity is one family,
and an intra-chunk link is simply one whose offsets are all zero.
`line`, `polyline`, and `streamline` use
`links_convention: implicit_sequential` — vertex order within a
fragment carries the topology — so they emit **only** non-zero-offset
links, and a store whose objects never cross a chunk has no `links/`
group at all.

There are no `is_directed` or `is_tree` root keys; `directed` is
per-family link policy, and tree-ness is carried by `geometry_type`
plus `links_convention`. See [Links](../object_model/links.md).

## Type selection guide

| Data description | Recommended type |
|-----------------|-----------------|
| Lidar, synchrotron point scan, single-molecule localisation | `point_cloud` |
| Pairs of connected points (short segments, contact sites) | `line` |
| Ordered vertex sequences without tractography metadata | `polyline` |
| MRI/synchrotron tractography streamlines with step size / seeding metadata | `streamline` |
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
