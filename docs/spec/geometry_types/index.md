# Geometry types

Zarr Vectors defines seven geometry types. Each type is identified by a string
constant stored in the root `zarr_vectors` attribute block under
`"geometry_types"` (a list, e.g. `["streamline"]`). A store normally holds one
geometry type, and the dataset API (`zarr_vectors.open`) reads a store as the
first type listed. A `zarr_vectors.composite` module can add further types
to one store, but treat multi-type stores as unsupported for the common
readers; use multiple stores.

Every geometry type has a `write_*` / `read_*` pair in `zarr_vectors.types`:

| Type | Writer | Reader | Dataset API |
|------|--------|--------|-------------|
| `point_cloud` | `write_points` | `read_points` | `ds.add_points` |
| `line` | `write_lines` | `read_lines` | `ds.add_lines` |
| `polyline`, `streamline` | `write_polylines(geometry_type=...)` | `read_polylines` | `ds.add_polylines(streamlines=...)` |
| `graph`, `skeleton` | `write_graph(kind=...)` | `read_graph` | `ds.add_graph(tree=...)` |
| `mesh` | `write_mesh` | `read_mesh` | `ds.add_mesh` |

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
| Discrete objects | optional† | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `links/` family | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `link_width` | — | 2 | 2 | 2 | 2 | 2 | ≥ 3 |
| `links_convention` | — | implicit_sequential | implicit_sequential | implicit_sequential | explicit | implicit_sequential_with_branches | explicit |
| Intra-chunk links (`0.0.0`) | — | — | — | — | ✓ | ✓‡ | ✓* |
| Cross-chunk links (non-zero offsets) | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Family `directed` | — | false | false | false | false | false (`write_graph`); **true** (streaming writers) | false |
| `object_index/` | optional† | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `object_attributes/` | optional† | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `groups/` written by `write_*` | ✓† | — | ✓ | ✓ | — | — | — |
| Per-vertex attributes | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Multiscale pyramid (core `per_object`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Draco compression | — | — | — | — | — | — | ✓ |

† Only when `object_ids=` / `object_attributes=` / `groups=` is passed to
`write_points`.

‡ A skeleton stores only the edges that depth-first order does not imply:
branch points and fragment starts intra-chunk, plus every cross-chunk edge.

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
per-family link policy, and tree-ness is carried by `geometry_types`
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
