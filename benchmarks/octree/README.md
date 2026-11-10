# Spatial read benchmark: point clouds, meshes, networks

Working branch `octree`: zarr-vectors with an optional octree sharding codec
(zarr-octree's `zarr_octree_sharding`) and a choice of shard reader.  Neither
is a format change to zarr-vectors; both act on one zarr array at a time.

| Setting | Values |
|---|---|
| `ZARR_VECTORS_SHARD_CODEC` | unset: `sharding_indexed` (default); `zarr_octree_sharding` (needs zarr-octree) |
| `ZARR_VECTORS_SHARD_CODEC_CONFIG` | JSON for the codec, e.g. `{"target": 100000}` (its default) |
| `ZARR_VECTORS_SHARD_READER` | `optimised` (default: each shard index once, coalesced ranges) or `zarr` (zarr's API only) |

## Running

```
python bench_kinds.py [--quick] [--layouts indexed octree] [--kinds points mesh network] [--vertices 4000000]
python figures_kinds.py            # needs matplotlib
```

`--layouts indexed` runs without zarr-octree.  Stores go to `_stores/`,
results to `results/bench_kinds.json`, figures to `results/figures/`.

Data (`kinds_data.py`, synthetic): point clouds, meshes (closed surfaces)
and networks (branching trees) with the same vertex count and the same
uneven density.  Queries: the whole level, one vertex (a 0.5-unit box), one
or more objects, boxes; local, 20 ms and 80 ms per request
(`latency_store.py`); every store and reader must return the same answer.

## The structure being read

Each level of a zarr-vectors store is a group of zarr arrays, all chunked
on the same spatial grid of cells (sharded `--shard_cells` per side):

| Array | Holds | Point cloud | Mesh | Graph |
|---|---|---|---|---|
| `vertices` | positions, one ragged row block per cell | yes | yes | yes |
| `vertex_fragments` | per cell: which rows belong to which fragment (object part) | yes | yes | yes |
| `links/0/<offsets>` | faces / edges, one array per neighbour offset between their vertices' cells | – | 61 arrays here | 15 arrays here |
| `link_fragments` | fragment index over the intra-cell links | – | yes | yes |
| `object_index/manifests` | per object: its (cell, fragment) pairs | yes | yes | yes |

Reads go: object manifest or box → cells → fragment indexes → vertex rows →
links in those cells, kept when every endpoint was read.

## Current limitations (as found)

* **Graphs cannot be read by object.**  The object index is written, but
  `read_graph(object_ids=)` was never implemented and refuses
  (`select(objects=)` on a graph or skeleton too).  Meshes can (they cannot
  be read by group).  The mesh object path would serve graphs as well.
* **Each array's metadata lists every non-empty cell** (`nonempty_chunks`):
  120 kB per array at 20k points, 685 kB at 4 M.  A cold read fetches it
  for each array it touches, so it dominates small reads.
* **Links are split into one array per neighbour offset** (61 for these
  meshes): each with its own metadata and shards, so a read touches many
  arrays, and links are most of a mesh's or graph's bytes.
* **Through zarr's API, a whole-level read re-reads a shard's index for every
  cell**: GBs read from a store of MBs, set by the cell count, not the data.
