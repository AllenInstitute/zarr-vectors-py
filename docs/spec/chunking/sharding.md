# Sharding

## Terms

**Shard**
: One storage object (a file, or an object in a bucket) holding the cells
  of several chunks of one per-chunk array.

**`shard_shape`**
: The number of chunk-grid cells per shard on each axis, for example
  `[4, 4, 4]` for 64 cells per shard. Counted in cells, not in coordinate
  units.

**Sharding codec**
: The Zarr v3 codec in an array's `zarr.json` that packs cells into
  shards: `sharding_indexed` (built into Zarr), or `zarr_octree_sharding`
  from the [zarr-octree](https://github.com/Andrew-Keenlyside/zarr-octree)
  package on a store marked octree.

**Shard index**
: The table in each shard giving the byte range of each cell, so a reader
  can fetch one cell without the whole shard.

---

## Introduction

Unsharded, each per-chunk array stores one object per non-empty chunk.
Many small objects are slow to list and copy, and costly on cloud
storage. Sharding packs a block of cells into one object.

Sharding changes only how bytes are packed: keys, cell contents and
metadata are unchanged, and any Zarr v3 reader reads a
`sharding_indexed` array. `zarr_vectors.sharding` is `internal`;
`zarr_vectors.building.shard_store` and `shard_object_layer` are
`supported`.

---

## Technical reference

### Declaring a shard shape

The root `zarr_vectors` block may hold `shard_shape`: an `int` for every
axis, or one value per axis. Every per-chunk array allocated afterwards
is sharded with it, including arrays a worker creates on first write.

```python
from zarr_vectors.building import create_store

create_store("scan.zarrvectors", bounds=[[0, 0, 0], [1000, 1000, 1000]],
             chunk_shape=(200., 200., 200.), geometry_types=["point_cloud"],
             shard_shape=4)
```

The writers take the same argument; their default,
`shard_shape="inherit"`, uses the root value. Each array's `zarr.json`
says how that array is actually packed.

### On disk

For a 5 × 5 × 5 chunk grid with shard shape 4, the `vertices` array's
`zarr.json` includes a Zarr chunk grid of one shard and a `sharding_indexed`
codec whose inner chunk is one cell:

```json
{
  "shape": [5, 5, 5],
  "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4, 4, 4]}},
  "codecs": [{
    "name": "sharding_indexed",
    "configuration": {
      "chunk_shape": [1, 1, 1],
      "codecs": [{"name": "vlen-bytes", "configuration": {}}]
    }
  }]
}
```

Shard objects sit at `0/vertices/c/<i>/<j>/<k>`, indexed by shard: this
array has `ceil(5 / 4)³ = 8` shard objects in place of 125 cell objects.
Array attributes (`nonempty_chunks`, `chunk_grid_origin`, `zv_array`) are
kept, so listing a level's chunks reads no shard index.

### Octree shards

A store marked octree has `octree_sharding` in its root `zarr_vectors`
block (the codec configuration, `{}` for defaults) and in
`format_capabilities`. Its sharded arrays use zarr-octree's
`zarr_octree_sharding` codec (by default `"layout": "octree"`,
`"target": 100000.0`, `"unit": "bytes"`). Keys and cells are as with
`sharding_indexed`. Inside a shard, cells are stored in Z order in runs
of about `target` bytes, and the index has one row per stored cell plus
a run table. A partial read fetches one byte range per run it touches.
Reading or writing these arrays needs zarr-octree.

`create_store(..., shard_shape=..., octree_sharding={})` creates a store
marked octree; `octree_sharding` without `shard_shape` raises
`MetadataError`.

### Converting a store

| Function | What it does |
|----------|--------------|
| `shard_store(path, *, shard_shape=8, arrays=None, object_shard_rows=None)` | Repacks every per-chunk array of every level (or those in `arrays`) with `sharding_indexed`. Skips arrays already sharded with that shape. Does not write the root `shard_shape` |
| `unshard_store(path, *, arrays=None)` | Rewrites every sharded array as one object per cell. Removes the octree marker once no octree array is left |
| `reshard(path, shard_shape, *, arrays=None)` | `shard_store` with the new shape, or `unshard_store` when `shard_shape` is `None` |
| `set_octree(path, octree=True, *, arrays=None)` | Marks the store octree and rewrites its sharded arrays with `zarr_octree_sharding`. `octree=False` reverses this. The store must be sharded with one shard shape, which is written to the root `shard_shape` if absent |
| `is_sharded(path)`, `get_shard_info(path)` | Report how the store's arrays are packed |

On an unsharded point cloud with a 5 × 5 × 5 chunk grid:

```python
from zarr_vectors.sharding import shard_store

shard_store("points.zarrvectors", shard_shape=4)
# {'arrays_sharded': 2, 'chunks_packed': 250, 'shard_shape': [4, 4, 4]}
```

The rewriting functions are coordinator operations: nothing else may
write the store meanwhile. `set_octree` builds each array in `<level>/__set_octree` and
swaps it in; rerun after an interruption, it finishes or discards that
copy.

`shard_store` finds cells through each array's `nonempty_chunks` list.
After workers write with `record_presence=False`, run it only once
`finalize_links` or `rebuild_presence` has rebuilt that list; before
then it silently packs nothing. On a level declared with
`defer_presence` it reads the cells from the store instead. See
[HPC pipelines](../../how_to/hpc_pipelines.md).

### The object layer

The object layer (`object_index/` arrays and `object_attributes/<name>`
columns) can be sharded along rows with `sharding_indexed`, in shards of
a positive multiple of 65,536 rows; other counts raise `ArrayError`.

```python
from zarr_vectors.building import shard_object_layer, shard_store

shard_store("scan.zarrvectors", shard_shape=2, object_shard_rows=65536)
shard_object_layer(level_group, None)    # one object per bucket again
```

Writers that add an object-layer array use the level's existing object
shard size.

### Choosing how shards are read

`ZARR_VECTORS_SHARD_READER` selects how sharded arrays are read:

| Value | Behaviour |
|-------|-----------|
| `auto` (default) | `optimised` for `zarr_octree_sharding` arrays, `standard` for others |
| `standard` | Local stores: each shard index once per batch, then one range per cell. Other stores: zarr's array API |
| `optimised` (or `optimized`) | Each shard index once per batch, cell byte ranges coalesced, on any store |
| `zarr` | zarr's array API only |

Any other value raises `ValueError` on the first sharded read.

### Validation

[`zarr_vectors.validate`](../validation/overview.md) reads sharded and
unsharded arrays alike and has no checks for `shard_shape`. On a level
with octree chunks, level 3 reports leaves that do not fit in one shard;
see [Octree chunks](octree_chunks.md#validation).
