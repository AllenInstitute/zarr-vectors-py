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
: The table at the end of each shard giving the byte range of each cell
  in it, so a reader can fetch one cell without the whole shard.

---

## Introduction

Each per-chunk array of a level stores one object per non-empty chunk:
`0/vertices/c/i/j/k`. A store with many small chunks has many objects,
which is slow to list and copy and, on cloud storage, costly per request.
Sharding packs the cells of a block of chunks into one object, so a
`[4, 4, 4]` shard shape cuts the object count of a dense array by up to
64.

Sharding changes only how each array's bytes are packed. Chunk keys,
cell contents, the fragment index and all Zarr Vectors metadata are the
same with or without it, and any Zarr v3 reader reads a
`sharding_indexed` array. The functions in `zarr_vectors.sharding` have
stability tier `internal`; `zarr_vectors.building.shard_store` and
`shard_object_layer` are `supported`.

---

## Technical reference

### Declaring a shard shape

A store records its shard shape as `shard_shape` in the root
`zarr_vectors` block. Every per-chunk array allocated afterwards is
sharded with it, including arrays a worker creates on first write. An
`int` applies to every axis; a list gives one value per axis.

```python
from zarr_vectors.building import create_store

create_store("scan.zarrvectors", bounds=[[0, 0, 0], [1000, 1000, 1000]],
             chunk_shape=(200., 200., 200.), geometry_types=["point_cloud"],
             shard_shape=4)
```

The root attributes then hold `"shard_shape": 4`. The writers take the
same argument (`write_points(..., shard_shape=4)` records it on a new
store). Their default, `shard_shape="inherit"`, uses the root value; an
explicit value applies to the arrays that call creates.

The root `shard_shape` is a default for writers. Each array's
`zarr.json` says how that array is packed.

### On disk

For a 5 × 5 × 5 chunk grid with shard shape 4, the `vertices` array's
`zarr.json` has a Zarr chunk grid of one shard and a `sharding_indexed`
codec whose inner chunk is one cell:

```json
{
  "shape": [5, 5, 5],
  "data_type": "variable_length_bytes",
  "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4, 4, 4]}},
  "codecs": [{
    "name": "sharding_indexed",
    "configuration": {
      "chunk_shape": [1, 1, 1],
      "codecs": [{"name": "vlen-bytes", "configuration": {}}],
      "index_codecs": [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "crc32c"}
      ],
      "index_location": "end"
    }
  }]
}
```

Shard objects sit at `0/vertices/c/<i>/<j>/<k>`, where the indices
count shards: this array has `ceil(5 / 4)³ = 8` shard objects in place
of 125 cell objects. The shard index locates each cell.

Array attributes (`nonempty_chunks`, `chunk_grid_origin`, `zv_array`
and so on) are kept, so listing a level's chunks does not read any shard
index.

### Octree shards

A store can be marked octree: the root `zarr_vectors` block gains
`octree_sharding` (the `zarr_octree_sharding` codec configuration, `{}`
for its defaults) and `format_capabilities` gains `octree_sharding`.
Its sharded arrays then use zarr-octree's codec:

```json
{"name": "zarr_octree_sharding",
 "configuration": {"version": 2, "chunk_shape": [1, 1, 1],
                   "codecs": [{"name": "vlen-bytes", "configuration": {}}],
                   "layout": "octree", "target": 100000.0, "unit": "bytes"}}
```

Keys and cells are the same as with `sharding_indexed`; only the bytes
inside a shard differ. Cells are stored in Z order and grouped into runs
(the leaves of an octree aimed at `target` bytes), and the shard index is
sparse: one row per stored cell plus a run table. A partial read fetches
one byte range per run it touches. Reading or writing such an array needs
the zarr-octree package.

`create_store(..., shard_shape=..., octree_sharding={})` creates a store
marked octree; `octree_sharding` without `shard_shape` raises
`MetadataError`.

### Converting a store

```python
from zarr_vectors.sharding import (shard_store, unshard_store, reshard,
                                   set_octree, get_shard_info, is_sharded)

shard_store("scan.zarrvectors", shard_shape=4)
# {'arrays_sharded': 2, 'chunks_packed': 250, 'shard_shape': [4, 4, 4]}
get_shard_info("scan.zarrvectors")
# {'sharded': True, 'arrays': [{'name': '0/vertex_fragments', 'grid_shape': [5, 5, 5],
#   'shard_shape': [4, 4, 4], 'shard_count': 8}, {'name': '0/vertices', ...}],
#  'shard_count': 16}
reshard("scan.zarrvectors", 2)          # {'action': 'shard', 'arrays_sharded': 2, ...}
set_octree("scan.zarrvectors")          # {'octree': {}, 'arrays_repacked': 2, 'chunks_repacked': 250}
set_octree("scan.zarrvectors", False)   # back to sharding_indexed
unshard_store("scan.zarrvectors")       # {'arrays_unsharded': 2, 'chunks_extracted': 250}
reshard("scan.zarrvectors", None)       # {'action': 'noop', 'message': 'already unsharded'}
```

| Function | What it does |
|----------|--------------|
| `shard_store(path, *, shard_shape=8, arrays=None, object_shard_rows=None)` | Repacks every per-chunk array of every level (or those named in `arrays`) with `sharding_indexed`. Arrays already sharded with that shape are skipped. Does not write the root `shard_shape`. |
| `unshard_store(path, *, arrays=None)` | Rewrites every sharded array as one object per cell. On a store marked octree, removes the marker once no octree array is left. |
| `reshard(path, shard_shape, *, arrays=None)` | `shard_store` with the new shape, or `unshard_store` when `shard_shape` is `None`. |
| `set_octree(path, octree=True, *, arrays=None)` | Marks the store octree and rewrites its sharded arrays with `zarr_octree_sharding`; `octree=False` rewrites them with `sharding_indexed` and removes the marker. The store must be sharded with one shard shape, which is written to the root `shard_shape` if absent. |
| `is_sharded(path)`, `get_shard_info(path)` | Report the packing of the store's arrays. |

`shard_store`, `unshard_store` and `set_octree` rewrite arrays and are
coordinator operations: nothing else may write the store meanwhile.
`set_octree` builds each array in a scratch group `<level>/__set_octree`
and swaps it in when complete; running it again after an interruption
finishes or discards that copy and continues.

`shard_store` finds each array's cells through its `nonempty_chunks`
list. Workers that write with `record_presence=False` leave that list
empty until the coordinator rebuilds it (`finalize_links`,
`rebuild_presence`), so run `shard_store` after the rebuild; before it,
`shard_store` finds no cells and packs nothing, without error. A level
declared with `defer_presence` has no list, and `shard_store` reads its
cells from the store. See [HPC pipelines](../../how_to/hpc_pipelines.md).
A store created with a root `shard_shape` is written sharded and needs no
repacking.

### The object layer

The object layer (`object_index/` arrays and every
`object_attributes/<name>` column) is one-dimensional along objects,
stored in row buckets of 16,384 rows (object index arrays) or 65,536
rows (dense `manifest_blocks` and attribute columns). It can be sharded
along rows with `sharding_indexed`, with shards a positive multiple of
65,536 rows:

```python
from zarr_vectors.building import shard_object_layer, shard_store

shard_store("scan.zarrvectors", shard_shape=2, object_shard_rows=65536)
shard_object_layer(level_group, None)    # one object per bucket again
```

Other row counts raise `ArrayError`. The layout is recorded only in each
array's `zarr.json`. Writers that add or rewrite an object-layer array
use the shard size the level's object arrays already have.

### Choosing how shards are read

The environment variable `ZARR_VECTORS_SHARD_READER` selects how
zarr-vectors reads sharded arrays:

| Value | Behaviour |
|-------|-----------|
| `auto` (default) | `optimised` for arrays with `zarr_octree_sharding`, `standard` for all others |
| `standard` | A local store's shards are read from the file system: each shard index once per batch, then one range per wanted cell. Other stores go through zarr's array API |
| `optimised` (or `optimized`) | Each shard index read once per batch, and the wanted cells' byte ranges coalesced, for local and remote stores alike |
| `zarr` | zarr's array API only |

Any other value raises `ValueError` on the first sharded read.

### Validation

[`zarr_vectors.validate`](../validation/overview.md) reads sharded and
unsharded arrays alike and has no checks of its own for `shard_shape`.
On a level with [octree chunks](octree_chunks.md), level 3 reports
leaves that do not fit inside one shard; see
[Octree chunks](octree_chunks.md#validation).
