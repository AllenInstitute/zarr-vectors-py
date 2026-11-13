# Octree chunks

## Terms

**Cell**
: One cell of the level's uniform chunk grid (`chunk_shape`). The
  smallest possible leaf.

**Root cube**
: A cube of `2**D` cells per side starting at `origin_cell`, where `D` is
  `max_depth`.

**Leaf**
: An aligned cube of `2**(D - d)` cells per side at depth `d`. Each leaf
  is one chunk. Its address is `d/i.j.k`, its position on the grid of
  depth `d`, or its octant path, one `xyz` bit triple per depth from the
  root: `111/001/010` is `3/4.5.6`.

**Anchor**
: A leaf's minimum-corner cell. A leaf's data is stored at its anchor's
  key, so every per-chunk array stays an array over the cell grid.

**Balanced**
: Leaves that share a face, edge or corner differ by at most one depth.

**Leaf cap**
: The widest leaf allowed, in cells per side (`max_leaf_cells`, a power
  of two).

---

## Introduction

A level normally has uniform chunks: one chunk per cell. Where data is
clustered, some cells hold far more vertices than others. An
octree-chunked level groups cells into leaves sized by content: cells are
merged while the leaf stays at or under a `target` vertex count, so
sparse regions become a few large chunks and dense regions stay fine. A
single cell heavier than `target` stays one leaf.

Octree chunks are optional and convert to and from uniform chunks
without loss. They apply to point clouds and graphs with three spatial
dimensions. Most writers and readers work on uniform cells, so an octree
level is read by its own functions and written only by the conversions.
The module `zarr_vectors.octree_chunks` has stability tier `undecided`.

---

## Technical reference

### On disk

After `to_octree(store, target=500)` on a point cloud with
`chunk_shape = 100` and 5,000 clustered points:

The root `zarr_vectors` block lists the capability in both lists:

```json
{"format_capabilities": ["octree_chunks"], "required_capabilities": ["octree_chunks"]}
```

The level's attributes gain an `octree_chunks` block:

```json
{
  "zarr_vectors_level": {
    "level": 0,
    "vertex_count": 5000,
    "octree_chunks": {
      "max_depth": 4,
      "origin_cell": [0, 0, 0],
      "target": 500.0,
      "balanced": true,
      "max_leaf_cells": null,
      "shard_shape": null,
      "shard_origin": null,
      "fit_shards": true,
      "weight": "vertices",
      "leaves": 69
    }
  }
}
```

| Key | Meaning |
|-----|---------|
| `max_depth` | `D`: the root cube is `2**D` cells per side |
| `origin_cell` | The root cube's minimum-corner cell |
| `target` | The largest leaf vertex count asked for |
| `balanced` | Whether the plan was balanced |
| `max_leaf_cells` | The leaf cap; `null` when uncapped |
| `shard_shape`, `shard_origin` | The shard grid the leaves were fitted to, in cells; `null` when the level was unsharded or not fitted |
| `fit_shards` | `false` when the conversion opted out of fitting leaves to shards |
| `weight` | What leaves are weighed by: `"vertices"` |
| `leaves` | Number of stored (non-empty) leaves |

During a conversion the block is `{"converting": "to_octree"}` or
`{"converting": "to_uniform"}`.

The level holds these arrays:

| Path | Contents |
|------|----------|
| `<level>/octree_chunks/leaves` | `(N, 5)` int64: `depth, i, j, k, weight` per stored leaf |
| `<level>/octree_chunks/cells` | `(C, 6)` int64: `i, j, k` (cell), `leaf`, `vertices`, `fragments`, in storage order |
| `vertices`, `vertex_fragments`, `object_index` | As for uniform chunks, keyed by each leaf's anchor. Manifest entries name `(anchor, fragment)` |
| `links/0/<segment>` | Keyed by the source leaf's anchor; see [Links](#links) |

A leaf's vertices are its cells' vertices end to end, cells in Z order,
and its fragments are its cells' fragments in the same order; the cell
table records where each cell's run starts.

### Marker codec

Every per-chunk array of an octree level ends its cell codecs with the
marker codec `{"name": "zarr_vectors_octree_chunks"}`. When the level is
sharded, the marker sits inside the sharding codec's inner codecs:

```json
[{"name": "sharding_indexed",
  "configuration": {"chunk_shape": [1, 1, 1],
                    "codecs": [{"name": "vlen-bytes", "configuration": {}},
                               {"name": "zarr_vectors_octree_chunks"}],
                    "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}},
                                     {"name": "crc32c"}],
                    "index_location": "end"}}]
```

The marker leaves bytes unchanged. A Zarr reader must know every codec of
an array to open it, so a reader that has not registered this one fails
on the level's first array (zarr-python raises
`UnknownCodecError: Unknown codec: 'zarr_vectors_octree_chunks'`) instead
of reading each leaf as one cell. Importing `zarr_vectors` registers it
(`zarr_vectors.encoding.octree_marker`).

`required_capabilities` gives a reader that checks it a clearer refusal.
Both capability lists hold `octree_chunks` while any level has an octree
block, including one mid-conversion; other entries are left as they
are, and `required_capabilities` is removed when `octree_chunks` was its
only entry. A store converted and converted back has the same root
`zarr.json` it had.

### Links

A link record is filed under its source leaf: the deeper endpoint, or at
equal depth the one with the smaller anchor. The other endpoint, `Δ`
depths coarser, is named by the offset `o` of its leaf from the source's
ancestor at that depth, always one grid step:

```
other = (d - Δ, (i >> Δ) + o_x, (j >> Δ) + o_y, (k >> Δ) + o_z),   o in {-1, 0, 1}³
```

The segment name is `o` as for uniform links, with `^Δ` appended when
`Δ > 0` (`format_link_segment`, `parse_link_segment`):

| Segment | Other endpoint |
|---------|----------------|
| `0.0.0` | The same leaf |
| `+1.0.0` | The same-depth leaf along +x |
| `-1.0.0^1` | The leaf along −x of the source's parent |

Rows are `(source row, other row)`, each local to its leaf. A balanced
level has at most 26 + 26 segments besides `0.0.0`. Links must join
adjacent cells, and the level must not hold links to other levels.

### Converting a level

```python
to_octree(store, *, target, level=0, balanced=True, max_leaf_cells=None,
          fit_shards=True, compressor="inherit", backend=None) -> dict
to_uniform(store, *, level=0, compressor="inherit", backend=None) -> dict
```

```python
from zarr_vectors.octree_chunks import to_octree, to_uniform

to_octree("points.zarrvectors", target=500)
# {'leaves': 69, 'cells': 114, 'links': 0, 'max_leaf_vertices': 418,
#  'max_cell_vertices': 355, 'depths': {3: 31, 4: 38}, 'max_leaf_cells': None}
to_uniform("points.zarrvectors")
# {'cells': 114, 'links': 0}
```

`to_octree` refuses, with `OctreeError` and without changing anything, a
level that:

- belongs to a store whose `geometry_types` are not all `point_cloud` or
  `graph`, or whose spatial dimensionality is not 3;
- has `vertex_attributes`, `fragment_attributes` or `link_attributes`;
- has directed links, links of width other than 2, or links not stored
  once (`canonical`);
- is [attribute-chunked](attribute_chunking.md);
- has links into or out of another level, or already has octree chunks.

Object ids are kept. `to_uniform` restores every cell's vertex rows,
fragments, manifest entries and links; it too refuses while links join
the level to another.

### Planning leaves

`plan_leaves(cells, weights, *, target, origin_cell=None, max_depth=None,
balanced=True, max_leaf_cells=None, shard_shape=None, shard_origin=None,
fit_shards=True)` returns the `LeafTable` a conversion would use, without
touching a store:

```python
import numpy as np
from zarr_vectors.octree_chunks import plan_leaves

t = plan_leaves(np.array([[0, 0, 0], [1, 0, 0], [5, 5, 5], [6, 6, 6], [7, 7, 7]]),
                np.array([10, 10, 50, 60, 5]), target=100)
t.to_rows().tolist()   # [[1, 0, 0, 0, 20], [2, 2, 2, 2, 50], [2, 3, 3, 3, 65]]
[t.key(i) for i in range(len(t))]   # ['1/0.0.0', '2/2.2.2', '2/3.3.3']
t.max_depth            # 3
```

The root cube is split while a node weighs more than `target` or, when it
holds anything, is wider than the leaf cap. Empty space is not stored.
Balancing only splits leaves.

### Leaves and shards

On a sharded level, `to_octree` fits the leaves to the shard grid of the
level's `vertices` array (or the root `shard_shape` when that array is
not sharded), so every leaf lies inside one shard:

- the root cube starts on a shard boundary;
- the leaf cap is the largest power of two dividing every shard side:
  4 for shards of `[4, 4, 4]`, 2 for `[4, 2, 8]` or `[6, 6, 6]`.

`max_leaf_cells` lowers the cap further, or sets one on an unsharded
level. It must be a power of two and, on a fitted sharded level, no
wider than the shard's cap; otherwise `to_octree` raises `OctreeError`.

With `fit_shards=False`, a sharded level is planned as if unsharded:
leaves may straddle shards, `max_leaf_cells` may be any power of two or
`None`, and the block records `"fit_shards": false` with `shard_shape`
and `shard_origin` `null`. A leaf's data is then in its anchor's shard
while its cells span several, so shards cannot be written or read on
their own.

Fitting costs chunk count where shards are small or not powers of two,
since large leaves are split at shard boundaries. Power-of-two shard
sides at least as wide as the largest leaves the target produces cost
nothing.

Both conversions keep the level's packing. `shard_store`, `reshard` and
`unshard_store` move cells unchanged, keep the marker codec, and are
allowed on an octree level; a new shard shape can split leaves, which
`validate` reports. Run `to_uniform` then `to_octree` to fit the leaves
to new shards.

### Reading an octree level

`read_points` and `read_graph` read an octree level whole, by `bbox`, or
by `object_ids`. They raise `OctreeError` for `chunks=`,
`attribute_filter=` and (`read_points`) `attribute_names=`.

`read_octree(store, *, level=0, leaves=None, bbox=None, object_ids=None,
halo=0, backend=None)` reads by leaf index or address, by box or by
object. `halo=1` with `leaves` adds the adjacent leaves. It returns
`positions`, `leaf` and `row` (where each vertex is stored), `edges` for
graphs, and `object_ids` on an object read.

`read_seams(store, leaf, *, level=0, backend=None)` returns the links
between one leaf and its neighbours, as arrays `row`, `other_leaf` and
`other_row`: what a per-leaf graph builder stitches across.

`leaf_table(level_group)`, `cell_table(level_group)` and
`octree_meta(level_group)` return the tables and block.

### What refuses an octree level

Only the two conversions and the shard repackers write an octree level.
These raise `OctreeError` before changing anything, with a message that
says to run `to_uniform` first:

- the type writers (`write_points`, `write_graph` and the others, and
  the `Dataset` methods that call them) when the level they write has
  octree chunks;
- edits (`Dataset.editing`, `EditSession`, the `zarr_vectors.ops`
  functions);
- `coarsen_level` and `build_pyramid`;
- `rechunk`, `rechunk_spatial` and `rebin_level`;
- the uniform-cell readers `Level.cells()`, `Query.cells()`,
  `read_neighbourhood`, `read_chunk_vertices` and `read_chunk_links`.

`shard_store`, `unshard_store` and `set_octree` refuse a store with a
level part way through a conversion. `zarr_vectors.octree_chunks.guard`
provides the checks: `is_octree_level`, `octree_levels`,
`refuse_octree_write`, `refuse_octree_read`, `refuse_octree_levels` and
`refuse_unfinished_conversions`, and the `converting` context that marks
the handles a conversion writes through.

### Interrupted conversions

A conversion plans and checks everything before it writes, then:

1. writes the new layout into `<level>/octree_chunks_staging` and seals
   it with an `octree_chunks_staging` attribute recording the direction,
   the arrays it holds and the final level metadata;
2. sets the level's block to `{"converting": ...}` and updates the root
   capabilities, in one write each (the commit point);
3. replaces `vertices`, `vertex_fragments`, `links`, `link_fragments`,
   `object_index` and `octree_chunks` with their staged copies, or
   removes them;
4. sets the level metadata from the record and removes the staging
   group.

If the process stops before step 2, the next call discards the staging
group and the level is as it was. After step 2, reads of the level raise
`OctreeError` ("part way through to_octree"), and calling the same
conversion again finishes the swap and returns its summary with
`"resumed": True`.

### Validation

[`zarr_vectors.validate`](../validation/overview.md) checks an octree
level with its own checks in place of the uniform ones:

| Level | Check | Result |
|-------|-------|--------|
| 2 | A level has an octree block but `format_capabilities` or `required_capabilities` lacks `octree_chunks`, or a list names it and no level has one | warning; `refresh_capabilities(store)` sets both lists from the levels |
| 3 | The level is mid-conversion | error |
| 3 | The leaf and cell tables are readable; every leaf anchor has a vertex chunk and every vertex chunk is at an anchor; the cell table names existing leaves | error |
| 3 | Each leaf's vertex and fragment counts match the cell and leaf tables; `vertex_count` matches the total | error |
| 3 | Every manifest entry names a leaf anchor and a fragment that leaf holds | error |
| 3 | No `links` to other levels; every link names a stored leaf and a row inside it; `num_physical_records` matches | error |
| 3 | `vertices` carries the marker codec | warning |
| 3 | No leaf is wider than the recorded `max_leaf_cells` | error |
| 3 | On a sharded level, every leaf lies inside one shard of the arrays' current shard grid | error when the level was converted with a cap; warning when converted without one; not checked with `fit_shards=False` |

After `reshard` to `[2, 2, 2]` on a level fitted to `[4, 4, 4]` shards,
level 3 reports:

```
ERROR: resolution_0: octree chunks: 1 leaves are not inside one shard of [2, 2, 2] cells,
e.g. leaf 1/0.0.0 (4 cells per side at cell (0, 0, 0)), ... run
zarr_vectors.octree_chunks.to_uniform, then to_octree, to fit the leaves to these shards
```
