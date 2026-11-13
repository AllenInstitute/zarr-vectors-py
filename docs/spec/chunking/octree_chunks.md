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
  depth `d`.

**Anchor**
: A leaf's minimum-corner cell. A leaf's data is stored at its anchor's
  key.

**Balanced**
: Leaves that share a face, edge or corner differ by at most one depth.

**Leaf cap**
: The widest leaf allowed, in cells per side (`max_leaf_cells`, a power
  of two).

---

## Introduction

An octree-chunked level merges uniform cells into leaves sized by
content. Cells merge while a leaf holds at most `target` vertices, so
sparse regions become a few large chunks and dense regions stay fine. A
single cell heavier than `target` stays one leaf.

Octree chunks convert to and from uniform chunks without loss. They
apply to 3-D point clouds and graphs. The module `zarr_vectors.octree_chunks` has stability tier
`undecided`.

---

## Technical reference

### On disk

After `to_octree(store, target=500)` on 5,000 clustered points with
`chunk_shape = 100`, the level's `zarr_vectors_level` attributes gain an
`octree_chunks` block:

```json
{
  "zarr_vectors_level": {
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
| `shard_shape`, `shard_origin` | The shard grid the leaves fit, in cells; `null` when unsharded or not fitted |
| `fit_shards` | `false` when the conversion did not fit leaves to shards |
| `weight` | What leaves are weighed by: `"vertices"` |
| `leaves` | Number of stored (non-empty) leaves |

During a conversion the block is `{"converting": "to_octree"}` or
`{"converting": "to_uniform"}`.

| Path | Contents |
|------|----------|
| `<level>/octree_chunks/leaves` | `(N, 5)` int64: `depth, i, j, k, weight` per stored leaf |
| `<level>/octree_chunks/cells` | `(C, 6)` int64: `i, j, k` (cell), `leaf`, `vertices`, `fragments`, in storage order |
| `vertices`, `vertex_fragments`, `object_index` | As for uniform chunks, keyed by each leaf's anchor. Manifest entries name `(anchor, fragment)` |
| `links/0/<segment>` | Keyed by the source leaf's anchor; see [Links](#links) |

A leaf holds its cells' vertices and fragments end to end, in Z order.
The cell table records where each cell's run starts.

### Marker codec

Every per-chunk array of an octree level ends its cell codecs (inside
the sharding codec, if any) with `{"name": "zarr_vectors_octree_chunks"}`.
The marker leaves bytes unchanged. A reader that does not know it fails
(zarr-python raises `UnknownCodecError`) instead of misreading leaves as
cells. Importing `zarr_vectors` registers it.

The root `format_capabilities` and `required_capabilities` both list
`octree_chunks` while any level has an octree block, including
mid-conversion. `required_capabilities` is removed when `octree_chunks`
was its only entry, so converting back restores the original root
`zarr.json`.

### Links

A link is filed under its source leaf: the deeper endpoint, or at equal
depth the one with the smaller anchor. The other endpoint, `Δ` depths
coarser, is named by its offset `o` from the source's ancestor at that
depth:

```
other = (d - Δ, (i >> Δ) + o_x, (j >> Δ) + o_y, (k >> Δ) + o_z),   o in {-1, 0, 1}³
```

The segment name is `o` as for uniform links, with `^Δ` appended when
`Δ > 0`: `0.0.0` is the same leaf, `+1.0.0` the same-depth leaf along +x,
`-1.0.0^1` the leaf along −x of the source's parent. Rows are
`(source row, other row)`, each local to its leaf. A balanced level has
at most 26 + 26 segments besides `0.0.0`. Links must join adjacent
cells, and the level must not link to other levels.

### API

| Function | What it does |
|----------|--------------|
| `to_octree(store, *, target, level=0, balanced=True, max_leaf_cells=None, fit_shards=True, compressor="inherit", backend=None)` | Converts a uniform level; returns a summary dict |
| `to_uniform(store, *, level=0, compressor="inherit", backend=None)` | Converts back, restoring every cell's rows, fragments, manifest entries and links |
| `plan_leaves(cells, weights, *, target, origin_cell=None, max_depth=None, balanced=True, max_leaf_cells=None, shard_shape=None, shard_origin=None, fit_shards=True)` | Returns the `LeafTable` a conversion would use, without touching a store |
| `read_octree(store, *, level=0, leaves=None, bbox=None, object_ids=None, halo=0, backend=None)` | Reads by leaf index or address, box or object. `halo=1` with `leaves` adds adjacent leaves. Returns `positions`, `leaf`, `row`, `edges` (graphs) and `object_ids` (object reads) |
| `read_seams(store, leaf, *, level=0, backend=None)` | Links between one leaf and its neighbours, as `row`, `other_leaf`, `other_row` |
| `leaf_table`, `cell_table`, `octree_meta` | Take a level group; return the tables and the block |

```python
from zarr_vectors.octree_chunks import to_octree, to_uniform

to_octree("points.zarrvectors", target=500)
# {'leaves': 69, 'cells': 114, 'links': 0, 'max_leaf_vertices': 418, ...}
to_uniform("points.zarrvectors")
# {'cells': 114, 'links': 0}
```

`read_points` and `read_graph` also read an octree level: whole, by
`bbox` or by `object_ids`.

### Rules

**Planning.** The root cube is split while a node weighs more than
`target` or, when it holds anything, is wider than the leaf cap. Empty
space is not stored. Balancing only splits leaves.

**What `to_octree` refuses.** It raises `OctreeError`, changing nothing,
for a level that:

- is in a store whose `geometry_types` are not all `point_cloud` or
  `graph`, or that is not 3-D;
- has `vertex_attributes`, `fragment_attributes` or `link_attributes`;
- has directed links, links of width other than 2, or links not stored
  once (`canonical`);
- is [attribute-chunked](attribute_chunking.md);
- has links to or from another level, or already has octree chunks.

`to_uniform` also refuses while links join the level to another. Object
ids are kept.

**Shards.** On a sharded level, `to_octree` fits leaves to the shard grid
of the level's `vertices` array (or the root `shard_shape`), so each leaf
lies in one shard. The root cube starts on a shard boundary. The leaf cap
is the largest power of two dividing every shard side: 4 for `[4, 4, 4]`,
2 for `[4, 2, 8]` or `[6, 6, 6]`. `max_leaf_cells` lowers the cap, or
sets one when unsharded. It must be a power of two no wider than the
shard's cap, or `to_octree` raises `OctreeError`.

With `fit_shards=False`, leaves may straddle shards and the block records
`shard_shape` and `shard_origin` as `null`. Shards then cannot be read or
written on their own.

`shard_store`, `reshard` and `unshard_store` work on an octree level and
keep the marker codec. A new shard shape can split leaves, which
`validate` reports. Run `to_uniform` then `to_octree` to refit.

**What refuses an octree level.** These raise `OctreeError` before
changing anything, saying to run `to_uniform` first:

- the type writers (`write_points`, `write_graph` and the others) and the
  `Dataset` methods that call them;
- edits (`Dataset.editing`, `EditSession`, `zarr_vectors.ops`);
- `coarsen_level` and `build_pyramid`;
- `rechunk`, `rechunk_spatial` and `rebin_level`;
- `Level.cells()`, `Query.cells()`, `read_neighbourhood`,
  `read_chunk_vertices` and `read_chunk_links`;
- `read_points` and `read_graph` with `chunks=` or `attribute_filter=`,
  and `read_points` with `attribute_names=`.

`shard_store`, `unshard_store` and `set_octree` refuse a store with a
level mid-conversion.

### Interrupted conversions

A conversion checks everything, writes the new layout into
`<level>/octree_chunks_staging`, then commits by setting the level's
block to `{"converting": ...}`. It then swaps the staged arrays in and
removes the staging group. If it stops before the commit, the next call
discards the staging group. After the commit, reads raise `OctreeError`
until the same conversion is called again, which finishes it and returns
`"resumed": True`.

### Validation

[`zarr_vectors.validate`](../validation/overview.md) runs these checks on
an octree level in place of the uniform ones:

| Level | Check | Result |
|-------|-------|--------|
| 2 | Capability lists name `octree_chunks` exactly when a level has an octree block (`refresh_capabilities(store)` fixes them) | warning |
| 3 | The level is mid-conversion | error |
| 3 | Leaf and cell tables are readable and match the stored chunks, counts and `vertex_count` | error |
| 3 | Every manifest entry and link names a stored leaf and a fragment or row in it; no links to other levels; `num_physical_records` matches | error |
| 3 | `vertices` carries the marker codec | warning |
| 3 | No leaf is wider than `max_leaf_cells` | error |
| 3 | On a sharded level, each leaf lies in one shard of the current shard grid | error if converted with a cap; warning without one; skipped with `fit_shards=False` |
