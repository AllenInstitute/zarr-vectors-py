# Octree chunks (optional extension)

A level normally has **uniform chunks**: one chunk per cell of a regular grid
of `chunk_shape`.  An octree-chunked level groups those cells into **leaves**
of an octree sized by content, so that no chunk holds more than a target
number of vertices (unless a single cell does).  It is optional, marked, and
converts to and from uniform chunks without loss.

## Terms

**Cell**
: A cell of the level's uniform grid (`chunk_shape`).  The finest possible
  leaf.

**Root cube**
: `2**D` cells per side from `origin_cell`; `D` is `max_depth`.

**Leaf**
: An aligned cube of `2**(D - d)` cells per side at depth `d`, addressed
  `d/i.j.k` (its position on the depth-`d` grid) or by its octant path, one
  `xyz` bit triple per depth from the root (`111/001/010` is `3/4.5.6`).
  Each leaf is one chunk.

**Anchor**
: A leaf's minimum-corner cell.  A leaf is stored at its anchor, so every
  per-chunk array stays a plain array over the cell grid.

**Balanced**
: Adjacent leaves (sharing a face, edge or corner) differ by at most one
  depth.

**Leaf cap**
: The widest leaf allowed, in cells per side (`max_leaf_cells`, a power of
  two).  On a sharded level it is at most the largest power of two
  dividing every shard side; see Leaves and shards.

## Layout

| Where | What |
|---|---|
| root `format_capabilities` | `octree_chunks` while any level has octree chunks |
| level `zarr_vectors_level.octree_chunks` | `max_depth`, `origin_cell`, `target`, `balanced`, `weight` (`"vertices"`), `leaves`, `max_leaf_cells` (`null`: uncapped), `shard_shape` and `shard_origin` (the shard grid the leaves were fitted to, in cells; `null` when unsharded) |
| `<level>/octree_chunks/leaves` | `(N, 5)` int64: `depth, i, j, k, weight` per stored (non-empty) leaf |
| `<level>/octree_chunks/cells` | `(C, 6)` int64: `i, j, k` (cell), `leaf`, `vertices`, `fragments`, in storage order |
| `vertices`, `vertex_fragments`, `object_index` | as for uniform chunks, keyed by the leaf's anchor; the object index keeps the level's object ids, layout and row sharding |
| `links/0/<segment>` | keyed by the source leaf's anchor; see below |
| every per-chunk array's cell codecs | end in `{"name": "zarr_vectors_octree_chunks"}` (see Compatibility) |

A leaf's vertex block is its cells' blocks end to end (cells in Z order), and
its fragments are theirs in the same order; `octree_chunks/cells` records the
runs.  Manifest entries name `(anchor, fragment)`.

## Leaves and shards

On a sharded level a leaf is never larger than a shard and never crosses
a shard boundary, so all of a leaf's data -- vertices, fragments, the links
filed under it -- is in one shard.  A shard can then be written by one
task without locks (as BRIDGE writes, one task per shard), rewritten on
its own after a failure, and read without touching its neighbours; a
leaf larger than a shard would put the data of several shards into its
anchor's shard.

`to_octree` fits the plan to the level's shard grid -- the shard shape of
its `vertices` array (or the store's declared `shard_shape` when the
level has none) and that array's grid origin (`chunk_grid_origin`, where
the first shard starts):

* the root cube starts on a shard boundary;
* no leaf is wider than the largest power of two dividing every shard
  side: 4 for shards of `(4, 4, 4)`, 2 for `(4, 2, 8)`, and 2 for
  `(6, 6, 6)`, since a leaf of 4 cells would cross the boundary at 6.
  Leaves are power-of-two cubes from a shard boundary, so they tile the
  shards exactly;
* a leaf that holds anything is split while wider than the cap; empty
  space is not stored and is left coarse.  Balancing only ever splits
  leaves, so it never makes one wider than the cap.

`max_leaf_cells=` caps the leaf side further (a power of two, at most the
shard's cap), or caps it on an unsharded level, which otherwise is not
capped.  The cap and the shard grid are recorded in the level's octree
block.  Both conversions keep the level's packing: the rewritten arrays
are sharded as the level's `vertices` were, also when that was done by
`shard_store` without a root declaration.

This is the rule Zarr's own sharding has for regular chunks -- a shard
shape must be a whole number of chunks, so every chunk is in exactly one
shard -- applied to chunks of several sizes.

The cap costs chunk count where space is sparse: leaves that would have
spanned several shards become one per shard (or finer, for small or
non-power-of-two shards).  On a clustered synthetic level of 10^6
vertices in 80^3 cells (140,299 occupied) with `target=5000`, the
uncapped plan has 1,665 leaves (side 1 to 16); capped, shards of 16 give
the same 1,665, shards of 8 give 2,141, shards of 4 give 8,546, and
shards of 2 (or `(6, 6, 6)`, `(4, 2, 8)`, capped at 2) give 58,151 leaves
averaging 17 vertices.  The largest leaf (4,936 vertices) is the same in
every case: the cap only splits leaves.

**Repacking an octree level.**  `shard_store`, `reshard` and
`unshard_store` move cells unchanged, so every leaf stays a correct
chunk at its anchor and they are allowed on an octree level.  A new
shard shape the recorded cap still divides (`(4, 4, 4)` to `(8, 8, 8)`)
keeps every leaf inside one shard; one that splits leaves (`(4, 4, 4)`
to `(2, 2, 2)`) does not, and `validate` reports each leaf not inside
one shard of the arrays' current shard grid -- as an error when the
level was converted with a cap, as a warning when it was converted
without one (unsharded at the time, or by a `to_octree` before the cap).
`to_uniform` then `to_octree` fits the leaves to the new shards.

## Links

A record is filed under its **source** leaf: the deeper endpoint, or at equal
depth the one with the smaller anchor.  The other endpoint, `Δ` depths
coarser, is named by the offset `o` of its leaf from the source's ancestor at
that depth, which is always one grid step:

    other = (d - Δ, (i >> Δ) + o_x, (j >> Δ) + o_y, (k >> Δ) + o_z),  o ∈ {-1, 0, 1}³

The segment is `o` as for uniform links, with `^Δ` appended when `Δ > 0`:

| Segment | Meaning |
|---|---|
| `0.0.0` | both endpoints in the leaf |
| `+1.0.0` | the same-depth leaf along +x |
| `+1.0.0^1` | the leaf along +x of the source's parent (one depth coarser) |

Rows are `(source row, other row)`, rows local to each leaf.  A balanced
level has at most 26 + 26 segments besides `0.0.0`.  Links must not be
longer than a cell, so that they only join adjacent leaves.

## Compatibility

A reader without `octree_chunks` must not read such a level: it would take
each leaf for a single cell.  Capabilities cannot stop it -- readers ignore
the ones they do not know, and zarr-vectors before octree chunks (e.g.
commit 467e80e) never refuses a store for a newer `zv_version` or an
unknown capability -- so the level stops it through zarr:

**The marker codec.**  Every per-chunk array of an octree level
(`vertices`, `vertex_fragments`, `links/0/<segment>`) ends its cell
codecs with `{"name": "zarr_vectors_octree_chunks"}` -- inside the
sharding codec's inner codecs when the level is sharded (either
`sharding_indexed` or `zarr_octree_sharding`).  It is the identity on
bytes, so the stored cells are exactly what they would be without it;
but a zarr reader must know every codec of an array to open it, so any
reader that does not register it fails on the level's first array
(zarr-python: `UnknownCodecError: Unknown codec:
'zarr_vectors_octree_chunks'`), for every read path -- whole level, box,
object, chunk -- of points and graphs alike.  Importing `zarr_vectors`
registers it (`zarr_vectors.encoding.octree_marker`).  `to_octree` adds
it, `to_uniform` writes the arrays without it -- a converted-back level
has the same `zarr.json` files as before -- and the shard repackers
(`shard_store`, `unshard_store`, `set_octree`) keep it.

Uniform levels are unchanged.  Every writer writes uniform chunks, and
nothing but the two conversions (and the shard repackers, which move cells
unchanged) may write an octree level.  Everything else refuses it before
it changes anything, with an error that says to `to_uniform` first:

* the type writers (`write_points`, `write_graph`, …, `Dataset.add_*`),
  when the level they write (level 0) has octree chunks;
* edits (`Dataset.editing`, `EditSession`, the `zarr_vectors.ops`
  functions), at the first edit staged for an octree level;
* `coarsen_level` (source or target level) and `build_pyramid` (any
  level): pyramid levels and links between levels are not defined for
  octree chunks;
* `rechunk`, `rechunk_spatial` and `rebin_level` (the levels they read):
  they read uniform cells.  `rechunk` and `rechunk_spatial` carry the
  store's packing to their output -- `shard_shape` and the
  `octree_sharding` marker -- so an octree-sharded store rechunks to an
  octree-sharded store;
* the uniform-cell level APIs `Level.cells()`, `Query.cells()` and
  `read_neighbourhood`, and the per-cell readers `read_chunk_vertices`
  and `read_chunk_links` that tools reading cell by cell (a viewer's
  chunk server, say) use;
* the shard repackers (`shard_store`, `unshard_store`, `set_octree`) on a
  store with a level part way through a conversion: they would repack
  its live arrays but not the staged ones swapped in later.  `select(limit=)` reads an octree level whole and
  keeps the first rows (its cell prefix cannot name leaves).

Underneath, `Group` refuses any cell, array or metadata write and any
delete on an octree level's handle unless the handle is marked by
`zarr_vectors.octree_chunks.guard.converting`.  `validate` checks an
octree level with checks of its own (leaf and cell tables, every leaf's
vertex and fragment counts, manifests, link rows, no links between
levels, the marker codec, and every leaf inside one shard) instead of
the uniform ones.
`zarr_vectors.octree_chunks.to_octree` / `to_uniform` convert a level,
and the way back restores every cell's vertex rows, fragments, manifest
entries and links.

## Converting safely

Nothing in the level changes until the whole conversion has been planned
and checked (supported content, every link joining adjacent leaves, every
manifest entry and link naming rows the level holds).  The new layout is
then written into a staging group, `<level>/octree_chunks_staging`,
carrying a copy of the level's metadata, and swapped in:

1. the staging group is sealed with a record (`octree_chunks_staging`
   attribute: direction, the arrays it holds, the level metadata to end
   with);
2. the level's octree block becomes `{"converting": "to_octree" |
   "to_uniform"}` and the root gains `octree_chunks` -- the commit point;
   from here a reader refuses the level instead of mixing layouts;
3. each of `vertices`, `vertex_fragments`, `links`, `link_fragments`,
   `object_index`, `octree_chunks` is replaced by its staged copy, or
   removed if the new layout has none (a directory rename on a local
   store; elsewhere a copy, then the array recorded as `moved` in the
   staging record, then the staged copy deleted -- so a delete cut short
   is never copied over the live array again);
4. the level's metadata is set from the record, the staging group
   removed, and the root's `octree_chunks` kept only while any level
   still has an octree block.

Running either conversion again after a failure or kill cleans up: a
staging group whose swap never began (no `converting` mark) is
discarded -- the level is as it was, and may have been written since --
and a swap that began is finished from the sealed record (a `to_octree`
or `to_uniform` call that finishes its own direction returns that
conversion's summary with `"resumed": true`).  So at any point the level
is either the old one, intact, or one the next call completes.

Writers address uniform cells, so they must not write an octree level.
`zarr_vectors.octree_chunks.guard` has `is_octree_level(level_group)`
(true for an octree level and for one mid-conversion), the
`converting(level_group)` / `is_converting(level_group)` marks that the
conversions put on every group handle they write through, and
`refuse_octree_write` / `refuse_octree_read` / `refuse_octree_levels`,
the refusals listed under Compatibility.
