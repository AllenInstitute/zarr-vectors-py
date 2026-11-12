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

## Layout

| Where | What |
|---|---|
| root `format_capabilities` | `octree_chunks` while any level has octree chunks |
| level `zarr_vectors_level.octree_chunks` | `max_depth`, `origin_cell`, `target`, `balanced`, `weight` (`"vertices"`), `leaves` |
| `<level>/octree_chunks/leaves` | `(N, 5)` int64: `depth, i, j, k, weight` per stored (non-empty) leaf |
| `<level>/octree_chunks/cells` | `(C, 6)` int64: `i, j, k` (cell), `leaf`, `vertices`, `fragments`, in storage order |
| `vertices`, `vertex_fragments`, `object_index` | as for uniform chunks, keyed by the leaf's anchor |
| `links/0/<segment>` | keyed by the source leaf's anchor; see below |

A leaf's vertex block is its cells' blocks end to end (cells in Z order), and
its fragments are theirs in the same order; `octree_chunks/cells` records the
runs.  Manifest entries name `(anchor, fragment)`.

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
each leaf for a single cell.  Uniform levels are unchanged, and every writer
writes uniform chunks; `zarr_vectors.octree_chunks.to_octree` /
`to_uniform` convert a level, and the way back restores every cell's vertex
rows, fragments, manifest entries and links.

Supported so far: point clouds and undirected graphs at one level, without
vertex, fragment or link attributes and without links to other levels.
