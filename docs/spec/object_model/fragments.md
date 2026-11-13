# Fragments

## Terms

**Fragment**
: A set of rows of one chunk's `vertices/` cell, listed in the chunk's
  `vertex_fragments/` cell. A fragment is either a contiguous range of
  rows or an explicit list of rows; the byte layout is on
  [Fragment-index arrays](../layout/fragment_index_arrays.md).

**Fragment address**
: The pair `(chunk_coords, fragment_index)` that identifies a fragment
  within a level. `fragment_index` is local to the chunk, in `[0, F)`
  for a chunk with `F` fragments.

**Shared fragment**
: A fragment named by more than one object's manifest.

---

## Introduction

Fragments let a reader take part of a chunk. A chunk's fragment index
divides its `vertices/` rows into fragments. Objects name their vertices
by fragment address in their [manifests](object_manifest.md).

---

## Technical reference

### Fragments by writer

| Writer | One fragment per |
|--------|------------------|
| `write_points` without object ids | Bin of the chunk, empty bins included (a range with `count == 0`). Fragment `f` is the bin at local bin coordinates `unravel(f, bins_per_chunk)` in C order |
| `write_points` with object ids, `write_polylines`, `write_graph` and the other object writers | Contiguous run of one object's vertices within one chunk; an object that leaves a chunk and returns has a fragment for each visit |
| `write_skeleton_chunk` | Unbranched path of a skeleton piece within the chunk |
| The core pyramid builder, at coarser levels | Metavertex, one vertex each |

A range fragment holds rows `[start, start + count)`; an explicit
fragment lists its rows. Every row index is within the `vertices` cell.

With `fragments_tile: true` in the level's attributes, each chunk's
fragments are ranges that cover its rows in order, with no gaps or
overlaps. A reader may then take the whole cell without the fragment
index.

Row `k` of a chunk's `vertex_attributes/<name>` cell belongs to vertex
row `k`, so a fragment's attributes use the same row indices. Row `f` of
`fragment_attributes/<name>` belongs to fragment `f` (see
[Chunk arrays](../layout/chunk_arrays.md)).

### Reading fragments

```python
from zarr_vectors.building import read_fragment, read_vertex_fragment_index
from zarr_vectors.core.store import get_resolution_level, open_store

lg = get_resolution_level(open_store("scan.zarrvectors"), 0)
fidx = read_vertex_fragment_index(lg, (0, 0, 0))   # ChunkFragmentIndex
print(len(fidx))                                    # F
rows = fidx.indices(4)                              # row indices of fragment 4
vertices = read_fragment(lg, (0, 0, 0), 4)          # (n, D) vertices of fragment 4
```

`ChunkFragmentIndex` (in `zarr_vectors.encoding.fragments`) also has
`is_range(f)`, `range(f)` for a range fragment, and `indices_view(f)`, a
view of an explicit fragment's stored indices. `read_fragment` does a
byte-range read of a range fragment where the storage allows it, and
otherwise reads the whole cell. To read an object, see
[Reading an object](object_manifest.md#reading-an-object).

### Shared fragments

At a level with `shared_fragments: true`, one fragment may appear in
several objects' manifests; its vertices are stored once. Readers must
not assume different objects have disjoint fragment addresses there.
The root's `format_capabilities` lists `shared_fragments` when any level
has it. The core pyramid builder writes every coarser level this way
(see [Pyramid construction](../multiscale/pyramid_construction.md)).

### Validation

**L1** warns when a level has no `vertex_fragments/`.

**L3**, for each chunk of each level:

| Check | Severity |
|-------|----------|
| Every fragment decodes to a `(n, sid_ndim)` array | error |
| A fragment holds `NaN` or infinite values | warning |
| With `fragments_tile: true`, the fragments tile the rows | error |
| With `link_fragments_aligned: true`, intra-chunk links are filed by vertex fragment | error |
| Point cloud with bins and no object index: no more fragments than bins per chunk | error |
| Same: a vertex lies outside its fragment's bin (checked in up to three chunks) | warning |

The byte-level checks of the fragment index are on
[Fragment-index arrays](../layout/fragment_index_arrays.md#validation).
