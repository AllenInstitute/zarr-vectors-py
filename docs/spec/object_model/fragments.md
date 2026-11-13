# Fragments

## Terms

**Fragment**
: A set of rows of one chunk's `vertices/` cell, listed in the chunk's
  `vertex_fragments/` cell. A fragment is either a contiguous range of
  rows or an explicit list of rows; the byte layout is on
  [Fragment-index arrays](../layout/fragment_index_arrays.md).

**Fragment address**
: The pair `(chunk_coords, fragment_index)` that identifies a fragment
  within one level. `chunk_coords` is the chunk's coordinate tuple and
  `fragment_index` is local to that chunk, in `[0, F)` where `F` is the
  chunk's fragment count.

**Shared fragment**
: A fragment named by more than one object's manifest.

---

## Introduction

Fragments are how a reader takes part of a chunk. A chunk's vertices are
stored as one cell of `vertices/`; its fragment index divides those rows
into fragments, and objects name their vertices by fragment address in
their [manifests](object_manifest.md). A reader that wants one object,
or one bin of a point cloud, reads the fragment index of each chunk
involved and takes only the rows it needs.

What a fragment stands for depends on the writer: a spatial bin in a
point cloud, a piece of one object in other geometry types, and a
metavertex at a coarser pyramid level.

---

## Technical reference

### Fragments by writer

| Writer | One fragment per |
|--------|------------------|
| `write_points` without object ids | Bin of the chunk, empty bins included (a range with `count == 0`). Fragment `f` is the bin at local bin coordinates `unravel(f, bins_per_chunk)` in C order |
| `write_points` with object ids, `write_polylines`, `write_graph` and the other object writers | Contiguous run of one object's vertices within one chunk; an object that leaves a chunk and returns has a fragment for each visit |
| `write_skeleton_chunk` | Unbranched path of a skeleton piece within the chunk |
| The core pyramid builder, at coarser levels | Metavertex, one vertex each |

The rows of a range fragment are `[start, start + count)`; an explicit
fragment lists its rows. Every row index lies in
`[0, rows in the vertices cell)`.

A level whose attributes say `fragments_tile: true` promises that the
chunk's fragments are ranges that cover its rows in order with no gaps
or overlaps, so a reader may take the whole cell without reading the
fragment index. L3 validation checks the promise.

### Per-chunk data aligned to vertices and fragments

Row `k` of a chunk's `vertex_attributes/<name>` cell belongs to vertex
row `k` of the same `vertices/` cell, so a fragment's attribute rows are
selected with the same row indices as its vertices. Row `f` of a chunk's
`fragment_attributes/<name>` cell belongs to fragment `f`. See
[Chunk arrays](../layout/chunk_arrays.md).

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

`ChunkFragmentIndex` (in `zarr_vectors.encoding.fragments`) answers
`len(fidx)`, `fidx.is_range(f)`, `fidx.range(f)` for a range fragment,
`fidx.indices(f)` (a new array) and `fidx.indices_view(f)` (a view of the
stored indices of an explicit fragment). `read_fragment` reads only a
range fragment's rows where the storage allows a byte-range read, and
otherwise reads the cell and selects the rows.

To read a whole object, follow its manifest; see
[Reading an object](object_manifest.md#reading-an-object).

### Shared fragments

At a level whose attributes say `shared_fragments: true`, one fragment
may appear in several objects' manifests, and its vertices are stored
once. The root's `format_capabilities` lists `shared_fragments` when any
level does. A reader treats a shared fragment like any other, but must
not assume that the fragment addresses of different objects at such a
level are disjoint. The core pyramid builder writes every coarser level
this way (see [Pyramid construction](../multiscale/pyramid_construction.md)).

### Validation

**L1** warns when a level has no `vertex_fragments/`.

**L3**, for each chunk of each level:

- every fragment of the chunk decodes to a `(n, sid_ndim)` array, and a
  fragment containing `NaN` or infinite values gives a warning;
- with `fragments_tile: true`, the chunk's fragments tile its rows;
- with `link_fragments_aligned: true`, the chunk's intra-chunk links are
  filed by vertex fragment;
- in a point cloud with bins and no object index, a chunk has no more
  fragments than bins per chunk, and in up to three chunks a warning is
  given for a vertex outside its fragment's bin.

The byte-level checks of the fragment index are on
[Fragment-index arrays](../layout/fragment_index_arrays.md#validation).
