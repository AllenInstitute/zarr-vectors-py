# Chunk arrays

## Terms

**Chunk array**
: A Zarr array within a resolution level group whose logical shape is
  expressed in terms of the chunk grid. The first `D` dimensions of a chunk
  array's shape correspond to the chunk grid dimensions; later dimensions
  carry per-chunk payload data (vertex coordinates, edge indices, etc.).

**Ragged array**
: An array where the size of the payload dimension varies per chunk. For
  example, `vertices/` has shape `(Cx, Cy, Cz, N_max, D)` in the Zarr
  metadata, but most chunks contain fewer than `N_max` vertices. Unused
  positions are filled with the array's fill value.

**`N_max`**
: The maximum number of vertices (or edges, faces) per chunk as declared in
  the Zarr array's `shape`. This is a soft upper bound; `zarr-vectors-py`
  resizes the array as needed when writing.

**Chunk grid shape**
: The number of chunks along each spatial axis. For a store with spatial
  extent `[E_0, E_1, …, E_{D-1}]` and `chunk_shape = [C_0, …, C_{D-1}]`,
  the chunk grid shape is `[ceil(E_i / C_i) for i in range(D)]`.

---

## Introduction

Every vertex, edge, face, and attribute value in a ZVF store is stored in a
Zarr array whose first dimensions index the chunk grid. Within a given chunk,
data is stored as a dense payload slice. Because different chunks contain
different numbers of vertices, the payload dimension is ragged: the Zarr
array declares a fixed maximum, but only the used portion is written (the
rest is fill-value padded or absent if the chunk is empty).

This page documents the dtype, shape, chunk grid, and fill value for every
array defined by the ZVF spec, for each geometry type.

---

## Technical reference

### `vertices/`

Stores the spatial positions of all vertices in the store, one Zarr chunk
per spatial chunk.

| Property | Value |
|----------|-------|
| Dtype | `float32` |
| Logical shape | `(*chunk_grid_shape, N_max, D)` |
| Zarr chunk shape | `(1, 1, …, 1, N_chunk_max, D)` — one Zarr chunk per spatial chunk |
| Fill value | `0.0` |
| Codec | `bytes → blosc(zstd, bitshuffle)` (default) |

`N_max` is set conservatively at write time based on the expected vertex
density and chunk volume; the array is resized if a chunk exceeds the
declared maximum.

Within each spatial chunk the vertices are stored in **fragment order**: all
vertices of bin (0,0,0) first, then bin (0,0,1), etc., in C-order bin
index. The `vertex_fragments/` array encodes one fragment per non-empty
bin describing its row range within this ordering (see
[Fragment-index arrays](fragment_index_arrays.md)).

**Example:** a 3-D store with a chunk grid of shape `(5, 6, 4)` and up to
65 536 vertices per chunk:

```json
{
  "shape": [5, 6, 4, 65536, 3],
  "data_type": "float32",
  "chunk_grid": {
    "name": "regular",
    "configuration": {"chunk_shape": [1, 1, 1, 65536, 3]}
  },
  "fill_value": 0.0
}
```

### `vertex_fragments/`

Stores the row partition of each `vertices/` chunk slice as a binary
fragment-index blob. See [Fragment-index arrays](fragment_index_arrays.md) for
the full byte layout, the decoder algorithm, and worked examples.

| Property | Value |
|----------|-------|
| Dtype | `uint8` |
| Layout | One opaque blob per chunk; addressed by chunk coordinate |
| Codec | none (bytes written directly; see [Codec pipeline](../foundations/codec_pipeline.md)) |
| `zv_array` metadata | `"vertex_fragments"`, `encoding == "fragment_index_v1"` |

Each blob is a v1 fragment-index header (magic `'ZVFG'`) followed by a
range bitmap, range table, and explicit CSR. At level 0 with the default
writer, each non-empty bin emits one range fragment; at coarsened levels
with `shared_fragments=True`, fragments may also represent metavertices
shared between objects' manifests.

### `link_fragments/`

Parallel structure for `links/0/<chunk>` rows. Same byte layout as
`vertex_fragments/`; present only where the geometry type has connectivity
and only at `<delta>=0`. Cross-level link arrays (`<delta> != 0`) keep
their inline self-describing header.

### `links/<delta>/`

Present for: polyline, streamline, graph, skeleton.

Stores pairs of vertex indices representing graph edges or polyline segment
connections, within a single chunk.

| Property | Value |
|----------|-------|
| Dtype | `int32` |
| Logical shape | `(*chunk_grid_shape, E_max, 2)` |
| Zarr chunk shape | `(1, 1, …, 1, E_max, 2)` |
| Fill value | `-1` |
| Codec | `bytes → blosc(zstd, byteshuffle)` |

Vertex indices in `edges/` are **local to the chunk**: index `k` refers to
the `k`-th vertex in the `vertices/` chunk slice. Inter-chunk connections are
stored separately in `cross_chunk_links/`.

For polylines and streamlines, edges are stored in traversal order: edge `i`
connects vertex `i` to vertex `i+1` along the polyline. For graphs and
skeletons, edge order is not semantically significant.

### `links/<delta>/`

Present for: mesh only.

Stores triangular face definitions as triplets of vertex indices, local to
the chunk.

| Property | Value |
|----------|-------|
| Dtype | `int32` |
| Logical shape | `(*chunk_grid_shape, F_max, 3)` |
| Zarr chunk shape | `(1, 1, …, 1, F_max, 3)` |
| Fill value | `-1` |
| Codec | `bytes → blosc(zstd, byteshuffle)` or `draco` |

Vertex winding order is consistent within a store (default: counter-clockwise
when viewed from outside the surface, i.e. outward-facing normals). The
winding order convention is stored in root `.zattrs` under `"winding_order"`:
`"ccw"` (default) or `"cw"`.

### `attributes/<name>/`

One sub-group per named per-vertex attribute. The attribute name is a
valid Python identifier (alphanumeric and underscores only).

| Property | Value |
|----------|-------|
| Dtype | Any numeric dtype declared in `zarr.json` |
| Logical shape | `(*chunk_grid_shape, N_max)` for scalar attributes; `(*chunk_grid_shape, N_max, K)` for vector attributes of width K |
| Zarr chunk shape | `(1, 1, …, 1, N_max)` or `(1, 1, …, 1, N_max, K)` |
| Fill value | `0` or `NaN` (declared per array) |
| Codec | Varies; default is `bytes → blosc(zstd, bitshuffle)` |

The vertex ordering within an attribute chunk must match the vertex ordering
in the corresponding `vertices/` chunk exactly. That is, attribute value `k`
in chunk `(i,j,l)` of `attributes/intensity/` corresponds to vertex `k` in
chunk `(i,j,l)` of `vertices/`.

### `fragment_attributes/<name>/`

One sub-group per named **per-fragment** attribute. Opt-in; absent unless
the writer was given explicit per-chunk values. Each chunk stores a dense
byte blob whose row count equals the number of fragments encoded in
`vertex_fragments/<chunk>`; the row count is derived at read time from the
byte length and the row stride (`dtype.itemsize * K`), so no sibling
offsets blob is written.

| Property | Value |
|----------|-------|
| Dtype | Any numeric dtype declared in `zarr.json` |
| Logical shape | `(F,)` for scalar attributes; `(F, K)` for vector attributes of width K, where `F = num_fragments_in_chunk` |
| Zarr chunk shape | One file per spatial chunk key (same key scheme as `vertices/`) |
| Fill value | `0` or `NaN` (declared per array) |
| Codec | `bytes → blosc(zstd, byteshuffle)` (same default as `attributes/`) |

Element `k` corresponds to fragment `k` as encoded in
`vertex_fragments/<chunk>`. Replace-only at the chunk level; not
auto-downsampled in the pyramid. The canonical opt-in use case is
materializing parent-IDs (e.g. the `object_id` owning each fragment) so
joins on fragment → object don't have to round-trip
`object_index/manifests`.

### `object_index/`

Present for: polyline, streamline, graph, skeleton, mesh.

Stores one **manifest blob** per object enumerating every chunk the object
touches and the fragments within each chunk. Two byte-keyed entries:

| Key | Contents |
|-----|----------|
| `data` | concatenated manifest blobs for all `num_objects` objects |
| `offsets` | `int64` array of length `num_objects`; entry `i` is the byte offset of object `i`'s blob within `data` |

Group-level `zv_array` metadata: `"object_index"`, plus `num_objects` and
`sid_ndim`. The arrays are written as opaque bytes (no Zarr codec pipeline).

See [Object manifest](../object_model/object_manifest.md) for the
manifest-blob byte layout and the read path.

### `object_attributes/<name>/`

One sub-group per named per-object attribute. Shape `(n_objects,)` for
scalar attributes.

| Property | Value |
|----------|-------|
| Dtype | Any numeric dtype |
| Logical shape | `(n_objects,)` or `(n_objects, K)` |
| Zarr chunk shape | `(65536,)` or `(65536, K)` |
| Fill value | `0` or `NaN` |

### `groupings/`

Maps group IDs to lists of object IDs.

| Property | Value |
|----------|-------|
| Dtype | `int64` |
| Logical shape | `(n_groups, max_group_size)` |
| Zarr chunk shape | `(1, max_group_size)` |
| Fill value | `-1` (padding for groups smaller than `max_group_size`) |

### `cross_chunk_links/`

Present for: polyline, streamline.

Stores pairs of global vertex IDs representing connections that cross a
chunk boundary.

| Property | Value |
|----------|-------|
| Dtype | `int64` |
| Logical shape | `(n_links, 2)` |
| Zarr chunk shape | `(65536, 2)` |
| Fill value | `-1` |

Each row is a `(src_global_vertex_id, dst_global_vertex_id)` pair where
`src` is the last vertex of a polyline segment in chunk A and `dst` is the
first vertex of the continuation segment in chunk B.

See [Cross-chunk links](../object_model/cross_chunk_links.md) for the
encoding of global vertex IDs and the reconstruction algorithm.
