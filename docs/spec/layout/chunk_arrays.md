# Chunk arrays

## Terms

**Chunk array**
: A single Zarr v3 **vlen-bytes** array within a resolution level group
  whose shape *is* the chunk grid: one array cell per spatial chunk. Each
  cell holds that chunk's payload as one opaque, variable-length byte blob
  (Zarr Vectors-encoded vertices, edges, fragment index, attribute values, …).
  Chunk files land at `<array>/c/i/j/k`.

**Chunk grid origin**
: `origin = floor(min_corner / chunk_shape)` per axis, stored in the array's
  `chunk_grid_origin` attribute (absent ⇒ all-zero). A spatial chunk at
  absolute coord `c` (= `floor(position / chunk_shape)`) is stored in cell
  `c - origin`, so data whose positions dip below the origin (negative
  coords) still maps onto the 0-indexed array.

**`nonempty_chunks`**
: An array attribute listing the dotted chunk keys (`"i.j.k"`) that hold a
  non-empty payload, so chunk enumeration is O(1) without scanning cells.

**Chunk grid shape**
: The number of chunks along each spatial axis: for spatial extent up to
  `max_corner` and `chunk_shape = [C_0, …]`, the grid spans
  `floor(max_corner_i / C_i) - origin_i + 1` cells along axis `i`.

---

## Introduction

Every per-spatial-chunk quantity in a Zarr Vectors store — vertices, edges, faces,
the fragment index, and per-vertex / per-fragment / per-edge attributes — is
stored as a **single** Zarr v3 vlen-bytes array whose shape is the level's
chunk grid. One array cell holds one spatial chunk's payload as an opaque
byte blob; empty chunks simply have no chunk file (the vlen fill value is
`b""`). This replaces the earlier "Option G" layout, where each spatial
chunk was its own single-chunk `uint8` sub-array under a per-array group.

Since 0.9.0 this pattern has **no exceptions**. `cross_chunk_links/<delta>/`
used to be one — its cells were keyed by canonical-sorted endpoint-chunk
tuples rather than a spatial grid, so each cell was its own small array
under a group. That family is gone: connectivity is a single family whose
arrays are ordinary chunk arrays, one cell per **source** chunk, with the
relationship between endpoints factored into the path instead
(see [Links](../object_model/links.md)).

This page documents the dtype, shape, chunk grid, and codec for every array
defined by the Zarr Vectors specification, for each geometry type.

---

## Technical reference

### `vertices/`

Stores the spatial positions of all vertices, one array cell per spatial
chunk.

| Property | Value |
|----------|-------|
| Dtype | `variable_length_bytes` (vlen-bytes serializer) |
| Shape | `chunk_grid_shape` (one cell per spatial chunk) |
| Zarr chunk shape | `(1, …, 1)` — one inner chunk per cell (file at `c/i/j/k`) |
| Fill value | `b""` (empty chunk ⇒ no chunk file) |
| Codec | `vlen-bytes` (+ optional `zstd` / `blosc` compressor; none by default) |
| Sharding | optional `shard_shape=` wraps cells in `sharding_indexed` |
| Attributes | `chunk_grid_origin`, `nonempty_chunks`, `zv_array="vertices"`, `dtype`, `encoding` |

Each cell's blob is the Zarr Vectors-encoded positions for that chunk. Within each
spatial chunk the vertices are stored in **fragment order**: all vertices
of bin (0,0,0) first, then bin (0,0,1), etc., in C-order bin index. The
`vertex_fragments/` array encodes one fragment per non-empty bin describing
its row range within this ordering (see
[Fragment-index arrays](fragment_index_arrays.md)).

**Example:** a 3-D store with a chunk grid of shape `(5, 6, 4)`, one cell
per spatial chunk (chunk file at `vertices/c/i/j/k`):

```json
{
  "shape": [5, 6, 4],
  "data_type": "variable_length_bytes",
  "chunk_grid": {
    "name": "regular",
    "configuration": {"chunk_shape": [1, 1, 1]}
  },
  "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
  "codecs": [{"name": "vlen-bytes", "configuration": {}}],
  "fill_value": "",
  "attributes": {
    "zv_array": "vertices",
    "dtype": "float32",
    "encoding": "raw",
    "chunk_grid_origin": [0, 0, 0],
    "nonempty_chunks": ["0.0.0", "0.0.1"]
  }
}
```

Note that `data_type` above is `variable_length_bytes` — that describes the
*container*, not the payload. The element type of the vertex positions is
the `dtype` **attribute** (`"float32"` here). See
[Decoding a cell blob](#decoding-a-cell-blob).

### Decoding a cell blob

A `vertices/` cell is a flat, C-order buffer of `N × D` elements with no
inline header, so a reader needs three pieces of metadata to recover the
`(N, D)` position array. Only one of them lives on the array itself:

| Input | Source |
|-------|--------|
| Element dtype | `vertices/` array attribute `dtype` (e.g. `"float32"`) |
| `D` (columns) | root metadata `zarr_vectors.spatial_dims` — **not** stored on the array; see [Dimensionality](../foundations/dimensionality.md) |
| `N` (rows) | derived: `len(blob) // (dtype.itemsize * D)` |
| Blob framing | `vertices/` array attribute `encoding`: `"raw"` (flat buffer, as above) or `"draco"` (compressed mesh payload; `dtype`/`D` describe the *decoded* result) |

For `encoding == "raw"` the decode is therefore:

```python
positions = np.frombuffer(blob, dtype=attrs["dtype"]).reshape(-1, spatial_dims)
```

Readers **must** honour the stored `dtype` rather than assuming `float32`:
a `float64` store decoded as `float32` yields garbage coordinates at twice
the row count, silently and without error.

Writers must keep the blob consistent with the declared `dtype`: the bytes
written are the positions cast to `dtype`, so a well-formed cell's length is
always an exact multiple of `dtype.itemsize * D`. Nothing in the format
records `N` independently, so this invariant is what makes the row count
recoverable — a blob that violates it decodes to the wrong shape rather
than raising.

The `vertex_fragments/`, `links/<delta>/<offsets>/`, and attribute arrays use
the same single-vlen-array shape; only the encoded blob and its `dtype` /
`encoding` attributes differ.

A link array's **physical row width** is not simply `link_width`. It is
`link_width + 1` when the array declares `has_perm` — the extra leading
column is `perm_idx`. So a link cell decodes as:

```python
W = attrs["link_width"] + (1 if attrs["has_perm"] else 0)
rows = np.frombuffer(blob, dtype=attrs["dtype"]).reshape(-1, W)
```

Readers **must** honour the stored `has_perm` rather than assuming `L`
columns: decoding a `1 + L` array at width `L` mis-parses every row in the
cell — silently, since the blob length is still a valid multiple. See
[`links/<delta>/<offsets>/`](#linksdeltaoffsets) below.

### `vertex_fragments/`

Stores the row partition of each `vertices/` chunk slice as a binary
fragment-index blob. See [Fragment-index arrays](fragment_index_arrays.md) for
the full byte layout, the decoder algorithm, and worked examples.

| Property | Value |
|----------|-------|
| Dtype | `variable_length_bytes` (vlen-bytes serializer) |
| Shape | `chunk_grid_shape` — same grid as `vertices/`, one cell per spatial chunk |
| Fill value | `b""` (empty chunk ⇒ no chunk file) |
| Codec | `vlen-bytes` (blob written as-is; see [Codec pipeline](../foundations/codec_pipeline.md)) |
| Attributes | `chunk_grid_origin`, `nonempty_chunks`, `zv_array="vertex_fragments"`, `encoding="fragment_index_v1"` |

Unlike `vertices/`, this array declares no `dtype` attribute: the blob is
not a typed buffer but a self-describing structure whose element widths are
fixed by the `fragment_index_v1` encoding and its `'ZVFG'` header.

Each blob is a v1 fragment-index header (magic `'ZVFG'`) followed by a
range bitmap, range table, and explicit CSR. At level 0 with the default
writer, each non-empty bin emits one range fragment; at coarsened levels
with `shared_fragments=True`, fragments may also represent metavertices
shared between objects' manifests.

### `link_fragments/`

Parallel structure for the rows of `links/0/<all-zero offsets>/` — the
intra-chunk array at delta 0, and **only** that array. Same byte layout as
`vertex_fragments/`; present only where the geometry type has connectivity.

Every other link array — any non-zero offset, and every `<delta> != 0` —
uses an inline self-describing blob and has no sidecar. The sidecar is
keyed by chunk **alone**, carrying neither a delta nor an offsets segment,
which is exactly why only one array per chunk may write it; see
[Fragment-index arrays](fragment_index_arrays.md).

### `links/<delta>/`

A **group**, not an array. Present for: polyline, streamline, graph,
skeleton (`link_width=2`); mesh (`link_width=3`); skeleton parent refs
(`link_width=1`).

Its children are one chunk array per distinct relative-offset segment. The
group's own metadata carries the family-wide policy that every child
decodes against:

| Attribute | Value |
|-----------|-------|
| `zv_array` | `"links_family"` |
| `level_delta` | signed int; `0` for intra-level |
| `link_width` | `L` — 2 for edges, 3 for triangles, 1 for parent refs |
| `sid_ndim` | spatial index dims — the arity of each offset |
| `directed` | `false` (default) / `true` — endpoint order is data |
| `store` | `"canonical"` (default) / `"duplicate"` |
| `num_links` | family-wide **logical** record count (after `finalize_links`) |
| `num_physical_records` | on-disk rows; `> num_links` under `"duplicate"` |

### `links/<delta>/<offsets>/`

The chunk array itself. One cell per **source** chunk — the chunk of the
endpoint the record is filed under. The `<offsets>` segment names where the
record's other `L - 1` endpoints sit relative to that source, so
`links/0/0.0.0/` holds intra-chunk records and `links/0/0.0.+1/` holds
records whose second endpoint is one chunk along `+z`.

| Property | Value |
|----------|-------|
| Dtype | `variable_length_bytes` (vlen-bytes serializer) |
| Shape | `chunk_grid_shape` (one cell per spatial chunk) |
| Fill value | `b""` (empty chunk ⇒ no chunk file) |
| Codec | `vlen-bytes` (+ optional `zstd` / `blosc` compressor; none by default) |
| Attributes | `chunk_grid_origin`, `nonempty_chunks`, `zv_array="links"`, `dtype` (default `int64`), `offsets`, `has_perm`, `link_width`, `level_delta` |

**Record width.** Rows are `L` ints — or `1 + L`, `[perm_idx, vi_0 …
vi_{L-1}]`, when `has_perm` is true. `perm_idx` is a Lehmer code that
exists only to undo a canonical sort, so it is stored exactly where a
non-identity placement is possible: **non-intra AND `delta == 0` AND
(`store == "duplicate"` OR not `directed`)**. Decode at the physical width
given by `has_perm` (see [Decoding a cell blob](#decoding-a-cell-blob));
never infer it.

**Cell encoding** turns on one condition — `delta == 0` and offsets all
zero:

| Condition | Encoding | Sidecar |
|-----------|----------|---------|
| `delta == 0` **and** offsets all-zero | flat concatenated rows | `link_fragments/<chunk>` |
| otherwise | inline self-describing ragged blob | none |

**Vertex indices are local to their own endpoint's chunk**: `vi_k` indexes
the `vertices/` slice of chunk `src + o_k`, with `o_0 = 0` — so `vi_0` is
local to the cell itself. This is what lets an inter-chunk record store
plain local indices rather than global vertex IDs.

For polylines and streamlines, edges are stored in traversal order: edge `i`
connects vertex `i` to vertex `i+1` along the polyline — which is why the
intra array never sorts its endpoints. For graphs and skeletons, edge order
is not semantically significant.

For meshes (`link_width=3`) each row is one triangle's three vertex
indices. Vertex winding order is consistent within a store (default:
counter-clockwise when viewed from outside the surface, i.e. outward-facing
normals). The winding order convention is stored in root `.zattrs` under
`"winding_order"`: `"ccw"` (default) or `"cw"`. Where a face is
canonical-sorted, `perm_idx` is what recovers its winding.

See [Links](../object_model/links.md) for the offsets grammar, the
placement rules, and the enumeration order.

### `link_attributes/<name>/<delta>/<offsets>/`

Per-record attribute data, mirroring `links/<delta>/<offsets>/`
cell-for-cell: same delta, same offsets segments, same cells, one row per
link record in the same order.

| Property | Value |
|----------|-------|
| Dtype | `variable_length_bytes` (vlen-bytes serializer) |
| Shape | `chunk_grid_shape` (one cell per spatial chunk) |
| Fill value | `b""` (empty chunk ⇒ no chunk file) |
| Codec | `vlen-bytes` (+ optional `zstd` / `blosc` compressor; none by default) |
| Attributes | `chunk_grid_origin`, `nonempty_chunks`, `zv_array="link_attribute"`, `name`, `dtype`, `offsets`, `level_delta` |

The parent `link_attributes/<name>/<delta>/` is a **group**
(`zv_array="link_attribute_family"`, plus `name` and `level_delta`).

Attribute rows carry no `perm_idx` — they are per-record values, not
endpoints, so a placement permutation does not reorder them. Their
alignment to link records rests entirely on the shared enumeration order
that `read_links` and `read_link_attributes` both use; see
[Enumeration order](../object_model/links.md#enumeration-order).

### `attributes/<name>/`

One sub-group per named per-vertex attribute. The attribute name is a
valid Python identifier (alphanumeric and underscores only).

| Property | Value |
|----------|-------|
| Dtype | `variable_length_bytes` (vlen-bytes serializer) |
| Shape | `chunk_grid_shape` (one cell per spatial chunk) |
| Fill value | `b""` (empty chunk ⇒ no chunk file) |
| Codec | `vlen-bytes` (+ optional `zstd` / `blosc` compressor; none by default) |
| Attributes | `chunk_grid_origin`, `nonempty_chunks`, `zv_array="attribute"`, `name`, `dtype`, optional `channel_names` |

As with `vertices/`, the Zarr `data_type` is the container type; the
attribute's element type is the `dtype` **attribute**, and the column count
`K` is `len(channel_names)` (scalar — one column — when `channel_names` is
absent). A cell decodes as
`np.frombuffer(blob, dtype=attrs["dtype"]).reshape(-1, K)`, with the row
count derived from the blob length (see
[Decoding a cell blob](#decoding-a-cell-blob)). Honouring the stored `dtype`
matters more here than anywhere else in the format, since attributes are
routinely non-`float32` (integer labels, `float64` measurements).

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

### `cross_chunk_links/` — removed in 0.9.0

This array no longer exists, and neither does
`cross_chunk_link_attributes/`. A link that crosses a chunk boundary is not
a separate kind of object: it is a record in
[`links/<delta>/<offsets>/`](#linksdeltaoffsets) whose offsets are non-zero.

The replacement stores **no global vertex IDs at all**. Each endpoint's
chunk is recovered from the cell coordinate plus the offsets segment, and
its vertex index is local to that chunk — so there is nothing to reconstruct
and no global-ID encoding to decode. See
[Links](../object_model/links.md).
