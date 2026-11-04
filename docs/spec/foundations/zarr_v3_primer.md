# Zarr v3 primer

## Terms

**Store**
: The root container of a Zarr hierarchy. A store maps string keys to bytes
  and can be backed by a local file system, a cloud object store, a ZIP
  archive, or an in-memory dictionary. In Zarr Vectors, a store corresponds to the
  `.zarrvectors` directory (or prefix); the extension is a convention, not
  read by the library.

**Group**
: A Zarr node that contains other groups and/or arrays. Analogous to a
  directory. In Zarr v3, a group's metadata and attributes are both in
  its `zarr.json`. Zarr Vectors uses groups for the store root, each
  resolution level, and namespace groups such as `links/<delta>/` and
  `vertex_attributes/` (whose children are the arrays).

**Array**
: A Zarr node that stores an n-dimensional typed array, divided into
  chunks. In Zarr Vectors the per-spatial-chunk arrays are variable-length
  byte (vlen-bytes) arrays: one array element per spatial chunk. Each chunk is stored as one key in the underlying store (or, with
  the sharding codec, multiple logical chunks per storage key).

**Chunk**
: The unit of storage for a Zarr array. Chunks are identified by a
  multi-dimensional integer index (the chunk coordinate). Each chunk is
  independently compressible and retrievable.

**Codec pipeline**
: An ordered sequence of transformations applied to chunk data before writing
  and reversed on reading. In Zarr v3, the pipeline is declared in the array
  metadata and may include array-to-bytes codecs (e.g. `bytes`), byte-to-byte
  codecs (e.g. `blosc`, `zstd`), and the `sharding_indexed` codec for
  sub-chunk addressing.

**Attributes**
: The JSON object in a node's `zarr.json` under `"attributes"` that stores
  arbitrary user-defined metadata (what Zarr v2 kept in `.zattrs`). Zarr
  Vectors uses the root attributes for the `zarr_vectors`, `multiscales`
  and `ome` blocks and each level group's attributes for
  `zarr_vectors_level`. Older ZV documentation and some code comments
  still say "`.zattrs`"; in a Zarr v3 store there is no such file.

**`zarr.json`**
: In Zarr v3, the per-node metadata file (replaces `.zarray`, `.zgroup`
  and `.zattrs` from Zarr v2). Stores array shape, dtype, chunk grid and
  codec pipeline, plus the node's attributes.

---

## Introduction

Zarr Vectors does not reinvent storage primitives. It is a *convention layer* built
on top of Zarr v3: every array inside a Zarr Vectors store is a standard Zarr v3
array, and every group is a standard Zarr v3 group. A reader that understands
only Zarr v3 can open any Zarr Vectors store, traverse its groups, and read raw array
data — it simply would not know how to interpret the Zarr Vectors-specific conventions
(the fragment index, the object model, the multiscale metadata).

This inheritance is intentional. It means Zarr Vectors benefits automatically from
Zarr's broad ecosystem: cloud backends (S3, GCS, Azure), codec libraries
(Blosc, Zstd, LZ4, Brotli), language bindings (Python, JavaScript, Java,
C++), and tools. Zarr Vectors adds only what Zarr itself does not provide: a spatial
index structure, an object model for discrete geometry, and a multi-
resolution metadata convention.

This page summarises the subset of Zarr v3 that Zarr Vectors relies on directly. It
is not a substitute for the
[Zarr v3 specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html);
consult that document for full detail.

---

## Technical reference

### Store model

A Zarr v3 store is an abstract key–value mapping. Keys are slash-separated
strings (e.g. `0/vertices/c/0/1/2`). Values are bytes. The
mapping is implemented by a *store backend*; Zarr Vectors supports the following
backends (see [Store types](store_types.md) for detail):

- Local file system (`zarr.storage.LocalStore`)
- Object stores via `obstore` (requires `zarr-vectors[obstore]`)
- S3 / GCS / Azure and other `fsspec` filesystems (requires
  `zarr-vectors[fsspec]`)
- `icechunk` transactional repositories (requires `zarr-vectors[icechunk]`)
- Any pre-built `zarr.abc.store.Store` passed directly, such as
  `zarr.storage.MemoryStore`

The store root of a Zarr Vectors store is identified by the path to the
`.zarrvectors` directory (or equivalent prefix in object storage).

### Groups and the hierarchy

A Zarr v3 group is represented in the store by a single
`<path>/zarr.json`:
`{"zarr_format": 3, "node_type": "group", "attributes": {...}}`.

Zarr Vectors uses the following group hierarchy (abbreviated):

```
zarr.json                   ← store root group; attributes: zarr_vectors, multiscales, ome
0/
    zarr.json               ← resolution level group; attributes: zarr_vectors_level
    vertices/
        zarr.json           ← array metadata (shape = chunk grid, vlen-bytes, codecs)
        c/0/0/0             ← cell (0,0,0): the vertex payload of spatial chunk (0,0,0)
        c/0/0/1
        …
    vertex_fragments/
        zarr.json
        c/0/0/0             ← fragment-index blob for chunk (0,0,0)
        c/0/0/1
        …
```

All paths are relative to the store root. With the sharding codec
(`shard_shape=`), several cells share one storage key.

### Array metadata (`zarr.json`)

Each per-spatial-chunk array is a Zarr v3 array of rank `D` whose **shape
is the level's chunk grid** (one array element per spatial chunk), with a
variable-length bytes data type. For a 3-D level with a 4 × 4 × 4 chunk
grid:

```json
{
  "zarr_format": 3,
  "node_type": "array",
  "shape": [4, 4, 4],
  "data_type": "bytes",
  "chunk_grid": {
    "name": "regular",
    "configuration": { "chunk_shape": [1, 1, 1] }
  },
  "chunk_key_encoding": {
    "name": "default",
    "configuration": { "separator": "/" }
  },
  "codecs": [ { "name": "vlen-bytes" } ],
  "attributes": { "chunk_grid_origin": [0, 0, 0], "nonempty_chunks": [...] }
}
```

(Illustrative: the exact `data_type` / codec spelling and the extra array
attributes come from the installed `zarr` version and from ZV; consult an
actual store's `zarr.json`. Without `shard_shape=` each cell is its own
chunk; with it the cells are wrapped in a `sharding_indexed` codec.) A
spatial chunk at absolute coordinate `c` lands in array cell
`c - chunk_grid_origin`; `nonempty_chunks` lists the cells that have been
written.

Zarr Vectors does not mandate a specific codec; any Zarr v3-compatible
codec pipeline is acceptable. The **type-level writers default to no
compression** (`bytes` only) and the dataset API defaults to `zstd`; see
[Codec pipeline](codec_pipeline.md). Draco mesh encoding is applied by the
mesh writer to the chunk payload and is not a Zarr codec.

### Chunk key encoding

Zarr Vectors uses the Zarr v3 **default** chunk key encoding with `/` as the
separator. A chunk at grid coordinate `(i, j, k)` in array
`0/vertices` is stored at:

```
0/vertices/c/i/j/k
```

The `c/` prefix is the Zarr v3 default; do not omit it.

### Metadata inheritance

All Zarr v3 metadata rules apply within Zarr Vectors stores:

- Array shape is the *logical* shape of the full array (all chunks combined).
  Chunks at the boundary of the array may be smaller than `chunk_shape` in
  one or more dimensions; Zarr v3 handles this transparently.
- Missing chunks (chunks that contain only the fill value) need not be
  stored. Zarr Vectors exploits this for sparse levels: cells with no data
  simply have no chunk key in the store, and read back as empty bytes
  (the vlen fill value).
- Dense object-level arrays (`object_attributes/<name>`) carry their
  absent-row marker in the array's `fill_value` (NaN for floats, a sentinel
  for integers); there is no sibling presence-mask array. Readers must treat
  missing chunks as filled with the declared fill value.

### Zarr v2 compatibility

Zarr Vectors targets Zarr v3 exclusively. Zarr v2 stores use a different metadata
format (`.zarray`, `.zgroup`, `.zattrs`, Fortran vs C order ambiguity) and a different
chunk key scheme. A v2 store will fail to open as a Zarr Vectors store (a
missing `zarr_vectors` root block is reported by `open_store` as a
`StoreError`).

If you have data in a Zarr v2 store, migrate it with the `zarr` library's
conversion utilities before ingesting into Zarr Vectors.
