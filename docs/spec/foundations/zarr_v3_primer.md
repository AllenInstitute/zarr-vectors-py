# Zarr v3 primer

## Terms

**Store**
: The root container of a Zarr hierarchy. A store maps string keys to bytes
  and can be backed by a local file system, a cloud object store, a ZIP
  archive, or an in-memory dictionary. In Zarr Vectors, a store corresponds to the
  `.zarrvectors` directory (or prefix).

**Group**
: A Zarr node that contains other groups and/or arrays. Analogous to a
  directory. Groups may carry metadata in a `.zattrs` file (Zarr v3:
  `zarr.json` under the group key). Zarr Vectors uses groups to represent the store
  root, each resolution level, and each array sub-group (e.g. `vertices/`,
  `attributes/`).

**Array**
: A Zarr node that stores an n-dimensional typed numeric array, divided into
  chunks. Each chunk is stored as one key in the underlying store (or, with
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

**`.zattrs`**
: A JSON file at a group path that stores arbitrary user-defined metadata.
  Zarr Vectors makes extensive use of `.zattrs` at the store root and at each
  resolution level group.

**`zarr.json`**
: In Zarr v3, the per-node metadata file (replaces `.zarray` and `.zgroup`
  from Zarr v2). Stores array shape, dtype, chunk shape, and codec pipeline.

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
- ZIP file (`zarr.storage.ZipStore`)
- In-memory (`zarr.storage.MemoryStore`)
- S3 via `s3fs` (requires `zarr-vectors[cloud]`)
- GCS via `gcsfs` (requires `zarr-vectors[cloud]`)

The store root of a Zarr Vectors store is identified by the path to the
`.zarrvectors` directory (or equivalent prefix in object storage).

### Groups and the hierarchy

A Zarr v3 group is represented in the store by:

- `<path>/zarr.json` — group metadata: `{"zarr_format": 3, "node_type": "group"}`
- `<path>/.zattrs` — optional user metadata (JSON object)

Zarr Vectors uses the following group hierarchy (abbreviated):

```
zarr.json                   ← store root group
.zattrs                     ← Zarr Vectors root metadata
0/
    zarr.json               ← resolution level group
    .zattrs                 ← per-level metadata (bin_ratio, object_sparsity)
    vertices/
        zarr.json           ← array metadata (shape, dtype, chunks, codecs)
        c/0/0/0             ← chunk (0,0,0) data bytes
        c/0/0/1             ← chunk (0,0,1) data bytes
        …
    vertex_fragments/
        zarr.json
        c/0/0/0             ← uint8 fragment-index blob for chunk (0,0,0)
        c/0/0/1
        …
```

All paths are relative to the store root.

### Array metadata (`zarr.json`)

Each Zarr v3 array carries a `zarr.json` file with at minimum:

```json
{
  "zarr_format": 3,
  "node_type": "array",
  "shape": [N, 3],
  "data_type": "float32",
  "chunk_grid": {
    "name": "regular",
    "configuration": { "chunk_shape": [65536, 3] }
  },
  "chunk_key_encoding": {
    "name": "default",
    "configuration": { "separator": "/" }
  },
  "codecs": [
    { "name": "bytes", "configuration": { "endian": "little" } },
    { "name": "blosc", "configuration": { "cname": "zstd", "clevel": 5 } }
  ],
  "fill_value": 0.0,
  "dimension_names": ["vertex", "spatial_dim"]
}
```

Zarr Vectors does not mandate a specific codec; any Zarr v3-compatible codec
pipeline is acceptable. The default pipeline used by `zarr-vectors-py` is
`bytes → blosc(zstd, level 5)`. Draco compression for mesh arrays uses a
custom codec registered with Zarr; see [Codec pipeline](codec_pipeline.md).

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
  stored. Zarr Vectors exploits this for sparse levels: chunks with no objects at a
  given resolution simply have no chunk keys in the store.
- The fill value for position arrays is `0.0`; for index arrays it is `-1`
  (indicating "no entry"). Readers must treat missing chunks as filled with
  the declared fill value.

### Zarr v2 compatibility

Zarr Vectors targets Zarr v3 exclusively. Zarr v2 stores use a different metadata
format (`.zarray`, `.zgroup`, Fortran vs C order ambiguity) and a different
chunk key scheme. `zarr-vectors-py` will raise `ValueError` if asked to open
a Zarr v2 store.

If you have data in a Zarr v2 store, migrate it with the `zarr` library's
conversion utilities before ingesting into Zarr Vectors.
