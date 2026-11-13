# Zarr v3 primer

This page summarises the parts of Zarr v3 that Zarr Vectors relies on
and shows how a store maps onto Zarr groups, arrays and chunk keys.

## Terms

**Store**
: A key–value mapping from slash-separated string keys to bytes, backed
  by a local file system, an object store or memory. A Zarr Vectors
  store is the `.zarrvectors` directory (or object-store prefix) and
  everything under it.

**Group**
: A Zarr node that contains other groups and arrays, described by a
  `zarr.json` file with `"node_type": "group"`. Zarr Vectors uses groups
  for the store root, each resolution level, and array families such as
  `vertex_attributes/` and `links/`.

**Array**
: A Zarr node holding an n-dimensional array divided into chunks,
  described by a `zarr.json` file with `"node_type": "array"`.

**Chunk**
: The unit of storage of a Zarr array, identified by its integer grid
  coordinate and stored under one key.

**Codec pipeline**
: The ordered list of codecs, declared in an array's `zarr.json`, that
  encodes chunk data on write and decodes it on read. See
  [Codec pipeline](codec_pipeline.md).

**Attributes**
: The free-form JSON object under the `attributes` key of a node's
  `zarr.json`. Zarr Vectors keeps all of its metadata there: the root
  attributes, each level's attributes and each array's attributes.

---

## Introduction

Zarr Vectors is a convention on top of Zarr v3: every node in a store
is a standard Zarr v3 group or array. A reader that knows only Zarr v3
can open a store, walk its groups and read each chunk's bytes; it needs
this specification to interpret those bytes and the metadata in the
attributes.

Zarr Vectors adds a spatial chunk layout for variable-length geometry,
a fragment index within each chunk, an object model, and metadata for
resolution pyramids. This page covers only the Zarr v3 features those
layers use. The
[Zarr v3 specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html)
is the full reference.

---

## Technical reference

### Store model

Keys are slash-separated strings such as `0/vertices/c/0/1/1`; values
are bytes. `zarr-vectors-py` reaches the key–value mapping through one
of several backends: the local file system, object stores (S3, Google
Cloud Storage, Azure, HTTP) through `obstore` or `fsspec`, Icechunk
repositories, and in-memory stores. See [Store types](store_types.md).

### Groups and the hierarchy

Each group and array is a `zarr.json` file at its path. A point cloud
written with `write_points` has this tree:

```
example.zarrvectors/
    zarr.json                  ← root group; root attributes hold
                                  "zarr_vectors", "multiscales", "ome"
    0/
        zarr.json              ← level group; level attributes hold
                                  "zarr_vectors_level"
        vertices/
            zarr.json          ← array metadata and array attributes
            c/0/0/0            ← the cell for spatial chunk (0, 0, 0)
            c/0/0/1
            …
        vertex_fragments/
            zarr.json
            c/0/0/0            ← fragment-index blob for chunk (0, 0, 0)
            …
```

The full set of nodes is in
[Directory structure](../layout/directory_structure.md); the root
attributes are in [Root metadata](../layout/root_metadata.md) and the
level attributes in [Level groups](../layout/level_groups.md).

### Array metadata (`zarr.json`)

Per-chunk arrays such as `vertices/` use the Zarr v3
`variable_length_bytes` data type. The array has one cell per spatial
chunk, so its shape is the spatial chunk grid and its Zarr chunk shape
is all ones; each cell holds one spatial chunk's encoded contents.
This is the `zarr.json` of `0/vertices` for 500 points in a 200-unit
cube written with `chunk_shape=(100.0, 100.0, 100.0)`:

```json
{
  "shape": [2, 2, 2],
  "data_type": "variable_length_bytes",
  "chunk_grid": {
    "name": "regular",
    "configuration": {"chunk_shape": [1, 1, 1]}
  },
  "chunk_key_encoding": {
    "name": "default",
    "configuration": {"separator": "/"}
  },
  "fill_value": "",
  "codecs": [
    {"name": "vlen-bytes", "configuration": {}}
  ],
  "attributes": {
    "nonempty_chunks": ["0.0.0", "0.0.1", "0.1.0", "0.1.1",
                        "1.0.0", "1.0.1", "1.1.0", "1.1.1"],
    "zv_array": "vertices",
    "dtype": "float32",
    "encoding": "raw"
  },
  "zarr_format": 3,
  "node_type": "array",
  "storage_transformers": []
}
```

The `dtype` attribute gives the element type inside each cell; the Zarr
`data_type` of a per-chunk array is always `variable_length_bytes`.
`nonempty_chunks` lists the cells that hold data. A compressor, when
one is set, follows `vlen-bytes` in `codecs`. The cell layout is in
[Chunk arrays](../layout/chunk_arrays.md) and the codec choices are in
[Codec pipeline](codec_pipeline.md).

Arrays that are not per-chunk, such as `object_attributes/<name>` and
the object index, are ordinary fixed-size Zarr arrays with a numeric
`data_type`. See [Object manifest](../object_model/object_manifest.md)
and [Object attributes](../object_model/object_attributes.md).

### Chunk key encoding

Arrays use the Zarr v3 `default` chunk key encoding with `/` as the
separator. The cell at grid coordinate `(i, j, k)` of `0/vertices` is
stored under the key:

```
0/vertices/c/i/j/k
```

The `c/` segment is part of the default encoding.

### Empty cells

A Zarr v3 chunk that was never written reads as the fill value. For a
per-chunk array the fill value is the empty byte string `""`, so a
spatial chunk with no data at a level has no key in the store. Readers
treat a missing key and an empty cell the same way.

### Zarr v2

Stores are Zarr v3 only. `zarr_vectors.open()` on a Zarr v2 hierarchy
raises `StoreError` ("Not a valid ZV store …"), because it finds no
Zarr v3 root group.

### Validation

Level 1 (`zarr_vectors/validate/structure.py`) opens the store through
Zarr and reports an error if the root attributes have no `zarr_vectors`
block, if there is no resolution level, or if a level has no `vertices`
array. It also checks that every per-chunk array in a level has the same
rank as `vertices`. See [Validation](../validation/overview.md).
