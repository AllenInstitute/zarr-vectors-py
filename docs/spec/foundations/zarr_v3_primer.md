# Zarr v3 primer

The parts of Zarr v3 that Zarr Vectors uses, and how a store maps onto
Zarr groups, arrays and chunk keys.

## Terms

**Store**
: A mapping from slash-separated keys to bytes, on a local file system,
  an object store or in memory. A Zarr Vectors store is the
  `.zarrvectors` directory (or object-store prefix) and everything in it.
  See [Store types](store_types.md).

**Group**, **array**
: Zarr nodes, each described by a `zarr.json` file with `"node_type"`
  set to `"group"` or `"array"`. A group holds other nodes; an array
  holds data split into chunks.

**Attributes**
: The JSON object under `attributes` in a node's `zarr.json`. All Zarr
  Vectors metadata is kept there.

**Codec pipeline**
: The list of codecs in an array's `zarr.json`. See
  [Codec pipeline](codec_pipeline.md).

## Introduction

Every node in a store is a standard Zarr v3 group or array. A Zarr v3
reader can walk the store and read every chunk's bytes; it needs this
specification to interpret them. The
[Zarr v3 specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html)
is the full reference.

## Technical reference

### Hierarchy

A point cloud written with `write_points` has this tree:

```
example.zarrvectors/
    zarr.json                  ← root group: "zarr_vectors", "multiscales", "ome"
    0/
        zarr.json              ← level group: "zarr_vectors_level"
        vertices/
            zarr.json          ← array metadata and attributes
            c/0/0/0            ← cell for spatial chunk (0, 0, 0)
            …
        vertex_fragments/
            zarr.json
            c/0/0/0            ← fragment index for chunk (0, 0, 0)
            …
```

See [Directory structure](../layout/directory_structure.md),
[Root metadata](../layout/root_metadata.md) and
[Level groups](../layout/level_groups.md).

### Per-chunk arrays

Arrays such as `vertices/` have one cell per spatial chunk. Their shape
is the spatial chunk grid, their Zarr chunk shape is all ones, and
their `data_type` is `variable_length_bytes`. The `dtype` attribute
gives the element type inside each cell. For 500 points in a 200-unit
cube with `chunk_shape=(100.0, 100.0, 100.0)`, `0/vertices/zarr.json`
is:

```javascript
{
  "shape": [2, 2, 2],
  "data_type": "variable_length_bytes",
  "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1, 1, 1]}},
  "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
  "fill_value": "",
  "codecs": [{"name": "vlen-bytes", "configuration": {}}],
  "attributes": {
    "nonempty_chunks": ["0.0.0", "0.0.1", ...],
    "zv_array": "vertices",
    "dtype": "float32",
    "encoding": "raw"
  },
  ...
}
```

`nonempty_chunks` lists the cells that hold data. The cell layout is in
[Chunk arrays](../layout/chunk_arrays.md).

Other arrays, such as `object_attributes/<name>` and the object index,
are ordinary fixed-size arrays with a numeric `data_type`.

### Chunk keys and empty cells

Arrays use the `default` chunk key encoding with `/`. The cell at
`(i, j, k)` of `0/vertices` has the key `0/vertices/c/i/j/k`.

The fill value of a per-chunk array is the empty byte string. A spatial
chunk with no data has no key, and readers treat a missing key as an
empty cell.

### Zarr v2

Stores are Zarr v3 only. `zarr_vectors.open()` on a Zarr v2 hierarchy
raises `StoreError` ("Not a valid ZV store …").

### Validation

Level 1 (`validate/structure.py`) reports an error if the root
attributes have no `zarr_vectors` block, if there is no level, or if a
level has no `vertices` array. It also checks that every per-chunk
array in a level has the same rank as `vertices`. See
[Validation](../validation/overview.md).
