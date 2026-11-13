# Chunk arrays

This page specifies the Zarr arrays inside a level group: the per-chunk
arrays (one cell per spatial chunk) and the per-object arrays, with their
`zarr.json` and how to decode a cell.

## Terms

**Per-chunk array**
: A Zarr v3 array of `data_type` `variable_length_bytes` whose shape is the
  level's chunk grid. Each cell holds one spatial chunk's payload as one
  byte blob; an empty chunk has no cell file. `vertices/`,
  `vertex_fragments/`, `link_fragments/`, `vertex_attributes/<name>/`,
  `fragment_attributes/<name>/`, `links/<delta>/<offsets>/` and
  `link_attributes/<name>/<delta>/<offsets>/` are per-chunk arrays.

**Chunk grid**
: The cells a level allocates for the store's bounds. Along axis `i`, with
  `lo` and `hi` the root `bounds` and `c` the chunk shape, the grid's
  **origin** is `floor(lo_i / c_i)` and its length is
  `floor(hi_i / c_i) - origin_i + 1`. The chunk at absolute coordinate `k`
  is cell `k - origin`. A level [chunked by an attribute](../chunking/attribute_chunking.md)
  adds a leading bin axis to every per-chunk array, so all arrays in a level
  share one rank.

**`nonempty_chunks`**
: An array attribute listing the dotted keys (`"i.j.k"`, in absolute chunk
  coordinates) of the cells that hold data, so a reader can list a level's
  chunks without scanning cells. When the attribute is absent, the level's
  presence is deferred (see [Level groups](level_groups.md#build-time-keys)):
  a reader must list or read the cells instead, and must not treat the array
  as empty.

## Introduction

Every per-chunk quantity — positions, connectivity, fragment indexes and
attributes — is one Zarr array with one cell per chunk, so reading a chunk
of any of them is one object read. Per-object data (the object index,
object and group attributes, group membership) is in ordinary
one-dimensional arrays indexed by object or group. Which arrays each
geometry type writes is on [Directory structure](directory_structure.md).

## Technical reference

### Per-chunk array metadata

The `zarr.json` of `vertices/` in an unsharded 3-D mesh store with a
3 × 3 × 3 grid:

```json
{
  "shape": [3, 3, 3],
  "data_type": "variable_length_bytes",
  "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1, 1, 1]}},
  "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
  "fill_value": "",
  "codecs": [{"name": "vlen-bytes", "configuration": {}}],
  "attributes": {
    "nonempty_chunks": ["0.0.1", "0.1.1", "1.0.0", "1.0.1", "1.1.0", "1.1.1"],
    "zv_array": "vertices",
    "dtype": "float32",
    "encoding": "raw"
  },
  "zarr_format": 3,
  "node_type": "array",
  "storage_transformers": []
}
```

Every per-chunk array shares this form; only the attributes differ.

- **Cells.** Cell `(i, j, k)` is the file `c/i/j/k`.
- **Compression.** None by default. A writer's `compressor=` argument
  appends a codec after `vlen-bytes`, for example
  `{"name": "zstd", "configuration": {"level": 0, "checksum": false}}`;
  see [Codec pipeline](../foundations/codec_pipeline.md).
- **Sharding.** With a root `shard_shape`, `chunk_grid.chunk_shape` is the
  shard shape and the codec list is one `sharding_indexed` codec with inner
  `chunk_shape` `[1, 1, 1]` and inner codec `vlen-bytes`; the files under
  `c/` are then shards. See [Sharding](../chunking/sharding.md).
- **Grid origin.** When the grid origin is not zero, every per-chunk array
  carries it as `chunk_grid_origin`. For bounds `[-300, 300]` and chunk
  shape 200 on each axis, the arrays have shape `[4, 4, 4]`,
  `"chunk_grid_origin": [-2, -2, -2]`, and `nonempty_chunks` keys such as
  `"-1.-1.-1"`. Absent means a zero origin.

The attributes of each per-chunk array:

| Array | Attributes besides `nonempty_chunks` and `chunk_grid_origin` |
|-------|----------------------------------------------------------------|
| `vertices/` | `zv_array` `"vertices"`, `dtype`, `encoding` (`"raw"` or `"draco"`) |
| `vertex_fragments/` | `zv_array` `"vertex_fragments"`, `encoding` `"fragment_index_v1"` |
| `link_fragments/` | `zv_array` `"link_fragments"`, `encoding` `"fragment_index_v1"` |
| `vertex_attributes/<name>/` | `zv_array` `"attribute"`, `name`, `dtype`, `row_shape`, and `channel_names` for a multi-column attribute |
| `fragment_attributes/<name>/` | `zv_array` `"fragment_attribute"`, `name`, `dtype`, `row_shape`, and `channel_names` when given |
| `links/<delta>/<offsets>/` | `zv_array` `"links"`, `dtype` (`"int64"`), `offsets`, `has_perm`, `link_width`, `level_delta` |
| `link_attributes/<name>/<delta>/<offsets>/` | `zv_array` `"link_attribute"`, `name`, `dtype`, `offsets`, `level_delta`, `row_shape` |

`row_shape` is the shape of one row after the first axis: `[]` for one
value per row, `[3]` for three. A three-column vertex attribute written
without names gets `"channel_names": ["ch0", "ch1", "ch2"]` and
`"row_shape": [3]`.

The groups that hold these arrays (`vertex_attributes/`,
`fragment_attributes/`, `links/`, `link_attributes/`,
`link_attributes/<name>/`) have empty attributes, except the two link
family groups below.

### Decoding a cell blob

Cells carry no header except where stated, and the Zarr `data_type`
describes only the container. A reader takes the element type from the
array's `dtype` attribute and the row width from the metadata:

| Array | Cell content |
|-------|--------------|
| `vertices/` (`encoding` `"raw"`) | `N × D` values of `dtype`, row-major; `D` is the number of axes in the root `multiscales[0].axes` |
| `vertices/` (`encoding` `"draco"`) | A Draco-compressed mesh; `dtype` and `D` describe the decoded positions |
| `vertex_fragments/`, `link_fragments/` | A [fragment index](fragment_index_arrays.md) |
| `vertex_attributes/<name>/` | One row of shape `row_shape` per vertex row of the same `vertices/` cell, in the same order |
| `fragment_attributes/<name>/` | One row of shape `row_shape` per fragment of the same `vertex_fragments/` cell |
| `links/<delta>/<offsets>/`, intra-chunk at delta 0 | Rows of `W` `dtype` values, no header |
| `links/<delta>/<offsets>/`, any other | `int64` group count `K`, `K` `int64` byte offsets of the groups within the data that follows, then rows of `W` values |
| `link_attributes/<name>/<delta>/<offsets>/` | One row of shape `row_shape` per record of the matching link cell, in the same order, no header |

`N` is never stored: it is the blob length divided by the row size, so a
reader must use the stored `dtype` (a `float64` cell read as `float32`
gives twice the rows and wrong values). For a link array the physical row
width is `W = link_width + 1` when `has_perm` is true (the extra leading
column is `perm_idx`) and `W = link_width` otherwise.

For example, decoding cell `(0, 0, 0)` of `vertices/` and of an intra-chunk
link array in a 3-D graph store:

```python
import numpy as np
import zarr

level = zarr.open_group("graph.zarrvectors", mode="r")["0"]

verts = level["vertices"]
blob = verts[0:1, 0:1, 0:1].ravel()[0]
positions = np.frombuffer(blob, dtype=verts.attrs["dtype"]).reshape(-1, 3)

links = level["links/0/0.0.0"]
a = links.attrs
width = a["link_width"] + (1 if a["has_perm"] else 0)
blob = links[0:1, 0:1, 0:1].ravel()[0]
edges = np.frombuffer(blob, dtype=a["dtype"]).reshape(-1, width)
```

### `links/<delta>/` and `link_attributes/<name>/<delta>/` groups

`links/<delta>/` is a group whose children are one per-chunk array per
offsets segment. `<delta>` is the level delta (`0`, `+1`, `-1`, …) and
`<offsets>` names where a record's other endpoints sit relative to the cell
that holds it: `0.0.0` within the cell, `0.0.+1` one chunk along the last
axis, `0.0.0_0.0.0` for a triangle with all three vertices in the cell. The
grammar, the placement rules and `perm_idx` are specified on
[Links](../object_model/links.md).

The group's attributes, from a graph store:

```json
{
  "zv_array": "links_family",
  "level_delta": 0,
  "link_width": 2,
  "directed": false,
  "store": "canonical",
  "sid_ndim": 3,
  "num_links": 396,
  "num_physical_records": 396
}
```

`link_width` is 2 for edges and 3 for mesh triangles. Cross-level groups
(`+1`, `-1`) written by `build_pyramid` have `"directed": true`.

`link_attributes/<name>/<delta>/` mirrors `links/<delta>/` array for array
and cell for cell. Its attributes are `zv_array`
`"link_attribute_family"`, `name`, `level_delta` and `num_links`.

### `links/<delta>/<offsets>/`

Within a cell, `vi_0` indexes the cell's own `vertices/` rows, and `vi_k`
indexes the rows of the chunk at the cell plus offset `k`, so records never
need global vertex IDs. The array's `offsets` attribute holds the parsed
segment, for example `[[0, 1, 1]]` for `0.+1.+1`.

`has_perm` is true exactly for a non-intra array at delta 0 whose family is
undirected or has `store` `"duplicate"`. The intra-chunk array at delta 0
is the only link array with a [`link_fragments/`](fragment_index_arrays.md)
sidecar.

### Per-object arrays

These are ordinary arrays indexed by object or group, not by chunk.

**`object_index/`** is a group with the attributes

```json
{
  "zv_array": "object_index",
  "num_objects": 20,
  "num_present": 20,
  "sid_ndim": 3,
  "layout": "vlen_manifests_v2",
  "object_ids_sorted": true
}
```

and two arrays: `manifests`, `variable_length_bytes` with shape
`[num_objects]`, one manifest per object; and `object_ids`, `int64` with
the same shape. Both have chunk shape `[16384]`, a `zstd` codec and no
attributes. The manifest format and the dense layout are on
[Object manifest](../object_model/object_manifest.md).

**`object_attributes/<name>/`** is one array per attribute with shape
`[num_objects]` (or `[num_objects, K]`), chunk shape `[65536]`, the
attribute's own `dtype`, and fill value `NaN` for floating-point types.
Attributes: `zv_array` `"object_attribute"`, `name`, `dtype`, `shape`, and
`fill_sentinel_meaning` `"absent"`.

**`groups`** is one `variable_length_bytes` array of shape `[num_groups]`
whose cell `g` holds group `g`'s object IDs as `int64`. Attributes:
`zv_array` `"groups"`, `num_groups`.

**`group_attributes/<name>/`** is one array per attribute with shape
`[num_groups]`. Attributes: `zv_array` `"groupings_attribute"`, `name`,
`dtype`, `shape`.

See [Object attributes](../object_model/object_attributes.md) for how these
are written and read.

### Validation

[L1](../validation/l1_structural.md) checks that every per-chunk array in a
level has the same rank as `vertices/`, that every `nonempty_chunks` key
has that many components, and, on a level chunked by an attribute, that the
leading axis has one entry per `chunk_attribute_values` label. It lists
each `links/<delta>/` group and warns when one has no arrays.

[L3](../validation/l3_consistency.md) decodes every `vertices/` cell and
checks that each row has `D` columns and finite values, that every manifest names an existing chunk and
fragment, that each offsets segment parses against the family's
`link_width` and `sid_ndim`, that an undirected canonical family at delta 0
has no lexicographically negative offset, that `num_physical_records`
matches the records on disk, and that each record's source chunk (and at
delta 0, every endpoint's chunk) exists.
