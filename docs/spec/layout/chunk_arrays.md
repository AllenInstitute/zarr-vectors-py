# Chunk arrays

This page specifies the arrays inside a level group: the per-chunk arrays,
one cell per spatial chunk, and the per-object arrays.

## Terms

**Per-chunk array**
: A Zarr v3 array of `data_type` `variable_length_bytes` whose shape is the
  level's chunk grid. Each cell holds one chunk's payload as one byte blob;
  an empty chunk has no cell file. `vertices/`, `vertex_fragments/`,
  `link_fragments/`, and the arrays under `vertex_attributes/`,
  `fragment_attributes/`, `links/` and `link_attributes/` are per-chunk
  arrays.

**Chunk grid**
: The cells a level allocates for the store's bounds. Along axis `i`, with
  `lo` and `hi` the root `bounds` and `c` the chunk shape, the grid's
  **origin** is `floor(lo_i / c_i)` and its length is
  `floor(hi_i / c_i) - origin_i + 1`. The chunk at absolute coordinate `k`
  is cell `k - origin`. A level [chunked by an attribute](../chunking/attribute_chunking.md)
  adds a leading bin axis to every per-chunk array.

**`nonempty_chunks`**
: An array attribute listing the dotted keys (`"i.j.k"`, absolute chunk
  coordinates) of the cells that hold data. When it is absent, the level's
  presence is deferred (see [Level groups](level_groups.md#build-time-keys)):
  a reader must list or read the cells, and must not treat the array as
  empty.

## Introduction

Each per-chunk quantity is one Zarr array with one cell per chunk, so
reading one chunk of it is one object read. Per-object data is in ordinary
arrays indexed by object or group. Which arrays each
geometry type writes is on [Directory structure](directory_structure.md).

## Technical reference

### Per-chunk array metadata

The `zarr.json` of `vertices/` in an unsharded 3-D store with a 3 × 3 × 3
grid:

```javascript
{
  "shape": [3, 3, 3],
  "data_type": "variable_length_bytes",
  "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1, 1, 1]}},
  "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
  "fill_value": "",
  "codecs": [{"name": "vlen-bytes", "configuration": {}}],
  "attributes": {
    "nonempty_chunks": ["0.0.1", "0.1.1", "1.0.0", ...],
    "zv_array": "vertices",
    "dtype": "float32",
    "encoding": "raw"
  },
  ...
}
```

Every per-chunk array has this form; only the attributes differ.

| Feature | Rule |
|---------|------|
| Cells | Cell `(i, j, k)` is the file `c/i/j/k` |
| Compression | None by default. A writer's `compressor=` appends a codec after `vlen-bytes`; see [Codec pipeline](../foundations/codec_pipeline.md) |
| Sharding | With a root `shard_shape`, `chunk_grid.chunk_shape` is the shard shape, and the only codec is `sharding_indexed` with inner `chunk_shape` `[1, 1, 1]` and inner codec `vlen-bytes`. See [Sharding](../chunking/sharding.md) |
| Grid origin | A non-zero origin is stored as `chunk_grid_origin`; absent means zero. For bounds `[-300, 300]` and chunk shape 200, the shape is `[4, 4, 4]`, the origin `[-2, -2, -2]`, and keys look like `"-1.-1.-1"` |

Attributes of each per-chunk array, besides `nonempty_chunks` and
`chunk_grid_origin`:

| Array | Attributes |
|-------|------------|
| `vertices/` | `zv_array` `"vertices"`, `dtype`, `encoding` (`"raw"` or `"draco"`) |
| `vertex_fragments/` | `zv_array` `"vertex_fragments"`, `encoding` `"fragment_index_v1"` |
| `link_fragments/` | `zv_array` `"link_fragments"`, `encoding` `"fragment_index_v1"` |
| `vertex_attributes/<name>/` | `zv_array` `"attribute"`, `name`, `dtype`, `row_shape`, and `channel_names` for a multi-column attribute |
| `fragment_attributes/<name>/` | `zv_array` `"fragment_attribute"`, `name`, `dtype`, `row_shape`, and `channel_names` when given |
| `links/<delta>/<offsets>/` | `zv_array` `"links"`, `dtype` (`"int64"`), `offsets`, `has_perm`, `link_width`, `level_delta` |
| `link_attributes/<name>/<delta>/<offsets>/` | `zv_array` `"link_attribute"`, `name`, `dtype`, `offsets`, `level_delta`, `row_shape` |

`row_shape` is the shape of one row after the first axis: `[]` for one
value per row, `[3]` for three. A three-column vertex attribute written
without names gets `"channel_names": ["ch0", "ch1", "ch2"]`.

The groups holding these arrays have empty attributes, except the link
family groups below.

### Decoding a cell blob

A reader takes the element type from the array's `dtype` attribute and the
row width from the metadata. Cells have no header except where stated.

| Array | Cell content |
|-------|--------------|
| `vertices/` (`"raw"`) | `N × D` values of `dtype`, row-major; `D` is the number of axes in `multiscales[0].axes` |
| `vertices/` (`"draco"`) | A Draco-compressed mesh; `dtype` and `D` describe the decoded positions |
| `vertex_fragments/`, `link_fragments/` | A [fragment index](fragment_index_arrays.md) |
| `vertex_attributes/<name>/` | One row of shape `row_shape` per row of the same `vertices/` cell, in the same order |
| `fragment_attributes/<name>/` | One row of shape `row_shape` per fragment of the same `vertex_fragments/` cell |
| `links/0/<all-zero offsets>/` | Rows of `W` `dtype` values |
| Any other `links/<delta>/<offsets>/` | `int64` group count `K`, `K` `int64` byte offsets of the groups within the data that follows, then rows of `W` values |
| `link_attributes/<name>/<delta>/<offsets>/` | One row of shape `row_shape` per record of the matching link cell, in the same order |

`N` is not stored: it is the blob length divided by the row size, so use
the stored `dtype`. For a link array, `W = link_width + 1` when `has_perm`
is true (the extra leading column is `perm_idx`), and `W = link_width`
otherwise.

Decoding cell `(0, 0, 0)` of `vertices/` and of the intra-chunk link array
in a 3-D graph store:

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

### Link arrays

`links/<delta>/` is a group with one per-chunk array per offsets segment.
`<delta>` is the level delta (`0`, `+1`, `-1`, …). `<offsets>` says where a
record's other endpoints are relative to the cell that holds it: `0.0.0`
within the cell, `0.0.+1` one chunk along the last axis, `0.0.0_0.0.0` for
a triangle wholly in the cell. Endpoint `vi_0` indexes the cell's own
`vertices/` rows; `vi_k` indexes the rows of the chunk at the cell plus
offset `k`. The array's `offsets` attribute holds the parsed segment, for
example `[[0, 1, 1]]` for `0.+1.+1`. The grammar, placement rules and
`perm_idx` are on [Links](../object_model/links.md).

The family group's attributes, from a graph store:

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

`link_width` is 2 for edges and 3 for triangles. Cross-level families
(`+1`, `-1`) from `build_pyramid` have `"directed": true`.

`has_perm` is true exactly for a non-intra array at delta 0 whose family is
undirected or has `store` `"duplicate"`. Only the intra-chunk array at
delta 0 has a [`link_fragments/`](fragment_index_arrays.md) sidecar.

`link_attributes/<name>/<delta>/` mirrors `links/<delta>/` array for array
and cell for cell. Its attributes are `zv_array`
`"link_attribute_family"`, `name`, `level_delta` and `num_links`.

### Per-object arrays

These are ordinary arrays indexed by object or group, not by chunk. See
[Object manifest](../object_model/object_manifest.md) and
[Object attributes](../object_model/object_attributes.md).

| Path | Shape and type | Attributes |
|------|----------------|------------|
| `object_index/` (group) | — | `zv_array` `"object_index"`, `num_objects`, `num_present`, `sid_ndim`, `layout` (`"vlen_manifests_v2"`), `object_ids_sorted` |
| `object_index/manifests` | `[num_objects]`, `variable_length_bytes`, one manifest per object; chunk `[16384]`, `zstd` | none |
| `object_index/object_ids` | `[num_objects]`, `int64`; chunk `[16384]`, `zstd` | none |
| `object_attributes/<name>/` | `[num_objects]` or `[num_objects, K]`, the attribute's `dtype`; chunk `[65536]`; fill `NaN` for floats | `zv_array` `"object_attribute"`, `name`, `dtype`, `shape`, `fill_sentinel_meaning` `"absent"` |
| `groups` | `[num_groups]`, `variable_length_bytes`; cell `g` holds group `g`'s object IDs as `int64` | `zv_array` `"groups"`, `num_groups` |
| `group_attributes/<name>/` | `[num_groups]` | `zv_array` `"groupings_attribute"`, `name`, `dtype`, `shape` |

### Validation

[L1](../validation/l1_structural.md) checks that every per-chunk array in a
level has the rank of `vertices/`, that every `nonempty_chunks` key has
that many components, and, on a level chunked by an attribute, that the
leading axis has one entry per `chunk_attribute_values` label. It warns
when a `links/<delta>/` group has no arrays.

[L3](../validation/l3_consistency.md) checks that every `vertices/` cell
decodes to rows of `D` finite values and that every manifest names an
existing chunk and fragment. For links it applies the checks listed under
[Geometry types](../geometry_types/index.md#validation).
