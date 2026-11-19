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

**Array metadata**
: The array's `zarr_vectors` attribute: an object whose `kind` names the
  array's role and whose other keys (`dtype`, `nonempty_chunks`, …)
  describe it. The node also registers the convention with
  `"zarr_conventions": [{"uuid": "66268a61-7833-4995-bbcb-233bb417dae5"}]`.
  In Python, `zarr_vectors.core.convention.unwrap_array_meta` flattens it
  into the in-memory form, where the role is the `zv_array` tag; on disk
  the key is `zarr_vectors.kind`.

**`nonempty_chunks`**
: A key of the array metadata listing the dotted keys (`"i.j.k"`, absolute chunk
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
    "zarr_conventions": [{"uuid": "66268a61-7833-4995-bbcb-233bb417dae5"}],
    "zarr_vectors": {
      "kind": "vertices",
      "nonempty_chunks": ["0.0.1", "0.1.1", "1.0.0", ...],
      "dtype": "float32",
      "encoding": "raw"
    }
  },
  ...
}
```

Every per-chunk array has this form; only the `zarr_vectors` object
differs.

| Feature | Rule |
|---------|------|
| Cells | Cell `(i, j, k)` is the file `c/i/j/k` |
| Compression | None by default. A writer's `compressor=` appends a codec after `vlen-bytes`; see [Codec pipeline](../foundations/codec_pipeline.md) |
| Sharding | With a root `shard_shape`, `chunk_grid.chunk_shape` is the shard shape, and the only codec is `sharding_indexed` with inner `chunk_shape` `[1, 1, 1]` and inner codec `vlen-bytes`. See [Sharding](../chunking/sharding.md) |
| Grid origin | A non-zero origin is stored as `chunk_grid_origin`; absent means zero. For bounds `[-300, 300]` and chunk shape 200, the shape is `[4, 4, 4]`, the origin `[-2, -2, -2]`, and keys look like `"-1.-1.-1"` |

Keys of each per-chunk array's `zarr_vectors` object, besides
`nonempty_chunks` and `chunk_grid_origin`:

| Array | Keys |
|-------|------|
| `vertices/` | `kind` `"vertices"`, `dtype`, `encoding` (`"raw"` or `"draco"`) |
| `vertex_fragments/` | `kind` `"vertex_fragments"`, `encoding` `"fragment_index_v1"` |
| `link_fragments/` | `kind` `"link_fragments"`, `encoding` `"fragment_index_v1"` |
| `vertex_attributes/<name>/` | `kind` `"vertex_attribute"`, `name`, `dtype`, `row_shape`, and `channel_names` for a multi-column attribute |
| `fragment_attributes/<name>/` | `kind` `"fragment_attribute"`, `name`, `dtype`, `row_shape`, and `channel_names` when given |
| `links/<delta>/<offsets>/` | `kind` `"links"`, `dtype` (`"int64"`), `offsets`, `has_perm`, `link_width`, `level_delta` |
| `link_attributes/<name>/<delta>/<offsets>/` | `kind` `"link_attribute"`, `name`, `dtype`, `offsets`, `level_delta`, `row_shape` |

The in-memory `zv_array` tag equals `kind` except for two arrays: the
tag `"attribute"` is stored as `"vertex_attribute"`, and
`"groupings_attribute"` as `"group_attribute"`.

`row_shape` is the shape of one row after the first axis: `[]` for one
value per row, `[3]` for three. A three-column vertex attribute written
without names gets `"channel_names": ["ch0", "ch1", "ch2"]`. An array
without `row_shape` (an older store) has `len(channel_names)` columns,
or one.

A categorical vertex attribute stores integer codes and adds
`encoding: "dictionary"`, `categories` (the values, indexed by code),
`ordered` and, optionally, `_FillValue` (the code read as missing).
Readers return `categories[code]`.

`fragment_attributes/object_id`, written by
`zarr_vectors.building.build_fragment_owner_index`, holds the `int64` id
of the object that owns each fragment: `-1` when no manifest names it,
`-2` when several do. It is derived from the manifests and is not
updated when they change.

The groups holding these arrays have empty attributes, except the link
family groups and `object_index/` below.

### Decoding a cell blob

A reader takes the element type from the `dtype` key of the array's
`zarr_vectors` object and the row width from the metadata. Cells have no header except where stated.

| Array | Cell content |
|-------|--------------|
| `vertices/` (`"raw"`) | `N × D` values of `dtype`, row-major; `D` is the number of axes in the root `zarr_vectors.axes` |
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
dtype = verts.attrs["zarr_vectors"]["dtype"]
positions = np.frombuffer(blob, dtype=dtype).reshape(-1, 3)

links = level["links/0/0.0.0"]
a = links.attrs["zarr_vectors"]
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
  "zarr_conventions": [{"uuid": "66268a61-7833-4995-bbcb-233bb417dae5"}],
  "zarr_vectors": {
    "kind": "links_family",
    "level_delta": 0,
    "link_width": 2,
    "directed": false,
    "store": "canonical",
    "sid_ndim": 3,
    "num_links": 396,
    "num_physical_records": 396
  }
}
```

`link_width` is 2 for edges and 3 for triangles. Cross-level families
(`+1`, `-1`) from `build_pyramid` have `"directed": true`.

`has_perm` is true exactly for a non-intra array at delta 0 whose family is
undirected or has `store` `"duplicate"`. Only the intra-chunk array at
delta 0 has a [`link_fragments/`](fragment_index_arrays.md) sidecar.

`link_attributes/<name>/<delta>/` mirrors `links/<delta>/` array for array
and cell for cell. Its `zarr_vectors` object has `kind`
`"link_attribute_family"`, `name`, `level_delta` and `num_links`.

### Per-object arrays

These are ordinary arrays indexed by object or group, not by chunk. See
[Object manifest](../object_model/object_manifest.md) and
[Object attributes](../object_model/object_attributes.md).

| Path | Shape and type | `zarr_vectors` keys |
|------|----------------|------------|
| `object_index/` (group) | — | `kind` `"object_index"`, `num_objects`, `num_present`, `sid_ndim`, `layout`, `object_ids_sorted`; `block_dtype` on an `int32` dense index |
| `object_index/manifests` | vlen layouts: `[num_objects]`, `variable_length_bytes`, one manifest per object; chunk `[16384]`, `zstd` | none |
| `object_index/manifest_spans` | dense layout: `[num_objects, 2]`, `int64` | none |
| `object_index/manifest_blocks` | dense layout: `[n_blocks, sid_ndim + 1]`, `int64` or `int32` | none |
| `object_index/object_ids` | `[num_objects]`, `int64`; chunk `[16384]`, `zstd` | none |
| `object_attributes/<name>/` | `[num_objects]` or `[num_objects, K]`, the attribute's `dtype`; chunk `[65536]`; fill `NaN` for floats | `kind` `"object_attribute"`, `name`, `dtype`, `shape`, `fill_sentinel_meaning` `"absent"` |
| `groups` | `[num_groups]`, `variable_length_bytes`; cell `g` holds group `g`'s object IDs as `int64` | `kind` `"groups"`, `num_groups`, optional `group_ranges` |
| `group_attributes/<name>/` | `[num_groups]` | `kind` `"group_attribute"`, `name`, `dtype`, `shape` |

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
