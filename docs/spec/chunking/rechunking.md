# Rechunking

## Terms

**Spatial rechunking**
: Writing a copy of a store with a different `chunk_shape`. Done by
  `rechunk_spatial`.

**Re-binning**
: Changing the bin shape of one level of a point-cloud store in place,
  keeping its chunks. Done by `rebin_level`.

**Prefix rechunking**
: Writing a copy of level 0 whose chunk keys gain a leading axis that
  groups objects by group, object id range or attribute value. Done by
  `rechunk` and `rechunk_by_attribute`. The result is an
  [attribute-chunked](attribute_chunking.md) level.

---

## Introduction

[`chunk_shape`](chunk_shape.md) fixes how a store is split into files,
so changing it means writing the data again. The functions on this page
do that from an existing store, without the original input arrays.

All of them are in `zarr_vectors.rechunk`, whose stability tier is
`internal` (`zarr_vectors.stability("zarr_vectors.rechunk")`): they may
change without notice.

---

## Technical reference

### Changing the chunk shape

```python
rechunk_spatial(store_path, output, chunk_shape=None, *,
                level_chunk_shapes=None, target_links_per_chunk=8_000_000,
                levels=None, force=False) -> dict
```

`rechunk_spatial` writes a new store at `output`; the source is not
changed. It raises `FileExistsError` if `output` exists, unless
`force=True`. When `chunk_shape` is `None`, it uses
`suggest_chunk_shape(store_path)`, which aims for about
`target_links_per_chunk` links per chunk. `level_chunk_shapes`
(`{level: shape}`) sets a different shape for a level, and `levels`
restricts which levels are copied.

What the output holds:

| Kept | Not kept |
|------|----------|
| Vertex positions of every object, written as float32 | Per-vertex attributes (`vertex_attributes/`) |
| Links, at their source `link_width` | |
| Object ids, object attributes, groups and group attributes | |
| `base_bin_shape` | |

Worked example, run on a two-level polyline store of 20 polylines with
`chunk_shape = 200`:

```python
from zarr_vectors.rechunk import rechunk_spatial

rechunk_spatial("lines.zarrvectors", "lines_400.zarrvectors",
                (400., 400., 400.), levels=[0])
# {'output_path': 'lines_400.zarrvectors',
#  'chunk_shape': (400.0, 400.0, 400.0),
#  'level_chunk_shapes': {0: (400.0, 400.0, 400.0)},
#  'levels': [{'level': 0, 'objects': 20, 'vertices': 1000, 'links': 180,
#              'link_width': 2, 'attributes': 1,
#              'chunk_shape': (400.0, 400.0, 400.0)}]}
```

To get a pyramid on the new chunk grid, copy level 0 only, as above, and
rebuild the coarser levels with
[`build_pyramid`](../multiscale/pyramid_construction.md).

### Re-binning a level

```python
rebin_level(store_path, target_bin_shape, *, level=0) -> dict
```

`rebin_level` rewrites the fragment index of each chunk of one level
and sorts the chunk's vertices into the new bins. Chunk files keep their
paths.

- It raises `StoreError` unless the root `geometry_types` is exactly
  `["point_cloud"]`. Other types use fragments for objects or segments,
  which a sort by bin would lose.
- It raises `MetadataError` unless `target_bin_shape` divides
  `chunk_shape` on every axis.
- It refuses a level with [octree chunks](octree_chunks.md).
- It writes vertices as float32.
- With `level=0`, it sets the root `base_bin_shape` to
  `target_bin_shape`.

```python
from zarr_vectors.rechunk import rebin_level

rebin_level("points.zarrvectors", (100., 100., 100.))
# {'chunks_processed': 125, 'vertices_processed': 5000, 'bins_created': 987,
#  'old_bin_shape': (50.0, 50.0, 50.0), 'new_bin_shape': (100.0, 100.0, 100.0),
#  'level': 0}
```

```{warning}
`rebin_level` reorders vertices inside each chunk but leaves
`vertex_attributes/` as it was, so per-vertex attributes do not line
up with their vertices. Rewrite them after re-binning, or re-bin only
stores without per-vertex attributes.
```

### Grouping objects under a leading axis

```python
rechunk(store_path, spec, output=None) -> dict
rechunk_by_attribute(store_path, attribute_name, *, output=None,
                     spatial_chunk_shape=None) -> dict
```

`rechunk` assigns each object of level 0 to a bin according to a
`RechunkSpec` and writes the result with chunk keys
`(bin, x, y, z)`:

| `RechunkSpec.by` | Bins |
|------------------|------|
| `"group"` | One per group; ungrouped objects go to a last bin labelled `-1` |
| `"object_id"` | Id ranges cut at `bins`, labelled by their lower edge |
| `"attribute:<name>"` | One per value when `categorical=True` or the per-object attribute has at most 10 distinct values; otherwise ranges cut at `bins`, or at quartiles when `bins` is `None` |
| `"spatial"` | One bin; only `spatial_chunk_shape` changes |

`spatial_chunk_shape` sets the output `chunk_shape`; `prefix_dim_name`
names the leading axis (by default the part of `by` before `:`).
`rechunk_by_attribute(store, name)` is `rechunk` with
`RechunkSpec(by="attribute:<name>", categorical=True, prefix_dim_name=name)`.

```python
from zarr_vectors.rechunk import rechunk_by_attribute

rechunk_by_attribute("cells.zarrvectors", "cls", output="cells_by_cls.zarrvectors",
                     spatial_chunk_shape=(250., 250., 250.))
# {'objects_rechunked': 30, 'bins_created': 3, 'total_vertices': 3000,
#  'rechunk_dims': ['cls', 'x', 'y', 'z'], 'output_path': 'cells_by_cls.zarrvectors'}
```

The output level's attributes record the leading axis:

```json
{
  "zarr_vectors_level": {
    "chunk_dims": ["cls", "x", "y", "z"],
    "chunk_attribute_name": "cls",
    "chunk_attribute_values": [0, 1, 2]
  }
}
```

and its `vertices` array has shape `[3, 4, 4, 4]`.

What the output holds: level 0's vertices, object index and groups, with
object ids renumbered densely. It does not hold links, vertex or object
attributes, or levels above 0, and the level's `vertex_count` is written
as 0. `rechunk` refuses a store whose level 0 has octree chunks.

Without `output`, the store is replaced in place: the result is written
to `<name>.rechunked`, the source is renamed to `<name>.backup`, the new
store is moved into place and the backup is deleted. With `output`, an
existing directory at that path is deleted first.

### Validation

Run `zarr_vectors.validate.validate(path, level=5)` on the output. The
checks that apply to chunk and bin shapes are listed in
[Chunk shape](chunk_shape.md#validation) and
[Bin shape](bin_shape.md#validation).
