# Rechunking

## Terms

**Spatial rechunking**
: Writing a copy of a store with a different `chunk_shape`
  (`rechunk_spatial`).

**Re-binning**
: Changing one level's bin shape in place (`rebin_level`).

**Prefix rechunking**
: Writing a copy of level 0 with a leading chunk axis that groups
  objects (`rechunk`, `rechunk_by_attribute`). The result is an
  [attribute-chunked](attribute_chunking.md) level.

---

## Introduction

These functions rewrite an existing store's chunks or bins, without the
original input arrays. They are in `zarr_vectors.rechunk`, whose
stability tier is `internal`.

---

## Technical reference

### Changing the chunk shape

```python
rechunk_spatial(store_path, output, chunk_shape=None, *,
                level_chunk_shapes=None, target_links_per_chunk=8_000_000,
                levels=None, force=False) -> dict
```

| Argument | Meaning |
|----------|---------|
| `output` | New store path. Raises `FileExistsError` if it exists, unless `force=True`. The source is not changed |
| `chunk_shape` | `None` uses `suggest_chunk_shape(store_path)`, which aims for about `target_links_per_chunk` links per chunk |
| `level_chunk_shapes` | `{level: shape}` for levels with a different shape |
| `levels` | Which levels to copy |

The output keeps vertex positions (as float32), links at their source
`link_width`, object ids, object attributes, groups, group attributes and
`base_bin_shape`. It drops `vertex_attributes/`.

```python
from zarr_vectors.rechunk import rechunk_spatial

rechunk_spatial("lines.zarrvectors", "lines_400.zarrvectors",
                (400., 400., 400.), levels=[0])
# {'output_path': 'lines_400.zarrvectors',
#  'chunk_shape': (400.0, 400.0, 400.0), ...,
#  'levels': [{'level': 0, 'objects': 20, 'vertices': 1000, 'links': 180, ...}]}
```

For a pyramid on the new grid, copy level 0 only and rebuild the coarser
levels with [`build_pyramid`](../multiscale/pyramid_construction.md).

### Re-binning a level

```python
rebin_level(store_path, target_bin_shape, *, level=0) -> dict
```

`rebin_level` sorts each chunk's vertices into the new bins and rewrites
its fragment index. Chunk files keep their paths. It writes vertices as
float32, and with `level=0` sets the root `base_bin_shape`. It raises:

| Error | When |
|-------|------|
| `StoreError` | The root `geometry_types` is not exactly `["point_cloud"]` |
| `MetadataError` | `target_bin_shape` does not divide `chunk_shape` on every axis |
| `OctreeError` | The level has [octree chunks](octree_chunks.md) |

```{warning}
`rebin_level` reorders vertices inside each chunk but leaves
`vertex_attributes/` as it was, so per-vertex attributes do not line
up with their vertices. Re-bin only stores without per-vertex attributes,
or rewrite them afterwards.
```

### Grouping objects under a leading axis

```python
rechunk(store_path, spec, output=None) -> dict
rechunk_by_attribute(store_path, attribute_name, *, output=None,
                     spatial_chunk_shape=None) -> dict
```

`rechunk` assigns each object of level 0 to a bin by a `RechunkSpec` and
writes chunk keys `(bin, x, y, z)`:

| `RechunkSpec.by` | Bins |
|------------------|------|
| `"group"` | One per group; ungrouped objects go to a last bin labelled `-1` |
| `"object_id"` | Id ranges cut at `bins`, labelled by their lower edge |
| `"attribute:<name>"` | One per value when `categorical=True` or the per-object attribute has at most 10 distinct values; otherwise ranges cut at `bins`, or at quartiles when `bins` is `None` |
| `"spatial"` | One bin; only `spatial_chunk_shape` changes |

`spatial_chunk_shape` sets the output `chunk_shape`. `prefix_dim_name`
names the leading axis (by default the part of `by` before `:`).
`rechunk_by_attribute(store, name)` is `rechunk` with
`RechunkSpec(by="attribute:<name>", categorical=True, prefix_dim_name=name)`.
Bin order and labels are listed in
[Attribute chunking](attribute_chunking.md#bin-labels-from-a-rechunk).

```python
from zarr_vectors.rechunk import rechunk_by_attribute

rechunk_by_attribute("cells.zarrvectors", "cls", output="cells_by_cls.zarrvectors",
                     spatial_chunk_shape=(250., 250., 250.))
# {'objects_rechunked': 30, 'bins_created': 3, 'total_vertices': 3000,
#  'rechunk_dims': ['cls', 'x', 'y', 'z'], 'output_path': 'cells_by_cls.zarrvectors'}
```

The output level records `chunk_dims`, `chunk_attribute_name` and
`chunk_attribute_values` (here `[0, 1, 2]`).

The output holds level 0's vertices, object index and groups, with object
ids renumbered densely. It drops links, vertex and object attributes, and
levels above 0, and writes the level's `vertex_count` as 0. `rechunk`
refuses a store whose level 0 has octree chunks.

Without `output`, the store is replaced in place: the result goes to
`<name>.rechunked`, the source is renamed `<name>.backup`, the new store
is moved into place and the backup is deleted. With `output`, an existing
directory at that path is deleted first.

### Validation

Run `zarr_vectors.validate.validate(path, level=5)` on the output. The
chunk and bin checks are in [Chunk shape](chunk_shape.md#validation) and
[Bin shape](bin_shape.md#validation).
