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
stability tier is `internal`. Each checks the whole source before it
writes anything, and refuses (rather than drops) data it cannot carry.

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
| `output` | New store path. Raises `ValueError` if it is the source, and `FileExistsError` if it exists, unless `force=True`. The source is not changed |
| `chunk_shape` | `None` uses `suggest_chunk_shape(store_path)`, which halves the chunk shape until the busiest chunk of any level holds about `target_links_per_chunk` links |
| `level_chunk_shapes` | `{level: shape}` for levels with a coarser shape; each must be an integer multiple of `chunk_shape` |
| `levels` | Which levels to copy. Links to levels not copied are dropped |

The output keeps vertex positions in the source dtype, links at their
source `link_width`, object ids, vertex attributes, object attributes,
groups, group attributes, `base_bin_shape`, `shard_shape` and the octree
sharding marker. Levels above 0 are marked `preserves_object_ids`.

It raises before writing:

| Error | When |
|-------|------|
| `NotImplementedError` | A copied level has `fragment_attributes/`, `link_attributes/` or another family it does not carry |
| `OctreeError` | A copied level has [octree chunks](octree_chunks.md) |

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

`rechunk_spatial` holds one whole level in memory while it writes it.
For a store larger than memory, copy one level per call or write the
store again from its source.

### Re-binning a level

```python
rebin_level(store_path, target_bin_shape, *, level=0) -> dict
```

`rebin_level` sorts each chunk's vertices, and their vertex attributes,
into the new bins and rewrites the fragment index with one fragment per
occupied bin, in bin order. Fragment `f` is then not the bin with
row-major index `f`. Chunk files keep their paths and vertices keep their
dtype. With `level=0` it sets the root `base_bin_shape` and each coarser level's
`bin_ratio`; on another level it
sets that level's `bin_shape` and `bin_ratio`. It raises:

| Error | When |
|-------|------|
| `StoreError` | The root `geometry_types` is not exactly `["point_cloud"]` |
| `MetadataError` | `target_bin_shape` does not divide `chunk_shape` on every axis |
| `OctreeError` | The level has [octree chunks](octree_chunks.md) |
| `StoreError` | The level has `object_index/` or `fragment_attributes/`, which name fragments; use `rechunk_spatial` |

### Grouping objects under a leading axis

```python
rechunk(store_path, spec, output=None, *, overwrite=False) -> dict
rechunk_by_attribute(store_path, attribute_name, *, output=None,
                     spatial_chunk_shape=None) -> dict
```

`rechunk` assigns each object of level 0 to a bin by a `RechunkSpec` and
writes chunk keys `(bin, x, y, z)`:

| `RechunkSpec.by` | Bins |
|------------------|------|
| `"group"` | One per group; ungrouped objects go to a last bin labelled `-1` |
| `"object_id"` | Id ranges cut at `bins`, labelled by their lower edge; one per object when `bins` is `None` |
| `"attribute:<name>"` | One per value when `categorical=True` or the per-object attribute has at most 10 distinct values; otherwise ranges cut at `bins`, or at quartiles when `bins` is `None` |
| `"spatial"` | No leading axis: `rechunk_spatial` to `spatial_chunk_shape` |

`spatial_chunk_shape` sets the output `chunk_shape`. `prefix_dim_name`
names the leading axis, recorded as both `chunk_dims[0]` and
`chunk_attribute_name` (by default the attribute's name for
`"attribute:<name>"`, otherwise `by`).
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

The output holds level 0's vertices, vertex attributes, object index,
object attributes, groups, group attributes and `links/0`. Object ids
are kept, so object attributes, groups and group attributes are copied
unchanged. Each vertex goes to its object's bin; fragments shared by
objects of one bin stay shared. Every `links/0` record is re-addressed
to its endpoints' new keys, whatever its `link_width`; a link between
objects in different bins joins two bins. Under
`links_convention="implicit_sequential"`, where the new grid cuts a
fragment, a link joins the two runs. The summary counts the records
written (`links`) and the source records naming no vertex, which are
dropped (`dropped_links`). `rechunk` raises before writing:

| Error | When |
|-------|------|
| `OctreeError` | Level 0 has octree chunks |
| `NotImplementedError` | The store has levels above 0 (remove them and run `build_pyramid` on the output) |
| `NotImplementedError` | Level 0 has a family it does not carry, such as `fragment_attributes/` or `link_attributes/` |
| `NotImplementedError` | Level 0 has links at a delta other than 0 (with no level to point at) |
| `NotImplementedError` | A fragment is shared by objects in different bins, or level 0 has vertices no object names; use `rechunk_spatial` |
| `ValueError` | `output` is the source |
| `FileExistsError` | `output` exists and `overwrite` is false |

Without `output`, the store is replaced in place: the result goes to
`<name>.rechunked`, the source is renamed `<name>.backup`, the new store
is moved into place and the backup is deleted. If either scratch name
already exists, `rechunk` raises `FileExistsError`.

### Validation

Run `zarr_vectors.validate.validate(path, level=5)` on the output. The
chunk and bin checks are in [Chunk shape](chunk_shape.md#validation) and
[Bin shape](bin_shape.md#validation).
