# Object attributes

## Terms

**Object attribute**
: A named array at `object_attributes/<name>` with one row per object of
  the level.

**Absent row**
: A row of an object attribute that holds the array's fill value, which
  marks the object as having no value for that attribute.

**Group**
: A named set of objects, stored as one row of the `groups` array.

**Group attribute**
: A named array at `group_attributes/<name>` with one row per group.

---

## Introduction

A level can carry data at four scopes: per vertex
(`vertex_attributes/<name>/`), per fragment (`fragment_attributes/<name>/`),
per link (`link_attributes/<name>/…`) and per object
(`object_attributes/<name>`). The per-chunk scopes are laid out in
[Chunk arrays](../layout/chunk_arrays.md) and link attributes in
[Links](links.md). This page specifies object attributes, groups and
group attributes: how rows map to objects, how an absent value is
stored, and what a coarser level keeps. Every attribute is optional and
independent of the others.

---

## Technical reference

### Array layout

Each object attribute is a plain Zarr array, readable without
zarr-vectors-py:

| Property | Value |
|----------|-------|
| Path | `<level>/object_attributes/<name>` |
| Shape | `[num_objects]`, or `[num_objects, K]` for `K` columns |
| Chunk shape | `[65536]`, or `[65536, K]` |
| `data_type` | The attribute's dtype |
| `fill_value` | The absent sentinel (see below) |
| Attributes | `zv_array: "object_attribute"`, `name`, `dtype`, `shape`, `fill_sentinel_meaning: "absent"` |

For example, a `float32` attribute of five objects has the array
attributes

```json
{
  "zv_array": "object_attribute",
  "name": "mean_fa",
  "dtype": "float32",
  "shape": [5],
  "fill_sentinel_meaning": "absent"
}
```

### Rows and object ids

Row `k` of an object attribute belongs to the object at row `k` of the
level's object index, whose id is row `k` of `object_index/object_ids`
(or `k` itself in a `vlen_manifests_v1` index; see
[Object manifest](object_manifest.md)). Rows follow ascending object id
when the index is written by a `write_*` function, so a store written
with object ids `3`, `7` and `20` stores their attribute values in rows
0, 1 and 2.

### Absent values

An absent row holds the array's `fill_value` in every column. The
default sentinel depends on the dtype:

| dtype kind | Sentinel |
|------------|----------|
| Floating point | `NaN` |
| Signed integer | The dtype's minimum (`-2147483648` for `int32`) |
| Unsigned integer | The dtype's maximum |

Other dtypes, such as `bool`, have no default; a writer passes
`fill_value=` explicitly. A chunk that was never written reads as the
fill value, so it is absent too. A row whose real value equals the
sentinel is indistinguishable from an absent one.

`read_object_attribute_present_mask(level_group, name)` returns a
`uint8` mask (`1` present, `0` absent), or `None` when every row is
present or the attribute does not exist.

### Writing and reading

The `write_*` functions take `object_attributes={name: array}` with one
row per object. On an existing level,
`write_object_attributes(level_group, name, data, *, present_mask=None,
fill_value=None, mode="replace", at=None)` writes one column:

- `present_mask` (one `0`/`1` per row) sets the rows marked `0` to the
  sentinel.
- `mode="replace"` writes `data` as the whole column.
- `mode="append"` adds rows at the end, or from row `at` when given:
  a column shorter than `at` is padded with absent rows, and one longer
  is cut back to `at`. Appends to one column must not run concurrently.
- `mode="place"` writes rows `[at, at + n)` of a column sized in advance
  by `reserve_object_rows`, so several writers can fill disjoint ranges.

`write_object_attribute_columns(level_group, {name: rows, …})` writes
several columns with the same modes.

`read_object_attributes(level_group, name)` returns the whole column,
sentinels included.

```python
import numpy as np

from zarr_vectors.building import (
    read_object_attribute_present_mask,
    read_object_attributes,
    write_object_attributes,
)
from zarr_vectors.core.store import get_resolution_level, open_store

lg = get_resolution_level(open_store("tracts.zarrvectors", mode="r+"), 0)
write_object_attributes(
    lg, "cls", np.array([1, 2, 3, 4, 5], dtype=np.int32),
    present_mask=np.array([1, 0, 1, 1, 0], dtype=np.uint8),
)
print(read_object_attributes(lg, "cls"))
# [          1 -2147483648           3           4 -2147483648]
print(read_object_attribute_present_mask(lg, "cls"))
# [1 0 1 1 0]
```

### Object attributes at coarser levels

A level built by the core `per_object` coarsener has a row for every
object id of its source level. A kept object's row is copied from the
source; a dropped object's row is absent. See
[Pyramid construction](../multiscale/pyramid_construction.md) and
[Object sparsity](../multiscale/sparsity.md).

### Groups and group attributes

`groups` is a `variable_length_bytes` array of shape `[num_groups]`;
cell `g` holds group `g`'s object ids as `int64`. Its attributes are
`zv_array: "groups"` and `num_groups`.

`group_attributes/<name>` is an array of shape `[num_groups]` with
attributes `zv_array: "groupings_attribute"`, `name`, `dtype` and
`shape`.

```python
write_polylines(
    "tracts.zarrvectors", polylines,
    chunk_shape=(200.0, 200.0, 200.0),
    groups={0: [0, 1, 2], 1: [3, 4]},
    group_attributes={"n_objects": np.array([3, 2], dtype=np.int32)},
)
```

`read_all_groupings(level_group)` returns every group's object ids,
`read_group_object_ids(level_group, g)` one group's, and
`read_groupings_attributes(level_group, name)` a group attribute. The
`read_*` functions select the objects of some groups with
`group_ids=[…]`.

### Validation

L1 reports which of `object_attributes/` and `groups` a level has. The
validators do not check object attribute shapes or values.
