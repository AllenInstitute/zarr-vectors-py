# Object attributes

## Terms

**Object attribute**
: A named array at `object_attributes/<name>` with one row per object of
  the level.

**Absent row**
: A row holding the array's fill value: the object has no value.

**Group**
: A named set of objects, stored as one row of the `groups` array.

---

## Introduction

Object attributes hold one value per object; groups and group
attributes hold sets of objects and one value per set. All are optional.
Per-vertex and per-fragment attributes are in
[Chunk arrays](../layout/chunk_arrays.md); link attributes are in
[Links](links.md).

---

## Technical reference

### Array layout

Each object attribute is a plain Zarr array:

| Property | Value |
|----------|-------|
| Path | `<level>/object_attributes/<name>` |
| Shape | `[num_objects]`, or `[num_objects, K]` for `K` columns |
| Chunk shape | `[65536]`, or `[65536, K]` |
| `fill_value` | The absent sentinel (see below) |
| Attributes | `zv_array: "object_attribute"`, `name`, `dtype`, `shape`, `fill_sentinel_meaning: "absent"` |

A `float32` attribute of five objects:

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

Row `k` belongs to row `k` of the level's
[object index](object_manifest.md): id `object_ids[k]`, or `k` in a
`vlen_manifests_v1` index. The `write_*` functions write rows in
ascending id order, so ids `3`, `7` and `20` use rows 0, 1 and 2.

### Absent values

An absent row holds `fill_value` in every column. Default sentinels:

| dtype kind | Sentinel |
|------------|----------|
| Floating point | `NaN` |
| Signed integer | The dtype's minimum (`-2147483648` for `int32`) |
| Unsigned integer | The dtype's maximum |

Other dtypes, such as `bool`, need an explicit `fill_value=`. Unwritten
chunks read as absent. A real value equal to the sentinel reads as
absent.

### Writing and reading

The `write_*` functions take `object_attributes={name: array}`, one row
per object. On an existing level,
`write_object_attributes(level_group, name, data, *, present_mask=None,
fill_value=None, mode="replace", at=None)` writes one column;
`write_object_attribute_columns(level_group, {name: rows, …})` writes
several. `present_mask` sets rows marked `0` to the sentinel.

| `mode` | Writes |
|--------|--------|
| `"replace"` | `data` as the whole column |
| `"append"` | rows at the end, or from row `at` (shorter columns are padded with absent rows, longer ones cut to `at`); not safe to run concurrently on one column |
| `"place"` | rows `[at, at + n)` of a column sized by `reserve_object_rows`; writers may fill disjoint ranges in parallel |

`read_object_attributes` returns the column with sentinels.
`read_object_attribute_present_mask` returns a `uint8` mask (`1`
present), or `None` when every row is present or the attribute does not
exist.

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
object id of its source level: kept objects are copied, dropped ones are
absent (see [Object sparsity](../multiscale/sparsity.md)).

### Groups and group attributes

| Array | Shape | Contents | Attributes |
|-------|-------|----------|------------|
| `groups` | `[num_groups]`, `variable_length_bytes` | cell `g`: group `g`'s object ids as `int64` | `zv_array: "groups"`, `num_groups` |
| `group_attributes/<name>` | `[num_groups]` | one value per group | `zv_array: "groupings_attribute"`, `name`, `dtype`, `shape` |

Writers take `groups={0: [0, 1, 2], 1: [3, 4]}` and
`group_attributes={name: array}`. `read_all_groupings`,
`read_group_object_ids(level_group, g)` and
`read_groupings_attributes(level_group, name)` read them. The `read_*`
functions take `group_ids=[…]` to select the objects of some groups.

### Validation

L1 reports which of `object_attributes/` and `groups` a level has. The
validators do not check object attribute shapes or values.
