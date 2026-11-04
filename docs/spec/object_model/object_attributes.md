# Attributes

## Terms

**Per-vertex attribute**
: A named array stored under `vertex_attributes/<name>/`. It is a
  **vlen-bytes chunk array** exactly like `vertices/`: shape = the level's
  chunk grid, one ragged cell per spatial chunk (file at `c/i/j/k`). Row `k`
  of a cell is the attribute value of row `k` of the same cell of
  `vertices/` — the two buffers are aligned 1:1 by row, *not* by fragment.
  (Fragments may name rows of the vertex buffer more than once; the attribute
  buffer is indexed the same way, via `vertex_fragments/`.)

**Per-fragment attribute**
: A named array stored under `fragment_attributes/<name>/`. Also a vlen-bytes
  chunk array, one cell per spatial chunk, but its rows are indexed by
  **fragment** rather than by vertex: row `f` of a cell belongs to fragment
  `f` of `vertex_fragments/<chunk>`. Opt-in; the common use is materialising
  the id of the object that owns each fragment (the `object_id` column)
  without decoding `object_index/manifests`.

**Per-object attribute**
: A named array stored at `object_attributes/<name>`. Unlike the two above it
  is **not** a chunk array: it is an ordinary dense Zarr array with one row per
  *manifest row* of `object_index/manifests` (see
  [Object manifest](object_manifest.md)). Absent objects are marked in-band by
  the array's `fill_value`.

**Group / group attribute**
: `groups` is a 1-D vlen-bytes array, one cell per group, holding the group's
  member object ids as `int64`. `group_attributes/<name>` is a dense array with
  one row per group. Both are optional.

**Attribute dtype**
: For the three per-chunk families the Zarr `data_type` is only the
  *container* (`variable_length_bytes`). The element type is the array
  attribute **`dtype`** (for example `"float32"`); the number of columns per
  row is the attribute **`row_shape`** (`[]` = scalar, `[K]` = `K` columns).

---

## Introduction

Zarr Vectors supports attributes at four granularities: per-vertex,
per-fragment, per-object and per-group. All are optional and independent — a
store may carry per-object `mean_fa` without any per-vertex values, or the
reverse.

The one thing to keep in mind is that **the layout depends on the scope**:

| Scope | Path | Container | Rows indexed by |
|-------|------|-----------|-----------------|
| vertex | `vertex_attributes/<name>` | chunk array (vlen-bytes, grid-shaped) | vertex row of the same cell |
| fragment | `fragment_attributes/<name>` | chunk array (vlen-bytes, grid-shaped) | fragment index of the same cell |
| link | `link_attributes/<name>/<delta>/<offsets>` | chunk array (vlen-bytes, grid-shaped) | link record of the same cell — see [Links](links.md) |
| object | `object_attributes/<name>` | dense Zarr array `(n_rows[, K])` | manifest row |
| group | `group_attributes/<name>` | dense Zarr array `(n_groups[, K])` | group id |

This page documents all scopes except link attributes, which live with their
link arrays on the [Links](links.md) page.

---

## Technical reference

### Per-vertex attributes (`vertex_attributes/<name>/`)

#### Per-vertex array schema

| Property | Value |
|----------|-------|
| Zarr `data_type` | `variable_length_bytes` |
| Shape | `chunk_grid_shape` — same grid as `vertices/` |
| Zarr chunk shape | `(1, …, 1)` — one inner chunk per cell |
| Fill value | `b""` (an empty cell has no chunk file) |
| Attributes | `zv_array="attribute"`, `name`, `dtype`, `row_shape`, optional `channel_names`, `chunk_grid_origin`, `nonempty_chunks`; optional dictionary-encoding keys (`encoding="dictionary"`, `categories`, `ordered`, `_FillValue`) |

A cell is a flat C-order buffer of `N × K` elements with no header, where `N`
is the number of rows in the matching `vertices/` cell. To decode it, take
`dtype` and `row_shape` from the array attributes and let `N` fall out of the
byte length:

```python
raw  = cell_bytes                                   # one cell of vertex_attributes/intensity
K    = attrs["row_shape"][0] if attrs["row_shape"] else 1
rows = np.frombuffer(raw, dtype=attrs["dtype"]).reshape(-1, K)   # (N, K)
assert len(rows) == n_vertices_in_this_cell          # 1:1 with vertices/
```

`row_shape` is the authoritative width. Stores written before it was stamped
fall back to `len(channel_names)`, then to `1`. The reference reader raises
when a caller-supplied `dtype` or width contradicts the stored one, because
decoding at the wrong element size does not fail by itself — it silently
reinterprets the bytes.

If a cell's element count is not a multiple of its vertex count the reader
raises: per-vertex attributes must align 1:1 with the vertex buffer.

Directory tree for a point cloud with two attributes:

```
scan.zarrvectors/0/
├── vertices/            # cells: (N, 3) float32
├── vertex_fragments/
└── vertex_attributes/
    ├── intensity/       # cells: (N,)   float32   row_shape = []
    │   ├── zarr.json
    │   └── c/0/0/0 …
    └── color/           # cells: (N, 3) uint8     row_shape = [3]
        ├── zarr.json
        └── c/0/0/0 …
```

There is no `N_max` padding and no `(*chunk_grid_shape, N_max)` shape: a cell
is exactly as long as its chunk's data.

#### Naming rules

The attribute name is a path segment (`vertex_attributes/<name>`). The library
does not validate it, but names that are valid Python identifiers
(alphanumeric and underscores, not starting with a digit) are the convention
and are always safe. No attribute names are reserved by the core library.

#### Common per-vertex attributes

| Attribute | Dtype | Description |
|-----------|-------|-------------|
| `intensity` | float32 | Absorption / fluorescence intensity |
| `fa` | float32 | Fractional anisotropy (per streamline vertex) |
| `md` | float32 | Mean diffusivity |
| `label` | int32 | Semantic segmentation label |
| `confidence` | float32 | Detection confidence score |
| `color` | uint8 `(N, 3)` | RGB colour per vertex |
| `normal` | float32 `(N, 3)` | Surface normal per vertex (mesh) |
| `uv` | float32 `(N, 2)` | UV texture coordinates (mesh) |

These are conventions of use, not names the library treats specially.

#### Categorical attributes

String (or other categorical) values can be stored as integer codes plus a
lookup table in the array's attributes (`encoding="dictionary"`,
`categories`, `ordered`, optional `_FillValue`). Readers decode the codes back
through `categories` automatically. See
[`zarr_vectors.encoding.categorical`](../../../zarr_vectors/encoding/categorical.py).

#### Writing per-vertex attributes

```python
import numpy as np
from zarr_vectors.types.points import write_points

n = 100_000
positions = np.random.default_rng(0).uniform(0, 1000, (n, 3)).astype(np.float32)
intensity = np.random.default_rng(1).uniform(0, 1, n).astype(np.float32)
color     = np.random.default_rng(2).integers(0, 256, (n, 3)).astype(np.uint8)

write_points(
    "scan.zarrvectors", positions,
    chunk_shape=(200., 200., 200.),
    vertex_attributes={
        "intensity": intensity,   # scalar float32
        "color":     color,       # 3 columns, uint8
    },
)
```

`write_points` also accepts the deprecated alias `attributes=`. For
polylines, the values are given per polyline:
`vertex_attributes={name: [array_for_polyline_0, …]}`.

#### Reading per-vertex attributes

```python
from zarr_vectors.types.points import read_points

result = read_points("scan.zarrvectors", attribute_names=["intensity", "color"])
intensity = result["vertex_attributes"]["intensity"]   # (M,)
color     = result["vertex_attributes"]["color"]       # (M, 3)

# A bbox query keeps attributes row-aligned with the returned positions
result = read_points("scan.zarrvectors", bbox=(lo, hi), attribute_names=["intensity"])
```

`attribute_names=None` (the default) reads **no** attributes — the parameter
names what to fetch; it does not mean "all".

### Per-fragment attributes (`fragment_attributes/<name>/`)

Same container as a per-vertex attribute (vlen-bytes, grid-shaped, one cell per
chunk, `zv_array="fragment_attribute"`, `dtype`, `row_shape`), but a cell holds
`F × K` elements, one row per fragment of `vertex_fragments/<chunk>`. The row
count is derived from the byte length and the row stride
(`dtype.itemsize × K`). The low-level `write_chunk_fragment_attributes` does
not check it against the fragment index, so a wrongly sized cell is only
detected on read; the high-level `write_polylines` does check and raises
`ArrayError`.

```python
# writer: {name: {combined_chunk_coords: ndarray of shape (F,) or (F, K)}}
# where F is the number of fragments this writer emits for that chunk
write_points(..., fragment_attributes={"quality": {(0, 0, 0): quality_for_chunk_000}})
```

Replace-only at the chunk level; not carried to coarser pyramid levels.

#### The `object_id` owner column

The reserved use is a column named `object_id` (`int64`) giving the owner of
each fragment, so "which object references this fragment" is one cell read
instead of an inversion of every manifest. Two sentinel values:

| Value | Meaning |
|-------|---------|
| `-1` | no object references the fragment |
| `-2` | more than one object references it (shared fragment) — fall back to the manifests |

It is materialised on demand by `zarr_vectors.core.arrays.write_fragment_owner_column(level_group)`
and read with `read_fragment_owners`. It is derived data and goes stale if the
manifests change.

### Per-object attributes (`object_attributes/<name>`)

#### Per-object array schema

```
path:        <N>/object_attributes/<name>        (a plain Zarr array, not a group)
shape:       (n_rows,)      scalar attribute
             (n_rows, K)    K-column attribute
chunks:      (65536,) or (65536, K)               # OBJECT_ATTRIBUTE_ROW_BUCKET rows
fill_value:  NaN (floats) | dtype minimum (signed ints) | dtype maximum (unsigned ints)
attributes:  zv_array="object_attribute", name, dtype, shape,
             fill_sentinel_meaning="absent"
```

`n_rows` equals `object_index/manifests.shape[0]` (`object_index` attribute
`num_objects`). Row `k` belongs to manifest row `k`. With the default dense id
space — ids `0 … n-1`, which every bulk writer produces — row `k` is object
id `k`. For stores whose ids are sparse or large, `object_index/object_ids`
maps row → id; use `object_ids_for_rows` / `object_rows_for_ids` rather than
assuming row == id. See [Object manifest](object_manifest.md#object-ids-and-rows).

The 65 536-row chunking is fixed so that appending more objects rewrites at
most one Zarr chunk. The default codec is Zarr v3's (`bytes` + `zstd`), unless
the write session asked for another.

#### Absent rows

There is no sibling `present_mask` array. A row is **absent** when every
channel equals the array's `fill_value`; the sentinel is chosen per dtype as
shown above (empty string for string dtypes). `bool` has no default sentinel
and needs an explicit `fill_value=`. Use `read_object_attribute_present_mask`
to recover a 0/1 mask — it returns `None` when every row is present.

This is the mechanism by which ID-preserving pyramid levels mark dropped
objects (see [Object manifest](object_manifest.md#object-ids-and-rows)).

#### Writing per-object attributes

```python
import numpy as np
from zarr_vectors.types.polylines import write_polylines

n_polylines = 500
polylines   = [...]   # list of (N_i, 3) arrays
lengths     = np.array([p.shape[0] for p in polylines], dtype=np.int32)
mean_fa     = np.random.default_rng(0).uniform(0.2, 0.8, n_polylines).astype(np.float32)

write_polylines(
    "tracts.zarrvectors", polylines,
    chunk_shape=(200., 200., 200.),
    object_attributes={
        "length":  lengths,    # int32 per polyline
        "mean_fa": mean_fa,    # float32 per polyline
    },
)
```

#### Reading per-object attributes

```python
from zarr_vectors.core.store import open_store, get_resolution_level
from zarr_vectors.core.arrays import (
    read_object_attributes, read_object_attribute_present_mask,
)

root  = open_store("tracts.zarrvectors", mode="r")
level = get_resolution_level(root, 0)

mean_fa = read_object_attributes(level, "mean_fa")                 # (n_rows,)
absent  = read_object_attribute_present_mask(level, "mean_fa")     # None, or uint8 (n_rows,)
```

`read_polylines` / `read_points` do not return per-object attributes; read
them as above.

#### Adding attributes to an existing store

Attributes can be added to an existing level without rewriting vertex data,
through the lazy writer (single-writer only):

```python
from zarr_vectors.lazy import open_zv

zv = open_zv("tracts.zarrvectors")
with zv[0].writer() as w:
    w.add_object_attribute_sync("cluster_id", cluster_ids)   # (n_objects,) int32
    w.add_attribute_sync("fa2", per_vertex_values)           # (n_vertices,), in stored vertex order
```

The per-vertex form splits the values by chunk using the level's existing
fragment layout and writes only `vertex_attributes/<name>`.

### Attributes across resolution levels

The core pyramid builder does **not** aggregate per-vertex attributes.
`coarsen_level` / `build_pyramid` write the metavertices' positions, the
manifests, and the **per-object** attributes only; a coarser level therefore
has no `vertex_attributes/` unless something writes them afterwards. (The
`AGG_*` aggregation-method constants in `zarr_vectors.constants` are not
consumed by the core coarsener, and `build_pyramid` has no per-attribute
aggregation argument.)

Per-object attributes are carried across with the *parent's* row space: every
source object keeps its row, surviving objects keep their values, and dropped
objects get the `fill_value` sentinel (`present_mask` semantics above). No
`level0_object_id` array is written, because ids are never renumbered.

### Groupings and group attributes

Groupings assign objects to named sets (tract bundles, cell types). They are
two optional arrays:

```
groups                       1-D vlen-bytes array, shape (n_groups,)
                             cell g = int64 member object ids, as raw bytes
                             attrs: zv_array="groups", num_groups, [group_ranges]
group_attributes/<name>      dense array (n_groups,) or (n_groups, K)
                             attrs: zv_array="groupings_attribute", name, dtype, shape
```

Group ids are contiguous from `0`; a missing id becomes an empty group. A
group whose members form a contiguous run may be passed as a `range`; it is
then stored implicitly as `[start, stop)` in the `group_ranges` attribute
(with an empty cell), so a billion-member group costs O(1).

```python
write_polylines(
    "tracts.zarrvectors", polylines,
    chunk_shape=(200., 200., 200.),
    groups={0: cst_ids, 1: af_ids},                  # {group_id: [object ids]}
    group_attributes={"n_objects": np.array([len(cst_ids), len(af_ids)])},
)
```

Readers: `read_group_object_ids(level, gid)`, `read_all_groupings(level)`,
`read_groupings_attributes(level, name)`. There is no built-in group-name
field; carry names in your own `group_attributes` array or outside the store.

### Validation

The validators do not currently check attribute shapes against `vertices/`
(L3 checks vertex counts, fragment bounds, manifests and links). L1 records
which of `vertex_attributes`, `fragment_attributes`, `object_index`,
`object_attributes` and `groups` exist at each level.

A writer must keep, and a reader may rely on:

- **per-vertex:** each cell has exactly as many rows as the matching `vertices/`
  cell;
- **per-fragment:** each cell has one row per fragment of the matching
  `vertex_fragments/` cell;
- **per-object:** `shape[0] == object_index.num_objects`, and absence expressed
  only through `fill_value`;
- **per-group:** `shape[0] == groups.num_groups`.
