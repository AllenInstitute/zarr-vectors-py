# Links

## Terms

**Link**
: A record of `link_width` (`L`) endpoints: `L = 2` is an edge, `L = 3`
  a triangle face, `L = 1` a single-vertex reference. Endpoints may sit
  in different chunks and, when `delta != 0`, at different levels.

**Source chunk**
: The chunk of the endpoint a record is filed under. It is the array
  cell that holds the record.

**Relative offset** (`o_k`)
: The chunk-grid displacement from the source chunk to endpoint `k`, a
  signed `sid_ndim`-tuple. `o_0 = 0` and is not stored, so a record
  carries `L - 1` offsets.

**Intra-chunk link**
: A link whose offsets are all zero. It lives in the all-zero
  `<offsets>` array of the same family.

**Level delta** (`<delta>`)
: A signed path segment: how many levels a record spans. `0` means all
  endpoints are at the owning level; `+N` means endpoints `1..L-1` are
  `N` levels coarser; `-N` means `N` levels finer. Written `"0"`,
  `"+1"`, `"-1"`, `"+2"`, ….

**Link attribute**
: Per-record data at `link_attributes/<name>/<delta>/<offsets>/`,
  mirroring the link array cell for cell.

---

## Introduction

Each level stores its connectivity in one family per level delta:

```
links/<delta>/                             GROUP  family-wide policy
links/<delta>/<offsets>/                   ARRAY  rank-D vlen; cell = source chunk
link_attributes/<name>/<delta>/            GROUP
link_attributes/<name>/<delta>/<offsets>/  ARRAY  mirrors the link array
```

The `<offsets>` segment says where the other endpoints sit. Each array
is a plain vlen array over the level's chunk grid, one cell per source
chunk. Each stored `vi_k` is a vertex index local to chunk `src + o_k`.
An intra-chunk link and a cross-chunk link differ only in their
`<offsets>` directory.

Worked example:
[`examples/07_multiscale_links.ipynb`](../../../examples/07_multiscale_links.ipynb).

---

## Technical reference

### Offsets grammar

The segment holds `link_width - 1` offsets, joined by `_`. Each offset
is `sid_ndim` signed components joined by `.`, written like `<delta>`
(`0`, `+1`, `-1`).

| Segment | `L` | Meaning |
|---------|-----|---------|
| `0.0.0` | 2 | intra-chunk edge |
| `0.0.+1` | 2 | edge to the neighbour along `+z` |
| `0.0.-1` | 2 | edge to the neighbour along `-z` |
| `0.0.+1_0.+1.0` | 3 | triangle spanning the source, `+z` and `+y` |
| `0.0.0_0.0.0` | 3 | intra-chunk triangle |
| `self` | 1 | no other endpoint; an empty segment would make `links/<delta>/` ambiguous |

Build and parse segments with the helpers in
[`zarr_vectors/core/paths.py`](../../../zarr_vectors/core/paths.py):
`format_delta` / `parse_delta`, `format_offsets` / `parse_offsets`,
`intra_offsets`, `is_intra`, `links_group_path`, `links_path`,
`link_attributes_group_path`, `link_attributes_path`.

`parse_offsets` rejects a segment unless:

1. it has exactly `link_width - 1` offsets, or is `self`;
2. every offset has exactly `sid_ndim` components;
3. `self` appears if and only if `link_width == 1`.

### Placement

`_cell_placements` in
[`zarr_vectors/spatial/boundary.py`](../../../zarr_vectors/spatial/boundary.py)
gives the permutations `sigma` a record is filed under
(`partition_records_by_offset` applies it to records,
`partition_link_arrays` to arrays). For each `sigma`:

| Item | Value |
|------|-------|
| Cell | chunk of endpoint `sigma[0]` (the source) |
| Offsets | `chunk(sigma[k]) - anchor(source)` for `k > 0` ([anchor](#cross-level-placement-the-anchor)) |
| Stored vertex indices | `[vi(sigma[0]), …, vi(sigma[L-1])]`, each local to its own chunk |
| `perm_idx` | `lehmer(sigma)`, when the array has that column |

To decode, endpoint `k` is in chunk `anchor(src) + o_k` with the `k`-th
stored `vi`.

### `perm_idx`

A record is `L` ints, or `1 + L` (`[perm_idx, vi_0 … vi_{L-1}]`) when
the array has a permutation column. `perm_idx` is a Lehmer code in
`[0, L!)` for `sigma`. Readers use it to restore input endpoint order
(face winding, edge direction).

[`links_has_perm`](../../../zarr_vectors/core/arrays.py) decides it, and
each array records the result as `has_perm`:

```python
def links_has_perm(offsets, *, delta, directed, store):
    if is_intra(offsets):    return False   # identity placement
    if delta != 0:           return False   # source is always endpoint 0
    if store == "duplicate": return True    # each copy leads with a different endpoint
    return not directed                     # undirected sorts; directed does not
```

Intra-chunk records keep input order, since polyline traversal order is
data.

### Cell encoding: two branches, one condition

[`write_chunk_links`](../../../zarr_vectors/core/arrays.py) picks the
encoding on `delta == 0 and is_intra(offsets)`:

| Condition | Encoding | Sidecar |
|-----------|----------|---------|
| `delta == 0` and offsets all zero | flat rows (`encode_ragged_ints`) | `link_fragments/<chunk>` |
| otherwise | self-describing ragged blob (`encode_ragged_blob`) | none |

[`link_fragments/<chunk>`](../layout/fragment_index_arrays.md) holds the
flat branch's per-group row ranges. Link groups need not match the
chunk's vertex fragments. The sidecar is keyed by chunk alone, so only
this branch writes it.

### `directed` and `store`

Both are family-wide, set on the `<delta>` group.
[`write_links`](../../../zarr_vectors/core/arrays.py) raises `ArrayError`
if a write would change either while other offsets arrays exist under
the group.

| Field | Value | Meaning |
|-------|-------|---------|
| `directed` | `false` (default) | Endpoints are sorted by `(chunk_coords, vi)`, so the stored offset is lexicographically positive and each record is stored once. |
| | `true` | Input order is data (for example fine→coarse pyramid links). No sort: `A→B` and `B→A` file under opposite offsets (`0.0.+1`, `0.0.-1`) in different cells. |
| `store` | `"canonical"` (default) | Each record is in one cell. Finding all links touching chunk `C` means scanning every offsets array for cells that reach `C`. |
| | `"duplicate"` | One copy per distinct incident chunk, each led by that chunk. All links touching `C` are in `C`'s cell across the offsets arrays, at up to `K`× the storage (`K` = chunks touched). `read_links` returns every copy. |

| `directed` | `store` | On disk |
|------------|---------|---------|
| `false` | `canonical` | one cell, sorted, `perm_idx` present |
| `false` | `duplicate` | one copy per incident chunk, `perm_idx` present |
| `true` | `canonical` | one cell, input order, no `perm_idx` |
| `true` | `duplicate` | one copy per incident chunk, `perm_idx` present |

### Endpoint order of skeleton links

| Writer | Family policy | Intra-chunk rows | Cross-chunk records |
|--------|---------------|------------------|---------------------|
| `write_graph(..., kind="skeleton")` | `directed: false`, `canonical` | `[child, parent]` | sorted with `perm_idx`; `read_links` returns `[child, parent]` |
| `write_skeleton_chunk`, `write_skeleton_cross_chunk_links` | `directed: true`, `canonical` | `[child, parent]` | `[parent, child]`, no `perm_idx` |

### Cross-level placement: the anchor

For `delta != 0` the source is always input endpoint 0, at the owning
level. Cross-level records are never sorted.

Levels may have different chunk grids: `LevelMetadata.chunk_shape` is an
optional per-level override, which
[`coarsen`](../../../zarr_vectors/multiresolution/coarsen.py) sets to
`source × chunk_scale_factor` (default factor 1). Chunk coordinates from
two such grids cannot be subtracted directly.

Example: root `chunk_shape = 100`, level-1 `chunk_shape = 200`. Both
rows have the target in the level-1 chunk after the one holding the
source:

| Source (level 0) | Target (level 1) | Naive `c_trg - c_src` | Anchored |
|---|---|---|---|
| chunk 3 = `[300, 400)` | chunk 2 = `[400, 600)` | `-1` | `+1` |
| chunk 1 = `[100, 200)` | chunk 1 = `[200, 400)` | `0` | `+1` |

[`anchor_chunk`](../../../zarr_vectors/spatial/boundary.py) maps the
source into the target level's grid first:

```
anchor = floor(c_src * r_src / r_trg)
o      = c_trg - anchor              # decode: c_trg = anchor + o
```

`r_L` is level `L`'s `chunk_shape` as an integer multiple of the root
`chunk_shape`
([`chunk_scale_from_root`](../../../zarr_vectors/core/metadata.py)).
When `r_src == r_trg`, `anchor == c_src`. Floor division rounds toward
`-inf`, as negative chunk coordinates need.

All endpoints of a store share `sid_ndim`; a mismatch raises
`ArrayError`. One `<delta>` covers all non-source endpoints, so a record
spanning three levels (such as a triangle at `(N, N, N+1)`) cannot be
stored.

### Enumeration order

Records are read in order of offsets segment (ASCII sort), then cell,
then write order within a cell. `+` and `-` sort before `0`, so the
intra segment comes last:

```python
sorted(['0.0.0', '0.0.+1', '0.0.-1', '+1.0.0', '-1.0.0'])
# ['+1.0.0', '-1.0.0', '0.0.+1', '0.0.-1', '0.0.0']
```

[`read_links`](../../../zarr_vectors/core/arrays.py) and
`read_link_attributes` use the same order; it is what aligns attribute
rows with link records. `write_link_attributes` without a
`LinkPartition` expects `attr_data` in this order. Pass the
`LinkPartition` returned by `write_links` to give data in input order.
Under `store="duplicate"` the partition is required.

### Metadata

`links/<delta>/` group (family policy):

```javascript
{
  "zv_array":    "links_family",
  "level_delta": 0,
  "link_width":  2,
  "directed":    false,
  "store":       "canonical",
  "sid_ndim":    3,
  "num_links":            12,   // logical record count
  "num_physical_records": 12    // on-disk rows; > num_links under "duplicate"
}
```

`links/<delta>/<offsets>/` array:

```javascript
{
  "nonempty_chunks": ["0.0.0", "1.0.0"],
  "zv_array":    "links",
  "dtype":       "int64",
  "offsets":     [[0, 0, 1]],   // parsed path segment
  "has_perm":    true,          // rows are 1 + L wide
  "link_width":  2,
  "level_delta": 0
}
```

`read_links` uses the stored `has_perm`, and computes it from the family
policy only when it is missing.

`link_attributes/<name>/<delta>/` group:

```javascript
{ "zv_array": "link_attribute_family", "name": "weight", "level_delta": 0, "num_links": 8 }
```

`link_attributes/<name>/<delta>/<offsets>/` array:

```javascript
{
  "nonempty_chunks": ["0.0.0", "1.0.0"],
  "zv_array":    "link_attribute",
  "name":        "weight",
  "dtype":       "float32",
  "offsets":     [[1, 0, 0]],
  "level_delta": 0,
  "row_shape":   []             // per-record shape; [] for a scalar
}
```

### Parallel writes and counts

`write_links` keeps `num_links` and `num_physical_records` up to date.
The per-cell writer `write_link_cells` does not; one
[`finalize_links`](../../../zarr_vectors/core/arrays.py) call sets them
afterwards and rebuilds `nonempty_chunks`. Under `duplicate` it counts
logical records by deduplicating decoded records.

In an unsharded array each cell is its own object. Workers that own
disjoint sets of source chunks can write in parallel; two workers must
not write the same cell. Workers pass `record_presence=False`, because
`nonempty_chunks` is shared by the whole array. `create_links_family`
sets the family policy without creating an offsets array.

### Reading

```python
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.building import (
    read_links, read_link_attributes, read_links_for_tuple,
    read_chunk_links, list_link_deltas, list_link_offsets,
)

root = open_store("graph.zarrvectors")
lg   = get_resolution_level(root, 0)

records = read_links(lg, delta=0)                       # whole family
weights = read_link_attributes(lg, "weight", delta=0)   # aligned to records
print(list_link_deltas(lg))            # e.g. [0, 1]
print(list_link_offsets(lg, 0))        # intra sorts last
```

`read_links` returns `[]` when the family is absent or has no policy.
`read_chunk_links` reads one cell. `read_links_for_tuple` reads only the
records spanning an exact `L`-chunk tuple. For an undirected `canonical`
family at `delta == 0` the tuple may be in any order; for any other
family it must lead with input endpoint 0's chunk.

### Validation

From [`zarr_vectors/validate/consistency.py`](../../../zarr_vectors/validate/consistency.py)
(see [L3 consistency](../validation/l3_consistency.md)). L1 lists the
`<delta>` segments under `links/` and checks nothing else.

L3, for every `<delta>` family with a `sid_ndim`:

| Check | Applies to |
|-------|------------|
| Every `<offsets>` segment parses against `sid_ndim` and `link_width` | all families |
| No offset is lexicographically negative; offsets are non-decreasing across the segment | undirected, `canonical`, `delta == 0` only |
| `num_physical_records`, if present, equals the rows `read_links` returns | all families |
| Every endpoint chunk exists in the level's chunk grid | `delta == 0` |
| The source-side endpoint chunk exists in the level's chunk grid | `delta != 0` |
