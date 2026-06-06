# Links and cross-chunk links

## Terms

**Intra-chunk link**
: An edge between two vertices that live in the **same** spatial chunk
  at the **same** resolution level. Stored as a pair of local vertex
  indices in `links/<delta>/<chunk_key>` with `delta=0`.

**Cross-chunk link**
: A record whose endpoints span two or more spatial chunks (possibly at
  different resolution levels). Stored as a fixed-size `(ci, vi)` row
  inside a leaf whose path lists the **sorted unique chunks** the
  record touches:
  the `cross_chunk_links/<delta>/kK` sharded vlen-bytes array.

**Level delta** (`<delta>`)
: A signed integer path segment that says how many pyramid levels the
  edges span. `0` = both endpoints at the owning level (the only kind
  written pre-0.4); `+N` = endpoint B is `N` levels coarser; `-N` =
  endpoint B is `N` levels finer. Filesystem-safe literal segments:
  `"0"`, `"+1"`, `"-1"`, `"+2"`, …

**`K`** — the number of **distinct** chunks a cross-chunk record
touches.  `1 ≤ K ≤ L` where `L = link_width`.  Every leaf is filed at
depth `K` under `<delta>`.

**Link attribute**
: Per-edge scalar or vector data parallel to a `links/<delta>/` array.
  Lives at `link_attributes/<name>/<delta>/<chunk_key>` for intra-chunk
  edges and at
  the `cross_chunk_link_attributes/<name>/<delta>/kK` sharded array
  for cross-chunk edges.

---

## Introduction

The 0.8 link layout is a family of four array trees, each parameterised
by a level delta and (for cross-chunk records) **partitioned by K**,
the number of distinct chunks the records touch:

```
/N/links/<delta>/<chunk_key>
/N/cross_chunk_links/<delta>/kK/                # sharded (sid_ndim*K)-D vlen-bytes zarr Array
/N/link_attributes/<name>/<delta>/<chunk_key>
/N/cross_chunk_link_attributes/<name>/<delta>/kK/  # parallel sharded vlen-bytes
```

When an edge's two endpoints share a chunk_key (after re-evaluation
against the target level's chunk grid), the edge goes into the
per-chunk `links/<delta>/<chunk_key>` array. Otherwise it goes into a
**cell of the `kK` array** keyed by the sorted unique chunks the record
touches; each cell holds a ragged vlen-bytes payload encoding all
records for that chunk set. Zarr v3 sharding packs many adjacent cells
into one outer shard file so the file count scales with the data,
**not** with the chunk-grid size.

This addresses two scaling concerns at once:

- **Many writers without conflict.** Two writers touching cells in
  different shards write to different files. A K=2 shard typically
  holds `4^(2*sid_ndim)` cells (e.g. 4096 for 3-D), so adjacent edits
  often share a shard but writers working in different spatial regions
  don't.
- **Variable-length per cell.** vlen-bytes lets each cell carry as many
  or as few records as it actually has — empty cells consume zero
  bytes inside their shard.

The level delta is encoded in the path — readers never need to inspect
the edge to know which level its target side lives at.

This page documents:

- the on-disk encoding of the four arrays at every `<delta>`,
- when each kind is generated and how the chunk-partitioning is decided,
- the parallel attribute leaves,
- canonicalization and validation rules,
- the path helpers and listing helpers callers should use,
- reader access patterns,
- and the 0.7 → 0.8 migration story.

For a worked end-to-end example, see
[`examples/07_multiscale_links.ipynb`](../../../examples/07_multiscale_links.ipynb).

---

## Technical reference

### Level-delta convention

| Segment | Meaning |
|---------|---------|
| `0`     | intra-level edges (the only kind written pre-0.4) |
| `+N`    | edges from this level to `this_level + N` (coarser) |
| `-N`    | edges from this level to `this_level - N` (finer) |

Compose paths with the helpers in
[`zarr_vectors/core/paths.py`](../../../zarr_vectors/core/paths.py) —
never hand-roll the `<delta>` segment:

```python
from zarr_vectors.core.paths import (
    format_delta,           # 0 -> "0";  1 -> "+1";  -2 -> "-2"
    parse_delta,            # inverse
    links_path,                          # links/<delta>
    cross_chunk_links_path,              # cross_chunk_links/<delta>
    link_attributes_path,                # link_attributes/<name>/<delta>
    cross_chunk_link_attributes_path,    # cross_chunk_link_attributes/<name>/<delta>
)
```

To enumerate which deltas exist under a level group, use:

```python
from zarr_vectors.core.arrays import (
    list_link_deltas,                    # [0, +1, -1, ...]
    list_cross_link_deltas,
    list_link_attribute_deltas,
    list_cross_chunk_link_attribute_deltas,
)
```

### `links/<delta>/<chunk_key>` — per-chunk array

Each chunk file is a contiguous int64 byte blob holding one or more
`(M_k, link_width)` row groups. `link_width` is declared on the
array's `.zattrs`:

| Geometry | `link_width` | Row meaning |
|----------|--------------|-------------|
| Graph, polyline, streamline, skeleton (branches) | 2 | `(src_local, dst_local)` |
| Triangle mesh | 3 | `(v0_local, v1_local, v2_local)` |
| Quad mesh | 4 | `(v0, v1, v2, v3)` |

**`.zattrs` schema** (see
[`zarr_vectors/core/arrays.py:create_links_array`](../../../zarr_vectors/core/arrays.py)):

```jsonc
{
  "zv_array":   "links",
  "dtype":      "int64",
  "link_width": 2,
  "level_delta": 0     // signed integer; 0 for intra-level
}
```

**Endpoint convention for non-zero deltas:** for a row in
`links/+N/<chunk_key>`, column 0 is a local vertex index in the source
chunk at the **owning level**, and column 1 is a local vertex index in
the **same chunk key** at level `owning_level + N`. The reader doesn't
need any cross-chunk-coords information — both sides share `<chunk_key>`.

**Self-describing blob.** Each `links/<delta>/<chunk_key>` file is a
self-describing ragged blob: an int64 header with `K` followed by the
`K` per-group byte offsets, then the concatenated link bytes. Readers
recover the per-vertex-group partition without consulting any sibling
table.

### `cross_chunk_links/<delta>/kK/` — sharded vlen-bytes Array

Cross-chunk records are partitioned across **K-separated sharded
vlen-bytes zarr Arrays**.  Each `kK` array's logical shape is
`(Cx, Cy, Cz)^K` (where `(Cx, Cy, Cz)` is the per-axis chunk-grid
extent and the same coords are repeated K times) and each cell holds a
ragged byte blob: the payload for one sorted-unique-chunks tuple.

**Cell index encoding.**  For a record whose sorted-unique chunks are
`(chunk_sorted_0, …, chunk_sorted_{K-1})`, the cell coord into the
`kK` array is the flat concatenation:

```
cell_coord = (chunk_sorted_0 - origin) ⧺ … ⧺ (chunk_sorted_{K-1} - origin)
```

where `origin` is the per-axis `chunk_origin` from the array's
`.zattrs` (defaults to all-zero; stores with negative chunk coords use
a negative `origin` so cell coords stay non-negative).

The chunk segments are in lex order: `chunk_sorted_0 < … <
chunk_sorted_{K-1}`, element-wise integer-tuple comparison.

**Per-cell payload.**  Each populated cell is a single byte blob:

```
cell = [ rec_0, rec_1, ..., rec_{N-1} ]            # N records, back-to-back
each record =
  [ ci_0, ci_1, ..., ci_{L-1},        # L * uint8   (chunk-index per endpoint)
    vi_0, vi_1, ..., vi_{L-1} ]       # L * int64   (vertex index, little-endian)
  = 9 * L bytes per record
```

- `L = link_width`, declared on the `cross_chunk_links/<delta>/` group
  (uniform across all K-buckets).
- `ci_i ∈ [0, K-1]` picks one of the K chunks listed in the cell coord;
  that's the chunk endpoint `i` lives in.
- `vi_i` is endpoint `i`'s local vertex index inside its chunk's
  vertex array.
- The two blocks are concatenated: all L chunk-indices first
  (contiguous uint8), then all L vertex indices (contiguous int64).
  Bulk reads see cache-friendly stride.

**Per-cell record count** is derivable from the cell's byte length:
`num_records = len(cell_bytes) / (9 * link_width)`.  No separate
`num_links` counter is stored anywhere (key change from 0.7).

**Sharding.**  Each `kK` array uses zarr v3's
`sharding_indexed` codec with **outer shards of size `4^(sid_ndim*K)`
cells** (default; tunable via the `shard_shape` field in the array's
`.zattrs`).  Inner chunks are `(1,)^(sid_ndim*K)` — one cell per
inner chunk — so a single-cell read fetches exactly one shard.

**Lazy per-K allocation.**  Only `kK` arrays for K values that have
records exist on disk.  A graph store (`link_width=2`) typically has
`k1` + `k2`; a mesh store may grow `k3` if any triangle face spans
three distinct chunks.

**Per-K array `.zattrs` schema:**

```jsonc
{
  "zv_array":     "cross_chunk_links_k",
  "level_delta":  0,
  "K":            2,
  "sid_ndim":     3,
  "link_width":   2,
  "shard_shape":  [4, 4, 4, 4, 4, 4],
  "chunk_origin": [0, 0, 0]
}
```

### Endpoint level convention

For cross-chunk records, endpoint roles are still determined by their
position in the record:

- Endpoint `0` (the row identified by `ci_0`, `vi_0`) lives at the
  **owning** resolution level — the level under whose
  `cross_chunk_links/<delta>/` it is written.
- Endpoints `1..L-1` live at `owning_level + level_delta`.

For `level_delta = 0` both endpoints are at the same level. For
`level_delta = +1` endpoint 0 is the fine vertex and endpoint 1 is the
coarse metanode; etc.

`link_width=2` (the default) encodes a classic cross-chunk edge;
`link_width=3` encodes a triangle face spanning chunks (used by mesh
writers); `link_width=4` a quad face; `link_width=1` a single
parent→child reference for pyramid metanode drill-down.

### Parent group `.zattrs` schema

Stored on the `cross_chunk_links/<delta>/` group itself (above all the
`kK` sub-arrays):

```jsonc
{
  "zv_array":    "cross_chunk_links",
  "sid_ndim":    3,
  "level_delta": 1,
  "link_width":  2,
  "layout":      "sharded_v1"
}
```

`layout = "sharded_v1"` is the only legal value in 0.8 and is the
on-disk discriminator that signals "this group contains `kK`
sub-arrays in the sharded vlen-bytes layout described above" —
readers that see any other value (including a legacy 0.7-era store
that wrote `cross_chunk_links/<delta>/data` directly) fail with a
clear "run the migration helper" error.  See
[Migration from 0.7 → 0.8](../migration/0.7_to_0.8.md).

**Sid-ndim assumption.** Source and target levels share `sid_ndim`
(uniform per store). The writer asserts every cell coord's chunk-tuple
arity equals `sid_ndim`; mismatched callers fail loudly with an
`ArrayError`. Chunk *spacing* may differ between levels (coarser chunks
are larger in physical units), but the chunk-coord arity does not.

### Worked examples

#### L=2, K=2, delta=0 — graph edge between two chunks

Edge between `(0,0,0):5` and `(1,0,0):2`. Sorted chunks
`(0,0,0) < (1,0,0)`, so the record lands in the **k2** array:

```
array:        cross_chunk_links/0/k2  (6-D shape (Cx,Cy,Cz, Cx,Cy,Cz))
cell coord:   (0, 0, 0,   1, 0, 0)
cell payload: ci = [0, 1]         ← endpoint 0 at first sorted chunk, endpoint 1 at second
              vi = [5, 2]
              total = 18 bytes
```

Today's 0.7 single-blob encoding spends 64 bytes per record for the
same case (sid_ndim=3, link_width=2, both endpoint chunk-coords baked
into the record). The new layout drops it to 18 plus per-shard
overhead, and many records share a shard file.

#### L=2, K=2, delta=+1 — cross-level edge

Fine vertex at `(2,3,1):7` (owning level) parents to coarse metanode
at `(1,1,0):3` (level + 1). Sorted: `(1,1,0) < (2,3,1)`.

```
array:        cross_chunk_links/+1/k2
cell coord:   (1, 1, 0,   2, 3, 1)
cell payload: ci = [1, 0]   ← endpoint 0 (fine) is at sorted-chunk index 1
              vi = [7, 3]
              total = 18 bytes
```

The `ci` permutation encodes which side is owning vs target — the cell
coord stays canonical for the lookup "records between (2,3,1) and
(1,1,0)?".

#### L=2, K=1, delta=0 — same-chunk bridge

Both endpoints in chunk `(0,0,0)`, local indices 2 and 5:

```
array:        cross_chunk_links/0/k1  (3-D shape (Cx,Cy,Cz))
cell coord:   (0, 0, 0)
cell payload: ci = [0, 0]
              vi = [2, 5]
              total = 18 bytes
```

(K=1 records are also storable in `links/0/<chunk>` — see
[Validation rules](#validation-rules) for when each is preferred.)

#### L=3, K=2, delta=0 — triangle face, two distinct chunks

Triangle V0→V1→V2 in winding order:
- V0 at `(0,0,0):5`, V1 at `(1,0,0):3`, V2 at `(0,0,0):7`

```
array:        cross_chunk_links/0/k2
cell coord:   (0, 0, 0,   1, 0, 0)
cell payload: ci = [0, 1, 0]    ← V0→sorted_0, V1→sorted_1, V2→sorted_0
              vi = [5, 3, 7]
              total = 27 bytes
```

A second triangle with vertices `(1,0,0):4`, `(0,0,0):8`, `(0,0,0):9`
lands in the **same cell** under a different `ci` permutation:

```
same cell coord
appended payload: ci = [1, 0, 0]
                  vi = [4, 8, 9]
total cell bytes: 54 (two 27-byte records concatenated)
```

Triangles with the same chunk pair but different winding orientations
share a single cell — there is **no** permutation fan-out in the
directory tree.

#### L=3, K=3, delta=0 — triangle face spanning three chunks

V0 at `(0,0,0):2`, V1 at `(1,0,0):4`, V2 at `(0,1,0):5`. Sorted:
`(0,0,0) < (0,1,0) < (1,0,0)`.

```
array:        cross_chunk_links/0/k3  (9-D shape (Cx,Cy,Cz)^3)
cell coord:   (0, 0, 0,   0, 1, 0,   1, 0, 0)
cell payload: ci = [0, 2, 1]   ← V0→sorted_0, V1→sorted_2, V2→sorted_1
              vi = [2, 4, 5]
              total = 27 bytes
```

#### L=4, K=2, delta=0 — quad face spanning two chunks

Quad with vertices `(0,0,0):2`, `(0,0,0):3`, `(1,0,0):8`, `(1,0,0):7`:

```
array:        cross_chunk_links/0/k2
cell coord:   (0, 0, 0,   1, 0, 0)
cell payload: ci = [0, 0, 1, 1]
              vi = [2, 3, 8, 7]
              total = 36 bytes
```

### Canonicalization

Two normalization rules — both about picking a single representative
when an alternative `ci` permutation would write the same logical
record under a different orientation:

1. **`delta = 0` AND `L = 2` (undirected edge).** Writers MUST emit
   `ci = [0, 1]`. This collapses the two orientations of an undirected
   edge into one canonical form, saves storage, and gives readers a
   guaranteed-no-duplicates view.

2. **`delta ≠ 0`.** No canonicalization. Endpoint 0 lives at the owning
   level; endpoints 1..L-1 live at the target level — direction is
   semantically meaningful and the `ci` permutation encodes which side
   each endpoint is on.

3. **`L ≥ 3`.** No canonicalization mandated. Higher-arity records
   carry geometry semantics (face winding, parent ordering) that the
   spec doesn't presume to standardize. Geometry writers MAY apply
   their own conventions (e.g. consistent winding direction) but the
   spec's only hard requirement is the *coverage invariant* below.

### Validation rules

L1 (structural) — walks every `<delta>` subdir under both `links/` and
`cross_chunk_links/`:

- `links/0/` exists for every geometry type that declares it in its
  `arrays_present` capability list (graph, polyline, streamline,
  skeleton, mesh).
- Any `links/<delta != 0>/` or any `kK` sub-array under
  `cross_chunk_links/<delta>/` triggers the `CAP_MULTISCALE_LINKS`
  capability check on root metadata, **and**, since 0.8, also the
  `CAP_PARTITIONED_CROSS_CHUNK_LINKS` check.
- The `layout` key in every `cross_chunk_links/<delta>/` group's
  `.zattrs` equals `"sharded_v1"`. Any other value (or absence) is
  a fatal error directing the user to the 0.7 → 0.8 migration helper.

L3 (consistency) — see
[`zarr_vectors/validate/consistency.py`](../../../zarr_vectors/validate/consistency.py):

- **Per-cell byte-length.** For every populated cell:
  `len(cell_bytes) % (9 * link_width) == 0`.
- **Per-record ci range.** Every `ci_i ∈ [0, K-1]`.
- **Coverage invariant.** For every record in a cell of the `kK`
  array, the set `{ci_0, …, ci_{L-1}}` equals `{0, 1, …, K-1}` —
  every chunk listed in the cell coord is referenced by at least one
  endpoint. Records that don't use every cell-listed chunk belong in
  a smaller-K array.
- **Lex-sorted cell coord.** In a `kK` array, the cell coord's K
  chunk-tuples are in strict lex order.
- **Canonical `ci` for `k2` at `delta=0`.** Every record has
  `ci = [0, 1]`.
- **Chunk-coord arity.** Each chunk-tuple in a cell coord has arity
  `sid_ndim` (verified against the array's declared shape).
- **Existence of chunks at each side's level.** For `delta != 0`, the
  validator checks that the chunk referenced by `ci_0` exists at the
  owning level and the chunks referenced by `ci_{i>0}` exist at level
  + delta.  For `delta == 0`, every chunk in the cell coord must name
  a chunk present at the level's chunk grid.
- **Attribute parity.** For every populated cell in
  `cross_chunk_link_attributes/<name>/<delta>/kK`, the row count
  matches the record count in the parallel
  `cross_chunk_links/<delta>/kK` cell at the same coord.
- **Same-chunk overlap warning.** Records in `kK=1` (same-chunk
  records) are legal but the validator emits a warning recommending
  `links/<delta>/<X>` for natural intra-chunk edges; reserve `k1`
  cells for special-case bridges.

L4 (semantic, opt-in): for each `delta > 0`, the union of source-side
endpoints in `links/+delta/*` and across every populated cell in every
`cross_chunk_links/+delta/kK` array must cover every vertex at the
source level — i.e. every fine vertex has at least one parent at level
`source_level + delta`. Useful as an ID-preservation cross-check for
stores written with the per-object pyramid regime; off by default
because it requires a full scan.

### `link_attributes/<name>/<delta>/<chunk_key>` — intra-chunk attrs

Parallel to `links/<delta>/<chunk_key>`. One ragged group per chunk
matching the link group layout exactly; rows are in the same order as
the link rows.

**`.zattrs` schema:**

```jsonc
{
  "zv_array":   "link_attribute",
  "name":       "weight",
  "dtype":      "float32",
  "level_delta": 0
}
```

### `cross_chunk_link_attributes/<name>/<delta>/kK/` — parallel sharded attrs

Parallel sharded vlen-bytes arrays mirroring the link `kK` arrays.
Each cell holds the dtype-typed row block for the matching link cell —
`num_records_in_link_cell` rows of either scalar (`dtype`) or
multi-channel (`dtype × C`) data, packed back-to-back.

**Parent group `.zattrs` schema** (stored on
`cross_chunk_link_attributes/<name>/<delta>/`):

```jsonc
{
  "zv_array":    "cross_chunk_link_attribute",
  "name":        "weight",
  "dtype":       "float32",
  "level_delta": 1,
  "link_width":  2,
  "sid_ndim":    3,
  "shape":       [num_links_total],   // or [num_links_total, C] for multi-channel
  "layout":      "sharded_v1"
}
```

**Per-cell parity invariant.** For every populated cell in
`cross_chunk_link_attributes/<name>/<delta>/kK` the row count equals
the record count of the parallel `cross_chunk_links/<delta>/kK` cell
at the same coord. The writer enforces this at runtime; a
desynchronised write fails with an `ArrayError`.

### Generation algorithm

**Intra-level (`delta == 0`).** Each geometry's writer (`write_graph`,
`write_polyline`, `write_mesh`, …) calls
[`partition_edges`](../../../zarr_vectors/spatial/boundary.py): for
each edge it compares the chunk indices of the two endpoints. Same
chunk → bucket into per-chunk `(M_local, link_width)` rows for
`links/0/<chunk_key>`. Different chunks → emit a cross-chunk record:

1. Compute `chunks_used = sorted(set(endpoint_i.chunk for i in range(L)))`.
2. `K = len(chunks_used)`.
3. Locate (or lazily create) the `cross_chunk_links/0/kK` sharded
   vlen-bytes array.
4. Compute the cell coord:
   `(chunks_used[0] - origin) ⧺ … ⧺ (chunks_used[K-1] - origin)`.
5. Compute the record `ci`: for endpoint `i`,
   `ci_i = chunks_used.index(endpoint_i.chunk)`.
6. Build `vi`: `vi_i = endpoint_i.vertex_index`.
7. For `L = 2` undirected edges, normalise `ci = [0, 1]` (swap `vi`
   if needed).
8. Append the `9 * L`-byte record to the cell's existing payload
   (read-modify-write at the shard level — multiple cells in one shard
   become one shard write when the writer batches per-shard).

**Cross-level (`delta != 0`).** Emitted by
[`_write_cross_level_edges`](../../../zarr_vectors/multiresolution/coarsen.py)
during pyramid construction. For each adjacent (fine, coarse) pair,
every fine vertex has exactly one trivial edge to its coarse parent
metanode. The edges are partitioned via
[`partition_cross_level_edges`](../../../zarr_vectors/spatial/boundary.py):
chunk-aligned edges (source chunk_key == target chunk_key when
re-evaluated against the coarser grid) become rows in
`links/+1/<chunk_key>`; the rest follow the same sorted-unique-chunks
+ `kK` bucketing as above. No canonicalization step for cross-level
(endpoint 0 = fine is semantically distinguished from endpoint 1 =
coarse).

When `cross_level_storage="explicit"`, the same edges are also mirrored
at the coarse level under `<-delta>` with endpoint roles swapped —
`links/-1/<chunk_key>` and `cross_chunk_links/-1/kK/` arrays. When
`cross_level_storage="implicit"`, only the `+delta` side is
materialised; readers reconstruct the `-delta` direction by walking
the `+delta` arrays at the target level.

See [Pyramid construction](../multiscale/pyramid_construction.md) for
the `cross_level_depth` / `cross_level_storage` API and
[`examples/07_multiscale_links.ipynb`](../../../examples/07_multiscale_links.ipynb)
for a full walkthrough.

**Concurrency.** Two writers writing records whose cells fall in
different outer shards write to **different shard files** and don't
conflict.  Writers whose cells share a shard collide at the shard
level (read-modify-write semantics).  Tune `shard_shape` smaller for
heavier write-contention workloads; larger for fewer total files.

### Reader access patterns

**Records between two specific chunks `A` and `B` (any L, K = 2):**

```python
from zarr_vectors.core.arrays import read_cross_chunk_link_leaf
from zarr_vectors.core.paths import cross_chunk_links_path

smaller, larger = sorted([A, B])  # lex compare
records = read_cross_chunk_link_leaf(
    lg, chunks=(smaller, larger), delta=0,
)
# Each record exposes ci + vi. ci tells you which endpoint is at smaller vs larger.
```

One lookup. No permutation scan.

**Records between three chunks `A`, `B`, `C` (any L, K = 3):**

```python
c0, c1, c2 = sorted([A, B, C])
records = read_cross_chunk_link_leaf(
    lg, chunks=(c0, c1, c2), delta=0,
)
```

One lookup; record `ci`s permute the three chunks across endpoints
according to the writer's winding-order convention.

**All records involving chunk `X` (any L, K and delta unknown):**

```python
from zarr_vectors.core.arrays import list_cross_chunk_link_leaves

# Walks every populated cell across all kN arrays at this delta,
# filtering to cells whose sorted-chunks tuple contains X.
for sorted_chunks in list_cross_chunk_link_leaves(lg, involves=X, delta=0):
    records = read_cross_chunk_link_leaf(lg, chunks=sorted_chunks, delta=0)
    ...
```

**Whole-level scan (delta given):**

```python
from zarr_vectors.core.arrays import read_cross_chunk_links

for record in read_cross_chunk_links(lg, delta=0):
    # record is ((chunk_0, vi_0), (chunk_1, vi_1), ...) in canonical order
    ...
```

Internally this walks each `kK` array's populated shards (cheap —
file count scales with data), then decodes each cell's records.  Sharding
keeps cold-storage random access reasonable: a reader interested in
one spatial region fetches only the shards covering that region.

Parallel CCL attributes use `read_cross_chunk_link_attributes`:

```python
from zarr_vectors.core.arrays import read_cross_chunk_link_attributes

weights = read_cross_chunk_link_attributes(lg, "weight", delta=1)
# Returns a flat (num_links_total,) or (num_links_total, C) array in
# the same canonical walk order as read_cross_chunk_links.
```

### Listing available deltas

```python
from zarr_vectors.core.arrays import (
    list_link_deltas,
    list_cross_link_deltas,
    list_link_attribute_deltas,
    list_cross_chunk_link_attribute_deltas,
)
print(list_link_deltas(lg))         # e.g. [0, +1]   at the bottom level
print(list_cross_link_deltas(lg))   # e.g. [0, +1]
```

---

## Migration from 0.7

The 0.7 monolithic-blob layout (`cross_chunk_links/<delta>/data` as a
single int64 byte array) is **not readable** by 0.8 readers.  A
one-shot migration utility re-bins records into the `kK` sharded
vlen-bytes arrays, applies the `delta=0, L=2` canonicalization, and
stamps `CAP_PARTITIONED_CROSS_CHUNK_LINKS` on root metadata.  See the
[0.7 → 0.8 migration guide](../migration/0.7_to_0.8.md) for details.

---
