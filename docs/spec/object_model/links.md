# Links

```{admonition} Format change in ZVF 0.9.0
:class: note

Prior to 0.9.0 connectivity was split across two families:
`links/<delta>/` (intra-chunk) and `cross_chunk_links/<delta>/`
(inter-chunk, keyed by endpoint-chunk tuples), each with its own
parallel attribute family. **Both `cross_chunk_links/` and
`cross_chunk_link_attributes/` are gone.** Connectivity is now one
family, `links/<delta>/<offsets>/`, in which an intra-chunk link is
simply a link whose relative offsets are all zero.

Stores written before 0.9.0 are not readable by 0.9+ readers; rewrite
from source.
```

## Terms

**Link**
: A record of `link_width` (`L`) endpoints. `L = 2` is an edge, `L = 3`
  a triangle face, `L = 1` a parent→child reference used by pyramid
  metanode drill-down. Endpoints may live in different spatial chunks
  and, when `delta != 0`, at different resolution levels.

**Source chunk**
: The chunk of the endpoint a record is *filed under* — the array cell
  that holds it. Every other endpoint is located relative to it.

**Relative offset** (`o_k`)
: The chunk-grid displacement from the source chunk to endpoint `k`, as
  a signed `sid_ndim`-tuple. `o_0 = 0` by definition (the source is its
  own reference) and is **never encoded**, so a record carries `L - 1`
  offsets.

**Intra-chunk link**
: A link whose relative offsets are **all zero** — every endpoint in the
  source chunk. Not a separate family: it is the all-zero-`<offsets>`
  array.

**Level delta** (`<delta>`)
: A signed path segment saying how many pyramid levels a record spans.
  `0` = every endpoint at the owning level; `+N` = endpoints `1..L-1`
  are `N` levels coarser; `-N` = `N` levels finer. Literal segments:
  `"0"`, `"+1"`, `"-1"`, `"+2"`, …

**Link attribute**
: Per-record data parallel to a link array, at
  `link_attributes/<name>/<delta>/<offsets>/` — mirroring the link
  array cell-for-cell.

---

## Introduction

Connectivity lives in exactly one family per resolution level:

```
links/<delta>/                             GROUP  — family-wide policy
links/<delta>/<offsets>/                   ARRAY  — rank-D vlen; cell = SOURCE chunk
link_attributes/<name>/<delta>/            GROUP
link_attributes/<name>/<delta>/<offsets>/  ARRAY  — mirrors it cell-for-cell
```

The relationship between a record's endpoints is factored **into the
path**. Because the `<offsets>` segment already says where the other
endpoints sit, each array is a plain rank-D vlen array over the level's
chunk grid — one cell per source chunk — and each stored `vi_k` is a
vertex index *local to chunk* `src + o_k`.

That factoring is what collapses the old two-family design. An
intra-chunk link is a link whose offsets are all zero; a cross-chunk
link is one whose offsets are not. They differ in a directory name, not
in kind.

This page documents:

- the [offsets grammar](#offsets-grammar) and the directory-name invariants,
- [where a record is filed](#placement-the-source-anchored-cell) and how it is
  [encoded](#cell-encoding-two-branches-one-condition),
- the [`perm_idx`](#perm_idx-present-only-where-a-sort-happened) rule,
- [`directed` / `store`](#directed-and-store) policy,
- the [enumeration order](#enumeration-order) that aligns attributes to links,
- the [cross-level anchor](#cross-level-placement-the-anchor) and why a naive
  coordinate difference is wrong,
- the [metadata](#metadata) schemas and the [validation](#validation) rules.

For a worked end-to-end example, see
[`examples/07_multiscale_links.ipynb`](../../../examples/07_multiscale_links.ipynb).

---

## Technical reference

### Offsets grammar

The `<offsets>` segment carries `link_width - 1` offsets. Each offset is
`sid_ndim` signed components joined by `.`; offsets are joined by `_`.
Components use the same signed convention as `<delta>` (`0`, `+1`, `-1`),
so the leading `+` is preserved.

| Segment | `L` | Meaning |
|---------|-----|---------|
| `0.0.0` | 2 | intra-chunk edge — both endpoints in the source chunk |
| `0.0.+1` | 2 | edge to the neighbour one chunk along `+z` |
| `0.0.-1` | 2 | edge to the neighbour one chunk along `-z` |
| `0.0.+1_0.+1.0` | 3 | triangle spanning the source, `+z`, and `+y` |
| `0.0.0_0.0.0` | 3 | intra-chunk triangle |
| `self` | 1 | `link_width == 1`; no other endpoint to locate |

`self` is a literal: with one endpoint there are zero offsets to encode,
and an empty segment would make `links/<delta>/` ambiguously an array
rather than a group.

Compose and parse these with the helpers in
[`zarr_vectors/core/paths.py`](../../../zarr_vectors/core/paths.py) —
never hand-roll the segments:

```python
from zarr_vectors.core.paths import (
    format_delta, parse_delta,        # 0 -> "0";  1 -> "+1";  -2 -> "-2"
    format_offsets, parse_offsets,    # ((0, 0, 1),) <-> "0.0.+1"
    intra_offsets,                    # the all-zero offsets for (sid_ndim, L)
    is_intra,                         # all offsets zero?
    links_group_path,                 # links/<delta>
    links_path,                       # links/<delta>/<offsets>
    link_attributes_group_path,       # link_attributes/<name>/<delta>
    link_attributes_path,             # link_attributes/<name>/<delta>/<offsets>
)
```

**Directory-name invariants.** `parse_offsets` rejects a segment whose
arity disagrees with the family, so a malformed listing fails fast
rather than decoding to the wrong geometry:

1. The segment has exactly `link_width - 1` offsets, or is `self` when
   `link_width == 1`.
2. Every offset has exactly `sid_ndim` components.
3. `self` appears **iff** `link_width == 1`.

An undirected `canonical` family at `delta == 0` carries two further
name invariants, enforced by the L3 validator — see
[Validation](#validation).

### Placement: the source-anchored cell

Where a record goes is decided in exactly one place:
[`partition_records_by_offset`](../../../zarr_vectors/spatial/boundary.py),
which for each record consults `_cell_placements` for the storage
permutations `sigma` it is filed under. For each `sigma`:

- the **source** is endpoint `sigma[0]`; its chunk is the array cell;
- the **offsets** are `chunk(sigma[k]) - anchor(source)` for `k > 0`
  (see [the anchor](#cross-level-placement-the-anchor));
- the stored vertex indices are `[vi(sigma[0]), …, vi(sigma[L-1])]`,
  each local to its own endpoint's chunk;
- `perm_idx = lehmer(sigma)`, when the array carries that column.

Decoding inverts this: endpoint `k`'s chunk is `anchor(src) + o_k`, and
its vertex index is the `k`-th stored `vi`. Since `o_0 = 0`, endpoint 0's
chunk is the cell itself.

### `perm_idx`: present only where a sort happened

A record's rows are `L` ints — or `1 + L` (`[perm_idx, vi_0 … vi_{L-1}]`)
when the array carries a permutation column. `perm_idx` is a **Lehmer
code**: an integer in `[0, L!)` packing the permutation `sigma`, so
readers can undo a canonical sort and recover input endpoint order
(mesh-face winding, edge direction).

It exists **only** to undo that sort, so it is stored exactly where a
non-identity placement is possible.
[`links_has_perm(offsets, delta, directed, store)`](../../../zarr_vectors/core/arrays.py)
is the single definition — writer, reader, and the attribute writer all
consult it, and it is mirrored in each array's `has_perm` metadata:

```python
def links_has_perm(offsets, *, delta, directed, store):
    if is_intra(offsets):    return False   # identity: nothing to canonicalise
    if delta != 0:           return False   # cross-level: source is always endpoint 0
    if store == "duplicate": return True    # each copy leads with a different endpoint
    return not directed                     # undirected sorts; directed does not
```

So `has_perm` is true exactly when the record is **non-intra AND
`delta == 0` AND (`store == "duplicate"` OR not `directed`)**.

The three false cases are forced identity placements:

- **Intra-chunk** — all endpoints share the source chunk, so there is
  nothing to canonicalise. Sorting would reorder endpoints and cost a
  column that is always `0`. Preserving input order here is also what
  keeps the all-zero-offsets array byte-identical to the pre-merge
  `links/<delta>/` (polyline traversal order is data).
- **Cross-level (`delta != 0`)** — endpoints are distinguished by
  *level*, not by coord order, so the source is always input endpoint 0.
- **Directed canonical** — input order is data and is preserved as-is.

Storing `perm_idx` unconditionally would add 8 bytes to every
intra-chunk link — the overwhelming majority of rows — for a value that
is always zero.

### Cell encoding: two branches, one condition

[`write_chunk_links`](../../../zarr_vectors/core/arrays.py) selects the
cell encoding on exactly one condition — `delta == 0 and is_intra(offsets)`:

| Condition | Encoding | Sidecar |
|-----------|----------|---------|
| `delta == 0` **and** offsets all zero | flat concatenated rows (`encode_ragged_ints`) | `link_fragments/<chunk>` |
| otherwise | inline self-describing ragged blob (`encode_ragged_blob`) | none |

That single condition reproduces **both** pre-merge layouts: the
all-zero array is byte-identical to the old `links/<delta>/`, and every
other offsets array matches the old `cross_chunk_links/<delta>/` cells.

The flat branch's per-group row ranges live in the sibling
[`link_fragments/<chunk>`](../layout/fragment_index_arrays.md) index.
Link groups need **not** be 1:1 with the chunk's vertex fragments.
`link_fragments/<chunk>` is keyed by chunk **alone** — no delta, no
offsets — which is why only this branch may write it; see
[Fragment-index arrays](../layout/fragment_index_arrays.md) for why a
second offsets array writing it would silently clobber the intra array's
fragment index.

### `directed` and `store`

Both are **family-wide**, stamped on the `<delta>` group, because every
offsets array under it decodes against them.
[`write_links`](../../../zarr_vectors/core/arrays.py) refuses to flip
either while sibling offset arrays survive — those siblings keep
decoding against the group's policy, so re-stamping it would silently
corrupt them. Appending with a conflicting policy raises `ArrayError`.

**`directed` (bool, default `false`).**

- `false` (undirected): endpoint order is not meaningful. A record's
  endpoints are canonical-sorted by `(chunk_coords, vi)`, which is what
  makes the stored offset lexicographically positive — so each record is
  stored exactly once. `perm_idx` recovers input order.
- `true` (directed): endpoint order **is** data — streamline
  predecessor→successor, skeleton child→parent. No sort: input order is
  kept, so `A→B` and `B→A` file under **opposite offsets** (`0.0.+1` vs
  `0.0.-1`) at **different cells**, and `perm_idx` is absent (implicitly 0).

**`store` (`"canonical"` | `"duplicate"`, default `"canonical"`).**

- `"canonical"`: each record is filed in exactly **one** cell. Fewest
  objects. Finding every link incident to a chunk `C` means scanning the
  offset arrays for cells that reach `C`.
- `"duplicate"`: each record is filed **once per distinct incident
  chunk** — that chunk leads, becoming the source — as independent
  physical copies. A reader then finds every record incident to `C` by
  scanning only `C`'s cell across the offset arrays, trading up to `K`×
  storage (`K` = distinct chunks touched) for direct incidence reads.
  Copies are **not** deduplicated on read: `read_links` returns each one.

All four combinations are valid:

| `directed` | `store` | On disk |
|------------|---------|---------|
| `false` | `canonical` | one cell, canonical-sorted, `perm_idx` present |
| `false` | `duplicate` | one copy per distinct incident chunk, `perm_idx` present |
| `true` | `canonical` | one cell in input order, no `perm_idx` |
| `true` | `duplicate` | one copy per distinct incident chunk, `perm_idx` present |

Note the bottom-right row: `duplicate` implies `perm_idx` **even when
directed**, because each copy leads with a different endpoint, so `sigma`
varies per copy and input order must still be recoverable.

### Cross-level placement: the anchor

For `delta != 0` the source is **always input endpoint 0**, which keeps
the source at the owning level. Records are never canonical-sorted
across levels: a sort would dedupe nothing (`A→B` and `B→A` live in
different delta arrays at *different levels*) and would only risk
promoting a target-level endpoint to source, flipping the anchor's
scale factors.

**A naive `o = c_trg - c_src` is wrong across levels.**
`LevelMetadata.chunk_shape` is an optional per-level override
([`coarsen`](../../../zarr_vectors/multiresolution/coarsen.py) sets it to
`source × chunk_scale_factor`; the default factor is 1), so two levels
may have different chunk grids. Their chunk coords then index cells of
different sizes and are **not commensurable** — differencing them
directly is meaningless.

**Counter-example.** Root `chunk_shape = 100`, level-1 `chunk_shape = 200`
(so `r_src = 1`, `r_trg = 2`). Take two pairs in the *same* geometric
relationship — the target is the level-1 chunk immediately after the one
containing the source:

| Source (level 0) | Target (level 1) | Container | Naive `c_trg - c_src` | Anchored |
|---|---|---|---|---|
| chunk 3 = `[300, 400)` | chunk 2 = `[400, 600)` | L1 chunk 1 | `-1` | `+1` |
| chunk 1 = `[100, 200)` | chunk 1 = `[200, 400)` | L1 chunk 0 | `0` | `+1` |

The naive difference gives `-1` and `0` for the same relationship: it is
**not translation-invariant**, so it would scatter geometrically
identical records across unrelated offset arrays. The anchored form
gives `+1` for both.

[`anchor_chunk`](../../../zarr_vectors/spatial/boundary.py) re-expresses
the source in the *target* level's grid before subtracting:

```
anchor = floor(c_src * r_src / r_trg)
o      = c_trg - anchor              # decode: c_trg = anchor + o
```

where `r_L` is level `L`'s `chunk_shape` as an integer multiple of the
root `chunk_shape` (see
[`chunk_scale_factor`](../../../zarr_vectors/core/metadata.py)).

This is a **strict generalisation, not a branch**: when `r_src == r_trg`
— same level, or any pyramid built with the default
`chunk_scale_factor = 1` — `anchor == c_src` and it reduces to the plain
difference. Integer floor division is used throughout: exact for large
coords, and it floors toward `-inf`, which is what negative chunk coords
require.

### Enumeration order

Records come back in **`(offsets segment, cell)` sorted order**: offsets
segments sorted lexicographically, then cells sorted within each segment,
then write order within a cell.

This has a counter-intuitive consequence worth internalising. Segment
sorting is plain ASCII string sorting, and `+` (`0x2b`) and `-` (`0x2d`)
both sort **before** `0` (`0x30`). So the all-zero intra segment sorts
**last**:

```python
sorted(['0.0.0', '0.0.+1', '0.0.-1', '+1.0.0', '-1.0.0'])
# ['+1.0.0', '-1.0.0', '0.0.+1', '0.0.-1', '0.0.0']
#                                          ^^^^^^^ intra is LAST
```

It is deterministic, and
[`read_links`](../../../zarr_vectors/core/arrays.py) and
`read_link_attributes` share it. **That shared order is the only thing
aligning attribute rows to link records** — the two must not drift.

Consequently, `write_link_attributes` called *without* a `LinkPartition`
requires `attr_data` already in this enumeration order, because the
writer can only re-derive on-disk order. Pass the `LinkPartition`
returned by the matching `write_links` to supply data in **input** order
instead. Under `store="duplicate"` a partition is **required**: physical
copies mean positional indices cannot express the fan-out back to input
records.

### Metadata

**`links/<delta>/` — family group.** Policy every offsets array under it
must agree on:

```jsonc
{
  "zv_array":    "links_family",
  "level_delta": 0,
  "link_width":  2,
  "directed":    false,
  "store":       "canonical",
  "sid_ndim":    3,

  // Added by finalize_links (absent until it runs):
  "num_links":            12,   // logical record count
  "num_physical_records": 12    // on-disk rows; > num_links under "duplicate"
}
```

**`links/<delta>/<offsets>/` — array.** Only what decodes this array's
own cells:

```jsonc
{
  "zv_array":    "links",
  "dtype":       "int64",
  "offsets":     [[0, 0, 1]],   // parsed form of the path segment
  "has_perm":    true,          // links_has_perm(...); rows are 1 + L wide
  "link_width":  2,
  "level_delta": 0
}
```

Readers trust the stored `has_perm`, falling back to recomputing it from
the family policy for an array written before it was stamped — guessing
would mis-parse every row in the cell. The `offsets` field and the path
segment agree; the field is preferred because decoding it needs no
`sid_ndim`.

**`link_attributes/<name>/<delta>/` — family group:**

```jsonc
{ "zv_array": "link_attribute_family", "name": "weight", "level_delta": 0 }
```

**`link_attributes/<name>/<delta>/<offsets>/` — array:**

```jsonc
{
  "zv_array":    "link_attribute",
  "name":        "weight",
  "dtype":       "float32",
  "offsets":     [[0, 0, 1]],
  "level_delta": 0
}
```

### Counts and finalization

`num_links` is the **logical** record count (one per input record);
`num_physical_records` is the on-disk row count. They are equal for a
`canonical` family and differ under `duplicate`.

[`write_links`](../../../zarr_vectors/core/arrays.py) maintains both.
The decentralized per-cell writers (`write_link_cells`) do not: they
leave the counts absent for a single
[`finalize_links`](../../../zarr_vectors/core/arrays.py) pass to
reconcile. `finalize_links` recovers the logical count under `duplicate`
by deduplicating decoded records, since every physical copy decodes to
the same input-order endpoints.

```{important}
`finalize_links` must run **before** `shard_store`. It rebuilds each
array's `nonempty_chunks` manifest via `derive_nonempty_chunks`, which
is unsharded-only — a shard packs many cells into one object whose inner
index is not derivable from key names. See
[Sharding](../chunking/sharding.md).
```

**Decentralized writes.** Race-freedom rests on the flat layout: each
cell is its own object, so workers writing **disjoint** cells never touch
the same file. A cell *is* a source chunk, so giving each worker
ownership of a disjoint set of source chunks makes every cell the
property of exactly one worker. Per-cell RMW is **not** safe for two
workers hitting the same cell. Workers pass `record_presence=False` —
`nonempty_chunks` is array-wide state whose read-modify-write two workers
race on even for disjoint cells — and a coordinator rebuilds it in
`finalize_links`. A coordinator may fix the policy up front with
`create_links_family`, which stamps the group without materialising any
offsets array.

### Reading

```python
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.core.arrays import (
    read_links,               # every record under links/<delta>/
    read_links_for_tuple,     # only records spanning an exact L-chunk tuple
    read_link_attributes,
    read_chunk_links,         # one cell
    list_link_deltas,         # [0, +1, -1, ...]
    list_link_offsets,        # ['+1.0.0', '-1.0.0', '0.0.0', ...]
)

root = open_store("graph.zarrvectors")
lg   = get_resolution_level(root, 0)

records = read_links(lg, delta=0)                       # intra + cross, one family
weights = read_link_attributes(lg, "weight", delta=0)   # row-aligned to records

# Which deltas / offset arrays exist?
print(list_link_deltas(lg))            # e.g. [0, +1]
print(list_link_offsets(lg, 0))        # intra sorts LAST — see Enumeration order
```

`read_links` returns `[]` when the family is absent or carries no policy.
`read_links_for_tuple` resolves an L-chunk tuple to the single
`(offsets, source)` cell that can hold it. For an undirected `canonical`
family at `delta == 0` the tuple may be passed in any order — a `vi` only
breaks ties between endpoints already in the same chunk, so it never
reorders the chunk sequence and sorting the chunks alone reproduces the
placement. Every other family leads with input endpoint 0, so the tuple
order **is** meaningful.

### Sid-ndim assumption

Source and target levels share `sid_ndim` (uniform per store). Every
endpoint's chunk-coord arity must match it; mismatched callers fail
loudly with `ArrayError`. Chunk *spacing* may differ between levels —
that is exactly what [the anchor](#cross-level-placement-the-anchor)
handles — but the chunk-key arity does not.

### Mixed-resolution records — current limitation

`<delta>` is **uniform** across the non-source endpoints of a record:
endpoints `1..L-1` all live at `owning_level + delta`. A triangle with
vertices at levels `(N, N, N+1)` cannot be expressed. A future revision
could promote each endpoint to a `(level_delta, chunk_offset, vi)`
triple, extending the offsets segment by one component per endpoint; the
`perm_idx` and blob conventions carry over unchanged.

### Validation

The rules below are the ones the shipped validator enforces (see
[`zarr_vectors/validate/consistency.py`](../../../zarr_vectors/validate/consistency.py)
and [L3 consistency](../validation/l3_consistency.md)). L1 lists the
`<delta>` segments present under `links/` but asserts nothing about them.

**L3 (consistency)**, for every `<delta>` family carrying a `sid_ndim`:

- Every `<offsets>` segment parses against the family's `sid_ndim` and
  `link_width` — a malformed segment is an error.
- **Canonical-offset invariants**, enforced **only** for an undirected,
  `canonical`, `delta == 0` family. Under the offsets layout these are
  properties of the *directory name*, so they cost one parse per offsets
  array rather than one check per cell:
  - no offset is lexicographically negative (a canonical family stores
    each record once, under the positive offset);
  - offsets are non-decreasing across the segment.

  All three excluded cases legitimately carry lex-negative offsets:
  directed segments key on input endpoint order, `duplicate` families
  lead with each incident chunk, and cross-level records are never sorted.
- When the family group records `num_physical_records`, it must equal the
  number of rows `read_links` returns.
- Every record's source-side endpoint chunk must exist in the level's
  chunk grid. For `delta == 0`, **every** endpoint's chunk must exist;
  for `delta != 0` only the source side is constrained here, since the
  other endpoints belong to the target level.
