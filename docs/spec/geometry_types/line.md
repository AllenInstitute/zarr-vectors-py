# Line (`line`)

## Terms

**Line segment**
: A pair of vertices `(A, B)` connected by a straight edge. A `line` store
  holds an unordered collection of such pairs. Unlike polylines, there is
  no sequential ordering between segments; each segment is independent.

**`GEOM_LINE`**
: The geometry type constant `"line"`. Stored in the root `zarr_vectors`
  attribute block as `"geometry_types": ["line"]`.

**`links/<delta>/<offsets>/`**
: The link family. For `line` it holds only the segments that were
  **split across a chunk boundary**: one record per split segment,
  filed under the `<offsets>` naming where the far endpoint's chunk
  sits relative to the near one. Segments contained in one chunk emit
  no record — their fragment's vertex order carries the topology. See
  *Segments that span chunks* below, and [Links](../object_model/links.md)
  for the family's layout.

---

## Introduction

The `line` type stores independent line segments: pairs of vertices with
no sequential relationship between segments. It is appropriate for contact
sites (two endpoints of a synapse), short connectors in circuit diagrams,
pairwise distance annotations, or any other dataset where the fundamental
unit is a two-point segment rather than an ordered path.

`line` is simpler than `polyline` because each object is exactly two
vertices with no ordered path to reconstruct. It is therefore faster to
write and read than `polyline` for data that genuinely consists of
independent segments. It does carry an object model (`object_index/`),
and it does support segments that span chunks — see *Segments that span
chunks* below.

---

## Technical reference

### Arrays present

| Array path | Required | Description |
|-----------|----------|-------------|
| `vertices/` | Yes | Endpoint positions, shape `(N, D)` per chunk (`float32` by default) |
| `vertex_fragments/` | Yes | Fragment index over `vertices/` rows |
| `object_index/` | Yes | Per-line manifest of `(chunk, fragment_index)` |
| `links/0/<offsets>/` | Only when a segment spans chunks | Non-zero-offset links joining a split segment's two halves |
| `vertex_attributes/<name>/` | No | Per-endpoint attributes |
| `object_attributes/<name>/` | No | Per-line attributes |

There is no `cross_chunk_links/` array: that family was merged into
`links/`. A cross-chunk segment is simply a link whose offsets are
non-zero.

`links_convention` is `implicit_sequential` and
`object_index_convention` is `standard`.

### Segments that span chunks

Segments crossing a chunk boundary **are** supported. `write_lines`
does not reject them and has no `split_cross_chunk` option; it always
handles both cases, and reports how many it split in the returned
summary's `cross_chunk_count`.

The two cases are written differently:

| Case | Fragments | Manifest | Link |
|------|-----------|----------|------|
| Both endpoints in one chunk | One fragment holding both vertices | `[(c, f)]` | **None** — the fragment's vertex order carries the segment |
| Endpoints in different chunks | Two fragments, one vertex each | `[(c_a, f_a), (c_b, f_b)]` | One record in `links/0/<offsets>/`, offsets naming `c_b` relative to `c_a` |

This is why `line` emits **no intra-chunk links**. Connectivity is
`implicit_sequential` — within a fragment, vertex order *is* the
topology, so a same-chunk segment needs no link record and the
all-zero-offsets array would never hold a row. `write_lines` therefore
does not create a links array up front; `write_links` creates exactly
the non-zero-offset arrays the split segments land in. A `line` store
whose segments all sit within one chunk has **no `links/` group at
all**.

The link family is written with the default policy — undirected,
`store="canonical"` — so a split segment is stored once, under the
lexicographically positive offset.

### Vertex ordering

Within a fragment, vertex order is the topology: index 0 is the
segment's first endpoint and index 1 its second. Fragments are assigned
in line-id order within each chunk, so `fragment_index` is stable
across a rewrite of the same input.

### Write API

`write_lines` takes one `(N, 2, D)` array: N segments, two endpoints each.
`chunk_shape` is required.

```python
import numpy as np
from zarr_vectors.types.lines import write_lines

rng = np.random.default_rng(0)
endpoints = rng.uniform(0, 500, (10_000, 2, 3)).astype(np.float32)  # (N, 2, D)

summary = write_lines(
    "contacts.zarrvectors",
    endpoints,
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
)
print(summary)   # {"line_count": 10000, "chunk_count": ..., "cross_chunk_count": ...}
```

Optional keywords: `bounds`, `vertex_attributes` (`{name: (N, 2)}` or
`(N, 2, C)` — one value per endpoint), `object_attributes`
(`{name: (N,)}` or `(N, C)` — one value per line), `dtype="float32"`,
`backend`, `chunk_by_attribute`, `out_of_bounds="raise"`, `compressor`,
`shard_shape`. The old `attributes=` / `line_attributes=` names still
work but emit a `DeprecationWarning`. There is no `edges=` argument:
topology is implicit in the `(N, 2, D)` shape.

### Read API

```python
import numpy as np
from zarr_vectors.types.lines import read_lines

result = read_lines("contacts.zarrvectors")
print(result["line_count"])          # int
print(result["endpoints"].shape)     # (N, 2, D)
print(result["vertex_attributes"])   # {name: array}, every attribute present

# Spatial query: a line is returned if EITHER endpoint is inside the box
result = read_lines(
    "contacts.zarrvectors",
    bbox=(np.array([0., 0., 0.]), np.array([200., 200., 200.])),
)
```

Other keywords: `level=0`, `object_ids=` (line ids), `attribute_filter=`,
`backend=`. Unlike `read_points`, `read_lines` returns all per-endpoint
attributes without being asked.

### Relationship to `polyline`

`line` and `polyline` share the `links/<delta>/<offsets>/` schema and
both use `links_convention: implicit_sequential`. The distinction is:

| Property | `line` | `polyline` |
|----------|--------|-----------|
| Object shape | Exactly 2 vertices | Ordered path of any length |
| Vertices per object per chunk | At most 1 when split | Any number (a run) |
| Cross-chunk links | One per split segment | One per boundary crossing |
| Object index | Yes | Yes |

If your data has ordered paths (e.g. vessel centrelines from which individual
segment pairs were extracted), `polyline` preserves the ordering and enables
object-level queries. Use `line` only for genuinely unordered, independent
segment pairs.

### Validation

L1: `vertices/` exists at every level. `links/` is **not** required —
a `line` store with no split segments legitimately has none.

L2: `links_convention` is a recognised token; `sid_ndim`, `chunk_shape`,
and any `bin_shape` agree dimensionally.

L3: every `links/0/` offsets segment parses; because the family is
undirected, canonical, and intra-level, every offset must be
lexicographically non-negative and non-decreasing. Every record's
source chunk exists at the level, as does every other endpoint's chunk
(`delta == 0`).

L4: `links_convention` MUST be `implicit_sequential` for `line`
([`GEOMETRY_LINK_REQ`](../../../zarr_vectors/validate/conformance.py)).

See [Validation overview](../validation/overview.md) for what each level
does and does not cover.
