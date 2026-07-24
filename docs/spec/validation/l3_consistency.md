# L3: Consistency validation

## Terms

**Consistency check**
: A validation check that reads array data and verifies that the values
  in one array are logically consistent with values in another. L3 checks
  read all chunks of all arrays at each level.

**Decode check**
: Verification that a chunk's blob decodes at all. L3 does not
  independently re-derive the fragment-index byte arithmetic; it decodes
  each chunk through the normal reader and reports any decoder exception
  as an error. Malformed magic, version, or offsets therefore surface as
  a decode failure rather than as a named per-field check.

**Manifest integrity**
: Verification that every `(chunk_coords, fragment_index)` pair in a
  decoded object manifest names a chunk present at the level and a
  fragment index below that chunk's fragment count.

**Directory-name invariant**
: An invariant that L3 checks by parsing an *array path segment* rather
  than by reading cell contents. Under the offset layout the
  canonical-sort property of a link family is a property of its
  `<offsets>` directory name, so checking it costs one parse per offset
  array rather than one per record.

---

## Introduction

L3 validation reads array data and checks that the store's internal
structure is logically consistent. It is the first level that can detect
bugs introduced by incorrect writer implementations, rechunking errors, or
manual store modifications.

L3 is substantially more expensive than L1–L2 because it reads all chunks.
For a 100 GB store, L3 may take 5–30 minutes depending on storage bandwidth.
For development and CI, run L3 on small synthetic stores; run L1–L2 on
full-size production stores unless a specific consistency issue is suspected.

---

## Technical reference

L3 is implemented by
[`validate_consistency`](../../../zarr_vectors/validate/consistency.py).
It walks every resolution level and, within each, performs the checks
below. Checks are reported as free-text messages on a
`ValidationResult`; they do not carry stable machine-readable check IDs
(see [Validation overview](overview.md#validationresult-api)).

### Chunk decode and vertex checks

For every chunk key at the level:

| Rule | Failure type |
|------|--------------|
| The chunk's vertex blob decodes via `read_chunk_vertices` | Error |
| Each decoded fragment is 2-D with `shape[1] == sid_ndim` | Error |
| No fragment contains NaN / Inf | Warning |
| The level's `vertex_count` attribute, **if present**, equals the total decoded vertex count | Error |

A level with no chunk keys emits a warning (`no chunk data`) and is
otherwise skipped.

### Bin-layout checks (point-cloud stores only)

These run **only** when the store's `geometry_types` contains
`point_cloud` and none of `polyline`, `streamline`, `line`, `graph`,
`skeleton`, `mesh`, **and** the level has no `object_index` — i.e. only
for undifferentiated point clouds where fragments really do correspond
to bins. Other types use fragments for segments, endpoints, or
per-object partitions, so a fragment-per-bin rule does not apply to
them. They also require at least one axis with `bins_per_chunk > 1`.

| Rule | Failure type |
|------|--------------|
| A chunk's fragment count does not exceed the product of `bins_per_chunk` | Error |
| Each fragment's points lie within their bin's bounds (tolerance `1e-4`) | Warning |

The bin-bounds check is a **spot check**: it examines at most the first
3 chunks per level, not every chunk.

`bins_per_chunk` is computed from the level's *effective* chunk shape
(honouring a per-level `chunk_shape` override) and the level's
effective bin shape.

### Object manifest checks

Read via `read_all_object_manifests`. Failures to read the manifests at
all are silently skipped (the store may legitimately have none).

| Rule | Failure type |
|------|--------------|
| Every manifest entry's `chunk_coords` names a chunk present at this level | Error |
| Every manifest entry's `fragment_index` is `<` that chunk's decoded fragment count | Error |

### Link checks

The walker enumerates every `<delta>` under `links/` via
`list_link_deltas` and validates each family independently. A family
whose metadata cannot be read, or whose `sid_ndim` is `0`, is skipped.
Family-wide policy (`link_width`, `sid_ndim`, `directed`, `store`) is
read from the `links/<delta>/` **group**.

#### Offsets-segment checks (directory names)

For every `<offsets>` array in the family:

| Rule | Failure type |
|------|--------------|
| The segment parses under the family's `sid_ndim` / `link_width` (`parse_offsets`) | Error |

The remaining segment checks are **gated** on the family being
undirected, single-copy, and intra-level:

```python
enforce_canonical = (not directed) and store == "canonical" and delta == 0
```

| Rule (only when `enforce_canonical`) | Failure type |
|------|--------------|
| Each offset is lexicographically non-negative — its first non-zero component is `> 0` | Error |
| The offsets within a segment are non-decreasing | Error |

The all-zero offsets segment (the intra-chunk array) is **legal** and
passes both: its lex-sign is `0`, not negative. Intra-chunk records are
deduplicated by the vertex-index tie-break in the canonical sort, not
by the offset sign.

The gate matters, because all three excluded cases legitimately carry
lexicographically negative offsets:

- **`directed=True`** — endpoint order is data, so `A→B` and `B→A` file
  under opposite offsets (`0.0.+1` vs `0.0.-1`).
- **`store="duplicate"`** — each incident chunk leads in its own copy.
- **`delta != 0`** — cross-level records are never sorted; their source
  is always input endpoint 0.

#### Record checks

| Rule | Failure type |
|------|--------------|
| `num_physical_records` in the family metadata, **if present**, equals the number of rows `read_links` returns | Error |
| Every record's endpoint-0 (source) chunk is present at this level | Error |
| For `delta == 0` only: every other endpoint's chunk is present at this level | Error |

`read_links` returns one row per **physical** record, so a
`store="duplicate"` family counts each copy — which is exactly what
`num_physical_records` records.

For `delta != 0` only endpoint 0 is constrained here: the other
endpoints live at level `owning + delta` and are validated against that
level's own grid when the walker reaches it.

### Example L3 report (abbreviated)

```
Level 3 validation: FAIL
  4 passed, 1 warnings, 2 errors
  ERROR: resolution_0: links[delta=0] segment '0.0.-1' offset 1 is
         lexicographically negative; a canonical family stores each
         record once, under the positive offset
  ERROR: resolution_0: links[delta=0] refs non-existent source chunk (7, 2, 1)
  WARN:  resolution_0: chunk (3, 0, 0) fragment[2] NaN/Inf
```

The first error is the characteristic cross-chunk-link writer bug under
the offset layout: an undirected canonical family must store each
record exactly once, under the lexicographically positive offset. Both
`0.0.+1` and `0.0.-1` existing means the writer canonicalised
inconsistently. See [Links](../object_model/links.md) for the correct
generation algorithm.
