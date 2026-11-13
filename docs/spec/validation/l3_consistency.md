# L3: Consistency validation

## Terms

**Consistency check**
: A check that reads array data and verifies that one array agrees with
  another. L3 reads every chunk of every level.

**Directory-name invariant**
: A property L3 checks by parsing an `<offsets>` array name instead of
  reading records (see [Links](../object_model/links.md)).

## Introduction

L3 reads every chunk and checks that the arrays agree with one another.
It catches writer bugs, rechunking errors and hand edits. Its cost grows
with the store.

## Technical reference

L3 is `validate_consistency` in `zarr_vectors/validate/consistency.py`.
Each check adds a message to the `ValidationResult` (see
[Validation overview](overview.md#validationresult-api)). Malformed
fragment-index blobs show up as decode errors. Messages start with
`resolution_N:`, left out below.

### Levels and chunks

| Check | Severity | Message | When it fires |
|-------|----------|---------|---------------|
| The level opens | error | `cannot open: ...` | |
| The level has chunks; otherwise the level is skipped | warning | `no chunk data` | |
| The chunk's vertices decode | error | `chunk <c> decode failed: ...` | |
| The fragments tile the vertex buffer | error | `fragments_tile is set but chunk <c> does not tile its buffer` | level has `fragments_tile`; once per level |
| Intra-chunk links are filed by vertex fragment | error | `link_fragments_aligned is set but chunk <c>'s intra-chunk links are not filed by vertex fragment` | level has `link_fragments_aligned`; once per level |
| Each fragment is 2-D with `shape[1] == sid_ndim` | error | `chunk <c> fragment[<i>] shape <s>` | |
| No fragment has NaN or Inf | warning | `chunk <c> fragment[<i>] NaN/Inf` | |
| `vertex_count` equals the decoded total | error | `metadata vertex_count=<v>, actual=<n>` | does not run on stores this package writes (octree levels excepted) |
| A chunk has no more fragments than bins | error | `chunk <c> has <n> fragments, exceeds bins_per_chunk product <m>` | point cloud only (below) |
| Each fragment's points lie in its bin (tolerance `1e-4`) | warning | `chunk <c> fragment[<i>] has <n> points outside bin <b> bounds` | point cloud only (below); first three chunks of each level |

The two bin checks run only when `geometry_types` has `point_cloud` and
no `line`, `polyline`, `streamline`, `graph`, `skeleton` or `mesh`, the
level has no `object_index/`, and some axis has more than one bin per
chunk. Bins per chunk come from the level's chunk shape and, at level 0,
the root `base_bin_shape`; coarser levels skip both checks.

### Object index

| Check | Severity | Message |
|-------|----------|---------|
| A dense index has its three arrays, no vlen `manifests`, and the dtypes and shapes in [Object manifest](../object_model/object_manifest.md#validation) | error | one per problem, such as `manifest_spans has a negative count` |
| Every manifest entry names a chunk present at the level | error | `obj <id> refs non-existent chunk <c>` |
| Every manifest entry's fragment index is below the chunk's fragment count | error | `obj <id> refs fragment_idx=<f> >= <n>` |

A level whose manifests cannot be read gets no message.

### Links

Each `links/<delta>/` family is checked with the `link_width`,
`sid_ndim`, `directed` and `store` of its group. A family whose metadata
cannot be read, or whose `sid_ndim` is `0`, is skipped. All findings
are errors; messages start with `links[delta=<d>]`.

| Check | Message | When it fires |
|-------|---------|---------------|
| The `<offsets>` name parses | `offsets segment '<s>' malformed: ...` | |
| No offset is lexicographically negative | `segment '<s>' offset <i> is lexicographically negative; ...` | undirected, `canonical`, `delta == 0` |
| The offsets in a name are non-decreasing | `segment '<s>' offsets are not non-decreasing; ...` | undirected, `canonical`, `delta == 0` |
| `num_physical_records` equals the rows on disk, every copy counted | `num_physical_records=<n> != <m> rows on disk` | the family records it |
| Each record's source chunk (endpoint 0) is present | `refs non-existent source chunk <c>` | |
| Every other endpoint's chunk is present | `refs non-existent chunk <c>` | `delta == 0` |

### Octree levels

A level with octree chunks replaces all the checks above with these.
All are errors unless marked.

- The level is not part way through a conversion.
- The leaf and cell tables read. Every leaf, and nothing else, has a
  vertex chunk at its anchor, with the vertex and fragment counts both
  tables give. The cell table names only existing leaves.
- `vertex_count` matches the leaves.
- Manifests name a leaf's anchor and a fragment it holds.
- There is no cross-level link family. Each link joins stored leaves at
  rows they hold; `num_physical_records` matches.
- `vertices/` carries the octree marker codec (warning).
- No leaf is wider than `max_leaf_cells`. On a sharded level each leaf
  lies inside one shard (warning when no `max_leaf_cells` is set).

### Example

A line store with one vertex chunk deleted and a wrong
`num_physical_records`:

```
Level 3 validation: FAIL
  23 passed, 0 warnings, 2 errors
  ERROR: resolution_0: fragments_tile is set but chunk (-1, -1, 2) does not tile its buffer
  ERROR: resolution_0: links[delta=0] num_physical_records=400 != 388 rows on disk
```
