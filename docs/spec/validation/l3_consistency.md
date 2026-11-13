# L3: Consistency validation

## Terms

**Consistency check**
: A check that reads array data and verifies that one array agrees with
  another. L3 reads every chunk of every level.

**Decode check**
: Decoding each chunk through the normal reader. L3 does not re-derive the
  fragment-index byte layout; a malformed blob surfaces as a decode error.

**Manifest integrity**
: Every `(chunk, fragment_index)` pair in an object manifest names a chunk
  present at the level and a fragment index below that chunk's fragment
  count.

**Directory-name invariant**
: A property L3 checks by parsing an `<offsets>` array name instead of
  reading records. For a canonical link family, the sort order of its
  records shows in the names, so the check costs one parse per offsets
  array.

## Introduction

L3 reads array data and checks that the store's arrays agree with one
another. It is the first level that catches writer bugs, rechunking errors
and hand edits. It reads every chunk, so its cost grows with the store; run
it on small stores in development and CI.

## Technical reference

L3 is `validate_consistency` in `zarr_vectors/validate/consistency.py`. It
walks every level; a level that cannot be opened is an error
(`resolution_N: cannot open`). Levels with octree chunks get their own checks
([below](#octree-levels)). Each check adds a message to the
`ValidationResult` (see
[Validation overview](overview.md#validationresult-api)).

### Chunk decode and vertex checks

A level with no chunks is a warning (`no chunk data`) and is otherwise
skipped. For every chunk:

| Rule | Failure |
|------|---------|
| The chunk's vertices decode | Error |
| When the level claims `fragments_tile`, the chunk's fragments tile its vertex buffer | Error, once per level |
| When the level claims `link_fragments_aligned`, the chunk's intra-chunk links are filed by vertex fragment | Error, once per level |
| Each fragment is 2-D with `shape[1] == sid_ndim` | Error |
| No fragment contains NaN or Inf | Warning |

L3 also compares a level's `vertex_count` with the decoded total, but reads
it from the top level of the level's attributes, where a store keeps only
the `zarr_vectors_level` block. The comparison therefore does not run on
stores written by this package (the octree branch reads it correctly).

### Bin-layout checks (point clouds only)

These run only when the store's `geometry_types` contains `point_cloud` and
no other type, the level has no `object_index/`, and at least one axis has
more than one bin per chunk:

| Rule | Failure |
|------|---------|
| A chunk has no more fragments than bins | Error |
| Each fragment's points lie inside its bin (tolerance `1e-4`) | Warning |

The bin-bounds check examines only the first three chunks of each level.
Bins per chunk come from the level's effective chunk shape and, at level 0,
the root `base_bin_shape`; coarser levels skip both checks.

### Object index checks

| Rule | Failure |
|------|---------|
| A dense object index has its arrays, dtypes and shapes as stamped, and no vlen `manifests` array beside them | Error per problem |
| Every manifest entry names a chunk present at the level | Error |
| Every manifest entry's fragment index is below that chunk's fragment count | Error |

A level whose manifests cannot be read is passed over without a message.

### Link checks

Every `links/<delta>/` family is checked. Family policy (`link_width`,
`sid_ndim`, `directed`, `store`) is read from the family group; a family
whose metadata cannot be read, or whose `sid_ndim` is `0`, is skipped.

For every `<offsets>` array:

| Rule | Failure |
|------|---------|
| The name parses under the family's `sid_ndim` and `link_width` | Error |

The next two checks run only when the family is undirected, single-copy and
within one level:

```python
enforce_canonical = (not directed) and store == "canonical" and delta == 0
```

| Rule (only when `enforce_canonical`) | Failure |
|------|---------|
| No offset is lexicographically negative: its first non-zero component is `> 0` | Error |
| The offsets in a name are non-decreasing | Error |

The all-zero offsets array (intra-chunk records) passes both. The excluded
families can carry negative offsets legitimately: a directed family keeps
endpoint order, so `A→B` and `B→A` file under opposite offsets; a duplicate
family stores a copy led by each incident chunk; a cross-level family is
never sorted. See [Links](../object_model/links.md).

For the records:

| Rule | Failure |
|------|---------|
| `num_physical_records`, when recorded, equals the rows on disk | Error |
| Every record's source chunk (endpoint 0) is present at the level | Error |
| For `delta == 0`, every other endpoint's chunk is present at the level | Error |

The row count includes every copy in a duplicate family. For `delta != 0`,
the other endpoints live at another level and are not checked here.

### Octree levels

A level with octree chunks (each leaf is one chunk at its anchor cell)
replaces the checks above with these, all errors unless noted:

- the level is not part way through a conversion;
- the leaf and cell tables read; every leaf, and nothing else, has a vertex
  chunk at its anchor, holding the vertex and fragment counts both tables
  give; the cell table names only existing leaves;
- the level's `vertex_count` matches the leaves;
- object manifests name a leaf's anchor and a fragment it holds;
- there is no cross-level link family; each link joins stored leaves at rows
  they hold, and the family's `num_physical_records` matches;
- `vertices/` carries the octree marker codec (a warning if not);
- on a sharded level, no leaf is wider than `max_leaf_cells` and every leaf
  lies inside one shard (a straddling leaf is a warning when no
  `max_leaf_cells` is set).

### Example L3 report

A line store with one vertex chunk deleted and a wrong
`num_physical_records`:

```
Level 3 validation: FAIL
  23 passed, 0 warnings, 2 errors
  ERROR: resolution_0: fragments_tile is set but chunk (-1, -1, 2) does not tile its buffer
  ERROR: resolution_0: links[delta=0] num_physical_records=400 != 388 rows on disk
```

L3 pass messages read like `resolution_0: 192 chunks decoded, 1999 vertices`,
`resolution_0: object_index validated (1000 objects)` and
`resolution_0: links[delta=0] validated (388 links across 11 offset arrays)`.
