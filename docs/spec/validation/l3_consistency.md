# L3: Consistency validation

## Terms

**Consistency check**
: A validation check that reads array data and verifies that the values
  in one array are logically consistent with values in another. L3 checks
  read all chunks of all arrays at each level.

**Fragment-index arithmetic check**
: Verification that the fragment-index blob in `vertex_fragments/<chunk>`
  is self-consistent: magic and version are correct, the range bitmap's
  popcount matches the header's `R`, CSR offsets are monotone, and every
  range fragment's `[start, start + count)` and every explicit fragment's
  indices lie in `[0, vertex_count_in_chunk)`.

**Manifest integrity**
: Verification that every block in `object_index/data` references a chunk
  that exists at the level and a fragment that exists in that chunk's
  `vertex_fragments/` blob, and that decoding the manifest yields exactly
  one fragment per fragment reference (no out-of-range indices).

**Attribute alignment**
: Verification that the length of each `attributes/<name>/` chunk slice
  equals the length of the corresponding `vertices/` chunk slice. Misaligned
  attributes indicate a bug in the writer's vertex-reordering logic.

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

### Fragment-index arithmetic checks

For every non-empty chunk in `vertices/` and `vertex_fragments/`:

| Check | Rule | Failure type |
|-------|------|--------------|
| `frag_magic` | First 4 bytes equal `0x5A56_4647` (`'ZVFG'`) | Error |
| `frag_version` | Header `version` field equals 1 | Error |
| `frag_popcount` | Header `num_range_fragments` equals `popcount(bitmap[0:F])` | Error |
| `frag_bitmap_padding` | Bitmap bytes beyond `ceil(F/8)` are zero | Warning |
| `frag_csr_monotone` | `explicit_offsets[0] == 0` and `explicit_offsets[i+1] >= explicit_offsets[i]` | Error |
| `frag_range_in_bounds` | Every range fragment's `start + count ≤ vertex_count_in_chunk` and `start >= 0` | Error |
| `frag_indices_in_bounds` | Every explicit fragment's indices lie in `[0, vertex_count_in_chunk)` | Error |
| `frag_indices_non_negative` | Every entry of `explicit_indices` is `>= 0` | Error |
| `frag_vg_order` | At level 0 with `shared_fragments=False`: vertices in fragment `f`'s row slice all share the same bin flat index | Error |

For `link_fragments/<chunk>`, the same checks apply with
`vertex_count_in_chunk` replaced by `link_count_in_chunk` (rows of
`links/0/<chunk>`).

The `frag_vg_order` check is the most expensive: it requires computing the
bin flat index of every vertex and comparing to the writer-emitted fragment
order. It runs by default at L3 but can be disabled with
`skip_vg_order_check=True` for large stores where only fragment-index
arithmetic is needed:

```python
result = validate("scan.zarrvectors", level=3, skip_vg_order_check=True)
```

### Attribute alignment checks

For every chunk at every level:

| Check | Rule | Failure type |
|-------|------|--------------|
| `attr_length_matches` | `len(attributes[name][chunk])` equals `len(vertices[chunk])` for all named attributes | Error |
| `attr_no_nan_default` | No NaN values in float attributes unless `fill_value = NaN` is declared | Warning |
| `obj_attr_length` | `len(object_attributes[name])` equals `object_index.shape[0]` | Error |

### Object index checks

| Check | Rule | Failure type |
|-------|------|--------------|
| `obj_index_offsets_monotone` | `object_index/offsets` is monotonically non-decreasing and `offsets[0] == 0` | Error |
| `obj_index_blob_decodes` | Every per-object manifest blob decodes via `decode_object_manifest_blocks` without error | Error |
| `obj_index_valid_chunks` | Every block's `chunk_coords` names a chunk in the level's chunk grid | Error |
| `obj_index_valid_fragments` | Every block's `fragment_index` (single mode), `[start, start + count)` (range mode), or index list (explicit mode) names fragments present in the chunk's `vertex_fragments/` blob | Error |
| `obj_index_no_double_share` | When `LevelMetadata.shared_fragments == False`: no `(chunk_coords, fragment_index)` pair appears in more than one manifest | Error |

### Cross-chunk link checks

The L3 walker enumerates every `<delta>` subdir under
`cross_chunk_links/` via `list_cross_link_deltas` (see
[`zarr_vectors/validate/consistency.py`](../../../zarr_vectors/validate/consistency.py))
and validates each independently.

| Check | Rule | Failure type |
|-------|------|--------------|
| `ccl_cell_chunk_arity` | Each `kK` array cell coord's K chunk-tuples each have arity `sid_ndim` (verified against the array's declared shape) | Error |
| `ccl_cell_lex_sorted` | The K chunk-tuples in the cell coord are in strict lex order (sorted-unique-chunks invariant) | Error |
| `ccl_cell_record_size` | Every populated cell's byte length is a multiple of `9 * link_width` | Error |
| `ccl_ci_range` | For every record, every `ci_i ∈ [0, K-1]` | Error |
| `ccl_coverage` | For every record, the set `{ci_0, …, ci_{L-1}}` equals `{0, 1, …, K-1}` — every chunk listed in the cell coord is referenced by at least one endpoint.  Records that don't use every cell-listed chunk belong in a smaller-K array. | Error |
| `ccl_canonical_l2_delta0` | For `k2` cells at `delta=0`, every record has `ci = [0, 1]` (canonical undirected orientation) | Error |
| `ccl_src_chunk_exists` | For every `<delta>`: the chunk in the cell coord referenced by `ci_0` exists in the owning level's `vertex_fragments/` | Error |
| `ccl_tgt_chunk_exists` | For `delta == 0` only: every chunk in the cell coord exists in the owning level's chunk grid | Error |
| `ccl_tgt_chunk_at_offset_level` | For `delta != 0`: chunks in the cell coord referenced by `ci_{i>0}` are validated when the walker reaches level `source_level + delta` | Error |
| `ccl_attribute_parity` | For every populated cell in `cross_chunk_link_attributes/<name>/<delta>/kK`, row count equals the parallel `cross_chunk_links/<delta>/kK` cell's record count | Error |
| `ccl_same_chunk_warning` | K=1 leaves (`cross_chunk_links/<delta>/<X>/data`) are allowed but trigger a warning recommending `links/<delta>/<X>` for natural intra-chunk edges; reserve K=1 cross-chunk-link leaves for special-case bridges | Warning |
| `ccl_no_polyline_cycles` | For polyline/streamline stores at `delta == 0`: the directed graph formed by intra-level cross-chunk links contains no cycles | Error |

### Edge index checks

For polyline, streamline, graph, and skeleton types:

| Check | Rule | Failure type |
|-------|------|--------------|
| `edge_indices_in_bounds` | All local vertex indices in `links/<delta>/` are in `[0, N_chunk)` or equal `−1` | Error |
| `no_self_loops` | No edge has `src == dst` | Error |
| `polyline_continuity` | For polyline/streamline: every non-terminal vertex has exactly one outgoing intra-chunk edge (or a cross-chunk link as continuation) | Error |

### Mesh face checks

| Check | Rule | Failure type |
|-------|------|--------------|
| `face_indices_valid` | All positive face indices in `[0, N_chunk)` | Error |
| `boundary_indices_decodable` | All negative face indices decode to valid global vertex IDs | Error |
| `no_degenerate_faces` | No face has two or more identical vertex indices | Error |

### Pyramid consistency checks (L3 component)

| Check | Rule | Failure type |
|-------|------|--------------|
| `vertex_count_non_increasing` | Total vertex count at level N ≤ total vertex count at level N-1 | Error |
| `attribute_names_match` | Attribute name sets are identical across all levels | Warning |

### Example L3 report (abbreviated)

```
Level 3 validation of scan.zarrvectors
========================================
Checking 0 (125 chunks)…
PASS  frag_magic [0]                all chunks: magic 0x5A56_4647 ✓
PASS  frag_popcount [0]              all chunks: R == popcount(bitmap) ✓
PASS  frag_range_in_bounds [0]      all range fragments in bounds
PASS  frag_vg_order [0]             all vertices in correct fragment order
PASS  attr_length_matches [0]       intensity/: 125/125 chunks aligned
ERROR ccl_different_chunks [0]      2 links found where src chunk == dst chunk
                                    (links at rows 14502, 87331)
PASS  edge_indices_in_bounds [0]    all intra-chunk edges valid
PASS  vertex_count_non_increasing   level 1 (82453) ≤ level 0 (100000) ✓

Level 3 validation: FAIL — 47 passed, 0 warnings, 1 error
```

The error above (`ccl_different_chunks`) indicates that the writer
erroneously generated cross-chunk links for same-chunk vertex pairs — the
most common correctness bug in cross-chunk link generation. See
[Cross-chunk links](../object_model/cross_chunk_links.md) for the
correct generation algorithm.
