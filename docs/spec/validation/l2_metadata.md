# L2: Metadata validation

## Terms

**Schema check**
: A validation check that reads a `.zattrs` or `zarr.json` file and
  verifies that its values conform to the ZVF specification — correct
  types, valid ranges, required keys present, cross-key consistency.

**Divisibility constraint**
: The requirement that `chunk_shape[d] % bin_shape[d] == 0` for all `d`.
  Checked at L2 using floating-point tolerance.

**Floating-point tolerance**
: The permissible difference between two floating-point values when
  checking equality. For divisibility checks, ZVF uses a tolerance of
  `1e-6 × chunk_shape[d]` per axis.

**Array metadata consistency**
: The agreement between the shape and dtype declared in each array's
  `zarr.json` and the values declared in the store's `.zattrs`.

---

## Introduction

L2 validation reads all `.zattrs` and `zarr.json` metadata files and
checks that their contents are internally consistent and conform to the
ZVF specification. No array data is read.

L2 catches the most common class of write-time bugs: mis-typed keys,
invalid ranges, dimensionality mismatches between chunk_shape and bin_shape,
array shapes that are inconsistent with the declared chunk grid, and
multiscale metadata that does not match the level groups actually present
in the store.

---

## Technical reference

### Root `.zattrs` checks

| Check | Rule | Failure type |
|-------|------|--------------|
| `version_present` | `zarr_vectors_version` key present | Error |
| `version_known` | `zarr_vectors_version == "1.0"` | Warning (unknown version) |
| `geometry_type_valid` | `geometry_type` is one of the seven constants | Error |
| `spatial_dims_type` | `spatial_dims` is a positive integer | Error |
| `chunk_shape_length` | `len(chunk_shape) == spatial_dims` | Error |
| `base_bin_shape_length` | `len(base_bin_shape) == spatial_dims` | Error |
| `chunk_shape_positive` | All elements of `chunk_shape` are > 0 | Error |
| `base_bin_shape_positive` | All elements of `base_bin_shape` are > 0 | Error |
| `divisibility` | `chunk_shape[d] % base_bin_shape[d] ≈ 0` for all `d` (tol = 1e-6 × chunk_shape[d]) | Error |
| `multiscales_present` | `multiscales` key is present and non-empty | Error |
| `level_0_present` | `multiscales` contains an entry with `level == 0` | Error |
| `level_0_bin_ratio` | Level 0 entry has `bin_ratio == [1, …, 1]` | Error |
| `level_0_sparsity` | Level 0 `object_sparsity == 1.0` (if present) | Error |
| `levels_ordered` | `multiscales` entries are in ascending `level` order | Error |
| `levels_match_groups` | Each `path` in `multiscales` references an existing level group | Error |
| `coordinate_system_type` | `coordinate_system`, if present, is a string | Warning |
| `bounding_box_shape` | `bounding_box.min` and `.max` are D-length arrays if present | Warning |

### Per-level `.zattrs` checks

| Check | Rule | Failure type |
|-------|------|--------------|
| `level_key_matches_name` | `level` in `.zattrs` equals N in `N/` | Error |
| `bin_ratio_length` | `len(bin_ratio) == spatial_dims` | Error |
| `bin_ratio_positive` | All elements of `bin_ratio` are positive integers | Error |
| `bin_shape_consistent` | `bin_shape[d] ≈ base_bin_shape[d] × bin_ratio[d]` for all `d` | Error |
| `bin_shape_divides_chunk` | `chunk_shape[d] % bin_shape[d] ≈ 0` for all `d` | Error |
| `bin_shape_le_chunk` | `bin_shape[d] ≤ chunk_shape[d]` for all `d` | Error |
| `sparsity_range` | `0.0 < object_sparsity ≤ 1.0` | Error |
| `sparsity_for_point_cloud` | Point cloud stores have `object_sparsity == 1.0` | Error |
| `sparsity_strategy_valid` | `sparsity_strategy` is one of the four constants if present | Warning |
| `ratio_monotone` | `bin_ratio[N][d] ≥ bin_ratio[N-1][d]` for all `d` and `N > 0` | Error |

### Array `zarr.json` checks

| Check | Rule | Failure type |
|-------|------|--------------|
| `vertices_dtype` | `vertices/` dtype is `float32` | Warning (other float types accepted with warning) |
| `vertices_shape_dims` | `vertices/` last dimension equals `spatial_dims` | Error |
| `vertex_fragments_dtype` | `vertex_fragments/` `.zattrs` declares `zv_array == "vertex_fragments"` and `encoding == "fragment_index_v1"` | Error |
| `vertex_fragments_blob_magic` | Every non-empty chunk's blob starts with magic `0x5A56_4647` (`'ZVFG'`) and `version == 1` | Error |
| `link_fragments_dtype` | `link_fragments/` (when present) `.zattrs` declares `zv_array == "link_fragments"` and `encoding == "fragment_index_v1"` | Error |
| `links_dtype` | `links/<delta>/` `.zattrs` declares `dtype` as an integer type (`int32`/`int64`) | Warning if `int64` |
| `links_link_width` | `links/<delta>/` `.zattrs` declares `link_width` ≥ 2 (2 for graph/poly/skeleton edges; 3 for triangle faces; 4 for quad faces) | Error |
| `links_level_delta` | `links/<delta>/` `.zattrs` declares `level_delta` matching the path segment | Error |
| `obj_index_meta` | `object_index/` `.zattrs` declares `zv_array == "object_index"`, `num_objects` (int ≥ 0), and `sid_ndim` (positive int) | Error |
| `obj_index_offsets_len` | `object_index/offsets` has length equal to `num_objects` and dtype `int64` | Error |
| `ccl_meta` | `cross_chunk_links/<delta>/` `.zattrs` declares `sid_ndim`, `level_delta`, `link_width`, and `layout == "sharded_v1"` (0.8+).  `num_links` is no longer at the group level — per-leaf counts are derived from leaf byte length. | Error |
| `ccl_layout` | Every `cross_chunk_links/<delta>/` group's `.zattrs` carries `layout == "sharded_v1"`.  Any other value (or absent key, including the 0.7 legacy single-blob layout) is fatal and directs the user to run the 0.7 → 0.8 migration helper. | Error |
| `ccl_leaf_record_size` | Every `cross_chunk_links/<delta>/kK` cell has byte length divisible by `9 * link_width`. | Error |
| `ccl_attr_layout` | Every `cross_chunk_link_attributes/<name>/<delta>/` group `.zattrs` carries `layout == "sharded_v1"` (0.8+). | Error |
| `ccl_attr_parity` | For every attribute leaf at `cross_chunk_link_attributes/<name>/<delta>/<…>/data`, record count equals the parallel `cross_chunk_links/<delta>/<same path>/data` leaf record count. | Error |
| `partitioned_ccl_capability` | If any `cross_chunk_links/<delta>/` group exists, root metadata `format_capabilities` carries both `multiscale_links` and `partitioned_cross_chunk_links`. | Error |

### Multiscale metadata checks

| Check | Rule | Failure type |
|-------|------|--------------|
| `coord_transforms_present` | Each level entry has `coordinateTransformations` | Error |
| `scale_translation_pair` | Each level has exactly one `scale` and one `translation` transform | Error |
| `scale_values` | `scale[d] ≈ bin_ratio[d]` for all `d` | Error |
| `translation_values` | `translation[d] ≈ bin_shape[d] / 2` for all `d` | Error |
| `axes_length` | `len(axes) == spatial_dims` | Error |
| `axes_type` | Each axis has `type` in `{"space", "time"}` | Warning |

### Streamline-specific checks

| Check | Rule | Failure type |
|-------|------|--------------|
| `step_size_positive` | `step_size` > 0 if present | Error |
| `step_size_unit_valid` | `step_size_unit` in known unit vocabulary if present | Warning |

### Example L2 report

```
Level 2 validation of tracts.zarrvectors
==========================================
PASS  version_present           zarr_vectors_version present
PASS  geometry_type_valid       geometry_type = "streamline"
PASS  spatial_dims_type         spatial_dims = 3
PASS  divisibility [d=0]        200.0 % 50.0 = 0.0 ✓
PASS  divisibility [d=1]        200.0 % 50.0 = 0.0 ✓
PASS  divisibility [d=2]        200.0 % 50.0 = 0.0 ✓
PASS  level_0_present           multiscales contains level 0
PASS  bin_ratio_consistent [1]  bin_shape [100,100,100] = base [50,50,50] × ratio [2,2,2] ✓
PASS  scale_values [level=1]    scale [2.0,2.0,2.0] = bin_ratio [2,2,2] ✓
PASS  translation_values [1]    translation [50,50,50] = bin_shape/2 [50,50,50] ✓
PASS  vertex_fragments_dtype [0] zv_array = "vertex_fragments", encoding = "fragment_index_v1" ✓
WARN  step_size_unit_valid       step_size_unit "voxels" not in recommended vocabulary
PASS  sparsity_range [level=1]  0.5 in (0, 1] ✓

Level 2 validation: PASS — 14 passed, 1 warning, 0 errors
```
