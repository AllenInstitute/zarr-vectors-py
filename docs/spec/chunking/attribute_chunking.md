# Attribute-based chunking

## Terms

**Attribute chunking**
: A chunking mode where a per-vertex categorical *attribute* (e.g. a gene
  label, a tract bundle label, a cell-type code) becomes the **leading
  axis** of the chunk key. Spatial coordinates follow as trailing axes.
  Reading all vertices for one attribute value becomes a prefix scan
  over chunk keys instead of a full-store scan.

**Leading chunk axis**
: The first dimension of a chunk key. For a spatial-only ZVF store the
  leading axis is the first spatial dim; for an attribute-chunked store
  the leading axis is the attribute bin index. Chunk keys go from
  `z.y.x` to `attr_bin.z.y.x`.

**Attribute bin**
: A dense integer index in `[0, K)` assigned to each unique value of the
  chunk-by attribute. Bin order is the sorted order returned by
  `numpy.unique`. The mapping bin → original value is stored in level
  metadata.

**Per-vertex grouping**
: When the chunk-by attribute differs across vertices of the *same
  object*, the object is split across the corresponding attribute
  chunks. A polyline whose first half is `bundle="A"` and second half
  is `bundle="B"` appears in two attribute chunks; the existing
  `cross_chunk_links` array stitches the segments back together at read
  time.

**Per-object uniformity**
: A restricted form of attribute chunking where all vertices of an
  object are required to share the same attribute value. Used for
  geometry types where per-vertex splitting would fragment the topology
  (mesh faces, graph edges, line endpoints). The writer raises if the
  attribute is mixed within any object.

---

## Introduction

ZVF chunks data spatially by default — points, vertices, faces, edges
are bucketed by their position in a fixed-size 3D grid. This is correct
for *spatial-locality* queries ("give me everything in this bounding
box") but wrong for *categorical* queries ("give me everything for gene
X"). Categorical queries against a spatially-chunked store have to scan
every spatial chunk that touches the gene's territory, which for
20 000-gene spatial transcriptomics is most of them.

Attribute chunking re-arranges the on-disk layout so that data sharing
an attribute value is **physically contiguous**: one disk prefix per
gene, one prefix per bundle label. A categorical query becomes a single
sub-tree listing followed by O(matching-chunks) reads.

The format change is small: chunk keys gain a leading dim and the level
metadata records the bin-to-value mapping. Existing readers that
iterate chunk keys via `list_chunk_keys` work unchanged on
attribute-chunked stores because the chunk-key parser is arity-agnostic
(`_parse_chunk_key` in `zarr_vectors/core/arrays.py`).

---

## Technical reference

### Chunk key layout

When attribute chunking is in use, chunk keys for every per-vertex array
(`vertices/`, `attributes/<name>/`, etc.) take the form:

```
<attr_bin>.<spatial_0>.<spatial_1>.<spatial_2>
```

`attr_bin` is an integer in `[0, K)`. Spatial coordinates follow in the
existing dot-separated order. Object index entries store the same
extended tuple — `sid_ndim` on the `object_index` `.zattrs` is `ndim +
1` for attribute-chunked stores.

`fragment_attributes/<name>/` follows the **same** extended key scheme:
each per-chunk blob is keyed by the same combined coords the type-writer
uses for `vertices/`. Callers passing `fragment_attributes={name:
{chunk_coords: ndarray}}` to a writer must key by the post-prefix
combined coords (the leading `attr_bin` plus the spatial coords), not
the raw spatial coords.

### Level metadata fields

Three new keys on the level `.zattrs`:

| Field                     | Type       | Meaning                                                                                   |
|---------------------------|------------|-------------------------------------------------------------------------------------------|
| `chunk_dims`              | `list[str]` | Names of chunk-key axes, leading axis first. e.g. `["gene", "x", "y", "z"]`.            |
| `chunk_attribute_name`    | `str`      | Name of the per-vertex attribute used as the leading axis.                                |
| `chunk_attribute_values`  | `list`     | Ordered list mapping `attr_bin` → original attribute value. `[i]` is the value for bin `i`. |

When `chunk_dims` is absent (legacy stores), readers default to a
spatial-only layout with `sid_ndim` axes.

The chunk-by attribute is **not** also stored as a per-chunk attribute
array; its value is implicit in the leading axis and recoverable from
`chunk_attribute_values`.

### Categorical-only constraint (v1)

Attribute chunking is **categorical only** in v1:

- Allowed dtypes: integer, boolean, string (Unicode or bytes).
- Floating-point attributes are rejected with a helpful error. For
  continuous values, bin externally and pass the bin index, or use the
  rechunk module's bin-edge support.
- Every unique value becomes its own bin, regardless of cardinality.
  20 000 unique genes produce 20 000 chunk prefixes.

### Per-vertex vs per-object semantics

| Type        | Semantics      | Behaviour on mixed-value objects                                |
|-------------|----------------|-----------------------------------------------------------------|
| points      | per-vertex     | Each point is its own object; no mixing possible.               |
| polylines   | per-vertex     | Polylines with mixed values are split; cross-chunk links bridge. |
| lines       | per-object     | Both endpoints of a line must share the same `line_attributes` value. |
| graphs      | per-object     | All nodes of one object must share the same value.              |
| meshes      | per-object     | All vertices of one mesh object must share the same value.      |

The per-object types raise `ArrayError` when the constraint is violated
— they do not silently drop or average values.

### Write API

Every writer that has per-vertex (or per-object) attributes accepts a
`chunk_by_attribute: str | None = None` kwarg:

```python
from zarr_vectors.types.points import write_points

write_points(
    "genes.zarrvectors",
    positions,
    chunk_shape=(50.0, 50.0, 50.0),
    attributes={"gene": gene_labels},
    chunk_by_attribute="gene",
)
```

The named attribute must appear in the writer's per-vertex / per-object
dict (`attributes`, `vertex_attributes`, `node_attributes`, or
`line_attributes` depending on the type).

### Read API

Eager readers (`read_points`, `read_polylines`, etc.) accept an
`attribute_filter: dict[str, Any] | None` kwarg:

```python
read_points("genes.zarrvectors", attribute_filter={"gene": "Gad1"})
```

The filter is a `{name: value}` dict with exactly one entry. The name
must match the store's `chunk_attribute_name`. The value is resolved to
a bin index via the stored `chunk_attribute_values` list, and the chunk
scan is restricted to keys with that leading coord. Unknown values
yield an empty result rather than an error.

Lazy readers (`ZVLevel`) expose:

- `chunk_dims` — the level's chunk-axis names, or `None` for legacy.
- `chunk_attribute_name` — the leading-axis attribute name.
- `attribute_values` — the ordered value list.
- `read_attribute_chunk(value)` — convenience: reads all fragments
  for chunks whose leading coord matches `value`.

### Post-hoc rechunking

For stores that already have a per-object attribute and need to be
re-laid out, `rechunk_by_attribute` is the ergonomic wrapper:

```python
from zarr_vectors import rechunk_by_attribute

rechunk_by_attribute(
    "tracts.zarrvectors",
    "bundle_label",
    output="tracts-by-bundle.zarrvectors",
)
```

It builds a `RechunkSpec(by="attribute:bundle_label", categorical=True,
prefix_dim_name="bundle_label")` and runs the standard rechunk engine.
`categorical=True` disables the legacy quartile fallback in
`DimensionMapper._map_by_attribute`, which would otherwise collapse
high-cardinality attributes (>10 unique values) to 4 quartile bins.

### Atomicity and writes

Attribute-chunked writes inherit the same atomicity guarantees as
spatial writes: each chunk is one file (or one object on a cloud
store), and per-chunk writes are atomic. A failed write can leave a
partially-populated store — there is no transactional rollback in v1.

### Compatibility notes

- A reader that ignores `chunk_dims` and treats the leading axis as
  spatial will read garbage. Always inspect `chunk_dims` before
  interpreting chunk keys.
- Bin order is fixed by `numpy.unique` at write time. Adding new
  attribute values to an existing attribute-chunked store requires a
  rechunk (the existing bin numbering does not accommodate inserts).
- `chunk_attribute_values` is the authoritative bin → value map. Do
  not infer it from chunk-key prefixes alone.
