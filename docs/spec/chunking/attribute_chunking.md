# Attribute-based chunking

## Terms

**Attribute chunking**
: A layout in which a categorical attribute becomes the leading axis of
  every chunk key, before the spatial axes. Chunk keys change from
  `i.j.k` to `b.i.j.k`.

**Attribute bin**
: A dense integer in `[0, K)` for one value of the chunk-by attribute,
  where `K` is the number of distinct values.

**Bin label**
: The value an attribute bin stands for, recorded in
  `chunk_attribute_values`.

---

## Introduction

By default a store is chunked by position only, which suits box queries
but not queries by category: reading every point of one gene from a
spatial-transcriptomics store means reading every chunk the gene's
points fall in, mixed with all other genes.

With attribute chunking, data sharing a value is stored together under
one key prefix. A query for one value reads only the chunks with that
leading coordinate. A level can be written this way directly
(`chunk_by_attribute=`) or rechunked into it afterwards
([`rechunk_by_attribute`](rechunking.md#grouping-objects-under-a-leading-axis)).

---

## Technical reference

### Chunk keys

On an attribute-chunked level every per-chunk array is keyed
`<bin>.<i>.<j>.<k>`. Object manifests carry the same extended
coordinates, so the object index has rank `1 + D`.

`fragment_attributes` passed to a writer must be keyed by these combined
coordinates, the bin first, not by spatial coordinates alone.

The chunk-by attribute is not also stored as a vertex attribute array;
`chunk_attribute_values` and the leading coordinate give each vertex's
value.

### Array shape

Every per-chunk array of an attribute-chunked level has rank `1 + D`,
with a leading axis of length `K` starting at 0:

```
vertices                  shape = (K, g_0, g_1, g_2)
vertex_fragments          shape = (K, g_0, g_1, g_2)
vertex_attributes/<name>  shape = (K, g_0, g_1, g_2)
links/<delta>/<offsets>   shape = (K, g_0, g_1, g_2)
```

`K` is `len(chunk_attribute_values)`, which writers pass as `bin_count`.
`chunk_dims` names the axes but does not give their sizes. Any code that
allocates an array in an existing level (a worker calling
`create_attribute_array`, an edit adding a links segment) must take the
rank and leading extent from the level; an array one axis short cannot be
addressed by the level's keys.

### Level metadata

The level's `zarr_vectors_level` block gains three keys. After writing
3,000 points labelled with three genes:

```json
{
  "zarr_vectors_level": {
    "chunk_dims": ["gene", "x", "y", "z"],
    "chunk_attribute_name": "gene",
    "chunk_attribute_values": ["Gad1", "Pvalb", "Slc17a7"]
  }
}
```

| Key | Type | Meaning |
|-----|------|---------|
| `chunk_dims` | `list[str]` | Names of the chunk-key axes, leading axis first |
| `chunk_attribute_name` | `str` | The attribute that `attribute_filter` names |
| `chunk_attribute_values` | `list` | Bin labels, in bin order: entry `i` is the value of bin `i` |

A level without `chunk_dims` is chunked by position only. A reader must
check `chunk_dims` before interpreting chunk keys, and must take bin
labels from `chunk_attribute_values`, not from key prefixes.

### Writing

The writers `write_points`, `write_polylines`, `write_lines`,
`write_graph` and `write_mesh` take `chunk_by_attribute: str | None`. It
names an entry of the writer's `vertex_attributes` (for
`write_lines`, also of `object_attributes`):

```python
import numpy as np
from zarr_vectors.types.points import write_points

rng = np.random.default_rng(0)
positions = rng.uniform(0, 200, (3000, 3)).astype("float32")
gene = rng.choice(np.array(["Gad1", "Slc17a7", "Pvalb"]), 3000)

write_points("genes.zarrvectors", positions, chunk_shape=(50., 50., 50.),
             vertex_attributes={"gene": gene}, chunk_by_attribute="gene")
```

The `vertices` array then has shape `[3, 4, 4, 4]`.

The attribute must be categorical: integer, boolean or string. A
floating-point attribute raises `ArrayError`; bin continuous values
yourself, or use `rechunk` with bin edges. Each distinct value becomes
one bin, in the sorted order of `numpy.unique`, however many there are.
Adding a value to a written level means rechunking it.

How each type treats an object whose vertices have different values:

| Type | Behaviour |
|------|-----------|
| Points | Each point is its own object |
| Polylines | The polyline is split between bins; links join the pieces, so it reads back whole |
| Lines | Both endpoints of a line must share a value, or `ArrayError` |
| Graphs | All vertices of one object must share a value, or `ArrayError`. Without `object_ids`, each vertex is its own object |
| Meshes | All vertices of one object must share a value, or `ArrayError` |

### Reading

The eager readers (`read_points`, `read_polylines`, and the others) take
`attribute_filter: dict[str, Any] | None`:

```python
from zarr_vectors.types.points import read_points

read_points("genes.zarrvectors", attribute_filter={"gene": "Gad1"})["vertex_count"]
# 1034
```

The filter must have exactly one entry, and its key must be the level's
`chunk_attribute_name`; otherwise it raises `ArrayError`. The value is
looked up in `chunk_attribute_values` and only chunks with that bin are
read. A value not in the list returns an empty result. A box query on an
attribute-chunked level reads every bin of the chunks the box touches.

The lazy level object (`ZVLevel`, from `zarr_vectors.lazy.open_zv`) has
`chunk_dims`, `chunk_attribute_name`, `attribute_values` and
`read_attribute_chunk(value)`, which returns the fragments of every chunk
in that value's bin.

### Bin labels from a rechunk

`rechunk` numbers its bins densely from 0 in the order below, and
allocates no bin that no object falls in. The label is the value
`attribute_filter` names to select the bin:

| `RechunkSpec` | Bin order | Label |
|---------------|-----------|-------|
| `by="attribute:<name>"`, categorical | Ascending value | The value |
| `by="attribute:<name>"` or `by="object_id"`, with `bins` | Ascending edge | The bin's lower edge |
| `by="attribute:<name>"`, more than 10 values and no `bins` | Ascending quartile edge | The bin's lower edge |
| `by="object_id"`, no `bins` | Ascending id | The object id |
| `by="group"` | Ascending group index, then ungrouped objects | The group index; `-1` for ungrouped |

A bin cut by edges spans `[edges[i], edges[i + 1])`. Values below the
first edge fall in the first bin and the last bin is open above, so
`bins=[0, 30, 80, inf]` puts a value of 45 in the bin labelled `30`.
`chunk_attribute_name` is the attribute's name, or `group`, `object_id`
or `prefix_dim_name` for the other kinds. `by="spatial"` records no
labels.

### Pyramids

`build_pyramid` and `coarsen_level` keep the bins at every coarser level:
the coarse level has the same leading axis and copies `chunk_dims`,
`chunk_attribute_name` and `chunk_attribute_values`, so
`attribute_filter` works on any level. Each coarse vertex aggregates
vertices of one attribute bin and one spatial bin, so a polyline that
changes value within a spatial bin gives one coarse vertex per value.

Links between levels (`links/+N`, `links/-N`) have rank `1 + D` at both
ends. A link never crosses attribute bins, so the bin component of its
offset is always 0 (segments such as `0.0.0.+1`).

A level with a leading axis but no `chunk_attribute_values` is carried
the same way; the coarse level takes the axis length from the fine
level's `vertices` shape.

### Validation

[`zarr_vectors.validate`](../validation/overview.md) has no checks
specific to attribute chunking. Level 3 reads every chunk key of any
rank, and the per-bin point checks of [Bin shape](bin_shape.md#validation)
apply only to point-cloud levels without an object index, which an
attribute-chunked point level always has.
