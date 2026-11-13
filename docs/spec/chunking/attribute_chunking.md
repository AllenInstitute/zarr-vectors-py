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

With attribute chunking, data sharing a value is stored under one key
prefix, so a query for one value (one gene, say) reads only the chunks
with that leading coordinate, not every chunk the value appears in. Write a level this way with
`chunk_by_attribute=`, or rechunk into it with
[`rechunk_by_attribute`](rechunking.md#grouping-objects-under-a-leading-axis).

---

## Technical reference

### Array shape

Every per-chunk array of an attribute-chunked level (`vertices`,
`vertex_fragments`, `vertex_attributes/<name>`, `links/<delta>/<offsets>`)
is keyed `<bin>.<i>.<j>.<k>` and has shape `(K, g_0, g_1, g_2)`: a
leading axis of length `K` before the spatial grid.

`K` is `len(chunk_attribute_values)`, which writers pass as `bin_count`.
Code that adds an array to an existing level must take the rank and
leading extent from the level; `chunk_dims` gives names, not sizes.

Object manifests use the same extended coordinates, so the object index
has rank `1 + D`. `fragment_attributes` passed to a writer are keyed by
these coordinates, bin first. The chunk-by attribute is not stored as a
vertex attribute array: the leading coordinate and
`chunk_attribute_values` give each vertex's value.

### Level metadata

After writing 3,000 points labelled with three genes, the level's
`zarr_vectors_level` block includes:

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
| `chunk_attribute_values` | `list` | Bin labels in bin order: entry `i` is the value of bin `i` |

A level without `chunk_dims` is chunked by position only. Readers must
check `chunk_dims` before interpreting chunk keys.

### Writing

`write_points`, `write_polylines`, `write_lines`, `write_graph` and
`write_mesh` take `chunk_by_attribute: str | None`, naming an entry of
`vertex_attributes` (for `write_lines`, also of `object_attributes`):

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

The attribute must be integer, boolean or string; a float raises
`ArrayError`. Each distinct value becomes one bin, in `numpy.unique`
order. An object whose vertices have different values is handled per
type:

| Type | Behaviour |
|------|-----------|
| Points | Each point is its own object |
| Polylines | Split between bins; links join the pieces, so it reads back whole |
| Lines | Both endpoints must share a value, or `ArrayError` |
| Graphs | All vertices of one object must share a value, or `ArrayError`. Without `object_ids`, each vertex is its own object |
| Meshes | All vertices of one object must share a value, or `ArrayError` |

### Reading

The eager readers (`read_points`, `read_polylines` and the others) take
`attribute_filter: dict[str, Any] | None`:

```python
from zarr_vectors.types.points import read_points

read_points("genes.zarrvectors", attribute_filter={"gene": "Gad1"})["vertex_count"]
# 1034
```

The filter must have exactly one entry, keyed by the level's
`chunk_attribute_name`, or it raises `ArrayError`. Only chunks of the
matching bin are read. A value not in `chunk_attribute_values` returns
an empty result. A box query reads every bin of the chunks it touches.

The lazy `ZVLevel` (`zarr_vectors.lazy.open_zv`) also has
`read_attribute_chunk(value)`.

### Bin labels from a rechunk

`rechunk` numbers its bins densely from 0 and allocates no empty bin.
The label is the value `attribute_filter` names:

| `RechunkSpec` | Bin order | Label |
|---------------|-----------|-------|
| `by="attribute:<name>"`, categorical | Ascending value | The value |
| `by="attribute:<name>"` or `by="object_id"`, with `bins` | Ascending edge | The bin's lower edge |
| `by="attribute:<name>"`, more than 10 values and no `bins` | Ascending quartile edge | The bin's lower edge |
| `by="object_id"`, no `bins` | Ascending id | The object id |
| `by="group"` | Ascending group index, then ungrouped objects | The group index; `-1` for ungrouped |

A bin cut by edges spans `[edges[i], edges[i + 1])`. Values below the
first edge fall in the first bin and the last bin is open above:
`bins=[0, 30, 80, inf]` puts 45 in the bin labelled `30`.
`chunk_attribute_name` is the attribute's name, or `group`, `object_id`
or `prefix_dim_name`. `by="spatial"` records no labels.

### Pyramids

`build_pyramid` and `coarsen_level` keep the leading axis and its
metadata at every level, so `attribute_filter` works on any level. Each
coarse vertex aggregates one attribute bin and one spatial bin. Links
between levels have rank `1 + D` and never cross attribute bins, so the
bin part of their offset is 0 (for example `0.0.0.+1`).

### Validation

[`zarr_vectors.validate`](../validation/overview.md) has no checks for
attribute chunking; level 3 reads chunk keys of any rank. The per-bin
checks of [Bin shape](bin_shape.md#validation) do not apply, since these
levels always have an object index.
