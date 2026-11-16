# Mesh (`mesh`)

## Terms

**Mesh**
: A piecewise-linear surface represented by a set of vertices and a set of
  triangular faces. Each face is a triplet of vertex indices. Zarr Vectors stores
  closed or open surface meshes; there is no requirement of watertightness.

**`GEOM_MESH`**
: The geometry type constant `"mesh"`.

**`links/<delta>/<offsets>/`**
: The link family holding faces. `link_width` is the face arity —
  taken from `faces.shape[1]`, so `3` for triangles and `4` for quads —
  and MUST be `>= 3`. Each record is `link_width` endpoints; endpoint
  `k`'s vertex index is local to chunk `src + o_k`.

**Winding order**
: The orientation convention for face normals — conventionally
  counter-clockwise when viewed from outside the surface, implying
  outward-facing normals. Zarr Vectors does **not** declare this in metadata; it
  preserves each face's input vertex order verbatim via `perm_idx`, so
  whatever winding the producer used survives a round-trip.

**Draco compression**
: An optional lossy encoding of each chunk's vertices and intra-chunk faces
  (`encoding="draco"`). Requires `zarr-vectors[draco]`.

**Boundary face**
: A face whose vertices do not all lie in one chunk. There is no special
  mechanism for it: a boundary face is an ordinary record in the one
  link family, filed under the `<offsets>` naming where its other
  vertices' chunks sit relative to its source chunk. An intra-chunk face
  is the same record with all-zero offsets.

---

## Introduction

The `mesh` type stores triangulated surface meshes in the Zarr Vectors spatial
chunking framework. Like other geometry types, the mesh is partitioned
into spatial chunks; each chunk holds the vertices that fall within its
spatial extent and the faces whose centroid falls within that extent.

Mesh chunking introduces a subtlety that does not arise for point clouds
or streamlines: a face may reference vertices in multiple chunks (the face
straddles a chunk boundary). Zarr Vectors handles this with the same link family
it uses for intra-chunk faces — the face is filed under the offsets
naming where its other vertices sit, and each endpoint's index stays
local to its own chunk. The `object_index/` maps each mesh object
(distinct connected surface) to its constituent chunks.

---

## Technical reference

### Arrays present

| Array path | Required | Description |
|-----------|----------|-------------|
| `vertices/` | Yes | Vertex positions, shape `(N, D)` per chunk (`float32` by default). Under Draco, a Draco bitstream per chunk |
| `vertex_fragments/` | Yes | Fragment index over `vertices/` rows |
| `links/0/0.0.0/` | Raw: yes | Faces whose vertices share a chunk. **Draco: absent** — see below |
| `links/0/<offsets>/` | When a face straddles chunks | Boundary faces; offsets name the other vertices' chunks |
| `link_fragments/` | Yes (with `links/0/0.0.0/`) | Fragment index over the intra array's rows |
| `object_index/` | Yes | Per-object manifest blobs naming fragments |
| `vertex_attributes/<name>/` | No | Per-vertex attributes (normals, UVs, colours) |
| `object_attributes/<name>/` | No | Per-mesh attributes (volume, surface area) |
| `link_attributes/<name>/0/<offsets>/` | No | Per-face attributes — not accepted by `write_mesh` (see Write API) |

There is no `cross_chunk_links/` array and no global-vertex-ID
mechanism: boundary faces are records with non-zero offsets in the same
family.

### Face storage and Draco

Which faces reach the link family depends on `encoding`:

| `encoding` | Intra-chunk faces | Boundary faces |
|------------|-------------------|----------------|
| `"raw"` (default) | `links/0/0.0.0/` | `links/0/<offsets>/` |
| `"draco"` (3-D only) | Embedded in the chunk's Draco bitstream, in chunk-local indices | `links/0/<offsets>/` |

Draco mode applies only when `encoding == "draco"` **and** `sid_ndim ==
3`; otherwise the raw path is used (silently — no error for a
non-3-D store). In Draco mode each chunk is a single fragment, so every
object present in that chunk names fragment 0. Draco removes only the *intra-chunk*
faces from the family — boundary faces have no chunk whose bitstream
could hold them, so they stay in `links/`. A Draco mesh with no
boundary-crossing face therefore has an **empty** link family;
`write_mesh` still calls `create_links_array` afterwards so the family
is advertised in `arrays_present` for the per-cell editors in `ops/`.

### Face policy and winding

The family is written **undirected** with `store="canonical"`, so
`write_links` canonical-sorts each face's endpoints and stores it once.
Winding is not lost: `perm_idx` records the permutation the sort
applied, and `apply_perm_inverse` recovers the original vertex order —
and therefore the face normal — on read. This is why the family can be
undirected despite winding being significant.

Records stay in input face order within each cell, so a parallel
`link_attributes/<name>/0/<offsets>/` array stays row-aligned.

### Root metadata

```json
{
  "zarr_vectors": {
    "geometry_types":   ["mesh"],
    "links_convention": "explicit"
  }
}
```

(Alongside the usual `zarr_vectors` keys; see
[Root metadata](../layout/root_metadata.md).) `mesh` declares no
type-specific keys of its own.

> **`winding_order` and `closed_surface` are not root metadata keys.**
> Neither is written, read, or validated by any shipped code, and
> `write_mesh` accepts neither as an argument. Winding is preserved
> per-face by `perm_idx` (see *Face policy and winding* above), not by a
> store-wide declaration; Zarr Vectors' convention is that a face's **input**
> vertex order is authoritative and is recovered exactly on read.
> Watertightness is not declared or checked.

### Face encoding and boundary faces

Every face is a record of `L` endpoints `(chunk_coords, local_vertex_index)`.
Each vertex lives in the chunk its own position falls in, and its
index is local to that chunk's vertex buffer.

**Intra-chunk faces:** all `L` vertices share a chunk. The record lands
in the all-zero-offsets array `links/0/0.0.0/` (raw encoding).

**Boundary faces:** the vertices span chunks. The record is canonical-sorted
by `(chunk_coords, vertex_index)`, filed under the chunk of its
lexicographically smallest endpoint (the *source*), in the offsets array
naming where the other endpoints' chunks sit relative to it. There are no
negative sentinel indices and no global vertex IDs.

`faces` passed to `write_mesh` use **global** vertex indices into the
`vertices` array you supply; `read_mesh` returns faces remapped to the
output vertex order.

### Draco compression

Pass `encoding="draco"` (and optionally `draco_quantization_bits=`,
default `11`) to `write_mesh`. The vertices and intra-chunk faces of each
chunk are then encoded jointly as one Draco bitstream. Draco applies only
when the store is 3-D, and requires the optional dependency
`zarr-vectors[draco]` (DracoPy). There is no `use_draco=` or
`draco_quantization=` keyword.

Draco compression is lossy (quantised). See
[Codec pipeline](../foundations/codec_pipeline.md).

### Write API

```python
import numpy as np
from zarr_vectors.types.meshes import write_mesh

# A small unit tetrahedron scaled to 100 units
vertices = 100.0 * np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
)
faces = np.array(
    [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64
)  # (F, 3) global vertex indices

summary = write_mesh(
    "brain.zarrvectors",
    vertices,
    faces,
    chunk_shape=(50.0, 50.0, 50.0),
    bin_shape=(25.0, 25.0, 25.0),
)
# summary: {"vertex_count", "face_count", "chunk_count",
#           "intra_face_count", "cross_face_count", "encoding"}
```

`chunk_shape` is required. `faces` is `(F, L)` with `L >= 3`; quads (`L=4`)
are accepted and set `link_width`. Other keywords: `bounds`, `encoding`
(`"raw"` default, or `"draco"`), `vertex_attributes`
(`{name: (V,)}` or `(V, C)`), `object_attributes`, `object_ids`
(`(V,)` **per vertex**, assigning vertices to mesh objects; default all 0),
`dtype`, `draco_quantization_bits`, `backend`, `chunk_by_attribute`,
`out_of_bounds` (`"ignore"` is rejected), `compressor`, `shard_shape`.
There is no per-face (link) attribute argument.

Through the dataset API: `ds.add_mesh(vertices, faces, ...)`.

### Ingest and export

OBJ, STL, and PLY converters live in the companion package
**`zarr-vectors-tools`**.

### Read API

```python
import numpy as np
from zarr_vectors.types.meshes import read_mesh

result = read_mesh("brain.zarrvectors")
print(result["vertex_count"])    # int
print(result["face_count"])      # int
print(result["vertices"].shape)  # (V, D)
print(result["faces"].shape)     # (F, L) indices into result["vertices"]

# Spatial query: bbox is applied to vertices; faces with a dropped
# vertex are dropped
result = read_mesh(
    "brain.zarrvectors",
    bbox=(np.array([0., 0., 0.]), np.array([500., 500., 500.])),
)
```

Other keywords: `level`, `chunks`, `attribute_filter`, `backend`.
**`read_mesh(object_ids=...)` raises `NotImplementedError`**: the filter is
not implemented, and rather than silently returning the whole level the
reader refuses. The dataset API likewise refuses `select(objects=...)`
for meshes.

### Multi-mesh stores

A single `mesh` store may contain many distinct mesh objects (e.g. one per
cell or organelle): pass `object_ids` (one entry per vertex) to
`write_mesh`. Objects are recorded in `object_index/`, but there is
currently no object-scoped mesh read; read the level and filter the
arrays.

### Validation

L1: `vertices/` exists at every level. `links/` and `object_index/` are
recorded when present but are **not** required at L1.

L3: offsets segments parse; the family being undirected, canonical and
intra-level, offsets are lex-non-negative and non-decreasing; every
record's endpoint chunks exist at the level.

L4 (mesh-specific): the `links/0/` family's `link_width` MUST be `>= 3`
— a `link_width` between 1 and 2 is an error. A mesh store with no link
metadata at all emits a warning, not an error (this is the Draco
no-boundary-face case). `links_convention` MUST be `explicit`.

**Not checked at any level:** face vertex indices within a chunk,
degenerate faces, watertightness, boundary edges, or winding
consistency. There is no `closed_surface` key in the shipped code. See
[Validation overview](../validation/overview.md).
