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
: An optional codec applied to mesh geometry (`vertices/` and `links/<delta>/`)
  that can achieve 6–15× compression over uncompressed float32 data.
  Requires `zarr-vectors[draco]`.

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
spatial extent and the faces whose vertices all lie within it; a face
whose vertices span chunks is stored once, as a boundary face.

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
| `vertices/` | Yes | Vertex positions, shape `(N, D)` float32 per chunk |
| `vertex_fragments/` | Yes | Fragment index over `vertices/` rows |
| `links/0/0.0.0/` | Raw: yes | Faces whose vertices share a chunk. **Draco: absent** — see below |
| `links/0/<offsets>/` | When a face straddles chunks | Boundary faces; offsets name the other vertices' chunks |
| `link_fragments/` | Yes (with `links/0/0.0.0/`) | Fragment index over the intra array's rows |
| `object_index/` | Yes | Per-object manifest blobs naming fragments |
| `attributes/<name>/` | No | Per-vertex attributes (normals, UVs, colours) |
| `object_attributes/<name>/` | No | Per-mesh attributes (volume, surface area) |

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
3`; otherwise the raw path is used. Draco removes only the *intra-chunk*
faces from the family — boundary faces have no chunk whose bitstream
could hold them, so they stay in `links/`. A Draco mesh with no
boundary-crossing face therefore has an **empty** link family;
`write_meshes` still calls `create_links_array` afterwards so the family
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

### Root `.zattrs` type-specific keys

```json
{
  "geometry_type":    "mesh",
  "links_convention": "explicit"
}
```

`mesh` declares no type-specific root keys of its own.

> **`winding_order` and `closed_surface` are not root metadata keys.**
> Neither is written, read, or validated by any shipped code, and
> `write_mesh` accepts neither as an argument. Winding is preserved
> per-face by `perm_idx` (see *Face policy and winding* above), not by a
> store-wide declaration; Zarr Vectors' convention is that a face's **input**
> vertex order is authoritative and is recovered exactly on read.
> Watertightness is not declared or checked.

### Face encoding and boundary faces

**Intra-chunk faces:** all vertices in one chunk. Stored in
`links/0/<all-zero offsets>/`, each vertex index local to that chunk's
vertex rows.

**Boundary faces:** vertices in more than one chunk. Filed by
[`write_links`](../object_model/links.md) under the offsets array that
names where the other vertices sit, with each endpoint's index local to
its own chunk; the canonical sort's permutation is kept in `perm_idx`
(see *Face policy and winding* above). There is no global vertex ID and
no sentinel encoding.

### Per-object face groups

`write_mesh` lays each chunk's vertices out object by object, one vertex
fragment per object, and cuts the chunk's intra-chunk faces into one link
group per fragment, in the same order. It then stamps
`fragment_link_groups` on the level (and `CAP_FRAGMENT_LINK_GROUPS` on the
root) after verifying it -- see [Links](../object_model/links.md#link-groups-that-follow-vertex-fragments).
An object's manifest names its vertex fragments; on such a level the same
indices name its face groups in `link_fragments/`, so a reader drawing one
object can range-read its faces instead of every object's in the chunk.
Boundary faces are not grouped; they stay in the offsets arrays.

Draco levels are not stamped: their intra-chunk faces live in the Draco
bitstream, not in the link family.

### Draco compression

Pass `encoding="draco"` (and optionally `draco_quantization_bits`) to
`write_mesh()`. Each chunk's vertices and intra-chunk faces are then one
Draco bitstream; boundary faces stay in `links/0/<offsets>/` (see
*Face storage and Draco* above).

Reading requires `zarr-vectors[draco]`. See
[Codec pipeline](../foundations/codec_pipeline.md) for quantisation
precision details.

Draco compression is applied per-chunk. The Draco codec receives the
combined (vertices, faces) data for one chunk and compresses them jointly,
exploiting vertex-face correlations for additional compression beyond what
independent array compression achieves.

### Write API

```python
import numpy as np
from zarr_vectors.types.meshes import write_mesh

write_mesh(
    "brain.zarrvectors",
    vertices=vertices,   # (N, 3) float32
    faces=faces,         # (F, 3) int32 — global vertex indices
    chunk_shape=(100.0, 100.0, 100.0),
    bin_shape=(25.0, 25.0, 25.0),
)
```

### Ingest and export

OBJ, STL, and PLY converters live in the companion package
**`zarr-vectors-tools`**.

### Read API

```python
from zarr_vectors.types.meshes import read_mesh

result = read_mesh("brain.zarrvectors")
print(result["vertex_count"])    # int
print(result["face_count"])      # int
print(result["vertices"].shape)  # (N, 3)
print(result["faces"].shape)     # (F, 3) global vertex indices

# Spatial query — reads the chunks the bbox touches
result = read_mesh(
    "brain.zarrvectors",
    bbox=(np.array([0., 0., 0.]), np.array([500., 500., 500.])),
)
```

### Multi-mesh stores

A single `mesh` store may contain many distinct mesh objects (e.g. one per
cell or organelle), each with its own manifest. `read_mesh` does not filter
by object yet (`object_ids=` raises `NotImplementedError` rather than
silently returning the whole level); `read_object_vertices` reads one
object's vertices, and on a level stamped `fragment_link_groups` its faces
are the link groups its fragments name (see *Per-object face groups*).

### Validation

L1: `vertices/` exists at every level. `links/` and `object_index/` are
recorded when present but are **not** required at L1.

L3: offsets segments parse; the family being undirected, canonical and
intra-level, offsets are lex-non-negative and non-decreasing; every
record's endpoint chunks exist at the level; a level stamped
`fragment_link_groups` keeps its link groups one per vertex fragment.

L4 (mesh-specific): the `links/0/` family's `link_width` MUST be `>= 3`
— a `link_width` between 1 and 2 is an error. A mesh store with no link
metadata at all emits a warning, not an error (this is the Draco
no-boundary-face case). `links_convention` MUST be `explicit`.

**Not checked at any level:** face vertex indices within a chunk,
degenerate faces, watertightness, boundary edges, or winding
consistency. There is no `closed_surface` key in the shipped code. See
[Validation overview](../validation/overview.md).
