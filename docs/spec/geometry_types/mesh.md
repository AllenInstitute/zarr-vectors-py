# Mesh (`mesh`)

## Terms

**Mesh**
: A piecewise-linear surface represented by a set of vertices and a set of
  triangular faces. Each face is a triplet of vertex indices. ZVF stores
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
  outward-facing normals. ZVF does **not** declare this in metadata; it
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

The `mesh` type stores triangulated surface meshes in the ZVF spatial
chunking framework. Like other geometry types, the mesh is partitioned
into spatial chunks; each chunk holds the vertices that fall within its
spatial extent and the faces whose centroid falls within that extent.

Mesh chunking introduces a subtlety that does not arise for point clouds
or streamlines: a face may reference vertices in multiple chunks (the face
straddles a chunk boundary). ZVF handles this with the same link family
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
> store-wide declaration; ZVF's convention is that a face's **input**
> vertex order is authoritative and is recovered exactly on read.
> Watertightness is not declared or checked.

### Face encoding and boundary faces

**Intra-chunk faces:** All three vertices are in the same chunk. Vertex
indices are local to the chunk (0-indexed within the chunk's vertex slice).

**Boundary faces:** One or more vertices are in a different chunk. For
boundary faces, vertex indices that refer to other chunks use a *negative
sentinel* encoding:

- Vertices in the current chunk: positive local index `[0, N_chunk)`.
- Vertices in other chunks: stored as a negative value `-(global_vertex_id + 1)`,
  where `global_vertex_id` is the vertex's position in the global vertex
  ID space.

At read time, the reader resolves negative indices by looking up the
corresponding global vertex IDs and fetching those vertices from their
respective chunks.

**Global vertex ID** for a vertex at local index `k` in chunk `(cx, cy, cz)`:

```
global_id = chunk_flat_index * N_max + k
```

where `chunk_flat_index = ravel_multi_index((cx, cy, cz), chunk_grid_shape)`
and `N_max` is the maximum vertices per chunk declared in the Zarr array shape.

### Face assignment to chunks

Each triangular face is assigned to the chunk containing its centroid:

```python
centroid = (vertices[i] + vertices[j] + vertices[k]) / 3
face_chunk = floor(centroid / chunk_shape).astype(int)
```

This ensures each face is stored exactly once. All three vertices of a face
are guaranteed to be within at most one chunk-width of the face's chunk (faces
cannot span more than two chunks in any dimension if all vertices are within
the face's chunk neighbourhood).

### Draco compression

Pass `use_draco=True` and a `draco_quantization` value when writing a
mesh (via `write_mesh()` or the format converters in
`zarr-vectors-tools`). The `vertices/` and `links/<delta>/` arrays then
use the `draco` codec.

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

# Spatial query — returns faces whose centroid is in bbox
result = read_mesh(
    "brain.zarrvectors",
    bbox=(np.array([0., 0., 0.]), np.array([500., 500., 500.])),
)
```

### Multi-mesh stores

A single `mesh` store may contain many distinct mesh objects (e.g. one per
cell or organelle). Each object is one connected surface:

```python
result = read_mesh("cells.zarrvectors", object_ids=[42, 107])
```

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
