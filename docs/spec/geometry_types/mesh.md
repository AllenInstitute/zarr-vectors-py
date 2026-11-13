# Mesh (`mesh`)

## Terms

**Mesh**
: A piecewise-linear surface given as vertices and faces. Each face is a
  tuple of vertex indices: three for a triangle, four for a quad. Open and
  closed surfaces are both allowed.

**`GEOM_MESH`**
: The geometry type constant `"mesh"`, stored in the `geometry_types` list
  of the root attributes.

**Face record**
: A link record holding one face. `link_width` is the face size
  (`faces.shape[1]`) and is at least 3. Endpoint `k`'s vertex index is local
  to the chunk `src + o_k`.

**Boundary face**
: A face whose vertices do not all lie in one chunk. It is an ordinary face
  record filed under the non-zero offsets naming its vertices' chunks.

**Draco**
: A mesh compression format. With `encoding="draco"`, each chunk's vertices
  and intra-chunk faces are stored as one Draco bitstream. It needs the
  `draco` extra (`pip install zarr-vectors[draco]`).

## Introduction

The `mesh` type stores surface meshes on the chunk grid. Each vertex goes to
the chunk that contains it, and every face is a record in the one link family
with `links_convention: explicit`. A face whose vertices share a chunk goes
in the all-zero offsets array `links/0/0.0.0/`; a boundary face goes in the
array named by the offsets of its vertices' chunks.

The family is undirected with `store: "canonical"`, so each face is stored
once. The writer sorts each face's endpoints and records the permutation, and
the reader undoes it, so faces come back with their written vertex order and
winding. See [Links](../object_model/links.md).

Objects come from the `object_ids=` argument, one id per vertex. Without it,
every vertex belongs to object 0. Within each chunk, the writer stores one
vertex fragment per object.

## Technical reference

### Arrays present

At level 0 of a store written by `write_mesh`:

| Path | Present | Description |
|------|---------|-------------|
| `vertices/` | Always | Vertex positions; with Draco, one bitstream per chunk |
| `vertex_fragments/` | Always | Fragment index over `vertices/` rows |
| `links/0/0.0.0/` | Raw encoding | Faces whose vertices share a chunk |
| `links/0/<offsets>/` | When a face crosses chunks | Boundary faces |
| `link_fragments/` | With face records | Fragment index over the link rows |
| `object_index/` | Always | One manifest per object |
| `vertex_attributes/<name>/` | With `vertex_attributes=` | One row per vertex |
| `object_attributes/<name>/` | With `object_attributes=` | One row per object |

Which faces reach the link family depends on `encoding`:

| `encoding` | Intra-chunk faces | Boundary faces |
|------------|-------------------|----------------|
| `"raw"` (default) | `links/0/0.0.0/` | `links/0/<offsets>/` |
| `"draco"` | In the chunk's Draco bitstream, chunk-local indices | `links/0/<offsets>/` |

Draco applies only to 3-D meshes; for any other dimensionality the raw path
is used. A Draco chunk holds one fragment shared by every object in it.

### Root attributes

A mesh store has no type-specific root keys. The writer sets these values in
the `zarr_vectors` block (see [Root metadata](../layout/root_metadata.md)):

```json
{
  "geometry_types": ["mesh"],
  "links_convention": "explicit",
  "object_index_convention": "standard",
  "cross_chunk_strategy": "explicit_links"
}
```

The level's attributes carry `link_fragments_aligned: true` for raw stores:
each chunk's intra-chunk faces are filed by vertex fragment, so an object
read takes only its own faces.

### Write and read API

```python
import numpy as np
from zarr_vectors.building import write_mesh
from zarr_vectors.types.meshes import read_mesh

# A 20 x 20 grid of vertices on a wavy sheet, two triangles per grid cell
n = 20
gx, gy = np.meshgrid(np.linspace(10, 390, n), np.linspace(10, 390, n), indexing="ij")
gz = 200 + 40 * np.sin(gx / 60) * np.cos(gy / 60)
vertices = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float32)
i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
a = (i * n + j).ravel()
faces = np.concatenate([
    np.stack([a, a + n, a + 1], axis=1),
    np.stack([a + 1, a + n, a + n + 1], axis=1),
]).astype(np.int64)

summary = write_mesh(
    "surface.zarrvectors",
    vertices,
    faces,                                  # (F, L), L >= 3, global vertex indices
    chunk_shape=(200.0, 200.0, 200.0),
    vertex_attributes={"height": vertices[:, 2].copy()},
)
print(summary)
# {'vertex_count': 400, 'face_count': 722, 'chunk_count': 8, 'intra_face_count': 512,
#  'cross_face_count': 210, 'encoding': 'raw'}

result = read_mesh("surface.zarrvectors")
print(result["vertex_count"], result["face_count"])   # 400 722

result = read_mesh(
    "surface.zarrvectors",
    bbox=(np.array([0.0, 0.0, 0.0]), np.array([200.0, 200.0, 400.0])),
)
print(result["vertex_count"], result["face_count"])   # 100 162
```

`faces` of any shape other than `(F, L)` with `L >= 3` raises `ArrayError`.
For Draco, pass `encoding="draco"` and optionally
`draco_quantization_bits=` (default 11); see
[Codec pipeline](../foundations/codec_pipeline.md).

`read_mesh` takes `level=`, `object_ids=`, `bbox=`, `chunks=` and
`attribute_filter=`, and returns `vertices`, `faces` (indices into
`vertices`), `vertex_count`, `face_count` and, when `object_ids=` is given,
`object_ids` (the owning object of each vertex). A `bbox=` read keeps the
vertices inside the box and the faces whose vertices are all kept.
`object_ids=` on a Draco store raises `ArrayError`, because each chunk is one
bitstream shared by every object in it.

OBJ, STL and PLY converters are in the companion package
[zarr-vectors-tools](https://zarr-vectors-tools.readthedocs.io/en/latest/convert/meshes.html).

### Validation

- L1: `vertices/` missing at a level is an error. `links/` and
  `object_index/` are not required.
- L3: each `links/0/` offsets name parses, no offset is lexicographically
  negative and the offsets are non-decreasing; each record's chunks exist at
  the level; when the level claims `link_fragments_aligned`, a chunk whose
  intra-chunk faces are not filed by vertex fragment is an error.
- L4: `links/0/` with `link_width` 1 or 2 is an error; a mesh store with no
  readable link metadata is a warning. `links_convention` other than
  `explicit` is an error.

No level checks face vertex indices within a chunk, degenerate faces,
watertightness or winding consistency. See
[Validation overview](../validation/overview.md).
