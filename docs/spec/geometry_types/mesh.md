# Mesh (`mesh`)

## Terms

**Mesh**
: A piecewise-linear surface given as vertices and faces. Each face is a
  tuple of vertex indices: three for a triangle, four for a quad. Open and
  closed surfaces are both allowed. The type constant is `GEOM_MESH`,
  `"mesh"`.

**Face record**
: A link record holding one face. `link_width` is the face size, at least
  3.

**Boundary face**
: A face whose vertices are not all in one chunk.

**Draco**
: A mesh compression format. With `encoding="draco"`, each chunk's vertices
  and intra-chunk faces are one Draco bitstream. It needs the `draco` extra
  (`pip install zarr-vectors[draco]`).

## Introduction

Each vertex goes to the chunk that contains it, and every face is a record
in one link family (`links_convention: explicit`). A face whose vertices
share a chunk goes in `links/0/0.0.0/`; a boundary face goes in the array
named by the offsets of its vertices' chunks. The reader returns each face
with its written vertex order and winding. See
[Links](../object_model/links.md).

Objects come from `object_ids=`, one ID per vertex; without it, every
vertex is in object 0. Each chunk has one vertex fragment per object. What
every type shares is on [Geometry types](index.md#common-to-every-type).

## Technical reference

### Arrays present

At level 0 of a store written by `write_mesh`:

| Path | `encoding="raw"` (default) | `encoding="draco"` |
|------|-----------------|--------------------|
| `vertices/` | Vertex rows | One bitstream per chunk, holding vertices and intra-chunk faces; one fragment shared by every object in the chunk |
| `links/0/0.0.0/` | Intra-chunk faces | — |
| `links/0/<offsets>/` | Boundary faces | Boundary faces |
| `link_fragments/` | With face records | With face records |
| `object_index/` | Always | Always |
| `vertex_attributes/<name>/`, `object_attributes/<name>/` | With the matching argument | With the matching argument |

Draco applies only to 3-D meshes; other dimensionalities use raw. A raw
store's level attributes carry `link_fragments_aligned: true`, so an object
read takes only its own faces.

The root attributes:

```javascript
{"geometry_types": ["mesh"], "links_convention": "explicit", ...}
```

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
)
print(summary)
# {'vertex_count': 400, 'face_count': 722, 'chunk_count': 8, 'intra_face_count': 512,
#  'cross_face_count': 210, 'encoding': 'raw'}

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

`read_mesh` returns `vertices`, `faces` (indices into `vertices`),
`vertex_count`, `face_count` and, with `object_ids=`, `object_ids` (the
owning object of each vertex). A `bbox=` read keeps the vertices inside the
box and the faces whose vertices are all kept. `object_ids=` on a Draco
store raises `ArrayError`.

OBJ, STL and PLY converters are in
[zarr-vectors-tools](https://zarr-vectors-tools.readthedocs.io/en/latest/convert/meshes.html).

### Validation

The [common checks](index.md#validation) apply, plus:

| Level | Check |
|-------|-------|
| L3 | When the level claims `link_fragments_aligned`, a chunk whose intra-chunk faces are not filed by vertex fragment is an error |
| L4 | `links/0/` with `link_width` 1 or 2 is an error; no readable link metadata is a warning |

No level checks degenerate faces, watertightness or winding consistency.
