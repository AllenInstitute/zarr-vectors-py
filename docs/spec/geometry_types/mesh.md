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

`faces` of any shape other than `(F, L)` with `L >= 3` raises `ArrayError`,
as does `out_of_bounds="ignore"`, since dropped vertices would leave faces
dangling. `write_mesh` takes no per-face attributes. Through the dataset
API, use `ds.add_mesh(vertices, faces, ...)`.
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

### Volume meshes

A mesh can also hold solid cells: one tetrahedron, pyramid, wedge or
hexahedron per record instead of one polygon face. Pass `element=` to
`write_mesh`; the `links/0/` family records it, and `read_mesh` returns
it as `element` (None for a surface mesh):

```python
write_mesh(path, vertices, tets, chunk_shape=(40.0,) * 3, element="tetra")
read_mesh(path)["element"]   # "tetra"
```

| `element` | `link_width` | Vertex order (VTK) |
|-----------|--------------|--------------------|
| `tetra` | 4 | `0, 1, 2` a triangle whose right-hand normal points towards `3` |
| `pyramid` | 5 | base quad `0-3` whose right-hand normal points towards the apex `4` |
| `wedge` | 6 | triangle `0, 1, 2` facing away from triangle `3, 4, 5`; `i` joined to `i + 3` |
| `hexahedron` | 8 | quad `0-3` facing towards quad `4-7`; `i` joined to `i + 4` |

In that order every cell has positive volume. `write_mesh` refuses
cells with non-positive signed volume before writing anything; pass
`check_orientation=False` to store them as given. Vertex order survives
storage: records are stored sorted, with a permutation column
(`has_perm`) that restores the order on read.

- Several element types: pass the cells as a dict,
  `write_mesh(path, vertices, {"tetra": tets, "hexahedron": hexes})`.
  The family is `element: "mixed"` with `element_types`; records are as
  wide as the widest type, and a shorter cell repeats its last vertex
  (a cell's vertices must be distinct, so the record's width before that
  run names its type). `read_mesh` returns the padded `faces` and
  `cells`, a dict of each type's cells.
- `face_attributes={name: (F,) or (F, C)}` stores per-face or per-cell
  values (material, quality) as link attributes, in the order of
  `faces` (for a dict of cells, each type's cells in turn).
- Draco encodes triangles only, so volume meshes are written `raw`.
- `building.boundary_faces(cells, element)` returns the faces that
  belong to one cell only, wound outward: the volume's surface.
  `building.signed_volumes(vertices, cells, element)` returns each
  cell's signed volume.
- An element is declared once. `building.stamp_link_element` refuses to
  retype records already written, for example quads as tetrahedra.
- Cells cross chunk boundaries like any link record. What `element`
  is limited to is one level: every vertex of a cell is on the same
  pyramid level.

#### Pyramids of a volume mesh

`build_pyramid` gives each coarse level its own cells, built on the
level's bins. Each coarse vertex is the centroid of one bin. For each
object, every 2 x 2 x 2 block of bins the object fills becomes one
hexahedron (from a hexahedral mesh) or six tetrahedra (from any other
element), joining those eight vertices. Cells that would be inverted at
the vertices' real positions are dropped; the build summary counts them
as `cells_dropped_inverted`.

Each coarse vertex on the boundary then moves to the centroid of the
fine boundary vertices in its bin, so the faces of the volume stay where
they were (edges and corners round off by up to a bin; the summary
counts `boundary_vertices_snapped`). A move that would invert a cell is
undone, and any cell still inverted is dropped.

The cells fill the occupied volume with no gaps, and each block depends
only on its own bins, so the build stays chunk-local. A feature thinner
than two bins has no cells at that level. A mixed source coarsens to
tetrahedra.

#### Surface pyramids and carried-down values

A surface mesh's coarse levels keep faces by vertex clustering: each
face is re-pointed at the coarse vertices of its vertices' bins, faces
left with a repeated vertex collapse, and faces over the same vertices
are kept once, with the first one's winding. The summary reports
`face_count`.

Face and cell attributes come down too. A coarse face takes the fine
faces that clustered onto it; a coarse cell takes the fine cells whose
centroid lies in its block. Floating-point values are averaged, weighted
by area or volume; integer and boolean values take the value with the
most area or volume. A coarse cell that no fine cell maps to takes the
level-wide mean or majority.

#### Coarse levels decimated elsewhere

For a better coarse mesh than the bin lattice (edge collapse,
TetGen, quadric decimation), make it with another tool and write it as
its own level:

```python
from zarr_vectors.building import write_mesh_level

write_mesh_level(path, 1, coarse_vertices, coarse_tets, element="tetra",
                 bin_shape=(20.0, 20.0, 20.0))
```

The level is marked `coarsening_method: "manual"` with `parent_level`
(default `level - 1`), and its `bin_shape` (default twice the
parent's) and `bin_ratio`. Its object ids must be ids of the parent
level. With `link_parents=True` (the default), every parent-level
vertex is linked to the nearest vertex of the same object on the new
level, within about two bins, in `links/+delta` and `links/-delta`, as
`build_pyramid` links its levels.

### Validation

The [common checks](index.md#validation) apply, plus:

| Level | Check |
|-------|-------|
| L3 | When the level claims `link_fragments_aligned`, a chunk whose intra-chunk faces are not filed by vertex fragment is an error |
| L3 | An `element` that is unknown, does not match `link_width`, or sits on a `delta != 0` family is an error |
| L4 | `links/0/` with `link_width` 1 or 2 is an error; no readable link metadata is a warning |

No level checks degenerate faces, watertightness or winding consistency; volume cells' orientation is checked when they are written.
