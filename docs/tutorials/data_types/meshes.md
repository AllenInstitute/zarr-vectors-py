# Meshes

The `mesh` type stores triangulated 3-D surface meshes: cell boundaries
from electron microscopy segmentation, brain or organ surfaces from MRI
reconstruction, organelle hulls from fluorescence segmentation, or any
other closed or open triangulated surface.

ZVF meshes support optional Draco compression for significant size
reductions, per-vertex attributes (normals, UV coordinates, scalars), and
multi-mesh stores that pack thousands of mesh objects into a single
spatially indexed store.

All examples on this page use only the core `zarr-vectors` API.
OBJ/STL/PLY converters live in **`zarr-vectors-tools`**; Draco
compression requires `zarr-vectors[draco]`.

---

## Writing a mesh

### Write from vertices and faces

```python
import numpy as np
from zarr_vectors.types.meshes import write_mesh

# Generate a simple icosphere (demonstration only)
# In practice, load from OBJ, STL, PLY, or a segmentation pipeline
vertices, faces = generate_icosphere(radius=200.0, subdivisions=4)
# vertices: (N, 3) float32   — vertex positions in µm
# faces:    (F, 3) int32     — triangle vertex index triplets (0-indexed, global)

write_mesh(
    "cell.zarrvectors",
    vertices=vertices.astype(np.float32),
    faces=faces.astype(np.int32),
    chunk_shape=(100.0, 100.0, 100.0),
    bin_shape=(25.0, 25.0, 25.0),
)
```

Face **winding is preserved automatically** — there is no
`winding_order` argument. A face is stored as an undirected link record
of `L` endpoints, and the permutation needed to restore your original
vertex order is kept alongside it as `perm_idx`; `read_mesh` applies it,
so the faces you read back have the winding you wrote.

`write_mesh` also has no `coordinate_system` or `axis_units` arguments —
those concepts do not exist anywhere in this package.

### Write with per-vertex attributes

Common mesh attributes include normals, curvature, UV texture coordinates,
and scalar overlays (e.g. cortical thickness):

```python
rng = np.random.default_rng(0)
n_verts = len(vertices)

# Compute vertex normals (simplified; use trimesh or open3d in practice)
normals   = compute_vertex_normals(vertices, faces)   # (N, 3) float32
curvature = rng.uniform(-0.1, 0.1, n_verts).astype(np.float32)
thickness = rng.uniform(1.5, 4.5, n_verts).astype(np.float32)

write_mesh(
    "brain_surface.zarrvectors",
    vertices=vertices,
    faces=faces,
    chunk_shape=(10.0, 10.0, 10.0),     # smaller chunks for dense mesh
    bin_shape=(5.0, 5.0, 5.0),
    vertex_attributes={
        "normal":    normals,           # vector attribute: (N, 3)
        "curvature": curvature,         # scalar attribute: (N,)
        "thickness": thickness,
    },
)
```

The keyword is `vertex_attributes` — `write_mesh` has no `attributes`
alias. (`write_points` does accept `attributes=`, but it is deprecated
there in favour of `vertex_attributes`.)

---

## Ingesting from external formats

Format converters for OBJ, STL, and PLY (and the `zarr-vectors` CLI)
live in the companion package **`zarr-vectors-tools`**.

---

## Draco compression

Draco is a geometry compression library that exploits vertex-face
correlations for significantly better compression than general-purpose
codecs on mesh data. Requires `zarr-vectors[draco]`.

Draco compression is enabled with `encoding="draco"` (the default is
`encoding="raw"`). There is no `use_draco` argument. Quantisation is set
with `draco_quantization_bits`, which defaults to `11`:

```python
from zarr_vectors.types.meshes import write_mesh

summary = write_mesh(
    "brain_draco.zarrvectors",
    vertices=vertices,
    faces=faces,
    chunk_shape=(100.0, 100.0, 100.0),
    encoding="draco",              # "raw" (default) or "draco"
    draco_quantization_bits=11,    # default 11
)
print(summary["encoding"])         # 'draco'
```

The Draco encode path is taken only for 3-D meshes.

### Compression ratio guidance

| `draco_quantization_bits` | Precision per axis | Typical compression vs float32 |
|--------------------|--------------------|-------------------------------|
| 8 | 1 / 256 of bbox | 12–18× |
| 11 | 1 / 2048 of bbox | 7–12× |
| 14 | 1 / 16384 of bbox | 4–7× |

For nanometre-resolution EM segmentation meshes with a bbox of ~100 µm,
11-bit quantisation gives sub-50 nm precision — more than sufficient for
most visualisation and analysis workflows.

```python
# Check whether a store's vertices are Draco-encoded.
# The encoding is recorded on the vertices array, not in root attrs —
# there is no "draco_compressed" root attribute.
from zarr_vectors.core.store import open_store, get_resolution_level

level_group = get_resolution_level(open_store("brain_draco.zarrvectors", mode="r"), 0)
meta = level_group.read_array_meta("vertices")
print(meta["encoding"])            # 'raw' or 'draco'
```

**Important:** Draco-compressed stores are not readable without
`zarr-vectors[draco]` installed. Communicate the compression requirement
clearly when distributing stores.

---

## Reading a mesh

### Read all data

```python
from zarr_vectors.types.meshes import read_mesh

result = read_mesh("brain.zarrvectors")

print(result["vertex_count"])     # int
print(result["face_count"])       # int
print(result["vertices"].shape)   # (N, 3) float32
print(result["faces"].shape)      # (F, 3) int32 — global vertex indices
```

The returned `faces` array contains global vertex indices (0-indexed into
the `vertices` array). Face indices are consistent: face `k` is defined by
`vertices[faces[k, 0]]`, `vertices[faces[k, 1]]`, `vertices[faces[k, 2]]`.

### Read per-vertex attributes

`read_mesh` takes no `attributes=` argument and never returns an
`"attributes"` key — it returns only `vertices`, `faces`, `vertex_count`,
and `face_count`. Read per-vertex attributes through the lazy API:

```python
from zarr_vectors.lazy import open_zv

level = open_zv("brain_surface.zarrvectors")[0]

curvature = level.attributes["curvature"].compute()   # (V,)
thickness = level.attributes["thickness"].compute()   # (V,)
```

**Vector attributes come back flat.** A `(V, C)` attribute such as
`normal` reads back as a 1-D `(V * C,)` array — the component shape is
not restored. Reshape it yourself:

```python
if "normal" in level.attributes:
    normals = level.attributes["normal"].compute()    # (V*3,) — flat!
    normals = normals.reshape(-1, 3)                  # (V, 3)
```

These arrays are in stored order for the whole level, so they align with
an unfiltered `read_mesh(...)["vertices"]` but not with a bbox-filtered
one.

### Spatial bbox query

A bbox query returns faces whose centroid is within the bounding box, plus
all vertices referenced by those faces (which may lie slightly outside the
bbox):

```python
result = read_mesh(
    "brain.zarrvectors",
    bbox=(np.array([0., 0., 0.]), np.array([50., 50., 50.])),
)
print(result["face_count"])      # faces with centroid in the bbox region
```

---

## Multi-mesh stores

For segmentation datasets with thousands of cell objects, a single ZVF
store is far more efficient than per-cell OBJ files:

### Writing a multi-mesh store

`write_mesh` takes **one** `(V, D)` vertex array and **one** `(F, L)`
face array — not lists of per-object arrays. Multiple objects are
expressed with `object_ids`, a `(V,)` array assigning each vertex to a
mesh object. Concatenate your per-cell meshes and offset each cell's face
indices into the combined vertex space:

```python
import numpy as np
from zarr_vectors.types.meshes import write_mesh

vert_blocks, face_blocks, oid_blocks = [], [], []
cell_volumes, cell_types = [], []
offset = 0

for cell_id, (verts, faces) in enumerate(cell_meshes):
    vert_blocks.append(verts)
    face_blocks.append(faces + offset)      # local -> global vertex indices
    oid_blocks.append(np.full(len(verts), cell_id, dtype=np.int64))
    offset += len(verts)
    cell_volumes.append(compute_volume(verts, faces))
    cell_types.append(cell_type_of(cell_id))

write_mesh(
    "cells.zarrvectors",
    vertices=np.concatenate(vert_blocks),     # (V, 3)
    faces=np.concatenate(face_blocks),        # (F, 3) global indices
    chunk_shape=(50., 50., 50.),
    object_ids=np.concatenate(oid_blocks),    # (V,) vertex -> object
    object_attributes={
        "volume":    np.array(cell_volumes, dtype=np.float32),
        "cell_type": np.array(cell_types,   dtype=np.int32),
    },
)
```

If `object_ids` is omitted, every vertex belongs to object 0 — except
when `chunk_by_attribute` is set, in which case each vertex becomes its
own object so the per-object uniformity check is trivially satisfied.

### Reading cells

> **Known limitation.** `read_mesh` accepts an `object_ids=` argument,
> but in the current version it has **no effect** — the parameter is
> never applied, so you get the whole level back regardless of what you
> pass. There is currently no way to read a single mesh object out of a
> multi-mesh store via `read_mesh`. Use `bbox=` or `chunks=` to restrict
> a read spatially, which does work.

```python
from zarr_vectors.types.meshes import read_mesh

result = read_mesh("cells.zarrvectors")
print(sorted(result))             # ['face_count', 'faces', 'vertex_count', 'vertices']
print(result["vertex_count"])
print(result["vertices"].shape)   # (V, 3)
print(result["faces"].shape)      # (F, L)
```

`read_mesh` returns exactly those four keys. It does **not** return an
`object_ids` key, and objects come back merged into one vertex/face
array with no per-object boundary.

### Spatial query in a multi-mesh store

```python
result = read_mesh(
    "cells.zarrvectors",
    bbox=(np.array([500., 500., 200.]),
          np.array([600., 600., 300.])),
)
print(result["face_count"])
```

There is no `return_object_ids` option — passing it raises `TypeError`.
To discover which cells have geometry in a region, use the lazy API's
object helpers:

```python
from zarr_vectors.lazy import open_zv

store = open_zv("cells.zarrvectors")
level = store[0]
print(level.present_oids)        # object IDs present at this level
print(level.has_object(42))      # True / False
```

---

## Exporting

OBJ and PLY exporters live in the companion package
**`zarr-vectors-tools`**.

---

## Validation

```python
from zarr_vectors.validate import validate

result = validate("brain.zarrvectors", level=4)
print(result.summary())
# Level 4 validation: PASS
#   29 passed, 0 warnings, 0 errors
```

For closed surfaces, level 4 additionally checks watertightness (every
edge shared by exactly two faces). Enable this check by setting
`closed_surface = true` in root `.zattrs`:

```python
from zarr_vectors.core.store import open_store

root = open_store("cell.zarrvectors", mode="r+")
root.attrs["closed_surface"] = True
# Now validate(cell.zarrvectors, level=4) checks watertightness
```

---

## Common pitfalls

**Face indices are global, not local.**
When calling `write_mesh` with a single mesh, `faces` must use 0-based
indices into the `vertices` array you are passing. When calling with
a list of per-object arrays, each face array uses local indices into
its own per-object `vertices` array; the writer handles global index
conversion automatically.

**Draco changes vertex positions slightly.**
Draco quantises vertex positions to integers before compression. Even at
the highest precision level (14 bits), there is a small rounding error.
Do not use Draco if you need exact float32 round-trip fidelity (e.g. for
downstream numerical computation on vertex coordinates). For visualisation,
11-bit quantisation is imperceptible.

**Winding order inconsistency between files.**
Different mesh tools use different winding conventions. `ingest_obj`
defaults to CCW (the OBJ standard) but some exporters produce CW meshes
without declaring it. If your rendered normals point inward, pass
`winding_order="cw"` at ingest time or flip normals post-hoc:

```python
ingest_obj("inverted.obj", "inverted.zarrvectors",
           chunk_shape=(10., 10., 10.),
           winding_order="cw")
```

**Boundary face resolution requires fetching extra chunks.**
A face whose centroid is in chunk A but one vertex is in chunk B requires
fetching chunk B to resolve the vertex position. The reader does this
automatically, but it means a bbox query may issue slightly more chunk
reads than the number of chunks in the bbox. This is expected behaviour.
