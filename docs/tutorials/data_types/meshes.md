# Meshes

The `mesh` type stores triangulated 3-D surface meshes: cell boundaries
from electron microscopy segmentation, brain or organ surfaces from MRI
reconstruction, organelle hulls from fluorescence segmentation, or any
other closed or open triangulated surface.

ZVF meshes support optional Draco compression for significant size
reductions, per-vertex attributes (normals, UV coordinates, scalars), and
multi-mesh stores that pack thousands of mesh objects into a single
spatially indexed store.

All examples on this page use only the core `zarr-vectors` package.
OBJ/STL/PLY converters live in **`zarr-vectors-tools`**; Draco
compression requires `zarr-vectors[draco]`.

`write_mesh` and `read_mesh` come from `zarr_vectors.types`, which is
**undecided** — neither promised nor disowned. The writers are also
exported from `zarr_vectors.building` and supported there; the readers
cannot be retired until the data api can carry per-vertex attributes for
every geometry, and pointing you at a lossy replacement would be worse
than leaving them here. Everything else on this page uses the two
supported surfaces: `zarr_vectors` itself for reading data, and
`zarr_vectors.building` for the physical layout. Ask at runtime with
`zv.stability("zarr_vectors.types")`.

---

## Writing a mesh

### Write from vertices and faces

```python
import numpy as np
from zarr_vectors.building import write_mesh

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
from zarr_vectors.building import write_mesh

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
from zarr_vectors.building import get_resolution_level, open_store

level_group = get_resolution_level(open_store("brain_draco.zarrvectors", mode="r"), 0)
meta = level_group.read_array_meta("vertices")
print(meta["encoding"])            # 'raw' or 'draco'
```

`open_store` and `get_resolution_level` are `zarr_vectors.building`
names. They used to be imported from `zarr_vectors.core.store`; that
module is internal, and the two functions are re-exported from
`building` unchanged.

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
and `face_count`. Read per-vertex attributes through the data API:

```python
import zarr_vectors as zv

level = zv.open("brain_surface.zarrvectors").level(0)
print(level.attribute_names("vertex"))   # ('curvature', 'normal', 'thickness')

result    = level.read()
curvature = result.attributes["curvature"]   # (V,)
thickness = result.attributes["thickness"]   # (V,)
```

Earlier versions of this page reached for `zarr_vectors.lazy.open_zv`
here. That module is internal, and `open_zv` now emits a
`DeprecationWarning` explaining why: the lazy layer reads chunk by chunk
in Python and opens no batched-read block, so against an object store it
is slower than the eager path it was meant to improve on. `zv.open`
returns a `Dataset` that drives the batching engine instead.

**Vector attributes do not come back.** `write_mesh` flattens a `(V, C)`
attribute such as `normal` into a `(V * C,)` column and records its
stored width as 1. The level still lists the name, but `read()` drops it,
because a column of `V * C` rows cannot be paired with `V` positions
without guessing:

```python
print(level.attribute_names("vertex"))   # ('curvature', 'normal', 'thickness')
print(result.attributes.names())         # ('curvature', 'thickness') — no 'normal'
```

Fetch it from the building surface, which hands back the flat column for
you to reshape. `chunk_local_to_global_offsets` gives the chunk order the
readers concatenate in, so the reshaped array lines up row-for-row with
`result.positions`:

```python
import numpy as np
from zarr_vectors.building import (
    chunk_local_to_global_offsets,
    get_resolution_level,
    open_store,
    read_chunk_attributes,
)

level_group = get_resolution_level(
    open_store("brain_surface.zarrvectors", mode="r"), 0,
)
_offsets, chunk_keys, total = chunk_local_to_global_offsets(level_group)

flat = np.concatenate([
    fragment
    for cc in chunk_keys
    for fragment in read_chunk_attributes(level_group, "normal", cc)
])                                       # (V*3,) — flat!
normals = flat.reshape(-1, 3)            # (V, 3)
```

Both routes are in stored order for the whole level, so they align with
an unfiltered read but not with a bbox-filtered one. A narrowed read says
so rather than mis-pairing values with positions: it comes back with
`attributes_read == False` and an empty `attributes`.

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
from zarr_vectors.building import write_mesh

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

> **Known limitation.** `read_mesh` accepts an `object_ids=` argument
> but does not implement it, and says so rather than pretending:
>
> ```pycon
> >>> read_mesh("cells.zarrvectors", object_ids=[3])
> NotImplementedError: read_mesh(object_ids=...) is not implemented: the filter
> would be silently ignored and you would get the whole level back. Filter the
> returned arrays yourself, or use read_polylines, which does implement object_ids.
> ```
>
> `level.objects[3]` on the data API goes through the same reader and
> raises the same error, so a mesh object cannot be read by id from
> either supported surface. Use `bbox=` or `chunks=` to restrict a read
> spatially, which does work, or read one object's vertices through
> `zarr_vectors.building` (below).

One object's vertices *can* be gathered by following its manifest, which
is what `object_index/manifests/` is for:

```python
import numpy as np
from zarr_vectors.building import (
    get_resolution_level, open_store, read_object_vertices,
)

level_group = get_resolution_level(open_store("cells.zarrvectors", mode="r"), 0)
fragments = read_object_vertices(level_group, 3, ndim=3)   # per-chunk pieces
cell_3 = np.concatenate(fragments)                         # (V_3, 3)
```

That gives you vertices, not faces: reassembling one object's faces still
means reading the level and filtering. This is the gap to report rather
than a reason to import from `core`.

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
To ask which cells a store holds, use the level's object catalogue:

```python
import zarr_vectors as zv

level = zv.open("cells.zarrvectors").level(0)
print(level.objects)                       # ObjectCatalog(level=0, count=6, slots=6)
print(level.objects.ids())                 # ids that actually hold geometry
print(level.objects.ids(present=False))    # every addressable slot
print(42 in level.objects)                 # True / False
print(len(level.objects))                  # slot count, from metadata; reads nothing
```

The catalogue answers for the whole level, not for a region.
`select(bbox=...).object_ids()` comes back **empty** on a mesh store —
the mesh reader does not carry per-vertex object ids — so there is no
per-region object listing for meshes on either surface.

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
#   26 passed, 0 warnings, 0 errors
```

The count moves with what the store contains — it is how many assertions
ran, not a score.

Beyond the generic checks, level 4 asks two things of a mesh: that the
declared `links_convention` is `explicit`, and that the delta-0 links
family has `link_width >= 3`. A mesh store with no link metadata at all
is a warning rather than an error — that is the Draco no-boundary-face
case.

**There is no `closed_surface` flag and no watertightness check.**
Earlier versions of this page said the check could be switched on by
setting `closed_surface = true` in root `.zattrs`. Both halves were
wrong. No shipped code writes, reads or validates that key, and `.zattrs`
is Zarr v2 spelling: a v3 store keeps its root fields in `zarr.json`,
under `attributes.zarr_vectors`. Watertightness, boundary edges,
degenerate faces and winding consistency are checked at no level, so
check them yourself before writing if they matter.

---

## Common pitfalls

**Face indices are global, not local.**
`faces` must use 0-based indices into the `vertices` array you are
passing — always, including in a multi-mesh store, where you offset each
cell's face indices into the combined vertex space yourself (see
[Writing a multi-mesh store](#writing-a-multi-mesh-store)). There is no
list-of-per-object-arrays form that would convert local indices for you;
passing lists fails on the shape check with `ValueError: too many values
to unpack (expected 2)`.

**Draco changes vertex positions slightly.**
Draco quantises vertex positions to integers before compression. Even at
the highest precision level (14 bits), there is a small rounding error.
Do not use Draco if you need exact float32 round-trip fidelity (e.g. for
downstream numerical computation on vertex coordinates). For visualisation,
11-bit quantisation is imperceptible.

**Winding order inconsistency between files.**
Different mesh tools use different winding conventions, and some
exporters produce CW meshes without declaring it. Nothing in this package
can help: `write_mesh` has no `winding_order` argument, no root key
records one, and no validation level checks winding consistency. What
`write_mesh` guarantees is narrower and more useful — the input vertex
order of each face is recovered exactly on read. If your rendered normals
point inward, fix it at ingest time (the OBJ/STL/PLY converters in
**`zarr-vectors-tools`** are where a winding option would live) or flip
the normals post-hoc.

**Boundary face resolution requires fetching extra chunks.**
A face whose centroid is in chunk A but one vertex is in chunk B requires
fetching chunk B to resolve the vertex position. The reader does this
automatically, but it means a bbox query may issue slightly more chunk
reads than the number of chunks in the bbox. This is expected behaviour.
