# Meshes

In this tutorial you write two triangulated cell surfaces, with a per-vertex
attribute, to a new store; read them back whole, by cell and by region; add a
coarser level and validate the result. The last section shows the
Draco-compressed encoding.

The steps run in order as one Python session. The
[quickstart](../../getting_started/quickstart.md) introduces the shared calls
on point clouds and polylines, and [Concepts](../../explanation/concepts.md)
explains chunks, bins and objects.

## 1. Create the store

```python
import numpy as np
import zarr_vectors as zv

cells = zv.create("cells.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    axes=(zv.Axis("x", unit="micrometer"),
          zv.Axis("y", unit="micrometer"),
          zv.Axis("z", unit="micrometer")),
    kind="mesh",
    vertex_attributes={"thickness": zv.AttributeSpec(dtype="float32", unit="micrometer")},
    layout=zv.Layout(cells=5),
))
print(cells.level(0).scale, cells.level(0).resolution)
```

```text
(200.0, 200.0, 200.0) (50.0, 50.0, 50.0)
```

[Choosing a layout](../../how_to/choose_chunk_and_bin.md) explains how to pick
the chunk and bin shapes.

## 2. Write the meshes

`add_mesh` takes the vertices as an `(V, 3)` array and the triangles as an
`(F, 3)` array of indices into it. To store several meshes in one call, put
their vertices in one array, offset each mesh's face indices by the number of
vertices before it, and pass one object id per vertex. This function builds a
UV sphere to stand in for a cell surface:

```python
def sphere(centre, radius, n_lat=16, n_lon=32):
    """A UV sphere: (V, 3) float32 vertices and (F, 3) int64 triangles."""
    lat = np.linspace(0.0, np.pi, n_lat + 1)[1:-1]
    lon = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    ring = np.stack([np.outer(np.sin(lat), np.cos(lon)),
                     np.outer(np.sin(lat), np.sin(lon)),
                     np.outer(np.cos(lat), np.ones(n_lon))], axis=-1).reshape(-1, 3)
    verts = np.concatenate([[[0, 0, 1]], ring, [[0, 0, -1]]]) * radius + centre
    faces = []
    south = len(verts) - 1
    for j in range(n_lon):
        k = (j + 1) % n_lon
        faces.append([0, 1 + j, 1 + k])
        faces.append([south, south - n_lon + k, south - n_lon + j])
        for i in range(n_lat - 2):
            a, b = 1 + i * n_lon + j, 1 + i * n_lon + k
            faces.append([a, a + n_lon, b])
            faces.append([b, a + n_lon, b + n_lon])
    return verts.astype(np.float32), np.array(faces, dtype=np.int64)

v0, f0 = sphere(np.array([350.0, 400.0, 500.0]), 150.0)
v1, f1 = sphere(np.array([650.0, 600.0, 500.0]), 120.0)
vertices = np.concatenate([v0, v1])
faces = np.concatenate([f0, f1 + len(v0)])
object_ids = np.repeat([0, 1], [len(v0), len(v1)])
thickness = np.random.default_rng(0).uniform(1.0, 3.0, len(vertices)).astype(np.float32)

print(cells.add_mesh(vertices, faces, attributes={"thickness": thickness},
                     object_ids=object_ids))
```

```text
{'vertex_count': 964, 'face_count': 1920, 'chunk_count': 19, 'intra_face_count': 1378, 'cross_face_count': 542, 'encoding': 'raw'}
```

Each face is stored once, with the chunk that owns it; `cross_face_count`
counts the faces whose vertices lie in more than one chunk. The
[mesh specification](../../spec/geometry_types/mesh.md) describes the layout.

## 3. Read it back

`read()` returns the vertices, the faces as indices into them, and every
per-vertex attribute:

```python
r = cells.read()
print(r)
print(r.positions.shape, r.faces.shape, r.attributes["thickness"].shape)
```

```text
ReadResult(kind='mesh', vertices=964, faces=1920, attributes=['thickness'])
(964, 3) (1920, 3) (964,)
```

Vertices come back in the stored order, which need not match the order you
wrote; use `r.faces` with `r.positions`, not your original indices. Each
face keeps its winding.

To read one cell, index the level's objects. The faces index that cell's own
vertices:

```python
one = cells.level(0).objects[1]
print(one, one.faces.max() < one.vertex_count)
```

```text
ReadResult(kind='mesh', vertices=482, faces=960) True
```

## 4. Query a region

A bounding-box query returns the vertices inside the box and the faces whose
three vertices are all inside it:

```python
box = ((300.0, 300.0, 300.0), (500.0, 500.0, 500.0))
q = cells.select(bbox=box)
print(q.explain())
part = q.read()
print(part, part.attributes_read)
```

```text
mesh read: level 0, bbox [300.0, 300.0, 300.0]..[500.0, 500.0, 500.0]
  via read_mesh(bbox, level)
ReadResult(kind='mesh', vertices=130, faces=215) False
```

A region read of a mesh returns no attributes: `attributes_read` is `False`.
Faces that cross the edge of the box are left out; to read every face that
touches a set of chunks, with the vertices those faces need, use
[`MeshReader`](mesh_streaming.md).

## 5. Per-vertex attributes

A whole read returns each scalar per-vertex attribute aligned with
`positions`. Store a vector quantity, such as a normal, as one scalar
attribute per component:

```python
normal = vertices - np.where(object_ids[:, None] == 0, v0.mean(axis=0), v1.mean(axis=0))
normal /= np.linalg.norm(normal, axis=1, keepdims=True)

shaded = zv.create("shaded.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    kind="mesh",
    layout=zv.Layout(cells=5),
))
shaded.add_mesh(vertices, faces, object_ids=object_ids, attributes={
    "normal_x": normal[:, 0], "normal_y": normal[:, 1], "normal_z": normal[:, 2],
})
s = shaded.read()
n_read = np.column_stack([s.attributes[f"normal_{a}"] for a in "xyz"])
print(n_read.shape, np.allclose(np.linalg.norm(n_read, axis=1), 1.0))
```

```text
(964, 3) True
```

## 6. Add coarser levels

`build_pyramid` writes coarser levels after level 0. For a mesh, a coarse
level holds the centroids of the occupied bins and no faces, which serves as
a point overview:

```python
print(cells.build_pyramid(factors=[(2.0, 1.0)])["levels_created"])
for i in cells.levels:
    print(i, cells.level(i).resolution, cells.read(level=i))
```

```text
1
0 (50.0, 50.0, 50.0) ReadResult(kind='mesh', vertices=964, faces=1920, attributes=['thickness'])
1 (100.0, 100.0, 100.0) ReadResult(kind='mesh', vertices=57, faces=0)
```

[Building pyramids](../multiscale/building_pyramids.md) covers the other
options.

## 7. Validate

```python
print(cells.validate(level=4).summary())
```

```text
Level 4 validation: PASS
  38 passed, 0 warnings, 0 errors
```

[Validation and repair](../../how_to/validate_and_repair.md) lists what each level
checks.

To convert OBJ, STL or PLY files to and from a store, use the companion
package `zarr-vectors-tools`.

## Draco compression

`zarr_vectors.building.write_mesh` can store the vertices and faces of each
chunk with Draco, a mesh codec, instead of as raw arrays. It needs the
`draco` extra (`pip install "zarr-vectors[draco]"`), both to write and to
read the store:

```python
from zarr_vectors import building

summary = building.write_mesh(
    "cells_draco.zarrvectors", vertices, faces,
    chunk_shape=(200.0, 200.0, 200.0),
    encoding="draco",
    draco_quantization_bits=11,
)
print(summary["encoding"])
```

Draco rounds each vertex coordinate to `draco_quantization_bits` bits, so
positions do not come back exactly as written. Reading a single object by id
and `MeshReader` need a raw store.

## Common mistakes

**Per-mesh face indices.** Faces index the whole `vertices` array you pass.
When you combine meshes, add each mesh's vertex offset to its faces, as in
step 2.

**A vector attribute.** An `(V, C)` per-vertex attribute is written and
listed by `attribute_names("vertex")`, but `read()` does not return it. Split
it into scalar attributes, as in step 5.
