# Meshes

In this tutorial you write two triangulated cell surfaces with a per-vertex
attribute, read them back, query a region, write a Draco-compressed copy, add
a coarser level and validate the store. Run the steps in order as one Python
session.

## 1. Create the store

```python
import numpy as np
import zarr_vectors as zv

cells = zv.create("cells.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    kind="mesh",
    vertex_attributes={"thickness": zv.AttributeSpec(dtype="float32")},
    layout=zv.Layout(cells=5),
))
```

[Choosing a layout](../../how_to/choose_chunk_and_bin.md) explains how to pick
the chunk and bin shapes.

## 2. Write the meshes

`add_mesh` takes the vertices as a `(V, 3)` array and the triangles as an
`(F, 3)` array of indices into it. To write several meshes in one call,
concatenate their vertices, add each mesh's vertex offset to its faces, and
pass one object id per vertex. This function builds a UV sphere as a stand-in
cell surface:

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

cells.add_mesh(vertices, faces, attributes={"thickness": thickness},
               object_ids=object_ids)
```

`read()` returns scalar per-vertex attributes only. Store a vector, such as a
normal, as one attribute per component (`normal_x`, `normal_y`, `normal_z`).

## 3. Read it back

`read()` returns the vertices, the faces and the attributes. Vertices come
back in stored order, so use `r.faces` with `r.positions`. Each face keeps its
winding.

```python
r = cells.read()
print(r)
print(cells.level(0).objects[1])
```

```text
ReadResult(kind='mesh', vertices=964, faces=1920, attributes=['thickness'])
ReadResult(kind='mesh', vertices=482, faces=960)
```

## 4. Query a region

A bounding-box query returns the vertices inside the box and the faces with
all three vertices inside it. It returns no attributes.

```python
box = ((300.0, 300.0, 300.0), (500.0, 500.0, 500.0))
print(cells.select(bbox=box).read())
```

```text
ReadResult(kind='mesh', vertices=130, faces=215)
```

To read every face near a region, including faces that cross its edge, use
[`MeshReader`](mesh_streaming.md).

## 5. Draco compression

`zarr_vectors.building.write_mesh` can store each chunk with Draco, a mesh
codec. Writing and reading need the `draco` extra
(`pip install "zarr-vectors[draco]"`).

```python
from zarr_vectors import building

building.write_mesh("cells_draco.zarrvectors", vertices, faces,
                    chunk_shape=(200.0, 200.0, 200.0),
                    encoding="draco", draco_quantization_bits=11)
```

Draco rounds each coordinate to `draco_quantization_bits` bits, so positions
do not come back exactly. Reading one object by id, and `MeshReader`, need a
raw store.

## 6. Add coarser levels

A coarse mesh level holds the centroids of the occupied bins and no faces. It
serves as a point overview.

```python
cells.build_pyramid(factors=[(2.0, 1.0)])
print(cells.read(level=1))
```

```text
ReadResult(kind='mesh', vertices=57, faces=0)
```

[Building pyramids](../multiscale/building_pyramids.md) covers the options.

## 7. Validate

```python
print(cells.validate(level=4).summary())
```

```text
Level 4 validation: PASS
  38 passed, 0 warnings, 0 errors
```

[Validation and repair](../../how_to/validate_and_repair.md) lists the checks.
To convert OBJ, STL or PLY files, use the companion package
`zarr-vectors-tools`.
