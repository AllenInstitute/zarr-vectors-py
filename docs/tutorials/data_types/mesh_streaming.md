# Stream mesh regions

`zarr_vectors.building.MeshReader` reads a mesh a few chunks at a time. It
returns every face owned by the chosen chunks, with all the vertices those
faces use, including vertices in neighbouring chunks. The
[meshes tutorial](meshes.md) covers writing and whole reads.

![Selected owner and neighbouring vertices; distant geometry stays unread](../../_static/mesh-owner-read.svg)

Write a sphere, then read the faces owned by the chunks that meet a box:

```python
import numpy as np
import zarr_vectors as zv
from zarr_vectors.building import MeshReader

lat = np.linspace(0.0, np.pi, 17)[1:-1]
lon = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
ring = np.stack([np.outer(np.sin(lat), np.cos(lon)),
                 np.outer(np.sin(lat), np.sin(lon)),
                 np.outer(np.cos(lat), np.ones(32))], axis=-1).reshape(-1, 3)
vertices = (np.concatenate([[[0, 0, 1]], ring, [[0, 0, -1]]]) * 300.0 + 500.0).astype(np.float32)
faces = []
for j in range(32):
    k = (j + 1) % 32
    faces += [[0, 1 + j, 1 + k], [481, 449 + k, 449 + j]]
    for i in range(14):
        a, b = 1 + i * 32 + j, 1 + i * 32 + k
        faces += [[a, a + 32, b], [b, a + 32, b + 32]]
faces = np.array(faces)

surface = zv.create("surface.zarrvectors", schema=zv.Schema(
    bounds=((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)),
    kind="mesh",
    layout=zv.Layout(cells=5),
))
surface.add_mesh(vertices, faces)

lo, hi = np.array([150.0, 150.0, 150.0]), np.array([450.0, 450.0, 450.0])
with MeshReader("surface.zarrvectors", level=0) as reader:
    near = [c.key for c in reader.chunks
            if np.all(np.array(c.bounds[0]) <= hi) and np.all(np.array(c.bounds[1]) >= lo)]
    mesh = reader.read(near, max_bytes=128 << 20)
    print(len(reader.chunks), len(near), mesh["vertex_count"], mesh["face_count"])
```

```text
26 11 264 458
```

`reader.chunks` lists the chunks that own faces. Each chunk's `bounds` cover
every chunk its faces reach, so filtering on them does not miss a face that
crosses into the box. Each result is a complete mesh: its `faces` index its
own `vertices`. Reading every chunk in turn returns each face once.

## Limits

- The level must be a raw-encoded 3-D mesh, not Draco and not chunked by
  attribute. Results hold geometry only, with no attributes.
- A read larger than `max_bytes` raises `MemoryError` before reading vertex
  data.
- The reader snapshots the store's metadata when it opens. Open a new reader
  after writing, and use one reader from one thread at a time.
