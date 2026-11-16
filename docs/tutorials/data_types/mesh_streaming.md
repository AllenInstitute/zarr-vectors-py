# Stream mesh regions (experimental)

`MeshReader` loads faces owned by selected cells, including neighboring
vertices. Existing `read_mesh` bbox filtering stays unchanged.

![Selected owner and neighboring vertices; distant geometry stays unread](../../_static/mesh-owner-read.svg)

```python
from zarr_vectors.building import MeshReader

with MeshReader("surface.zarrvectors", level=0) as reader:
    for chunk in reader.chunks:
        mesh = reader.read([chunk.key], max_bytes=128 << 20)
        # mesh["vertices"] and mesh["faces"] are self-contained.
```

Owner reads partition faces and preserve winding. Shared vertices may repeat
between results. `chunk.bounds` conservatively covers all endpoint chunks.

## Limits

- Raw spatial 3-D surfaces with canonical ownership only. No Draco, attribute
  chunking, object selection or attributes. Each level needs stored faces.
- `max_bytes` covers dependency vertices and output face indices. It requires
  the writer-verified `fragments_tile` flag; other layouts allow unbudgeted reads.
  Connectivity, indices and metadata are read first; scratch memory is extra.
- Discovery lists all cells. `close()` releases the reader's metadata snapshot
  and index. Use a fresh reader after mutations; serialize calls to one reader.
- Missing vertices and invalid face indices raise errors.

[Reproduce measured reads](../../upstream/mesh-streaming-review.md) ·
[Existing mesh API](meshes.md#reading-a-mesh)
