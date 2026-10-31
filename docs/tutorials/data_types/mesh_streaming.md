# Ownership-based mesh streaming (experimental)

`zarr_vectors.building.MeshReader` is an additive low-level interface for
out-of-core renderers. It leaves the existing `read_mesh` bbox semantics intact.

```python
from zarr_vectors.building import MeshReader

reader = MeshReader("surface.zarrvectors", level=0)
for chunk in reader.chunks:
    print(chunk.key, chunk.bounds)
    mesh = reader.read([chunk.key], max_bytes=128 << 20)
    # mesh["vertices"] and mesh["faces"] are self-contained.
reader.close()
```

Each chunk key names a **stored canonical face source cell**, not a centroid
region or a filter on vertex positions. Each face has one owner. Independent
owner reads therefore partition faces without losing triangles crossing chunk
boundaries. Shared vertices may be duplicated across independently read results.
Input winding is preserved by the existing link permutation decoder.

The metadata-only index derives conservative bounds from the endpoint chunk
coordinates of every offsets family present at an owner. The reader fetches
only selected connectivity payloads, then fetches all referenced vertex chunks.
It does not read the complete face-link family for a regional request.
Metadata discovery still lists all connectivity cells for that level.

`max_bytes` checks decoded dependency vertex bytes plus face-index bytes before
fetching vertex payloads. Discovering that requirement reads the selected
connectivity and vertex-fragment indices first. Connectivity decode/remapping
scratch and metadata are not included; this is not a total RSS limit.

Only raw spatial 3-D meshes with canonical ownership are supported. Draco is
rejected until decoder ordering and its embedded intra-cell faces are handled.
Missing referenced vertices and invalid face indices raise; incomplete surfaces
are not returned as successful reads. No object-ID selection or mesh attributes
are promised. Construct independent readers for concurrent reads, or serialize
calls to one reader.

Metadata is snapshotted for the reader lifetime. It reuses the root-relative
node and cell-listing caches and passes pre-resolved offsets families into the
link reader. Construct a fresh reader after mutations. Use `with MeshReader(...)`
or `close()` to end the read-only metadata scope. A repeated query still reads
geometry payloads; this is metadata reuse, not an unbounded decoded-mesh cache.

The accompanying Lodstone branch uses one reader for each explicitly selected
surface level. A coarse metavertex pyramid without triangle connectivity is
not a mesh LOD and is rejected. Geometry simplification remains the concern of
the dataset producer or companion tools.
