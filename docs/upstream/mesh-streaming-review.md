---
orphan: true
---

# Complete boundary faces without a global connectivity read

Selecting one vertex chunk with `read_mesh(chunks=...)` can drop a triangle
whose other vertices live in neighboring chunks. Selecting all endpoint chunks
recovers that triangle, but still reads unrelated connectivity.

The additive [`MeshReader`](../tutorials/data_types/mesh_streaming.md) selects
stored canonical face owners and resolves their vertex dependencies. Existing
`read_mesh` behavior and the storage format stay unchanged.

## Reproduce

From a checkout, with [uv](https://docs.astral.sh/uv/):

```bash
uv run --extra dev python examples/mesh_read_reproducer.py --output mesh-read-results.json
```

The fixture has one boundary triangle and 64 distant triangles. No Lodstone,
GPU, credentials or scientific download is needed. It checks oriented triangle
coordinates and counts returned Zarr Store buffers, including batched reads.

| Query | Faces | Payload bytes | Metadata bytes |
| --- | ---: | ---: | ---: |
| Existing reader, one vertex chunk | 0 | — | — |
| Existing reader, all three endpoint chunks | 1 | 2,320 | 21,149 |
| Owner reader, first query after discovery | 1 | 272 | 1,719 |
| Owner reader, repeated query | 1 | 272 | 0 |

Discovery costs another 12,476 metadata bytes on this fixture. Both complete
queries return the same oriented triangle. These are local encoded-buffer
measurements, not latency or HTTP wire bytes; sizes depend on store encoding.
[Recorded results](mesh-read-results.json).

## Review scope

Tests cover complete boundary faces, winding, disjoint owner reads, unrelated
payload exclusion, missing dependencies, budget preflight, metadata reuse and
reader cleanup. The reader adds no renderer dependency.

The main API question is whether `building.MeshReader` is the right home for
this low-level read interface. The format-spec page's historical centroid
wording also needs maintainer review; this change corrects the reader tutorial
without changing normative format text.

[Downstream biological demo and geometry checks](https://github.com/kephale/lodstone/blob/880f1d9/docs/biological-mesh.md).
