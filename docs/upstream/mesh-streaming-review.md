---
orphan: true
---

# Reproduce mesh reads

From a checkout, with [uv](https://docs.astral.sh/uv/):

```bash
uv run --extra dev python examples/mesh_read_reproducer.py --output mesh-read-results.json
```

One triangle crosses three chunks; 64 triangles sit farther away. The script
checks winding and counts returned Zarr Store buffers. No viewer is needed.

| Query | Faces | Payload bytes | Metadata bytes |
| --- | ---: | ---: | ---: |
| Existing reader, one vertex chunk | 0 | — | — |
| Existing reader, all endpoint chunks | 1 | 2,320 | 21,149 |
| Owner reader, first query | 1 | 272 | 1,719 |
| Owner reader, repeat | 1 | 272 | 0 |

Discovery adds 12,476 metadata bytes. Complete queries return the same oriented
triangle. Sizes depend on encoding; these are local buffers, not latency or HTTP
wire bytes. [Recorded results](mesh-read-results.json).

[Reader usage and limits](../tutorials/data_types/mesh_streaming.md) ·
[Biological viewer demo](https://github.com/kephale/lodstone/blob/880f1d9/docs/biological-mesh.md)
