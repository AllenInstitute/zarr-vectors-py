# Codec pipeline

The Zarr v3 codecs a store's arrays declare, the compression choices
`zarr-vectors-py` offers, and Draco, an optional mesh encoding inside
chunk cells.

## Terms

**Codec**
: A Zarr v3 transformation applied to chunk data on write and reversed
  on read, listed in `codecs` in an array's `zarr.json`. The
  array-to-bytes codec (`vlen-bytes` or `bytes`) comes first; a
  compressor such as `zstd`, `blosc` or `gzip` may follow.

**`sharding_indexed`**
: The Zarr v3 sharding codec. It packs several cells into one stored
  object with an index. See [Sharding](../chunking/sharding.md).

**Encoding**
: The `encoding` attribute of a `vertices` array: `"raw"` (positions as
  little-endian values of the array's `dtype`) or `"draco"` (each cell
  holds a Draco bitstream). It is not a Zarr codec.

## Introduction

Every array uses only codecs that zarr-python 3 provides (`zarr>=3.0`,
`numcodecs>=0.12`), so a generic Zarr reader can decode every cell.
Compression is chosen when an array is created and does not change.
Draco is applied by the mesh writer inside each `vertices` cell; Zarr
sees the result as ordinary variable-length bytes.

## Technical reference

### Codecs by array kind

| Array kind | Array-to-bytes | Compressor |
|------------|----------------|------------|
| Per-chunk arrays (`vertices`, `vertex_fragments`, `link_fragments`, links, vertex, fragment and link attributes) | `vlen-bytes` | the writer's `compressor` (none by default) |
| Object-index integer arrays (for example `object_index/object_ids`) | `bytes` | `zstd` at level 9 or higher, unless the write session sets another (`manifest_codecs` in `zarr_vectors/core/dense_manifests.py`) |
| Other fixed-size arrays (`object_attributes/<name>`, `object_index/manifests`) | `bytes` or `vlen-bytes` | the write session's compressor |

### The `compressor` argument

The type writers and `create_store` take `compressor`, resolved by
`zarr_vectors.encoding.compression.resolve_compressor`:

| `compressor` | Codecs after `vlen-bytes` |
|--------------|---------------------------|
| `None` (default), `"none"`, `False` | none |
| `"zstd"` | `{"name": "zstd", "configuration": {"level": 0, "checksum": false}}` |
| `"blosc"` | `{"name": "blosc", "configuration": {"cname": "zstd", "clevel": 5, "shuffle": "bitshuffle", "typesize": 4, "blocksize": 0}}` |
| a list of codec dicts | the list as given |

Any other value raises `ValueError`. `zarr_vectors.create()` uses
`Layout(compression="auto")`, which resolves to `"zstd"`, or to the
`ZARR_VECTORS_COMPRESSION` environment variable when it is set.

```python
import json

import numpy as np
from zarr_vectors.types.points import write_points

rng = np.random.default_rng(0)
positions = rng.uniform(0, 200, (500, 3)).astype(np.float32)

write_points("gzip.zarrvectors", positions,
             chunk_shape=(100.0, 100.0, 100.0),
             compressor=[{"name": "gzip", "configuration": {"level": 5}}])
print(json.load(open("gzip.zarrvectors/0/vertices/zarr.json"))["codecs"])
```

```
[{'name': 'vlen-bytes', 'configuration': {}}, {'name': 'gzip', 'configuration': {'level': 5}}]
```

### With sharding

A sharded array has a single `sharding_indexed` codec. Its
configuration holds the per-cell codecs and an index encoded with
`bytes` and `crc32c`. For `compressor="zstd"` and `shard_shape=2` on a
3-D store:

```json
[{"name": "sharding_indexed",
  "configuration": {
    "chunk_shape": [1, 1, 1],
    "codecs": [{"name": "vlen-bytes", "configuration": {}},
               {"name": "zstd", "configuration": {"level": 0, "checksum": false}}],
    "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}},
                     {"name": "crc32c"}],
    "index_location": "end"}}]
```

The array's `chunk_grid` then has chunk shape `[2, 2, 2]`. An octree
store uses the `zarr_octree_sharding` codec from the `zarr-octree`
package instead; see [Octree chunks](../chunking/octree_chunks.md).

### Draco mesh encoding

`write_mesh(..., encoding="draco", draco_quantization_bits=11)` writes
each chunk's vertices and intra-chunk faces as one Draco bitstream in
the chunk's `vertices` cell, and sets `"encoding": "draco"`. Faces that
cross chunks stay in `links/`.

- Draco applies only to 3-D meshes.
- Positions are quantised to `draco_quantization_bits`, so they are lossy.
- A chunk is a single fragment shared by all its objects, so
  `read_mesh(object_ids=...)` raises `ArrayError`.
- Encoding needs `DracoPy` (extra `zarr-vectors[draco]`) or the
  `draco_encoder` and `draco_decoder` tools; with neither, `DracoError`
  is raised.

See [Mesh](../geometry_types/mesh.md).

### Validation

No level inspects codec lists. Level 3 (`validate/consistency.py`)
decodes cells, so a cell that cannot be decoded is reported there. See
[Validation](../validation/overview.md).
