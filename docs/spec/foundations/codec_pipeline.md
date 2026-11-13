# Codec pipeline

This page describes the Zarr v3 codecs a store's arrays declare, the
compression choices `zarr-vectors-py` offers, and Draco, the optional
mesh encoding applied inside chunk cells.

## Terms

**Codec**
: A Zarr v3 transformation applied to chunk data on write and reversed
  on read, declared in the `codecs` list of an array's `zarr.json`.

**Array-to-bytes codec**
: The codec that turns an array chunk into bytes. Per-chunk arrays use
  `vlen-bytes`; fixed-size numeric arrays use `bytes`.

**Bytes-to-bytes codec (compressor)**
: A codec that transforms bytes, such as `zstd`, `blosc` or `gzip`. It
  follows the array-to-bytes codec in `codecs`.

**`sharding_indexed`**
: The Zarr v3 sharding codec. It packs several cells into one stored
  object with an index; the cell's own codecs move inside its
  configuration. See [Sharding](../chunking/sharding.md).

**Encoding**
: The `encoding` attribute of a `vertices` array: `"raw"` (positions as
  little-endian values of the array's `dtype`) or `"draco"` (each cell
  holds a Draco bitstream). It describes the cell contents, not a Zarr
  codec.

---

## Introduction

Every array in a store declares an ordinary Zarr v3 codec list, built
only from codecs that zarr-python provides, so a generic Zarr reader
can decode the bytes of every cell. Compression is a choice made when
an array is created, and an array keeps its codecs afterwards.

Draco is different: it is applied by the mesh writer to the contents of
each `vertices` cell, and is recorded in the array's `encoding`
attribute. Zarr sees a Draco cell as ordinary variable-length bytes.

---

## Technical reference

### Codecs by array kind

| Array kind | Array-to-bytes codec | Compressor |
|------------|----------------------|------------|
| Per-chunk arrays (`vertices`, `vertex_fragments`, `link_fragments`, `links/<delta>/<offsets>`, `vertex_attributes/<name>`, `fragment_attributes/<name>`, `link_attributes/…`) | `vlen-bytes` | the writer's `compressor` (none by default) |
| Object-index integer arrays (for example `object_index/object_ids`) | `bytes` | `zstd` at level 9 or higher, unless the write session uses another compressor |
| Other fixed-size arrays (`object_attributes/<name>`, `object_index/manifests`) | `bytes` or `vlen-bytes` | the write session's compressor |

The object-index rule is in `manifest_codecs`
(`zarr_vectors/core/dense_manifests.py`).

### Choosing a compressor

The type writers (`write_points`, `write_polylines`, `write_lines`,
`write_graph`, `write_mesh`, the skeleton writers) and `create_store`
take a `compressor` argument, resolved by
`zarr_vectors.encoding.compression.resolve_compressor`:

| `compressor` | Codecs after `vlen-bytes` |
|--------------|---------------------------|
| `None` (default), `"none"`, `False` | none |
| `"zstd"` | `{"name": "zstd", "configuration": {"level": 0, "checksum": false}}` |
| `"blosc"` | `{"name": "blosc", "configuration": {"cname": "zstd", "clevel": 5, "shuffle": "bitshuffle", "typesize": 4, "blocksize": 0}}` |
| a list of codec dicts | the list as given |

Any other value raises `ValueError`. A store created through
`zarr_vectors.create()` uses `Layout(compression="auto")`, which
resolves to `"zstd"`, or to the value of the
`ZARR_VECTORS_COMPRESSION` environment variable when it is set.

This example writes the same points with `zstd` and with a `gzip`
codec list, then prints the codecs of each `vertices` array:

```python
import json

import numpy as np
from zarr_vectors.types.points import read_points, write_points

rng = np.random.default_rng(0)
positions = rng.uniform(0, 200, (500, 3)).astype(np.float32)

write_points("zstd.zarrvectors", positions,
             chunk_shape=(100.0, 100.0, 100.0), compressor="zstd")
write_points("gzip.zarrvectors", positions,
             chunk_shape=(100.0, 100.0, 100.0),
             compressor=[{"name": "gzip", "configuration": {"level": 5}}])

for path in ("zstd.zarrvectors", "gzip.zarrvectors"):
    meta = json.load(open(f"{path}/0/vertices/zarr.json"))
    print(meta["codecs"])
    assert read_points(path)["vertex_count"] == 500
```

```
[{'name': 'vlen-bytes', 'configuration': {}}, {'name': 'zstd', 'configuration': {'level': 0, 'checksum': False}}]
[{'name': 'vlen-bytes', 'configuration': {}}, {'name': 'gzip', 'configuration': {'level': 5}}]
```

### With sharding

When a store is sharded (`shard_shape=` on a writer, or `Layout` with
shard cells), the array's `codecs` is a single `sharding_indexed`
entry. Its configuration holds the per-cell codecs and an index
encoded with `bytes` and `crc32c`. For `compressor="zstd"` and
`shard_shape=2` on a 3-D store:

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

The array's own `chunk_grid` then has chunk shape `[2, 2, 2]`. An octree
store uses the `zarr_octree_sharding` codec from the `zarr-octree`
package instead of `sharding_indexed`; see
[Octree chunks](../chunking/octree_chunks.md).

### Draco mesh encoding

`write_mesh(..., encoding="draco", draco_quantization_bits=11)` writes
each chunk's vertices and its intra-chunk faces as one Draco bitstream
in that chunk's `vertices` cell, and stamps `"encoding": "draco"` on the
`vertices` array. Faces whose vertices fall in different chunks stay in
the `links/` family. Rules:

- Draco applies only to 3-D meshes.
- Each chunk is one bitstream shared by every object in it, so a chunk
  is a single fragment and `read_mesh(object_ids=...)` raises
  `ArrayError` on a Draco store.
- `draco_quantization_bits` sets the position quantisation; positions
  are lossy at that precision.
- Encoding uses the `DracoPy` package (extra `zarr-vectors[draco]`), or
  the `draco_encoder` and `draco_decoder` command-line tools when
  `DracoPy` is missing; with neither, `DracoError` is raised.

The mesh layout is in [Mesh](../geometry_types/mesh.md).

### Availability

`vlen-bytes`, `bytes`, `zstd`, `blosc`, `gzip`, `crc32c` and
`sharding_indexed` are codecs that zarr-python 3 provides; the package
requires `zarr>=3.0` and `numcodecs>=0.12`. Zarr may emit
`UnstableSpecificationWarning` for the `variable_length_bytes` data
type; `zarr-vectors-py` filters it on import.

### Validation

No validation level inspects codec lists. Level 3
(`validate/consistency.py`) decodes cells through the declared codecs,
so a cell whose codecs cannot be decoded is reported there. See
[Validation](../validation/overview.md).
