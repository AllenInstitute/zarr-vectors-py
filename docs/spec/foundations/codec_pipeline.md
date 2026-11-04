# Codec pipeline

## Terms

**Codec**
: A single transformation applied to chunk data. A codec takes bytes (or an
  array) as input and produces bytes as output. Codecs are composable;
  an ordered sequence of codecs forms a pipeline.

**Array-to-bytes codec**
: The first codec in a Zarr v3 pipeline, which converts a typed N-dimensional
  array into a flat byte sequence. Zarr v3 requires exactly one such codec.
  The standard array-to-bytes codec is `bytes`, which serialises the array
  in C-order with a declared byte order.

**Byte-to-byte codec**
: A codec that transforms one byte sequence into another — typically
  compression or encryption. Multiple byte-to-byte codecs may be stacked.
  Common codecs: `blosc`, `zstd`, `gzip`, `lz4`.

**Blosc**
: A high-performance meta-compressor that internally uses one of several
  compression algorithms (zstd, lz4, blosclz, zlib). Blosc also performs
  byte-shuffling before compression, which improves compression ratios on
  typed numeric arrays. In `zarr-vectors-py` it is available as
  `compressor="blosc"` but is **not** the default.

**Sharding codec** (`sharding_indexed`)
: A Zarr v3 codec that packs multiple logical chunks into a single storage
  key. Reduces file count and cloud storage request overhead. Discussed in
  detail in [Sharding](../chunking/sharding.md).

**Draco**
: Google's geometry compression library, optimised for 3-D mesh data.
  `write_mesh(encoding="draco")` encodes each chunk's vertices and
  intra-chunk faces into one Draco bitstream (via DracoPy, or the
  `draco_encoder` / `draco_decoder` command-line tools as a fallback).
  It is applied by the mesh writer, **not** registered as a Zarr codec.
  Requires `zarr-vectors[draco]`.

**Fill value**
: The value a chunk position reads back as when it was never written. In
  Zarr v3 it is declared per array in `zarr.json`. The per-spatial-chunk
  arrays (`vertices`, `vertex_fragments`, `links/...`,
  `vertex_attributes/...`) are vlen-bytes arrays, so an unwritten cell
  reads as empty bytes. Dense object-level arrays such as
  `object_attributes/<name>` use NaN for float dtypes and a sentinel for
  integer dtypes to mark absent rows.

---

## Introduction

A Zarr v3 codec pipeline controls how chunk data is serialised to bytes and
compressed before storage. Zarr Vectors does not mandate a specific codec; any Zarr
v3-compatible pipeline is valid. `zarr-vectors-py` exposes one knob,
`compressor=`, and one geometry-specific encoding (Draco for meshes).

Understanding the codec pipeline matters for contributors adding new
geometry types and for users who want to tune compression ratios or
compatibility with downstream tools.

---

## Technical reference

### Defaults depend on the entry point

| Entry point | Default compression |
|-------------|---------------------|
| Type-level writers (`write_points`, `write_lines`, `write_polylines`, `write_graph`, `write_mesh`) | **None**: `bytes` only |
| Dataset API (`zarr_vectors.create` / `Dataset.add_*`) | `zstd` (`Layout(compression="auto")`), or the value of `$ZARR_VECTORS_COMPRESSION` when set |

Both can be overridden: `compressor=` on a type-level writer, and
`zv.Layout(compression=...)` on the schema for the dataset API.
There is no `codec_config=` argument.

### Per-array codec configuration

`zarr-vectors-py` writes every chunk array with the **same codec
pipeline**, set per-write via the `compressor=` kwarg on the type-level
write functions (`write_points`, `write_graph`, `write_polylines`,
`write_lines`, `write_mesh`), or via `create_store(compressor=...)`. The
type-level default is `bytes`-only — no compression. This keeps the fast
async PUT path active (compression CPU is typically much smaller than
cloud PUT latency, so the win is negligible against object stores). Opt
into compression explicitly when disk size matters more than throughput.

Because a store's chunk arrays are created once and reused, the codec
chosen when an array is first created applies for the life of that array
(`create_store(compressor=...)` fixes it for `vertices` /
`vertex_fragments`).

Accepted `compressor=` values (see
[`resolve_compressor`](../../../zarr_vectors/encoding/compression.py)):

| Value | Resulting `codecs` list |
|-------|-------------------------|
| `None` *(default)*, `"none"`, or `False` | `[{"name": "bytes"}]` — uncompressed; fast async PUT path. |
| `"zstd"` | `[{"name": "bytes"}, {"name": "zstd", "configuration": {"level": 0, "checksum": false}}]` — zarr v3's own default compressor.  Triggers the sync codec-encoding path. |
| `"blosc"` | `bytes` + Blosc (`cname=zstd`, `clevel=5`, `shuffle=bitshuffle`, `typesize=4`, `blocksize=0`). |
| `list[dict]` | Caller-supplied Zarr V3 codec specs, passed through; a leading `bytes` codec is added if absent. |

Any other value raises `ValueError`.

```python
import numpy as np
from zarr_vectors.types.points import write_points

positions = np.random.default_rng(0).uniform(0, 100, (10_000, 3)).astype(np.float32)
write_points(
    "scan.zarrvectors",
    positions,
    chunk_shape=(50., 50., 50.),
    compressor="blosc",
)
```

Per-array-type codec selection (different codecs for `vertices/`
vs `vertex_fragments/` vs `links/`) is **not supported**: every chunk
array in a single `write_*` call gets the same codec.

```{note}
**`vertex_fragments/` and `link_fragments/`** are pre-packed binary
layouts with little redundancy for a general-purpose compressor to
exploit. They still go through whatever `compressor=` was passed. See
[Fragment-index arrays](../layout/fragment_index_arrays.md) for the byte
layout and [Object manifest](../object_model/object_manifest.md) for the
manifest-blob format.

**`object_index/manifests`** is a vlen-bytes array (variable-length byte
array). Each entry is one object's encoded manifest blob; per-object
random access reads only the zarr chunk containing the requested OID.

The Zarr V3 specification for variable-length byte arrays is still
in development (tracked at
[zarr-extensions](https://github.com/zarr-developers/zarr-extensions/tree/main/data-types));
stores written with `vlen-bytes` may need to be re-encoded if the
eventual spec lands incompatibly. `zarr-vectors-py` silences zarr's
`UnstableSpecificationWarning` for this reason on import.
```

### Draco encoding (mesh geometry)

`write_mesh(..., encoding="draco", draco_quantization_bits=11)` replaces
each chunk's vertices and intra-chunk faces with one Draco bitstream. It
applies only to 3-D stores and requires `zarr-vectors[draco]`; boundary
faces stay in `links/`. The encoder's compression level is fixed at 7
(`draco_encode_mesh` defaults). There is no `use_draco=` or
`draco_quantization=` keyword.

A Draco-encoded store cannot be decoded without DracoPy (or the Draco CLI
tools); the decoder raises `DracoError`. Generic Zarr tools can open the
arrays but cannot interpret the Draco payload.

#### Quantisation and precision

Draco uses integer quantisation of vertex positions: the more
`quantization_bits`, the higher the positional precision (relative to the
chunk's bounding box, roughly `1 / 2^bits` per axis) and the lower the
compression ratio. The default is 11 bits. Compression ratios depend on
the mesh; this documentation does not promise specific figures.

### Choosing a codec

| Scenario | Recommended |
|----------|-------------|
| Default / general purpose, local | `compressor=None` or `"zstd"` |
| Float-heavy data where disk size matters | `compressor="blosc"` |
| Custom pipeline | `compressor=[{...}, ...]` of Zarr v3 codec specs |
| Mesh geometry with size budget (3-D) | `encoding="draco"` |
| Compatibility with non-Zarr tools | `compressor=None` (`bytes` only) |
| Cloud with many small chunks | Consider [sharding](../chunking/sharding.md) first |

### Codec availability and compatibility

Compression codecs are provided by the installed `zarr` (v3) and
`numcodecs` packages; `zarr-vectors-py` passes the specs through. The
named shorthands above (`bytes`, `zstd`, `blosc`) are always accepted.
Beyond that, anything a custom `list[dict]` names must be available to
every consumer.

Codecs requiring extras:

- Draco encoding — requires `zarr-vectors[draco]`.
- `sharding_indexed` — built into Zarr v3; selected via `shard_shape=`,
  no extra required.

When distributing Zarr Vectors stores, prefer `bytes` or `zstd` for
maximum compatibility, and say so wherever you document the data if you
use anything else. There is no format-defined `notes` field; you can use
`Dataset.metadata` (user metadata) for such notes.
