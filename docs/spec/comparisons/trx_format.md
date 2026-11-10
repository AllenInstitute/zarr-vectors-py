# Zarr Vectors and TRX format

## Terms

**TRX format**
: Tractography Exchange format — a tractography file format designed to
  replace TRK and TCK for cross-software data exchange. A TRX file is a
  ZIP archive (`.trx`) or an equivalent plain folder in which every array
  is its own file. Specified at https://github.com/tee-ar-ex/trx-spec
  (project page: https://tee-ar-ex.github.io). The reference
  implementation is `trx-python`; DIPY and scilpy read and write it.

**Array files**
: Raw little-endian, C-order binary with no file header — not `.npy`. The
  file name carries the array's name, optional second dimension and dtype:
  `positions.3.float16` is an `(NB_VERTICES, 3)` float16 array,
  `offsets.uint64` a 1-D uint64 array, `dps/color.3.uint8` an
  `(NB_STREAMLINES, 3)` uint8 array. Accepted dtypes are int8–64,
  uint8–64, float16/32/64 and `bit` (boolean).

**`positions`**
: `positions.3.<float16|float32|float64>` at the root of the archive: every
  vertex of every streamline, concatenated, in world space (RAS
  millimetres), shape `(NB_VERTICES, 3)`.

**Offset array**
: `offsets.<uint32|uint64>` at the root, of length `NB_STREAMLINES + 1`.
  Streamline `k` spans rows `[offsets[k], offsets[k+1])` of `positions`;
  the last entry is a sentinel equal to `NB_VERTICES`.

**`dps` (data per streamline)**
: One array file per key under `dps/`, shape `(NB_STREAMLINES,)` or
  `(NB_STREAMLINES, N)`. Equivalent to Zarr Vectors `object_attributes/`.

**`dpv` (data per vertex)**
: One array file per key under `dpv/`, shape `(NB_VERTICES,)` or
  `(NB_VERTICES, N)`, rows aligned with `positions`. Equivalent to Zarr
  Vectors `vertex_attributes/`.

**`groups`**
: `groups/<name>.uint32` — a 1-D array of streamline indices. Groups may
  overlap and need not cover every streamline. Equivalent to one row of
  the Zarr Vectors `groups/` array, named through its `group_names`.

**`dpg` (data per group)**
: `dpg/<group name>/<key>.<dtype>`, shape `(1,)` or `(N,)`. Not every group
  has to carry every key. Equivalent to Zarr Vectors `group_attributes/`.

**TRX header**
: `header.json` at the root with exactly four fields: `VOXEL_TO_RASMM`
  (a 4x4 affine, the reference image's voxel-to-world transform),
  `DIMENSIONS` (the reference image's shape, three uint16),
  `NB_STREAMLINES` and `NB_VERTICES`. The affine is *not* applied to
  `positions`, which are already in world space; it is kept so a
  tractogram can be put back into voxel space of its reference image.

**Memory mapping**
: Because array files have no header, each one can be memory-mapped
  directly, at a byte offset inside the ZIP when the archive is stored
  uncompressed (`ZIP_STORED`). A deflated archive has to be extracted
  first. This is the primary efficiency advantage of TRX for local I/O.

---

## Introduction

TRX and Zarr Vectors both store tractography streamline data, but they were designed
for fundamentally different access patterns. TRX prioritises fast sequential
access on local storage via memory mapping — it is ideal for software
pipelines that process all streamlines in sequence. Zarr Vectors prioritises spatial
random access and cloud-native serving — it is ideal for spatial queries
on large datasets, multi-resolution visualisation, and scalable cloud pipelines.

Most tractography workflows begin with data in TRK, TCK, or TRX format.
The converters in the companion package `zarr-vectors-tools` (`ingest_trx`,
`ingest_trk`, `ingest_tck`) convert these formats to Zarr Vectors,
carrying `dps`, `dpv`, groups and `dpg` across as `object_attributes/`,
`vertex_attributes/`, `groups/` and `group_attributes/` respectively.

---

## Technical reference

### Format structure comparison

| Property | TRX | Zarr Vectors (`streamline`) |
|----------|-----|--------------------|
| Container | ZIP archive (`.trx`, stored or deflated) or a plain folder | Zarr v3 group (`.zarrvectors`) |
| Vertex storage | `positions.3.<float>`: flat concatenated array in streamline order, plus `offsets.<uint>` | Spatially chunked `vertices/`, fragment-indexed |
| Spatial index | None | Chunk grid + fragment index |
| Per-vertex attributes (`dpv`) | `dpv/<name>.<dtype>` files, rows aligned with `positions` | `vertex_attributes/<name>/` arrays, rows aligned with `vertices/` cells |
| Per-streamline attributes (`dps`) | `dps/<name>.<dtype>` files, one row per streamline | `object_attributes/<name>/` arrays |
| Groupings | `groups/<name>.uint32` index arrays; per-group data under `dpg/<name>/` | `groups/` ragged array with `group_names`; per-group data in `group_attributes/` |
| Reference image | `header.json`: `VOXEL_TO_RASMM` and `DIMENSIONS`, descriptive only | Not a format concept; the TRX header is kept under `headers/trx/` |
| Positions' coordinate space | Always RAS millimetres | Whatever the writer declares in `multiscales[0].axes` |
| Multi-resolution | No | Yes |
| Cloud access | Not designed for it; the archive is downloaded whole in practice | Native (chunk-level range requests) |
| Write API | Yes (`trx-python`) | Yes (`zarr-vectors-py`) |
| Memory mapping | Yes (fast local sequential access) | No (but chunk caching is equivalent for sequential reads) |

### Data model mapping

#### Vertex positions

TRX stores all streamline vertices as one `(NB_VERTICES, 3)` array in
`positions.3.<dtype>` (float16 by default; float32 and float64 are also
allowed), in streamline order. Streamline boundaries are given by
`offsets.<uint32|uint64>`, which has `NB_STREAMLINES + 1` entries ending in
a sentinel equal to `NB_VERTICES`.

Zarr Vectors stores vertices chunked spatially. The equivalent of the TRX offset
table is `object_index/`, whose per-streamline manifest enumerates every
`(chunk, fragment)` the streamline touches, in traversal order. (Before
Zarr Vectors 0.6.0 this took a primary fragment plus a cross-chunk-link walk;
the manifest now enumerates the chunks directly.)

#### `dpv` → `vertex_attributes/`

Each `dpv/<name>.<dtype>` (or `dpv/<name>.<N>.<dtype>`) file in TRX contains
one row per vertex in the same order as `positions`. In Zarr Vectors, the
equivalent is `vertex_attributes/<name>/`, which stores one row per vertex
in fragment order (per chunk, spatially sorted).

| TRX | Zarr Vectors |
|-----|-----|
| `dpv/fa.float32` | `vertex_attributes/fa/` (float32) |
| `dpv/md.float16` | `vertex_attributes/md/` (float16, or float32 if the writer widens) |
| `dpv/color.3.uint8` | `vertex_attributes/color/` (uint8, shape `(n, 3)`) |

#### `dps` → `object_attributes/`

Each `dps/<name>.<dtype>` file in TRX contains one row per streamline
(row `k` corresponds to streamline `k`). In Zarr Vectors, the equivalent is
`object_attributes/<name>/`.

| TRX | Zarr Vectors |
|-----|-----|
| `dps/mean_fa.float32` | `object_attributes/mean_fa/` (float32) |
| `dps/cluster_id.uint16` | `object_attributes/cluster_id/` (uint16) |
| `dps/commit_colors.3.uint8` | `object_attributes/commit_colors/` (uint8, shape `(n, 3)`) |

TRX `dps` arrays are one- or two-dimensional: `(NB_STREAMLINES,)` or
`(NB_STREAMLINES, N)`. A matrix per streamline, such as a `(2, 3)` pair of
endpoints, is stored flattened as `N = 6`. Zarr Vectors `object_attributes/`
can hold the `(n, 2, 3)` shape directly, so a converter going back to TRX
has to flatten the trailing dimensions.

#### Groups and `dpg`

TRX keeps one `groups/<name>.uint32` file per group, listing the streamline
indices it contains, and optional per-group data under `dpg/<name>/`, one
file per key. Zarr Vectors stores groups as rows of one ragged `groups/`
array; row `g` holds that group's object ids and the array's `group_names`
metadata gives each row its name, so the TRX group name survives.
Per-group data becomes one `group_attributes/<key>/` column of shape
`(G,)` or `(G, C)` across all groups. A TRX group that does not carry a key
gets the column's fill value in its row.

### Metadata and coordinate space

TRX positions are RAS millimetres by definition. `header.json` carries
the reference image's `VOXEL_TO_RASMM` affine and `DIMENSIONS` so that the
tractogram can be related back to that image; the affine is not applied to
the positions on read or write.

Zarr Vectors does not store or apply affine transforms. The TRX ingest in
`zarr-vectors-tools` writes the positions unchanged, declares millimetre
axes on the store, and keeps the TRX header (affine and dimensions) under `headers/trx/`
through the `HeaderRegistry`, from where the TRX export reads it back.
Converting a store whose positions are in another space (for example
TrackVis voxel-millimetres from a TRK ingest) to TRX requires transforming
the positions to RAS millimetres first.

### Performance comparison

| Operation | TRX | Zarr Vectors |
|-----------|-----|-----|
| Sequential read of all streamlines | Very fast (memory mapped) | Fast (sequential chunk reads, ~same throughput) |
| Random access to one streamline by ID | O(1) offset lookup + linear read | O(1) object_index lookup + fragment read |
| Spatial bbox query | O(n) — scan all streamlines | O(chunks × bins) — spatial index |
| Cloud access (S3/GCS) | Archive downloaded whole in practice | Native range requests |
| Spatial query on 1M-streamline dataset | Seconds to minutes | < 1 second for a small bbox |

Zarr Vectors' spatial query advantage is most pronounced for datasets with millions
of streamlines where only a small spatial region is needed. For pipelines
that process all streamlines sequentially, TRX's memory-mapped access
pattern has comparable or better throughput than Zarr Vectors' chunked reads.

### Ingest and export

TRX converters live in the companion package **`zarr-vectors-tools`**.
TRX ingest carries `dps`, `dpv`, groups (with their names) and `dpg`
across; TRX export writes them back. Coarser resolution levels are not
exported to TRX (TRX has no multi-resolution concept), and the exported
TRX does not include the Zarr Vectors chunk layout metadata.

The current converters are not dtype-exact: float16 positions and
attributes are widened to float32, `bit` (boolean) arrays become uint8, and
`dpg` values are stored as float32. A TRX file round-tripped through Zarr
Vectors therefore comes back with equal values but, for those dtypes,
larger files.
