# ZVF and TRX format

## Terms

**TRX format**
: Tractography Exchange format — a ZIP-based tractography file format
  designed to replace TRK and TCK for cross-software data exchange. Defined
  at https://tractography-file-format.github.io. Implemented in
  `trx-python` and supported by DIPY, MRtrix, Mrtrix3Tissue, and others.

**`dps` (data per streamline)**
: Per-streamline scalar or matrix arrays in TRX, stored under `dps/` in
  the ZIP archive. Each array has one row per streamline. Equivalent to
  ZVF `object_attributes/`.

**`dpp` (data per point)**
: Per-vertex scalar or matrix arrays in TRX, stored under `dpp/` in the
  ZIP archive. Each array has one value per vertex across all streamlines,
  in concatenated order. Equivalent to ZVF `attributes/`.

**TRX header**
: A JSON file `header.json` at the root of the ZIP archive, storing the
  affine transform, voxel dimensions, voxel-to-rasmm transform, and
  data space identifier.

**Memory mapping**
: TRX arrays are stored as raw binary `.npy` files within the ZIP,
  allowing direct memory mapping on POSIX systems without decompression.
  This is the primary efficiency advantage of TRX for local I/O.

**Offset array**
: TRX stores the lengths (or cumulative offsets) of individual streamlines
  in `streamlines/offsets.int64.npy`. Given offsets `[0, 40, 85, …]`,
  streamline `k` spans points `[offsets[k], offsets[k+1])` of
  `streamlines/data.float32.npy`.

---

## Introduction

TRX and ZVF both store tractography streamline data, but they were designed
for fundamentally different access patterns. TRX prioritises fast sequential
access on local storage via memory mapping — it is ideal for software
pipelines that process all streamlines in sequence. ZVF prioritises spatial
random access and cloud-native serving — it is ideal for spatial queries
on large datasets, multi-resolution visualisation, and scalable cloud pipelines.

Most tractography workflows begin with data in TRK, TCK, or TRX format.
The ZVF ingest pipeline (`ingest_trx`, `ingest_trk`, `ingest_tck`) converts
these formats to ZVF, preserving all `dps` and `dpp` attributes as
`object_attributes/` and `attributes/` respectively.

---

## Technical reference

### Format structure comparison

| Property | TRX | ZVF (`streamline`) |
|----------|-----|--------------------|
| Container | ZIP archive (`.trx`) | Directory tree (`.zarrvectors`) |
| Vertex storage | Flat concatenated array + offset table | Spatially chunked, fragment-indexed |
| Spatial index | None | Chunk grid + fragment index |
| Per-vertex attributes (`dpp`) | Separate `.npy` files (same layout as vertices) | `attributes/<name>/` arrays |
| Per-streamline attributes (`dps`) | Separate `.npy` files (one row per streamline) | `object_attributes/<name>/` arrays |
| Affine transform | Stored in `header.json` | Not stored; apply before writing |
| Multi-resolution | No | Yes |
| Cloud access | Not designed for (ZIP download required) | Native (chunk-level range requests) |
| Write API | Yes (`trx-python`) | Yes (`zarr-vectors-py`) |
| Memory mapping | Yes (fast local sequential access) | No (but chunk caching is equivalent for sequential reads) |
| Groupings | `groups/<name>/offsets.npy` | `groupings/` array |

### Data model mapping

#### Vertex positions

TRX stores all streamline vertices as a flat `(total_points, 3)` float32
array in `streamlines/data.float32.npy`. Streamline boundaries are given
by `streamlines/offsets.int64.npy` (cumulative sum of streamline lengths).

ZVF stores vertices chunked spatially. The equivalent of the TRX offset
table is `object_index/`, whose per-streamline manifest enumerates every
`(chunk, fragment)` the streamline touches, in traversal order. (Before
ZVF 0.6.0 this took a primary fragment plus a cross-chunk-link walk;
the manifest now enumerates the chunks directly.)

#### `dpp` → `attributes/`

Each `dpp/<name>.<dtype>.npy` file in TRX contains one value per vertex in
the same concatenated order as `streamlines/data`. In ZVF, the equivalent
is `attributes/<name>/`, which stores one value per vertex in fragment order
(per-chunk, spatially sorted).

| TRX | ZVF |
|-----|-----|
| `dpp/fa.float32.npy` | `attributes/fa/` (float32) |
| `dpp/md.float32.npy` | `attributes/md/` (float32) |
| `dpp/t1.float32.npy` | `attributes/t1/` (float32) |

#### `dps` → `object_attributes/`

Each `dps/<name>.<dtype>.npy` file in TRX contains one value per streamline
(row `k` corresponds to streamline `k`). In ZVF, the equivalent is
`object_attributes/<name>/`.

| TRX | ZVF |
|-----|-----|
| `dps/mean_fa.float32.npy` | `object_attributes/mean_fa/` (float32) |
| `dps/cluster_id.int16.npy` | `object_attributes/cluster_id/` (int16) |
| `dps/endpoints.float32.2x3.npy` | `object_attributes/endpoints/` (float32, shape (n, 2, 3)) |

Matrix `dps` attributes (shape `(n_streamlines, K, M)`) are supported in
both formats. ZVF stores them as `object_attributes/<name>/` with shape
`(n_objects, K, M)`.

#### Groups

TRX groups are sub-ZIP directories: `groups/<name>/offsets.int64.npy`
lists the indices of streamlines in the group. ZVF stores groups in
`groupings/` with group names in `groupings_attributes/name/`.

### TRX metadata not preserved in ZVF

TRX stores an affine transform from voxel space to RAS mm in `header.json`.
ZVF does not store or apply affine transforms; the caller must apply the
affine to vertex positions before writing to ZVF. The affine can be stored
in root `.zattrs` under `custom_metadata.affine` for provenance, but it
will not be automatically applied by `zarr-vectors-py`.

TRX ingest (via the companion package **`zarr-vectors-tools`**) exposes
an `apply_affine` keyword that controls whether the TRX header affine
is applied to vertex positions before they are written to ZVF.

### Performance comparison

| Operation | TRX | ZVF |
|-----------|-----|-----|
| Sequential read of all streamlines | Very fast (memory mapped) | Fast (sequential chunk reads, ~same throughput) |
| Random access to one streamline by ID | O(1) offset lookup + linear read | O(1) object_index lookup + fragment read |
| Spatial bbox query | O(n) — scan all streamlines | O(chunks × bins) — spatial index |
| Cloud access (S3/GCS) | Requires full download | Native range requests |
| Spatial query on 1M-streamline dataset | Seconds to minutes | < 1 second for a small bbox |

ZVF's spatial query advantage is most pronounced for datasets with millions
of streamlines where only a small spatial region is needed. For pipelines
that process all streamlines sequentially, TRX's memory-mapped access
pattern has comparable or better throughput than ZVF's chunked reads.

### Ingest and export

TRX converters live in the companion package **`zarr-vectors-tools`**.
TRX ingest preserves all dps and dpp attributes; TRX export is lossless
for the base level (level 0). Coarser resolution levels are not exported
to TRX (TRX has no multi-resolution concept), and the exported TRX does
not include the ZVF chunk layout metadata.
