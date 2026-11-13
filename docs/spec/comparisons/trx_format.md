# Zarr Vectors and TRX

This page compares Zarr Vectors streamline stores with the TRX
tractography format, maps TRX arrays onto Zarr Vectors arrays, and
describes the converters between them.

## Terms

**TRX**
: A tractography file format defined at
  <https://tractography-file-format.github.io>, with a reference
  implementation in `trx-python`. A TRX file is a ZIP archive or a
  directory of uncompressed binary arrays plus a `header.json`.

**`positions` and `offsets`**
: The TRX arrays holding every streamline's vertices concatenated, and
  the index of each streamline's first vertex.

**`dpv`, `dps`, `dpg`**
: TRX data per vertex, per streamline and per group: arrays aligned
  with the vertices, the streamlines or the groups.

**`groups`**
: Named TRX lists of streamline indices, such as bundles.

**TRX header**
: `header.json`, holding the reference image's `VOXEL_TO_RASMM` affine
  and `DIMENSIONS` and the streamline and vertex counts. TRX positions
  are in RAS millimetres.

---

## Introduction

TRX keeps each array contiguous so that a program can memory-map a
file and stream every streamline in order. A Zarr Vectors store chunks
vertices by position, so a program can read the streamlines that pass
through a region, read one streamline by ID, or read a coarser
resolution level, without reading the rest. TRX suits whole-file
processing on local storage; a Zarr Vectors store suits spatial access
and object-store hosting. The converters in `zarr-vectors-tools` move
data between the two.

---

## Technical reference

### Structure

| Property | TRX | Zarr Vectors `streamline` |
|----------|-----|---------------------------|
| Container | ZIP archive or directory | Zarr v3 hierarchy (`.zarrvectors`) |
| Vertices | One `positions` array, concatenated | Per-chunk cells, sorted into fragments (see [Chunk arrays](../layout/chunk_arrays.md)) |
| Streamline boundaries | `offsets` | Object manifests listing each streamline's chunks and fragments (see [Object manifest](../object_model/object_manifest.md)) |
| Spatial index | None | Chunk grid and fragment index |
| Per-vertex data | `dpv/` | `vertex_attributes/<name>` |
| Per-streamline data | `dps/` | `object_attributes/<name>` |
| Groups | `groups/` | `groups` array, with names in its `group_names` attribute |
| Per-group data | `dpg/` | `group_attributes/<name>` |
| Reference image | `header.json` | Kept as a `trx` header by `zarr-vectors-tools` (see below) |
| Resolution levels | One | Several (see [Multiscale metadata](../multiscale/multiscale_metadata.md)) |

### Data model mapping

`ingest_trx` (`zarr_vectors_tools.convert.ingest.trx`) writes a
`streamline` store with `write_polylines`:

- `positions` and `offsets` become one object per streamline, in file
  order.
- Each `dpv` array becomes a vertex attribute, kept in the file's dtype
  and reordered with the vertices.
- Each `dps` array becomes an object attribute, in the file's dtype.
- Each TRX group becomes a row of the `groups` array, named after the
  group; `dpg` arrays become group attributes.
- `VOXEL_TO_RASMM`, `DIMENSIONS` and the array names are kept in a
  `trx` header (the `zarr_vectors.headers` registry) so that an export
  can write them back.
- The axes record millimetres. Positions are stored as given, in RAS
  millimetres.

`ingest_trx` can also compute per-streamline length and endpoints,
drop streamlines outside a length range, and average `dpv` scalars per
streamline.

### Export

`export_trx` (`zarr_vectors_tools.convert.export.trx`) writes one
resolution level to a TRX file: vertex attributes as `dpv`, object
attributes as `dps`, named groups as `groups` with their attributes as
`dpg`, and the reference image from the store's TRX or TRK header. A
store holding TrackVis voxel-millimetre positions is converted to RAS
millimetres. It can export a subset by object ID, group ID or chunk.
The chunk layout and the other levels are not part of TRX and are not
written.

The commands are:

```bash
zvtools convert bundles.trx bundles.zarrvectors --chunk-shape 20,20,20
zvtools convert bundles.zarrvectors cst.trx --group-id 0
```

Both need the `zarr-vectors-tools[trx]` extra (`trx-python`). TRK and
TCK files have their own converters (`ingest_trk`, `ingest_tck`,
`export_trk`). See the zarr-vectors-tools
[streamlines page](https://zarr-vectors-tools.readthedocs.io/en/latest/convert/streamlines.html).

### Validation

`zarr-vectors-py` does not validate TRX files. A store written by
`ingest_trx` is a `streamline` store; level 4 of
`zarr_vectors.validate.validate` checks that its `links_convention` is
`implicit_sequential`. See [Validation](../validation/overview.md).
