# Zarr Vectors and TRX

How Zarr Vectors streamline stores differ from TRX files, and when to
use each.

## What TRX is for

[TRX](https://github.com/tee-ar-ex/trx-spec) is a tractography exchange
format, with a reference implementation in `trx-python`; DIPY and scilpy
also read and write it. A TRX file is a ZIP archive (`.trx`) or a plain
directory with the same contents: a `header.json` and one file per array.
Each array file is raw little-endian, C-order binary with no header; its
name gives the array's name, optional second dimension and dtype, as in
`positions.3.float16` or `offsets.uint64`. Because the arrays have no
header, a program can memory-map them and stream every streamline in
order. That works inside a ZIP only when it is stored uncompressed; a
deflated archive must be extracted first.

`header.json` holds exactly `VOXEL_TO_RASMM`, `DIMENSIONS`,
`NB_STREAMLINES` and `NB_VERTICES`. Positions are already in RAS
millimetres; the affine is not applied to them, and only relates the
tractogram back to its reference image.

A Zarr Vectors store chunks vertices by position. A program can read
the streamlines that pass through a region, one streamline by ID, or a
coarser resolution level, without reading the rest.

## Comparison

| | TRX | Zarr Vectors `streamline` |
|---|---|---|
| Container | ZIP archive or directory | Zarr v3 hierarchy |
| Vertices | One concatenated `positions` array | Per-chunk cells, sorted into fragments |
| Streamline boundaries | `offsets`, `NB_STREAMLINES + 1` entries | [Object manifests](../../spec/object_model/object_manifest.md) |
| Spatial index | None | Chunk grid and fragment index |
| Per-vertex data | `dpv/`, rows aligned with `positions` | `vertex_attributes/<name>` |
| Per-streamline data | `dps/`, 1-D or 2-D | `object_attributes/<name>`, any trailing shape |
| Groups | `groups/<name>.uint32` streamline indices; may overlap | One row of the `groups` array per group, named by `group_names` |
| Per-group data | `dpg/<group>/<key>`; a group may lack a key | One `group_attributes/<key>` column; a missing key gets the fill value |
| Reference image | `header.json` (`VOXEL_TO_RASMM`, `DIMENSIONS`) | Kept in a `trx` header by `zarr-vectors-tools` |
| Positions | RAS millimetres | Stored as given; axes record millimetres |
| Resolution levels | One | Several |

## When to use which

Use TRX for whole-file processing on local storage. Use Zarr Vectors
for spatial queries and hosting in an object store.

`zarr-vectors-tools` converts both ways (extra `zarr-vectors-tools[trx]`):

```bash
zvtools convert bundles.trx bundles.zarrvectors --chunk-shape 20,20,20
zvtools convert bundles.zarrvectors cst.trx --group-id 0
```

Ingest carries `dpv`, `dps`, groups with their names, `dpg` and the TRX
header across, and export writes them back. It is not dtype-exact:
float16 positions and attributes widen to float32, `bit` arrays become
uint8, and `dpg` values are stored as float32. Integer arrays keep their
dtype. A multi-dimensional `object_attributes` array is flattened to 2-D
on export. Export writes one resolution level; the chunk layout and other
levels are not part of TRX. See the zarr-vectors-tools
[streamlines page](https://zarr-vectors-tools.readthedocs.io/en/latest/convert/streamlines.html).
