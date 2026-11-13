# Zarr Vectors and TRX

How Zarr Vectors streamline stores differ from TRX files, and when to
use each.

## What TRX is for

[TRX](https://tractography-file-format.github.io) is a tractography
format, with a reference implementation in `trx-python`. A TRX file is
a ZIP archive or a directory of uncompressed binary arrays plus a
`header.json`. It keeps each array contiguous, so a program can
memory-map the file and stream every streamline in order.

A Zarr Vectors store chunks vertices by position. A program can read
the streamlines that pass through a region, one streamline by ID, or a
coarser resolution level, without reading the rest.

## Comparison

| | TRX | Zarr Vectors `streamline` |
|---|---|---|
| Container | ZIP archive or directory | Zarr v3 hierarchy |
| Vertices | One concatenated `positions` array | Per-chunk cells, sorted into fragments |
| Streamline boundaries | `offsets` | [Object manifests](../../spec/object_model/object_manifest.md) |
| Spatial index | None | Chunk grid and fragment index |
| Per-vertex data | `dpv/` | `vertex_attributes/<name>` |
| Per-streamline data | `dps/` | `object_attributes/<name>` |
| Groups and per-group data | `groups/`, `dpg/` | `groups` array and `group_attributes/<name>` |
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

Ingest keeps dtypes, groups and the reference image, so an export can
write them back. Export writes one resolution level; the chunk layout
and other levels are not part of TRX. See the zarr-vectors-tools
[streamlines page](https://zarr-vectors-tools.readthedocs.io/en/latest/convert/streamlines.html).
