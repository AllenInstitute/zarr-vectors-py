# Ingesting other formats

`zarr-vectors` reads and writes Zarr Vectors stores only. Converters between
stores and file formats are in the companion package `zarr-vectors-tools`,
with its command-line tool `zvtools`:

```bash
pip install zarr-vectors-tools
zvtools convert points.csv points.zarrvectors --chunk-shape 250,250,250   # ingest
zvtools convert points.zarrvectors exported.csv                           # export
```

| Geometry | Formats |
|----------|---------|
| Points | CSV/XYZ, PLY, LAS/LAZ, H5AD |
| Streamlines | TRK, TRX, TCK |
| Skeletons | SWC, Neuroglancer precomputed |
| Meshes and surfaces | OBJ, STL, Neuroglancer precomputed, GIFTI, FreeSurfer |
| Graphs and line segments | GraphML, edge lists |

Some formats are read only, and several need an extra, such as
`pip install "zarr-vectors-tools[las]"`. The `zarr-vectors-tools`
documentation lists each format's direction, extra and options.

To write data you already hold as arrays, use the `zarr_vectors` API
directly; see the [Quickstart](../../getting_started/quickstart.md).
