# Precomputed export

Stock Neuroglancer reads its own precomputed format but not Zarr Vectors
stores. The companion package `zarr-vectors-tools` exports a store to a
precomputed layer (`zarr_vectors_tools.convert.export.precomputed`):

```bash
zvtools convert STORE OUTPUT --format precomputed
```

`zarr-vectors-py` itself has no precomputed export. See the zarr-vectors-tools
[Visualise](https://zarr-vectors-tools.readthedocs.io/en/latest/visualise.html)
page for the geometry types it exports and how to view the result.
