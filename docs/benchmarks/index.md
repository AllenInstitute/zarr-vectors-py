# Benchmarks

Most benchmarks for Zarr Vectors live in the companion package
`zarr-vectors-tools`, next to the third-party readers they compare against.
This repository keeps one benchmark suite and one performance regression
test.

## In zarr-vectors-tools

The suites are in
[`benchmarks/`](https://github.com/AllenInstitute/zarr-vectors-tools/tree/main/benchmarks)
of the zarr-vectors-tools repository. They need `pandas`, `matplotlib` and
`jupyter`, which this package does not depend on.

| Suite | Question it answers | Contents |
|-------|---------------------|----------|
| `paper/` | The comparisons made in the paper | Scripts |
| `formats/` | How does Zarr Vectors compare with PLY, CSV, TRX, GraphML, SWC and OBJ? | 3 notebooks |
| `internals/` | How does Zarr Vectors scale with size, geometry type, backend, pyramid depth, bounding-box query, chunk shape, codec and edit cost? | 8 notebooks |
| `filtering/` | Is numpy under Pyodide fast enough to filter streamlines in a browser? | Scripts |

## In this repository

- [`benchmarks/octree/`](https://github.com/AllenInstitute/zarr-vectors-py/tree/main/benchmarks/octree)
  measures spatial reads of point clouds, meshes and networks for each
  shard layout and shard reader. Its `README.md` says how to run it and
  how to read the results.
- `tests/test_perf_writes.py` is a regression gate, not a benchmark. It
  writes and reads 10,000 lines, polylines, graphs and meshes and checks
  each wall time against a loose ceiling (`PERF_BUDGET`), so it catches
  order-of-magnitude slowdowns, such as an O(N²) loop, without failing on
  a slow CI runner. It runs with the rest of the `pytest` suite.
