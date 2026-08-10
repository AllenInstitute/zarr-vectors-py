# Benchmarks

```{admonition} Benchmarks have moved
:class: note

The benchmark notebooks and their published results now live in the
companion package **`zarr-vectors-tools`**, not in this repository.

See the [zarr-vectors-tools benchmarks
page](https://zarr-vectors-tools.readthedocs.io/en/latest/benchmarks/),
or [`benchmarks/`](https://github.com/AllenInstitute/zarr-vectors-tools/tree/main/benchmarks)
in that repository's source tree.
```

## Why they moved

This package deliberately keeps its runtime dependencies to `numpy`,
`numcodecs`, and `zarr`. The benchmarks need `pandas`, `matplotlib`,
and `jupyter`, and the more useful comparisons — zarr-vectors against
PLY, CSV, TRX, GraphML, SWC, and OBJ — additionally need the
third-party readers that `zarr-vectors-tools` already wraps. Keeping
both suites next to those readers means neither repository carries a
dependency it does not otherwise need.

## What's there

| Suite | Question it answers | Notebooks |
|-------|--------------------|-----------|
| `benchmarks/formats/` | How does ZVF compare to the format I use today? | 3 |
| `benchmarks/internals/` | How does ZVF scale along one axis? — size, geometry type, backend, pyramid depth, bbox query, chunk shape, codec, edit cost | 8 |

The vertex-scaling analysis that used to be published on this page —
zarr-vectors versus a pandas / CSV baseline, with the crossover table
and the dtype / on-disk-encoding discussion — is now the
`internals/01_size_scaling` section of that page.

## What stayed here

`tests/test_perf_writes.py` remains in this repository. It is not a
benchmark: it is a regression gate that writes and reads `N = 10 000`
lines, polylines, graphs, and meshes, and asserts each wall time
against a loose ceiling (`PERF_BUDGET`, set ~3× over measured
post-fix times). It runs as part of the normal `pytest` suite and is
sized to catch order-of-magnitude regressions — a reintroduced O(N²)
loop — without flaking on shared CI runners. It pulls in no
dependency beyond `numpy` and `pytest`.
