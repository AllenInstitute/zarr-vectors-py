# Benchmarks

Notebooks that exercise the `zarr-vectors-py` write/read/edit path
along one scaling axis each:

| Notebook | Axis | Fixed | Swept |
|----------|------|-------|-------|
| [`01_size_scaling.ipynb`](01_size_scaling.ipynb) | size (N) | point cloud, local backend | `N ∈ {1e3, 1e4, 1e5, 1e6}` |
| [`02_data_types.ipynb`](02_data_types.ipynb)     | geometry type | `N = 50_000`, local backend | all 6 types |
| [`03_backends.ipynb`](03_backends.ipynb)         | backend | point cloud, `N = 100_000` | `local` always; `obstore`/`fsspec` if `ZV_BENCH_S3_URL` is set |
| [`08_edit_operations.ipynb`](08_edit_operations.ipynb) | edit kind, atomicity, N_edits, concurrency, index scaling | `N = 50_000` baseline points | `kind ∈ {move_in_chunk, move_cross_chunk, add, soft_delete}`, `atomic ∈ {True, False}`, `N_edits ∈ {1, 10, 100, 1_000}`; multi-writer sub-row covers 1 vs 4 cooperating editors on disjoint chunks via `oid_prefix=`; index-scaling sub-row sweeps `N_OBJECTS ∈ {1k, 10k, 50k}` to document the flatness of the post-Iter4 fragment→OID inverted index |

Each notebook follows the same shape: **setup → sweep → table →
plot**. ~10 cells, ~1 plot, no surprises.

## Running

```bash
pip install zarr-vectors pandas matplotlib
jupyter lab examples/benchmarks/
```

Then open one and run all cells. Expected runtime on a laptop:

- `01_size_scaling`: a few minutes (the 1M case dominates)
- `02_data_types`: ~30 s
- `03_backends`: ~10 s without cloud, longer with
- `08_edit_operations`: ~3-6 minutes (`move_cross_chunk` + `atomic=True` at `N_edits=1_000` plus the concurrency sub-row dominate)

## Optional cloud backend benchmarking

Notebook 03 benchmarks the `obstore` and `fsspec` cloud backends
**only when** the `ZV_BENCH_S3_URL` env var is set:

```bash
export ZV_BENCH_S3_URL="s3://my-bucket/zv-bench/"
jupyter lab examples/benchmarks/03_backends.ipynb
```

Both `obstore` and `fsspec` are optional installs:

```bash
pip install "zarr-vectors[obstore]"   # preferred
# OR
pip install "zarr-vectors[cloud]"     # fsspec fallback
```

When the env var is unset, the notebook prints a skip note for the
cloud rows and reports a one-row local-only result.

## Caveats

These numbers are machine-dependent and **not benchmarks of
underlying algorithms** — different geometry types do genuinely
different work. Treat them as "what to expect on my machine" sanity
plots, not as cross-format comparisons.

No CI gating, no pytest-benchmark integration, no memory profiling
(disk bytes only). To regenerate the notebooks from source:

```bash
python examples/benchmarks/_build.py
```
