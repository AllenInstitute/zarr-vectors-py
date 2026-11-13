# Compliance testing

How the test suite checks that `zarr-vectors-py` writes conforming
stores and reads them back, and how to run it.

## Terms

**Conformance test**
: A test that writes a store from synthetic data and asserts that
  `zarr_vectors.validate.validate(path, level=5)` passes.

**Round-trip test**
: A test that writes a store and asserts that reading it returns the
  data written.

**Byte-level comparison**
: A test that writes the same input through two code paths and asserts
  the stores are identical on disk (`assert_stores_identical` in
  `tests/_store_compare.py`).

## Introduction

The validator is the executable form of this specification. So the
main compliance check is that every writer's output validates at
level 5 and reads back unchanged. Tests build data with seeded random
generators; the repository holds no reference stores.

## Technical reference

### Test layout

| Path | Content |
|---|---|
| `tests/conftest.py` | Shared fixtures and the `--manifest-layout` option |
| `tests/_store_compare.py` | `assert_stores_identical` |
| `tests/_reference.py` | Frozen copies of write paths, for byte-level comparison |
| `tests/test_<feature>.py` | One module per feature or geometry type, such as `test_points.py` |
| `tests/test_linkml_schema.py` | Stored metadata against the JSON Schema; `VALID_*` sets against the schema enums; pins `FORMAT_VERSION` |
| `tests/integration/` | Longer end-to-end tests, not run by the workflow |
| `tests/gpu/` | Tests marked `gpu`, skipped without CuPy and a CUDA device |

Fixtures in `conftest.py`: `rng` (seed 42), `simple_points_3d`,
`two_chunk_points`, `simple_streamlines`, `simple_skeleton`,
`simple_graph`, `simple_mesh`, `tmp_store_path`, `tmp_dir`,
`assert_arrays_close` and `assert_arrays_equal`.

### Example

```python
import numpy as np

from zarr_vectors.types.points import read_points, write_points
from zarr_vectors.validate import validate


def test_points_round_trip_and_validate(tmp_path):
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 500, (2_000, 3)).astype(np.float32)
    path = str(tmp_path / "points.zarrvectors")

    write_points(path, positions, chunk_shape=(100.0, 100.0, 100.0))

    result = validate(path, level=5)
    assert result.ok, result.summary()

    lo, hi = np.zeros(3), np.full(3, 100.0)
    box = read_points(path, bbox=(lo, hi))
    inside = np.all((positions >= lo) & (positions < hi), axis=1)
    assert box["vertex_count"] == int(inside.sum())
```

`validate` returns a `ValidationResult`: `ok` is true when there are no
errors, and `summary()` lists every pass, warning and error.

### Running the tests

The test workflow (`.github/workflows/tests.yml`) runs on Python 3.11
and 3.12:

```bash
uv sync --frozen --extra all --extra dev
uv run python schema/regen.py --check
uv run pytest tests/ --ignore=tests/integration -v
```

`pytest tests/ --manifest-layout dense` (or `vlen`) forces the layout
of every new object index, except in tests marked `vlen_only`.
