# Compliance testing

This page describes how the test suite checks that `zarr-vectors-py`
writes conforming stores and reads them back, how the tests are laid
out, and how to run them.

## Terms

**Writer conformance test**
: A test that writes a store with synthetic data and asserts that
  `zarr_vectors.validate.validate(path, level=5)` passes.

**Round-trip test**
: A test that writes a store and asserts that reading it returns the
  data that was written.

**Byte-level comparison**
: A test that writes the same input through two code paths and asserts
  the stores are identical on disk, using `assert_stores_identical` in
  `tests/_store_compare.py`.

---

## Introduction

The validator is the executable form of this specification, so the
main compliance check is that every writer's output validates at
level 5 and reads back unchanged. Tests build their data with seeded
random generators; the repository holds no pre-written reference
stores. Pull requests that change behaviour add tests, and the test
workflow must pass before merging (see
[Spec change process](spec_change_process.md)).

---

## Technical reference

### Test layout

```
tests/
├── conftest.py            ← shared fixtures and the --manifest-layout option
├── _store_compare.py      ← assert_stores_identical
├── _reference.py          ← frozen copies of replaced write paths, for comparison
├── test_points.py         ← one module per feature or geometry type
├── test_polylines.py
├── test_graphs.py
├── test_skeletons.py
├── test_meshes.py
├── test_linkml_schema.py  ← stored metadata against the JSON Schema
├── test_validate_rank.py
├── …
├── integration/           ← longer end-to-end tests (not run by the workflow)
└── gpu/                   ← tests marked gpu
```

`tests/conftest.py` provides small seeded datasets as fixtures:
`rng` (seed 42), `simple_points_3d`, `two_chunk_points`,
`simple_streamlines`, `simple_skeleton`, `simple_graph` and
`simple_mesh`, plus `tmp_store_path`, `tmp_dir`,
`assert_arrays_close` and `assert_arrays_equal`.

### Writing a conformance test

A test writes synthetic data, validates at level 5, and checks a full
read and a bounding-box read:

```python
import numpy as np

from zarr_vectors.types.points import read_points, write_points
from zarr_vectors.validate import validate


def test_points_round_trip_and_validate(tmp_path):
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 500, (2_000, 3)).astype(np.float32)
    intensity = rng.uniform(0, 1, 2_000).astype(np.float32)
    path = tmp_path / "points.zarrvectors"

    write_points(str(path), positions,
                 chunk_shape=(100.0, 100.0, 100.0),
                 bin_shape=(25.0, 25.0, 25.0),
                 vertex_attributes={"intensity": intensity})

    result = validate(str(path), level=5)
    assert result.ok, result.summary()

    out = read_points(str(path))
    assert out["vertex_count"] == len(positions)
    np.testing.assert_allclose(
        np.sort(out["positions"], axis=0), np.sort(positions, axis=0),
    )

    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([100.0, 100.0, 100.0])
    box = read_points(str(path), bbox=(lo, hi))
    inside = np.all((positions >= lo) & (positions < hi), axis=1)
    assert box["vertex_count"] == int(inside.sum())
```

`validate` returns a `ValidationResult`; `ok` is true when there are no
errors, and `summary()` lists every pass, warning and error. For a new
geometry type, also corrupt a store on purpose and assert that
validation fails at the expected level.

### Running the tests

The test workflow (`.github/workflows/tests.yml`) installs every extra
and the development tools with `uv`, checks the generated schema, and
runs the suite on Python 3.11 and 3.12:

```bash
uv sync --frozen --extra all --extra dev
uv run python schema/regen.py --check
uv run pytest tests/ --ignore=tests/integration -v
```

Other useful invocations:

```bash
pytest tests/test_points.py -v               # one module
pytest tests/ --manifest-layout dense        # every new object index in the dense layout
pytest tests/integration                     # the longer end-to-end tests
```

`--manifest-layout` (`vlen` or `dense`) forces the layout of every new
object index, except in tests marked `vlen_only`. Tests marked `gpu`
are skipped without CuPy and a CUDA device.

### Schema agreement

`tests/test_linkml_schema.py` validates metadata written by the package
against `schema/zarr_vectors.schema.json`, checks that the `VALID_*`
constant sets match the schema enums, and pins `FORMAT_VERSION`. After
changing the LinkML source, run `python schema/regen.py` (see the
[machine-readable schema](../schema.md)).
