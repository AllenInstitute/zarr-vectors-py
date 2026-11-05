# Compliance testing

## Terms

**Fixture (pytest)**
: A function in `tests/conftest.py` (or a test module) that builds small,
  deterministic synthetic data, such as seeded random point clouds. The
  repository currently has **no committed reference stores**:
  `tests/fixtures/` holds only an empty package marker. Tests write their
  stores into pytest's `tmp_path` and read them back.

**Round-trip test**
: A pytest test that writes a store with `zarr-vectors-py`, reads it back
  and asserts the data survived.

**Writer conformance test**
: A pytest test that writes a store and then runs the validator
  (`zarr_vectors.validate.validate`) to confirm the output is conforming.
  These tests guard against regressions in the writer.

**Reference fixture / golden output**
: Not yet in use. A committed store plus expected outputs would let a reader be
  tested against stores written by *another* implementation. If you add
  one, keep it small and generate it deterministically.

---

## Introduction

The test suite guards the reader and writer against regressions, and
documents the format in executable form: the tests are concrete examples of
valid stores that contributors can inspect. It does not yet include
cross-implementation compliance fixtures.

For contributors adding a new feature or fixing a bug, the test suite is
the primary quality gate. All PRs must pass the full test suite. PRs that
add new functionality must include new tests.

---

## Technical reference

### Repository test layout

Tests are flat modules in `tests/`, named by area, plus a small
`tests/integration/` directory:

```
tests/
├── conftest.py            ← shared fixtures: tmp_store_path, rng, simple_points_3d, ...
├── fixtures/              ← empty package marker (no committed stores)
├── integration/           ← slower tests, excluded from CI (see below)
├── test_points.py         ← per-geometry: test_points / test_lines_parametric /
├── test_polylines.py        test_polylines / test_graphs / test_skeletons / test_meshes
├── test_graphs.py
├── test_meshes.py
├── test_multiscale_links.py, test_per_object_pyramid.py, ...   ← pyramids
├── test_links_*.py, test_link_offsets.py, test_paths.py ...    ← link families
├── test_icechunk_backend.py, test_backends.py ...              ← backends
├── test_linkml_schema.py  ← keeps schema/ artefacts in sync with the code
└── …                      ← validators, API facade, edits, rechunking, etc.
```

Use `ls tests/` for the current list.

### Writing a round-trip and writer-conformance test

```python
# tests/test_points.py (illustrative)
import numpy as np
from zarr_vectors.types.points import write_points, read_points
from zarr_vectors.validate import validate


def test_write_conformance(tmp_path):
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 500, (10_000, 3)).astype(np.float32)
    intensity = rng.uniform(0, 1, 10_000).astype(np.float32)

    path = str(tmp_path / "test.zarrvectors")
    write_points(
        path, positions,
        chunk_shape=(100., 100., 100.),
        bin_shape=(25., 25., 25.),
        vertex_attributes={"intensity": intensity},
    )

    # Validate at full depth: result.ok is True when there are no errors
    result = validate(path, level=5)
    assert result.ok, result.summary()

    # Round-trip (attributes must be requested by name)
    out = read_points(path, attribute_names=["intensity"])
    assert out["vertex_count"] == len(positions)
    np.testing.assert_allclose(
        np.sort(out["positions"], axis=0),
        np.sort(positions, axis=0),
        rtol=1e-6,
    )
    assert out["vertex_attributes"]["intensity"].shape == (10_000,)
```

`ValidationResult` has `ok`, `passed`, `warnings`, `errors` and
`summary()`; it has no `is_valid` or `report()`.

### Running the test suite

```bash
# Full suite (what CI runs)
uv run pytest tests/ --ignore=tests/integration -v

# Everything, including integration tests
uv run pytest tests/ -v

# Single test module
uv run pytest tests/test_polylines.py -v

# With coverage (if pytest-cov is installed)
uv run pytest tests/ --cov=zarr_vectors --cov-report=term-missing
```

The repository defines no custom pytest markers, so there is no
`-m "not slow"` selection.

### CI pipeline

`.github/workflows/tests.yml` runs on pushes and pull requests (an approval
gate lets outside contributors' PRs run once a maintainer applies the
`ci-approved` label). One job, `pytest`, runs on Python 3.11 and 3.12 after
`uv sync --frozen --extra all --extra dev` and executes
`pytest tests/ --ignore=tests/integration -v`. There is no nightly job.
`.github/workflows/release.yml` builds and publishes releases and checks the
release version against `FORMAT_VERSION`.

### Adding tests for a new geometry type

1. Add a `tests/test_<type>.py` with a seeded-random round trip, a bbox
   read, a pyramid build/read, and a `validate(path, level=5)` assertion.
2. Add shared data builders to `tests/conftest.py` if several tests need
   them.
3. If you change the format, update the LinkML schema and re-run
   `python schema/regen.py` (`--check` reports drift);
   `tests/test_linkml_schema.py` checks the dataclasses against the
   committed JSON Schema.
