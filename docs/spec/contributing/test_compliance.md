# Compliance testing

## Terms

**Reference fixture**
: A small, pre-written ZVF store committed to the repository under
  `tests/fixtures/`. Reference fixtures are the ground truth for
  compliance testing: any conforming reader must produce correct output
  when reading them.

**Compliance test**
: A pytest test that opens a reference fixture and asserts that the reader
  produces the expected output (vertex positions, attribute values, object
  IDs, etc.). Compliance tests guard against regressions in the reader.

**Writer conformance test**
: A pytest test that writes a store using `zarr-vectors-py` and then runs
  the validator at level 5 to confirm the output is conforming. These tests
  guard against regressions in the writer.

**Golden output**
: The expected output of a read operation on a reference fixture, stored
  as a companion `.npz` file. Used by compliance tests to assert exact
  numerical equality.

---

## Introduction

The compliance test suite serves two purposes. First, it ensures that
`zarr-vectors-py` correctly reads ZVF stores written by any conforming
implementation. Second, it documents the ZVF format in executable form:
the reference fixtures are concrete examples of valid stores that
contributors can inspect.

For contributors adding a new feature or fixing a bug, the test suite is
the primary quality gate. All PRs must pass the full test suite. PRs that
add new functionality must include new tests.

---

## Technical reference

### Repository test layout

```
tests/
├── fixtures/                       ← reference stores (committed to repo)
│   ├── point_cloud_3d/
│   │   ├── store.zarrvectors/      ← reference ZVF store (small)
│   │   └── expected.npz           ← golden outputs
│   ├── streamline_3d_multiscale/
│   │   ├── store.zarrvectors/
│   │   └── expected.npz
│   ├── skeleton_swc/
│   │   ├── store.zarrvectors/
│   │   ├── source.swc             ← input file (for ingest round-trip)
│   │   └── expected.npz
│   └── …                          ← one fixture per geometry type
├── conftest.py                     ← shared fixtures and helpers
├── test_point_cloud.py
├── test_polylines.py
├── test_streamlines.py
├── test_graphs.py
├── test_skeletons.py
├── test_meshes.py
├── test_validation.py
├── test_rechunking.py
└── test_multiresolution.py
```

### Reference fixture requirements

Each reference fixture must:

1. Be small enough to commit to the repository (< 5 MB per fixture).
2. Cover all arrays defined for its geometry type (no missing optional
   arrays for the primary fixture).
3. Include at least two resolution levels for types that support pyramids.
4. Include at least one object that spans chunks, for types that
   support it — so the fixture exercises a non-zero-offset link array
   and not just the all-zero one.
5. Include at least two named attributes.
6. Pass L5 validation (asserted in `test_validation.py`).

### Generating reference fixtures

Use the `generate_fixtures.py` script in `tests/`:

```bash
python tests/generate_fixtures.py --output tests/fixtures/
```

This script generates all reference fixtures from scratch using
`zarr-vectors-py` with deterministic random seeds. It also writes the
`expected.npz` golden outputs.

To regenerate a single fixture after a spec change:

```bash
python tests/generate_fixtures.py --type point_cloud_3d \
    --output tests/fixtures/
```

After regenerating, inspect the diff carefully: a change in reference
fixture content means either a correctness fix (good) or a format change
that was not intended (bad).

### Writing a compliance test

```python
# tests/test_point_cloud.py
import numpy as np
import pytest
from zarr_vectors.types.points import read_points
from zarr_vectors.validate import validate

FIXTURE_PATH = "tests/fixtures/point_cloud_3d/store.zarrvectors"
EXPECTED_PATH = "tests/fixtures/point_cloud_3d/expected.npz"

@pytest.fixture(scope="module")
def expected():
    return np.load(EXPECTED_PATH, allow_pickle=True)

def test_read_all_vertices(expected):
    result = read_points(FIXTURE_PATH)
    assert result["vertex_count"] == int(expected["vertex_count"])
    np.testing.assert_allclose(
        result["positions"],
        expected["positions"],
        rtol=1e-6,
    )

def test_read_bbox_query(expected):
    lo = expected["bbox_lo"]
    hi = expected["bbox_hi"]
    result = read_points(FIXTURE_PATH, bbox=(lo, hi))
    np.testing.assert_array_equal(
        np.sort(result["positions"], axis=0),
        np.sort(expected["bbox_positions"], axis=0),
    )

def test_attribute_values(expected):
    result = read_points(FIXTURE_PATH)
    np.testing.assert_allclose(
        result["attributes"]["intensity"],
        expected["intensity"],
        rtol=1e-5,
    )

def test_level_1_read(expected):
    result = read_points(FIXTURE_PATH, level=1)
    assert result["vertex_count"] < int(expected["vertex_count"])
    assert result["vertex_count"] == int(expected["level1_vertex_count"])

def test_l5_validation():
    result = validate(FIXTURE_PATH, level=5)
    assert result.is_valid, result.report()
```

### Writing a writer conformance test

```python
# tests/test_point_cloud.py (continued)
import tempfile, numpy as np
from zarr_vectors.types.points import write_points, read_points
from zarr_vectors.validate import validate

def test_write_conformance():
    rng = np.random.default_rng(0)
    positions  = rng.uniform(0, 500, (10_000, 3)).astype(np.float32)
    intensity  = rng.uniform(0, 1, 10_000).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/test.zarrvectors"
        write_points(path, positions,
                     chunk_shape=(100., 100., 100.),
                     bin_shape=(25., 25., 25.),
                     attributes={"intensity": intensity})

        # Validate at full depth
        result = validate(path, level=5)
        assert result.is_valid, result.report()

        # Round-trip
        out = read_points(path)
        np.testing.assert_allclose(
            np.sort(out["positions"], axis=0),
            np.sort(positions, axis=0),
            rtol=1e-6,
        )
```

### Running the test suite

```bash
# Full suite
pytest tests/ -v

# Fast checks only (no L3+ validation)
pytest tests/ -v -m "not slow"

# Single test module
pytest tests/test_streamlines.py -v

# With coverage
pytest tests/ --cov=zarr_vectors --cov-report=term-missing
```

Tests marked `@pytest.mark.slow` read full reference fixtures through
the validator and may take 10–60 seconds each. They are excluded from
the fast CI job but included in the nightly job.

### CI pipeline

The repository uses GitHub Actions with two jobs:

**Fast job (every push and PR):**
- L1–L2 validation of all fixtures.
- All writer conformance tests.
- All round-trip tests.
- Runs in < 2 minutes.

**Nightly job:**
- Full L5 validation of all fixtures.
- Slow compliance tests.
- Coverage report.
- Runs in < 15 minutes.

Both jobs run on Python 3.10, 3.11, and 3.12.

### Adding a fixture for a new geometry type

1. Add a generator function to `tests/generate_fixtures.py`.
2. Run `python tests/generate_fixtures.py --type <new_type>`.
3. Inspect the generated store and validate manually at L5 with
   `zarr_vectors.validate.validate(store, level=5)`.
4. Commit both the store and the `expected.npz`.
5. Add compliance tests in `tests/test_<new_type>.py`.
6. Ensure the nightly CI job runs the new tests.
