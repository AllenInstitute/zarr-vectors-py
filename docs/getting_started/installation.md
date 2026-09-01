# Installation

`zarr-vectors` requires Python 3.10 or later and depends on `zarr>=3.0`,
`numpy>=1.24`, and `numcodecs`. All mandatory dependencies are installed
automatically by `pip`.

## Standard install

```bash
pip install zarr-vectors
```

This installs the core read/write/validate API for all seven geometry types.
It does not include format converters, cloud-store drivers, or Draco mesh
compression — those are available as optional extras described below.

## Optional extras

### Ingest and export formats

Format converters and the `zarr-vectors` CLI live in the companion
package **`zarr-vectors-tools`** — install that separately if you need
to read/write LAS, PLY, CSV/XYZ, TRK, TCK, TRX, SWC, GraphML, OBJ, or
STL files.

### Draco mesh compression

```bash
pip install "zarr-vectors[draco]"
```

Enables Google Draco encoding and decoding for the `mesh` geometry type.
Requires `DracoPy`. Draco-compressed stores are not readable without this
extra installed.

### Cloud object-store backends

```bash
pip install "zarr-vectors[cloud]"
```

Enables reading from and writing to Amazon S3, Google Cloud Storage, and
Azure Blob Storage via `s3fs` and `gcsfs`. See
[Cloud stores](../tutorials/io/cloud_stores.md) for configuration details.

### Everything

```bash
pip install "zarr-vectors[all]"
```

Installs all optional extras in a single command.

## Development install

To install from source with all development dependencies:

```bash
git clone https://github.com/BRIDGE-Neuroscience/zarr-vectors-py.git
cd zarr-vectors-py
pip install -e ".[all]"
pip install -r docs/requirements-docs.txt   # if building the docs locally
```

Run the test suite to verify the install:

```bash
pytest tests/ -v
```

All tests should pass. The suite does not require network access or external
data files.

## Verifying the install

```python
import zarr_vectors
print(zarr_vectors.__version__)

# Confirm geometry type constants are available
from zarr_vectors.constants import GEOM_POINT_CLOUD, GEOM_STREAMLINE
print(GEOM_POINT_CLOUD, GEOM_STREAMLINE)
```

## Dependency notes

`zarr-vectors` targets **Zarr v3** exclusively. Zarr v2 stores are not
supported and cannot be opened with this package. If you have an existing
v2 workflow, migrate the store using `zarr`'s built-in conversion utilities
before ingesting into the Zarr Vectors format.

NumPy 2.x is supported from `zarr-vectors` 0.2 onward. Earlier releases
require NumPy 1.x.
