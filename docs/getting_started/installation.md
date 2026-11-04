# Installation

`zarr-vectors` requires **Python 3.11 or later** (3.11 and 3.12 are the
tested versions) and depends on `zarr>=3.0`, `numpy>=1.24`, and `numcodecs`.
All mandatory dependencies are installed automatically by `pip`. Check yours
with `python --version`; a virtual environment (`python -m venv .venv`, or
`uv venv`) keeps it separate from other projects.

## Standard install

```bash
pip install zarr-vectors
```

This installs the core read/write/validate API for all seven geometry types
(point clouds, lines, polylines, streamlines, graphs, skeletons and meshes). It
does not include format converters, cloud-store drivers, or Draco mesh
compression — those are available as optional extras described below. The
`zarr-vectors` *command-line tool* is also not part of this package: this
package installs no console scripts (see below).

Once installed, the [Quickstart](quickstart.md) is the next stop.

## Optional extras

### Ingest and export formats

Format converters and the `zarr-vectors` CLI live in the companion
package **`zarr-vectors-tools`** (`pip install zarr-vectors-tools`) — install
that separately if you need to read/write LAS, PLY, CSV/XYZ, TRK, TCK, TRX,
SWC, GraphML, OBJ, or STL files.

### Draco mesh compression

```bash
pip install "zarr-vectors[draco]"
```

Enables Google Draco encoding and decoding for the `mesh` geometry type.
Requires `DracoPy`. Draco-compressed stores are not readable without this
extra installed.

### Cloud object-store backends

```bash
pip install "zarr-vectors[obstore]"   # preferred (Rust-based); "[cloud]" is an alias for it
pip install "zarr-vectors[fsspec]"    # fallback: fsspec + s3fs, gcsfs, adlfs
```

Enables reading from and writing to Amazon S3, Google Cloud Storage, and
Azure Blob Storage, either through `obstore` or through `fsspec` with
`s3fs`/`gcsfs`/`adlfs`. See [Cloud stores](../tutorials/io/cloud_stores.md)
for configuration details.

### Transactional stores (Icechunk)

```bash
pip install "zarr-vectors[icechunk]"
```

Selected explicitly with `StorageOptions(backend="icechunk")`; it is never
auto-detected from a URL. The extra resolves only on Python 3.12 or later.

### Everything

```bash
pip install "zarr-vectors[all]"
```

Installs all optional extras (`draco`, `obstore`, `fsspec`, `icechunk`) in a
single command.

## Development install

To install from source with all development dependencies:

```bash
git clone https://github.com/AllenInstitute/zarr-vectors-py.git
cd zarr-vectors-py
pip install -e ".[all,dev]"
pip install -r docs/requirements-docs.txt   # if building the docs locally
```

Run the test suite to verify the install:

```bash
pytest tests/ -v
```

`dev` adds `pytest`, `ruff` and `mypy` (and the schema tooling used by one
test module). The suite is intended to run without network access or external
data files.

## Verifying the install

```python
import zarr_vectors as zv
from zarr_vectors.constants import GEOM_POINT_CLOUD, GEOM_STREAMLINE

print(zv.__version__)
print(GEOM_POINT_CLOUD, GEOM_STREAMLINE)
```

```text
<your installed version>
point_cloud streamline
```

(The version comes from the git tag the release was built from, so a
development checkout prints a `.devN+g<hash>` version.)

## Dependency notes

`zarr-vectors` targets **Zarr v3** exclusively (`zarr>=3.0`; Zarr 2 cannot read
these stores, because the layout uses Zarr v3's variable-length bytes codec).
Zarr v2 stores are not supported and cannot be opened with this package; the
data has to be re-written into a Zarr Vectors store from its source.

`numpy>=1.24` is required, with no upper bound declared. Importing the package
silences zarr's `UnstableSpecificationWarning` (the variable-length dtypes it
complains about are a deliberate choice of the format).
