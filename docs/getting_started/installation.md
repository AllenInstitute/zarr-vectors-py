# Installation

`zarr-vectors` requires Python 3.11 or later. `pip` installs its required
dependencies, `zarr>=3.0`, `numpy>=1.24` and `numcodecs>=0.12`.

## Standard install

```bash
pip install zarr-vectors
```

This installs reading, writing and validation for all seven geometry types.
Cloud-store drivers, Draco mesh compression and GPU support are optional
extras, described below. Format converters are in a separate package.

## Optional extras

### Ingest and export formats

Format converters and the `zvtools` command-line tool are in the companion
package `zarr-vectors-tools`. Install it separately to read or write LAS,
PLY, CSV/XYZ, TRK, TCK, TRX, SWC, GraphML, OBJ or STL files; see
[Ingesting other formats](../tutorials/io/ingest_formats.md).

### Draco mesh compression

```bash
pip install "zarr-vectors[draco]"
```

Installs `DracoPy` for Google Draco encoding and decoding of meshes. You need
it to read a Draco-compressed store.

### Cloud object-store backends

```bash
pip install "zarr-vectors[obstore]"
```

Reads and writes Amazon S3, Google Cloud Storage and Azure Blob Storage
through `obstore`. `[cloud]` is another name for the same extra.

```bash
pip install "zarr-vectors[fsspec]"
```

Uses the `fsspec` backends instead: `s3fs`, `gcsfs` and `adlfs`.

```bash
pip install "zarr-vectors[icechunk]"
```

Adds the Icechunk transactional store, selected with
`StorageOptions(backend="icechunk")`; it is never chosen from a URL. It needs
Python 3.12 or later.

See [Cloud stores](../tutorials/io/cloud_stores.md) for configuration.

### GPU arrays

```bash
pip install "zarr-vectors[gpu]"
```

Lets the array-form readers and writers in `zarr_vectors.building` take and
return device (cupy) arrays: `device="cuda"` on a reader, or a cupy array
passed to a writer. With it, `read_cells` decodes uncompressed cells on the
device, and writers encode device arrays there. It is Linux only and installs
`cupy-cuda12x`. In a conda environment, install cupy from conda-forge instead
(`conda install -c conda-forge cupy`), because the extra would add a second,
pip-built cupy.

Two more extras build on it:

- `[gpu-codecs]` adds nvCOMP, so `read_cells(..., decode="device")` can
  decompress zstd cells on the device. Its wheel carries only nvCOMP, so
  it is safe in a conda environment too.
- `[gpu-io]` adds kvikio, for reading local files straight into device
  memory where the system has GPUDirect Storage. It is for pip
  environments only; its wheel pins its own cupy, so in conda install
  kvikio from the `rapidsai` channel instead.

Every call also works with numpy, and a store written from device arrays is
byte-identical to one written from numpy. See [GPU arrays](../how_to/gpu.md).

### Everything

```bash
pip install "zarr-vectors[all]"
```

Installs `[draco]`, `[obstore]`, `[fsspec]` and `[icechunk]`. The GPU extras
are tied to a CUDA version, so you always ask for them by name.

## Development install

To install from source with the test and lint tools:

```bash
git clone https://github.com/AllenInstitute/zarr-vectors-py.git
cd zarr-vectors-py
pip install -e ".[all,dev]"
pip install -r docs/requirements-docs.txt   # if building the docs locally
```

Run the test suite to check the install:

```bash
pytest tests/
```

## Verifying the install

```python
import zarr_vectors
from zarr_vectors.constants import GEOM_POINT_CLOUD, GEOM_STREAMLINE

print(zarr_vectors.__version__)
print(GEOM_POINT_CLOUD, GEOM_STREAMLINE)
```

The first line prints the installed version; the second prints
`point_cloud streamline`.

## Zarr version

`zarr-vectors` reads and writes Zarr v3 only; it cannot open Zarr v2
stores.
