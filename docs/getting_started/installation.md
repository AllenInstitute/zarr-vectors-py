# Installation

`zarr-vectors` requires Python 3.11 or later. It reads and writes Zarr v3
stores only.

```bash
pip install zarr-vectors
```

This installs `zarr>=3.0`, `numpy>=1.24` and `numcodecs>=0.12`, and gives you
reading, writing and validation for all seven geometry types.

## Optional extras

Install an extra with `pip install "zarr-vectors[<extra>]"`.

| Extra | Adds | See |
|-------|------|-----|
| `draco` | `DracoPy`, needed to read Draco-compressed meshes | |
| `obstore` (alias `cloud`) | Amazon S3, Google Cloud Storage and Azure through `obstore` | [Cloud stores](../how_to/cloud_stores.md) |
| `fsspec` | The same clouds through `s3fs`, `gcsfs` and `adlfs` | [Cloud stores](../how_to/cloud_stores.md) |
| `icechunk` | The Icechunk transactional store; needs Python 3.12 or later | [Cloud stores](../how_to/cloud_stores.md) |
| `gpu` | cupy (`cupy-cuda12x`) device arrays; Linux only | [GPU arrays](../how_to/gpu.md) |
| `gpu-codecs` | `gpu` plus nvCOMP, to decompress zstd cells on the device | [GPU arrays](../how_to/gpu.md) |
| `gpu-io` | `gpu` plus kvikio, to read local files into device memory | [GPU arrays](../how_to/gpu.md) |
| `all` | `draco`, `obstore`, `fsspec` and `icechunk` | |

`all` leaves out the GPU extras, because they are tied to a CUDA version. In a
conda environment, install cupy from conda-forge instead of using `[gpu]`, and
kvikio from the `rapidsai` channel instead of using `[gpu-io]`.

Format converters (LAS, PLY, TRK, SWC, OBJ and others) and the `zvtools`
command are in the separate package `zarr-vectors-tools`; see
[Ingesting other formats](../how_to/convert_files.md).

## Development install

```bash
git clone https://github.com/AllenInstitute/zarr-vectors-py.git
cd zarr-vectors-py
pip install -e ".[all,dev]"
pip install -r docs/requirements-docs.txt   # to build the docs
pytest tests/
```

## Check the install

```python
import zarr_vectors

print(zarr_vectors.__version__)
```

This prints the installed version.
