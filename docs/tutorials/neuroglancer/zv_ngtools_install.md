# Installing zv-ngtools

`zv-ngtools` is a fork of [`ngtools`](https://github.com/neuroscales/ngtools)
maintained by BRIDGE Neuroscience. It is not yet published to PyPI; install
directly from GitHub.

---

## Requirements

| Requirement | Minimum version | Notes |
|-------------|----------------|-------|
| Python | 3.10 | Matches `zarr-vectors` requirement |
| `zarr-vectors` | 0.1.0 | Must be installed first |
| `neuroglancer` | 2.39 | The Google Neuroglancer Python bindings |
| Browser | Chrome 110 / Firefox 115 | WebGL 2 required |

---

## Install from GitHub

```bash
pip install git+https://github.com/BRIDGE-Neuroscience/zv-ngtools.git
```

This installs the `ngtools` package (the fork is still importable as
`ngtools`) along with its dependencies: `neuroglancer`, `tornado`,
`prompt_toolkit`, and `numpy`.

### Install with zarr-vectors

If you have not yet installed `zarr-vectors`, install both together:

```bash
pip install "zarr-vectors[all]" \
    git+https://github.com/BRIDGE-Neuroscience/zv-ngtools.git
```

### Development install

To contribute to `zv-ngtools` or run the examples:

```bash
git clone https://github.com/BRIDGE-Neuroscience/zv-ngtools.git
cd zv-ngtools
pip install -e ".[dev]"
```

---

## Verifying the install

```python
import ngtools
print(ngtools.__version__)

# Check that the zarr_vectors layer type is registered
from ngtools.datasources import list_datasources
print("zarr_vectors" in list_datasources())   # True
```

From the shell:

```bash
nglocal --version
# ngtools 0.x.y (zv-ngtools fork)

nglocal --help
```

---

## What the fork changes

`zv-ngtools` diverges from upstream `ngtools` in the following areas.
All changes are additive; no upstream behaviour is modified.

### New `zarr_vectors://` URL scheme

The upstream `ngtools` supports `zarr://` (for OME-Zarr image volumes).
`zv-ngtools` additionally registers a `zarr_vectors://` scheme that is
recognised by `LocalNeuroglancer` and the `nglocal` shell:

```bash
# Upstream ngtools — loads an OME-Zarr image volume
[1] load zarr:///path/to/image.zarr

# zv-ngtools — loads a Zarr Vectors store
[1] load zarr_vectors:///path/to/tracts.zarrvectors
[1] load zarr_vectors://s3://my-bucket/tracts.zarrvectors
```

When a path ending in `.zarrvectors` is passed without a scheme,
`zv-ngtools` infers the `zarr_vectors://` scheme automatically.

### New Zarr Vectors-aware layer types

The `zarr_vectors://` datasource registers a layer type appropriate for
the store's `geometry_type`:

| Zarr Vectors `geometry_type` | Neuroglancer layer | Rendering |
|--------------------|--------------------|-----------|
| `point_cloud` | `annotation` | 3-D points (size, colour from attributes) |
| `line` | `annotation` | Line segment pairs |
| `polyline`, `streamline` | `annotation` | Connected polylines |
| `graph`, `skeleton` | `annotation` (lines) | Node–edge network |
| `mesh` | `segmentation` | Triangulated surface |

### LOD integration

The Zarr Vectors resolution pyramid drives Neuroglancer's level-of-detail rendering.
As the user zooms out, `zv-ngtools` automatically selects a coarser level
whose `bin_shape` is commensurate with the on-screen pixel size. This is
transparent to the user: zooming in reveals finer detail without any manual
level switching.

### Precomputed export

Precomputed export (and the `zarr-vectors` CLI subcommand for it) lives
in the companion package **`zarr-vectors-tools`**.

---

## Upgrading

Because `zv-ngtools` is installed from GitHub, upgrades require
re-running the install command:

```bash
pip install --upgrade git+https://github.com/BRIDGE-Neuroscience/zv-ngtools.git
```

To pin a specific commit (for reproducibility):

```bash
pip install "git+https://github.com/BRIDGE-Neuroscience/zv-ngtools.git@<commit_sha>"
```

---

## Troubleshooting

**`ImportError: No module named 'neuroglancer'`**

```bash
pip install neuroglancer
```

**`WebGL 2 not available` in browser**

Some headless or SSH-forwarded browser sessions do not support WebGL 2.
Use Chrome or Firefox on a desktop with a GPU. On remote systems,
consider using a remote desktop session (VNC, NoMachine) rather than X
forwarding.

**Neuroglancer window does not open automatically**

By default `LocalNeuroglancer` attempts to open the browser automatically
using `webbrowser.open()`. If this fails (e.g. on a headless server), the
URL is printed to stdout:

```
neuroglancer: http://127.0.0.1:9321/v/1/
```

Copy the URL and open it manually.

**`zarr_vectors` datasource not found**

```python
from ngtools.datasources import list_datasources
print(list_datasources())
```

If `zarr_vectors` is absent, the zv-ngtools fork was not installed correctly.
Confirm with `pip show ngtools` that the source is the BRIDGE-Neuroscience
fork (the installed version will show `Location: .../zv-ngtools`).
