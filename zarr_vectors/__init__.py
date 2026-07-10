"""
zarr-vectors: Python utilities for the Zarr Vectors (ZV) format.

Cloud-native storage for points, lines, streamlines, graphs, and meshes
built on Zarr v3.
"""

import warnings

from zarr.errors import UnstableSpecificationWarning

from zarr_vectors.core.backends import StorageBackend, detect_scheme
from zarr_vectors.core.group import Group
from zarr_vectors.core.store import (
    FsGroup,
    create_store,
    open_store,
    rebind,
)
from zarr_vectors.lazy.writer import ZVWriter
from zarr_vectors.rechunk import RechunkSpec, rechunk, rechunk_by_attribute

# zarr-vectors' per-chunk-array layout is built on Zarr v3's
# ``variable_length_bytes`` dtype (see :doc:`/spec/chunking/sharding`), which
# zarr-python flags as spec-unstable on every array create *and* open. This
# is a permanent, deliberate design choice rather than something callers can
# act on, so silence it once here instead of wrapping every array create/open
# call site throughout the package.
warnings.filterwarnings("ignore", category=UnstableSpecificationWarning)

# Version resolution.  Three sources in priority order:
#   1. ``zarr_vectors/_version.py`` — written by setuptools-scm at build
#      time from the git tag (e.g. ``v0.1.0`` → ``0.1.0``).  Present in
#      wheels and in any checkout that has been ``pip install -e``'d.
#   2. ``importlib.metadata`` — the version recorded in the installed
#      package's metadata; covers the case where _version.py is absent
#      but the package is installed.
#   3. ``"0.0.0+unknown"`` — running from a raw source checkout that
#      has never been built or installed.  Tools that key on version
#      should treat this as a sentinel.
try:
    from zarr_vectors._version import __version__  # type: ignore[no-redef]
except ImportError:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version
    try:
        __version__ = _pkg_version("zarr-vectors")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

__all__ = [
    "StorageBackend",
    "Group",
    "FsGroup",
    "create_store",
    "open_store",
    "rebind",
    "detect_scheme",
    "RechunkSpec",
    "rechunk",
    "rechunk_by_attribute",
    "ZVWriter",
    "__version__",
]
