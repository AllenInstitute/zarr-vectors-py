"""
zarr-vectors: Python utilities for the Zarr Vectors (ZV) format.

Cloud-native storage for points, lines, streamlines, graphs, and meshes
built on Zarr v3.
"""

# Silence zarr's UnstableSpecificationWarning once, at import.
#
# The ZV format stores every per-chunk payload in a Zarr v3 vlen-bytes
# array (and some string metadata in fixed-length UTF-32), and zarr warns
# that those dtypes have no finalised v3 spec.  That is an informed,
# load-bearing choice of this library, not something a caller can act on —
# you cannot use zarr-vectors without triggering it — and zarr re-emits it
# on essentially every array access, so it floods (10k+ lines) on any real
# read.  Filter the category so it fires zero times instead of once per
# cell.  Scoped to this one category; every other warning is untouched.
# To see it anyway, re-enable after importing zarr_vectors:
#     warnings.simplefilter("always", UnstableSpecificationWarning)
import warnings as _warnings

try:
    from zarr.errors import UnstableSpecificationWarning as _UnstableDtypeWarning

    _warnings.filterwarnings("ignore", category=_UnstableDtypeWarning)
except Exception:  # pragma: no cover - older/newer zarr without the class
    pass

from zarr_vectors.core.backends import detect_scheme
from zarr_vectors.core.group import Group
from zarr_vectors.core.store import (
    FsGroup,
    create_store,
    open_store,
    rebind,
)
from zarr_vectors.lazy.writer import ZVWriter
from zarr_vectors.rechunk import RechunkSpec, rechunk, rechunk_by_attribute

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
