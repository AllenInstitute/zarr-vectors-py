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

# The data-oriented API.  Prefer these: they describe the data, and they
# are the surface that stays stable when the storage layer changes.
from zarr_vectors.api import (
    Attributes,
    AttributeSpec,
    Axis,
    CellRef,
    CellSet,
    Dataset,
    EditPlan,
    FormatError,
    Grid,
    GridCapacity,
    Layout,
    Level,
    ObjectCatalog,
    Query,
    ReadError,
    ReadResult,
    Schema,
    SchemaConflict,
    Selection,
    SizeHints,
    StorageOptions,
    aopen,
    coarsen_methods,
    create,
    create_dataset,
    open,
    open_dataset,
    open_or_create,
    require_format,
)

# The builder surface, as an attribute of the package.  ``import
# zarr_vectors as zv`` did not give you ``zv.building`` -- it had to be
# imported by its full path -- while ``zv.FsGroup`` and ``zv.ZVWriter``
# were right there, so the two names the split most wanted people to stop
# using were the discoverable ones.
from zarr_vectors import building  # noqa: F401
from zarr_vectors._api_version import (  # noqa: F401
    FEATURES,
    __api_version__,
    require_api,
)
from zarr_vectors._stability import stability  # noqa: F401

# The storage layer.  Kept importable from here for one release, but no
# longer advertised in ``__all__``: a caller reaching for these is
# describing where bytes live rather than what the data is, which is
# precisely the coupling that makes every internal change a downstream
# break.  ``building`` is where a tool that genuinely needs them should
# get them -- it re-exports create_store / open_store / Group with a
# promise attached -- and ``FsGroup`` in particular is a local-store
# implementation detail that no annotation should name.
from zarr_vectors.core.backends import detect_scheme  # noqa: F401
from zarr_vectors.core.group import Group  # noqa: F401
from zarr_vectors.core.store import (  # noqa: F401
    FsGroup,
    create_store,
    open_store,
    rebind,
)
from zarr_vectors.lazy.writer import ZVWriter  # noqa: F401
from zarr_vectors.rechunk import (  # noqa: F401
    RechunkSpec,
    rechunk,
    rechunk_by_attribute,
)

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
    # --- data-oriented API ---
    "open",
    "open_dataset",
    "create",
    "create_dataset",
    "open_or_create",
    "aopen",
    "require_format",
    "Dataset",
    "EditPlan",
    "Level",
    "Grid",
    "GridCapacity",
    "CellRef",
    "CellSet",
    "ObjectCatalog",
    "Query",
    "Selection",
    "ReadResult",
    "ReadError",
    "Attributes",
    "Schema",
    "Layout",
    "StorageOptions",
    "AttributeSpec",
    "SizeHints",
    "Axis",
    "SchemaConflict",
    "FormatError",
    "coarsen_methods",
    # --- version + capability negotiation ---
    "__version__",
    "__api_version__",
    "FEATURES",
    "require_api",
    # --- the builder surface, as a module ---
    "building",
    "stability",
]

# Retired from ``__all__`` but still importable for one release:
#   Group, FsGroup, create_store, open_store, rebind, detect_scheme,
#   RechunkSpec, rechunk, rechunk_by_attribute, ZVWriter
# Get the first eight from ``zarr_vectors.building``; ``ZVWriter`` is
# deprecated outright (see its own warning for what replaces each method).
