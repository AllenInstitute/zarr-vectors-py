"""The data-oriented public API.

Everything here is stable surface.  ``zarr_vectors.core``,
``zarr_vectors.lazy``, ``zarr_vectors.ops``, ``zarr_vectors.encoding`` and
``zarr_vectors.spatial`` are internal and may change between releases;
this package and :mod:`zarr_vectors.building` are the two that will not.

The names are re-exported from :mod:`zarr_vectors` itself, so the usual
spelling is::

    import zarr_vectors as zv

    ds = zv.open("scan.zarrvectors")
    result = ds.select(bbox=(lo, hi)).read()
"""

from __future__ import annotations

from zarr_vectors.api.dataset import (
    Dataset,
    FormatError,
    aopen,
    create,
    open,
    open_or_create,
    require_format,
)
from zarr_vectors.api.edit import EditPlan
from zarr_vectors.api.grid import CellRef, CellSet, Grid, GridCapacity
from zarr_vectors.api.level import Level
from zarr_vectors.api.objects import ObjectCatalog
from zarr_vectors.api.result import Attributes, ReadError, ReadResult
from zarr_vectors.api.schema import (
    AttributeSpec,
    Axis,
    Layout,
    Schema,
    SchemaConflict,
    SizeHints,
    StorageOptions,
)
from zarr_vectors.api.select import Query, Selection

# Aliases for callers who do not `import zarr_vectors as zv` and would
# otherwise shadow the builtin.
open_dataset = open
create_dataset = create

__all__ = [
    "AttributeSpec",
    "Attributes",
    "Axis",
    "CellRef",
    "CellSet",
    "Dataset",
    "EditPlan",
    "FormatError",
    "Grid",
    "GridCapacity",
    "Layout",
    "Level",
    "ObjectCatalog",
    "Query",
    "ReadError",
    "ReadResult",
    "Schema",
    "SchemaConflict",
    "Selection",
    "SizeHints",
    "StorageOptions",
    "aopen",
    "create",
    "create_dataset",
    "open",
    "open_dataset",
    "open_or_create",
    "require_format",
]
