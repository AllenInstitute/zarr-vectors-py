"""Lazy zarr vectors store access.

Opens a store without reading data. All vertex, attribute, and object
access is deferred until ``.compute()`` is called, optionally using
Dask for parallel I/O.

Usage::

    from zarr_vectors.lazy import open_zv

    zv = open_zv("scan.zarrvectors")
    zv[0].vertices.compute()          # materialise all level-0 vertices
    zv[0].vertices[0, 0, 0].compute() # single chunk
    zv[0].attributes["intensity"].compute()
"""

from zarr_vectors.lazy.store import ZVStore, open_zv
from zarr_vectors.lazy.views import ZVPolylineCollection, ZVView

__all__ = ["ZVStore", "ZVView", "ZVPolylineCollection", "open_zv"]
