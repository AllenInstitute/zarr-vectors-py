Geometry types
==============

Read and write functions for each Zarr Vectors geometry type. All write functions
accept either a local path string, a ``zarr.storage.Store`` object, or
an fsspec mapper as the first argument. All read functions return a typed
result dictionary documented on each function's page.

``zarr_vectors.types`` is *undecided* tier in ``zarr_vectors/_stability.py``
— superseded by the :doc:`api` surface, but not deprecated: "The five
store-creating writers are promoted into ``building`` and are supported
there. The readers are superseded by ``Level.read()`` / ``ReadResult``, but
cannot be deprecated until the api can carry per-vertex attributes for every
geometry — pointing callers at a lossy replacement is worse than leaving
them here."

Point clouds
------------

.. automodule:: zarr_vectors.types.points
   :members:
   :undoc-members:
   :show-inheritance:

Lines
-----

.. automodule:: zarr_vectors.types.lines
   :members:
   :undoc-members:
   :show-inheritance:

Polylines and streamlines
-------------------------

.. automodule:: zarr_vectors.types.polylines
   :members:
   :undoc-members:
   :show-inheritance:

Graphs and skeletons
--------------------

.. automodule:: zarr_vectors.types.graphs
   :members:
   :undoc-members:
   :show-inheritance:

Meshes
------

.. automodule:: zarr_vectors.types.meshes
   :members:
   :undoc-members:
   :show-inheritance:
