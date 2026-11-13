Geometry types
==============

Read and write functions for each Zarr Vectors geometry type. Each function
takes as its first argument a path or URL, a Zarr store object, or an open
store group. Read functions return a dictionary described on each function's
entry.

``zarr_vectors.types`` is in the *undecided* tier: treat it as internal. Its
five store-creating writers are supported through :doc:`building`, and
``Level.read()`` and ``ReadResult`` in :doc:`api` cover its readers, except for
per-vertex attributes on some geometries.

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
