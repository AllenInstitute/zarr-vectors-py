Geometry types
==============

Read and write functions per geometry type. Each takes a path or URL, a
Zarr store or an open group, and the readers return a dict.

``zarr_vectors.types`` is *undecided*: treat it as internal. Its writers
are supported through :doc:`building`. ``Level.read()`` in :doc:`api`
covers its readers, except per-vertex attributes on some geometries.

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
