Headers
=======

.. automodule:: zarr_vectors.headers
   :no-members:
   :no-special-members:

Source-format headers (for example from ``.trk``, ``.swc`` or Neuroglancer
precomputed files) kept with a store. Each is stored whole, uninterpreted,
as the attributes of ``/headers/<format>/``.

.. autoclass:: zarr_vectors.headers.registry.HeaderRegistry
   :members:
   :show-inheritance:

Reached from a dataset as :attr:`~zarr_vectors.api.dataset.Dataset.headers`::

    import zarr_vectors as zv

    ds = zv.open("tracts.zarrvectors", mode="r+")
    ds.headers.add("trk", {"voxel_to_rasmm": [[1, 0, 0, 0], [0, 1, 0, 0],
                                              [0, 0, 1, 0], [0, 0, 0, 1]]})
    ds.headers.available_formats      # ['trk']
    ds.headers.get("trk")             # the dict back, unchanged
