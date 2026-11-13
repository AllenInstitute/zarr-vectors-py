Headers
=======

.. automodule:: zarr_vectors.headers
   :no-members:
   :no-special-members:

Source-format headers kept with a store. A store converted from a ``.trk``
file, an ``.swc`` file or a Neuroglancer precomputed volume can keep that
format's header, for which the Zarr Vectors format has no field.

Each header is kept whole, one group per format under ``/headers/<format>/``,
with the header dict stored as that group's attributes. This package stores
and returns them without interpreting them; parsing a header belongs to the
package that understands its format.

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
