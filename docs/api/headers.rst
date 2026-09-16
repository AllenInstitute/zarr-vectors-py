Headers
=======

.. automodule:: zarr_vectors.headers
   :no-members:

A store often comes from somewhere — a ``.trk`` file, an ``.swc``, a
Neuroglancer precomputed volume — and that source format has a header the
Zarr Vectors format has no field for. Discarding it loses provenance;
inventing fields for it would make every source format a change to the
spec.

So headers are kept whole and opaque, one group per format under
``/headers/<format>/``, with the header dict stored as that group's
attributes. Core round-trips them and looks inside none of them; typed
(de)serialisation belongs to whichever package understands the format.

.. autoclass:: zarr_vectors.headers.registry.HeaderRegistry
   :members:
   :show-inheritance:

Reached from a dataset as :attr:`~zarr_vectors.api.dataset.Dataset.headers`::

    import zarr_vectors as zv

    ds = zv.open("tracts.zarrvectors")
    ds.headers.add("trk", {"voxel_to_rasmm": [[1, 0, 0, 0], ...]})
    ds.headers.available_formats      # ['trk']
    ds.headers.get("trk")             # the dict back, unchanged
