Validation
==========

Five-level conformance validator for Zarr Vectors stores. See
:doc:`/spec/validation/overview` for a full description of each level.

``zarr_vectors.validate`` is *undecided* tier in
``zarr_vectors/_stability.py``: "Stable in practice and widely used, but its
result objects have never been given a compatibility promise." Treat it as
internal until that promise is made.

.. automodule:: zarr_vectors.validate
   :members:
   :undoc-members:
   :show-inheritance:

ValidationResult
----------------

.. autoclass:: zarr_vectors.validate.ValidationResult
   :members:
   :undoc-members:
   :show-inheritance:

Individual validation modules
------------------------------

.. automodule:: zarr_vectors.validate.structure
   :members:
   :undoc-members:

.. automodule:: zarr_vectors.validate.metadata
   :members:
   :undoc-members:

.. automodule:: zarr_vectors.validate.consistency
   :members:
   :undoc-members:
