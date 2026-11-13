Validation
==========

Conformance validator for Zarr Vectors stores, with levels 1 to 5, each
including the ones below it. See :doc:`/spec/validation/overview` for what each
level checks.

``zarr_vectors.validate`` is in the *undecided* tier: its result objects have
no compatibility promise, so treat it as internal.

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
   :exclude-members: ValidationResult

.. automodule:: zarr_vectors.validate.metadata
   :members:
   :undoc-members:

.. automodule:: zarr_vectors.validate.consistency
   :members:
   :undoc-members:
