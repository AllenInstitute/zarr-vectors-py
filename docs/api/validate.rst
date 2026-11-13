Validation
==========

Conformance validator with levels 1 to 5, each including those below.
:doc:`/spec/validation/overview` says what each level checks.
``zarr_vectors.validate`` is *undecided*: treat it as internal.

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
