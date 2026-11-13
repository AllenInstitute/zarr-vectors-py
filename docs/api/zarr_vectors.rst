zarr\_vectors
=============

.. automodule:: zarr_vectors
   :no-members:
   :no-special-members:

The top-level package re-exports the :doc:`api` surface, so
``zarr_vectors.open`` is ``zarr_vectors.api.open``. Those names are
documented on :doc:`api`; ``zarr_vectors.building`` on :doc:`building`.

The names below exist only at the top level and describe the installed
package. ``zarr_vectors.__version__`` is the package version and
``zarr_vectors.__api_version__`` the API version that
:func:`~zarr_vectors.require_api` compares against.
``zarr_vectors.concurrency_contract`` is
:func:`zarr_vectors.building.concurrency_contract`.

.. autofunction:: zarr_vectors.stability

.. autofunction:: zarr_vectors.require_api

.. autodata:: zarr_vectors.FEATURES

.. autofunction:: zarr_vectors.runtime_capabilities
