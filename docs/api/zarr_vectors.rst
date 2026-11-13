zarr\_vectors
=============

.. automodule:: zarr_vectors
   :no-members:
   :no-special-members:

The flat top-level namespace. Importing ``zarr_vectors`` re-exports the whole
:doc:`api` surface — ``Dataset``, ``Level``, ``Query``, ``Selection``,
``ReadResult``, ``Schema``, ``Layout``, ``Grid``, ``ObjectCatalog``,
``EditPlan``, and the ``open`` / ``create`` entry-points — so
``zarr_vectors.open(...)`` and ``zarr_vectors.api.open(...)`` are the same
object under two names.

Those names are documented once, on the page that carries the reasoning for
them: :doc:`api` for reading, querying and editing stores, and :doc:`building`
for tools that create them. ``zarr_vectors.building`` is the same module as
the one documented there.

The names below exist only at the top level and describe the installed
package rather than a store. ``zarr_vectors.__version__`` is the package
version and ``zarr_vectors.__api_version__`` the API version that
:func:`~zarr_vectors.require_api` compares against.
``zarr_vectors.concurrency_contract`` is the same function as
:func:`zarr_vectors.building.concurrency_contract`.

.. autofunction:: zarr_vectors.stability

.. autofunction:: zarr_vectors.require_api

.. autodata:: zarr_vectors.FEATURES

.. autofunction:: zarr_vectors.runtime_capabilities
