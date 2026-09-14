zarr\_vectors
=============

.. automodule:: zarr_vectors
   :no-members:

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

What is left below is the handful of names that exist only at the top level:
they answer questions about the installed package rather than about a store.

.. autofunction:: zarr_vectors.stability

.. autofunction:: zarr_vectors.require_api

.. autodata:: zarr_vectors.FEATURES
