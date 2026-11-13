API reference
=============

Two surfaces are supported:

- :doc:`api` (``zarr_vectors.api``) is for **using** data: opening a store,
  selecting a region or objects, reading and editing. It works in bounds,
  axes, geometry kinds and levels, not chunks or shards. Its names are
  re-exported from ``zarr_vectors``.
- :doc:`building` (``zarr_vectors.building``) is for **making** stores:
  ingest converters, pyramid builders, exporters. It names cells, bins,
  shards and arrays directly.

``zarr_vectors.constants`` (:doc:`constants`), ``zarr_vectors.typing``
(:doc:`typing`), ``zarr_vectors.headers`` (:doc:`headers`) and
``zarr_vectors.exceptions`` are also supported.

Other modules
-------------

``zarr_vectors.types`` (:doc:`types`) and ``zarr_vectors.validate``
(:doc:`validate`) are **undecided**: treat them as internal.
``zarr_vectors.composite``, ``zarr_vectors.octree_chunks`` and
``zarr_vectors.gpu`` are also undecided and have no page; reach the GPU
extension through ``device=`` (see :doc:`../how_to/gpu`).

``zarr_vectors.core``, ``.encoding``, ``.spatial``, ``.lazy``, ``.ops``,
``.sharding``, ``.multiresolution``, ``.rechunk`` and the underscore
modules are **internal** and change without notice. Import the names the
package re-exports from them (``stability``, ``require_api``,
``FEATURES``, ``runtime_capabilities``, ``__api_version__``,
``__version__``) from ``zarr_vectors``.

Asking at runtime
-----------------

:func:`zarr_vectors.stability` gives a name's tier::

    >>> import zarr_vectors as zv
    >>> zv.stability("zarr_vectors.api")
    'supported'
    >>> zv.stability("zarr_vectors.core.arrays")
    'internal'

It answers for the module containing a dotted name, and raises
``KeyError`` for names reached on the top-level package, such as
``zarr_vectors.open``; check those through their defining module
(``zarr_vectors.api.open``). :func:`zarr_vectors.require_api` checks an
API version range or feature names from :data:`zarr_vectors.FEATURES`.

Pages
-----

.. toctree::
   :maxdepth: 1

   api
   building
   zarr_vectors
   constants
   typing
   headers
   types
   validate
