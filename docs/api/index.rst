API reference
=============

There are two supported surfaces, split by what the caller is doing.

:doc:`api` — ``zarr_vectors.api`` — is for **using** data: opening a store,
selecting a region or a set of objects, reading it back, editing it. It is
addressed in bounds, axes, geometry kinds and resolution levels rather than in
chunks, bins, shards and compressors: those are defaults you can override
(``Layout``, ``build_pyramid(chunk_scale_factors=...)``) and counts a write
reports back, never arguments you are obliged to supply. An application belongs
here, and its names are re-exported from the top-level package, so
``zarr_vectors.open(...)`` and ``zarr_vectors.api.open(...)`` are the same
function.

:doc:`building` — ``zarr_vectors.building`` — is for **making** stores: ingest
converters, pyramid builders, exporters. That code works on the physical layout, so
this surface names cells, bins, shards and arrays directly.

Four other modules are supported, three with pages of their own:
``zarr_vectors.constants`` (:doc:`constants`), ``zarr_vectors.typing``
(:doc:`typing`) and ``zarr_vectors.headers`` (:doc:`headers`). The fourth,
``zarr_vectors.exceptions``, has no page; its classes are named in the
docstrings of the functions that raise them.

What is *not* here
------------------

``zarr_vectors.core``, ``.encoding``, ``.spatial``, ``.lazy``, ``.ops``,
``.sharding``, ``.multiresolution``, ``.rechunk``, ``._engine``, ``._xp``,
``._runtime``, ``._gds``, ``._api_version``, ``._stability`` and ``._version``
are **internal**: they change without notice. The names the package
re-exports from the private modules (``stability``, ``require_api``,
``FEATURES``, ``runtime_capabilities``, ``__api_version__``, ``__version__``)
are meant to be imported from ``zarr_vectors``. If you need a name that
exists only in an internal module, ask for it to be promoted into ``api`` or
``building`` rather than importing it.

Two pages below document **undecided** modules, neither promised nor
disowned. Treat them as internal:

``zarr_vectors.types`` (:doc:`types`)
    Its five store-creating writers are supported through ``building``. Its
    readers are covered by ``Level.read()`` and ``ReadResult``, except for
    per-vertex attributes on some geometries.

``zarr_vectors.validate`` (:doc:`validate`)
    Its result objects have no compatibility promise.

Three more undecided modules have no page: ``zarr_vectors.composite``
(multi-geometry stores, whose on-disk namespacing is not specified),
``zarr_vectors.octree_chunks`` (octree chunking of a level, an extension the
specification does not carry), and ``zarr_vectors.gpu`` (the optional GPU
extension; reach it through ``device=`` and
:func:`zarr_vectors.runtime_capabilities` rather than by importing it, see
:doc:`../how_to/gpu`).

Asking at runtime
-----------------

The tiers are a manifest in ``zarr_vectors/_stability.py``, and
:func:`zarr_vectors.stability` reads it::

    >>> import zarr_vectors as zv
    >>> zv.stability("zarr_vectors.api")
    'supported'
    >>> zv.stability("zarr_vectors.core.arrays")
    'internal'

It accepts a dotted name under any submodule and answers for the module that
contains it, so a lint rule or an import audit can check a call site without a
hard-coded list. It raises ``KeyError`` for a name the manifest does not cover,
which includes names reached directly on the top-level package, such as
``zarr_vectors.open`` or ``zarr_vectors.runtime_capabilities``; check those
through the module that defines them (``zarr_vectors.api.open``). To check that an
install provides what your code needs, use :func:`zarr_vectors.require_api`
with an API version range or feature names from :data:`zarr_vectors.FEATURES`.

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

The pages themselves are generated from docstrings with
``sphinx.ext.autodoc``, so what is rendered is what is in the source.
