API Reference
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
converters, pyramid builders, exporters. That code's job *is* the physical
layout, so hiding the layout from it would hide the thing it works on. It gets
the smaller, blunter surface instead, with the same promise attached.

Four other modules are supported. Two have pages of their own —
``zarr_vectors.constants`` (:doc:`constants`) and ``zarr_vectors.typing``
(:doc:`typing`). Two do not: ``zarr_vectors.exceptions``, whose classes are
named in the signatures that raise them, and ``zarr_vectors.headers``, which is
promised but currently has no rendered page at all. That is a documentation gap,
not a weaker promise.

What is *not* here
------------------

``zarr_vectors.core``, ``.encoding``, ``.spatial``, ``.lazy``, ``.ops``,
``.sharding``, ``.multiresolution``, ``._engine`` and ``.rechunk`` are
**internal**. They change without notice — they always did — so they are no
longer advertised here. Earlier versions of this page listed ``lazy`` and
``spatial`` as public API; that was wrong, and importing from them is what made
previous refactors into downstream breaks. If you need a name that only exists
in one of them, ask for it to be promoted into ``api`` or ``building`` rather
than importing it.

Two pages below document **undecided** modules — neither promised nor disowned:

``zarr_vectors.types`` (:doc:`types`)
    "The five store-creating writers are promoted into ``building`` and are
    supported there. The readers are superseded by ``Level.read()`` /
    ``ReadResult``, but cannot be deprecated until the api can carry per-vertex
    attributes for every geometry — pointing callers at a lossy replacement is
    worse than leaving them here."

``zarr_vectors.validate`` (:doc:`validate`)
    "Stable in practice and widely used, but its result objects have never been
    given a compatibility promise."

Treat both as internal until that changes. (``zarr_vectors.composite`` is
undecided too, and undocumented: multi-geometry stores round-trip, but the
layout they use to namespace each geometry has not been specified.)

Asking at runtime
-----------------

The tiers are not prose here and prose somewhere else: they are a manifest in
``zarr_vectors/_stability.py``, and :func:`zarr_vectors.stability` reads it::

    >>> import zarr_vectors as zv
    >>> zv.stability("zarr_vectors.api")
    'supported'
    >>> zv.stability("zarr_vectors.core.arrays")
    'internal'

It accepts any dotted name and answers for the module that contains it, so a
lint rule or an import audit can check a call site without a hard-coded list.
:func:`zarr_vectors.require_api` is the assertion form.

Pages
-----

.. toctree::
   :maxdepth: 1

   api
   building
   zarr_vectors
   constants
   typing
   types
   validate

The pages themselves are generated from docstrings with
``sphinx.ext.autodoc``, so what is rendered is what is in the source.
