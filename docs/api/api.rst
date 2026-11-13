Data API
========

.. automodule:: zarr_vectors.api
   :no-members:
   :no-special-members:

Every name here is also importable from ``zarr_vectors``.

Datasets
--------

:func:`~zarr_vectors.api.dataset.open` and
:func:`~zarr_vectors.api.dataset.create` return a
:class:`~zarr_vectors.api.dataset.Dataset`, the entry point to everything
else. ``open_dataset`` and ``create_dataset`` are aliases.

.. automodule:: zarr_vectors.api.dataset
   :members:
   :show-inheritance:

Schema and layout
-----------------

A :class:`~zarr_vectors.api.schema.Schema` describes the data and is all a
new store needs. :class:`~zarr_vectors.api.schema.Layout` sets the grid when
you need to fix it.

.. automodule:: zarr_vectors.api.schema
   :members:
   :show-inheritance:

Selections and queries
----------------------

A :class:`~zarr_vectors.api.select.Selection` describes a read (a bbox,
object ids, an attribute filter). A :class:`~zarr_vectors.api.select.Query`
is a lazy handle on one; nothing is read until you ask.

.. automodule:: zarr_vectors.api.select
   :members:
   :show-inheritance:

Read results
------------

One result type for every geometry kind, with per-geometry views such as
``result.polylines``.

.. automodule:: zarr_vectors.api.result
   :members:
   :show-inheritance:

Resolution levels
-----------------

A :class:`~zarr_vectors.api.level.Level` is one resolution of a dataset,
and the handle for its objects, groups and grid.

.. automodule:: zarr_vectors.api.level
   :members:
   :show-inheritance:

Objects
-------

Objects by id, as ``level.objects``. Indexing it reads only what you ask
for.

.. automodule:: zarr_vectors.api.objects
   :members:
   :show-inheritance:

Object groups
-------------

Named sets of object ids, as ``level.groups``.

.. automodule:: zarr_vectors.api.groups
   :members:
   :show-inheritance:

The chunk grid
--------------

Grid questions in physical units: will this fit, and which region is
this. Most callers do not need it.

.. automodule:: zarr_vectors.api.grid
   :members:
   :show-inheritance:

Editing
-------

Edits addressed by object and vertex rather than by storage location.
Read the copy-on-write warning below before assuming an edit failed.

.. automodule:: zarr_vectors.api.edit
   :members:
   :show-inheritance:

Coarsening methods
------------------

The method names ``build_pyramid(method=...)`` accepts in this
installation, including any a strategy package registered.

.. autofunction:: zarr_vectors.api.coarsen_methods
