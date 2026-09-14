Data API
========

.. automodule:: zarr_vectors.api
   :no-members:

The sections below follow the order you meet them in: open a dataset, say what
you want stored or what you want back, read it, then work with the objects,
the grid, and edits. Every name here is also importable from ``zarr_vectors``
itself.

Datasets
--------

The entry point. :func:`~zarr_vectors.api.dataset.open` and
:func:`~zarr_vectors.api.dataset.create` hand back a
:class:`~zarr_vectors.api.dataset.Dataset`, which is the object every other
page here is reached from. (``open_dataset`` and ``create_dataset`` are aliases
of the same two functions, for callers who do not want to shadow the builtin.)

.. automodule:: zarr_vectors.api.dataset
   :members:
   :show-inheritance:

Schema and layout
-----------------

What the data *is*, kept separate from how it is stored — so creating a store
needs a :class:`~zarr_vectors.api.schema.Schema` and nothing else, and
:class:`~zarr_vectors.api.schema.Layout` is only reached for when the grid is
fixed from outside.

.. automodule:: zarr_vectors.api.schema
   :members:
   :show-inheritance:

Selections and queries
----------------------

How a read is described. A :class:`~zarr_vectors.api.select.Selection` is a
value — a bbox, some object ids, an attribute filter — and a
:class:`~zarr_vectors.api.select.Query` is a lazy handle on one that can be
narrowed and passed around before anything is read.

.. automodule:: zarr_vectors.api.select
   :members:
   :show-inheritance:

Read results
------------

What comes back, in one shape for all five geometries, with the per-geometry
views kept as derived properties so existing code still reads
``result.polylines``.

.. automodule:: zarr_vectors.api.result
   :members:
   :show-inheritance:

Resolution levels
-----------------

Where reads actually happen. A :class:`~zarr_vectors.api.level.Level` is one
resolution of a dataset, and it is also the handle through which objects,
groups and the grid are reached.

.. automodule:: zarr_vectors.api.level
   :members:
   :show-inheritance:

Objects
-------

Objects addressed by id rather than by which chunks their fragments landed in.
Reached as ``level.objects``; indexing it reads only what was asked for.

.. automodule:: zarr_vectors.api.objects
   :members:
   :show-inheritance:

Object groups
-------------

Named rows of object ids, reached as ``level.groups``. Naming them is what
makes a store readable without the writing application's source beside it.

.. automodule:: zarr_vectors.api.groups
   :members:
   :show-inheritance:

The chunk grid
--------------

Most callers never need this. It is here for the two questions about the grid
that are real — *will this fit?* and *which region is this?* — asked and
answered in physical units.

.. automodule:: zarr_vectors.api.grid
   :members:
   :show-inheritance:

Editing
-------

Edits named by what a thing is — "the third vertex of object 7" — instead of by
its physical address. Read the copy-on-write warning below before assuming an
edit failed.

.. automodule:: zarr_vectors.api.edit
   :members:
   :show-inheritance:

Coarsening methods
------------------

Which method names ``build_pyramid(method=...)`` will accept in this
installation, including any a strategy package registered on import.

.. autofunction:: zarr_vectors.api.coarsen_methods
