Building stores
===============

This surface is for code that *writes* a store: ingest converters, pyramid
builders, exporters, repair tools. It is deliberately physical — chunk shapes,
fragments, shard layouts, manifests — because that is what such code is for.
Applications that only read should use the :doc:`api` surface instead.

The alternative to this page is reaching into ``zarr_vectors.core``, which is
internal: a name imported from there is a name nobody knows is load-bearing,
which is exactly how the last few layout refactors turned into downstream
breaks. Everything listed below is promised. If something you need is missing,
that is a gap to report rather than a reason to import from ``core``.

.. automodule:: zarr_vectors.building
   :no-members:

Format constants and array names
--------------------------------

The array-family names and layout sentinels the on-disk format is spelled
in.  Import these rather than writing the strings, so code keeps working if
a name moves.  All of them are importable from ``zarr_vectors.building``.
The :doc:`constants` page documents the values of some but not all of them —
``constants.rst`` renders only the constants that carry their own docstring.

.. hlist::
   :columns: 3

   * ``FORMAT_VERSION``
   * ``VERTICES``
   * ``VERTEX_ATTRIBUTES``
   * ``VERTEX_FRAGMENTS``
   * ``FRAGMENT_ATTRIBUTES``
   * ``OBJECT_INDEX``
   * ``OBJECT_ATTRIBUTES``
   * ``GROUPS``
   * ``GROUP_ATTRIBUTES``
   * ``LINKS``
   * ``LINK_FRAGMENTS``
   * ``LINK_ATTRIBUTES``
   * ``OBJECT_INDEX_LAYOUT_V1``
   * ``LINKS_IMPLICIT_SEQUENTIAL``
   * ``LINKS_IMPLICIT_BRANCHES``
   * ``XLEVEL_NONE``
   * ``XLEVEL_EXPLICIT``
   * ``DEFAULT_CROSS_LEVEL_DEPTH``
   * ``DEFAULT_CROSS_LEVEL_STORAGE``
   * ``COARSEN_PER_OBJECT``
   * ``CAP_FRAGMENT_INDEX``
   * ``CAP_MULTISCALE_LINKS``
   * ``CAP_PRESERVED_OBJECT_IDS``
   * ``CAP_SHARED_FRAGMENTS``

Stores, resolution levels and transactions
------------------------------------------

Opening and creating stores, adding and removing levels, and reading or
updating the two metadata documents.  :class:`~zarr_vectors.building.Group`
is the storage handle every other function here takes; only the methods
named in :data:`~zarr_vectors.building.GROUP_SUPPORTED_METHODS` are
promised.

.. autoclass:: zarr_vectors.building.Group
   :members:
   :show-inheritance:

.. autodata:: zarr_vectors.building.GROUP_SUPPORTED_METHODS

.. autofunction:: zarr_vectors.building.create_store

.. autofunction:: zarr_vectors.building.open_store

.. autofunction:: zarr_vectors.building.commit

.. autofunction:: zarr_vectors.building.session_for

.. autofunction:: zarr_vectors.building.create_resolution_level

.. autofunction:: zarr_vectors.building.get_resolution_level

.. autofunction:: zarr_vectors.building.remove_resolution_level

.. autofunction:: zarr_vectors.building.list_resolution_levels

.. autofunction:: zarr_vectors.building.list_available_ratios

.. autofunction:: zarr_vectors.building.read_root_metadata

.. autofunction:: zarr_vectors.building.update_root_metadata

.. autofunction:: zarr_vectors.building.read_level_metadata

.. autofunction:: zarr_vectors.building.update_level_metadata

Metadata records and grid arithmetic
------------------------------------

The two metadata records, and the functions that derive a level's chunk
shape, bin shape and grid extent from them.

.. autoclass:: zarr_vectors.building.RootMetadata
   :no-members:

.. autoclass:: zarr_vectors.building.LevelMetadata
   :no-members:

.. autofunction:: zarr_vectors.building.compute_bin_shape

.. autofunction:: zarr_vectors.building.compute_bin_ratio

.. autofunction:: zarr_vectors.building.get_level_bin_shape

.. autofunction:: zarr_vectors.building.get_level_chunk_shape

.. autofunction:: zarr_vectors.building.chunk_scale_factor

.. autofunction:: zarr_vectors.building.chunk_scale_from_root

.. autofunction:: zarr_vectors.building.level_chunk_scale

.. autofunction:: zarr_vectors.building.level_factor

.. autofunction:: zarr_vectors.building.validate_bin_shape_divides_chunk

.. autofunction:: zarr_vectors.building.validate_level_chunk_shape_against_root

.. autofunction:: zarr_vectors.building.level_grid_layout

.. autofunction:: zarr_vectors.building.compute_grid_shape

Multiscale metadata and coarsening strategies
---------------------------------------------

The NGFF-style multiscale document, and the registries a coarsening or
selection strategy package installs itself into.

.. autofunction:: zarr_vectors.building.read_multiscale_metadata

.. autofunction:: zarr_vectors.building.write_multiscale_metadata

.. autofunction:: zarr_vectors.building.upsert_level_transform

.. autofunction:: zarr_vectors.building.get_level_scale

.. autofunction:: zarr_vectors.building.get_level_translation

.. autofunction:: zarr_vectors.building.register_coarsen_strategy

.. autofunction:: zarr_vectors.building.register_selection_strategy

Array allocation
----------------

Creating the zarr arrays a level is made of, before anything is written
into them.

.. autofunction:: zarr_vectors.building.create_vertices_array

.. autofunction:: zarr_vectors.building.create_attribute_array

.. autofunction:: zarr_vectors.building.create_fragment_attribute_array

.. autofunction:: zarr_vectors.building.create_object_index_array

.. autofunction:: zarr_vectors.building.create_object_attributes_array

.. autofunction:: zarr_vectors.building.create_groupings_array

.. autofunction:: zarr_vectors.building.create_groupings_attributes_array

.. autofunction:: zarr_vectors.building.create_links_array

.. autofunction:: zarr_vectors.building.create_links_family

.. autofunction:: zarr_vectors.building.create_link_attributes_array

.. autofunction:: zarr_vectors.building.attribute_layout

Chunk I/O: vertices, fragments and attributes
---------------------------------------------

Reading and writing one grid cell at a time -- the per-chunk verbs an
ingest worker runs in parallel.

.. autofunction:: zarr_vectors.building.open_write_session

.. autofunction:: zarr_vectors.building.list_chunk_keys

.. autofunction:: zarr_vectors.building.chunk_vertex_count

.. autofunction:: zarr_vectors.building.read_chunk_vertices

.. autofunction:: zarr_vectors.building.read_chunk_vertex_buffer

.. autofunction:: zarr_vectors.building.read_fragment

.. autofunction:: zarr_vectors.building.read_attribute_fragment

.. autofunction:: zarr_vectors.building.read_chunk_attributes

.. autofunction:: zarr_vectors.building.read_chunk_fragment_attributes

.. autofunction:: zarr_vectors.building.read_vertex_fragment_index

.. autofunction:: zarr_vectors.building.write_chunk_vertices

.. autofunction:: zarr_vectors.building.write_chunk_fragments

.. autofunction:: zarr_vectors.building.write_chunk_attributes

.. autofunction:: zarr_vectors.building.write_chunk_fragment_attributes

Links
-----

Edge storage: the nested ``links/<delta>/<offsets>`` path vocabulary, the
per-cell readers and writers, and the finalisation pass.

.. autofunction:: zarr_vectors.building.links_path

.. autofunction:: zarr_vectors.building.links_group_path

.. autofunction:: zarr_vectors.building.link_attributes_path

.. autofunction:: zarr_vectors.building.link_attributes_group_path

.. autofunction:: zarr_vectors.building.intra_offsets

.. autofunction:: zarr_vectors.building.is_intra

.. autofunction:: zarr_vectors.building.parse_offsets

.. autofunction:: zarr_vectors.building.link_family_policy

.. autofunction:: zarr_vectors.building.links_has_perm

.. autofunction:: zarr_vectors.building.list_link_deltas

.. autofunction:: zarr_vectors.building.list_link_offsets

.. autofunction:: zarr_vectors.building.list_link_attribute_offsets

.. autofunction:: zarr_vectors.building.iter_link_cells

.. autofunction:: zarr_vectors.building.cell_endpoint_chunks

.. autofunction:: zarr_vectors.building.read_links

.. autofunction:: zarr_vectors.building.read_links_for_tuple

.. autofunction:: zarr_vectors.building.read_link_attributes

.. autofunction:: zarr_vectors.building.read_chunk_links

.. autofunction:: zarr_vectors.building.read_chunk_link_attributes

.. autofunction:: zarr_vectors.building.write_links

.. autofunction:: zarr_vectors.building.write_link_cells

.. autofunction:: zarr_vectors.building.write_link_attributes

.. autofunction:: zarr_vectors.building.write_link_attribute_cells

.. autofunction:: zarr_vectors.building.write_chunk_links

.. autofunction:: zarr_vectors.building.write_chunk_link_attributes

.. autofunction:: zarr_vectors.building.finalize_links

Object index, manifests and object attributes
---------------------------------------------

The index that says which fragments belong to which object, and the
per-object attribute arrays beside it.

.. autoclass:: zarr_vectors.building.ObjectIndexAppender
   :members:
   :show-inheritance:

.. autofunction:: zarr_vectors.building.object_count

.. autofunction:: zarr_vectors.building.encode_object_manifest_blocks

.. autofunction:: zarr_vectors.building.decode_object_manifest_blocks

.. autofunction:: zarr_vectors.building.expand_manifest_blocks

.. autofunction:: zarr_vectors.building.read_object_manifests

.. autofunction:: zarr_vectors.building.read_all_object_manifests

.. autofunction:: zarr_vectors.building.read_object_vertices

.. autofunction:: zarr_vectors.building.write_object_index

.. autofunction:: zarr_vectors.building.write_object_manifests

.. autofunction:: zarr_vectors.building.read_object_attributes

.. autofunction:: zarr_vectors.building.read_object_attribute_present_mask

.. autofunction:: zarr_vectors.building.write_object_attributes

Object groups
-------------

Named rows of object ids, and the parallel array of names that makes a
store self-describing.

.. autofunction:: zarr_vectors.building.read_all_groupings

.. autofunction:: zarr_vectors.building.read_group_object_ids

.. autofunction:: zarr_vectors.building.read_groupings_attributes

.. autofunction:: zarr_vectors.building.write_groupings

.. autofunction:: zarr_vectors.building.write_groupings_attributes

Presence manifests and array inventory
--------------------------------------

The coordinator-side verbs: what is actually on disk, and rebuilding the
``nonempty_chunks`` and ``arrays_present`` records after a parallel write
phase that deliberately did not stamp them.

.. autofunction:: zarr_vectors.building.refresh_arrays_present

.. autofunction:: zarr_vectors.building.rebuild_presence

.. autofunction:: zarr_vectors.building.per_chunk_array_paths

.. autofunction:: zarr_vectors.building.array_is_sharded

.. autofunction:: zarr_vectors.building.is_sharded

Sharding and rechunking
-----------------------

Re-layout of a store that already exists.  Both rewrite where bytes live,
so both are coordinator operations -- never run one while workers are
writing.

.. autofunction:: zarr_vectors.building.get_shard_info

.. autofunction:: zarr_vectors.building.shard_store

.. autofunction:: zarr_vectors.building.unshard_store

.. autofunction:: zarr_vectors.building.reshard

.. autoclass:: zarr_vectors.building.RechunkSpec
   :members:
   :show-inheritance:

.. autofunction:: zarr_vectors.building.rechunk

.. autofunction:: zarr_vectors.building.rechunk_by_attribute

Spatial partitioning
--------------------

Assigning geometry to cells, and repairing what crosses a cell boundary.

.. autofunction:: zarr_vectors.building.assign_chunks

.. autofunction:: zarr_vectors.building.chunks_intersecting_bbox

.. autofunction:: zarr_vectors.building.neighbouring_chunk_keys

.. autofunction:: zarr_vectors.building.split_polyline_at_boundaries

.. autofunction:: zarr_vectors.building.partition_cross_level_edges

.. autofunction:: zarr_vectors.building.chunk_local_to_global_offsets

.. autofunction:: zarr_vectors.building.apply_perm_inverse

.. note::

   ``zarr_vectors.building.build_vertex_chunk_mapping`` is exported and
   supported, but is not rendered here: its ``Returns:`` block is malformed
   and Napoleon mis-parses it. Read its docstring in the source until that is
   fixed.

Store-creating writers
----------------------

One call, one store.  These take the full physical vocabulary --
``chunk_shape``, ``bin_shape``, ``shard_shape``, ``compressor``, ``dtype``
-- which is why they live here rather than behind
:class:`~zarr_vectors.api.dataset.Dataset`.  The matching *readers* are not promoted:
:class:`~zarr_vectors.api.result.ReadResult` covers all five geometries.

.. autofunction:: zarr_vectors.building.write_points

.. autofunction:: zarr_vectors.building.write_lines

.. autofunction:: zarr_vectors.building.write_polylines

.. autofunction:: zarr_vectors.building.write_graph

.. autofunction:: zarr_vectors.building.write_mesh

Skeletons: the streaming writer
-------------------------------

The four-call streaming path for tree-structured morphology, which has no
one-shot counterpart.

.. autofunction:: zarr_vectors.building.init_skeleton_store

.. autofunction:: zarr_vectors.building.write_skeleton_chunk

.. autofunction:: zarr_vectors.building.write_skeleton_cross_chunk_links

.. autofunction:: zarr_vectors.building.finalize_skeleton_store

.. autofunction:: zarr_vectors.building.decompose_tree_to_paths

.. autofunction:: zarr_vectors.building.get_coordinate_offset

.. autofunction:: zarr_vectors.building.set_coordinate_offset

.. autofunction:: zarr_vectors.building.read_skeleton_by_segment_id

