.. zarr-vectors documentation master file

.. image:: zarr-vectors.png
   :width: 55%
   :align: center
   :alt: zarr-vectors

----

**zarr-vectors** stores spatial geometry (point clouds, streamlines, graphs,
skeletons and meshes) in chunked, multiscale Zarr v3 stores. A spatial query
reads only the chunks it needs, on local disk or in cloud storage, and
resolution pyramids let viewers such as Neuroglancer stream data at any scale.
It implements `Zarr Vectors <https://github.com/AllenInstitute/zarr_vectors>`_,
specified by Forrest Collman at the Allen Institute for Brain Science.

.. figure:: _static/figures/zarr-vectors-overview.png
   :width: 100%
   :align: center
   :alt: Overview of Zarr Vectors. Panel a plots file size against dataset
         complexity, crossing the memory ceiling. Panel b shows point clouds,
         skeletons and meshes decomposed into vertices, links, faces and
         attributes, stacked into a resolution pyramid that supports scalable
         read/write and multiscale visualisation.
   :figclass: zv-figure

   **Overview of Zarr Vectors.**
   **a** — Complex and detailed derivative datasets exceed GPU and system
   memory. **b** — Summary of the Zarr Vectors concept, showing input data
   types, construction, and uses.

.. raw:: html

   <div class="zv-explainer-wrap">
     <iframe src="explainer.html" class="zv-explainer-frame"
             frameborder="0" allowfullscreen></iframe>
   </div>

----

| `Link to the GitHub repository <https://github.com/AllenInstitute/zarr-vectors-py>`__

Where to start
--------------

.. list-table::
   :widths: 35 65

   * - :doc:`getting_started/installation`
     - Install the package and its optional extras.
   * - :doc:`getting_started/quickstart`
     - Create, write, query and validate your first store.
   * - :doc:`explanation/concepts`
     - The terms used throughout: chunks, bins, fragments, objects and levels.
   * - :doc:`api/index`
     - The API reference, and which modules are supported.
   * - :doc:`spec/index`
     - The full Zarr Vectors specification.


.. toctree::
   :maxdepth: 1
   :caption: Getting started
   :hidden:

   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorials/data_types/point_clouds
   tutorials/data_types/polylines_streamlines
   tutorials/data_types/graphs_skeletons
   tutorials/data_types/meshes
   tutorials/data_types/mesh_streaming
   tutorials/multiscale/building_pyramids
   tutorials/multiscale/lazy_loading

.. toctree::
   :maxdepth: 1
   :caption: How-to guides
   :hidden:

   how_to/choose_chunk_and_bin
   how_to/memory_efficient_writes
   how_to/hpc_pipelines
   how_to/cloud_stores
   how_to/validate_and_repair
   how_to/convert_files
   how_to/view_in_neuroglancer
   how_to/gpu
   how_to/cite

.. toctree::
   :maxdepth: 1
   :caption: Explanation
   :hidden:

   explanation/concepts
   explanation/faq
   explanation/comparisons/neuroglancer_precomputed
   explanation/comparisons/ome_zarr
   explanation/comparisons/trx_format

.. toctree::
   :maxdepth: 1
   :caption: Specification
   :hidden:

   spec/index

.. toctree::
   :maxdepth: 1
   :caption: API reference
   :hidden:

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Benchmarks
   :hidden:

   benchmarks/index
