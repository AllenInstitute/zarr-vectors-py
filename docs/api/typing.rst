Type aliases
============

Type aliases used throughout the ``zarr-vectors`` API. All are defined in
``zarr_vectors.typing`` (a supported module) and are *not* re-exported from the
top-level package: import them from ``zarr_vectors.typing``.

.. automodule:: zarr_vectors.typing
   :members:
   :undoc-members:
   :show-inheritance:

Reference table
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Alias
     - Definition
     - Notes
   * - ``Vertices``
     - ``NDArray[floating]``
     - ``(N, D)`` vertex positions
   * - ``Faces``
     - ``NDArray[integer]``
     - ``(F, L)`` face indices (3 = tri, 4 = quad)
   * - ``Edges``
     - ``NDArray[integer]``
     - ``(M, 2)`` edge endpoint indices
   * - ``ParentArray``
     - ``NDArray[integer]``
     - ``(N,)``; ``parent[i]`` is the parent of node ``i`` (-1 for root)
   * - ``ChunkCoords``
     - ``tuple[int, ...]``
     - Integer chunk grid coordinate
   * - ``ChunkShape``
     - ``tuple[float, ...]``
     - Physical chunk size per dimension
   * - ``BinCoords``
     - ``tuple[int, ...]``
     - Integer bin coordinate (bins tile chunks)
   * - ``BinShape``
     - ``tuple[float, ...]``
     - Physical bin edge lengths; must evenly divide ``ChunkShape``
   * - ``BinRatio``
     - ``tuple[int, ...]``
     - Per-dimension integer fold-change of bin shape relative to level 0
   * - ``BoundingBox``
     - ``tuple[NDArray, NDArray]``
     - ``(min_corner, max_corner)``, each shape ``(D,)``
   * - ``FragmentIndex``
     - ``int``
     - 0-based fragment index within a chunk
   * - ``FragmentRef``
     - ``tuple[ChunkCoords, FragmentIndex]``
     - Reference to one fragment
   * - ``ObjectManifest``
     - ``list[FragmentRef]``
     - Ordered fragment references composing one object
   * - ``ObjectID``
     - ``int``
     - Object identifier
   * - ``GroupID``
     - ``int``
     - Group-of-objects identifier
   * - ``CrossChunkLink``
     - ``tuple[tuple[ChunkCoords, int], tuple[ChunkCoords, int]]``
     - One 2-endpoint link as ``(chunk_coords, local_vertex_index)`` pairs
   * - ``LinksConvention``
     - ``str``
     - ``explicit``, ``implicit_sequential``, ``implicit_sequential_with_branches``
   * - ``ObjectIndexConvention``
     - ``str``
     - ``standard`` or ``identity``
   * - ``CrossChunkStrategy``
     - ``str``
     - ``boundary_deduplication``, ``explicit_links``, ``both``
   * - ``GeometryType``
     - ``str``
     - One of the ``GEOM_*`` constants
   * - ``AggregationMethod``
     - ``str``
     - ``mean``, ``sum``, ``mode``, ``count``, ``min``, ``max``
   * - ``AttributeAggregation``
     - ``dict[str, AggregationMethod]``
     - Attribute name to coarsening aggregation
