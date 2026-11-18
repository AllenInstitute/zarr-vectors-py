# Specification

This section is the reference for the Zarr Vectors on-disk format as
`zarr-vectors-py` implements it: what a store contains, how it is
chunked, how levels and objects are recorded, and how a store is
validated. Each page has **Terms**, an **Introduction** and a
**Technical reference** ending in **Validation**.

It describes format version **0.10.0**
(`zarr_vectors.constants.FORMAT_VERSION`, whose docstring records the
version history). Its metadata is the
[Zarr Vectors convention](https://github.com/Andrew-Keenlyside/zarr-vectors)
v0.1.0. 0.9.x stores are refused until migrated in place with
`zarr_vectors.building.migrate_metadata_0_10` (see
[Migrating from 0.9.x](layout/root_metadata.md#migrating-from-09x));
older stores cannot be read. Where a page
and the code disagree, the code is authoritative. Start with
[Directory structure](layout/directory_structure.md).

## Design goals

- **Spatial random access.** A query reads only the chunks it overlaps,
  and only the bins it needs within them, on local disk or in an object
  store.
- **Pyramids in the format.** Coarser levels live in the same store,
  listed by a `multiscales` block from the
  [multiscales convention](https://github.com/zarr-conventions/multiscales). See
  [Multiscale metadata](multiscale/multiscale_metadata.md).
- **One layout for every geometry type.** Points, streamlines, graphs,
  skeletons and meshes share one chunk layout. They differ only in
  which arrays a level holds.
- **Plain Zarr v3.** The `zarr` library can open any store; no custom
  codec is needed. The per-chunk arrays hold Zarr Vectors encodings in
  `vlen-bytes` cells, a serializer whose Zarr spec is not final. See
  [Zarr v3 primer](foundations/zarr_v3_primer.md) and
  [Codec pipeline](foundations/codec_pipeline.md).
- **Implementable from this spec.** These pages are enough to write a
  conforming reader, writer or validator.

## Related specifications

| Specification | Relationship |
|---------------|-------------|
| [Zarr v3 core spec](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html) | Every store is a Zarr v3 hierarchy. |
| [Zarr Vectors convention](https://github.com/Andrew-Keenlyside/zarr-vectors) | Since 0.10.0, every node's metadata is this convention's `zarr_vectors` attribute, registered in `zarr_conventions`. |
| [Zarr conventions: multiscales](https://github.com/zarr-conventions/multiscales), [proj](https://github.com/zarr-conventions/proj) | The root's `multiscales` block lists the levels; a CRS is `proj:projjson`. |
| [OME-Zarr NGFF](https://ngff.openmicroscopy.org/) | The root carries an RFC 8 `ome` node (since 0.9.2) and no NGFF `multiscales`. See [OME-Zarr comparison](../explanation/comparisons/ome_zarr.md). |
| [Allen Institute Zarr Vectors specification](https://github.com/AllenInstitute/zarr_vectors) | The specification by Forrest Collman that `zarr-vectors-py` implements and extends. |
| [Neuroglancer precomputed](https://github.com/google/neuroglancer/tree/master/src/datasource/precomputed) | See [Neuroglancer precomputed comparison](../explanation/comparisons/neuroglancer_precomputed.md). |
| [TRX format](https://tractography-file-format.github.io/) | See [TRX comparison](../explanation/comparisons/trx_format.md). |

## Specification sections

```{toctree}
:maxdepth: 1

foundations/zarr_v3_primer
foundations/store_types
foundations/dimensionality
foundations/codec_pipeline
layout/directory_structure
layout/root_metadata
layout/level_groups
layout/chunk_arrays
layout/fragment_index_arrays
geometry_types/index
object_model/fragments
object_model/object_manifest
object_model/links
object_model/object_attributes
chunking/chunk_shape
chunking/bin_shape
chunking/chunk_vs_bin
chunking/attribute_chunking
chunking/rechunking
chunking/sharding
chunking/chunk_sources
chunking/octree_chunks
multiscale/multiscale_metadata
multiscale/pyramid_construction
multiscale/sparsity
validation/overview
validation/l1_structural
validation/l2_metadata
validation/l3_consistency
contributing/spec_change_process
contributing/adding_geometry_types
contributing/test_compliance
contributing/implementation_notes
schema
```
