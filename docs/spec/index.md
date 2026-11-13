# Specification

This section is the reference for the Zarr Vectors on-disk format as
`zarr-vectors-py` implements it: what a store contains, how it is
chunked, how levels and objects are recorded, and how a store is
validated. Each page has **Terms**, an **Introduction** and a
**Technical reference** ending in **Validation**.

## Design goals

- **Spatial random access.** A query reads only the chunks it overlaps,
  and only the bins it needs within them, on local disk or in an object
  store.
- **Pyramids in the format.** Coarser levels live in the same store,
  described by an OME-Zarr `multiscales` block. See
  [Multiscale metadata](multiscale/multiscale_metadata.md).
- **One layout for every geometry type.** Points, streamlines, graphs,
  skeletons and meshes share one chunk layout. They differ only in
  which arrays a level holds.
- **Plain Zarr v3.** The `zarr` library can open any store; no custom
  codec is needed. See [Zarr v3 primer](foundations/zarr_v3_primer.md).
- **Implementable from this spec.** These pages are enough to write a
  conforming reader, writer or validator.

## Related specifications

| Specification | Relationship |
|---------------|-------------|
| [Zarr v3 core spec](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html) | Every store is a Zarr v3 hierarchy. |
| [OME-Zarr NGFF](https://ngff.openmicroscopy.org/) | The root attributes carry a `multiscales` block and an RFC 8 `ome` node. See [OME-Zarr comparison](../explanation/comparisons/ome_zarr.md). |
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
chunking/rechunking
chunking/sharding
chunking/chunk_sources
chunking/attribute_chunking
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
schema
```
