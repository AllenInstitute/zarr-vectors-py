# Specification

This section is the reference for the Zarr Vectors on-disk format as
implemented in `zarr-vectors-py`: what a store contains, how it is
chunked, how resolution levels and objects are recorded, and how a
store is validated. It is written for people choosing chunk sizes and
pyramid settings, and for contributors writing readers, writers or
validators.

Each page has a **Terms** section defining its vocabulary, an
**Introduction**, and a **Technical reference** with schemas, worked
examples and a **Validation** subsection.

---

## Design goals

**Spatial random access.** A spatial query reads only the chunks it
overlaps and, within a chunk, only the bins it needs, on a local file
system or in an object store.

**Resolution pyramids in the format.** Coarser levels are part of the
store, described by an OME-Zarr `multiscales` block in the root
attributes. See [Multiscale metadata](multiscale/multiscale_metadata.md).

**One layout for every geometry type.** Points, streamlines, graphs,
skeletons and meshes share the same chunk layout; types differ only in
which arrays a level holds. A point cloud has no `links/` group, for
example, while a mesh stores its faces under `links/`.

**Plain Zarr v3.** A store is a Zarr v3 hierarchy that the `zarr`
library can open; the base format needs no custom codec. See
[Zarr v3 primer](foundations/zarr_v3_primer.md).

**Implementable from this spec.** The pages in this section are meant
to be enough to write a conforming reader, writer or validator.

---

## Relationship to other specifications

| Specification | Relationship |
|---------------|-------------|
| [Zarr v3 core spec](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html) | Every store is a Zarr v3 hierarchy; array storage, codecs and metadata follow Zarr v3. |
| [OME-Zarr NGFF](https://ngff.openmicroscopy.org/) | The root attributes carry an NGFF `multiscales` block for the pyramid and an RFC 8 `ome` node. See [OME-Zarr comparison](../explanation/comparisons/ome_zarr.md). |
| [Allen Institute Zarr Vectors specification](https://github.com/AllenInstitute/zarr_vectors) | The specification by Forrest Collman that `zarr-vectors-py` implements and extends. |
| [Neuroglancer precomputed](https://github.com/google/neuroglancer/tree/master/src/datasource/precomputed) | A format for static Neuroglancer hosting. See [Neuroglancer precomputed comparison](../explanation/comparisons/neuroglancer_precomputed.md). |
| [TRX format](https://tractography-file-format.github.io/) | A streamline exchange format. See [TRX comparison](../explanation/comparisons/trx_format.md). |

---

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
