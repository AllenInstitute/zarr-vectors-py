# Specification

This section is the authoritative reference for the **Zarr Vectors
(ZV)** as implemented in `zarr-vectors-py`. It is written to serve two
audiences simultaneously:

- **Users** who want to understand precisely what the format does and why,
  so they can make informed decisions about chunk sizes, metadata, and
  resolution pyramids.
- **Contributors** who want to implement new geometry types, extend the
  validation suite, or propose changes to the format itself.

Each page follows a consistent structure: a **Terms** section defining all
vocabulary used on that page, a plain-English **Introduction**, and a
detailed **Technical reference** section with schemas, pseudocode, and
worked examples.

---

## Design goals

Zarr Vectors was designed to satisfy the following constraints simultaneously:

**Cloud-native random access.** Spatial queries should require only the
minimum number of I/O operations, whether the store is on a local file
system or in an object store (S3, GCS). This is achieved through spatial
chunking at the chunk level and sub-chunk indexing at the bin level.

**Multi-resolution out of the box.** Viewers that render large-scale 3-D
data (Neuroglancer, napari, custom WebGL) need coarser representations for
overview rendering. Zarr Vectors encodes resolution pyramids natively, with
OME-Zarr-compatible metadata so any NGFF-aware viewer can discover them.

**Geometry-type agnostic layout.** Points, streamlines, graphs, and meshes
all use the same spatial chunking layout. This means a single I/O layer
handles all types; type-specific logic lives only in the arrays present
within each chunk group (e.g. meshes have a `links/<delta>` array that point
clouds do not).

**Format transparency.** A Zarr Vectors store is a plain Zarr v3 directory tree.
No binary wrapper, no proprietary codec required for the base format. Every
array can be opened with the standard `zarr` library.

**Contributor-accessible.** The format is fully documented in this spec so
that an external contributor can implement a compliant reader, writer, or
validator without access to the original authors.

---

## Relationship to upstream specifications

| Specification | Relationship |
|---------------|-------------|
| [Zarr v3 core spec](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html) | Zarr Vectors stores are valid Zarr v3 stores. All array storage, codec, and metadata conventions are inherited from Zarr v3. |
| [OME-Zarr NGFF](https://ngff.openmicroscopy.org/) | Zarr Vectors borrows the `multiscales` JSON block for resolution pyramid metadata. Zarr Vectors-specific keys are additive extensions. |
| [Allen Institute Zarr Vectors specification](https://github.com/AllenInstitute/zarr_vectors) | The original specification by Forest Collman. `zarr-vectors-py` implements and extends this spec. |
| [Neuroglancer precomputed](https://github.com/google/neuroglancer/tree/master/src/datasource/precomputed) | A related format for static Neuroglancer hosting. See [comparison](comparisons/neuroglancer_precomputed.md). |
| [TRX format](https://tractography-file-format.github.io/) | A streamline-specific format. See [comparison](comparisons/trx_format.md). |

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
chunking/chunk_shape
chunking/bin_shape
chunking/chunk_vs_bin
chunking/rechunking
chunking/sharding
multiscale/multiscale_metadata
multiscale/pyramid_construction
multiscale/sparsity
geometry_types/index
object_model/fragments
object_model/object_manifest
object_model/links
object_model/object_attributes
validation/overview
comparisons/neuroglancer_precomputed
comparisons/trx_format
comparisons/ome_zarr
contributing/spec_change_process
contributing/adding_geometry_types
contributing/test_compliance
```
