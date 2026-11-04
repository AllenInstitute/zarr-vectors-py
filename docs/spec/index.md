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

This specification describes format version **0.9.2**
(`zarr_vectors.constants.FORMAT_VERSION`). Stores older than 0.9.0 are not
readable; the version history is recorded in the docstring of that constant.
Where a page and the code disagree, the code is authoritative.

---

## A store at a glance

```
tracts.zarrvectors/
├── zarr.json                  root attributes: zarr_vectors, multiscales, ome
├── 0/                         resolution level 0 (finest); 1/, 2/ … are coarser
│   ├── zarr.json              level attributes: zarr_vectors_level
│   ├── vertices/              ┐ chunk arrays: one vlen-bytes array each, shape =
│   ├── vertex_fragments/      │ the chunk grid, one cell per spatial chunk
│   ├── vertex_attributes/<n>/ │ (files at <array>/c/i/j/k)
│   ├── links/<delta>/<offs>/  │
│   ├── link_fragments/        ┘
│   ├── object_index/          per-object manifests (which chunks/fragments)
│   ├── object_attributes/<n>  per-object columns (plain arrays)
│   └── groups, group_attributes/<n>
└── parametric/                optional algebraic objects
```

Start with [Directory structure](layout/directory_structure.md), then
[Chunk arrays](layout/chunk_arrays.md) for the container every spatial array
shares, and [Fragment-index arrays](layout/fragment_index_arrays.md) for how a
chunk's rows are partitioned.

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
within each level group (e.g. meshes have a `links/<delta>/<offsets>` family
of arrays that point clouds do not).

**Format transparency.** A Zarr Vectors store is a plain Zarr v3 directory tree.
No binary wrapper, no proprietary codec is required for the base format, and
every array can be opened with the standard `zarr-python` v3 library. The
spatial arrays use the Zarr v3 `vlen-bytes` serializer, for which the Zarr
specification is not yet final: zarr-python emits an
`UnstableSpecificationWarning` when opening them, and the payload inside each
cell is a Zarr Vectors encoding (documented in this specification) rather than
a typed Zarr array.

**Contributor-accessible.** The format is fully documented in this spec so
that an external contributor can implement a compliant reader, writer, or
validator without access to the original authors.

---

## Relationship to upstream specifications

| Specification | Relationship |
|---------------|-------------|
| [Zarr v3 core spec](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html) | Zarr Vectors stores are valid Zarr v3 stores. All array storage, codec, and metadata conventions are inherited from Zarr v3. |
| [OME-Zarr NGFF](https://ngff.openmicroscopy.org/) | Zarr Vectors borrows the `multiscales` JSON block (which also holds the store's axes) for resolution pyramid metadata, and since 0.9.2 writes an additive RFC 8 `ome` node so an OME collection can name the store. Zarr Vectors-specific keys are additive extensions. |
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
chunking/attribute_chunking
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
