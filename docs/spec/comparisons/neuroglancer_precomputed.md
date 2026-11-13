# Zarr Vectors and Neuroglancer precomputed

This page compares Zarr Vectors with the Neuroglancer precomputed
formats for annotations, skeletons and meshes, and describes how data
moves between them.

## Terms

**Neuroglancer precomputed**
: A family of formats that the Neuroglancer viewer reads directly over
  HTTP, defined in the
  [Neuroglancer repository](https://github.com/google/neuroglancer/tree/master/src/datasource/precomputed).
  Each layer is a directory or bucket prefix with an `info` JSON file
  that declares its type and layout.

**Precomputed annotations**
: Point, line, axis-aligned bounding box and ellipsoid annotations, with
  typed per-annotation `properties`, related-segment lists, a
  multi-level spatial index and an index by annotation ID.

**Precomputed skeletons**
: The `neuroglancer_skeletons` format: one encoded skeleton per segment
  (vertex positions, edges and optional per-vertex attributes),
  stored as one file per segment or in shards.

**Precomputed meshes**
: The `neuroglancer_legacy_mesh` format (a JSON manifest per segment
  naming its fragment files) and the multi-resolution
  `neuroglancer_multilod_draco` format.

**Segment properties**
: The `neuroglancer_segment_properties` format, which attaches labels
  and numeric values to segment IDs.

---

## Introduction

Both formats store chunked, spatially organised 3-D geometry for
viewing at scale. Precomputed is laid out for Neuroglancer to fetch;
it is written once by an export or pipeline step. A Zarr Vectors store
is a Zarr v3 hierarchy that a program reads and updates through a
library, with one chunk layout and object model shared by every
geometry type, including streamlines and graphs.

Stock Neuroglancer does not read Zarr Vectors stores. They are viewed
with ngplatform, a Neuroglancer build with a `zarr-vectors` data source
that opens a store from its URL followed by `/|zarr-vectors:`, or
exported to precomputed with `zarr-vectors-tools`. See the
[Neuroglancer overview](../../tutorials/neuroglancer/overview.md).

---

## Technical reference

### Geometry types

| Precomputed | Closest Zarr Vectors type | Notes |
|-------------|---------------------------|-------|
| `point` annotation | `point_cloud` | |
| `line` annotation | `line` | |
| `axis_aligned_bounding_box`, `ellipsoid` annotations | none | The parametric registry has `plane`, `line` and `sphere` types only |
| `neuroglancer_skeletons` | `skeleton`, `graph` | |
| `neuroglancer_legacy_mesh`, `neuroglancer_multilod_draco` | `mesh` | |
| none | `polyline`, `streamline` | |

See [Geometry types](../geometry_types/index.md) for each Zarr Vectors
type.

### Structural differences

| Property | Zarr Vectors | Precomputed |
|----------|--------------|-------------|
| Spatial unit of storage | A per-chunk cell holding every object's vertices in that chunk, with a fragment index | Skeletons and meshes: one object per file or shard entry; annotations: spatially chunked records |
| Objects that cross chunks | Fragments in each chunk, joined by the object manifest and by links whose cell is the source chunk (see [Links](../object_model/links.md)) | Skeletons are stored whole; legacy meshes are split into fragments named by a manifest |
| Resolution levels | Coarser levels in the same store (see [Multiscale metadata](../multiscale/multiscale_metadata.md)) | Per format: annotation spatial-index levels, multi-resolution Draco meshes |
| Per-vertex attributes | Any NumPy dtype | Skeleton `vertex_attributes` |
| Per-object attributes | `object_attributes/<name>` arrays | Segment properties, annotation `properties` |
| Metadata | Root, level and array attributes in `zarr.json` | `info` JSON per layer |

### Mesh faces and chunks

A Zarr Vectors mesh face whose vertices all lie in one chunk is stored
with that chunk. A face whose vertices span chunks is stored in a
links array whose cell is the face's source chunk, with the offsets to
the other vertices' chunks in the array's path (see
[Links](../object_model/links.md)). With
`encoding="draco"`, intra-chunk faces are inside each chunk's Draco
bitstream instead (see [Codec pipeline](../foundations/codec_pipeline.md)).

### Converting between the formats

The converters are in the companion package `zarr-vectors-tools`
(see its [Visualise](https://zarr-vectors-tools.readthedocs.io/en/latest/visualise.html)
page):

- **Export** (`zvtools convert STORE OUTPUT --format precomputed`, or
  `export_precomputed`): skeleton and graph stores become an unsharded
  `neuroglancer_skeletons` layer; mesh stores become a single-resolution
  `neuroglancer_legacy_mesh` layer with one fragment per segment.
  Object attributes become segment properties. Other geometry types,
  Draco-encoded mesh stores, sharded output and multi-resolution Draco
  meshes are not written; export a coarser level to a second layer.
- **Ingest**: precomputed skeleton and mesh layers can be read into
  Zarr Vectors stores.

`zarr-vectors-py` itself has no precomputed converter; see
[Precomputed export](../../tutorials/neuroglancer/precomputed_export.md).

### Validation

Precomputed output is not validated by `zarr-vectors-py`. Before
exporting, `zarr_vectors.validate.validate(store, level=4)` checks the
geometry rules the exporter relies on, such as a mesh's `link_width`
of at least 3. See [Validation](../validation/overview.md).
