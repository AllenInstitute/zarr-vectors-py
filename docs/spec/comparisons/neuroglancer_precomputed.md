# Zarr Vectors and Neuroglancer precomputed

## Terms

**Neuroglancer precomputed**
: A family of static binary formats for serving volumetric image data,
  annotation data, skeleton data, and mesh data to the Neuroglancer 3-D
  viewer. Data is stored as a directory of binary chunk files served over
  HTTP. Defined by Google at
  https://github.com/google/neuroglancer/tree/master/src/datasource/precomputed.

**Precomputed annotations**
: A precomputed sub-format for point, line, axis-aligned bounding box, and
  ellipsoid annotations. Each annotation is a small fixed-size binary record.
  Annotations are stored in spatial chunks indexed by a regular 3-D grid.

**Precomputed skeletons**
: A precomputed sub-format for neuronal morphology skeletons. Each skeleton
  object is a binary blob containing vertex positions and edge pairs, stored
  in a spatially indexed shard file.

**Precomputed meshes**
: A precomputed sub-format for triangulated surface meshes. Stored as
  draco-compressed or raw binary per-fragment files, indexed via a manifest
  JSON. Two sub-formats exist: the legacy per-fragment format and the newer
  sharded mesh format.

**`info` file**
: The JSON metadata file at the root of a Neuroglancer precomputed layer
  that declares the data type, scales, voxel resolution, chunk sizes, and
  layer-specific configuration. Analogous to Zarr Vectors' root `.zattrs`.

---

## Introduction

Zarr Vectors and Neuroglancer precomputed share the same fundamental goal —
spatially indexed, multi-resolution storage of 3-D geometry data — but
were designed for different constraints. Precomputed is a read-only static
serving format: data is written once and served over HTTP to Neuroglancer
clients. Zarr Vectors is a read-write cloud-native format: data can be read and
written from any Zarr-compatible environment, and the format is not tied
to any specific viewer.

Understanding the relationship between the two formats is important for
users of `zv-ngtools`, which translates Zarr Vectors stores into Neuroglancer layers
and optionally exports them to precomputed format for static hosting.

---

## Technical reference

### Annotation data

#### Precomputed annotation format

Precomputed annotations support four geometric primitives: `point`,
`line`, `axis_aligned_bounding_box`, and `ellipsoid`. Each annotation
is serialised as a compact fixed-size binary record:

- **Point:** 12 bytes (three float32 x,y,z).
- **Line:** 24 bytes (two point3).
- **Bounding box:** 24 bytes (min point3 + max point3).
- **Ellipsoid:** 36 bytes (centre point3 + radii point3 + angles point3).

Annotations are stored in spatial chunks; the chunk grid is declared in
the `info` file. Multiple spatial scales (levels of detail) are supported
via separate scale entries in `info`.

Precomputed annotations do not support:
- Ordered paths (no polyline or streamline type).
- Per-annotation scalar attributes (only `id`, `type`, and geometry).
- Discrete object model with per-object retrieval by ID.

#### Zarr Vectors equivalent

| Precomputed type | Closest Zarr Vectors type | Notes |
|-----------------|-----------------|-------|
| `point` | `point_cloud` | Zarr Vectors adds per-vertex attributes and multi-resolution |
| `line` | `line` | Zarr Vectors adds per-vertex attributes |
| `axis_aligned_bounding_box` | `mesh` (degenerate box) | No native Zarr Vectors bbox type; store as 8-vertex mesh |
| `ellipsoid` | `parametric/` group | Stored as parameter tuple; no native render in Zarr Vectors viewers |

Zarr Vectors is strictly richer than precomputed annotations: all precomputed
annotation types can be represented in Zarr Vectors (with some loss of compactness),
but not vice versa (streamlines, graphs, and skeletons have no precomputed
annotation equivalent).

#### `zv-ngtools` translation

`zv-ngtools` serves Zarr Vectors point clouds and line stores as Neuroglancer
annotation layers by translating fragment slices to annotation chunk binaries
on the fly:

```python
from ngtools.local.viewer import LocalNeuroglancer

viewer = LocalNeuroglancer()
viewer.add("points.zarrvectors")   # served as Neuroglancer point annotations
```

### Skeleton data

#### Precomputed skeleton format

Precomputed skeletons are stored in the `skeletons/` sub-directory. Each
skeleton object has a binary file containing:

- A header with vertex count and edge count.
- A `(n_vertices, 3)` float32 vertex position array.
- A `(n_edges, 2)` uint32 edge array (local vertex index pairs).

No spatial indexing: to retrieve one skeleton, fetch its binary file
directly. Spatial queries require fetching all skeleton files in a region.

Precomputed skeletons also support a sharded format (`sharding_index` +
shard files) that packs many skeletons into fewer files for cloud efficiency.

Attribute support: precomputed skeletons support per-vertex `vertex_attributes`
(float scalars and float vectors) stored alongside vertex positions.

#### Zarr Vectors skeleton vs precomputed skeleton

| Property | Zarr Vectors `skeleton` | Precomputed skeleton |
|----------|---------------|---------------------|
| Spatial index | Yes (fragment index + chunk grid) | No (direct file per object) |
| Multi-resolution | Yes | No |
| Per-vertex attributes | Yes (arbitrary dtypes) | Yes (float32 only) |
| Per-object attributes | Yes | No |
| SWC compatibility | Yes (`swc_type`, `radius`) | No |
| Cloud efficiency | Chunk-level parallelism | Shard-level parallelism |
| Write API | Yes | No (static format) |
| Viewer support | Via zv-ngtools | Native Neuroglancer |

#### Export to precomputed skeletons

The precomputed-skeletons exporter (and its CLI subcommand) lives in the
companion package **`zarr-vectors-tools`**. It translates a Zarr Vectors skeleton
store to the Neuroglancer sharded skeleton format, writing one shard
file per chunk. The output can be served directly by any HTTP server;
add the layer URL to Neuroglancer as a `skeletons` data source.

### Mesh data

#### Precomputed mesh formats

Two precomputed mesh formats exist:

**Legacy per-fragment format:** One binary file per mesh object, stored
at `meshes/<object_id>:0`. Each file contains:
- A `(n_vertices, 3)` float32 position array.
- A `(n_faces * 3,)` uint32 face array (no face structure; flat triplets).

**Sharded mesh format:** Meshes are split into spatial *fragments*, each
fragment covering one spatial chunk. Fragments are packed into shard files.
Draco compression is supported. A per-object manifest JSON declares which
fragment shards contain each object.

#### Zarr Vectors mesh vs precomputed mesh

| Property | Zarr Vectors `mesh` | Precomputed mesh (sharded) |
|----------|-----------|---------------------------|
| Spatial indexing | Chunk + fragment index | Fragment grid |
| Draco compression | Optional | Supported |
| Multi-resolution | Yes | No (single resolution) |
| Per-vertex attributes | Yes | No |
| Write API | Yes | No |
| Viewer support | Via zv-ngtools | Native Neuroglancer |

The Zarr Vectors mesh fragment model (faces assigned to chunks by centroid) is
intentionally similar to the Neuroglancer sharded fragment model. The
`precomputed_export` tool in `zv-ngtools` exploits this similarity to
produce precomputed shard files directly from Zarr Vectors chunk data with minimal
transformation.

#### Export to precomputed meshes

The precomputed-meshes exporter (and its CLI subcommand) lives in the
companion package **`zarr-vectors-tools`**.

### Summary: choosing between Zarr Vectors and precomputed

| Use case | Zarr Vectors | Precomputed |
|----------|-----|-------------|
| Cloud-native read-write pipeline | ✓ | ✗ |
| Static Neuroglancer serving (no write needed) | Via export | ✓ |
| Multi-resolution with object sparsity | ✓ | ✗ |
| Per-vertex float attributes (non-float32) | ✓ | ✗ |
| Streamlines / tractography | ✓ | ✗ |
| Native Neuroglancer support (no extra tools) | Via zv-ngtools | ✓ |
| Arbitrary query API (Python, Julia, R) | ✓ | ✗ |

For most new projects that originate data computationally, Zarr Vectors is the
primary format. Precomputed export is a one-way publication step for
sharing with Neuroglancer users who do not have `zv-ngtools` installed.
