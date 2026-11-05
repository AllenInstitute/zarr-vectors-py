# Zarr Vectors and OME-Zarr

## Terms

**OME-Zarr (NGFF)**
: Open Microscopy Environment Next-Generation File Format — a Zarr v2/v3
  based format for bioimaging data, primarily volumetric fluorescence and
  electron microscopy images. Defined at https://ngff.openmicroscopy.org.

**`multiscales` block**
: A JSON array in a Zarr group's attributes that declares a multi-resolution
  image pyramid, with one entry per resolution level. Shared between OME-Zarr
  and Zarr Vectors (Zarr Vectors writes it bare at the store root, in the
  NGFF 0.4 layout).

**`ome` node (RFC 8)**
: An optional root attribute block (format 0.9.2) describing the store as
  an OME-Zarr RFC 8 `collection` node: one `zv:level` leaf per resolution
  level and a `scene` declaring a `world` coordinate system with the
  store's axes and units. It lets an OME collection reference a Zarr
  Vectors store by path.

**`coordinateTransformations`**
: Per-level or per-dataset affine transformations declared within a
  `multiscales` entry. Used by OME-Zarr-aware tools to map between pixel
  space and physical space.

**OME-Zarr axes**
: An array of axis descriptor objects (name, type, unit) that declares
  the semantic meaning and physical units of each array dimension. Present
  in the `multiscales` block.

**NGFF label**
: An OME-Zarr sub-format for integer segmentation label images stored as a
  multi-resolution Zarr array. Commonly used for cell or nucleus
  segmentation masks.

---

## Introduction

Zarr Vectors and OME-Zarr target different data types (sparse vector geometry vs
dense image volumes) but share a common metadata convention: the
`multiscales` JSON block. Zarr Vectors writes this block following the
OME-Zarr 0.4 layout; the per-level vector-specific parameters
(`bin_ratio`, `bin_shape`, `object_sparsity`, ...) are not extra keys in
it. `bin_ratio` and `bin_shape` are encoded in the `scale` and
`translation` transforms, and the rest live in each level group's
`zarr_vectors_level` attribute.

This shared metadata convention makes Zarr Vectors stores partially visible to
OME-Zarr-aware tools, enables joint discovery with image data registered
in the same coordinate space, and positions Zarr Vectors as a complement to OME-Zarr
in multi-modal bioimaging workflows.

---

## Technical reference

### What Zarr Vectors borrows from OME-Zarr

| OME-Zarr feature | Zarr Vectors adoption | Notes |
|-----------------|-------------|-------|
| `multiscales` block structure | Adopted | `version` (`"0.4"`), `name`, `axes`, `datasets` (each with `path` and `coordinateTransformations`). No top-level `coordinateTransformations` is written |
| `axes` array (name, type, unit) | Adopted | The only place a store's axes are kept. Default names are `x`, `y`, `z`, `w`, in that order. Unit is omitted when not declared |
| `coordinateTransformations` | Adopted with different semantics | `scale` and `translation` per level; `translation` omitted when zero. `scale` is a dimensionless bin ratio, not world-units-per-index |
| RFC 8 node (`ome`) | Added in format 0.9.2, optional | See the `ome` node term above |

### What Zarr Vectors adds to the `multiscales` block

One key, which OME-Zarr-aware tools will ignore:

| Key | Level | Description |
|-----|-------|-------------|
| `metadata.format: "zarr_vectors"` | Entry | Identifies the block as belonging to a Zarr Vectors store. (NGFF reserves `type` for the downsampling method, so `type` is not used.) |

The vector-specific per-level values `bin_ratio` and `bin_shape` are
derived from the transforms (`bin_ratio = scale`, `bin_shape =
2 × translation`); `object_sparsity` and the rest are in
`zarr_vectors_level` on each level group.

### Coordinate transforms: Zarr Vectors encoding

Zarr Vectors uses `scale` and `translation` to encode the physical meaning of coarser
resolution metanodes. For a level with `bin_ratio = [r_0, r_1, r_2]` and
`bin_shape = [b_0, b_1, b_2]`:

```javascript
"coordinateTransformations": [
  {"type": "scale",       "scale":       [r_0, r_1, r_2]},
  {"type": "translation", "translation": [b_0/2, b_1/2, b_2/2]}
]
```

**Scale** encodes the bin ratio: a coarser-level metanode position must be
multiplied by `r_d` to recover physical position. This is analogous to the
OME-Zarr convention for image downscaling, where `scale[d]` grows with the
downscale factor. (Level 0 has `scale = 1` and a translation of half the
base bin.)

**Translation** encodes the centroid offset: metanodes are positioned at
bin centroids, so they are offset by half a bin width from the bin origin.
OME-Zarr uses `translation` for the same purpose in image pyramids (the
centre of a downscaled pixel is offset from the corner).

This encoding means that an OME-Zarr-aware viewer correctly positions Zarr Vectors
metanodes in physical space using the standard `scale → translate` pipeline,
without any Zarr Vectors-specific code.

### What OME-Zarr tools see when opening a Zarr Vectors store

An OME-Zarr reader (e.g. `ome-zarr-py`, napari with `napari-ome-zarr`,
`zarr-viewer`) opening a Zarr Vectors store will:

1. Find a `multiscales` block in the root attributes. ✓
2. Read `axes`, `datasets`, and `coordinateTransformations`. ✓
3. Ignore `metadata.format` (unknown, skipped). ✓
4. Attempt to open `0/vertices/` as an image array.
   - It is a rank-`D` variable-length-bytes array whose shape is the
     level's chunk grid (one element per spatial chunk), not an image.
   - Image viewers will most likely reject it or show a meaningless
     volume. ✗

The practical result: OME-Zarr metadata tools (validators, metadata
registries, data portals) can index and discover Zarr Vectors stores. Interactive
image viewers will not render the data usefully but will not crash on the
metadata.

### What Zarr Vectors does not borrow from OME-Zarr

**Image arrays.** OME-Zarr image arrays are dense (C × Z × Y × X) arrays
where every pixel has a value. Zarr Vectors vertex arrays are ragged (one
variable-length byte payload per spatial chunk, with a variable number of
vertices). There is no compatibility at the array
data level.

**Label format.** OME-Zarr labels (segmentation masks) are stored as
integer image arrays with OME-Zarr conventions. Zarr Vectors does not adopt this
convention; semantic labels in Zarr Vectors are stored as per-vertex or per-object
attributes.

**Plate/Well convention.** OME-Zarr has HCS (High Content Screening)
conventions for plate/well/field data. Zarr Vectors has no equivalent.

### Using Zarr Vectors alongside OME-Zarr image data

A common workflow in connectomics and neuroscience registers Zarr Vectors
data against an OME-Zarr image volume in the same physical coordinate space:

```
experiment/
├── image.zarr/               ← OME-Zarr image (EM, light microscopy)
│   ├── zarr.json             ← multiscales with spatial calibration
│   └── 0/ 1/ 2/              ← resolution levels
└── tracts.zarrvectors/       ← Zarr Vectors streamlines in the same RAS space
    ├── zarr.json             ← multiscales with matching axes/units
    └── 0/ 1/
```

Both stores declare matching `axes`, e.g.
`[{"name": "x", "type": "space", "unit": "micrometer"}, …]`. (Axis *order*
follows each store's own array layout, so check it before overlaying.)
A viewer that understands both formats (such as Neuroglancer via
`zv-ngtools`) can overlay them in physical space using the coordinate
information from each store.

### OME-Zarr compatibility checklist

To maximise compatibility of a Zarr Vectors store with OME-Zarr tools:

- [ ] Declare `axes` with `name`, `type`, and a `unit` for all D axes
      (via `create_store(axes=...)` or `zv.Schema(axes=...)`); the
      type-level `write_*` functions leave units unset.
- [ ] Make sure the root has an `ome` node (format 0.9.2). Stores written
      by this version have one; for an older store run
      `zarr_vectors.building.stamp_ome_node(store)`, which is
      metadata-only. `validate()` (L1) warns when it is missing.

Stores passing these checks can be referenced from an OME collection and
will carry correct axes and units for viewers that read them.
