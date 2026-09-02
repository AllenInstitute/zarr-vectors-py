# Zarr Vectors and OME-Zarr

## Terms

**OME-Zarr (NGFF)**
: Open Microscopy Environment Next-Generation File Format — a Zarr v2/v3
  based format for bioimaging data, primarily volumetric fluorescence and
  electron microscopy images. Defined at https://ngff.openmicroscopy.org.

**`multiscales` block**
: A JSON array in a Zarr group's `.zattrs` that declares a multi-resolution
  image pyramid, with one entry per resolution level. Shared between OME-Zarr
  and Zarr Vectors.

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
`multiscales` JSON block. Zarr Vectors borrows this block from OME-Zarr 0.5,
extending it with vector-geometry-specific keys (`bin_ratio`, `bin_shape`,
`object_sparsity`) while preserving full compatibility with the OME-Zarr
schema as seen by tools that do not understand the extensions.

This shared metadata convention makes Zarr Vectors stores partially visible to
OME-Zarr-aware tools, enables joint discovery with image data registered
in the same coordinate space, and positions Zarr Vectors as a complement to OME-Zarr
in multi-modal bioimaging workflows.

---

## Technical reference

### What Zarr Vectors borrows from OME-Zarr

| OME-Zarr feature | Zarr Vectors adoption | Notes |
|-----------------|-------------|-------|
| `multiscales` block structure | Adopted verbatim | `version`, `name`, `axes`, `datasets`, `coordinateTransformations` |
| `axes` array (name, type, unit) | Adopted | Zarr Vectors axis order follows OME-Zarr convention (slowest to fastest) |
| `coordinateTransformations` | Adopted | `scale` and `translation` transforms per level |
| OME-Zarr `version` field | Adopted | Value `"0.5"` |

### What Zarr Vectors adds to the `multiscales` block

Zarr Vectors adds the following keys to `multiscales` that are not part of the
OME-Zarr spec. OME-Zarr-aware tools will ignore these unknown keys:

| Key | Level | Description |
|-----|-------|-------------|
| `type: "zarr_vectors_multiscale"` | Top-level | Identifies this as a Zarr Vectors multiscales block |
| `datasets[].bin_ratio` | Per-level | Vector binning ratio (Zarr Vectors-specific) |
| `datasets[].bin_shape` | Per-level | Effective bin shape (redundant but explicit) |
| `datasets[].object_sparsity` | Per-level | Object thinning fraction |

### Coordinate transforms: Zarr Vectors encoding

Zarr Vectors uses `scale` and `translation` to encode the physical meaning of coarser
resolution metanodes. For a level with `bin_ratio = [r_0, r_1, r_2]` and
`bin_shape = [b_0, b_1, b_2]`:

```json
"coordinateTransformations": [
  {"type": "scale",       "scale":       [r_0, r_1, r_2]},
  {"type": "translation", "translation": [b_0/2, b_1/2, b_2/2]}
]
```

**Scale** encodes the bin ratio: a coarser-level metanode position must be
multiplied by `r_d` to recover physical position. This matches the OME-Zarr
convention for image downscaling, where `scale[d] = downscale_factor[d]`.

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

1. Find a valid `multiscales` block in `.zattrs`. ✓
2. Read `axes`, `datasets`, and `coordinateTransformations`. ✓
3. Ignore `type: "zarr_vectors_multiscale"` (unknown type, skipped). ✓
4. Attempt to open `0/vertices/` as an image array.
   - The array is `(Cx, Cy, Cz, N_max, D)` float32 — 5-D.
   - Most image viewers expect 2–4-D arrays. The viewer may reject the
     array or display it as a meaningless 5-D volume. ✗

The practical result: OME-Zarr metadata tools (validators, metadata
registries, data portals) can index and discover Zarr Vectors stores. Interactive
image viewers will not render the data usefully but will not crash on the
metadata.

### What Zarr Vectors does not borrow from OME-Zarr

**Image arrays.** OME-Zarr image arrays are dense (C × Z × Y × X) arrays
where every pixel has a value. Zarr Vectors vertex arrays are ragged (one array per
chunk, variable number of vertices). There is no compatibility at the array
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
│   ├── .zattrs               ← multiscales with spatial calibration
│   └── 0/ 1/ 2/              ← resolution levels
└── tracts.zarrvectors/       ← Zarr Vectors streamlines in the same RAS space
    ├── .zattrs               ← multiscales with matching axes/units
    └── 0/ 1/
```

Both stores declare `"axes": [{"name": "z", "unit": "micrometer"}, …]`.
A viewer that understands both formats (such as Neuroglancer via
`zv-ngtools`) can overlay them in physical space using the coordinate
transforms from each store's `multiscales` block.

### OME-Zarr compatibility checklist

To maximise compatibility of a Zarr Vectors store with OME-Zarr tools:

- [ ] Include `multiscales` block with `version: "0.5"`.
- [ ] Include `axes` with `name`, `type`, and `unit` for all D axes.
- [ ] Ensure `coordinateTransformations` has exactly one `scale` and one
      `translation` per level, in that order.
- [ ] Set `axis_units` in root `.zattrs` consistently with the `axes` unit
      declarations.
- [ ] Run `zarr.consolidate_metadata(store)` after writing for fast
      metadata discovery.

Stores passing these checks will be correctly indexed by OME-Zarr metadata
tools and will display correct physical coordinates in viewers that read
the transforms.
