# Zarr Vectors and OME-Zarr

This page describes the OME-Zarr metadata a Zarr Vectors store carries,
what it does and does not mean to an OME-Zarr reader, and how the two
formats differ.

## Terms

**OME-Zarr (NGFF)**
: The Open Microscopy Environment's Zarr-based format for bioimaging
  data, defined at <https://ngff.openmicroscopy.org>. Its core content
  is dense multi-resolution image arrays.

**`multiscales` block**
: The OME-Zarr metadata that lists the resolution levels of a pyramid,
  with `axes`, one `datasets` entry per level, and per-level
  `coordinateTransformations`.

**`ome` node**
: The `ome` object in the root attributes, written as an OME-Zarr
  RFC 8 `collection` node so that a store can be listed in an OME
  collection.

---

## Introduction

Zarr Vectors stores sparse vector geometry, and OME-Zarr stores dense
images, so the two share no array layout. A store does carry two pieces
of OME-Zarr metadata in its root attributes: a `multiscales` block
describing its resolution levels, and an RFC 8 `ome` node. An OME-Zarr
tool can find both, but it cannot read the arrays as images.

---

## Technical reference

### The `multiscales` block

`write_multiscale_metadata` (`zarr_vectors/core/multiscale.py`) writes
one entry. For a 3-D store with one level whose bin shape is
`[100.0, 100.0, 100.0]`:

```json
"multiscales": [{
  "version": "0.4",
  "name": "default",
  "axes": [
    {"name": "x", "type": "space"},
    {"name": "y", "type": "space"},
    {"name": "z", "type": "space"}
  ],
  "datasets": [{
    "path": "0",
    "coordinateTransformations": [
      {"type": "scale", "scale": [1.0, 1.0, 1.0]},
      {"type": "translation", "translation": [50.0, 50.0, 50.0]}
    ]
  }],
  "metadata": {"format": "zarr_vectors"}
}]
```

- `axes` is the store's axis list and the source of its dimensionality
  (see [Dimensionality](../../spec/foundations/dimensionality.md)). Each axis
  has a `unit` only when one was given.
- Each `datasets` entry names a level group. Its `scale` is the
  level's `bin_ratio` and its `translation` is half the level's
  `bin_shape`; the translation is omitted when it is all zeros.
- `metadata.format = "zarr_vectors"` marks the block as written by this
  format. No other key is added to the block. A level's `bin_ratio` and
  `bin_shape` are read back from these transforms; its other values,
  such as `object_sparsity`, are in the level's `zarr_vectors_level`
  attributes (see [Level groups](../../spec/layout/level_groups.md)).

The meaning of `scale` differs between the formats. In OME-Zarr,
`scale` converts array indices to physical units. Zarr Vectors stores
vertex positions in world coordinates at every level, so its `scale` is
a dimensionless bin ratio and is not needed to place vertices. A
reader that applies these transforms to Zarr Vectors positions would
misplace them. The full block is described in
[Multiscale metadata](../../spec/multiscale/multiscale_metadata.md).

### The `ome` node

The root attributes also hold an `ome` object
(`zarr_vectors/core/ome.py`):

```json
"ome": {
  "version": "0.6",
  "type": "collection",
  "name": "example",
  "attributes": {
    "scene": {
      "coordinateSystems": [{
        "id": "world",
        "name": "world",
        "axes": [
          {"name": "x", "type": "space"},
          {"name": "y", "type": "space"},
          {"name": "z", "type": "space"}
        ]
      }],
      "coordinateTransformations": []
    }
  },
  "nodes": [{"type": "zv:level", "name": "0"}]
}
```

- `name` is the store's directory name without its suffix.
- `nodes` has one `zv:level` leaf per resolution level. The `zv:`
  prefix marks a node type defined by this format, whose structure
  RFC 8 does not constrain. Nodes added by other tools are kept when
  the levels are refreshed.
- The `world` coordinate system carries the store's axes and units.
  `coordinateTransformations` is empty because vertices are already in
  world coordinates.

The `zarr_vectors` and `multiscales` blocks remain the source of every
value the package reads; the `ome` node repeats information for RFC 8
readers. `zarr_vectors.building.stamp_ome_node(store)` adds the node to
a store that lacks one.

### What OME-Zarr content has no counterpart

- **Image arrays.** OME-Zarr images are dense arrays with a value at
  every index. Zarr Vectors per-chunk arrays hold one variable-length
  byte cell per spatial chunk (see
  [Zarr v3 primer](../../spec/foundations/zarr_v3_primer.md)), so an image
  reader cannot display them.
- **Labels.** OME-Zarr labels are integer image arrays. Zarr Vectors
  stores labels as vertex or object attributes (see
  [Object attributes](../../spec/object_model/object_attributes.md)).
- **High-content screening.** OME-Zarr's plate and well layout has no
  equivalent.

### Placing a store beside an image

To overlay a Zarr Vectors store on an OME-Zarr image, give the store
the same axis names, order and units as the image's physical space
(through `axes` at creation, see
[Dimensionality](../../spec/foundations/dimensionality.md)) and write vertex
positions in those physical units. An RFC 8 collection elsewhere can
then reference the store's `world` coordinate system by its `id`.

### Validation

Level 1 (`validate/structure.py`) warns when the root attributes have
no `ome` node. No level reads the `multiscales` block itself. See [Validation](../../spec/validation/overview.md).
