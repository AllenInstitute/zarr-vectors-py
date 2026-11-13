# Zarr Vectors and OME-Zarr

How Zarr Vectors relates to OME-Zarr, what OME-Zarr metadata a store
carries, and when to use each.

## What OME-Zarr is for

[OME-Zarr (NGFF)](https://ngff.openmicroscopy.org) is the Open
Microscopy Environment's Zarr format for bioimaging. Its core content is
dense, multi-resolution image arrays, with labels and plate layouts for
high-content screening.

Zarr Vectors stores sparse vector geometry: points, lines, graphs and
meshes. The two share no array layout, but a store's root attributes
carry two pieces of OME-Zarr metadata:

- a `multiscales` block (NGFF 0.4) listing the resolution levels. Each
  level's `scale` is its `bin_ratio` and its `translation` is half its
  `bin_shape`. See [Multiscale metadata](../../spec/multiscale/multiscale_metadata.md).
- an `ome` node (RFC 8, version 0.6) of type `collection`, with a
  `world` coordinate system and one `zv:level` node per level, so a
  store can be listed in an OME collection.

An OME-Zarr tool can find both blocks but cannot read the arrays as
images.

## Comparison

| | Zarr Vectors | OME-Zarr |
|---|---|---|
| Content | Vertices, links and objects | Dense images |
| Arrays | One variable-length byte cell per spatial chunk | A value at every index |
| Positions | World coordinates at every level | Array indices, converted by `scale` |
| `scale` in `multiscales` | Dimensionless bin ratio; not used to place vertices | Converts indices to physical units |
| Labels | Vertex or [object attributes](../../spec/object_model/object_attributes.md) | Integer label images |
| Plates and wells | None | Yes |

Because the meaning of `scale` differs, a reader that applies these
transforms to Zarr Vectors positions misplaces them.

## When to use which

Use OME-Zarr for images and Zarr Vectors for geometry extracted from
them, such as detected cells, traced neurons or meshes.

To overlay a store on an image, give the store the image's axis names,
order and units (`axes` at creation; see
[Dimensionality](../../spec/foundations/dimensionality.md)), and write
positions in those physical units. An RFC 8 collection can then refer to
the store's `world` coordinate system by its `id`.
