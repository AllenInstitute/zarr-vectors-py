# Frequently asked questions

## General

### Can I open a Zarr Vectors store with plain `zarr`?

Yes. A store is a valid Zarr v3 group, so `zarr.open_group(path)` opens every
array. You then decode fragments, manifests and links yourself, so use
`zv.open(path).read()` to read data; see [Core concepts](concepts.md).

### How does Zarr Vectors relate to OME-Zarr?

It uses the OME-Zarr `multiscales` metadata, so OME-Zarr viewers can find its
levels. OME-Zarr is for dense images; Zarr Vectors is for sparse geometry. See
[Zarr Vectors and OME-Zarr](comparisons/ome_zarr.md).

### Is the `.zarrvectors` extension required?

No, it is a convention. ngplatform uses it, or the root `zarr_vectors`
attributes, to recognise a store in a folder.

### Which modules are stable?

`zarr_vectors.api` (re-exported from `zarr_vectors`), `zarr_vectors.building`,
`constants`, `exceptions`, `typing` and `headers` are supported. Other modules
can change between releases; `zv.stability("zarr_vectors.<module>")` reports a
module's tier. See the {doc}`API reference <../api/index>`.

## Installation

### Which Python versions are supported?

Python 3.11 or later; the tests run on 3.11 and 3.12. The `[icechunk]` extra
needs Python 3.12 or later. See [Installation](../getting_started/installation.md).

### Does it work on Windows?

The base install does: `numpy`, `numcodecs` and `zarr` all support Windows.
The `[gpu]` extras are Linux only.

### I get an error about `DracoPy`. What do I do?

Install the Draco extra: `pip install "zarr-vectors[draco]"`. You need it to
read a Draco-compressed store.

## Layout and pyramids

### How do I choose the chunk size?

Choose it by access pattern. Smaller chunks suit interactive queries; larger
chunks mean fewer files for batch processing. See
[Choosing a layout](../how_to/choose_chunk_and_bin.md).

### How many resolution levels should I build?

Enough that the coarsest level fits in memory for an overview. See
[Building pyramids](../tutorials/multiscale/building_pyramids.md).

### Can I add a level after building the pyramid?

`build_pyramid` always starts from level 0 and does not overwrite existing
levels. Remove the coarser levels, highest first, then call it again with the
full list of factors; see
[Building pyramids](../tutorials/multiscale/building_pyramids.md).

### Why does my point cloud have a `links/` group?

`build_pyramid` adds links from level 0 to the next level, so level-5
validation then warns `Point cloud but links array exists`. The warning is
expected after building a pyramid.

## Formats and viewing

### Can I convert to and from TRK, SWC, OBJ and other formats?

Yes, with the companion package `zarr-vectors-tools` and its `zvtools`
command. See [Ingesting other formats](../how_to/convert_files.md).

### How do I view a store in Neuroglancer?

Open it in [ngplatform](https://github.com/AllenInstitute/ngplatform), whose
Zarr Vectors plugin reads local stores and URLs. See
[View stores in Neuroglancer](../how_to/view_in_neuroglancer.md) and
[Zarr Vectors and Neuroglancer precomputed](comparisons/neuroglancer_precomputed.md).

## Performance

### My writes are slow. What should I check?

Write the base level in one call: `add_points` and the other writes replace
chunks rather than append. On network file systems use fewer, larger chunks.
For cloud stores, see [Cloud stores](../how_to/cloud_stores.md).

### Validation is slow. Can I run less of it?

Yes. `ds.validate(level=1)` runs only the structural checks. Levels 2 to 5 add
checks in turn, so level 5 is the slowest; see
[Validation and repair](../how_to/validate_and_repair.md).

### How do I report a bug?

Open an issue on [GitHub](https://github.com/AllenInstitute/zarr-vectors-py/issues).
For a problem with a store, include the output of
`print(ds.validate(level=3).summary())`.
