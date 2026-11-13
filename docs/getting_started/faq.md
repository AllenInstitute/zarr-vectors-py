# Frequently asked questions

## General

### What is the relationship between zarr-vectors and Zarr v3?

`zarr-vectors` is built on Zarr v3. A Zarr Vectors store is a valid Zarr v3
store, and the standard `zarr` Python library can open every array in it.
`zarr-vectors` adds conventions on top: the directory layout, the metadata in
each `zarr.json`, the fragment index arrays and OME-Zarr multiscale metadata.
The root `zarr.json` holds the store-level fields under
`attributes.zarr_vectors` (`zv_version`, `bounds`, `chunk_shape`,
`base_bin_shape`, `geometry_types`, …) and the per-level transforms under
`attributes.multiscales`; each level's own `zarr.json` holds
`attributes.zarr_vectors_level`. The format is Zarr v3 only.

### What is the relationship between zarr-vectors and OME-Zarr?

Zarr Vectors uses the `multiscales` JSON block from the OME-Zarr NGFF
specification, so an OME-Zarr-aware viewer can find its resolution pyramid.
It is not a subset of OME-Zarr: OME-Zarr is mainly for dense image volumes,
Zarr Vectors for sparse vector geometry.
[Zarr Vectors and OME-Zarr](../spec/comparisons/ome_zarr.md) lists which
fields are shared and which are specific to Zarr Vectors.

### Why does the store have a `.zarrvectors` extension? Is it required?

The `.zarrvectors` extension is a convention; `zarr-vectors` does not require
it, and you can name a store anything. The extension identifies a store at a
glance, and ngplatform uses it, or the root `zarr_vectors` attributes, to
recognise a store in a local folder.

### Can I open a Zarr Vectors store with plain `zarr.open()`?

Yes. The arrays are standard Zarr v3, so `zarr.open_group(path)` works and the
group's `attrs` hold the `zarr_vectors` and `multiscales` blocks described
above. You then decode the fragment indices, object manifests, the links under
`links/<delta>/<offsets>/` and the level transforms yourself.

To read data, use `zv.open(path).read()`, or
`zv.open(path).select(bbox=...).read()` for a region.

---

## Installation

### Which Python versions are supported?

Python 3.11 and 3.12. The package requires Python 3.11 or later, and the tests
run on 3.11 and 3.12. The `[icechunk]` extra needs Python 3.12 or later.

### Does zarr-vectors work on Windows?

The base install needs only `numpy`, `numcodecs` and `zarr`, all of which
support Windows. The optional extras listed in
[Installation](installation.md) have their own platform requirements; the
`[gpu]` extras are Linux only.

### I get an error about `DracoPy` — what do I do?

Install the Draco extra: `pip install "zarr-vectors[draco]"`. When neither
`DracoPy` nor a `draco_encoder` command-line tool is available, the library
raises `DracoError: Neither DracoPy nor draco_encoder CLI found`. If DracoPy has
no binary wheel for your platform, build it from source.

---

## Chunks and bins

### How do I choose the cell size?

A **chunk** (a *cell* on the `zarr_vectors.api` surface) is the I/O unit, one
file on disk, so choose it by access pattern. For interactive spatial queries,
such as a viewer fetching what is on screen, smaller cells load less data per
request. For batch processing that reads a whole dataset, larger cells mean
fewer files to open.

`zv.Layout(cells=5)` cuts the bounds into five cells per axis and lets the
extent set the size; `zv.Layout(cell_size=(200.0, 200.0, 200.0))` sets the size
directly, which you need when the grid must line up with an image volume.
`cells=` also takes one count per axis, for example `cells=(5, 5, 2)`.

See [Choosing a layout](../how_to/choose_chunk_and_bin.md) for
worked examples and heuristics.

### What is the relationship between cells and bins — `chunk_shape` and `bin_shape`?

Both come from `Layout`. `bin_shape` must evenly divide
`chunk_shape` in every dimension, which is why `zv.Layout` divides rather than
asking for an absolute size: `subcells=n` puts `n` bins per cell *per axis*
(default 4, so 64 bins per cell in 3-D). `chunk_shape` controls the on-disk
file layout; `bin_shape` controls spatial query granularity and grows with the
coarsening factor at each coarser level.

By default every level keeps the base `chunk_shape`; passing
`chunk_scale_factors=` to `build_pyramid` gives each level its own, which
`ds.resolution(scale=...)` needs to tell the levels apart. Each level reports
both:

```python
level = ds.level(0)
level.scale        # the chunk shape, e.g. (200.0, 200.0, 200.0)
level.resolution   # the bin shape,   e.g.  (50.0,  50.0,  50.0)
```

### What happens if I pass no `Layout` at all?

`Layout()` has `cells="auto"`. Without a size hint that is one cell for the
whole volume; with `Schema(expected=zv.SizeHints(n_vertices=...))` it aims for
about 64 MB per cell. The bin shape defaults to a quarter of the cell per axis,
so a store with 1000 µm bounds and neither layout nor hint reports
`scale == (1000.0, 1000.0, 1000.0)` and `resolution == (250.0, 250.0, 250.0)`.

`subcells=1` gives one bin per cell (`bin_shape == chunk_shape`), so spatial
queries work at whole-cell granularity.

---

## Multi-resolution

### How many resolution levels should I build?

Build enough levels that the coarsest one fits in memory for overview
rendering. Levels come from
`Dataset.build_pyramid(factors=[...])`, where `factors[i]` is a
`(coarsen_factor, sparsity_factor)` pair applied to level `i` to produce level
`i+1`:

```python
ds.build_pyramid(factors=[(2.0, 1.0), (2.0, 1.0)], chunk_scale_factors=[2, 2])
```

`coarsen_factor` is a per-level ratio against the level below, so factors
compound: `[(2, 1), (2, 1), (2, 1)]` bins at 2×, 4× and 8× the root bin. `1.0`
is the identity for either factor.

Coarsening replaces the vertices falling in a bin with a single metavertex, so
a coarse level's size is the number of *occupied* bins rather than a fixed
fraction of the level below. Doubling the bin per axis divides a sparse level
by up to 8 in 3-D; a dense one saturates at the bin count instead. In the
quickstart's store, 100 000 points become 1 000 and then 125 metavertices —
10³ and 5³ bins, every one of them occupied.

### How does object sparsity work?

`sparsity_factor`, the second element of each `factors` pair, is a divisor of
at least 1: `2.0` keeps half the objects at that level and leaves the dropped
ones as empty manifest slots. The level stores its reciprocal as
`object_sparsity` (`sparsity_factor=2.0` gives `object_sparsity=0.5`), and
values below `1.0` are rejected: `factors=[(2.0, 0.5)]` raises
`MetadataError: object_sparsity must be in (0, 1], got 2.0`. Survivors keep
their object IDs across levels.

### Does the coarsening factor have to be isotropic (same in all dimensions)?

The bin shape need not be: set it anisotropically at level 0 with
`zv.Layout(cells=(5, 5, 2))` or `cell_size=`, and every level keeps the
anisotropy. The coarsening ratio of each level is one number applied to every
axis, so `build_pyramid` cannot take a per-axis ratio such as `(1, 2, 2)`;
passing a tuple raises
`TypeError: float() argument must be a string or a real number, not 'tuple'`.
The chunk shape can grow per axis: `chunk_scale_factors=[(2, 2, 1)]`.

### Can I add a resolution level after writing the base level?

Yes, by rebuilding the coarser levels. `ds.build_pyramid(factors=[...])`
always starts from level 0 and will not write into a level that exists, so
remove the coarser levels first, highest first, and call it again with the
full factor list:

```python
import zarr_vectors as zv
from zarr_vectors import building

ds = zv.open("scan.zarrvectors", mode="r+")
for index in sorted(ds.levels, reverse=True)[:-1]:
    building.remove_resolution_level(ds.store, index)
ds = zv.open(ds.url, mode="r+")
ds.build_pyramid(factors=[(2.0, 1.0)] * 3, chunk_scale_factors=[2, 2, 2])
```

The base level's geometry is not changed, but the pyramid adds `links/+1/`
(the edges to the parent level) to it. That is why a point cloud has a
`links/` group after `build_pyramid`, and why level-5 validation then warns
`Point cloud but links array exists`.

`zarr_vectors.building` also has `create_resolution_level`,
`get_resolution_level` and `list_resolution_levels` for builder code that
manages levels by hand. The supported surfaces have no call that coarsens a
single level on its own.

---

## Formats and interoperability

### Can I convert a Zarr Vectors store back to TRK / SWC / OBJ?

Yes, with the converters and the `zvtools` command-line tool in the companion
package `zarr-vectors-tools`.

### How do I visualise a Zarr Vectors store in Neuroglancer?

Open it in [ngplatform](https://github.com/AllenInstitute/ngplatform), a
browser workspace around Neuroglancer whose Zarr Vectors plugin reads
stores from your computer or a URL. See
[View stores in Neuroglancer](../tutorials/neuroglancer/overview.md).

### Is Zarr Vectors compatible with the Neuroglancer precomputed format?

They are separate formats with shared goals, such as spatial chunking and
multiscale levels. `zarr-vectors-tools` exports skeleton and mesh stores as
Neuroglancer precomputed layers; see
[Precomputed export](../tutorials/neuroglancer/precomputed_export.md). See
[Zarr Vectors and Neuroglancer precomputed](../spec/comparisons/neuroglancer_precomputed.md)
for a comparison of the formats.

---

## Performance

### My writes are slow — what should I check?

- Write the base level once. `add_points` and its siblings are one-shot
  writes, not appends: a second call replaces the chunks its data falls in.
  Assemble the full array first rather than writing in a loop.
- On networked file systems (NFS, SMB), use fewer, larger cells (a smaller
  `Layout(cells=...)`) to create fewer files.
- For cloud writes, install `zarr-vectors[obstore]` (or `[fsspec]`) and pass
  the URL to `zv.create` or `zv.open`, for example
  `zv.open("s3://bucket/scan.zarrvectors")`. The backend comes from the URL
  scheme, or from `StorageOptions(backend=...)` to force one.
  `Layout(pack=...)` packs several cells into one object; it is on by default
  for non-local stores. See [Cloud stores](../tutorials/io/cloud_stores.md).
- Build the pyramid as a separate pass over the finished base level, and thin
  the coarse levels with a `sparsity_factor` above 1.0 if they are the cost.

### Validation is slow on large stores. Can I run only level 1?

Yes. `ds.validate(level=1)` runs only the structural check (which files and
arrays are present). Levels are cumulative: 2 to 5 add metadata, consistency,
conformance and multiresolution checks in turn, so level 5 is the most
thorough and the slowest.

---

## Development and contribution

### How do I report a bug?

Open an issue on the
[GitHub repository](https://github.com/AllenInstitute/zarr-vectors-py/issues).
If the issue concerns a store, include the output of
`print(ds.validate(level=3).summary())`.

### How do I propose a change to the specification?

See [Spec change process](../spec/contributing/spec_change_process.md).
Spec changes require an RFC-style discussion issue before a pull request.

### Is zarr-vectors stable?

Partly. Two module surfaces are **supported** and carry a compatibility
promise: `zarr_vectors.api` (re-exported from the top-level package) for using
data, and `zarr_vectors.building` for making stores, together with
`constants`, `exceptions`, `typing` and `headers`. `core`, `encoding`,
`spatial`, `lazy`, `ops`, `sharding`, `multiresolution` and `rechunk` are
**internal** and change without notice. `types`, `validate`, `composite`,
`gpu` and `octree_chunks` are **undecided**: they carry no promise, so treat
them as internal.

The package reports each module's tier, so a lint rule or an import audit can
check a call site:

```python
import zarr_vectors as zv

zv.stability("zarr_vectors.api")          # 'supported'
zv.stability("zarr_vectors.core.arrays")  # 'internal'
zv.stability("zarr_vectors.validate")     # 'undecided'
```

See the {doc}`API reference <../api/index>` for what each surface covers.

Three version numbers move independently: `zv.__version__` (the package),
`zv.__api_version__` (the API surface) and `ds.format_version` (the on-disk
format of one store). `zv.require_api(...)` and `zv.require_format(ds, ...)`
check the last two.
