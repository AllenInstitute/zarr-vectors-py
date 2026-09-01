# Frequently asked questions

## General

### What is the relationship between zarr-vectors and Zarr v3?

`zarr-vectors` is built *on top of* Zarr v3. A ZVF store is a valid Zarr v3
store: all arrays inside it can be opened with the standard `zarr` Python
library. `zarr-vectors` adds conventions on top of Zarr — the directory
layout, the metadata blocks carried in each `zarr.json`, the fragment index
arrays, and the OME-Zarr-compatible multiscale metadata. The root
`zarr.json` holds the store-level fields under `attributes.zarr_vectors`
(`zv_version`, `bounds`, `chunk_shape`, `base_bin_shape`, `geometry_types`, …)
and the per-level scale/translation transforms under `attributes.multiscales`;
each level's own `zarr.json` holds `attributes.zarr_vectors_level`. There is no
`.zattrs` anywhere in a ZVF store — that is the Zarr **v2** spelling, and ZVF
is v3-only.

You could read the raw position arrays directly with `zarr.open()`, but you
would not get the spatial indexing, object model, or multi-resolution support
that `zarr-vectors` provides.

### What is the relationship between zarr-vectors and OME-Zarr?

ZVF borrows the `multiscales` JSON block from the OME-Zarr NGFF
specification so that resolution pyramids are discoverable by any OME-Zarr-
aware viewer. ZVF is not a strict subset of OME-Zarr: the two formats target
different data (OME-Zarr is primarily for dense image volumes; ZVF is for
sparse vector geometry). The comparison page
[ZVF and OME-Zarr](../spec/comparisons/ome_zarr.md) documents exactly which
fields are shared and which are ZVF-specific extensions.

### Why does the store have a `.zarrvectors` extension? Is it required?

The `.zarrvectors` extension is a convention, not a requirement enforced by
the file system or the library. You can name your store anything. The
extension helps tools (and humans) identify ZVF stores at a glance and is
used by `zv-ngtools` to auto-detect the store type when loading layers.

### Can I open a ZVF store with plain `zarr.open()`?

Yes. The underlying arrays are standard Zarr v3, so `zarr.open_group(path)`
works and the group's `attrs` are the `zarr_vectors` and `multiscales` blocks
described above. What you will not get is interpretation: fragment indices,
the object manifests, the link families under `links/<delta>/<offsets>/` and
the level transforms all have to be decoded by hand.

For reading data, use the supported `zarr_vectors.api` surface instead —
`zv.open(path).read()`, or `zv.open(path).select(bbox=...).read()` for a
region. (Earlier versions of this page pointed at the `zarr_vectors.types.*`
read functions. Those still work, but the module is *undecided* rather than
supported; see [Is zarr-vectors stable?](#is-zarr-vectors-stable) below.)

---

## Installation

### Which Python versions are supported?

Python 3.11 and 3.12 — that is what `requires-python = ">=3.11"` allows, what
the classifiers claim, and what CI runs. Python 3.10 and earlier are not
supported: the single-array layout uses Zarr v3's vlen-bytes codec, so
`zarr>=3.0` is required, and with it Python 3.11+.

### Does zarr-vectors work on Windows?

Yes. All core functionality is cross-platform, and the base install needs only
`numpy`, `numcodecs` and `zarr`. The optional extras — `[draco]` for
Draco-compressed meshes, `[obstore]` (aliased as `[cloud]`) and `[fsspec]` for
cloud backends, `[icechunk]` for the transactional store — have their own
platform requirements; consult their documentation if you encounter issues.
Format converters (LAS, PLY, TRK, SWC, OBJ, …) live in the companion package
`zarr-vectors-tools` and carry their own dependencies.

### I get an error about `DracoPy` — what do I do?

Install the Draco extra: `pip install "zarr-vectors[draco]"`. When neither
`DracoPy` nor a `draco_encoder` CLI is available the library raises
`DracoError: Neither DracoPy nor draco_encoder CLI found`. DracoPy has binary
wheels for most platforms; if a wheel is not available for yours, you may need
to build it from source.

---

## Chunks and bins

### How do I choose the cell size?

A **chunk** (a *cell*, on the `zarr_vectors.api` surface) is the I/O unit —
one file on disk — and the primary consideration is I/O pattern. For
interactive spatial queries (e.g. viewport-driven fetching in a visualiser),
smaller cells reduce the amount of data loaded per request. For batch
processing (e.g. reading an entire dataset sequentially), larger cells reduce
overhead from opening many files. A common starting point for 3-D biological
data is 200–500 physical units per axis.

Say it either way round. `zv.Layout(cells=5)` cuts the bounds into about five
cells per axis and lets the extent decide the size; `zv.Layout(cell_size=(200.0,
200.0, 200.0))` fixes the size directly, which is what a pipeline whose grid
must line up with an image volume needs. `cells=` also takes one count per
axis, e.g. `cells=(5, 5, 2)`.

See [Choosing a layout](../how_to/choose_chunk_and_bin.md) for
worked examples and heuristics.

### What is the relationship between cells and bins — `chunk_shape` and `bin_shape`?

Nothing about the format changed; the two are simply derived from `Layout`
now instead of being passed to every call. `bin_shape` must evenly divide
`chunk_shape` in every dimension, which is why `zv.Layout` divides rather than
asking for an absolute size: `subcells=n` puts `n` bins per cell *per axis*
(default 4, so 64 bins per cell in 3-D). `chunk_shape` controls the on-disk
file layout; `bin_shape` controls spatial query granularity and grows with the
coarsening factor at each coarser level.

By default every level inherits the root `chunk_shape`; passing
`chunk_scale_factors=` to `build_pyramid` opts out and gives each level its own,
which is what you want if `ds.resolution(scale=...)` is to tell the levels
apart. Both are readable per level, under their data-shaped names:

```python
level = ds.level(0)
level.scale        # the chunk shape, e.g. (200.0, 200.0, 200.0)
level.resolution   # the bin shape,   e.g.  (50.0,  50.0,  50.0)
```

### What happens if I pass no `Layout` at all?

`Layout()` means `cells="auto"`, which is **one cell for the whole volume** —
the honest default, and almost never what you want. Pass `cells=`. The bin
shape still defaults to a quarter of the cell per axis, so a store created with
bounds of 1000 µm and no layout reports `scale == (1000.0, 1000.0, 1000.0)` and
`resolution == (250.0, 250.0, 250.0)`.

`subcells=1` is the other end: one bin per cell, i.e. `bin_shape ==
chunk_shape`. Spatial queries then return data at whole-cell granularity. This
is the backward-compatible mode for stores that do not require sub-chunk
spatial indexing.

---

## Multi-resolution

### How many resolution levels should I build?

A common choice is to build enough levels so that the coarsest level fits
comfortably in memory for overview rendering. Levels come from
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

`sparsity_factor` — the second element of each `factors` pair — is a divisor
`≥ 1`: `2.0` keeps half the objects at that level and leaves the dropped ones
as empty manifest slots. It is stored on the level as its reciprocal,
`object_sparsity` (so `sparsity_factor=2.0` → `object_sparsity=0.5`), and
values below `1.0` are rejected — `factors=[(2.0, 0.5)]` raises
`MetadataError: object_sparsity must be in (0, 1], got 2.0`. Survivors keep
their object IDs across levels.

### Does the coarsening factor have to be isotropic (same in all dimensions)?

The *bin shape* need not be: set it anisotropically at level 0 with
`zv.Layout(cells=(5, 5, 2))` or `cell_size=`, and every level inherits the
anisotropy. The per-level *coarsening ratio* is a single scalar applied to
every axis, though — `factors=[(2.0, 1.0)]` — so a per-axis ratio like
`(1, 2, 2)` is not expressible through `build_pyramid` (passing a tuple raises
`TypeError: float() argument must be a string or a real number, not 'tuple'`).
Per-axis growth of the chunk grid *is* available:
`chunk_scale_factors=[(2, 2, 1)]`.

### Can I add a resolution level after writing the base level?

Yes. `ds.build_pyramid(factors=[...])` always starts from level 0, so to add a
third level to a store that already has two, call it again with the full factor
list — it rewrites the coarser levels rather than appending to them. The base level's geometry is never re-derived, but the pyramid does add
`links/+1/` to it (the cross-level parent edges), which is why a point cloud
grows a `links/` group after `build_pyramid` and why level-5 validation then
warns `Point cloud but links array exists`.

Builder code that manages levels by hand has
`create_resolution_level`, `get_resolution_level`, `list_resolution_levels` and
`remove_resolution_level` in `zarr_vectors.building`. Coarsening a *single*
level in isolation has no supported spelling: `coarsen_level` lives in
`zarr_vectors.multiresolution.coarsen`, which is internal, and reaching for it
means reaching past the contract — a gap to report rather than a reason to
import from an internal module. `build_pyramid` covers the common case.

---

## Formats and interoperability

### Can I convert a ZVF store back to TRK / SWC / OBJ?

Yes, using the converters and CLI in the companion package
**`zarr-vectors-tools`**.

### How do I visualise a ZVF store in Neuroglancer?

Use [`zv-ngtools`](https://github.com/BRIDGE-Neuroscience/zv-ngtools), a
fork of `ngtools` that adds a ZVF layer type. It can serve a local
`.zarrvectors` store to a Neuroglancer instance running in your browser.
See [Neuroglancer integration](../tutorials/neuroglancer/overview.md).

### Is ZVF compatible with the Neuroglancer precomputed format?

ZVF and Neuroglancer precomputed are distinct formats that share some
design goals (spatial chunking, multiscale support). `zv-ngtools` includes
a precomputed export tool that converts a ZVF store to the Neuroglancer
precomputed annotation or skeleton format for static hosting.
See [Format comparisons](../spec/comparisons/neuroglancer_precomputed.md)
for a detailed comparison.

---

## Performance

### My writes are slow — what should I check?

- Let `zv.Layout` derive the bin shape. It divides the cell (`subcells`), so
  the bin shape divides the chunk shape by construction; a store assembled by
  hand through `zarr_vectors.building` can break that rule, and level-2
  validation will say so.
- Write the base level once. `add_points` and its siblings are one-shot
  writes, not appends — a second call re-derives the grid from its own batch
  and overwrites what was there — so assemble the full array first rather than
  looping.
- On networked file systems (NFS, SMB), use fewer, larger cells
  (a smaller `Layout(cells=...)`) to reduce the number of file creates.
- For cloud writes, install `zarr-vectors[obstore]` (or `[fsspec]` for the
  fsspec backends) and pass the URL directly to `zv.create` / `zv.open`:
  `zv.open("s3://bucket/scan.zarrvectors")`. The backend is resolved from the
  URL scheme, or from `StorageOptions(backend=...)` when you want to force it.
  `Layout(pack=...)` packs several cells into one object, which defaults on for
  non-local stores because one object per cell is expensive there.
  See [Cloud stores tutorial](../tutorials/io/cloud_stores.md).
- Build the pyramid as its own pass over the finished base level (there is no
  inline mode), and thin the coarse levels with a `sparsity_factor` above 1.0
  if they are what is costing you.

### Validation is slow on large stores. Can I run only level 1?

Yes. `validate("scan.zarrvectors", level=1)` runs only the structural check
(file/array presence). Levels are cumulative — each adds metadata, consistency,
conformance and multiresolution checks on top — so level 5 is the most thorough
and the slowest.

---

## Development and contribution

### How do I report a bug?

Open an issue on the
[GitHub repository](https://github.com/BRIDGE-Neuroscience/zarr-vectors-py/issues).
Include the output of `zarr_vectors.validate.validate(store, level=3)` if
the issue is store-related.

### How do I propose a change to the specification?

See [Spec change process](../spec/contributing/spec_change_process.md).
Spec changes require an RFC-style discussion issue before a pull request.

### Is zarr-vectors stable?

Partly, and the package will tell you which part. Two module surfaces are
**supported** and carry a compatibility promise: `zarr_vectors.api` (also
re-exported from the top-level package) for using data, and
`zarr_vectors.building` for making stores, alongside `constants`, `exceptions`,
`typing` and `headers`. `core`, `encoding`, `spatial`, `lazy`, `ops`,
`sharding`, `multiresolution` and `rechunk` are **internal** and change without
notice. `types`, `validate` and `composite` are **undecided** — neither
promised nor disowned, and best treated as internal until that changes.

The tiers are a manifest in `zarr_vectors/_stability.py`, not prose, so a lint
rule or an import audit can check a call site:

```python
import zarr_vectors as zv

zv.stability("zarr_vectors.api")          # 'supported'
zv.stability("zarr_vectors.core.arrays")  # 'internal'
zv.stability("zarr_vectors.validate")     # 'undecided'
```

See the {doc}`API reference <../api/index>` for what each surface covers.

Stores written with the current version will be readable by future versions;
no store format migration is planned for the 0.x series. The three version
numbers move independently — `zv.__version__` (the package),
`zv.__api_version__` (the API surface) and `ds.format_version` (the on-disk
format of one store) — and `zv.require_api(...)` / `zv.require_format(ds, ...)`
assert against the last two.
