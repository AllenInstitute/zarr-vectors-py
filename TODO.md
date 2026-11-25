# Known issues and next steps

What is left for the next few releases. The previous list (lazy-layer
bugs, `build_pyramid` atomicity, `rechunk` ids, attributes and links,
the GPU fragment decode and device readers, the stale schema reference,
deprecated APIs in tests, lint in CI) was cleared; see the git log.

Each entry says what is missing, why it matters, and anything already
decided. Grouped by when it should land.

---

## Before the next release

**Tag the convention where the stores point.** Format 0.10.0 stamps
every root with the convention's `schema_url` and `spec_url`
(`core/convention.py`, `CONVENTION_CMO`), which point at tag `v0.1.0` of
`github.com/Andrew-Keenlyside/zarr-vectors`. Every store keeps these
URLs, so merge the convention's `schema-fixes` branch and push the tag
before releasing 0.10.0.

**Format 0.10.0 is a cut-over for downstream users.** 0.9.x packages refuse 0.10.0
stores, and 0.10.0 refuses 0.9.x stores until
`building.migrate_metadata_0_10` has run on them. Move downstream users' pinned
zarr-vectors to 0.10.0 first, then migrate their stores.

**Publish zarr-octree to PyPI, then declare it.** zarr-octree is not on
PyPI, so it is not a dependency or an extra: users get a git URL from an
error message, and CI installs it from a pinned commit
(`ZARR_OCTREE_REF` in `.github/workflows/tests.yml`). Once 0.4.0 is on
PyPI:
- add an `octree` extra (`zarr-octree>=0.4,<0.5`) and use it in CI in
  place of the git install;
- change the "pip install git+..." message in `core/group.py`
  (`_octree_codec_class`) to `pip install zarr-vectors[octree]`.

**Bump `ZARR_OCTREE_REF`** whenever zarr-octree's `main` moves. It
points at the commit that renamed the codec `octree_sharding`; after that
merges, point it at `main`'s merge commit.

**Decide on Python 3.11.** zarr-octree needs Python >= 3.12 (as does
zarr >= 3.2), so the 3.11 CI job skips every octree test. Either drop
3.11 (`requires-python >= 3.12`, one CI job) or keep it and accept that
octree sharding is 3.12+ only. Dropping it is simpler.

---

## Next few releases

### Deprecations

**Remove the lazy layer.** `open_zv()`, `ZVLevel`, `ZVView` and
`ZVWriter` are deprecated in favour of `zarr_vectors.open()` /
`Dataset`. Pick the removal release and say it in the warnings. Before
removing:
- `add_node_attribute_sync` / `add_face_attribute_sync` have no
  `Dataset` replacement; the warning points at
  `building.write_vertex_attribute`. Face attributes have none: add a
  `building` entry point for per-face values (they are
  `link_attributes/<name>/0`, as `ZVWriter.add_face_attribute` now
  writes them).
- the tests of the lazy layer go through `tests/_deprecated.py`; they go
  with it.

**`metadata.chunk_scale_factor()`** (renamed `chunk_scale_from_root()`)
and **`write_points(attributes=)`** (now `vertex_attributes=`): remove in
the same release.

Deprecations now use `zarr_vectors.exceptions.ZVDeprecationWarning`,
which the test suite turns into errors, so new uses fail.

**The legacy codec name `zarr_octree_sharding`.** zarr-octree 0.4 writes
`octree_sharding` and still reads the old name, so
`constants.OCTREE_CODEC_NAMES` holds both. Keep reading it for at least
two minor releases after zarr-octree 0.4, and only drop it once
zarr-octree does.

### API decisions

**Per-array codecs** (`TODO` at `core/group.py`, "per-array-type codec
dict"). One session-wide codec list (`_active_codecs`) is used for every
array. Per-array codecs need:
- a dict shape (keys for vertices, fragments, links, attributes,
  object_index);
- `_ensure_array_dir` / `_ensure_chunk_arrays` (`core/arrays.py`) and
  `Group._resolve_codecs` to pick by array name;
- `create_sharded_chunk_arrays`, which creates arrays in one batch with
  one codec list, to group them by codec;
- a rule for `dense_manifests.manifest_codecs`, which already raises the
  zstd level for manifests;
- the public side: `Layout(compression=)` / Schema and the writers'
  `compressor=`.

**`link_fragments` on the device.** `read_cells(fragments=True)` reads
`vertex_fragments` only. The device decoder is generic, so adding
`link_fragments` is a matter of the API (a parameter naming the
fragments array), not kernels.

### Lazy writer

`ZVWriter.append_vertices` extends positions and per-vertex attributes
(`vertex_attributes=`), but not `fragment_attributes`. It goes with the
lazy layer's removal unless that slips.

**Bbox views on binned levels** now read every touched chunk whole and
mask it (the per-bin shortcut was wrong with object ids). Exact, but a
small box decodes whole chunks; `Dataset` reads are the fast path.

### Remaining refusals in `rechunk`

Non-spatial `rechunk` now keeps object ids, object attributes,
groupings and links. It refuses, before touching an existing output:
- a fragment shared by objects in different bins (a vertex lives in one
  bin);
- vertices that no object names;
- levels above 0, and link or fragment attributes.
`rechunk_spatial` is the lossless path for such stores. Lift a refusal
only when a real store needs it.

### Volume meshes (zarr_vectors#10)

Elements (one type or `mixed`), coarse cells with boundary snapping,
surface clustering, carried-down face/cell attributes and
`write_mesh_level` are in. Left:
- **Edges and corners** of a coarse volume round off by up to a bin
  (snapping moves each boundary vertex to its bin's fine-surface
  centroid, which fixes faces, not creases). A per-bin quadric fit
  (dual contouring) would keep sharp features.
- **`rechunk_spatial` refuses levels with link attributes**, so a mesh
  with face attributes cannot be rechunked spatially yet.

### When OME-Zarr RFC 8 settles

RFC 8 is under review (R1), unimplemented and without a JSON Schema.
The root `ome` node (`core/ome.py`) meets its current text, but:
- **Levels are not reachable.** Each `zv:level` entry has only `type`
  and `name`, and level groups carry no `ome` node, so a generic RFC 8
  resolver sees opaque names it cannot follow. Add a `path` to each
  entry (the level group, e.g. `"0"`) once RFC 8 fixes what a node's
  `path` looks like (the draft has both a string and a
  `{"type": "zarr", "path": ...}` object). Additive: stamp it in place
  with `building.stamp_ome_node`, no format bump.
- **`version: "0.6"`** is a guess at the NGFF release RFC 8 lands in
  (the draft says `"0.x"`); change it to the real one.
- **Axes in `scene.coordinateSystems`** are RFC 5 objects; the draft's
  table says strings. Follow whichever the final text keeps.
- Validate the node against RFC 8's schema once one is published.

### GPU

- **Readers still on the host:** `read_chunk_attribute_rows` (its row
  width comes from the fragment index's vertex extent, not the vertex
  row count) and `read_link_arrays` (it filters and assembles link
  records). Each needs its own device path, not a reroute.
- **zstd under `decode="auto"`** still decodes on the host and uploads.
  `decode="device"` decodes on the GPU; it stays opt-in because nvCOMP
  can hang or kill the CUDA context on a damaged block. Stores that opt
  into zstd checksums (`{"name": "zstd", "configuration": {"checksum":
  true}}`) now have them verified on the device after decode, which
  catches wrong bytes but cannot prevent a hang.
- **nvCOMP 6 rejects empty `out=` buffers** in `decode_zstd`
  ("NULL CUDA buffer not accepted"). Skip the call for zero-length
  outputs.
- The per-cell header check and the XXH64 checksum run one thread per
  cell; make them per-fragment / per-block if cells get very large.

---

## Design notes (no work planned)

**Byte-range reads of compressed cells.** `read_fragment` reads an
uncompressed cell by range, but a compressed cell whole. A zstd frame has
no block index and its blocks refer back to earlier data, so part of a
cell cannot be decoded without the data before it. Range reads would
need independent frames per cell (or zstd's seekable format with a seek
table), i.e. a new codec that other Zarr tools could not read. Stores
that need per-object range reads can leave `vertices` uncompressed.
Revisit only if large cells read one object at a time become common
(for example a viewer streaming single objects from big shards).

**The `bytes` (variable-length) data type has no Zarr v3 spec yet.**
zarr-python warns that its representation may change. Every per-chunk
array uses it. Follow the zarr-extensions data-types registry; nothing
to do until a spec lands.

**The ruff version.** CI pins `ruff==0.17.5`; the dev extra says
`ruff>=0.4` and `uv.lock` has 0.15.12. Bump the lockfile to match
when convenient.
