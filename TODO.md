# Known issues

What is still open after the attribute-chunking backlog (validator rank
rule, rechunk bins, pyramids, the lazy writer deadlock, spatial cell
matching) was fixed on this branch. Each entry says what fails, why, and
anything already decided. Line references are against this branch.

In order of priority: 1 returns wrong data silently; 2 and 3 fail loudly
or only in narrow cases; 4 is a recorded constraint; 5 is housekeeping.

---

## 1. The lazy layer's leftovers

**Severity: medium.**

- **The lazy readers read vertices as float32 whatever the level declares**,
  and return the decoded positions, so a float64 level reads back as
  garbage: `ZVLevel.vertices` (`lazy/level.py:262`), the view read path
  (`lazy/views.py:203`), and `_read_polyline` (`lazy/views.py:527`). The
  writer had the same hardcoded dtype but used only row counts, which come
  from the fragment index, so it was harmless there; it now reads at the
  declared dtype.
- **`add_face_attribute` always raises `StoreError`.** `face_attributes/<n>`
  is not a per-chunk array (`_is_per_chunk_array`, `core/arrays.py:885`
  accepts only `vertex_attributes` and `fragment_attributes`), and the
  writer creates it as a group (`lazy/writer.py:430`), so the first cell
  write fails.
- **`append_vertices` writes rank-3 keys into an attribute-chunked level.**
  It assigns chunks with the root's spatial `chunk_shape` and no bin, so
  every key is one component short of the level's arrays.

---

## 2. `build_pyramid` leftovers

**Severity: low.**

- **A pyramid is not atomic across levels.** Each `coarsen_level` step rolls
  itself back, but a failure at level k keeps levels 1..k-1, and the
  finalize pass (`±N`, N ≥ 2) does not run.
- **Registered coarsen strategies** (`zarr-vectors-tools`) over an
  attribute-chunked source are untested. They receive the same kwargs as
  before; if they reach `_write_cross_level_edges` with mismatched ranks they
  now get a `CoarseningError` naming both levels instead of the
  partitioner's shape error.
- **A pre-existing `links/+1` family on the source** cannot be restored by
  the rollback once `write_links(mode="replace")` has rewritten it. It is
  unreachable today: a target level must not exist, and only an old,
  half-removed pyramid leaves a `+1` family behind.

---

## 3. `rechunk` leftovers

**Severity: low.**

- **`by="spatial"`** still writes a pointless extent-1 leading axis with
  `chunk_dims[0] == "spatial"`, and records no labels. `rechunk_spatial`
  (`rechunk/spatial.py:355`) does the spatial job losslessly; decide whether
  `by="spatial"` should delegate to it or be removed.
- **Object ids are renumbered** by a running counter in bin order, so they
  no longer join back to the source, and object attributes are not
  copied. Links are never copied either, so only a point cloud survives a
  non-spatial rechunk intact (see the module docstring of
  `rechunk/spatial.py`).
- **`chunk_attribute_name` vs `chunk_dims[0]`** disagree for an explicit
  `RechunkSpec(by="attribute:x")` without `prefix_dim_name`: the name is `x`,
  the axis is `attribute`.

---

## 4. Device-side decode and GPU-direct reads

**Severity: none today — the next step for the optional GPU extension.**

Device arrays in and out now work (`zarr_vectors.gpu`, the `[gpu]` extra):
readers take `device="cuda"`, writers take cupy arrays, `read_cells` /
`read_neighbourhood` read many cells in one prefetch, and
`runtime_capabilities()` says which of it an install has. But the bytes
still go through the host. A read fetches and decompresses each cell there
and uploads the result once; a write downloads each argument once and
encodes there. `runtime_capabilities()` reports what is missing as
`gpu_encode`, `gpu_io` and `gpu_codecs`.

Why not zarr's own GPU buffers: every per-chunk array is vlen bytes, and
zarr's GPU buffer rejects the object dtype a vlen decode produces
(`zarr/core/buffer/gpu.py`), while the sharding codec asserts the default
host prototype. A configurable prototype would only help the numeric
standalone arrays (object attributes, the object-id table).

There is also no partial-cell read. `read_fragment` advertises a byte-slice
fast path for range fragments but fetches the whole cell and slices it on the
host. The only genuinely sub-cell read is row selection on 1-D standalone
arrays (`Group.read_vlen_elements`).

**If taken up:** decode fragment indices and vlen payloads with cupy
(`encoding/fragments.py` is already array-shaped); nvCOMP for a zstd cell
payload, so decompression moves too; kvikio for uncompressed, unsharded
local cells (the direct reader already opens those files itself,
`core/_batch_reader.py`); then byte-range reads for range fragments, which
only exist for an uncompressed cell.

---

## 5. Housekeeping

**`schema/reference.md` is stale.** Regenerating it with `schema/regen.py`
(linkml 1.11.0) changes about 10k lines before any schema edit, and CI only
byte-checks the JSON Schema, so the `chunk_attribute_values` change was not
carried into it. A plain regeneration belongs in a commit of its own.
