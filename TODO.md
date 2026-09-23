# Known issues

Found while fixing build-time sharding and the attribute-chunked grid
derivation on this branch, and deliberately left out of those changes. Each
entry says what fails, why, and what has already been decided — the research
is done, the work is not.

Line references are against `cf4c2f1`.

---

## 1. `build_pyramid` leftovers after the attribute-chunked fix

**Severity: low.** Pyramids over attribute-chunked levels, the depth-0
gate, per-step rollback and the empty-level `chunk_shape` were fixed on
`todo-backlog`; these were left out of scope.

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

## 2. Lazy-layer leftovers after the deadlock fix

**Severity: medium.** The `*_sync` deadlock and the `_prefetch_cache` race
were fixed on `todo-backlog`; these were found on the way and left out.

- **The lazy readers read vertices as float32 whatever the level declares**,
  and return the decoded positions, so a float64 level reads back as garbage:
  `ZVLevel.vertices` (`lazy/level.py:262`), the view read path
  (`lazy/views.py:203`), and `_read_polyline` (`lazy/views.py:527`). The
  writer had the same hardcoded dtype but used only row counts, which come
  from the fragment index, so it was harmless there.
- **`add_face_attribute` always raises `StoreError`.** `face_attributes/<n>`
  is not a per-chunk array (`_is_per_chunk_array`, `core/arrays.py:880`), and
  the writer pre-creates it as a group (`lazy/writer.py`,
  `_write_per_face_attribute`), so the first cell write fails.
- **`append_vertices` writes rank-3 keys into an attribute-chunked level.** It
  assigns chunks with the root's spatial `chunk_shape` and no bin, so every
  key is one component short of the level's arrays.

---

## 3. `rechunk` leftovers after the bin fix

**Severity: low.** The non-dense bins, per-object reads, group rewrite and
dtype were fixed on `todo-backlog`; these were left out of scope.

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

## 6. No device-side read path

**Severity: none today — a recorded constraint, not a request.** Asked for by
BRIDGE (its D8, GPU-direct reads), which has dropped it from its own backlog
because nothing it can do reaches past this.

Every read ends in `np.frombuffer` on host memory, so a device buffer handed up
by zarr (`zarr.config.enable_gpu()`, kvikio) is copied to the host on its first
contact with this package. The batched reader and writer already pass zarr a
buffer prototype, but it is fixed at import to the host-side
`default_buffer_prototype()` (`core/_batch_reader.py:61`,
`core/_batch_writer.py:90`).

There is also no partial-cell read. `read_fragment` advertises a byte-slice
fast path for range fragments but fetches the whole cell and slices it on the
host. The only genuinely sub-cell read is row selection on 1-D standalone
arrays (`Group.read_vlen_elements`).

**If taken up:** make the prototype configurable, then a read path that honours
it instead of calling `np.frombuffer`, then byte-range plumbing so a range
fragment is fetched without its cell. The last only works for an uncompressed
cell — under a compressor there is no byte range to ask for — so it is a
codec-dependent fast path, not a general one.
