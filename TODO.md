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

## 2. `ZVWriter.add_attribute_sync` deadlocks on an attribute-chunked store

**Severity: high.** A permanent hang — no traceback, no timeout, indistinguishable
from slowness.

**Decided:** fix it properly rather than failing fast, despite the lazy layer
being deprecated.

**Root cause, verified.** Thread-pool exhaustion of zarr's shared loop's default
executor:

1. `add_attribute_sync` (`lazy/writer.py:622-629`) runs the outer coroutine on
   zarr's **process-global** loop via `sync()`.
2. `_write_per_vertex_attribute` (`writer.py:202-299`) does `asyncio.gather` over
   one task per chunk key, unbounded (`writer.py:287`). Each task's first act is
   `asyncio.to_thread`, which resolves to **that loop's** default executor —
   `min(32, cpu_count+4)` = 16 threads on this machine.
3. `read_chunk_vertices` opens a `batched_reads` block **per call** for a single
   key (`core/arrays.py:4089-4092`) — the degenerate case `_maybe_batched_reads`'
   own docstring says it exists to avoid.
4. `batched_reads` entry calls `sync()` (`core/group.py:556-561` →
   `_lookup_node` → zarr's `_sync` → `wait(timeout=None)`), and that inner
   coroutine's `LocalStore.get` needs a thread from **the same** executor, which
   has none free.

Deadlocks at **≥16 chunk keys**; 15 or fewer completes. Attribute chunking
triggers it purely by cardinality — populated cells ≈ spatial cells × bins, so
the 8-cell store `tests/test_lazy_writer.py:24-33` uses becomes 24+ with 3 bins,
which is why the existing test passes. The async `add_attribute` does **not**
deadlock: the caller's loop and zarr's loop have different executors.

**Candidate fixes, in preference order:**
- Hoist one `batched_reads` over all keys before the gather.
  `_maybe_batched_reads` is a no-op when `_prefetch_cache` is set
  (`arrays.py:189-191`), so inner calls stop opening their own blocks. Check
  whether the *write* half (`writer.py:259-263`) re-opens the cycle.
- Run the `*_sync` wrappers on a private loop so the two executors differ — the
  same asymmetry that already makes the async form safe. Smallest change, fixes
  every `*_sync` method at once.
- Bound the gather with a semaphore. Machine-dependent threshold; a band-aid.

**Two more bugs in the same function:**
- **`_prefetch_cache` race.** It is plain instance state on the shared `Group`
  (`group.py:152,216,235`). The nesting check at `:525-526` and the set at `:557`
  are not atomic, and the `finally` at `:572` lets one thread null it while
  another is inside its block. Below 16 keys the fan-out can spuriously raise
  `StoreError("batched_reads() does not support nesting")`.
- **Hardcoded dtype.** `writer.py:237` passes `np.float32` to
  `read_chunk_vertices`, overriding the store's declared dtype.
  `arrays.py:4070-4074` documents why that is dangerous: a float64 cell read as
  float32 decodes to garbage at twice the row count, silently.

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

## 5. `Level.grid` reports the spatial grid on an attribute-chunked level

**Severity: low.** `Level.grid` (`api/level.py:270-284`) calls
`Grid.plan(bounds, cell_size=self.scale)`, which computes a purely spatial grid
(`api/grid.py:160-162`). On an attribute-chunked level `grid.shape` is rank 3
while the arrays are rank 4, and `grid.cells` under-reports by a factor of K.

`Grid.plan` itself is a pre-write prediction and is right to be spatial. The
question is only what `Level.grid` should return for a store that exists. Either
carry the leading axis, or document that `grid` is the spatial grid and that
`cell_of` / `cells_in` / `holds` are spatial predicates.

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
