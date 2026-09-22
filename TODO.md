# Known issues

Found while fixing build-time sharding and the attribute-chunked grid
derivation on this branch, and deliberately left out of those changes. Each
entry says what fails, why, and what has already been decided — the research
is done, the work is not.

Line references are against `cf4c2f1`.

---

## 1. `build_pyramid` fails on an attribute-chunked level

**Severity: high.** Public API (`Dataset.build_pyramid`), documented feature
combination, fails outright and leaves a half-built store behind.

```
ChunkingError: partition_arrays_by_offset: src_chunks has shape (M, 4);
expected (M, 3)
```
from `zarr_vectors/spatial/boundary.py:736-741`.

Three things disagree:

| | rank | source |
|---|---|---|
| `sid_ndim` passed to the link writer | 3 | `root_meta.sid_ndim`, set once at `multiresolution/coarsen.py:195` |
| `src_chunks` (`fine_cc`, `coarsen.py:996-1010`) | 4 | the fine level's real on-disk keys, via `_reconstruct_chunk_assignments` → `chunk_local_to_global_offsets` |
| `trg_chunks` (`coarse_cc`, `coarsen.py:1012`) | 3 | `assign_chunks(meta_positions, target_chunk_shape)` at `coarsen.py:407`, purely spatial |

The partitioner's single-rank contract is load-bearing, not incidental: the
group-by key is `concatenate([src_chunks, offsets])` and it splits back with
`head[:sid_ndim]` / `head[sid_ndim:]` (`boundary.py:758,775-776`), then
`format_offsets` (`core/paths.py:78`) encodes the offset tuple into the on-disk
array path and `parse_offsets` validates arity against `sid_ndim` on read.

**Decided:** make cross-level links work, rather than failing fast or skipping
them. That makes the coarse level itself attribute-chunked with the same bins,
so both endpoints are rank `sid_ndim + 1`.

**Open design question, must be settled first.** Coarsening currently groups per
object. A polyline with mixed attribute values *is* split across bins by design
(`docs/spec/chunking/attribute_chunking.md`), so a per-object metavertex can
draw vertices from several bins. The bin has to become part of the grouping key
so a metavertex never mixes bins — otherwise a categorical query at a coarse
level returns the wrong set, which defeats the purpose of attribute chunking.
Confirm where the grouping key is formed in `_per_object_coarsen`
(`coarsen.py:195-560`) before committing to this.

**Precedent to reuse:** `_pad_scale` (`core/arrays.py:2727-2745`) already states
the rule — the bin axis indexes bins, not space, so it never rescales across
levels; pad with 1s at the front. It is currently inert on this path only
because the coarsener passes the spatial `sid_ndim`. Passing the widened rank
should make the partitioner arithmetic work unchanged.

**Also fix while here:** `coarsen.py:428-448` and `:701-718` build the coarse
`LevelMetadata` without `chunk_dims` / `chunk_attribute_name` /
`chunk_attribute_values`. They need propagating.

**Related, worth folding in:**
- **Partial write with no rollback.** By the time it raises, the coarse level
  group, `vertices`, `object_index`, every coarse vertex chunk,
  `object_attributes` and two `CAP_*` root tokens are already written
  (`coarsen.py:449-546`). There is no cleanup anywhere in the module, and
  `create_resolution_level` uses `require_group`, so a retry silently merges
  into the half-built level.
- **`cross_level_depth=0` does not avoid it.** `build_pyramid` never forwards
  `cross_level_depth` to `coarsen_level` (`coarsen.py:1126-1138`), so the inline
  ±1 emission still runs. Only `cross_level_storage="none"` escapes;
  `"implicit"` fails at the same line as `"explicit"`.

**No test covers a pyramid over an attribute-chunked store.** The sets of tests
touching `build_pyramid` and `chunk_by_attribute` are disjoint.

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

## 3. `rechunk` produces non-dense bins that readers resolve wrongly

**Severity: high — silent wrong data.** `rechunk` / `rechunk_by_attribute` /
`RechunkSpec` are all on the supported surface (`building/__init__.py:200`).

`rechunk/engine.py` carries **four** notions of bin count:

| | site | value |
|---|---|---|
| bins actually used | `engine.py:134-137` | `sorted(set(obj_to_bin.values()))` |
| grid extent | `engine.py:211` | `max(unique_bins) + 1` |
| `chunk_attribute_values` | `engine.py:183-185` | compacted `[bin_to_value[b] for b in sorted(...)]`, length `len(unique_bins)` |
| chunk key prefixes | `engine.py:230,268,287` | the raw `bin_idx` |

Readers resolve a value positionally through the compacted list —
`chunk_attribute_values.index(v)` at `types/points.py:904`, `lazy/level.py:234`
and the four sibling type readers. So with a hole:

- `unique_bins = [0, 2]` → asking for bin 2's value gives index 1 → filters
  `k[0] == 1` → **empty result, no error**.
- `unique_bins = [0, 2, 3]` → asking for bin 3's value gives index 2 →
  **returns bin 2's data under bin 3's label**.

Verified producers of holes: `_map_by_attribute` with explicit `bins`
(`spec.py:162-165` — e.g. `bins=[0,30,80,inf]` gives `unique_bins=[0,2]`),
`_map_by_object_id` with `bins`, and `_map_by_group` (empty groupings).
`assign_attribute_bins` (`core/attr_chunking.py:82-87`) is dense by
construction, which is why the non-rechunk writers never trip this.

**Suggested fix:** canonicalise once — remap `obj_to_bin` through
`{old: new for new, old in enumerate(sorted(unique_bins))}` before anything
consumes it, so keys, grid extent and the values list agree by construction.

**Related in the same file:**
- **`-1` bins.** `_map_by_group` assigns `-1` to ungrouped objects
  (`spec.py:114-117`). The key `(-1, z, y, x)` raises `StoreError` from
  `_check_coords_in_bounds` — and since the rank now matches, it raises with the
  *rank* diagnosis, which is the wrong message. All-ungrouped gives
  `max([-1])+1 == 0`, i.e. a zero-extent leading axis. A remap makes `-1` simply
  bin 0; decide whether that is the wanted semantics and document it.
- **`by="object_id"` with `bins=None`** gives `{oid: oid}` (`spec.py:123-125`).
  10 000 objects → a 10 000-wide leading axis, and the loop at
  `engine.py:230-293` is O(n²). Scaling bug, not correctness.
- **`bin_to_value` records the value of the lowest oid in each bin**
  (`engine.py:173-181`), not a bin edge. For a continuous `bins=[...]` rechunk
  the stored `chunk_attribute_values` is a list of arbitrary sample values, so
  `.index(v)` can only match by luck even when bins are dense.
- **`by="spatial"`** yields a pointless extent-1 leading axis and
  `chunk_dims[0] == "spatial"`.

**Tests:** no test in `tests/test_rechunk_by_attribute.py` reads data back — all
six stop at `read_level_metadata`, and the one that asserts the values list uses
a sorted-set comparison, which is blind to a positional shift. The
non-dense-capable config is exercised at
`tests/integration/test_lazy_sharding_rechunk.py:339-363` but only asserts
`bins_created >= 2`.

---

## 4. Add a validator rule for array rank vs key arity

**Severity: medium.** **Decided: do it.**

A store corrupted by the rank bug fixed in `cf4c2f1` passes all four validators
clean. `validate/structure.py:67-172` never opens an array node;
`validate/consistency.py:169-170` checks the decoded fragment's column count
against `sid_ndim`, which is unrelated to key arity.

Proposed L1 rule: every per-chunk array in a level shares one rank, and the keys
in its `nonempty_chunks` have that many components. Flags already-broken stores,
which nothing else can.

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
