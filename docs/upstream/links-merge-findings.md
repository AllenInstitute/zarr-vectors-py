# Links-merge findings from a downstream consumer

Two defects in the `links/<delta>/<offsets>/` merge, found from the vantage
of `zarr-vectors-tools` rather than from inside core. Neither blocked the
tools migration — both had tools-side workarounds — but both are core
defects, and **core's own tests could not catch either one**, which is the
part worth keeping.

Reproduced against branch `links-offset-merge`, zarr 3.2.1.

**Status: both fixed in core.** This document is the record of what they
were and why the suite was blind to them. Regressions live in
`tests/test_links_merge_regressions.py`, each verified to fail against the
reintroduced defect.

---

## 1. `read_chunk_links` silently returned fabricated links

**Severity:** silent data corruption in a public API. No exception, no
warning, wrong row count, wrong values.

### Cause

`read_chunk_links` resolved its row width from the family group's
`link_width` — the **logical** width L. But `links_has_perm` makes the
**physical** row width `1 + L` for any array that is non-intra, `delta == 0`
and undirected-canonical: the row is `[perm_idx, vi_0, … vi_{L-1}]`. The
reader decoded physical bytes at the logical width.

`links_has_perm`'s own docstring asks that it be kept *"in lockstep with
`_cell_placements`: it is the single definition the writer and reader both
consult"*. This reader never consulted it.

### Why it was silent rather than loud

With N records a cell holds `(1 + L)·N` elements, decoded at `ncols = L`.
Since `(1 + L)·N ≡ N (mod L)`:

- raises iff `N % L != 0`
- **silently returns `(1 + L)·N / L` fabricated rows iff `N % L == 0`**

At `link_width=2` that is loud on odd record counts and silent on every
even one — roughly half of all cells. A cell with many records is no safer
than one with two. For a triangle mesh (L=3) it goes silent whenever a cell
holds a multiple of 3 records.

### Observed

```
N=1 truth=1 -> raised: Element count 3 is not divisible by ncols=2
N=2 truth=2 -> SILENT, 3 rows      # [[0,0],[0,0],[1,1]] — 2 real records
N=3 truth=3 -> raised: Element count 9 is not divisible by ncols=2
N=4 truth=4 -> SILENT, 6 rows
```

For N=2 the true records are `((0,0,0),0)–((2,0,0),0)` and
`((0,0,0),1)–((2,0,0),1)`. On disk: `[0,0,0, 0,1,1]`. Returned:
`[[0,0],[0,0],[1,1]]` — three edges that do not exist, including a
self-loop.

### Why the suite was green

Core never called this reader with non-intra offsets. All three call sites
passed intra, where `links_has_perm` is False and physical == logical. **The
bug was unreachable from inside core and reachable from the first line of
downstream code that reads a cross cell per-chunk** — and `read_chunk_links`
is the natural thing to reach for, its `offsets=` parameter advertising
exactly that use.

This is the shape worth remembering: a defect can be structurally invisible
to a suite that only exercises the paths its own callers take.

### Fix

`read_chunk_links` now takes the width from the array's own `has_perm`
stamp, never inferring it from L. Rows come back **as stored**, and the
docstring says so plainly: `1 + L` wide with `perm_idx` in column 0, in
*placement* order rather than input order, pointing at `read_links` /
`read_links_for_tuple` for whole records.

Returning as-stored (rather than raising) keeps `read_chunk_links`
symmetric with `write_chunk_links`, which writes rows as given.

The prior docstring — *"this reader does not reverse a canonical sort, so it
never strips a `perm_idx` column"* — placed a burden on callers with no
signal that a burden existed, while the default width silently guaranteed
they got it wrong.

---

## 2. The decentralized-write manifest protocol had no callers

**Severity:** incomplete contract; the documented entry point did nothing.

### Cause

Core had both halves of a coordinator/worker protocol for `nonempty_chunks`
and connected neither:

| Piece | Role | Callers |
|---|---|---|
| `write_bytes(..., record_presence=False)` | worker half — skip the racing shared manifest | **zero** |
| `derive_nonempty_chunks(array_name)` | coordinator half — rebuild it from the listing | **zero** |

`group.py` told workers:

> "Decentralized writers therefore pass False and leave the manifest to a
> coordinator's `derive_nonempty_chunks` (see
> `zarr_vectors.core.arrays.finalize_links`), which rebuilds it from the
> store listing after all workers finish."

`finalize_links` never called `derive_nonempty_chunks`. It enumerated cells,
counted rows, merged counts into the family meta — and that was all. The
docstring pointed at a coordinator that did not coordinate, while
`finalize_links` was already walking every cell it would have needed.

### Why the merge made this worse, not better

`_is_per_chunk_array` now returns true for `links/<delta>/<offsets>`, so
cross-chunk links became chunk-grid arrays **with a `nonempty_chunks`
manifest that decentralized writers race on**. Before the merge they were
tuple-keyed and carried no manifest at all. Any parallel writer following
the documented advice got a stale manifest — and `list_chunks` trusts it, so
cells silently vanish from enumeration while their payloads sit on disk.

Downstream evidence that the contract had been published before it was
wired: `zarr-vectors-tools` called `level_group.rebuild_nonempty_manifests()`
at five sites — a method that exists in neither the tree nor its history.

### Fix

- `finalize_links` now calls `derive_nonempty_chunks` per offsets segment
  before enumerating. Segment discovery already used `children()` (a store
  listing), so it was race-free; only the per-array manifest needed rebuilding.
- `write_chunk_links` gained `record_presence`, and `write_link_cells` passes
  `False` — the worker half now exists at a call site rather than only in a
  signature.

**Ordering constraint** (documented at the call site): `derive_nonempty_chunks`
is unsharded-only — a shard packs many cells into one object whose inner
index is not derivable from key names — so `shard_store` must run *after*
`finalize_links`.

Verified: two workers writing disjoint cells of the *same* offsets array,
both unstamped, coordinator rebuilds, both records survive.

---

## 3. "Empty cell" and "absent chunk" became indistinguishable

Not in the original report, but the same family, and found by following it.

Collapsing each per-chunk family to one vlen array made absence and
emptiness **byte-identical**: an allocated-but-never-written cell reads back
`b""`, which is also the vlen fill value and also what an explicitly-written
empty payload returns. Under the old per-cell layout absence was a missing
node, so it raised. Several call sites still keyed off that raise:

- **`read_chunk_fragment_attributes` ignored `default=`.** `b""` passes the
  stride check (`len(b"") % row_bytes == 0`) and `np.frombuffer(b"")` yields
  an empty array — so the `except` never fired and the caller got
  `array([], dtype=float32)` where it asked for `None`. Fixed with a
  `chunk_exists` guard: the presence manifest is what separates the two now.
- **`_read_modify_write_blob` treated an empty cell as corrupt.** It caught
  only `StoreError`, but `decode_fn(b"")` raises `ArrayError`, so
  `write_chunk_fragments(mode="append")` to a fresh cell died with
  `Fragment-index blob too short: 0 < 16` instead of starting from
  `initial`. Fixed by treating empty and absent alike.
- **`Group.read_bytes` no longer raises for an unwritten in-grid cell** — it
  returns `b""`. Out-of-grid coords still raise, which keeps the useful
  distinction: `b""` means "addressable, nothing written"; `StoreError` means
  "not addressable". Callers that used the raise to detect absence must use
  `chunk_exists`. Pinned by
  `test_batched_reads.py::{test_batched_reads_missing_chunk_omitted_from_cache,
  test_read_bytes_raises_for_out_of_grid_coords}`.

The links code had this right from the start (`if not raw: return []`); the
older call sites did not, and nothing forced them to agree. If you are
auditing for more of these, the search is: any `except StoreError` or
`except Exception` whose purpose is "the chunk isn't there".

## 4. `write_lines` wrote the same link for every line in a chunk

Pre-existing, unrelated to the merge, and the most serious defect found —
silently wrong data in a public writer. Found by a doc agent reading
`types/lines.py` while reconciling the geometry-type spec, then confirmed by
running it.

`lines.py` appended each cross-chunk endpoint as its own single-vertex
fragment, then wrote the link as:

```python
cross_links.append(((ca, 0), (cb, 0)))   # vertex index hardcoded to 0
```

Link endpoints are **chunk-local vertex indices**. A chunk's vertices are its
fragments concatenated in order, so the *k*-th line's endpoint sits at
chunk-local index *k* — not 0. Three lines crossing the same boundary
therefore decoded to three *identical* records, all pointing at the chunk's
first vertex:

```
chunk (0,0,0) holds 3 vertices -> [1.0, 2.0, 3.0]
  link: (((0, 0, 0), 0), ((1, 0, 0), 0))
  link: (((0, 0, 0), 0), ((1, 0, 0), 0))
  link: (((0, 0, 0), 0), ((1, 0, 0), 0))
```

Fixed to `((ca, fragment_idx_a), (cb, fragment_idx_b))`; each link now
resolves to its own line's endpoints. Pinned by
`TestLinesCrossChunkEndpointIndices`.

**Why nothing caught it:** `read_lines` reconstructs geometry from
`object_index/` manifests, not from the links family. The links were wrong
on disk but no read path consulted them, so every round-trip test passed.
Same shape as defect #1 — a bug living on a path the package writes but
never reads.

## Also addressed

**Privates every consumer had to re-implement.** `link_family_policy`,
`iter_link_cells` and `cell_endpoint_chunks` are now public.
`link_family_policy` is the only source of
`(link_width, sid_ndim, directed, store)`, and a caller needs all four to use
`parse_offsets`, `links_has_perm` or `create_links_array` correctly.
`iter_link_cells` is the only way to ask "which cells exist" without an
O(records) `read_links`. Tools was copying ~35 lines against undocumented
meta keys — the standard way a downstream drifts out of sync with a format.

**`create_links_family`.** `create_links_array` could not stamp family policy
without materialising an offsets array, so a cross-only family had to invent
an empty intra array or let workers race to create one. `create_links_family`
stamps the group alone; it is idempotent, merges rather than overwrites (so
`finalize_links` counts survive), and raises on a conflicting re-stamp rather
than stranding arrays already written under the old policy.

**Doc drift, behaviour was correct:** `read_chunk_link_attributes` (now states
*why* it is intra-only — the fragment sidecar is keyed by chunk alone, so the
all-zero array is the only one it partitions), `reorder_vertices_implicit`,
and `_ensure_array_dir` (its "non-per-chunk" example still named
`cross_chunk_links`; now explains the depth-aware rule — `links/0` is a group,
`links/0/0.0.+1` an array).

---

## 5. `read_graph(object_ids=)` / `read_mesh(object_ids=)` were silent no-ops

Pre-existing, unrelated to the merge. Both declare and document an
`object_ids` filter and **never reference it in their bodies** — proven by
AST rather than by reading:

```
read_graph:     declares object_ids -> SILENT NO-OP
read_mesh:      declares object_ids -> SILENT NO-OP
read_polylines: declares object_ids -> used 8x
```

So `read_graph(store, object_ids=[5])` returned the *entire* level while the
caller believed they had scoped the read — the worst possible outcome, since
the result is plausible.

Implementing the filter is out of scope here, so the fix is minimal and
honest: both now raise `NotImplementedError` pointing at `read_polylines`,
which does implement it. That converts a silent wrong answer into a loud
one. Unfiltered reads are unaffected.

## Open — needs a decision, not a guess

**`skeletons.py` contradicts itself about endpoint order, and nothing on
disk resolves it.** Two writers populate the *same* `links/0/` family with
opposite conventions:

- `write_skeleton_chunk` — "``branch_links`` are ``(child_order_idx,
  parent_order_idx)``", i.e. endpoint 0 is the **child** (stated in three
  places).
- `write_skeleton_cross_chunk_links` — "endpoint 0 the parent and endpoint 1
  the child", and its own docstring says it writes "within the same
  ``links/0/`` family :func:`write_skeleton_chunk` writes branch links into".

The family is `directed=True`, which preserves input order verbatim — that is
the whole point, so parent→child survives a canonical sort. The consequence
is that **a reader cannot tell which convention a given record used**: both
orders are legal, stored identically, and indistinguishable.

Left unfixed deliberately. `write_skeleton_cross_chunk_links` has no
in-package caller, so it is public API whose docstring *is* the contract;
flipping either side would silently invert an external caller's parent/child
relationships with no error. Which one is authoritative is a question for
whoever owns the skeleton format.

**Also dead, pending removal:** `spatial/boundary.py::partition_cross_level_edges`
has zero callers and its docstring still describes the removed
`links/<delta>/<chunk_key>` + `cross_chunk_links/<delta>/data` split.

## The through-line

Both defects share a shape: **core could not have caught either from the
inside.** #1 was unreachable from core's own call sites; #2 was a contract
whose two halves were each individually plausible and jointly inert. A
downstream consumer found both on contact.

Worth carrying forward:

- A default that is *usually* right (`link_width` as a row width) is more
  dangerous than one that is always wrong, because it fails on a schedule —
  here, every odd N — that looks like a flake rather than a defect.
- A docstring that names a collaborator (`finalize_links`,
  `rebuild_nonempty_manifests`) is an assertion. It should be tested like one.
- **Data the package writes but never reads is unverified by construction.**
  Both #1 and #4 lived on write-only paths: `read_lines` rebuilds geometry
  from manifests and never consults the links it wrote, so `write_lines`
  emitted three identical links for three distinct lines and every
  round-trip test passed. A round-trip test only covers the paths it
  round-trips through.
- **"Pre-existing failure" is not the same as "not our problem."** Three
  things here were filed that way and all three were wrong: the Windows
  `PermissionError`s were a real shared-manifest race; the perf failures hid
  a real 3x regression behind an already-red threshold; and `read_chunk_links`
  was fabricating rows on a path core never called. An already-failing
  neighbour is camouflage.
- **The docs asserted more than the code enforced, repeatedly.** Reconciling
  them removed an entire fictional `zarr_vectors.repair` module (four
  functions, `ModuleNotFoundError` on import), several invented L1/L2 check
  IDs, and a `CAP_MULTISCALE_LINKS` validation that is only ever stamped,
  never read. A CI check that every documented symbol imports would have
  caught most of it.
