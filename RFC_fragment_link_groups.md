# [RFC] Link groups that follow vertex fragments (`fragment_link_groups`)

*Draft for a zarr-vectors-py `[RFC]` issue, following
`docs/spec/contributing/spec_change_process.md`. An implementation is on the
local branch `feat/fragment-link-groups`.*

## Motivation

A level stores each chunk's intra-chunk links (for a mesh, its faces) as
flat rows plus a `link_fragments/<chunk>` index of group row ranges. The
spec deliberately lets groups be anything ("link groups need not be 1:1 with
the chunk's vertex fragments"), and manifests reference vertex fragments
only. So a reader that wants one object's links has no way to find them:
it must read the whole cell, which holds every object's links in that
chunk, and filter.

For meshes that is most of the cost of drawing an object. On a MICrONS
mesh pyramid (65 M vertices at level 0), drawing three cells read 4.2 GB at
full resolution and 415 MB at a coarse level, almost all of it other
objects' faces. With groups that follow fragments and a reader that
range-reads only the object's rows, the same views read 75 MB and 4.7 MB.

Writers already emit one object's links together, so most stores are one
small sidecar rewrite away from the layout; what is missing is a way to
*promise* it, so a reader may rely on it.

## Proposed change

1. **Level metadata flag** `fragment_link_groups: bool` (absent ⇒ `false`).
   It promises, for every chunk of the level:
   - `link_fragments/<chunk>` has exactly as many groups as
     `vertex_fragments/<chunk>` has fragments, and
   - group `k` holds exactly the intra-chunk links whose endpoints all lie
     in vertex fragment `k` (so no intra-chunk link joins two fragments).

   ```json
   {"zarr_vectors_level": {"level": 0, "...": "...", "fragment_link_groups": true}}
   ```

2. **Root capability token** `CAP_FRAGMENT_LINK_GROUPS = "fragment_link_groups"`
   in `format_capabilities` when any level is stamped. A hint; the level
   flag is authoritative.

3. **A claim, handled exactly like `fragments_tile`:**
   - stamped only after the writes, by `stamp_fragment_link_groups`, which
     verifies every chunk (using per-group endpoint bounds the writer
     records, so the rows just written are not re-read);
   - cleared by `Group.write_bytes` / `write_cells` on any later write to
     `vertex_fragments`, `link_fragments` or the intra-chunk link array, so
     a writer that does not know about it (edits, appends, rechunking)
     withdraws it rather than leaving it stale;
   - checked by `validate_consistency` (L3).

4. **Writers.** `write_mesh` lays out one fragment per object (as it
   already does) and cuts its intra-chunk faces into one group per fragment,
   then stamps. `write_links` gains `intra_group_sizes`, a callback that
   cuts a fresh intra-chunk cell into groups without reordering rows (so
   the returned partition stays row-exact).

5. **Retrofit.** `index_fragment_link_groups(store, levels, dry_run, verify)`
   brings existing levels into the layout with the least rewriting: nothing
   where groups already follow fragments, a new `link_fragments` cell where
   rows are already in fragment order, and row reordering only where needed
   and only when the level has no row-aligned intra-chunk link attributes.
   A level where some link joins two fragments is left unstamped.

Spec pages touched: `layout/level_groups.md` (new *Claims* table),
`object_model/links.md` (new section), `layout/fragment_index_arrays.md`,
`validation/l3_consistency.md`, `geometry_types/mesh.md`, plus the LinkML
slot.

## Backward compatibility

Additive. Link rows are unchanged; only how they are grouped, which readers
that concatenate groups (all of zarr-vectors-py's) never see. A reader that
does not know the flag reads the store correctly. Existing stores stay
valid and simply lack the flag; `index_fragment_link_groups` adds it.

## Alternatives considered

- **Manifests that also reference link groups.** Exact, but a new manifest
  layout (`vlen_manifests_v3`) for every writer and reader, to express what
  a 1:1 rule gives for free.
- **An array mapping each link group to its object/fragment.** A new array
  per level and a second index to keep consistent.
- **A declaration on the intra-chunk array's own attributes** (what the
  first prototype did). Invisible to L2/L3, which read group attributes,
  and not cleared by edits.
- **An unversioned tool-namespaced key** (`zarr_vectors_tools.…`). Workable
  as a stopgap, but the guarantee is only safe if the core write path
  clears it, which needs core support anyway.

## Open questions

- `write_chunk_vertices` rewrites `vertex_fragments` with every vertex
  write, so an edit that only moves vertices also clears the claim, though
  the fragments are unchanged. Conservative, at the cost of a re-stamp;
  should that write path skip an unchanged fragment index?
- Should `read_mesh(object_ids=...)` be implemented on top of this? It
  currently raises `NotImplementedError`.
