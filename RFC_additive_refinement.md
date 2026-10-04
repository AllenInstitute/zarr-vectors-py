# [RFC] Additive pyramid refinement (`refinement: "add"`)

*Draft for a zarr-vectors-py `[RFC]` issue, following
`docs/spec/contributing/spec_change_process.md`.  Implemented on the local
branch `feat/additive-refinement` (format, readers, conversion) and in
zarr-vectors-tools `feat/additive-pyramids` (construction).*

## Motivation

Every pyramid level today is a *replacement* for the one below it.  A
sparsity pyramid keeps half the streamlines at level 1, a quarter at level
2, and stores each of them again at every finer level.  Two costs follow:

* **Storage.**  With a sparsity factor of 2 per level the levels sum to
  about 1 + 1/2 + 1/4 + ... of level 0 -- up to 2x for a deep pyramid, and
  1.14-1.33x for the factors used in practice (8 or 4 per level).
* **Re-fetching.**  A viewer that has drawn level 2 and zooms in discards
  it and fetches level 1, which contains every streamline of level 2 again;
  zooming once more fetches them a third time.

When the levels differ only in *which* objects they keep -- not in how they
are drawn -- nothing in level 1 that level 2 already has needs to be
stored or fetched again.

## Proposed change

1. **Level field** `refinement`, optional, `"replace"` (the default, and the
   meaning of every existing level) or `"add"`.  On an `"add"` level `L`:

       complete content of L = L's own data  ∪  complete content of L + 1
       chain(L) = [L] + chain(L + 1)   if L is "add"   else [L]

   The coarsest level cannot be `"add"`.  `vertex_count` and the object
   index describe the level's own data.

2. **Root token** `additive_levels` in `format_capabilities`.

3. **Root list** `required_capabilities` (new, optional, default empty):
   capabilities a reader MUST implement to read the store correctly; a
   reader MUST refuse a store listing one it does not implement.  An
   additive store lists `additive_levels` there.

4. **Readers** return a level's complete content by default -- the union
   over its chain, vertex and face indices offset, object ids preserved --
   and take an opt-out for one level's stored data (`own_level_only=True`
   in zarr-vectors-py).

5. **Writers** never write a level from a union they read, never rebuild an
   additive level from its neighbours, never remove a level an additive
   level depends on, and add the required capability before marking a
   level additive.

6. **Validation**: `refinement` values; the coarsest level is not `"add"`;
   an additive store requires the capability (L2).  An object of an
   additive level is not also in the next level's complete content (L3).
   Pyramid counts compare complete content (L5).

Spec pages touched: `multiscale/additive_refinement.md` (new),
`multiscale/pyramid_construction.md`, `multiscale/sparsity.md`,
`layout/level_groups.md`, `layout/root_metadata.md`,
`foundations/store_types.md`, `validation/l2_metadata.md`,
`validation/l3_consistency.md`, `validation/overview.md`, the LinkML slots
`refinement` and `required_capabilities` (plus the `Refinement` enum), and
the regenerated JSON Schema.

## Where it is meaningful

* **Object sparsity with unchanged geometry.**  A streamline or polyline
  pyramid built with coarsen factor 1: each kept streamline is
  byte-identical at every level that keeps it, so the union over a chain is
  exactly the replacement level.  zarr-vectors-tools tests this byte for
  byte, with and without chunk scaling.
* **Nested point subsets.**  Each coarser level a subset of the finer one.

## Where it is not

* **Meshes.**  Every mesh coarsener re-tessellates.  An object kept at a
  coarse level is drawn coarse in the finest view, and coarse and fine
  surfaces of neighbouring objects mix.
* **Spatially coarsened geometry** -- metanode binning (the per-object
  coarsener, even at coarsen factor 1), RDP simplification or resampling of
  polylines, skeleton simplification, mesh clustering or decimation.  The
  objects kept at a coarse level exist only in coarse form, so the
  "complete" finest view shows them at the coarse resolution.

The format does not forbid these; zarr-vectors-tools builds them on request
and warns.

## Backward compatibility

**Not backward compatible for readers, and the mechanism that says so is
new.**  A store with additive levels is valid only to a reader that follows
the chain.  A reader that does not shows level 0 as only the objects no
coarser level kept -- a plausible-looking, silently incomplete picture.

`required_capabilities` is how a store says "do not read me without this".
But readers written before the list existed -- every zarr-vectors-py before
this change, and the Neuroglancer data source before it adds the check --
do not consult it, so **they will read level 0 of an additive store as if
it were complete**.  Only readers from this version on refuse what they do
not understand.  There is no way around this short of a format-version
bump that old readers already reject; the RFC proposes accepting it, since
additive stores are opt-in at construction and new.

Existing stores are unaffected: no level of theirs is additive, they list
no required capability, and every reader reads them as before.  The
additions to `RootMetadata` and `LevelMetadata` are optional keys.

## Alternatives considered

* **A different level kind under a new key** (e.g. `delta_levels/`).
  Clearer to old readers -- they would not find the data at all -- but it
  duplicates the level layout, every per-level tool, and the NGFF
  `multiscales` block.
* **Bumping the format version** so old readers refuse the store.  Heavy:
  every other store would have to carry the new version too, or readers
  would have to accept two.  `required_capabilities` gives the same refusal
  to every reader from now on, at the cost stated above.
* **Store-level instead of level-level flag.**  A per-level flag lets a
  pyramid keep replacement levels where the geometry is coarsened and add
  only where it is not.

## Open questions

* Should `object_count`/`num_present`-style per-level counts gain a
  *complete* twin, so a viewer can show totals without walking the chain?
* Should `required_capabilities` take over the role of some existing
  informational tokens (none of them changes what data means today)?
