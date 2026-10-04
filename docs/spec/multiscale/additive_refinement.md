# Additive refinement

*Proposed for 0.9.3; see `RFC_additive_refinement.md`.*

## Terms

| Term | Meaning |
|------|---------|
| **Replacement level** | A level whose own data is its complete content (`refinement: "replace"`, the default). Every level written before this page existed is one. |
| **Additive level** | A level whose complete content is its own data together with the complete content of the next coarser level (`refinement: "add"`). |
| **Own data** | What a level stores: its vertices, fragments, links, attributes and object manifests. |
| **Complete content** | What a reader of the level shows: its own data for a replacement level, the union over its chain for an additive one. |
| **Chain** | `chain(L) = [L] + chain(L + 1)` if level `L` is additive, else `[L]`. |

## Introduction

In an ordinary pyramid each coarser level *replaces* the finer one: a
sparsity pyramid stores the half of the streamlines kept at level 1 twice,
once at level 0 and once at level 1, and a viewer zooming in discards what
it drew from level 1 to fetch level 0, which contains the same streamlines
again.  Storage is 1.14-1.33x of level 0 for typical pyramids, and the bytes
re-fetched on every zoom grow with it.

An additive level stores only what the next coarser level does not.  Every
object (or point) is stored at exactly one level -- the coarsest that keeps
it -- so the pyramid costs about what level 0 alone did, and a viewer that
has drawn level `L + 1` fetches only level `L`'s own data when zooming in.

## Technical reference

### Level metadata

`zarr_vectors_level.refinement` is optional:

| Value | Meaning |
|-------|---------|
| absent, `"replace"` | The level's own data is its complete content. |
| `"add"` | The level's complete content is its own data together with the complete content of level `L + 1`. |

Writers omit the key for `"replace"`.  The coarsest level of a store MUST
NOT be `"add"`, and level `L + 1` MUST exist for an `"add"` level `L`.

`vertex_count` and the object index (`object_index/manifests`, its
`num_present`) describe the level's **own** data.  `object_sparsity` keeps
describing the level's complete content, as in the replacement pyramid it
was built from.

### Root capabilities

A store with at least one additive level MUST list `"additive_levels"` in
both `zarr_vectors.format_capabilities` and
`zarr_vectors.required_capabilities`.

`required_capabilities` (optional, default empty) names the capabilities a
reader MUST implement to read the store correctly.  A reader that does not
implement one of them MUST refuse to open the store, rather than read it as
something else.  Additive levels are the first capability that needs it:
a reader unaware of them reads level 0 of an additive store as if it were
complete, and silently shows part of the data.

**Readers that predate `required_capabilities` do not check it.**  Opening
an additive store with one of them shows each level's own data only.  That
is the price of declaring the capability in an additive field; see the RFC.

### Reading

A reader asked for level `L` of a store with additive levels returns the
union of the own data of every level in `chain(L)`:

- vertices, links and per-vertex attributes are concatenated, finest level
  first, with each level's link and face indices offset by the vertices
  before them;
- an object keeps its id: it is stored at exactly one level of the chain;
- a per-vertex attribute is part of the union only where every level of
  the chain that has vertices carries it.

A spatial selection (a bounding box) means the same region at every level.
A selection of chunks of level `L`'s grid selects, at a coarser level of the
chain, the chunks of that level's grid that overlap them; with a coarser
chunk grid that covers more than the requested region.

`zarr-vectors-py` readers (`read_points`, `read_polylines`, `read_lines`,
`read_graph`, `read_mesh`, `read_skeleton_by_segment_id`, and
`Level.read` / `Level.select` in the API) return the complete content by
default and take `own_level_only=True` for a level's own data.
`zarr_vectors.building.level_chain(store, level)` gives the chain.

### Writing

A conforming writer:

1. never writes a level from a union it read: a writer that rewrites level
   `L` from data it read uses `own_level_only`;
2. never rebuilds an additive level from the level below or above it (an
   additive level is not complete, and a level built from it would be
   wrong), and never removes a level an additive level is completed by;
3. adds `"additive_levels"` to `required_capabilities` before it marks the
   first level additive, and marks a level additive before it removes the
   data the coarser level duplicates -- so an interrupted conversion reads
   back with some objects twice, never with any missing.

`zarr_vectors.multiresolution.additive.make_levels_additive` converts a
finished replacement pyramid in place, finest level first.  On a level with
an object index it drops every object whose manifest at `L + 1` is
non-empty, with the fragments and vertices only they referenced.  On an
object-less point cloud it drops the multiset of exact vertex coordinates
present at `L + 1`.  What stays is byte-identical.  Cross-level link
families (`delta != 0`) are removed: they join one object's vertices at two
levels, and an object of an additive pyramid lives at one.

### Where it is meaningful

The union over a chain is the picture the replacement pyramid draws only
when the levels differ in *which* objects or points they keep, not in *how*
they are drawn:

- **object sparsity with unchanged geometry** -- a streamline or polyline
  pyramid built with coarsen factor 1 keeps every kept streamline
  byte-identical;
- **nested point subsets** -- each coarser level a subset of the finer one.

It is not meaningful for meshes, or for any level that spatially coarsens
geometry (binning into metanodes, RDP simplification or resampling of
polylines, skeleton simplification, mesh clustering or decimation): the
objects kept at a coarse level are then stored only in coarse form, so the
complete finest view draws them coarse beside the fine ones, and for meshes
coarse and fine surfaces of neighbouring objects mix.  The format permits
it; `zarr-vectors-tools` warns.

### Validation

| Level | Rule | Failure |
|-------|------|---------|
| L2 | `refinement` is `"replace"` or `"add"` | Error |
| L2 | An `"add"` level has a level `L + 1` (the coarsest level is not `"add"`) | Error |
| L2 | A store with an `"add"` level lists `"additive_levels"` in `required_capabilities` | Error |
| L2 | ... and in `format_capabilities` | Warning |
| L3 | No object of an `"add"` level is also in level `L + 1`'s complete content | Error |
| L5 | Vertex and object counts are non-increasing across levels, counted over complete content | Error |
