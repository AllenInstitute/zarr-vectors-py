# Pyramid construction

## Terms

**Metanode**
: A synthetic vertex at a coarser resolution level that represents
  a group of source vertices co-located within a supervoxel bin.
  Its position is the centroid (or another aggregation) of the source
  vertices in the bin. Metanodes are the fundamental unit of spatial
  coarsening.

**Coarsening factor**
: Per-level isotropic vertex-reduction target — `coarsen_factor=2.0`
  means each axis is binned 2× coarser, giving up to ~8× fewer
  vertices in 3D (exact reduction depends on the actual vertex
  distribution).

**Sparsity factor**
: Per-level object-keep target — `sparsity_factor=3.0` keeps every
  third object on average. `1.0` keeps all objects (the default).

**Aggregation mode**
: How per-vertex attributes are combined within a bin when producing a
  metanode's attribute value. Zarr Vectors defines the token set
  `mean`, `sum`, `mode`, `count`, `min`, `max` —
  see [`constants.VALID_AGGREGATIONS`](../../../zarr_vectors/constants.py).
  The vocabulary is normative; **core does not consume it** — see
  *Aggregation modes* below.

**Coarsening method**
: How the writer reconciles object identity across levels.  Core
  provides `per_object` (default; OID-stable, metavertices may be
  shared).  Other methods (e.g. `cross_object_metanode` / `grid_metanode`,
  fresh OID space per level) are provided by `zarr-vectors-tools` and
  selected by name via `method=`.

**Cross-level link**
: An edge from a fine-level vertex to its coarse-level parent
  metanode. Stored in the single link family at the fine level under a
  non-zero level delta (`links/<delta>/<offsets>/`) — see
  [Links](../object_model/links.md).

**Chunk scale factor** (`chunk_scale_factor`)
: Per-axis positive-integer multiplier applied to the *source* level's
  `chunk_shape` to derive the *target* level's `chunk_shape`. The
  default `1` leaves the chunk grid unchanged, so the level inherits
  the root `chunk_shape`. Any other value gives that level a **different
  chunk grid** from the root, which is what makes cross-level offsets
  anchor-projected rather than plain differences (see
  *Cross-level offsets are anchor-projected* below).

**Anchor**
: A source chunk coordinate re-expressed in the target level's chunk
  grid, `floor(c_src * r_src / r_trg)`. Cross-level offsets are measured
  from the anchor, not from the raw source coordinate.

---

## Introduction

Building a resolution pyramid means constructing one or more coarser
representations of the base-level data. Coarsening operates
independently on two axes:

- **Spatial coarsening** — merge vertices in the same supervoxel bin
  into a single metanode (per the `coarsen_factor`).
- **Object thinning** — drop a fraction of discrete objects (per the
  `sparsity_factor`).

Every coarsening pass can additionally emit **cross-pyramid-level
links** that record which fine vertex maps to which coarse metanode.
There is no separate cross-chunk family: cross-level links live in the
same `links/<delta>/<offsets>/` family as every other link, under a
non-zero `<delta>`. See [Links](../object_model/links.md) for the
on-disk layout and
[`examples/07_multiscale_links.ipynb`](../../../examples/07_multiscale_links.ipynb)
for a walkthrough.

A pyramid may also give each level its own **chunk grid** via
`chunk_scale_factor`. This page documents the coarsening algorithm per
geometry type, the two public APIs (`build_pyramid` and
`coarsen_level`), the per-level chunk grid, and the cross-level link
options.

---

## Technical reference

### Core vs `zarr-vectors-tools`

Core ships only the simplest, **dependency-free** coarsening:

- the `per_object` metavertex-binning coarsener (`build_pyramid` /
  `coarsen_level` with the default `method="per_object"`), and
- the `random` object-selection strategy (`sparsity_strategy="random"`).

Everything more elaborate — the geometry-type-specific coarsening described
below (quadric mesh decimation, Douglas–Peucker polyline simplification,
graph/point metanodes), alternative coarsening methods (e.g.
`cross_object_metanode`), and non-random object selection (spatial coverage,
length/attribute-ranked, point-thinning) — lives in **`zarr-vectors-tools`**.
That package registers its implementations through
[`zarr_vectors/multiresolution/registry.py`](../../../zarr_vectors/multiresolution/registry.py)
on import; `build_pyramid` / `coarsen_level` then dispatch to them by
`method=` / `sparsity_strategy=` name. Requesting one without
`zarr-vectors-tools` installed raises a clear error naming the package to
install. Core takes no dependency on it.

### Coarsening by geometry type

> These geometry-type-specific coarseners are provided by
> `zarr-vectors-tools`, not core. They register as `method=` values (see
> *Core vs `zarr-vectors-tools`* above).

#### Point cloud

Pure spatial coarsening. For each bin at the coarser level, all
vertices in the corresponding bin at the base level are merged into a
single metanode:

```
metanode_position  = mean(positions in bin)        # centroid
metanode_attribute = <aggregation mode>(attributes in bin)
```

(The aggregation mode is named from the vocabulary in
*Aggregation modes* below; how a given method selects it is that
method's own contract.)

There are no discrete objects in a point cloud; `sparsity_factor` is
ignored (always 1.0).

#### Polyline and streamline

Two-stage coarsening:

1. **Spatial:** for each object, the vertex sequence is downsampled by
   replacing each contiguous run of vertices within the same bin with
   one metanode at their centroid. Edge connectivity is preserved
   (consecutive metanodes are connected).
2. **Object thinning** (when `sparsity_factor > 1.0`): a subset of
   objects is selected for the coarser level using
   `sparsity_strategy`. Non-selected objects are not written.

After both stages, the coarser level's `links/0/` family is recomputed
(metanode positions may fall in different chunks than the originals, so
a pair that was intra-chunk at the fine level may land under a non-zero
offsets array at the coarse level, or vice versa).

#### Graph and skeleton

For each bin, all vertices in the bin merge into one metanode. Edges
are remapped: an edge `(u, v)` at the base level becomes an edge
between the metanodes of `u` and `v`. Self-loops (both endpoints in
the same bin) are removed; duplicate edges deduplicated.

For skeletons, the tree structure is preserved across coarsening.

Multi-skeleton stores support `sparsity_factor > 1.0` for thinning
when the store contains many independent skeletons.

#### Mesh

Mesh coarsening uses **vertex merging**:

1. Identify all vertices in each bin and replace them with one
   metanode.
2. Update face vertex indices to reference metanodes.
3. Remove degenerate faces (faces with two or more identical
   vertices after merging).
4. Remove duplicate faces.

This is conservative — it preserves topology at the cost of some
geometric quality. For high-quality mesh decimation, pre-process
with a dedicated mesh tool before writing.

### `build_pyramid` API

The recommended entry point. Pass `factors=[(coarsen, sparsity), ...]`
where the *i*-th tuple produces level `i+1` from level `i`:

```python
from zarr_vectors.multiresolution.coarsen import build_pyramid

build_pyramid(
    "scan.zarrvectors",
    factors=[
        (2.0, 1.0),                 # level 1: 2× coarsen, no sparsity
        (2.0, 1.0),                 # level 2: 2× coarsen
        (2.0, 3.0),                 # level 3: 2× coarsen + drop 2/3 objects
    ],
    method="per_object",            # core default; see "Methods"
    sparsity_strategy="random",     # core default; advanced ones via tools
    sparsity_seed=None,
    cross_level_depth=1,            # ±1 cross-level edges per pair
    cross_level_storage="explicit", # write both +1 (fine) and -1 (coarse)
)
```

Each factor pair `(coarsen, sparsity)` opts out by passing `1.0` on
that axis. Passing a non-default `method=` / `sparsity_strategy=` dispatches
to a `zarr-vectors-tools`-registered implementation.

**Signature** (see
[`zarr_vectors/multiresolution/coarsen.py:build_pyramid`](../../../zarr_vectors/multiresolution/coarsen.py)):

```python
def build_pyramid(
    store_path: str | Path,
    *,
    factors: list[tuple[float, float]],
    chunk_scale_factors: list[int | tuple[int, ...]] | None = None,
    sparsity_strategy: str = "random",     # core built-in; else via tools
    sparsity_seed: int | None = None,
    cross_level_depth: int = 1,
    cross_level_storage: str = "explicit",
    method: str = "per_object",            # core built-in; else via tools
) -> dict[str, Any]
```

### `coarsen_level` API

For one level at a time:

```python
from zarr_vectors.multiresolution.coarsen import coarsen_level

coarsen_level(
    "scan.zarrvectors",
    source_level=0,
    target_level=1,
    coarsen_factor=2.0,
    sparsity_factor=1.0,
    method="per_object",            # core default; advanced ones via tools
)
```

`source_level` does not have to be 0 — you can incrementally build
deeper pyramids by chaining `coarsen_level` calls.

`coarsen_level` accepts `cross_level_storage=` but defaults it to
`"none"`, so a standalone call emits **no** cross-level arrays unless
you opt in. Passing `"implicit"` / `"explicit"` emits the adjacent
`±1` families inline. Deltas of `±2` and beyond are only produced by
`build_pyramid(..., factors=...)`, which runs the composing finalize
pass over the whole pyramid.

`coarsen_level` also takes `chunk_scale_factor=` (default `1`) to give
the target level its own chunk grid — see
*[Per-level chunk grids](#per-level-chunk-grids-chunk_scale_factor)*.

### Methods

| Method | Provided by | Behaviour | OID stability |
|--------|-------------|-----------|---------------|
| `per_object` (default) | **core** | Per-object pyramid; metavertices may be shared between objects | **Stable** — each surviving object keeps its OID across levels |
| `cross_object_metanode` (alias: `grid_metanode`), and others | `zarr-vectors-tools` | Registered via the strategy registry; e.g. grid-binning that merges vertices across object boundaries for the smallest coarse representation | Method-defined |

Choose `per_object` (core) when downstream consumers need to track the
"same object" across resolution levels (e.g. drill-down navigation in
Neuroglancer, ID-preserving analytics). Advanced methods are selected by
passing their registered name as `method=`; they require
`zarr-vectors-tools` to be installed (see *Core vs `zarr-vectors-tools`*).

Core implementation:
[`_per_object_coarsen`](../../../zarr_vectors/multiresolution/coarsen.py).

### Aggregation modes

Zarr Vectors defines a canonical vocabulary of aggregation tokens
([`zarr_vectors.constants.VALID_AGGREGATIONS`](../../../zarr_vectors/constants.py)):

| Token | Description | Use case |
|-------|-------------|----------|
| `mean`     | Mean of values in bin | Continuous scalars (FA, intensity, concentration) |
| `sum`      | Sum of values in bin | Counts, densities |
| `mode`     | Most frequent value | Categorical / label attributes |
| `count`    | Number of source vertices in bin | Density tracking |
| `min` / `max` | Bin extrema | Thresholds, peak activations |

> **Core does not implement attribute aggregation.** Neither
> `build_pyramid` nor `coarsen_level` takes an `agg_mode=` argument.
> The core `per_object` coarsener aggregates **positions only**, always
> as the bin centroid (arithmetic mean), and carries **per-object**
> attributes across unchanged for each surviving OID (with a
> `present_mask` for dropped objects). It does not coarsen per-vertex
> attributes at all.
>
> The vocabulary above exists so that coarsening methods registered by
> `zarr-vectors-tools` name their modes consistently. Consult that
> package for the arguments its methods accept.

### Per-level chunk grids (`chunk_scale_factor`)

By default a pyramid coarsens *positions* but leaves the **chunk grid**
alone: every level inherits the root `chunk_shape`, so a coarse level
simply has fewer vertices spread over the same grid. Passing
`chunk_scale_factor > 1` instead gives the target level a chunk grid of
its own:

```
target_chunk_shape = source_chunk_shape × chunk_scale_factor   # per axis
```

`coarsen_level` takes a single `chunk_scale_factor=`; `build_pyramid`
takes `chunk_scale_factors=`, a list aligned with `factors` (one entry
per coarser level). Each entry is a scalar int (uniform across axes) or
a per-axis tuple. Every multiplier MUST be a **positive integer** —
chunk grids are required to nest — and a tuple's rank MUST equal
`sid_ndim`; violations raise `CoarseningError`.

The result is written as a **per-level `chunk_shape` override** on the
target level's `LevelMetadata`. The field is omitted (written as
`None`) when the derived shape equals the root `chunk_shape`, so a
default pyramid stamps nothing and every level inherits root. Readers
resolve the effective value with
[`get_level_chunk_shape`](../../../zarr_vectors/core/metadata.py); the
per-axis integer multiple of root is
[`chunk_scale_factor(root_meta, level_meta)`](../../../zarr_vectors/core/metadata.py),
written `r_L` below. A level that inherits root has `r_L = (1, …, 1)`.

Because `chunk_scale_factor` compounds down the pyramid (each level
scales the *source* level's shape, not root's), `r_L` MUST be read from
metadata rather than assumed from the factor list.

#### Cross-level offsets are anchor-projected

Chunk coordinates are indices into a level's grid. When two levels
carry different `chunk_shape`, their coordinates index cells of
different sizes and **are not commensurable** — subtracting them
directly is meaningless. So a cross-level offset is not
`c_trg - c_src`. The source is first re-anchored into the target's grid:

```
anchor = floor(c_src * r_src / r_trg)
o      = c_trg - anchor              # decode: c_trg = anchor + o
```

where `r_src` / `r_trg` are the source and target levels' chunk scales.
This is implemented by
[`anchor_chunk`](../../../zarr_vectors/spatial/boundary.py) using
integer floor division (exact at large coordinates, and floors toward
−∞ as negative chunk coordinates require). The scales are derived by
[`_derive_level_scales`](../../../zarr_vectors/core/arrays.py) and
fitted to `sid_ndim` by `_link_scales`.

When `r_src == r_trg` — the same level, or *any* pyramid built with the
default `chunk_scale_factor=1` — the anchor reduces to `c_src` and the
offset is the plain difference. **Anchor projection is therefore a
strict generalisation, not a branch**: intra-level offsets are the
`r_src == r_trg` case of the same formula.

**Why the naive difference fails.** The naive offset is not
translation-invariant across grids, so it cannot name a relationship.
Take root `chunk_shape = 100` and a level-1 `chunk_shape = 200`
(`r_0 = 1`, `r_1 = 2`), and two pairs that are *geometrically adjacent*
in exactly the same way:

| Pair | `c_src` (level 0) | `c_trg` (level 1) | Naive `c_trg − c_src` | `anchor = ⌊c_src·1/2⌋` | Anchored `o` |
|------|------------------|-------------------|----------------------|------------------------|--------------|
| A | 3 (spans 300–400) | 2 (spans 400–600) | `2 − 3 = −1` | `⌊3/2⌋ = 1` | `2 − 1 = +1` |
| B | 1 (spans 100–200) | 1 (spans 200–400) | `1 − 1 = 0` | `⌊1/2⌋ = 0` | `1 − 0 = +1` |

Both pairs point from a level-0 chunk to the level-1 chunk immediately
above it, yet the naive offsets disagree (`−1` vs `0`) — the same
geometric relationship would scatter across two different offset
arrays, and neither value would let a reader recover `c_trg`. The
anchored offset gives `+1` for both, so they share one array and decode
correctly.

> **Ordering constraint (writers).** `_derive_level_scales` falls back
> to all-ones when it cannot read the target level's metadata. All-ones
> is correct for a default pyramid but **silently mis-anchors every
> record** on a scaled one. A writer emitting cross-level links MUST
> therefore ensure the level it references across already exists on
> disk *with its `LevelMetadata`* — including any `chunk_shape`
> override — before writing. Core satisfies this by emitting the `±1`
> families only after the target level's `create_resolution_level`.

#### Source endpoint and sort order for `delta != 0`

For `delta != 0` the source is **always input endpoint 0**, which keeps
the source at the owning level so the anchor's `r_src` is that level's
scale. Cross-level records are **never canonical-sorted**, regardless
of the family's `directed` / `store` policy: the endpoints are already
distinguished by *level* rather than by coordinate order, so a sort
dedupes nothing (`A→B` and `B→A` live in different delta arrays at
different levels) and would only risk promoting a target-level endpoint
to source, flipping the anchor's scale factors. Their `perm_idx` is
therefore always `0`.

Core writes cross-level families with `directed=True`: fine→coarse
parenthood is data, not an undirected pair.

### Cross-level link emission

Two kwargs on `build_pyramid` control whether and how cross-pyramid-
level edges are materialised at each adjacent level pair:

```python
build_pyramid(
    "scan.zarrvectors",
    factors=[(2.0, 1.0), (2.0, 1.0)],
    cross_level_depth=2,                    # ±1, ±2 per applicable pair
    cross_level_storage="explicit",         # both +N and -N
)
```

**`cross_level_depth: int = 1`** controls how far the cross-level
emission reaches:

| Value | Meaning |
|-------|---------|
| `0`   | Disabled (same as `cross_level_storage="none"`) |
| `N`   | Materialise up to `±N` for every adjacent pair we can reach |
| `-1`  | Walk **all** available pyramid levels |

For `depth >= 2`, parents are composed across coarsening steps —
`grandparent[i] = parent_at_L1[parent_at_L0[i]]` — so a single
edge goes from a level-0 vertex straight to its level-2 metanode
(instead of forcing readers to chain two `+1` lookups).

**`cross_level_storage: Literal["none","implicit","explicit"]`**
controls direction:

| Mode | `+N` at fine level | `-N` at coarse level |
|------|--------------------|----------------------|
| `none`     | no  | no  |
| `implicit` | yes | no  (reconstruct on read by flipping `+N` at target) |
| `explicit` (default) | yes | yes |

`implicit` saves disk; `explicit` gives O(1) drill-down *and* drill-up
reads.

**Algorithm.** Emission happens in two stages.

*Stage 1 — adjacent `±1`, inline during coarsening.* Each
`coarsen_level` call emits its own `±1` families as it goes (see
[`_emit_inline_cross_level_links`](../../../zarr_vectors/multiresolution/coarsen.py)),
after the target level has been created:

1. Re-walks the source level in chunk-major order, re-bins each vertex,
   and looks up its metavertex to build a flat fine→coarse `parent[]`
   array. Orphaned fine vertices (`parent < 0`) are dropped.
2. Builds the trivial edge list `[(i, parent[i])]` and hands it to
   [`write_links`](../../../zarr_vectors/core/arrays.py) in global
   `(chunk_coords, vertex_idx)` form with `delta=+1`, `link_width=2`,
   `directed=True`.
3. `write_links` routes each record to the offsets array for the gap
   between its endpoints — anchor-projected through `anchor_chunk`
   using both levels' real chunk scales — so intra- and cross-chunk
   cases are one call with no hand-partitioning.
4. When `cross_level_storage="explicit"`, repeats the call at the
   coarse level with the endpoints swapped and `delta=-1`, letting
   `write_links` re-derive the offsets from the coarse grid.

*Stage 2 — deeper deltas, post-hoc.* After the coarsening loop,
`build_pyramid` calls
[`_finalize_cross_level_for_store`](../../../zarr_vectors/multiresolution/coarsen.py),
which:

1. Stamps the root cross-level metadata (always), then returns early if
   `cross_level_storage="none"` or `cross_level_depth == 0`.
2. Stamps the `CAP_MULTISCALE_LINKS` capability and decodes each
   adjacent pair's `+1` family back into a flat `parent[]` array
   (`_decode_parent_from_plus_one`, reading the **fine** level's
   `links/+1/` records — endpoint 0 is the fine side).
3. Returns without emitting anything more when `max_delta < 2`: `±1`
   was already written inline.
4. Otherwise composes parents step-by-step
   (`composed[i] = inter_parent[parent[i]]`) and emits `±2`, `±3`, …
   up to `cross_level_depth` through the same `write_links` path.

Because stage 1 runs inside `coarsen_level`, a **standalone**
`coarsen_level` call can emit `±1` too — pass `cross_level_storage=`
explicitly (it defaults to `"none"` there, unlike on `build_pyramid`).
Deltas of `±2` and beyond require `build_pyramid`, which is the only
caller of the finalize pass.

**Persistence.** Root metadata gains `cross_level_depth` and
`cross_level_storage`, and the `CAP_MULTISCALE_LINKS` capability
token is stamped on `format_capabilities` whenever any `delta != 0`
array is emitted. Readers MAY use the capability to short-circuit
walks of stores that don't carry cross-level edges at all.

See [`examples/07_multiscale_links.ipynb`](../../../examples/07_multiscale_links.ipynb)
for a walkthrough that builds a 3-level pyramid, inspects the
resulting `<delta>` subdirs, and reads both intra- and cross-level
edges.

### Validation

`build_pyramid` rejects, before writing anything:

| Condition | Raises |
|-----------|--------|
| `cross_level_storage` not in `{"none", "implicit", "explicit"}` | `ValueError` |
| `cross_level_depth < -1` | `ValueError` |
| `len(chunk_scale_factors) != len(factors)` | `ValueError` |
| A `factors[i]` that is not a 2-tuple | `ValueError` |

`coarsen_level` additionally rejects a `chunk_scale_factor` tuple whose
rank is not `sid_ndim`, or any multiplier `< 1`, with `CoarseningError`.

`factors=` is **required** — there is no auto-planning fallback.
Passing a non-default `method=` or `sparsity_strategy=` without
`zarr-vectors-tools` installed raises a clear error naming the package.
