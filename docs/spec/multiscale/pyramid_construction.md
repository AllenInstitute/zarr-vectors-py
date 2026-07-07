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

**Aggregation mode** (`agg_mode`)
: How per-vertex / per-object attributes are combined within a bin
  when producing the metanode's attribute value. Options:
  `mean` (default), `sum`, `mode`, `count`, `min`, `max` —
  see [`constants.VALID_AGGREGATIONS`](../../../zarr_vectors/constants.py).

**Coarsening method**
: How the writer reconciles object identity across levels.  Core
  provides `per_object` (default; OID-stable, metavertices may be
  shared).  Other methods (e.g. `cross_object_metanode` / `grid_metanode`,
  fresh OID space per level) are provided by `zarr-vectors-tools` and
  selected by name via `method=`.

**Cross-level link**
: An edge from a fine-level vertex to its coarse-level parent
  metanode. Materialised as a separate array family at each level
  (`links/<delta>/`, `cross_chunk_links/<delta>/`) — see
  [Links and cross-chunk links](../object_model/cross_chunk_links.md).

---

## Introduction

Building a resolution pyramid means constructing one or more coarser
representations of the base-level data. Coarsening operates
independently on two axes:

- **Spatial coarsening** — merge vertices in the same supervoxel bin
  into a single metanode (per the `coarsen_factor`).
- **Object thinning** — drop a fraction of discrete objects (per the
  `sparsity_factor`).

In 0.4, every coarsening pass can additionally emit **cross-pyramid-
level links** that record which fine vertex maps to which coarse
metanode. Cross-level links live alongside the per-level intra-chunk
edges under `links/<delta>/` and `cross_chunk_links/<delta>/`; see
[Links and cross-chunk links](../object_model/cross_chunk_links.md)
for the on-disk layout and
[`examples/07_multiscale_links.ipynb`](../../../examples/07_multiscale_links.ipynb)
for a walkthrough.

This page documents the coarsening algorithm per geometry type, the
two public APIs (`build_pyramid` and `coarsen_level`), and the
cross-level link options.

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
metanode_attribute = agg_mode(attributes in bin)
```

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

After both stages, the per-level intra-chunk `links/0/` and
`cross_chunk_links/0/` arrays are recomputed at the coarser level
(metanode positions may fall in different chunks than the originals).

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
deeper pyramids by chaining `coarsen_level` calls. Cross-level
emission for individual `coarsen_level` calls is **not** automatic;
to materialise `±delta` arrays, call `build_pyramid(..., factors=...)`
instead (which post-processes the whole pyramid in one pass via
`_finalize_cross_level_for_store`).

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

Set globally via `agg_mode=` on `build_pyramid` or `coarsen_level`.
Applies to every per-vertex / per-object attribute:

| `agg_mode` | Description | Use case |
|------------|-------------|----------|
| `mean`     | Mean of values in bin | Continuous scalars (FA, intensity, concentration) |
| `sum`      | Sum of values in bin | Counts, densities |
| `mode`     | Most frequent value | Categorical / label attributes |
| `count`    | Number of source vertices in bin | Density tracking |
| `min` / `max` | Bin extrema | Thresholds, peak activations |

Canonical token set:
[`zarr_vectors.constants.VALID_AGGREGATIONS`](../../../zarr_vectors/constants.py).

### Cross-level link emission (0.4+)

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

**Algorithm.** After all coarser levels have been written by the
normal coarsening loop, `build_pyramid` calls
[`_finalize_cross_level_for_store`](../../../zarr_vectors/multiresolution/coarsen.py)
which:

1. Walks every adjacent `(fine, coarse)` level pair.
2. Reconstructs the fine→parent map from the coarse level's
   `cross_chunk_links/<delta=-1>/` records (each record pairs a
   coarse metanode to one of its fine children).
3. Builds the trivial edge list `[(i, parent[i]) for i in range(n_fine)]`.
4. Partitions via
   [`partition_cross_level_edges`](../../../zarr_vectors/spatial/boundary.py)
   into chunk-aligned (`links/+delta/<chunk>`) and cross-chunk
   (`cross_chunk_links/+delta/data`) buckets.
5. Writes the `+delta` arrays at the fine level; if
   `cross_level_storage="explicit"`, also writes the swapped-endpoint
   `-delta` arrays at the coarse level.
6. For `depth >= 2`, composes parents as above and emits `±2`, `±3`,
   … in the same way.

**Persistence.** Root metadata gains `cross_level_depth` and
`cross_level_storage`, and the `CAP_MULTISCALE_LINKS` capability
token is stamped on `format_capabilities` whenever any `delta != 0`
array is emitted. Readers MAY use the capability to short-circuit
walks of stores that don't carry cross-level edges at all.

See [`examples/07_multiscale_links.ipynb`](../../../examples/07_multiscale_links.ipynb)
for a walkthrough that builds a 3-level pyramid, inspects the
resulting `<delta>` subdirs, and reads both intra- and cross-level
edges.

### Legacy automatic pyramid

`build_pyramid` with no `factors=` and no `level_configs=` falls back
to a legacy auto-planner driven by `target_volume_reduction`,
`sparsity_weight`, `reduction_factor`, `max_levels`, and
`min_vertices`:

```python
build_pyramid("scan.zarrvectors")               # legacy: auto-select levels
```

Convenient for exploratory use; for production pipelines prefer
explicit `factors=`.
