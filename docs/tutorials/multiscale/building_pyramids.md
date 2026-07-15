# Building multi-resolution pyramids

A multi-resolution pyramid stores the same dataset at progressively
coarser spatial resolutions. Viewers and analysis pipelines select the
appropriate level based on viewport size, memory budget, or query
scale — loading only the data density they need.

This tutorial covers pyramid construction for point clouds,
streamlines, and graphs; manual single-level coarsening; and the 0.4
cross-pyramid-level link materialisation. See the
[Pyramid construction](../../spec/multiscale/pyramid_construction.md)
spec page for the algorithm and on-disk layout.

---

## Concepts recap

**Coarsen factor** scales the supervoxel bin size at each level. A
factor of `2.0` means each axis is binned 2× coarser, giving up to
~8× fewer metanodes per unit volume in 3D.

**Sparsity factor** thins discrete objects (streamlines, skeletons,
meshes) at coarser levels. A factor of `3.0` keeps every third object
on average. `1.0` (the default) keeps all objects.

**Aggregation** is not configurable. Coarsening aggregates a bin's
source vertices into their **centroid** (a metavertex); there is no
`agg_mode`-style choice of `mean` / `sum` / `mode` / `min` / `max`.

**Coarsening method** (`method`) selects the coarsening implementation.
Core ships exactly one: `per_object` (the default) — OID-stable, with
metavertices shared between objects. Any other name must be registered by
**`zarr-vectors-tools`**; core raises a clear `ValueError` if it is not
installed. See the spec page for the trade-off.

**Cross-level links** are edges from a fine-level vertex to its
coarse-level parent metanode, materialised at each adjacent level pair.
Controlled by `cross_level_depth` and `cross_level_storage`. See
[`examples/07_multiscale_links.ipynb`](../../../examples/07_multiscale_links.ipynb)
for a worked example.

---

## Point cloud pyramids

Point clouds use spatial coarsening only; `sparsity_factor` is always
ignored.

### Quick three-level pyramid

```python
import numpy as np
from zarr_vectors.types.points import write_points
from zarr_vectors.multiresolution.coarsen import build_pyramid

rng = np.random.default_rng(0)
positions  = rng.uniform(0, 2000, (500_000, 3)).astype(np.float32)
intensity  = rng.uniform(0, 1, 500_000).astype(np.float32)
label      = rng.integers(0, 16, 500_000).astype(np.int32)

write_points(
    "synchrotron.zarrvectors",
    positions,
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
    attributes={"intensity": intensity, "label": label},
)

build_pyramid(
    "synchrotron.zarrvectors",
    factors=[
        (2.0, 1.0),         # level 1: 8× vertex reduction in 3D
        (2.0, 1.0),         # level 2: another 8× reduction
        (2.0, 1.0),         # level 3
    ],
)
```

After building, the resolution summary is roughly:

```
0:  500000 vertices
1:   63000 vertices   (8×  reduction)
2:    8000 vertices   (64× reduction)
3:    1000 vertices   (512× reduction)
```

Actual counts at each level are less than `500000 / 8^N` because bins
near the data boundary contain fewer vertices to merge.

### Choosing coarsen factors

Each `(coarsen, sparsity)` tuple controls one level. For 3D data the
per-level vertex reduction is approximately `coarsen ** 3`. A balanced
pyramid uses the same factor at every level:

| Per-level target reduction | `coarsen_factor` | 3D effect |
|----------------------------|------------------|-----------|
| 8×    | 2.0 | each axis halved |
| 27×   | 3.0 | each axis thirded |
| 64×   | 4.0 | each axis quartered |

`factors` is isotropic: each entry scales every axis by the same
`coarsen_factor`. For finer control, call `coarsen_level` per level
yourself — it takes an explicit `source_level` / `target_level` pair
plus `coarsen_factor`, `sparsity_factor`, and a `chunk_scale_factor`
that may be a per-axis tuple:

```python
from zarr_vectors.multiresolution.coarsen import coarsen_level

coarsen_level(
    "scan.zarrvectors",
    source_level=0,
    target_level=1,
    coarsen_factor=2.0,
    chunk_scale_factor=(2, 2, 1),   # per-axis
)
```

`build_pyramid` also accepts `chunk_scale_factors=` — one entry per
level, each a scalar or a per-axis tuple — if you only need to vary the
chunk shape rather than the coarsening itself.

### A note on attributes

There is no per-attribute aggregation setting, and no `agg_mode`
parameter on either `build_pyramid` or `coarsen_level`. Every bin's
vertices collapse to a centroid metavertex under the one built-in
`per_object` method. If you need categorical labels or counts handled
differently, that behaviour is not in core.

---

## Streamline / polyline pyramids

Polylines and streamlines use both spatial coarsening (vertex
metanodes) and object thinning (dropping individual streamlines at
coarser levels).

### Two-stage pyramid with increasing thinning

```python
import numpy as np
from zarr_vectors.types.polylines import write_polylines
from zarr_vectors.multiresolution.coarsen import build_pyramid

rng = np.random.default_rng(0)
streamlines = [
    rng.normal(0, 30, (rng.integers(30, 120), 3)).cumsum(0).astype(np.float32)
    for _ in range(10_000)
]

write_polylines(
    "tracts.zarrvectors",
    streamlines,
    chunk_shape=(100.0, 100.0, 100.0),
    bin_shape=(25.0, 25.0, 25.0),
    geometry_type="streamline",
)

build_pyramid(
    "tracts.zarrvectors",
    factors=[
        (2.0, 1.0),     # L1: 8× fewer vertices, all streamlines kept
        (2.0, 4.0),     # L2: another 8× + drop 3/4 of streamlines
        (2.0, 4.0),     # L3: another 8× + drop another 3/4
    ],
    method="per_object",            # keep OIDs stable across levels
    sparsity_strategy="random",
    sparsity_seed=42,
)
```

Expected output:

```
0:  10000 streamlines, ~750 000 vertices
1:  10000 streamlines, ~96 500 vertices   (8× vertex reduction)
2:   2500 streamlines, ~3 100 vertices    (8× × 4× thinning)
3:    625 streamlines, ~155 vertices
```

`per_object` preserves OIDs: a streamline kept at level 3 has the same
OID it had at level 0, with each surviving level holding the
appropriate coarser metanode trajectory.

### Sparsity strategies

`sparsity_strategy` picks which objects survive at each level. **Core
ships exactly one strategy: `"random"`** (the default).

```python
build_pyramid(
    "tracts.zarrvectors",
    factors=[(2.0, 1.0), (2.0, 4.0)],
    sparsity_strategy="random",
    sparsity_seed=42,
)
```

Non-random strategies — spatial-coverage, length-ranked,
attribute-ranked, point-thinning — live in **`zarr-vectors-tools`**,
which registers them on import. Requesting one without that package
installed raises a `ValueError` naming the strategy and telling you to
install the tools package; it does not silently fall back to `random`.

```python
# Requires `pip install zarr-vectors-tools`
build_pyramid(
    "tracts.zarrvectors",
    factors=[(2.0, 1.0), (2.0, 4.0)],
    sparsity_strategy="spatial_coverage",
    sparsity_seed=42,
)
```

The same applies to `method=`: `"per_object"` is core, anything else is
tools-registered.

---

## Cross-pyramid-level link materialisation (0.4+)

`build_pyramid` materialises edges between fine vertices and their
coarse-level parent metanodes. These are stored under
`links/<delta>/<offsets>/` at every adjacent level pair — one family,
whether or not an edge's endpoints share a chunk. See
[Links](../../spec/object_model/links.md) for the on-disk layout.

### Default: ±1 explicit

```python
build_pyramid(
    "scan.zarrvectors",
    factors=[(2.0, 1.0), (2.0, 1.0)],
    cross_level_depth=1,                  # default
    cross_level_storage="explicit",       # default
)
```

This emits, at every adjacent (fine, coarse) pair:

- `links/+1/<offsets>/` at the fine level — the drill-up edges. Edges
  whose target metanode shares the source chunk_key land in the all-zero
  segment (`links/+1/0.0.0/`); those whose target sits in a different
  chunk_key land in the segment naming that displacement (e.g.
  `links/+1/0.0.+1/`).
- `links/-1/<offsets>/` at the coarse level — the same edges with
  endpoints swapped (drill-down direction).

Both directions are one family per delta. There is no separate
cross-chunk array: the `<offsets>` segment *is* how a cross-chunk edge is
distinguished from an intra-chunk one.

### Storage modes

| Mode | `+N` at fine level | `-N` at coarse level |
|------|--------------------|----------------------|
| `none`     | no  | no  |
| `implicit` | yes | no (readers reconstruct by flipping `+N`) |
| `explicit` | yes | yes |

Use `"implicit"` to halve disk usage; readers will need to find the
`+N` array at the target level and flip endpoints to drill down.

```python
build_pyramid(
    "scan.zarrvectors",
    factors=[(2.0, 1.0), (2.0, 1.0)],
    cross_level_storage="implicit",
)
```

### Multi-step deltas

```python
build_pyramid(
    "scan.zarrvectors",
    factors=[(2.0, 1.0), (2.0, 1.0), (2.0, 1.0)],     # 4 levels total
    cross_level_depth=2,                              # emit ±1 AND ±2
    cross_level_storage="explicit",
)
```

`depth=2` composes parent maps across two coarsening steps so a
level-0 vertex points straight to its level-2 metanode (single hop, no
chained lookup). Pass `cross_level_depth=-1` to walk all available
adjacent and skip-one pairs.

### Opting out

```python
build_pyramid(
    "scan.zarrvectors",
    factors=[(2.0, 1.0), (2.0, 1.0)],
    cross_level_depth=0,                  # no <delta != 0> arrays
)
```

Use `cross_level_depth=0` when downstream consumers don't need
drill-up/drill-down navigation — saves disk and a small post-build
pass.

See [`examples/07_multiscale_links.ipynb`](../../../examples/07_multiscale_links.ipynb)
for a notebook walkthrough of reading both intra-level and cross-level
arrays at each `<delta>`.

---

## Manual single-level coarsening

For fine-grained control over individual levels, use `coarsen_level`
directly:

```python
from zarr_vectors.multiresolution.coarsen import coarsen_level

coarsen_level(
    "synchrotron.zarrvectors",
    source_level=3,
    target_level=4,
    coarsen_factor=2.0,
    sparsity_factor=1.0,
    method="per_object",
    sparsity_strategy="random",
    sparsity_seed=42,
)
```

`source_level` does not have to be 0 — chain `coarsen_level` calls to
build pyramids one level at a time. **Note:** `coarsen_level` does
*not* emit cross-level link arrays on its own; only `build_pyramid`
runs the post-build `_finalize_cross_level_for_store` step. To
materialise `<delta>` arrays after a sequence of manual `coarsen_level`
calls, call `build_pyramid(..., factors=[(1.0, 1.0)])` once at the end
to trigger the finalize pass.

### Listing existing levels

```python
from zarr_vectors.core.store import open_store, list_resolution_levels

root = open_store("synchrotron.zarrvectors", mode="r")
print(list_resolution_levels(root))      # [0, 1, 2, 3]
```

### Removing a level

```python
from zarr_vectors.core.store import remove_resolution_level

root = open_store("synchrotron.zarrvectors", mode="r+")
remove_resolution_level(root, level_index=4)
```

---

## Reading pyramid levels

```python
from zarr_vectors.types.points import read_points

# Read each level and compare vertex counts
for level in range(4):
    result = read_points("synchrotron.zarrvectors", level=level)
    print(f"Level {level}: {result['vertex_count']:>8d} vertices")
```

Reading a specific level with a bounding box:

```python
import numpy as np

# Quick overview: coarsest level, full volume
overview = read_points("synchrotron.zarrvectors", level=3)

# Drill-down: finest level, region of interest
detail = read_points(
    "synchrotron.zarrvectors",
    level=0,
    bbox=(np.array([400., 400., 400.]),
          np.array([600., 600., 600.])),
)
```

---

## Performance tips

**Build levels from finest to coarsest.** `build_pyramid` does this
automatically — each level coarsens from the previous one, not from
level 0, so per-level work decreases as the pyramid grows.

**Skip cross-level emission when not needed.** If downstream consumers
don't navigate between levels, pass `cross_level_depth=0` to skip the
post-build pass.

**`method="per_object"` is the only core method, and it is OID-stable.**
It is what you want when you need to track "the same object" across
resolution levels (Neuroglancer drill-down, ID-preserving analytics),
and it is the default. Alternative methods exist only if
**`zarr-vectors-tools`** is installed to register them.

**Build pyramids on the same machine as the store.** For cloud stores
(S3 / GCS), run `build_pyramid` from a VM in the same region as the
bucket — pyramid building is I/O-bound on cloud, and same-region
latency is ~10× lower than from a laptop.
