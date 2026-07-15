# Point clouds

Point clouds are the simplest ZVF geometry type: a collection of spatial
positions with optional per-vertex scalar or vector attributes. They arise
in synchrotron absorption imaging (HiP-CT, micro-CT), single-molecule
localisation microscopy (STORM, PALM, MINFLUX), spatial transcriptomics
(Visium HD, Xenium, MERFISH), and lidar scanning.

This tutorial covers writing, reading, spatial querying, attribute handling,
ingesting from external formats, and building multi-resolution pyramids.
All examples on this page use only the core `zarr-vectors` API. Format
converters for LAS/PLY/CSV/XYZ live in the companion package
**`zarr-vectors-tools`**.

---

## Writing a point cloud

### Minimal write

```python
import numpy as np
from zarr_vectors.types.points import write_points

rng = np.random.default_rng(42)

# 100 000 points in a 1 000³ µm volume
positions = rng.uniform(0, 1000, (100_000, 3)).astype(np.float32)

write_points(
    "scan.zarrvectors",
    positions,
    chunk_shape=(200.0, 200.0, 200.0),   # one chunk file per 200³ µm
    bin_shape=(50.0, 50.0, 50.0),        # spatial index at 50³ µm
)
```

After writing, the store summary is:

```
geometry_type:  point_cloud
spatial_dims:   3
chunk_shape:    [200.0, 200.0, 200.0]
bin_shape:      [50.0, 50.0, 50.0]
0:   100000 vertices, 125 chunks
```

### Write with per-vertex attributes

Any number of named float or integer attribute arrays can be attached.
The arrays must have the same length as `positions` (one value per vertex):

```python
rng = np.random.default_rng(42)
n = 100_000
positions   = rng.uniform(0, 1000, (n, 3)).astype(np.float32)
intensity   = rng.uniform(0, 1, n).astype(np.float32)          # absorption
label       = rng.integers(0, 8, n).astype(np.int32)           # class label
rgb         = rng.integers(0, 256, (n, 3)).astype(np.uint8)    # colour
confidence  = rng.uniform(0.5, 1, n).astype(np.float32)

write_points(
    "scan.zarrvectors",
    positions,
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
    vertex_attributes={
        "intensity":  intensity,
        "label":      label,
        "color":      rgb,        # vector attribute: shape (N, 3)
        "confidence": confidence,
    },
)
```

`write_points` still accepts `attributes=` as a deprecated alias for
`vertex_attributes=`; it emits a `DeprecationWarning`. Passing both
raises `TypeError`.

There are no `coordinate_system` or `axis_units` arguments — those
concepts do not exist anywhere in this package.

### Choosing `chunk_shape` and `bin_shape`

A practical starting point: `chunk_shape` should be large enough that
each chunk contains ~10 000–100 000 vertices; `bin_shape` should be
roughly `chunk_shape / 4` per axis.

```python
# Estimate expected vertices per chunk
total_vertices    = 10_000_000
volume            = 4000 ** 3        # µm³
chunk_volume      = 500 ** 3         # µm³ with chunk_shape = 500
expected_per_chunk = total_vertices * chunk_volume / volume
# ≈ 7 812 — within the target range

write_points(
    "large_scan.zarrvectors",
    positions_10M,
    chunk_shape=(500.0, 500.0, 500.0),
    bin_shape=(125.0, 125.0, 125.0),   # 4×4×4 = 64 bins/chunk
)
```

---

## Reading a point cloud

### Read all data

```python
from zarr_vectors.types.points import read_points

result = read_points("scan.zarrvectors")

print(result["vertex_count"])                     # 100000
print(result["positions"].shape)                  # (100000, 3)
print(result["attributes"]["intensity"].shape)    # (100000,)
print(result["attributes"]["label"].dtype)        # int32
print(result["attributes"]["color"].shape)        # (100000, 3)
```

### Read a specific level

If the store has a multi-resolution pyramid, pass `level=N` to read a
coarser representation:

```python
coarse = read_points("scan.zarrvectors", level=1)
print(coarse["vertex_count"])     # fewer vertices — spatially coarsened
print(coarse["level"])            # 1
```

### Reading attributes

**Attributes are opt-in.** `read_points` loads *no* attribute data unless
you name what you want: the returned `vertex_attributes` dict is empty by
default. Name them with `attribute_names` (not `attributes`):

```python
result = read_points("scan.zarrvectors")
print(result["vertex_attributes"])      # {} — nothing loaded

result = read_points(
    "scan.zarrvectors",
    attribute_names=["intensity"],      # only load intensity
)
print(sorted(result["vertex_attributes"]))   # ['intensity']
assert "label" not in result["vertex_attributes"]
```

The result key is `vertex_attributes`, not `attributes`. `read_points`
returns exactly three keys:

```python
result = read_points("scan.zarrvectors", attribute_names=["intensity", "color"])
print(sorted(result))                              # ['positions', 'vertex_attributes', 'vertex_count']
print(result["positions"].shape)                   # (N, 3)
print(result["vertex_attributes"]["color"].shape)  # (N, 3) — shape preserved
```

Vector attributes keep their `(N, C)` shape through `read_points`. (The
lazy API flattens them — see
[Lazy loading](../multiscale/lazy_loading.md).)

---

## Spatial bounding-box queries

ZVF queries target individual bins — not full chunks — so the amount of
data loaded is proportional to the query volume, not the chunk volume.

```python
lo = np.array([100.0, 100.0, 100.0])
hi = np.array([200.0, 200.0, 200.0])

result = read_points(
    "scan.zarrvectors",
    bbox=(lo, hi),
)
print(result["vertex_count"])   # ≈ 100 (100³/1000³ × 100 000)
print(result["positions"].min(axis=0))  # all >= lo
print(result["positions"].max(axis=0))  # all <= hi
```

The query is exact: only vertices within the half-open interval
`[lo, hi)` per axis are returned.

### Combining bbox and level

A coarser level with bbox is the key pattern for overview-first rendering:
load a low-resolution overview of the full volume, then switch to the
finer level only for the region of interest.

```python
# Overview: coarse level, full volume
overview = read_points("scan.zarrvectors", level=2)

# Detail: full resolution, small region
detail = read_points(
    "scan.zarrvectors",
    level=0,
    bbox=(np.array([400., 400., 400.]),
          np.array([600., 600., 600.])),
)
```

---

## Multi-resolution pyramids

### Building a pyramid

```python
from zarr_vectors.multiresolution.coarsen import build_pyramid

build_pyramid(
    "scan.zarrvectors",
    factors=[(2.0, 1.00), (4.0, 1.00)],
)
```

Bin aggregation is fixed: source vertices collapse to their centroid.
There is no aggregation-mode parameter.

After building, the resolution summary is:

```
0:  100000 vertices  (bin_ratio 1×1×1)
1:  12890 vertices   (bin_ratio 2×2×2)
2:  1613 vertices    (bin_ratio 4×4×4)
```

### Anisotropic pyramids

For data with anisotropic resolution (e.g. 4×4×25 nm voxels), use an
anisotropic `bin_ratio` that coarsens proportionally in each axis:

```python
# Anisotropic: coarsen z 2× less than x,y (data is 6× coarser in z already)
build_pyramid(
    "aniso.zarrvectors",
    factors=[(2.0, 1.00), (4.0, 1.00)],
)
```

---

## Ingesting and exporting external formats

Format converters for LAS, PLY, CSV, and XYZ (and the `zarr-vectors`
CLI) live in the companion package **`zarr-vectors-tools`**.

---

## Validation

```python
from zarr_vectors.validate import validate

result = validate("scan.zarrvectors", level=5)
print(result.summary())
# Level 5 validation: PASS
#   42 passed, 0 warnings, 0 errors
```

---

## Common pitfalls

**`bin_shape` does not divide `chunk_shape`.**
The writer raises `ValueError` immediately. Check that
`chunk_shape[d] / bin_shape[d]` is an integer for every axis.

```python
# This will raise ValueError:
write_points("bad.zarrvectors", positions,
             chunk_shape=(200., 200., 200.),
             bin_shape=(60., 60., 60.))   # 200/60 = 3.33... ✗
```

**Writing float64 positions.**
ZVF stores positions as float32 by default. If your coordinates require
float64 precision (sub-nanometre accuracy at kilometre scale), pass
`dtype=np.float64` to preserve precision:

```python
write_points("precise.zarrvectors", positions_f64,
             chunk_shape=(200., 200., 200.),
             dtype=np.float64)
```

Be aware that float64 doubles storage size and reduces Blosc compression
ratio.

**Attribute array has wrong length.**
The write function checks that each attribute array has the same length as
`positions`. If you have a mismatch, check whether your data pipeline
dropped or duplicated rows.

**Reading a large store without bbox.**
`read_points` without `bbox` loads all vertices into memory. For stores
with > 50M vertices this may exhaust RAM. Use a bbox query or the lazy
API (see [Lazy loading](../multiscale/lazy_loading.md)).
