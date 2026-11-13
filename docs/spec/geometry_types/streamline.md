# Streamline (`streamline`)

## Terms

**Streamline**
: A polyline from a tractography algorithm, tracing an estimated fibre
  pathway through a diffusion MRI or other orientation field. The type
  constant is `GEOM_STREAMLINE`, `"streamline"`.

## Introduction

`streamline` uses the [`polyline`](polyline.md) layout unchanged; the type
tells readers the paths are tractography output. `write_polylines` writes
`streamline` by default, and `read_polylines` reads it.

The format has no root keys for tractography provenance. Store
per-streamline values (mean FA, length, step size) in `object_attributes/`,
per-point values (FA, MD) in `vertex_attributes/`, and bundles in
`groups/`. Units belong to the axes in `multiscales`.

TRK, TCK and TRX converters are in
[zarr-vectors-tools](https://zarr-vectors-tools.readthedocs.io/en/latest/convert/streamlines.html).
[TRX format comparison](../../explanation/comparisons/trx_format.md) maps
TRX fields onto this layout.

## Technical reference

### Arrays present

As for [`polyline`](polyline.md#arrays-present). The root attributes:

```javascript
{"geometry_types": ["streamline"], "links_convention": "implicit_sequential", ...}
```

### Write and read API

```python
import numpy as np
from zarr_vectors.building import write_polylines
from zarr_vectors.types.polylines import read_polylines

rng = np.random.default_rng(0)
streamlines = [
    (rng.uniform(100, 400, 3) + rng.normal(0, 5, (rng.integers(20, 80), 3)).cumsum(0)).astype(np.float32)
    for _ in range(500)
]
fa = [rng.uniform(0.2, 0.8, len(s)).astype(np.float32) for s in streamlines]

summary = write_polylines(
    "tracts.zarrvectors",
    streamlines,
    chunk_shape=(100.0, 100.0, 100.0),
    vertex_attributes={"fa": fa},                       # one array per streamline
    object_attributes={"mean_fa": np.array([f.mean() for f in fa], dtype=np.float32)},
)
print(summary)
# {'polyline_count': 500, 'vertex_count': 24426, 'chunk_count': 103,
#  'cross_chunk_link_count': 2890, 'group_count': 0}

result = read_polylines("tracts.zarrvectors")
print(result["polyline_count"], result["vertex_attributes"]["fa"].shape)   # 500 (24426,)
```

### Validation

As for [`polyline`](polyline.md#validation). No level checks tractography
metadata.
