# Streamline (`streamline`)

## Terms

**Streamline**
: A polyline produced by a tractography algorithm, tracing an estimated
  fibre pathway through a diffusion MRI or other orientation field.

**`GEOM_STREAMLINE`**
: The geometry type constant `"streamline"`, stored in the `geometry_types`
  list of the root attributes.

## Introduction

`streamline` uses the [`polyline`](polyline.md) layout unchanged: ordered
vertex runs, one object per streamline, cross-chunk links between runs,
optional groups for bundles. The geometry type tells readers that the paths
are tractography output. `write_polylines` writes `streamline` unless told
otherwise.

The format reserves no root keys for tractography provenance (step size,
seeding, algorithm, software). Per-streamline values go in
`object_attributes/` and per-point values (FA, MD) in `vertex_attributes/`.
Physical units belong to the axes in `multiscales`; see
[Root metadata](../layout/root_metadata.md).

TRK, TCK and TRX converters are in the companion package
[zarr-vectors-tools](https://zarr-vectors-tools.readthedocs.io/en/latest/convert/streamlines.html).
[TRX format comparison](../comparisons/trx_format.md) maps TRX fields onto
this layout.

## Technical reference

### Arrays present

The same as [`polyline`](polyline.md#arrays-present). In a typical
tractography store:

| Path | Holds |
|------|-------|
| `vertex_attributes/<name>/` | Per-point values such as FA |
| `object_attributes/<name>/` | Per-streamline values such as mean FA, length, step size |
| `groups/` | Bundle membership |

### Root attributes

A streamline store has no type-specific root keys. The writer sets these
values in the `zarr_vectors` block:

```json
{
  "geometry_types": ["streamline"],
  "links_convention": "implicit_sequential",
  "object_index_convention": "standard",
  "cross_chunk_strategy": "explicit_links"
}
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
    object_attributes={
        "mean_fa": np.array([f.mean() for f in fa], dtype=np.float32),
        "step_size": np.full(500, 0.5, dtype=np.float32),
    },
)
print(summary)
# {'polyline_count': 500, 'vertex_count': 24426, 'chunk_count': 103,
#  'cross_chunk_link_count': 2890, 'group_count': 0}

result = read_polylines("tracts.zarrvectors")
print(result["polyline_count"], result["vertex_attributes"]["fa"].shape)   # 500 (24426,)
```

`read_polylines` and its return value are described under
[Polyline](polyline.md#write-and-read-api).

### Validation

The same as [`polyline`](polyline.md#validation). No level checks
tractography metadata.
