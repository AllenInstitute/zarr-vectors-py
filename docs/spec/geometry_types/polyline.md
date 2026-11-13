# Polyline (`polyline`)

## Terms

**Polyline**
: An ordered sequence of vertices joined by consecutive edges,
  `V_0 — V_1 — … — V_{n-1}`. Each polyline is one object, identified by its
  index in the input list. The type constant is `GEOM_POLYLINE`,
  `"polyline"`.

**Run**
: The consecutive vertices of one polyline that lie in one chunk. Each run
  is one fragment.

**Cross-chunk link**
: A link record joining the last vertex of one run to the first vertex of
  the next run, in another chunk.

## Introduction

The `polyline` type stores ordered paths of any length. Each path can be
read by ID, put in groups and given attributes.

Within a run, consecutive vertices are joined by definition, so no link
record is stored. A polyline that crosses chunk boundaries *n* times has
*n + 1* runs, listed in path order in its
[manifest](../object_model/object_manifest.md), and *n* cross-chunk links.
[`streamline`](streamline.md) uses the same layout. What every type shares
is on [Geometry types](index.md#common-to-every-type).

## Technical reference

### Arrays present

At level 0 of a store written by `write_polylines`:

| Path | Present | Description |
|------|---------|-------------|
| `vertices/` | Always | Each run in path order |
| `object_index/` | Always | One manifest per polyline |
| `links/0/<offsets>/` | When a polyline crosses a chunk | Cross-chunk links |
| `vertex_attributes/<name>/` | With `vertex_attributes=` | |
| `object_attributes/<name>/` | With `object_attributes=` | |
| `fragment_attributes/<name>/` | With `fragment_attributes=` | One row per run |
| `groups/` | With `groups=` | Group ID → object IDs |
| `group_attributes/<name>/` | With `group_attributes=` | One row per group |

The root attributes:

```javascript
{"geometry_types": ["polyline"], "links_convention": "implicit_sequential", ...}
```

### Write and read API

```python
import numpy as np
from zarr_vectors.building import write_polylines
from zarr_vectors.types.polylines import read_polylines

rng = np.random.default_rng(0)
polylines = [
    (rng.uniform(100, 400, 3) + rng.normal(0, 5, (rng.integers(10, 60), 3)).cumsum(0)).astype(np.float32)
    for _ in range(100)
]

summary = write_polylines(
    "paths.zarrvectors",
    polylines,
    chunk_shape=(100.0, 100.0, 100.0),
    geometry_type="polyline",          # the default is "streamline"
    groups={0: list(range(50)), 1: list(range(50, 100))},
)
print(summary)
# {'polyline_count': 100, 'vertex_count': 3180, 'chunk_count': 63,
#  'cross_chunk_link_count': 344, 'group_count': 2}

result = read_polylines("paths.zarrvectors", group_ids=[0])
print(result["polyline_count"])                    # 50
first = np.concatenate(result["polylines"][0])     # join the runs
print(len(result["polylines"][0]), np.array_equal(first, polylines[0]))   # 3 True
```

`read_polylines` returns:

| Key | Type | Description |
|-----|------|-------------|
| `polylines` | `list[list[ndarray]]` | Per polyline, its runs in path order |
| `object_ids` | `ndarray` | Object ID of each returned polyline |
| `vertex_attributes` | `dict[str, ndarray]` | Per-vertex attributes, concatenated over the result |
| `ndim` | `int` | Spatial dimensionality |
| `polyline_count`, `vertex_count` | `int` | Sizes of the result |

A `bbox=` read returns the whole of every polyline with a run in a chunk
the box touches. A `chunks=` read returns only the runs in the named
chunks, each unbroken stretch as its own polyline, so `polyline_count` can
exceed the number of objects.

### Validation

Only the [common checks](index.md#validation) apply. No level checks that
a polyline's runs join up.
