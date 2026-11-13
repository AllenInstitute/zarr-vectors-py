# Line (`line`)

## Terms

**Line segment**
: A pair of vertices `(A, B)` joined by a straight edge. Each segment is
  one object. The type constant is `GEOM_LINE`, `"line"`.

**Split segment**
: A segment whose endpoints are in different chunks.

## Introduction

The `line` type stores independent two-point segments, such as contact
sites, connectors or distance annotations. Each segment has its own
manifest, so segments can be read by ID. For paths of any length, use
[`polyline`](polyline.md). What every type shares is on
[Geometry types](index.md#common-to-every-type).

## Technical reference

### Arrays present

At level 0 of a store written by `write_lines`:

| Path | Present | Description |
|------|---------|-------------|
| `object_index/` | Always | One manifest per segment |
| `links/0/<offsets>/` | When a segment is split | One record per split segment |
| `vertex_attributes/<name>/` | With `vertex_attributes=` | One row per endpoint |
| `object_attributes/<name>/` | With `object_attributes=` | One row per segment |

| Case | Fragments | Manifest | Link |
|------|-----------|----------|------|
| Both endpoints in one chunk | One fragment holding both vertices | `[(c, f)]` | None |
| Endpoints in chunks `c_a` and `c_b` | One single-vertex fragment in each chunk | `[(c_a, f_a), (c_b, f_b)]` | One record in `links/0/<offsets>/` |

Within each chunk, fragments are in segment order. The root attributes:

```javascript
{"geometry_types": ["line"], "links_convention": "implicit_sequential", ...}
```

### Write and read API

```python
import numpy as np
from zarr_vectors.building import write_lines
from zarr_vectors.types.lines import read_lines

rng = np.random.default_rng(0)
starts = rng.uniform(0, 500, (1000, 3))
endpoints = np.stack([starts, starts + rng.normal(0, 20, (1000, 3))], axis=1).astype(np.float32)
strength = rng.uniform(0, 1, 1000).astype(np.float32)

summary = write_lines(
    "contacts.zarrvectors",
    endpoints,                     # (N, 2, D): one row per segment
    chunk_shape=(100.0, 100.0, 100.0),
    object_attributes={"strength": strength},
)
print(summary)
# {'line_count': 1000, 'chunk_count': 192, 'cross_chunk_count': 388}

result = read_lines("contacts.zarrvectors", object_ids=[0, 5, 42])
print(result["line_count"], result["endpoints"].shape)   # 3 (3, 2, 3)
```

`cross_chunk_count` is the number of split segments. `read_lines` returns
`endpoints` `(M, 2, D)`, `vertex_attributes` and `line_count`. A `bbox=`
read keeps a segment when either endpoint is inside the box.

### Validation

Only the [common checks](index.md#validation) apply.
