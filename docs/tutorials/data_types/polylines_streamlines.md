# Polylines and streamlines

Polylines and streamlines store ordered vertex sequences — paths through
3-D space. Use `polyline` for general ordered paths (vascular centrelines,
cell migration tracks, traced axons). Use `streamline` when the paths come
from MRI or synchrotron tractography and you need to store propagation
metadata (step size, seeding strategy, reference image).

The two types are technically identical in their on-disk arrays; the
distinction is in the `geometry_type` constant and the optional
tractography-specific metadata keys. All write/read functions in this
tutorial apply equally to both.

All examples on this page use only the core `zarr-vectors` API.
TRK/TCK/TRX converters live in the companion package
**`zarr-vectors-tools`**.

---

## Writing streamlines

### Minimal write

```python
import numpy as np
from zarr_vectors.types.polylines import write_polylines

rng = np.random.default_rng(0)

# 500 streamlines, each a random walk of 40 steps in a 200³ µm volume
streamlines = [
    rng.normal(0, 25, (40, 3)).cumsum(axis=0).astype(np.float32)
    for _ in range(500)
]

write_polylines(
    "tracts.zarrvectors",
    streamlines,
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
    geometry_type="streamline",    # declares tractography type
)
```

### Write with groups and attributes

Groups allow you to tag streamlines as belonging to named bundles or
experimental conditions. Per-streamline attributes (e.g. mean FA, length)
are stored in `object_attributes/`.

```python
n = 1000
streamlines = [
    rng.normal(0, 30, (rng.integers(20, 80), 3)).cumsum(0).astype(np.float32)
    for _ in range(n)
]

# Compute arc lengths for use as an attribute
lengths = np.array([
    np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1))
    for s in streamlines
], dtype=np.float32)

# Per-streamline FA (simulated)
mean_fa = rng.uniform(0.2, 0.8, n).astype(np.float32)

# Per-vertex FA (simulated) — one value per vertex along each streamline
per_vertex_fa = [rng.uniform(0.1, 0.9, len(s)).astype(np.float32)
                 for s in streamlines]

write_polylines(
    "tracts.zarrvectors",
    streamlines,
    chunk_shape=(200.0, 200.0, 200.0),
    bin_shape=(50.0, 50.0, 50.0),
    geometry_type="streamline",
    groups={
        0: list(range(300)),                # corticospinal tract
        1: list(range(300, 600)),           # arcuate fasciculus
        2: list(range(600, 1000)),          # uncinate fasciculus
    },
    object_attributes={
        "length":  lengths,
        "mean_fa": mean_fa,
    },
    vertex_attributes={
        "fa": per_vertex_fa,               # list of per-vertex arrays
    },
)
```

Three things to note:

- **Group keys are integers**, not names: `groups` is
  `{group_id: [polyline_indices]}`. Passing string keys such as
  `{"CST": [...]}` raises `TypeError`. A group whose members are
  contiguous may be passed as a `range` object, which is stored compactly.
  To attach human-readable names, use `group_attributes={"name": ...}`.
- The per-vertex keyword is `vertex_attributes`. `write_polylines` has no
  `attributes=` alias (unlike `write_points`, where it exists but is
  deprecated).
- There is no `streamline_metadata=` argument, and no place in the core
  API for free-form acquisition metadata such as step size or seeding
  strategy. That concept does not exist in this package.

---

## Reading streamlines

### Read all streamlines

```python
from zarr_vectors.types.polylines import read_polylines

result = read_polylines("tracts.zarrvectors")
print(result["polyline_count"])           # 1000
print(len(result["polylines"]))           # 1000 — list of (N_i, 3) arrays
print(result["polylines"][0].shape)       # (N_0, 3) — first streamline
```

### Read by object ID

Object IDs are stable integer identifiers assigned at write time (0-indexed):

```python
result = read_polylines("tracts.zarrvectors", object_ids=[0, 42, 99])
print(result["polyline_count"])            # 3
print(result["polylines"][1].shape)        # shape of streamline 42
```

### Read by group

```python
result = read_polylines("tracts.zarrvectors", group_ids=["CST"])
print(result["polyline_count"])            # 300

# Read from multiple groups at once
result = read_polylines("tracts.zarrvectors", group_ids=["CST", "AF"])
print(result["polyline_count"])            # 600
```

### Read object attributes without fetching vertices

For large stores where you only need the per-streamline metadata (e.g. to
filter by length before loading geometry):

```python
from zarr_vectors.core.store import open_store

root = open_store("tracts.zarrvectors", mode="r")
lengths  = root["0"]["object_attributes"]["length"][:]
mean_fa  = root["0"]["object_attributes"]["mean_fa"][:]

# Select long, high-FA streamlines
good_ids = np.where((lengths > 100) & (mean_fa > 0.4))[0]

# Now fetch only those streamlines
result = read_polylines("tracts.zarrvectors", object_ids=good_ids.tolist())
print(result["polyline_count"])
```

---

## Spatial bounding-box queries

A bbox query returns all streamlines that have **at least one vertex**
in the bounding box. The full streamline geometry is returned (not clipped
to the bbox), preserving path continuity.

```python
lo = np.array([-50.0, -50.0, -50.0])
hi = np.array([ 50.0,  50.0,  50.0])

result = read_polylines(
    "tracts.zarrvectors",
    bbox=(lo, hi),
)
print(result["polyline_count"])   # streamlines passing through the bbox
```

There is no `clip=` option — streamlines are returned whole, never split
at the bbox boundary.

### What `read_polylines` returns

`read_polylines` returns exactly three keys:

```python
result = read_polylines("tracts.zarrvectors")
print(sorted(result))            # ['polyline_count', 'polylines', 'vertex_count']
```

It has no `include_object_attributes=` option and returns neither an
`object_attributes` nor an `object_ids` key.

### Combining bbox and object attributes

A common analysis pattern is a spatial query followed by an attribute
filter. Object attributes are read separately, with
`read_object_attributes`, which returns a dense array indexed by object
ID:

```python
import numpy as np
from zarr_vectors.core.store import open_store, get_resolution_level
from zarr_vectors.core.arrays import read_object_attributes

level_group = get_resolution_level(open_store("tracts.zarrvectors", mode="r"), 0)

mean_fa = read_object_attributes(level_group, "mean_fa")   # (O,) by object id
high_fa_ids = np.nonzero(mean_fa > 0.5)[0]

# Then read just those streamlines — object_ids filtering works for polylines.
result = read_polylines("tracts.zarrvectors", object_ids=high_fa_ids.tolist())
print(f"{result['polyline_count']} high-FA streamlines")
```

Do **not** reach for object attributes through the lazy
`level.attributes[...]` accessor: that path is for *per-vertex* arrays
and silently returns an empty array for an object attribute rather than
raising.

---

## Multi-resolution pyramids

```python
from zarr_vectors.multiresolution.coarsen import build_pyramid

build_pyramid(
    "tracts.zarrvectors",
    factors=[(2.0, 1.00), (4.0, 4.00)],
)
```

Bin aggregation is fixed: source vertices collapse to their centroid.
There is no aggregation-mode parameter.

After building, the resolution summary looks like:

```
0:  1000 streamlines, ~40 000 vertices
1:  1000 streamlines, ~5 800 vertices (8× reduction)
2:  250 streamlines,  ~365 vertices   (64× × 4× = 256× total)
```

---

## Ingesting and exporting tractography formats

Format converters for TRK, TCK, and TRX (and the `zarr-vectors` CLI)
live in the companion package **`zarr-vectors-tools`**.

---

## Common pitfalls

**Links do not contain every edge of a streamline.**
Streamlines are `implicit_sequential`: within a chunk, consecutive
vertices are connected by *vertex order alone*, and no link record is
written for them. Only a transition that crosses a chunk boundary
becomes a stored link, under a non-zero offsets segment such as
`links/0/0.0.+1/`. The all-zero (intra-chunk) segment `links/0/0.0.0/`
is never even created for this type — there is no row that could live in
it.

Inspecting the links family therefore shows you the chunk-boundary
bridges, not the streamline. Use `read_polylines(object_ids=[k])` to
retrieve the complete vertex sequence.

**Streamlines entirely outside the bbox are not returned.**
A bbox query returns streamlines with at least one vertex inside the bbox.
A streamline that passes through the bbox interior but has no vertices
inside (because vertices are spaced far apart) will not be returned. Reduce
`step_size` or `bin_shape` relative to the streamline vertex spacing to
ensure all passing streamlines are captured.

**Group IDs are integers everywhere — names are not resolved.**
`groups={...}` takes integer keys at write time, and `group_ids=[...]`
takes integers at read time. Passing a string name to either fails:
`write_polylines(groups={"CST": ...})` raises `TypeError`, and
`read_polylines(group_ids=["CST"])` raises
`TypeError: '<' not supported between instances of 'str' and 'int'`.

Names can be *stored* alongside groups with
`group_attributes={"name": ...}` (they land in `groupings_attributes/`),
but nothing in the reader maps a name back to an ID. Keep your own
name→ID mapping and pass integers:

```python
GROUPS = {"CST": 0, "AF": 1, "UF": 2}
result = read_polylines("tracts.zarrvectors", group_ids=[GROUPS["CST"]])
```

**Lost attributes after rechunking.**
Rechunking reorders vertices within each chunk. Per-vertex attribute arrays
are reordered with the same permutation automatically. However, if you
manually wrote attribute arrays without going through the write functions,
the reordering will not be applied and attributes will be misaligned after
rechunking. Always use the provided write API.
