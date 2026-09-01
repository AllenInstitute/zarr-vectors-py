# Directory structure

## Terms

**Store root**
: The top-level directory (or object-store prefix) of a ZVF store.
  Conventionally named with a `.zarrvectors` extension. Contains the root
  `zarr.json`, root `.zattrs`, all resolution level groups, and
  `metadata.json`.

**Resolution level group**
: A Zarr group at path `<N>/` within the store root, where `N`
  is a non-negative integer. Level 0 is always the full-resolution level.
  Higher levels are progressively coarser.

**Per-chunk array**
: Every per-spatial-chunk array (`vertices/`, `vertex_fragments/`,
  `links/<delta>/<offsets>/`, `vertex_attributes/<name>/`, …) is a
  **single** Zarr v3 vlen-bytes array whose shape is the level's chunk
  grid. One cell holds one spatial chunk's payload bytes; the chunk files
  live under the `c/` sub-tree (`c/i/j/k`). A spatial chunk at absolute
  coord `c` maps to cell `c - origin`, where
  `origin = floor(min_corner / chunk_shape)` is stored in the array's
  `chunk_grid_origin` attribute (absent ⇒ zero origin). The set of
  non-empty cells is listed in the array's `nonempty_chunks` attribute.

  This pattern is now **universal — there is no exception.** Before 0.9.0
  `cross_chunk_links/<delta>/` was one: its cells were keyed by
  endpoint-chunk tuples rather than a spatial grid, so each cell was its
  own small array under a group. That family is gone. Connectivity is a
  single family whose arrays are ordinary rank-D grids over the chunk
  grid, one cell per **source** chunk — so every per-chunk array in a
  store now shares the same shape, the same `c/i/j/k` key scheme, and the
  same sharding behaviour. See [Links](../object_model/links.md).

**Family group**
: `links/<delta>/` and `link_attributes/<name>/<delta>/` are **groups**,
  not arrays. Their children are one per-chunk array per distinct
  relative-offset segment (`links/<delta>/<offsets>/`). The group carries
  the family-wide policy (`link_width`, `sid_ndim`, `directed`, `store`);
  the arrays carry only what decodes their own cells.

**`metadata.json`**
: A plain-text JSON file at the store root containing human-readable
  summary information about the store (total vertex count, bounding box,
  creation timestamp). Not used by the read/write API; present for
  inspection and provenance.

**`parametric/`**
: An optional sub-group at the store root for algebraic (non-vertex-based)
  geometry objects such as planes, spheres, and ellipsoids. Not chunked
  spatially; stores a single array of object parameter tuples.

---

## Introduction

The on-disk layout of a ZVF store follows a clear hierarchy: store root →
resolution levels → array groups → chunk files. Every path in the hierarchy
has a defined meaning; there are no opaque binary blobs. This page documents
every node in the tree for each supported geometry type.

Understanding the directory structure is essential for contributors
implementing new geometry types, validation tools, or custom readers. It is
also useful for debugging: if a store fails validation, the first step is
often to inspect the directory tree directly.

```{figure} ../../_static/figures/zarr-vectors-store-structure.png
:alt: Panel a, the directory tree of a three-level Zarr Vectors store with the base level expanded into vertices, vertex_attributes, links by level and chunk offset, vertex_fragments, link_fragments, object_index and groups. Panel b, vertices and links drawn across levels 0, 1 and 2, showing links that stay within a chunk, links carrying a chunk offset, links carrying a level offset, and objects split into fragments across chunks.
:width: 100%
:name: fig-store-structure
:figclass: zv-figure

**Zarr Vectors data structure.** How vector items are divided across the Zarr
arrays of a store. **a** — Directory structure of an example store containing
three levels, with the array structure of the base level expanded. **b** —
Items a store can represent: vertices, links and their offsets between chunks
and between resolution levels, and objects and object fragments, across the
same three-level store.
```

---

## Technical reference

### Full annotated tree (point cloud)

```
dataset.zarrvectors/
│
├── zarr.json                    # Zarr v3 root group metadata
├── .zattrs                      # ZVF root metadata (see root_metadata.md)
├── metadata.json                # human-readable summary
│
├── 0/                # full-resolution level
│   ├── zarr.json                # Zarr v3 group metadata
│   ├── .zattrs                  # per-level metadata (bin_ratio, sparsity)
│   │
│   ├── vertices/                # single vlen-bytes array; shape = chunk grid
│   │   ├── zarr.json            # codecs=[vlen-bytes(, compressor)]; attrs:
│   │   │                        #   nonempty_chunks, chunk_grid_origin
│   │   └── c/
│   │       ├── 0/0/0            # chunk at grid coord (0,0,0)
│   │       ├── 0/0/1            # chunk at grid coord (0,0,1)
│   │       └── …
│   │
│   ├── vertex_fragments/        # fragment index — one vlen array, cell/chunk
│   │   ├── zarr.json
│   │   └── c/ …
│   │
│   ├── vertex_attributes/       # per-vertex attribute arrays
│   │   ├── intensity/           # one single vlen array per named attribute
│   │   │   ├── zarr.json
│   │   │   └── c/ …
│   │   └── label/
│   │       ├── zarr.json
│   │       └── c/ …
│   │
│   └── fragment_attributes/     # per-fragment attribute arrays (opt-in)
│       └── object_id/           # e.g. OID owning each fragment
│           ├── zarr.json
│           └── c/ …
│
└── 1/                # coarser level (bin_ratio declared in .zattrs)
    └── [same structure as 0]
```

### Full annotated tree (streamline / polyline)

The streamline tree adds connectivity and object-model arrays. Every
link-family path carries two segments: a signed `<delta>` saying how many
pyramid levels its records span (`0` = intra-level, `+N` / `-N` = N levels
coarser / finer), and an `<offsets>` segment saying where the record's
other endpoints sit relative to its **source** chunk. An intra-chunk link
is simply one whose offsets are all zero (`0.0.0` for an edge in a 3-D
store) — there is no separate cross-chunk family. See
[Links](../object_model/links.md).

```
tracts.zarrvectors/
│
├── zarr.json
├── .zattrs
├── metadata.json
│
└── 0/
    ├── zarr.json
    ├── .zattrs
    │
    ├── vertices/                # vertex positions
    ├── vertex_fragments/        # fragment index over vertices/ rows
    │
    ├── links/                   # connectivity — ONE family
    │   └── 0/                   # <delta>=0 GROUP; zarr.json carries the
    │       │                    #   family policy: link_width=2, sid_ndim,
    │       │                    #   directed, store (+ counts after finalize)
    │       ├── 0.0.0/           # <offsets> all-zero → intra-chunk edges
    │       │   ├── zarr.json    #   has_perm=false; flat rows + link_fragments
    │       │   └── c/ …         #   one file per SOURCE chunk
    │       ├── 0.0.+1/          # edges to the +z neighbour
    │       │   ├── zarr.json    #   has_perm=true (undirected canonical)
    │       │   └── c/ …         #   inline ragged blob; no sidecar
    │       └── 0.+1.0/          # edges to the +y neighbour
    │           ├── zarr.json
    │           └── c/ …
    │
    ├── link_fragments/          # fragment index over links/0/0.0.0/ rows ONLY
    │   ├── zarr.json            #   (keyed by chunk alone — no delta/offsets)
    │   └── c/ …
    │
    ├── link_attributes/         # per-record attrs, mirroring links/ cell-for-cell
    │   └── weight/
    │       └── 0/               # <delta> GROUP
    │           ├── 0.0.0/       # same <offsets> segments as links/0/
    │           │   ├── zarr.json
    │           │   └── c/ …
    │           └── 0.0.+1/
    │               ├── zarr.json
    │               └── c/ …
    │
    ├── attributes/              # per-vertex attributes (e.g. FA, MD)
    │
    ├── fragment_attributes/     # per-fragment attributes (opt-in)
    │   └── object_id/           # e.g. OID owning each fragment in a chunk
    │
    ├── object_index/            # per-object manifest blobs
    │   ├── data                 # concatenated manifest bytes
    │   └── offsets              # int64 array of per-object byte offsets
    │
    ├── object_attributes/       # per-object scalars (e.g. mean FA)
    │   ├── mean_fa/
    │   └── tract_length/
    │
    ├── groupings/               # group ID → [object IDs]
    │
    └── groupings_attributes/    # per-group metadata
```

Pyramids built with `cross_level_depth >= 1` add `<delta>` siblings
to the link arrays. A typical level-0 tree under
`build_pyramid(..., cross_level_depth=1, cross_level_storage="explicit")`:

```
0/
└── links/
    ├── 0/                   # intra-level records
    │   ├── 0.0.0/           #   both endpoints in the source chunk
    │   └── 0.0.+1/          #   endpoint one chunk along +z
    └── +1/                  # cross-level: source → level+1
        ├── 0.0.0/           #   parent in the anchored chunk
        └── 0.0.+1/          #   parent one COARSE chunk along +z
```

Under `<delta> != 0` the offsets are measured against the source chunk
**re-anchored into the target level's grid**, not against the raw
coordinate difference — see
[the anchor](../object_model/links.md#cross-level-placement-the-anchor).
Every `<delta> != 0` array has `has_perm=false` and uses the inline blob
encoding with no `link_fragments/` sidecar.

At an intermediate level (e.g. `1`), both `+1` (drill up to
level 2) and `-1` (drill down to level 0) appear. See
[`examples/07_multiscale_links.ipynb`](../../../examples/07_multiscale_links.ipynb).

### Full annotated tree (graph / skeleton)

```
neuron.zarrvectors/
├── zarr.json
├── .zattrs
├── metadata.json
└── 0/
    ├── vertices/
    ├── vertex_fragments/
    ├── links/
    │   └── 0/                   # GROUP — link_width=2 for graphs / skeletons
    │       ├── 0.0.0/           #   intra-chunk edges
    │       ├── 0.0.+1/          #   edges crossing into the +z neighbour
    │       └── 0.+1.0/          #   … one array per distinct offset
    ├── link_fragments/          # fragment index over links/0/0.0.0/ rows
    ├── link_attributes/
    │   └── weight/
    │       └── 0/
    │           ├── 0.0.0/
    │           └── 0.0.+1/
    ├── attributes/
    ├── object_index/
    └── object_attributes/
```

A skeleton's parent references use `link_width=1`, whose single endpoint
leaves no offsets to encode — those arrays are named by the literal
segment `links/<delta>/self/`.

### Full annotated tree (mesh)

```
brain.zarrvectors/
├── zarr.json
├── .zattrs
├── metadata.json
└── 0/
    ├── vertices/
    ├── vertex_fragments/
    ├── links/
    │   └── 0/                       # GROUP — link_width=3 for triangle meshes
    │       ├── 0.0.0_0.0.0/         #   face wholly inside the source chunk
    │       ├── 0.0.+1_0.0.+1/       #   face straddling the +z boundary
    │       └── 0.0.+1_0.+1.0/       #   face spanning source, +z and +y
    ├── link_fragments/              # fragment index over links/0/0.0.0_0.0.0/ rows
    ├── attributes/
    ├── object_index/
    └── object_attributes/
```

A face carries `link_width - 1 = 2` offsets, joined by `_` — so a mesh's
offsets segments are twice as long as an edge's.

### Parametric objects

The optional `parametric/` group is not spatially chunked. It holds
algebraic objects (planes, spheres, ellipsoids) as a flat array of parameter
tuples:

```
dataset.zarrvectors/
├── …
└── parametric/
    ├── zarr.json
    ├── objects/                 # (n_parametric, param_dim) float64
    │   ├── zarr.json
    │   └── c/0
    └── object_attributes/
        └── label/
```

### Naming rules

Resolution level directories must be named `<N>` where `N` is a
non-negative integer. There is no requirement that levels be contiguous (a
store may have `0` and `2` without `1`),
but contiguous numbering from 0 is strongly recommended.

Array group names within a level are fixed by this specification. Custom
arrays may not be added at the array group level without a spec extension.
Per-vertex and per-object custom attributes must be placed under
`attributes/` and `object_attributes/` respectively.

### Required vs optional nodes

| Path | Required for | Notes |
|------|-------------|-------|
| `zarr.json` (root) | All types | Zarr v3 group node |
| `.zattrs` (root) | All types | ZVF root metadata |
| `metadata.json` | All types | Recommended; not read by API |
| `0/` | All types | At least one level required |
| `vertices/` | All types | |
| `vertex_fragments/` | All types | Required for spatial queries; see [Fragment-index arrays](fragment_index_arrays.md) |
| `link_fragments/` | polyline, streamline, graph, skeleton, mesh | Pairs with `links/0/<all-zero offsets>/` **only**; keyed by chunk alone. See [Fragment-index arrays](fragment_index_arrays.md) |
| `links/<delta>/` | polyline, streamline, graph, skeleton (`link_width=2`); mesh (`link_width=3`) | A **group**, not an array. Carries the family policy; `<delta>=0` intra-level, `±N` cross-pyramid-level |
| `links/<delta>/<offsets>/` | as above | One per-chunk array per distinct offset. All-zero offsets = intra-chunk; `self` when `link_width=1` |
| `link_attributes/<name>/<delta>/` | Any geometry that wrote `edge_attributes` | A **group**, mirroring `links/<delta>/` |
| `link_attributes/<name>/<delta>/<offsets>/` | as above | Mirrors `links/<delta>/<offsets>/` cell-for-cell |
| `attributes/` | All types | Optional if no per-vertex attributes |
| `object_index/` | polyline, streamline, graph, skeleton, mesh | |
| `object_attributes/` | Any type | Optional |
| `groupings/` | Any discrete-object type | Optional |
| `parametric/` | Any type | Optional |
