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
  `links/<delta>/`, `vertex_attributes/<name>/`, …) is a **single** Zarr v3
  vlen-bytes array whose shape is the level's chunk grid. One cell holds
  one spatial chunk's payload bytes; the chunk files live under the `c/`
  sub-tree (`c/i/j/k`). A spatial chunk at absolute coord `c` maps to cell
  `c - origin`, where `origin = floor(min_corner / chunk_shape)` is stored
  in the array's `chunk_grid_origin` attribute (absent ⇒ zero origin). The
  set of non-empty cells is listed in the array's `nonempty_chunks`
  attribute. `cross_chunk_links/<delta>/` is the one exception — its cells
  are keyed by endpoint-chunk tuples (not a spatial grid), so each cell is
  its own small array under a group.

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

The streamline tree adds connectivity and object-model arrays. Under
the 0.4 multiscale-links layout, every link-family array carries a
signed `<delta>` segment that says how many pyramid levels its edges
span (`0` = intra-level, `+N` / `-N` = N levels coarser / finer). See
[Links and cross-chunk links](../object_model/cross_chunk_links.md).

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
    ├── links/                   # connectivity (per spatial chunk)
    │   └── 0/                   # <delta>=0 → intra-level edges
    │       ├── zarr.json        # link_width=2 for streamline/polyline
    │       └── c/ …             # one file per chunk_key
    │
    ├── link_fragments/          # fragment index over links/0/ rows (delta=0)
    │   ├── zarr.json
    │   └── c/ …
    │
    ├── cross_chunk_links/       # inter-chunk edges (global flat blob)
    │   └── 0/
    │       ├── zarr.json        # num_links, sid_ndim, level_delta=0
    │       └── data             # 2*(sid_ndim+1) int64s per link
    │
    ├── link_attributes/         # per-edge attrs, parallel to links/<delta>/
    │   └── weight/
    │       └── 0/
    │           ├── zarr.json
    │           └── c/ …
    │
    ├── cross_chunk_link_attributes/    # per-CCL attrs (NEW in 0.4)
    │   └── weight/                     # parallel to cross_chunk_links/<delta>/data
    │       └── 0/
    │           ├── zarr.json           # num_links matches CCL meta
    │           └── data
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
├── links/
│   ├── 0/                   # intra-level edges
│   └── +1/                  # cross-level: source local → coarse local (same chunk_key)
├── cross_chunk_links/
│   ├── 0/                   # intra-level inter-chunk edges
│   └── +1/                  # cross-level inter-chunk edges
└── …
```

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
    │   └── 0/                   # link_width=2 for graphs / skeletons
    ├── link_fragments/          # fragment index over links/0/ rows
    ├── cross_chunk_links/
    │   └── 0/
    ├── link_attributes/
    │   └── weight/
    │       └── 0/
    ├── cross_chunk_link_attributes/
    │   └── weight/
    │       └── 0/
    ├── attributes/
    ├── object_index/
    └── object_attributes/
```

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
    │   └── 0/                   # link_width=3 for triangle meshes
    ├── link_fragments/          # fragment index over links/0/ rows
    ├── cross_chunk_links/
    │   └── 0/
    ├── attributes/
    ├── object_index/
    └── object_attributes/
```

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
| `link_fragments/` | polyline, streamline, graph, skeleton, mesh | Present at `<delta>=0` whenever `links/0/` is present |
| `links/<delta>/` | polyline, streamline, graph, skeleton (`link_width=2`); mesh (`link_width=3`) | `<delta>=0` for intra-level edges; `<delta>=±N` for cross-pyramid-level edges (0.4+) |
| `cross_chunk_links/<delta>/` | Any geometry whose objects can span multiple chunks | `<delta>=0` always; `±N` when `cross_level_depth > 0` |
| `link_attributes/<name>/<delta>/` | Any geometry that wrote `edge_attributes` | Parallel to `links/<delta>/` |
| `cross_chunk_link_attributes/<name>/<delta>/` | Any geometry with cross-chunk per-edge attrs (0.4+) | Parallel to `cross_chunk_links/<delta>/data` |
| `attributes/` | All types | Optional if no per-vertex attributes |
| `object_index/` | polyline, streamline, graph, skeleton, mesh | |
| `object_attributes/` | Any type | Optional |
| `groupings/` | Any discrete-object type | Optional |
| `parametric/` | Any type | Optional |
