# Directory structure

## Terms

**Store root**
: The top-level directory (or object-store prefix) of a Zarr Vectors store.
  Contains the root `zarr.json` and all resolution level groups.

  **Where metadata lives.** Zarr v3 has no `.zattrs` file. Every group and
  array keeps its user metadata in the `"attributes"` block of its own
  `zarr.json`; elsewhere in these pages "`.zattrs`" is shorthand for that
  block. The root's block holds `zarr_vectors`, `multiscales` and `ome` (see
  [Root metadata](root_metadata.md)); a level's holds `zarr_vectors_level`
  (see [Level groups](level_groups.md)).

  **File extension.** `.zarrvectors` is the canonical extension and the one
  to prefer in documentation, examples and generated output. `.zv` is an
  accepted short form, useful where path length matters — deep scratch
  hierarchies on HPC filesystems, say. The two are interchangeable: nothing
  in the format reads the extension, and a store opens the same either way.
  A store with no extension at all is still a valid store.

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

**`parametric/`**
: An optional sub-group at the store root for algebraic (non-vertex-based)
  geometry objects such as planes, lines and spheres. Not chunked
  spatially and outside the resolution-level hierarchy; stores one dense array
  of object parameter rows. Created lazily on first use.

---

## Introduction

The on-disk layout of a Zarr Vectors store follows a clear hierarchy: store root →
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
├── zarr.json                    # Zarr v3 root group; attributes: zarr_vectors,
│                                #   multiscales, ome  (see root_metadata.md)
│
├── 0/                           # full-resolution level
│   ├── zarr.json                # group; attributes: zarr_vectors_level
│   │
│   ├── vertices/                # single vlen-bytes array; shape = chunk grid
│   │   ├── zarr.json            # codecs=[vlen-bytes(, compressor)]; attrs:
│   │   │                        #   zv_array, dtype, encoding,
│   │   │                        #   nonempty_chunks, [chunk_grid_origin]
│   │   └── c/
│   │       ├── 0/0/0            # chunk at grid coord (0,0,0)
│   │       ├── 0/0/1            # chunk at grid coord (0,0,1)
│   │       └── …
│   │
│   ├── vertex_fragments/        # fragment index — one vlen array, cell/chunk
│   │   ├── zarr.json
│   │   └── c/ …
│   │
│   ├── vertex_attributes/       # per-vertex attribute arrays (group)
│   │   ├── intensity/           # one vlen array per named attribute
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
└── 1/                           # coarser level (bin ratio lives in the root
    └── [same structure as 0]    #   multiscales block, not in the level)
```

A freshly created store (`create_store`) already holds `0/vertices/` and
`0/vertex_fragments/` as empty arrays: the "warm" shell.

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
│
└── 0/
    ├── zarr.json
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
    ├── vertex_attributes/       # per-vertex attributes (e.g. FA, MD)
    │   └── fa/                  #   one vlen chunk array per name
    │
    ├── fragment_attributes/     # per-fragment attributes (opt-in)
    │   └── object_id/           # e.g. OID owning each fragment in a chunk
    │
    ├── object_index/            # GROUP: per-object manifests
    │   ├── manifests/           #   1-D vlen-bytes, one blob per object row
    │   └── object_ids/          #   int64, row -> object id
    │
    ├── object_attributes/       # per-object columns (plain arrays, NOT groups)
    │   ├── mean_fa/             #   shape (n_objects,)
    │   └── tract_length/
    │
    ├── groups/                  # 1-D vlen-bytes: group id -> int64 object ids
    │
    └── group_attributes/        # per-group columns (plain arrays)
        └── n_objects/
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
    ├── vertex_attributes/
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
└── 0/
    ├── vertices/
    ├── vertex_fragments/
    ├── links/
    │   └── 0/                       # GROUP — link_width=3 for triangle meshes
    │       ├── 0.0.0_0.0.0/         #   face wholly inside the source chunk
    │       ├── 0.0.+1_0.0.+1/       #   face straddling the +z boundary
    │       └── 0.0.+1_0.+1.0/       #   face spanning source, +z and +y
    ├── link_fragments/              # fragment index over links/0/0.0.0_0.0.0/ rows
    ├── vertex_attributes/
    ├── object_index/
    └── object_attributes/
```

A face carries `link_width - 1 = 2` offsets, joined by `_` — so a mesh's
offsets segments are twice as long as an edge's.

### Parametric objects

The optional `parametric/` group is not spatially chunked. It holds
algebraic objects (planes, lines, spheres, or custom types) as a flat array of
parameter rows `[type_id, coeff_0, coeff_1, …]`, padded with `NaN` to the
widest type:

```
dataset.zarrvectors/
├── …
└── parametric/
    ├── zarr.json                # attributes: the type registry
    ├── objects/                 # (n_objects, 1 + max_coefficients) float64
    │   └── zarr.json            #   zv_array="parametric_objects", num_objects, max_row_length
    ├── names/                   # 1-D vlen-bytes, UTF-8 name per object
    ├── object_attributes/       # optional
    │   └── label/
    ├── groups/                  # optional, same encoding as level groups
    └── group_attributes/        # optional
```

### Naming rules

Resolution level directories must be named `<N>` where `N` is a
non-negative integer. There is no requirement that levels be contiguous (a
store may have `0` and `2` without `1`),
but contiguous numbering from 0 is strongly recommended.

Array group names within a level are fixed by this specification. Custom
arrays may not be added at the array group level without a spec extension.
Per-vertex and per-object custom attributes must be placed under
`vertex_attributes/` and `object_attributes/` respectively.

### Required vs optional nodes

| Path | Required for | Notes |
|------|-------------|-------|
| `zarr.json` (root) | All types | Zarr v3 group node; its `attributes` must contain the `zarr_vectors` block (L1 error otherwise) |
| `ome` attribute (root) | — | Recommended; its absence is an L1 *warning* (store stays valid) |
| `0/` | All types | At least one level required |
| `vertices/` | All types | L1 error if missing |
| `vertex_fragments/` | All types | L1 warning if missing; required for spatial queries. See [Fragment-index arrays](fragment_index_arrays.md) |
| `link_fragments/` | polyline, streamline, graph, skeleton, mesh | Pairs with `links/0/<all-zero offsets>/` **only**; keyed by chunk alone. See [Fragment-index arrays](fragment_index_arrays.md) |
| `links/<delta>/` | polyline, streamline, graph, skeleton (`link_width=2`); mesh (`link_width=3`) | A **group**, not an array. Carries the family policy; `<delta>=0` intra-level, `±N` cross-pyramid-level |
| `links/<delta>/<offsets>/` | as above | One per-chunk array per distinct offset. All-zero offsets = intra-chunk; `self` when `link_width=1` |
| `link_attributes/<name>/<delta>/` | Any geometry that wrote link attributes | A **group**, mirroring `links/<delta>/` |
| `link_attributes/<name>/<delta>/<offsets>/` | as above | Mirrors `links/<delta>/<offsets>/` cell-for-cell |
| `vertex_attributes/` | — | Optional (only if per-vertex attributes exist) |
| `fragment_attributes/` | — | Optional, opt-in |
| `object_index/` | polyline, streamline, graph, skeleton, mesh; point clouds written with objects | A group with `manifests` and `object_ids` |
| `object_attributes/` | — | Optional |
| `groups`, `group_attributes/` | — | Optional |
| `parametric/` | — | Optional |

A level's `zarr_vectors_level.arrays_present` lists the *family* names present
(`"vertices"`, `"vertex_attributes"`, `"links"`, `"object_index"`, …), never
`"<family>/<name>"`; `building.refresh_arrays_present` re-derives it from disk.
