> [!NOTE]
> This package is under active development.

<img src="assets/zarr-vectors.png" alt="zarr-vectors" width="60%" />

**Tools for Zarr Vectors Data**

`zarr-vectors-py` is a Python package for reading, writing, and managing large-scale vector geometry data in the zarr vectors format — a chunked, cloud-native format built on Zarr v3 for multiscale points, lines, streamlines, graphs, skeletons, and meshes.

The package supports supervoxel-level spatial binning with separated chunk and bin sizes, per-level object sparsity for balanced multi-resolution pyramids, and OME-Zarr-compatible multiscale metadata.

*Aligned to the Zarr Vectors specification by Forrest Collman, Allen Institute for Brain Sciences*
[Link to specification GitHub](https://github.com/AllenInstitute/zarr_vectors)

---
## Documentation

**link to readthedocs:** https://zarr-vectors-py.readthedocs.io/en/latest

---

## Install

```bash
pip install zarr-vectors
```

---

## Quick start

```python
import numpy as np
import zarr_vectors as zv

# A Schema says what the data is; a Layout says how it is cut up.
schema = zv.Schema(
    bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
    kind="point_cloud",
    layout=zv.Layout(cells=8),  # ~8 chunks per axis
)

ds = zv.create("demo.zarrvectors", schema=schema)

rng = np.random.default_rng(0)
positions = rng.uniform(0, 1000, size=(10_000, 3)).astype("float32")
ds.add_points(
    positions,
    attributes={"intensity": rng.random(10_000).astype("float32")},
    object_ids=np.repeat(np.arange(100), 100),
)
ds.build_pyramid(factors=[(2.0, 1.0)])  # add a coarser level

# Read back only what falls inside a box.
result = zv.open("demo.zarrvectors").select(bbox=([0, 0, 0], [250, 250, 250])).read()

print(result.kind, result.positions.shape, result.attributes.names())
print(result.attributes["intensity"][:3])
```

```
point_cloud (122, 3) ('intensity',)
[0.58416754 0.17340323 0.28906345]
```

A read always comes back as a `ReadResult`, whatever the geometry: `positions` is
`(N, D)`, `parts` slices it into polylines or segments, and `edges` / `faces` /
`attributes` are filled in when the store carries them. Per-vertex `object_ids`
are populated only on the reader paths that supply them — a whole-store read of a
point cloud leaves them `None` even when ids were written. Reach for
`level.objects[...]` to read by id.

There are two supported surfaces. `zarr_vectors.api` — re-exported from the top-level
package, which is what the example above uses — is for reading and writing data.
`zarr_vectors.building` is for tools that construct stores. Alongside them
`constants`, `exceptions`, `typing` and `headers` are supported too. `core`,
`encoding`, `spatial`, `lazy`, `ops`, `sharding`, `multiresolution` and `rechunk` are
internal and may change between releases, and `types`, `validate` and `composite` are
undecided — neither promised nor disowned. `zv.stability("zarr_vectors.<module>")`
answers for any of them at runtime.

Fuller walkthroughs live in the
[documentation](https://zarr-vectors-py.readthedocs.io/en/latest).

---

## Store Layout

A store is a plain Zarr v3 group. Only five entries always exist; everything else
appears when the matching data is written.

```
dataset.zarrvectors/
├── zarr.json                     [always]  Zarr v3 group metadata.  Store-level fields sit under
│                                           attributes.zarr_vectors (zv_version, bounds, chunk_shape,
│                                           base_bin_shape, geometry_types, conventions, optional crs);
│                                           per-level scale/translation under attributes.multiscales
│                                           (OME-NGFF 0.4).
├── 0/                            [always]  Full resolution.  Levels are bare integers, no prefix.
│   ├── zarr.json                 [always]  Level metadata under attributes.zarr_vectors_level
│   │                                       (vertex_count, object_sparsity, coarsening_method,
│   │                                       parent_level, arrays_present, ...).
│   │
│   │  ── spatial arrays: each is one Zarr v3 array whose shape IS the level's chunk grid, holding
│   │     one vlen-bytes cell per occupied chunk at  c/<i>/<j>/<k> .  Occupancy is listed in the
│   │     array's nonempty_chunks attribute; chunk_grid_origin (absent means 0) maps negative
│   │     chunk coordinates onto it.
│   │
│   ├── vertices/                 [always]  c/i/j/k = every vertex in that chunk, packed.  One cell
│   │   └── c/0/0/0 ...                     per chunk, not per bin.
│   ├── vertex_fragments/         [always]  Fragment index into the sibling vertices cell: per
│   │                                       fragment either a row range [start, count) or an explicit
│   │                                       list of row indices (which is how two fragments can share
│   │                                       a vertex).  Row indices, not byte offsets.  Fragments are
│   │                                       per-bin, or per-(object, chunk) once object ids exist.
│   ├── vertex_attributes/        [written] One child array per name, same grid, rows aligned 1:1
│   │   └── intensity/                      with the vertices cell.
│   ├── links/                    [written] Connectivity.  A GROUP two levels deep, never an array:
│   │   ├── 0/0.0.0/                        links/<delta>/<offsets>/.  <delta> is how many pyramid
│   │   ├── 0/0.0.+1/                       levels the record spans: 0 intra-level, +1/-1 for the
│   │   └── +1/0.0.0/                       parent/child metavertex edges a pyramid adds.  <offsets>
│   │                                       is where the record's other endpoints sit relative to its
│   │                                       source chunk (the cell holding it): link_width-1 signed
│   │                                       tuples, components joined by ".", offsets by "_", or
│   │                                       "self" when link_width is 1.  So 0.0.0 is intra-chunk and
│   │                                       0.0.+1 reaches one chunk along +z.  A cross-chunk link is
│   │                                       simply one with non-zero offsets — there is no separate
│   │                                       array family for them.
│   ├── link_fragments/           [written] Fragment index over the one intra-chunk link array
│   │                                       (delta 0, all-zero offsets), so it is keyed by chunk alone
│   │                                       and carries no delta or offsets segment.  Every other
│   │                                       offsets array self-describes and needs no sidecar.
│   ├── link_attributes/          [written] Mirrors links/ exactly — <name>/<delta>/<offsets>, same
│   │   └── weight/0/0.0.0/                 cells, same row order, so rows align 1:1 with links.
│   ├── fragment_attributes/      [builder] One row per fragment in the chunk.
│   │   └── object_id/
│   │
│   │  ── non-spatial arrays: rank-1 over object or group ids, so their chunk keys are  c/0, c/1, ...
│   │
│   ├── object_index/             [written] A GROUP whose only child is manifests: a ragged array with
│   │   └── manifests/                      one row per object slot, row i being object i's ordered
│   │       └── c/0                         (chunk_coords, fragment_index) references.  num_objects
│   │                                       and num_present sit on the group.
│   ├── object_attributes/        [written] One (O,) or (O,C) array per name; absent objects hold the
│   │   └── cell_type/                      array's fill value.
│   ├── groups/                   [written] One ragged (G,) array: row g is that group's object ids.
│   └── group_attributes/         [builder] One (G,) or (G,C) array per name.
│       └── region/
├── 1/                            [written] A coarser level from build_pyramid.  Same layout, but not
│   ├── vertices/                           the same arrays: a coarsened level gets vertices,
│   ├── vertex_fragments/                   vertex_fragments, object_index and links/-1, and inherits
│   ├── object_index/manifests/             object_attributes.  It does not get vertex_attributes,
│   ├── object_attributes/                  groups, group_attributes, link_attributes or
│   └── links/-1/0.0.0/                     link_fragments.
├── parametric/                   [written] Root-level, created lazily on the first parametric write.
│   ├── zarr.json                           Its attributes carry the type registry (plane, line,
│   ├── objects/                            sphere).  objects/ is (O, 1 + max_coeffs): a type id, then coefficients.
│   ├── names/
│   ├── object_attributes/
│   ├── groups/
│   └── group_attributes/
└── headers/                      [written] Root-level, one group per source format; the header dict
    └── swc/                                is stored as that group's attributes.
```

- `[always]` — there the moment `zv.create(...)` returns, before anything is written.
  A fresh store is exactly these five entries, and a bare `add_points(positions)` with
  no ids, attributes, groups or pyramid adds nothing more.
- `[written]` — appears only once the matching data is written, so which of these a
  store has depends on its geometry kind and on what the caller passed. A point cloud
  grows a `links/` only if you build a pyramid; polylines and streamlines never write
  `link_fragments/`, because their intra-chunk edges are implicit.
- `[builder]` — no `zarr_vectors.api` writer produces it; only `zarr_vectors.building`
  does. The api can still read them back: `Level.attribute_names("fragment")` and
  `Level.attribute_names("group")` are both accepted.

Sharding does not change this tree. The keys stay `c/i/j/k`; each file just becomes a
shard covering several cells.
