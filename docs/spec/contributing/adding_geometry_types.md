# Adding a new geometry type

## Terms

**Geometry type constant**
: The string value stored in the `geometry_types` list of the root
  `zarr_vectors` attribute block that identifies the type. Must be a lowercase alphanumeric string
  with underscores (e.g. `"point_cloud"`, `"streamline"`).

**Type-specific arrays**
: Arrays that are required or optional for a given geometry type but not
  for all types (e.g. `object_index/` for types with discrete objects, or
  a `links/<delta>/` family with `link_width >= 3` for `mesh`).

**Write function**
: The `write_<type>()` function in `zarr_vectors/types/<type>.py` that
  accepts raw geometry data and writes a conforming Zarr Vectors store.

**Read function**
: The `read_<type>()` function that reads a Zarr Vectors store of the given type
  and returns a typed result dict.

---

## Introduction

Adding a new geometry type to Zarr Vectors touches several layers: a new
type constant, a writer, a reader, the dataset-API reader table, the
geometry conformance table, the LinkML schema, tests, and documentation.
This page is a checklist for contributors, using the existing types as
reference implementations. It names real extension points in the code;
there is no plug-in registry for geometry types, so each step is an edit
to the library itself.

Before beginning implementation, ensure the new type has been ratified via
the RFC process (see [Spec change process](spec_change_process.md)).

---

## Technical reference

### Step 0 — Choose a type constant and arrays

Decide:
1. The type constant string (e.g. `"voxel_cloud"` for a new type).
2. Which existing arrays it requires (every type writes `vertices/`; the
   writers also produce `vertex_fragments/`).
3. What new arrays, if any, it introduces. Prefer reusing the existing
   families (`vertex_attributes/`, `object_attributes/`, `links/`) over new
   arrays.
4. Whether it has discrete objects (requires `object_index/`).
5. Its link-family policy: `link_width`, and whether the family is
   `directed` (endpoint order is data) or undirected (canonical-sorted
   and deduplicated, with `perm_idx` recovering input order). The policy is
   set by `write_links(directed=..., store=...)` /
   `create_links_array(...)`.
6. Whether its connectivity is explicit or implicit — an
   `implicit_sequential` type emits no intra-chunk links at all, since
   vertex order within a fragment carries the topology. The value goes in
   `links_convention` (one of `explicit`, `implicit_sequential`,
   `implicit_sequential_with_branches`).
7. Whether it needs any root-metadata keys. Prefer none: existing types
   declare none beyond `geometry_types` and the three convention fields.

### Step 1 — Add the type constant

```python
# zarr_vectors/constants.py
GEOM_VOXEL_CLOUD: str = "voxel_cloud"    # ← new constant

VALID_GEOMETRY_TYPES: frozenset[str] = frozenset({
    GEOM_POINT_CLOUD, GEOM_LINE, GEOM_POLYLINE, GEOM_STREAMLINE,
    GEOM_SKELETON, GEOM_GRAPH, GEOM_MESH,
    GEOM_VOXEL_CLOUD,    # ← register new constant
})
```

`RootMetadata.validate()` rejects any `geometry_types` entry not in
`VALID_GEOMETRY_TYPES`. Also add the token to the `GeometryType` enum in
`schema/zarr_vectors.linkml.yaml` and regenerate the artefacts with
`python schema/regen.py` (`--check` verifies they are current).

### Step 2 — Write the write function

Create `zarr_vectors/types/voxel_clouds.py`. The existing writers
(`write_points` is the simplest reference) share one skeleton, built from
these real helpers:

1. Validate arguments; reject unsupported `out_of_bounds` policies
   *before* creating anything.
2. `_create_or_open_store(store_path, backend=..., bounds=..., chunk_shape=..., ndim=...)`
   from `zarr_vectors.core.store`.
3. `_apply_out_of_bounds_policy(root, positions, policy=out_of_bounds)`.
4. `_ensure_root_metadata_for_write(root, inferred_ndim=ndim, geometry_type=GEOM_VOXEL_CLOUD, base_bin_shape=bin_shape, links_convention=..., object_index_convention=..., cross_chunk_strategy=...)`.
5. `create_resolution_level(root, 0, LevelMetadata(level=0, vertex_count=..., arrays_present=[...]))`.
6. Inside `open_write_session(level_group, compressor=..., shard_shape=..., bounds=..., chunk_shape=...)`
   (from `zarr_vectors.core.arrays`): `create_vertices_array`, then
   per chunk `write_chunk_vertices(level_group, chunk_coords, fragments, dtype=...)`,
   `create_attribute_array` / `write_chunk_attributes` for vertex attributes,
   `write_links` for any connectivity, and
   `write_object_index(level_group, manifests, sid_ndim=...)` for objects.
7. `stamp_fragments_tile(level_group, ndim)` (only when each chunk's
   fragments tile its buffer) and `_finalize_write(root, "write_voxel_cloud")`.
8. Return a summary dict (`vertex_count`, `chunk_count`, ...).

Signature conventions (keyword-only after the data arguments):
`chunk_shape`, `bin_shape=None`, `bounds=None`, `vertex_attributes=None`,
`object_attributes=None`, `dtype="float32"`, `backend=None`,
`chunk_by_attribute=None`, `out_of_bounds=DEFAULT_OOB_POLICY`,
`compressor=None`, `shard_shape=None`. Name per-vertex attributes
`vertex_attributes` to match the on-disk directory (the older
`attributes=` spelling is deprecated everywhere).

Key invariants to maintain in the writer:
- Fragment indices recorded in manifests must match the fragment order
  actually written for each chunk.
- All per-vertex attribute arrays must be split with the same fragment
  boundaries and row order as `vertices`.
- Links are given to `write_links` in global `(chunk_coords, vertex_index)`
  form; it routes each record to its `<offsets>` array.

### Step 3 — Write the read function

`read_voxel_clouds(store_path, *, level=0, bbox=None, object_ids=None, chunks=None, attribute_filter=None, backend=None)`
returns a dict. Follow the existing readers:

- `open_store(store_path, backend=backend)` accepts a path, URL, `Store` or
  `Group`; wrap the body in `root.cached_nodes()`.
- Resolve the chunks to read with `resolve_chunk_keys(level_group, level_chunk_shape, bbox=..., chunks=...)` (from `zarr_vectors.core.arrays`), using
  `get_level_chunk_shape(root_meta, level_meta)` for the level's effective chunk shape.
- If a filter is not implemented, **raise** (`NotImplementedError`) rather than
  ignoring it: `read_mesh` and `read_graph` do this for `object_ids=`.
- Return the shared keys where they apply (`positions` or `vertices`,
  `vertex_attributes`, `vertex_count`, ...). There is no uniform return
  schema across types; the dataset API adapts each (next step).

### Step 4 — Plug into the dataset API

`zarr_vectors/api/level.py` holds `_READERS`: for each geometry constant,
the reader's module and function, the keyword arguments it supports, and
the name of a `ReadResult.from_<type>` classmethod in
`zarr_vectors/api/result.py` that maps its dict into the uniform
`ReadResult`. Add an entry and an adapter, and an `add_<type>` method on
`Dataset` (`zarr_vectors/api/dataset.py`) if the type should be writable
from `Schema(kind=...)`.

### Step 5 — Add geometry conformance checks

Validation levels L1–L3 are geometry-independent. Type-specific rules live
at L4 in `zarr_vectors/validate/conformance.py`:

```python
GEOMETRY_LINK_REQ: dict[str, set[str]] = {
    ...
    GEOM_VOXEL_CLOUD: {LINKS_EXPLICIT},   # allowed links_convention values
}
```

Add any further per-type check inside `validate_conformance` (it receives
the root metadata and level 0). Messages are free text appended to the
`ValidationResult` (`add_pass` / `add_warning` / `add_error`); there are no
stable check IDs, and no `REQUIRED_ARRAYS` table to extend. L1 is
`validate_structure` and requires only `vertices/` for every type.

### Step 6 — Multiresolution

Core's `per_object` coarsener is geometry-agnostic: it rebuilds `vertices`,
`object_index` and per-object attributes. Geometry-specific coarsening
(connectivity rewrite, attribute aggregation) is added by registering a
coarsen method through `zarr_vectors.multiresolution.registry`
(`register_coarsen_strategy(name, fn)`), selected by `method=name`;
registered names must not shadow the built-in `per_object`. See
[Pyramid construction](../multiscale/pyramid_construction.md).

### Step 7 — Write tests

Tests live flat in `tests/` (`tests/test_<area>.py`, e.g.
`tests/test_meshes.py`, `tests/test_graphs.py`) with integration tests
under `tests/integration/`. Add `tests/test_voxel_clouds.py` covering:

- [ ] Round-trip write → read produces identical data.
- [ ] Bbox query returns correct vertices.
- [ ] A pyramid can be built and read at level 1.
- [ ] `validate(path, level=5)` passes on a freshly written store.
- [ ] A deliberately corrupted store is rejected at the level that should
      catch it.
- [ ] The dataset-API path (`Dataset.add_*`, `Level.read`) works.

See [Compliance testing](test_compliance.md).

### Step 8 — Write documentation

Add these pages (following the spec page template):

- `docs/spec/geometry_types/voxel_cloud.md` — type spec page.
- A section in `docs/tutorials/data_types/` — tutorial.

Update `docs/spec/geometry_types/index.md` (constants, comparison matrix,
writer/reader table, selection guide) and its toctree.

### Checklist summary

- [ ] Type constant in `constants.py` and `VALID_GEOMETRY_TYPES`
- [ ] LinkML `GeometryType` enum updated and `schema/regen.py` re-run
- [ ] Write function in `types/<type>.py`
- [ ] Read function in `types/<type>.py`
- [ ] Entry in `api/level.py` `_READERS` plus a `ReadResult.from_<type>` adapter
- [ ] `GEOMETRY_LINK_REQ` entry (and any extra L4 checks) in `validate/conformance.py`
- [ ] Coarsening strategy registered (if `per_object` is not enough)
- [ ] Test module with round-trip, bbox, pyramid, and validation tests
- [ ] Spec page in `docs/spec/geometry_types/`
- [ ] Tutorial in `docs/tutorials/data_types/`
- [ ] Comparison matrix updated
