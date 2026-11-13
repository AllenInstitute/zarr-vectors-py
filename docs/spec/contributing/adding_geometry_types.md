# Adding a geometry type

This page lists the places in `zarr-vectors-py` that a new geometry
type has to be added to, in the order a contributor meets them, using
the existing types as reference implementations.

## Terms

**Geometry type**
: A string in the root `geometry_types` list that says how a store's
  vertices and links are to be read, such as `"point_cloud"` or
  `"streamline"`. The valid values are `VALID_GEOMETRY_TYPES` in
  `zarr_vectors/constants.py`. See [Geometry types](../geometry_types/index.md).

**Links convention**
: The root `links_convention` value saying how connectivity is stored:
  `implicit_sequential` (vertex order within a fragment is the
  topology), `implicit_sequential_with_branches`, or `explicit` (a
  `links/` family). See [Links](../object_model/links.md).

**Writer and reader**
: The functions in `zarr_vectors/types/<module>.py` that write a store
  of a type from raw geometry and read it back, such as `write_points`
  and `read_points`.

---

## Introduction

All geometry types share one chunk layout, so a new type is mostly a
new writer and reader over the existing arrays, plus registrations in
the constants, the schema, the validator and the `Dataset` facade.
A new type is a format change: propose it in an RFC first (see
[Spec change process](spec_change_process.md)).

---

## Technical reference

### Decide the design

Answer these in the RFC:

1. The type string: lowercase letters, digits and underscores.
2. Which arrays it writes beyond `vertices` and `vertex_fragments`
   (see [Chunk arrays](../layout/chunk_arrays.md)), and whether it has
   discrete objects and so an object index.
3. Its links convention. For an explicit `links/` family: the
   `link_width` (2 for edges, 3 or more for faces) and whether the
   family is `directed` (endpoint order is data) or undirected
   (canonically sorted, with `perm_idx` recovering input order).
4. Any root or level attribute keys it adds.

### Register the type string

- Add a `GEOM_<NAME>` constant to `zarr_vectors/constants.py` and add it
  to `VALID_GEOMETRY_TYPES`. `RootMetadata.validate`
  (`zarr_vectors/core/metadata.py`) rejects any type not in that set.
- Add the value to the `GeometryType` enum in
  `schema/zarr_vectors.linkml.yaml` and run `python schema/regen.py`.
  `test_constant_set_matches_schema_enum` in
  `tests/test_linkml_schema.py` fails until both agree.

### Write the writer and reader

Add `write_<type>` and `read_<type>` to a module in
`zarr_vectors/types/`, following the closest existing type:

| Closest to | Module | Functions |
|------------|--------|-----------|
| Unconnected points | `points.py` | `write_points`, `read_points` |
| Ordered vertex sequences | `polylines.py`, `lines.py` | `write_polylines`, `read_polylines`, `write_lines`, `read_lines` |
| Explicit edges | `graphs.py` | `write_graph`, `read_graph` |
| Faces | `meshes.py` | `write_mesh`, `read_mesh` |

The writer must keep the invariants every reader relies on: vertices
in each chunk are sorted into fragments, the fragment index describes
the sorted rows, and every per-vertex attribute is reordered with the
same permutation (see [Fragments](../object_model/fragments.md)). It
records the type in `geometry_types` and the links convention in the
root attributes, and updates each level's `vertex_count`. The reader
returns the same result-dict keys as the closest existing reader.

### Register the type with the validator

- Level 4 (`zarr_vectors/validate/conformance.py`): add the type to
  `GEOMETRY_LINK_REQ` with the links conventions it allows. A type
  missing from that dict produces the warning
  `Unknown geometry type: '<type>'`. Add any type-specific rule beside
  the existing mesh and point-cloud rules in `validate_conformance`.
- Levels 1 to 3 (`structure.py`, `metadata.py`, `consistency.py`) check
  arrays and metadata without reference to the type. Add a check there
  only for a new array or key.

See [Validation](../validation/overview.md) for what each level checks.

### Register the type with the `Dataset` facade

- `zarr_vectors/api/level.py`: add an entry to `_READERS` naming the
  reader module and function, the keyword arguments it accepts, and the
  `ReadResult` constructor that adapts its result.
- `zarr_vectors/api/result.py`: add a `ReadResult.from_<type>`
  constructor if no existing one fits.
- `zarr_vectors/api/dataset.py`: add a `Dataset.add_<type>` method if
  users should write the type through the facade.

### Pyramids

`build_pyramid` and `coarsen_level` in
`zarr_vectors/multiresolution/coarsen.py` build coarser levels with the
`per_object` method for every type. Type-specific coarsening, such as
mesh decimation or polyline simplification, is registered from
`zarr-vectors-tools` through `zarr_vectors/multiresolution/registry.py`.
Check that `per_object` produces a store that passes level 5 validation
for the new type.

### Tests

Add `tests/test_<type>.py` covering at least:

- a write and read round trip, including objects that cross chunks;
- a bounding-box read;
- `zarr_vectors.validate.validate(path, level=5)` passing on a freshly
  written store and on a store with a pyramid;
- a corrupted store failing validation at the expected level.

See [Compliance testing](test_compliance.md).

### Documentation

- Add a spec page `docs/spec/geometry_types/<type>.md` following the
  [spec documentation standards](spec_change_process.md), and add it to
  the toctree and comparison matrix in `docs/spec/geometry_types/index.md`.
- Add a tutorial in `docs/tutorials/data_types/`.
