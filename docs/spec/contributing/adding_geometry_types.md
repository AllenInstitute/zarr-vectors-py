# Adding a geometry type

The places in `zarr-vectors-py` a new geometry type must be added to,
in the order a contributor meets them.

## Terms

**Geometry type**
: A string in the root `geometry_types` list, such as `"point_cloud"`
  or `"streamline"`, that says how to read the store's vertices and
  links. Valid values are `VALID_GEOMETRY_TYPES` in
  `zarr_vectors/constants.py`. See [Geometry types](../geometry_types/index.md).

**Links convention**
: The root `links_convention` value: `implicit_sequential` (vertex
  order in a fragment is the topology),
  `implicit_sequential_with_branches`, or `explicit` (a `links/`
  family). See [Links](../object_model/links.md).

## Introduction

All geometry types share one chunk layout, so a new type is mostly a
writer and reader over the existing arrays, plus registrations. It is a
format change, so propose it in an RFC first (see
[Spec change process](spec_change_process.md)).

## Technical reference

### Design questions for the RFC

1. The type string: lowercase letters, digits and underscores.
2. Which arrays it writes beyond `vertices` and `vertex_fragments` (see
   [Chunk arrays](../layout/chunk_arrays.md)), and whether it has
   objects and so an object index.
3. Its links convention. For a `links/` family: the `link_width` (2 for
   edges, 3 or more for faces) and whether it is `directed`.
4. Any root or level attribute keys it adds.

### Checklist

| Step | Where | What |
|---|---|---|
| Type string | `zarr_vectors/constants.py` | Add `GEOM_<NAME>` to `VALID_GEOMETRY_TYPES`. `RootMetadata.validate` rejects any other type. |
| Schema | `schema/zarr_vectors.linkml.yaml` | Add the value to the `GeometryType` enum, then run `python schema/regen.py`. `test_constant_set_matches_schema_enum` fails until both agree. |
| Writer and reader | `zarr_vectors/types/<module>.py` | Add `write_<type>` and `read_<type>`, following the closest existing type (below). |
| Validator | `zarr_vectors/validate/conformance.py` | Add the type to `GEOMETRY_LINK_REQ` with its allowed links conventions; a missing type gives the warning `Unknown geometry type: '<type>'`. Add type-specific rules in `validate_conformance`. Levels 1 to 3 need a change only for a new array or key. |
| `Dataset` facade | `zarr_vectors/api/level.py`, `result.py`, `dataset.py` | Add a `_READERS` entry; a `ReadResult.from_<type>` constructor if none fits; a `Dataset.add_<type>` method if users write the type through the facade. |
| Pyramids | `zarr_vectors/multiresolution/` | `build_pyramid` uses `per_object` for every type; check its output passes level 5 validation. Type-specific coarsening is registered from `zarr-vectors-tools` through `registry.py`. |
| Tests | `tests/test_<type>.py` | See below. |
| Docs | `docs/spec/geometry_types/<type>.md`, `docs/tutorials/data_types/` | A spec page following the [spec documentation standards](spec_change_process.md), added to the toctree and matrix in `geometry_types/index.md`; a tutorial. |

Closest existing types:

| Closest to | Module | Functions |
|------------|--------|-----------|
| Unconnected points | `points.py` | `write_points`, `read_points` |
| Ordered vertex sequences | `polylines.py`, `lines.py` | `write_polylines`, `read_polylines`, `write_lines`, `read_lines` |
| Explicit edges | `graphs.py` | `write_graph`, `read_graph` |
| Faces | `meshes.py` | `write_mesh`, `read_mesh` |

### Writer rules

The writer sorts each chunk's vertices into fragments, writes a
fragment index for the sorted rows, and reorders every per-vertex
attribute the same way (see [Fragments](../object_model/fragments.md)).
It records the type and links convention in the root attributes and
updates each level's `vertex_count`. The reader returns the same result
keys as the closest existing reader.

### Tests

`tests/test_<type>.py` covers at least:

- a write and read round trip, with objects that cross chunks;
- a bounding-box read;
- `validate(path, level=5)` passing on a new store and on a store with
  a pyramid;
- a corrupted store failing at the expected level.

See [Compliance testing](test_compliance.md) and
[Validation](../validation/overview.md).
