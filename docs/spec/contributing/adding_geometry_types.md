# Adding a new geometry type

## Terms

**Geometry type constant**
: The string value stored in `"geometry_type"` in root `.zattrs` that
  uniquely identifies the type. Must be a lowercase alphanumeric string
  with underscores (e.g. `"point_cloud"`, `"streamline"`).

**Type-specific arrays**
: Arrays that are required or optional for a given geometry type but not
  for all types (e.g. `object_index/` for types with discrete objects, or
  a `links/<delta>/` family with `link_width >= 3` for `mesh`).

**Write function**
: The `write_<type>()` function in `zarr_vectors/types/<type>.py` that
  accepts raw geometry data and writes a conforming ZVF store.

**Read function**
: The `read_<type>()` function that reads a ZVF store of the given type
  and returns a typed result dict.

---

## Introduction

Adding a new geometry type to ZVF requires changes at every layer of the
stack: a new type constant, type-specific arrays, a writer, a reader, a
validator, a lazy loader, and documentation. This page provides a step-by-
step checklist for contributors adding a new type, using the existing types
as reference implementations.

Before beginning implementation, ensure the new type has been ratified via
the RFC process (see [Spec change process](spec_change_process.md)).

---

## Technical reference

### Step 0 — Choose a type constant and arrays

Decide:
1. The type constant string (e.g. `"voxel_cloud"` for a new type).
2. Which existing arrays it requires (all types need `vertices/` and
   `vertex_fragments/`; types with intra-chunk connectivity also need
   `link_fragments/`).
3. What new arrays, if any, it introduces.
4. Whether it has discrete objects (requires `object_index/`).
5. Its link-family policy: `link_width`, and whether the family is
   `directed` (endpoint order is data) or undirected (canonical-sorted
   and deduplicated, with `perm_idx` recovering input order).
6. Whether its connectivity is explicit or implicit — an
   `implicit_sequential` type emits no intra-chunk links at all, since
   vertex order within a fragment carries the topology.
7. What type-specific metadata keys it adds to root `.zattrs`.

### Step 1 — Add the type constant

```python
# zarr_vectors/constants.py
GEOM_VOXEL_CLOUD = "voxel_cloud"    # ← new constant

GEOMETRY_TYPES = {
    GEOM_POINT_CLOUD, GEOM_LINE, GEOM_POLYLINE, GEOM_STREAMLINE,
    GEOM_GRAPH, GEOM_SKELETON, GEOM_MESH,
    GEOM_VOXEL_CLOUD,    # ← register new constant
}
```

### Step 2 — Define required and optional arrays

```python
# zarr_vectors/validate/structure.py
REQUIRED_ARRAYS = {
    ...
    "voxel_cloud": [
        "vertices/",
        "vertex_fragments/",
        "links/voxel_ids/",      # new type-specific array
    ],
}

OPTIONAL_ARRAYS = {
    ...
    "voxel_cloud": [
        "attributes/",
        "object_index/",
    ],
}
```

### Step 3 — Write the write function

Create `zarr_vectors/types/voxel_cloud.py`:

```python
def write_voxel_cloud(
    store_path: str,
    positions: npt.NDArray[np.floating],
    voxel_ids: npt.NDArray[np.integer],
    *,
    chunk_shape: ChunkShape,
    bin_shape: BinShape | None = None,
    attributes: dict[str, npt.NDArray] | None = None,
    **kwargs,
) -> None:
    """Write a voxel cloud store.

    Parameters
    ----------
    store_path:
        Path to the output .zarrvectors directory.
    positions:
        Vertex positions, shape (N, D).
    voxel_ids:
        Integer voxel ID for each vertex, shape (N,).
    chunk_shape:
        Spatial extent of each chunk in coordinate units.
    bin_shape:
        Supervoxel bin shape. Defaults to chunk_shape.
    attributes:
        Optional per-vertex attribute arrays.
    """
    # Validate inputs
    _validate_write_args(positions, chunk_shape, bin_shape)

    # Open or create the store
    root = _open_or_create_store(store_path, geometry_type=GEOM_VOXEL_CLOUD,
                                 chunk_shape=chunk_shape, bin_shape=bin_shape)

    # Partition vertices into chunks
    chunk_map = _partition_into_chunks(positions, chunk_shape)

    for chunk_coord, (chunk_verts, chunk_indices) in chunk_map.items():
        # Sort into fragment order and emit one range fragment per non-empty bin
        sorted_verts, order, fragments = build_level0_fragments(
            chunk_verts, chunk_coord, chunk_shape, bin_shape
        )
        # Write vertices
        _write_chunk_array(root, "0/vertices", chunk_coord,
                           sorted_verts)
        # Write fragment index (binary blob, one per chunk)
        _write_chunk_bytes(root, "0/vertex_fragments",
                           chunk_coord, encode_fragments(fragments))
        # Write type-specific array
        sorted_voxel_ids = voxel_ids[chunk_indices][order]
        _write_chunk_array(root, "0/links/voxel_ids",
                           chunk_coord, sorted_voxel_ids)
        # Write attributes
        if attributes:
            for name, values in attributes.items():
                _write_chunk_array(root, f"0/attributes/{name}",
                                   chunk_coord, values[chunk_indices][order])

    # Write multiscale metadata
    write_multiscale_metadata(root)
    write_metadata_json(root)
```

Key invariants to maintain in the writer:
- Vertices must be in fragment order (bin-sorted) within each chunk.
- The fragment list passed to `encode_fragments` must reference the
  sorted vertex rows; emit one `(start, count)` range per non-empty bin.
- All attribute arrays must be reordered with the same `order` permutation.

See [`zarr_vectors.encoding.fragments`](../../api/index.rst) for
`encode_fragments` and the canonical `build_level0_fragments` helper.

### Step 4 — Write the read function

```python
def read_voxel_cloud(
    store_path: str,
    *,
    bbox: BoundingBox | None = None,
    level: int = 0,
    attributes: list[str] | None = None,
) -> dict[str, Any]:
    """Read a voxel cloud store."""
    root = open_store(store_path, mode="r")
    _assert_geometry_type(root, GEOM_VOXEL_CLOUD)

    verts, fragments = _read_bbox_fragments(root, level, bbox)
    voxel_ids = _read_bbox_array(root, level, "links/voxel_ids", bbox,
                                 fragments)
    attrs = _read_attributes(root, level, attributes, fragments)

    return {
        "vertex_count": len(verts),
        "positions":    verts,
        "voxel_ids":    voxel_ids,
        "attributes":   attrs,
        "level":        level,
        "bbox_used":    bbox,
    }
```

### Step 5 — Add L1 structural checks

```python
# zarr_vectors/validate/structure.py
def check_type_specific_l1(root, geometry_type, result):
    ...
    if geometry_type == GEOM_VOXEL_CLOUD:
        _assert_array_exists(root, "0/links/voxel_ids/", result)
```

### Step 6 — Add L2 metadata checks

```python
# zarr_vectors/validate/metadata.py
def check_type_specific_l2(root, geometry_type, result):
    ...
    if geometry_type == GEOM_VOXEL_CLOUD:
        voxel_ids_array = root["0"]["links"]["voxel_ids"]
        _check_dtype(voxel_ids_array, expected="int64", result=result,
                     check_name="voxel_ids_dtype")
        _check_last_dim(voxel_ids_array, expected=None,  # scalar per vertex
                        result=result, check_name="voxel_ids_shape")
```

### Step 7 — Add L3 consistency checks

```python
# zarr_vectors/validate/consistency.py
def check_type_specific_l3(root, geometry_type, chunk_coord, result):
    ...
    if geometry_type == GEOM_VOXEL_CLOUD:
        voxel_ids = _read_chunk(root, "links/voxel_ids", chunk_coord)
        vertex_count = _read_vertex_count(root, chunk_coord)
        _check_equal(len(voxel_ids), vertex_count,
                     "voxel_ids_length_matches", result)
```

### Step 8 — Add lazy loader support

```python
# zarr_vectors/lazy.py  (or lazy/voxel_cloud.py)
class LazyVoxelCloud(LazyStore):
    @property
    def voxel_ids(self) -> LazyArray:
        return LazyArray(self._root, "links/voxel_ids", self._level)
```

### Step 9 — Add coarsening support

If the type supports multi-resolution pyramids (it should), add a
coarsening function:

```python
# zarr_vectors/multiresolution/voxel_cloud.py
def coarsen_voxel_cloud_level(root, source_level, target_level, bin_ratio):
    """Coarsen a voxel cloud level. Voxel IDs are aggregated by majority vote."""
    ...
```

And register it in `zarr_vectors/multiresolution/coarsen.py`:

```python
COARSEN_FUNCTIONS = {
    GEOM_POINT_CLOUD:  coarsen_point_cloud_level,
    ...
    GEOM_VOXEL_CLOUD:  coarsen_voxel_cloud_level,
}
```

### Step 10 — Write tests

Add a test module `tests/test_voxel_cloud.py` covering:

- [ ] Round-trip write → read produces identical data.
- [ ] Bbox query returns correct vertices.
- [ ] Multi-resolution pyramid is correctly constructed.
- [ ] L1–L3 validation passes on a freshly written store.
- [ ] L3 validation catches intentionally corrupted `voxel_ids` alignment.
- [ ] Ingest/export round-trip (if applicable).

Add a reference fixture store (see [Compliance testing](test_compliance.md)).

### Step 11 — Write documentation

Add these pages (following the spec page template):

- `docs/spec/geometry_types/voxel_cloud.md` — type spec page.
- `docs/tutorials/data_types/voxel_cloud.md` — tutorial.

Update `docs/spec/geometry_types/index.md` to include the new type in the
comparison matrix and type selection guide.

### Checklist summary

- [ ] Type constant in `constants.py`
- [ ] Registered in `GEOMETRY_TYPES` set
- [ ] Required/optional arrays in `validate/structure.py`
- [ ] Write function in `types/<type>.py`
- [ ] Read function in `types/<type>.py`
- [ ] L1 structural checks
- [ ] L2 metadata checks
- [ ] L3 consistency checks
- [ ] Lazy loader support
- [ ] Coarsening support
- [ ] Test module with round-trip, bbox, pyramid, and validation tests
- [ ] Reference fixture store
- [ ] Spec page in `docs/spec/geometry_types/`
- [ ] Tutorial in `docs/tutorials/data_types/`
- [ ] Comparison matrix updated
- [ ] Changelog entry
