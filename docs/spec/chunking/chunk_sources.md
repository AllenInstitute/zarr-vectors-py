# Chunk sources (composite levels)

## Terms

**Composite level**
: A level whose objects are made partly or wholly of fragments stored in
  other stores.

**Source axis**
: The leading chunk axis of a composite level, named `"source"`. Value 0
  is the level's own cells; each other value is one other store.

**Fingerprint**
: What a source looked like when the composite level was declared, used
  to detect a source that has been rebuilt since.

---

## Introduction

A composite level presents objects whose geometry lives in other stores,
without copying it. A manifest entry `((s, i, j, k), f)` with `s >= 1`
means fragment `f` of cell `(i, j, k)` in source `s`, which readers
fetch from the source store.

The functions (`set_chunk_sources`, `chunk_sources`, `is_composite`,
`verify_chunk_sources`) are in `zarr_vectors.building`, stability tier
`supported`. A build that supports composite levels lists
`"chunk-sources"` in `zarr_vectors.FEATURES`.

---

## Technical reference

### Level metadata

| Key | Type | Meaning |
|-----|------|---------|
| `chunk_dims` | `list[str]` | `"source"` followed by the spatial axes, for example `["source", "dim0", "dim1", "dim2"]` |
| `chunk_sources` | `list[dict]` | One entry per value of the source axis, in order |

`chunk_sources[0]` is `{"self": true}`. Every other entry names a source:

```json
{
  "ref": {"path": "../../a/source.zarrvectors", "id": "world"},
  "level": 0,
  "transform": [[2.0, 0.0, 0.0, 100.0],
                [0.0, 2.0, 0.0, -3.0],
                [0.0, 0.0, 2.0, 7.0],
                [0.0, 0.0, 0.0, 1.0]],
  "fingerprint": {"object_count": 2}
}
```

| Key | Meaning |
|-----|---------|
| `ref` | An OME-Zarr RFC 8 reference. `path` is absolute, or relative to the composite store's own directory (from `c/composite.zarrvectors`, the example names `a/source.zarrvectors`). `id` is the coordinate system; a zarr-vectors store declares `world` |
| `level` | The source level whose cells are referenced |
| `transform` | `(D+1) × (D+1)` affine from the source's stored positions to the composite level's frame |
| `fingerprint` | `object_count` is recorded and checked; other keys are kept but not checked |

Per-chunk arrays have the source axis first, of length
`len(chunk_sources)`: with two entries and a 21 × 21 × 21 grid,
`vertices` has shape `[2, 21, 21, 21]`. The object index has rank
`len(chunk_dims)`. A composite level cannot also be attribute-chunked.

### Writing a composite level

`set_chunk_sources(level_group, sources, *, fingerprint=True)` records
the table with each source's fingerprint. Call it before allocating the
level's arrays:

```python
import numpy as np
from zarr_vectors.building import (
    create_store, get_resolution_level, open_store, open_write_session,
    create_vertices_array, create_object_index_array, read_all_object_manifests,
    set_chunk_sources, write_object_index)

src_manifests = read_all_object_manifests(
    get_resolution_level(open_store("a/source.zarrvectors"), 0))

root = create_store("c/composite.zarrvectors", bounds=([0.] * 3, [200.] * 3),
                    chunk_shape=(10., 10., 10.), geometry_types=["polyline"])
lg = get_resolution_level(root, 0)
M = np.eye(4); M[:3, :3] *= 2; M[:3, 3] = (100, -3, 7)
table = set_chunk_sources(lg, [
    {"self": True},
    {"ref": {"path": "../../a/source.zarrvectors", "id": "world"},
     "level": 0, "transform": M.tolist()},
])
with open_write_session(lg, bounds=([0.] * 3, [200.] * 3),
                        chunk_shape=(10., 10., 10.), bin_count=len(table)):
    create_vertices_array(lg)
    create_object_index_array(lg)
    write_object_index(lg, {oid: [((1, *cc), f) for cc, f in blocks]
                            for oid, blocks in enumerate(src_manifests)}, 4)
```

Only source-0 cells can be written. Writing a cell with `s >= 1` raises
`ArrayError`.

### Reading

A cell key `s.i.j.k` with `s >= 1` reads source `s`'s cell `i.j.k`.
`vertices` come back through the source's `transform`, in the composite
level's vertex dtype. Every other per-chunk array comes back as stored,
because its row numbers refer to the source's vertex buffer.

Readers that go through `Group.read_bytes`, including the object
readers, need no changes. Each source opens on first use. If its
`object_count` differs from the fingerprint, the read raises
`ArrayError`.

`verify_chunk_sources(level_group)` checks every source at once: it
raises `StoreError` for a source that cannot be opened and `ArrayError`
for a changed one, and returns the table (`None` on an ordinary level).
Call it first in a reader that treats unreadable cells as empty.

### Limits

- A spatial box query over a composite level raises `ArrayError`. Read
  its objects through their manifests.
- Relative source paths resolve only on a local store; other stores
  raise `StoreError`.
- A reader without the `chunk-sources` feature reads source cells as
  empty.

### Validation

Writing level metadata raises `MetadataError` unless `chunk_sources` is
a non-empty list, `chunk_dims` starts with `"source"`, entry 0 is
`{"self": true}`, and every other entry has a `ref` with a non-empty
`path`, a `level` of at least 0, a square `transform` of size at least
2, and a dict `fingerprint` if present (`LevelMetadata.validate`).

[`zarr_vectors.validate`](../validation/overview.md) does not open
sources. On a composite level with no cells of its own, level 3 warns
"no chunk data".
