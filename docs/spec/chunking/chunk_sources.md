# Chunk sources (composite levels)

A **composite level** holds objects made partly of *another store's*
fragments. Its chunk keys have a leading `source` axis, declared the way an
attribute-chunked level declares its bin axis
([attribute chunking](attribute_chunking.md)); object manifests are unchanged
apart from that one extra coordinate. Nothing is copied from a source: a cell
of the level names a source's cell, and readers resolve it.

## Metadata

`LevelMetadata` (the level group's `zarr_vectors_level` attribute) gains:

| Field           | Type         | Description |
|-----------------|--------------|-------------|
| `chunk_dims`    | `list[str]`  | Leads with `"source"`, then the spatial axes, e.g. `["source", "dim0", "dim1", "dim2"]`. |
| `chunk_sources` | `list[dict]` | One entry per value of the source axis, in order. |

`chunk_sources[0]` is `{"self": true}`: the level's own cells, written and
read as any level's. Every other entry names a source:

```json
{"ref": {"path": "../member.zarrvectors", "id": "world"},
 "level": 0,
 "transform": [[2, 0, 0, 100], [0, 2, 0, -3], [0, 0, 2, 7], [0, 0, 0, 1]],
 "fingerprint": {"object_count": 1234}}
```

- `ref` is an OME-Zarr RFC-8 `Reference`: `path` is relative to this store's
  root (or absolute), `id` the coordinate system it binds to (a zarr-vectors
  store declares `world`). The same reference can sit as a child node of the
  store's `ome` collection node, so a generic collection reader finds the
  sources too.
- `level` is the source level whose cells and fragments are referenced.
- `transform` is a `(D+1) x (D+1)` affine taking the source's stored vertex
  positions to this level's.
- `fingerprint` records what the source was when the level was written.
  zarr-vectors fills in and checks `object_count`; any other key a writer
  records is carried, not checked.

A composite level is not also attribute-chunked: its one leading axis is the
source. Its per-chunk arrays are allocated with the leading axis
(`bin_count = len(chunk_sources)`); only source-0 cells are ever written. Its
object index's `sid_ndim` is `len(chunk_dims)` (the source and the spatial
axes), which the manifest writers and `commit_object_index` take by default.

## Reading

A cell key `s.z.y.x` with `s >= 1` is the source level's cell `z.y.x`:

- `vertices` come back through the source's `transform`, in this level's frame
  and its declared vertex dtype;
- every other per-chunk array -- fragment indexes, vertex attributes, links --
  comes back as stored, since its row numbers refer to the source's vertex
  buffer, which is what is read.

So an object fragment `(s, z, y, x, f)` is fragment `f` of the source's cell
`(z, y, x)`, and every reader that goes through `Group.read_bytes` (the object
readers, fragment indexes, attributes) reads composite objects unchanged. A
source is opened on first use; one whose `object_count` no longer matches its
fingerprint has been rebuilt, and reading it raises rather than returning
geometry from a different build. A reader that treats a cell it cannot read as
empty would see a rebuilt source as missing objects instead, so it calls
`verify_chunk_sources(level)` first, which opens and checks every source at
once (and does nothing on an ordinary level).

## Not supported (v1)

- **Writing a source cell.** A composite writes only source-0 cells; a write
  to `s >= 1` raises.
- **Spatial box queries.** A source's cells are in the source's own grid, so the
  composite's grid cannot locate them; a box query raises. Read a composite's
  objects through their manifests.
- **Remote composites.** Relative source paths resolve on a local store.
- **Older readers.** A build without the `chunk-sources` feature
  (`zarr_vectors._api_version.FEATURES`) does not know the field and reads
  source cells as empty; a consumer opening composites requires the feature.
