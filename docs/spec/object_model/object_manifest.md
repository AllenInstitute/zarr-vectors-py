# Object manifest

```{admonition} Format change in ZVF 0.6.0
:class: note

Prior to ZVF 0.6.0 the `object_index/` array was a fixed-shape
`(n_objects, 2)` `int64` table storing one *primary-fragment address*
per object; reading a multi-chunk object required walking
`cross_chunk_links/` forwards from the primary fragment. ZVF 0.6.0
replaced this with a self-contained per-object manifest stored
as a ragged `uint8` blob: every chunk the object touches, and
every fragment within each chunk, is enumerated directly.

The cross-chunk-link walk is no longer required during object
read. Links still encode geometric edges across chunks — as
records with non-zero offsets in the `links/0/<offsets>/` family
(see [Links](links.md)) — but they are not used to discover which
chunks an object touches.
```

```{admonition} Format change in ZVF 0.9.0
:class: note

The `cross_chunk_links/` and `cross_chunk_link_attributes/` families
were merged into `links/` and `link_attributes/`. **Manifests are
unaffected**: they reference vertex fragments only, and have not
referenced links since the walk was removed in 0.6.0.
```

## Terms

**Object manifest**
: The full description of where an object's vertices live across
  the ZVF store. Encoded as a sequence of **manifest blocks**, one
  per chunk the object touches.

**Manifest block**
: One entry inside a manifest: a chunk's coordinates plus a
  *fragment reference* naming the fragments in that chunk that
  belong to this object.

**Manifest block mode**
: How the fragment reference is encoded. Three modes:
  - `mode 0` (single) — exactly one fragment index
  - `mode 1` (range) — a contiguous run `[start, start + count)`
    of fragment indices
  - `mode 2` (explicit) — an arbitrary list of fragment indices

**Manifest blob**
: The byte sequence that encodes one object's full manifest.
  Stored as the OID-th element of the ragged `object_index/manifests`
  array (vlen-bytes codec, ~16K objects per zarr chunk so per-object
  random access reads only one chunk).

**Shared fragment**
: A fragment named by more than one object's manifest. The fragment's
  vertices are stored once and referenced by each manifest. Marked
  per-level with `LevelMetadata.shared_fragments=True` and at the
  store level with the `CAP_SHARED_FRAGMENTS` capability token.

**Primary fragment**
: The first fragment of an object in traversal order. Useful for
  preview rendering and as a stable per-object handle, but no longer
  the only addressable thing in `object_index/` — the new manifest
  format enumerates every fragment.

---

## Introduction

The object manifest is the mechanism that turns "object ID `k`" into
a concrete read plan. Without it, fetching one streamline from a
large tractography store would require scanning every chunk to find
the streamline's vertices.

The pre-0.6 design used a thin primary-address row in `object_index/`
plus a recursive walk of `cross_chunk_links/` to discover continuation
chunks. This walk was the bottleneck for streamline rendering across
many chunks: each hop required a round-trip to fetch the next link.

The post-0.6 design enumerates every chunk and every fragment the
object touches directly in the manifest itself. Reading an object
becomes a fixed-cost decode plus parallel chunk reads — no chain
of dependent fetches. The storage cost is small (a handful of bytes
per chunk per object) and is recovered immediately when shared
fragments at coarsened levels eliminate vertex duplication.

A second consequence: different objects' manifests can both name
the same `(chunk_coords, fragment_index)` pair. The fragment's
vertex rows are stored once; each object recovers them via its
own manifest. This is the **shared fragments** feature, the central
gain of the 0.6.0 rewrite at coarsened pyramid levels.

---

## Technical reference

### `object_index/` group layout

`object_index/` is a Zarr group containing a single ragged Zarr array
named `manifests`:

| Key | Type | Contents |
|-----|------|----------|
| `manifests` | 1-D `vlen-bytes` zarr array of length `num_objects`, chunked at ~16K objects per chunk | Entry `i` is the encoded manifest blob for object `i` |

Group-level metadata (`zv_array`):

```json
{
  "zv_array":    "object_index",
  "num_objects": <int>,
  "sid_ndim":    <int>,
  "layout":      "vlen_manifests_v1"
}
```

The `layout` field discriminates the on-disk container. The value
`"vlen_manifests_v1"` selects the ragged-array layout described here.
Its absence signals the legacy `data` + `offsets` byte-blob layout
described in the *Legacy layout* section below.

To read object `i`'s manifest blob, slice the array (scalar indexing
on a zarr vlen-bytes array yields a 0-d object ndarray; slicing yields
a 1-D ndarray whose element is the bytes object directly):

```python
manifests = level_group.zarr_group["object_index/manifests"]
blob      = manifests[i:i + 1][0]  # one chunk fetch
```

An empty manifest is legal and occupies 4 bytes (`B = 0`).
ID-preserving sparsified levels emit empty manifest blobs for
dropped object IDs rather than removing rows; see the *Object ID
assignment* section below.

#### Why a single ragged array

Pre-vlen ZVF stored `object_index` as two single-chunked byte blobs
(`data` and `offsets`), so reading one object's manifest required
loading the entire offsets table (8 × `num_objects` bytes) and the
entire concatenated data blob, regardless of which OID was requested.
At 1M objects that is 8 MB of offsets plus the full data blob on every
single-object read.

The vlen-bytes ragged array eliminates that amplification. Each zarr
chunk holds ~16K manifest blobs; per-object random access reads only
the chunk holding the requested OID. Bulk reads (the whole `manifests`
array sliced with `[:]`) fetch every chunk in parallel via zarr's
async pipeline.

```{warning}
The Zarr V3 specification for variable-length byte arrays is still in
development (tracked at
[zarr-extensions](https://github.com/zarr-developers/zarr-extensions/tree/main/data-types)).
ZVF 0.x stores written with `vlen-bytes` may need to be re-encoded if
the eventual spec lands incompatibly.
```

#### Legacy layout

Stores written before the `vlen_manifests_v1` layout was introduced
have two byte-blob children instead of `manifests`:

| Key | Contents |
|-----|----------|
| `data` | concatenated manifest blobs for all `num_objects` objects, in OID order |
| `offsets` | flat `int64` array of length `num_objects`; entry `i` is the byte offset of object `i`'s blob within `data` |

To read object `i`'s manifest blob from the legacy layout:

```python
start = offsets[i]
end   = offsets[i + 1] if i + 1 < num_objects else len(data)
blob  = data[start:end]
```

```{important}
`offsets` has length `num_objects`, **not** `num_objects + 1`. The
last object's blob extends from `offsets[num_objects - 1]` to the
end of `data`. Readers MUST handle the final entry as an open-ended
slice.
```

Readers in `zarr_vectors.core.arrays` (`read_object_manifest` and
`read_all_object_manifests`) auto-detect both layouts via the `layout`
group attribute and dispatch accordingly. Writers always emit the new
layout going forward; no migration tool is provided — legacy stores
remain readable as-is.

### Manifest blob byte layout

```
HEADER
  uint32 num_blocks B

For each block (1 of B):
  int64  chunk_coords[sid_ndim]
  uint8  mode
  if mode == 0:   # single fragment
    int64 fragment_index
  elif mode == 1: # contiguous range of fragments
    int64 start, int64 count
  elif mode == 2: # explicit list
    uint32 count
    int64  fragment_indices[count]
```

All fragment references are **chunk-local** — they index into the
named chunk's `vertex_fragments/<chunk_coords>` blob only. This
preserves chunk-write independence: a chunk can be written without
coordinating fragment numbering with any other chunk.

An empty manifest is 4 bytes: `B = 0`.

### Mode selection rules

The reference writer (`encode_object_manifest_blocks` in
[`zarr_vectors.encoding.fragments`](../../api/index.rst)) chooses
the mode per block:

| Mode | When |
|------|------|
| Single (0) | The object touches exactly one fragment in this chunk. Most common for small or point-like objects whose every bin contributes exactly one fragment. |
| Range (1)  | The object touches a contiguous run `[start, start + count)` of fragment indices in this chunk. Most common for long polylines / streamlines whose path through one chunk visits adjacent bins in succession. |
| Explicit (2) | The object touches a non-contiguous set of fragment indices in this chunk. Most common after coarsening, where the object's metavertices may be scattered across the chunk's fragment list. |

A writer that passes an `int` chooses mode 0; a `(start, count)`
tuple chooses mode 1; a 1-D integer array auto-detects mode 1 (if
the array equals `arange(start, start + len)`) or mode 2 otherwise.
Pass `force_explicit=True` to disable the auto-promotion — useful
for round-trip testing of the explicit path.

### Reading an object's vertices

```python
def read_object_vertices(level_group, object_id):
    """Read and concatenate all vertices of an object, in manifest order."""
    meta     = level_group.read_array_meta(OBJECT_INDEX)
    sid_ndim = meta["sid_ndim"]

    # New (vlen_manifests_v1) layout — one chunk fetch.
    manifests = level_group.zarr_group["object_index/manifests"]
    blob      = manifests[object_id:object_id + 1][0]

    out = []
    for chunk_coords, fragment_ref in decode_object_manifest_blocks(blob, sid_ndim):
        fidx     = read_fragment_index(
            level_group, VERTEX_FRAGMENTS, chunk_coords,
        )
        vertices = level_group.read_vertices(chunk_coords)

        if isinstance(fragment_ref, int):
            rows = fidx.indices(fragment_ref)
        elif isinstance(fragment_ref, tuple):
            s, c = fragment_ref
            rows = np.concatenate([fidx.indices(f) for f in range(s, s + c)])
        else:  # 1-D int64 array
            rows = np.concatenate([fidx.indices(int(f)) for f in fragment_ref])

        out.append(vertices[rows, :])

    return np.concatenate(out, axis=0)
```

The reference reader in `zarr_vectors.core.arrays` is `read_object_manifest`
(returns the decoded block list) and downstream geometry-type readers
that consume it. Note the simplification compared with pre-0.6: there
is no link-walk loop, no `visited_chunks` cycle guard, no chain of
dependent reads — the manifest itself enumerates every chunk.

### Shared fragments

When a level is written with `LevelMetadata.shared_fragments=True`:

- A fragment in `vertex_fragments/<chunk_coords>` MAY be named by
  more than one object's manifest.
- The fragment's vertex rows are stored once in `vertices/<chunk_coords>`.
- The store advertises `CAP_SHARED_FRAGMENTS` in
  `RootMetadata.format_capabilities`.

**Writer responsibility.** The coarsening pipeline emits one fragment
per *distinct* metavertex per chunk, then writes each object's
manifest to reference the metavertices it touches. When two coarsened
objects share a metavertex, both manifests name the same fragment
index in the same chunk.

**Reader responsibility.** None. The shared-fragments case is fully
transparent to read paths — the same `fidx.indices(f)` call resolves
the vertex rows whether the fragment is shared or not. Code that
naively unions `(chunk_coords, fragment_index)` pairs across all
manifests must not assume the union is disjoint when
`shared_fragments=True`.

**Storage rationale.** At a typical coarsening level with `bin_ratio
= (2, 2, 2)` and an `object_sparsity` retention of ~50%, naive
replication would multiply storage by the per-metavertex
participation count `k` (often 4–16 for densely connected tract
data). Shared fragments eliminate this multiplier; the cost is the
per-block manifest bytes naming the metavertex, which is amortised
over `k` objects.

### Object ID assignment

Object IDs are non-negative integers assigned at write time. At
level 0 they are dense (no gaps) starting from 0. The maximum object
ID is `num_objects − 1`.

For levels with `preserves_object_ids=True` (the ID-preserving
sparsification regime), `num_objects` equals the parent level's
`num_objects`. Dropped objects appear as empty manifest blobs
(4 bytes, `B = 0`); their OIDs are *preserved* rather than
*remapped*. The companion `present_mask` sidecar in
`object_attributes/` flags which rows are real. See
[Object attributes](object_attributes.md).

For levels with `preserves_object_ids=False`, surviving objects are
renumbered to `[0, num_retained)`. The mapping back to level-0 OIDs
is written as a per-object attribute.

Object IDs are stable across reads of a given store but MAY change
when a store is rechunked.

### Decoding a manifest

Given a manifest blob `raw`, the decoder is straightforward — there
is no recursion, no graph walk, no joins across arrays:

```python
def decode_object_manifest_blocks(raw, sid_ndim):
    b = struct.unpack_from("<I", raw, 0)[0]
    offset = 4
    blocks = []
    for _ in range(b):
        coords = tuple(int(c) for c in np.frombuffer(
            raw, dtype=np.int64, count=sid_ndim, offset=offset,
        ))
        offset += sid_ndim * 8

        mode = struct.unpack_from("<B", raw, offset)[0]
        offset += 1

        if mode == 0:  # single
            idx = struct.unpack_from("<q", raw, offset)[0]
            offset += 8
            blocks.append((coords, int(idx)))
        elif mode == 1:  # range
            start, count = struct.unpack_from("<qq", raw, offset)
            offset += 16
            blocks.append((coords, (int(start), int(count))))
        else:  # explicit
            count = struct.unpack_from("<I", raw, offset)[0]
            offset += 4
            indices = np.frombuffer(
                raw, dtype=np.int64, count=count, offset=offset,
            ).copy()
            offset += count * 8
            blocks.append((coords, indices))
    return blocks
```

Decoded `fragment_ref` types: `int` for mode 0, `(start, count)` tuple
for mode 1, `np.ndarray[int64]` for mode 2. Callers that prefer a
uniform shape (always an array of indices) can map over the result.

### Validation

**L1 (structural).**
- `object_index/` group exists for all discrete-object geometry types.
- Exactly one of these layouts is present:
  - `object_index/manifests` array (new layout: `layout ==
    "vlen_manifests_v1"`); or
  - both `object_index/data` and `object_index/offsets` byte entries
    (legacy layout: `layout` absent).
- `object_index/`'s `zv_array` metadata names `num_objects` and
  `sid_ndim`.

**L2 (metadata).**
- New layout: `manifests.shape == (num_objects,)` and `manifests.dtype
  == object` (vlen-bytes).
- Legacy layout: `len(offsets) == num_objects`; `offsets` is
  monotonically non-decreasing; `offsets[0] == 0` when `num_objects >
  0`; `offsets[i] <= len(data)` for all `i`.

**L3 (consistency).**
- Each manifest blob decodes without error via
  `decode_object_manifest_blocks(blob, sid_ndim)`.
- For every decoded block: `chunk_coords` is a valid chunk in the
  level's chunk grid.
- For every decoded `fragment_index`: `fragment_index <
  ChunkFragmentIndex.num_fragments` in the named chunk's
  `vertex_fragments/<chunk_coords>` blob.
- For range mode: `start + count <=` the chunk's `num_fragments`.
- When `shared_fragments == False` at this level: the union of
  `(chunk_coords, fragment_index)` pairs across all manifests is
  disjoint (no fragment is named twice).
- Legacy layout only: trailing bytes in `data` after
  `offsets[num_objects - 1]` plus the last blob's encoded length are
  zero.

### Object index at coarser pyramid levels

`object_index/` is recomputed per-level: each level has its own
manifest list referring to that level's `vertex_fragments/` blobs.
At levels with `shared_fragments=True`, manifests for closely
spaced objects converge on shared metavertices; reading either
object yields the same vertex row from one underlying chunk.

If a coarse level uses ID-preserving sparsification
(`preserves_object_ids=True`), dropped objects have empty
manifest blobs. Otherwise the surviving objects are renumbered
densely and a mapping back to level-0 OIDs is stored in
`object_attributes/level0_object_id/` at the coarse level.
