# Object manifest

```{admonition} Format change in Zarr Vectors 0.6.0
:class: note

Prior to Zarr Vectors 0.6.0 the `object_index/` array was a fixed-shape
`(n_objects, 2)` `int64` table storing one *primary-fragment address*
per object; reading a multi-chunk object required walking
`cross_chunk_links/` forwards from the primary fragment. Zarr Vectors 0.6.0
replaced this with a self-contained per-object manifest stored
as one variable-length byte blob per object: every chunk the object
touches, and every fragment within each chunk, is enumerated directly.

The cross-chunk-link walk is no longer required during object
read. Links still encode geometric edges across chunks — as
records with non-zero offsets in the `links/0/<offsets>/` family
(see [Links](links.md)) — but they are not used to discover which
chunks an object touches.
```

```{admonition} Format change in Zarr Vectors 0.9.0
:class: note

The `cross_chunk_links/` and `cross_chunk_link_attributes/` families
were merged into `links/` and `link_attributes/`. **Manifests are
unaffected**: they reference vertex fragments only, and have not
referenced links since the walk was removed in 0.6.0.
```

```{admonition} Object ids are stored, not implied (layout v2)
:class: note

The current writer stamps `layout = "vlen_manifests_v2"`. Under v1 an object's
id *was* its row index, so one object with id 20 000 000 allocated twenty
million rows. Under v2 rows are dense and the ids live in a sibling
`object_ids` array, so ids may be sparse, large or 64-bit. v1 stores read
unchanged: a v1 store is simply one whose id table is the identity. See
[Object ids and rows](#object-ids-and-rows).
```

## Terms

**Object manifest**
: The full description of where an object's vertices live across
  the Zarr Vectors store. Encoded as a sequence of **manifest blocks**, one
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
  Stored as one row of the ragged `object_index/manifests` array
  (vlen-bytes codec, 16 384 rows per Zarr chunk so a single-object read
  fetches at most one chunk).

**Row vs. object id**
: A *row* is a position in `object_index/manifests`; an *object id* is the
  name a user knows the object by. They are equal in a densely numbered store
  (the default) and differ otherwise. `object_index/object_ids[row]` is the
  id stored at a row.

**Shared fragment**
: A fragment named by more than one object's manifest. The fragment's
  vertices are stored once and referenced by each manifest. Marked
  per-level with `LevelMetadata.shared_fragments=True` and at the
  store level with the `CAP_SHARED_FRAGMENTS` capability token.

**Primary fragment**
: The first fragment of an object in traversal order (the first block of its
  manifest). Useful for preview rendering and as a stable per-object handle;
  the manifest enumerates every other fragment too.

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

`object_index/` is a Zarr group holding two 1-D arrays and a block of group
attributes:

| Key | Type | Contents |
|-----|------|----------|
| `manifests` | 1-D `vlen-bytes` array, shape `(num_objects,)`, Zarr chunk `min(16384, num_objects)` | Row `r` is the encoded manifest blob of the object stored at row `r` |
| `object_ids` | 1-D `int64` array, shape `(num_objects,)` | Row `r` is the object id stored at row `r` (layout v2 only) |

Group attributes:

```json
{
  "zv_array":         "object_index",
  "num_objects":      <int>,      // number of ROWS (slots), present or not
  "num_present":      <int>,      // rows whose manifest is non-empty
  "sid_ndim":         <int>,      // rank of chunk coordinates in the manifests
  "layout":           "vlen_manifests_v2",
  "object_ids_sorted": true       // ids ascend, so lookups can binary-search
}
```

`num_objects` is the **slot** count and `num_present` the count of slots that
actually hold an object. They differ by exactly the objects an ID-preserving
pyramid level dropped (see [Object ids and rows](#object-ids-and-rows)).
`sid_ndim` is `ndim + 1` for an attribute-chunked store, because the leading
attribute-bin axis is part of every chunk key (see
[Attribute-based chunking](../chunking/attribute_chunking.md)).

The `layout` field discriminates the container. `"vlen_manifests_v2"` is what
the writer emits; `"vlen_manifests_v1"` (no `object_ids` array, id == row) is
still read. Any other value, including the pre-0.8.1 `data` + `offsets` pair,
is **not readable** and raises an error telling the caller to rewrite from
source.

A directory tree for a two-object store:

```
0/object_index/
├── zarr.json          # attributes above
├── manifests/
│   ├── zarr.json      # variable_length_bytes, shape [2], chunks [2]
│   └── c/0            # both manifests, vlen-encoded together
└── object_ids/
    ├── zarr.json      # int64, shape [2]
    └── c/0
```

To read the manifest of object id `i`, resolve the id to a row and slice the
array (scalar indexing on a Zarr vlen-bytes array yields a 0-d object ndarray;
slicing yields a 1-D ndarray whose element is the `bytes` object):

```python
from zarr_vectors.core.arrays import object_rows_for_ids, read_object_manifest

found, rows = object_rows_for_ids(level_group, [i])    # identity for a v1 store
manifests = level_group.zarr_group["object_index/manifests"]
blob      = manifests[int(rows[0]):int(rows[0]) + 1][0]   # one chunk fetch

# or, decoded in one call:
manifest = read_object_manifest(level_group, i)
# -> [((0, 0, 0), 0), ((0, 0, 1), 3), ...]   (chunk_coords, fragment_index)
```

#### Why a single ragged array

Pre-vlen Zarr Vectors stored `object_index` as two single-chunked byte blobs
(`data` and `offsets`), so reading one object's manifest required loading the
entire offsets table (8 × `num_objects` bytes) and the entire concatenated data
blob, regardless of which object was requested. At 1M objects that is 8 MB of
offsets plus the full data blob on every single-object read.

The vlen-bytes ragged array eliminates that amplification. Each Zarr chunk
holds 16 384 manifest blobs; per-object random access reads only the chunk
holding the requested row. Bulk reads fetch every chunk in parallel via zarr's
async pipeline.

```{warning}
The Zarr V3 specification for variable-length byte arrays is still in
development (tracked at
[zarr-extensions](https://github.com/zarr-developers/zarr-extensions/tree/main/data-types)).
Zarr Vectors 0.x stores written with `vlen-bytes` may need to be re-encoded if
the eventual spec lands incompatibly. zarr-python emits an
`UnstableSpecificationWarning` when opening these arrays; the library
silences it internally.
```

#### Limits

Because rows are dense under v2, the index is as long as the number of
objects, not the largest id. Writers still refuse more than `2**26`
(`OBJECT_INDEX_MAX_ROWS`, about 67 million) rows, and warn when an index of at
least `2**20` rows is more than 98 % empty padding. Remap segmentation-style
ids to a dense range and keep the originals as an object attribute.

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

The bulk writer `write_object_index` currently emits **mode 0 only**: it
encodes each `(chunk_coords, fragment_index)` pair of a manifest as its own
single-fragment block. Modes 1 and 2 are part of the format and every reader
must accept them, but nothing in the core writers produces them yet; they are
available through `encode_object_manifest_blocks`.

A worked example. With `sid_ndim = 3`, the manifest
`[((0, 0, 0), 0), ((0, 0, 1), (2, 3))]` — fragment 0 of chunk `(0,0,0)` and
fragments 2, 3, 4 of chunk `(0,0,1)` — encodes to 78 bytes:

```python
from zarr_vectors.encoding.fragments import encode_object_manifest_blocks

blob = encode_object_manifest_blocks(
    [((0, 0, 0), 0),        # int      -> mode 0: 24 + 1 + 8  = 33 bytes
     ((0, 0, 1), (2, 3))],  # (s, n)   -> mode 1: 24 + 1 + 16 = 41 bytes
    sid_ndim=3,
)
assert len(blob) == 4 + 33 + 41          # uint32 block count + the two blocks
```

A writer that passes an `int` chooses mode 0; a `(start, count)`
tuple chooses mode 1; a 1-D integer array auto-detects mode 1 (if
the array equals `arange(start, start + len)`) or mode 2 otherwise.
Pass `force_explicit=True` to disable the auto-promotion — useful
for round-trip testing of the explicit path.

### Reading an object's vertices

```python
from zarr_vectors.core.arrays import read_object_manifest, read_object_vertices

manifest = read_object_manifest(level_group, object_id)
# [((i, j, k), fragment_index), ...]  -- ranges and explicit lists are expanded
# to one tuple per fragment, in manifest order

fragments = read_object_vertices(level_group, object_id, dtype="float32", ndim=3)
# list of (n_k, 3) arrays, one per fragment, in reconstruction order
```

Internally this is: resolve the id to a row, fetch that row of
`object_index/manifests`, decode its blocks with
`decode_object_manifest_blocks(blob, sid_ndim)`, then for each
`(chunk_coords, fragment_ref)` decode that chunk's `vertex_fragments` blob,
turn the reference into row indices with `ChunkFragmentIndex.indices(f)` (for
a mode-1 block, once per `f` in `range(start, start + count)`), and slice the
chunk's `vertices` cell. Chunks the object touches are prefetched in one
batch.

Note the simplification compared with pre-0.6: there is no link-walk loop, no
`visited_chunks` cycle guard, no chain of dependent reads — the manifest
itself enumerates every chunk.

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

### Object ids and rows

An object id is a non-negative integer chosen at write time. Bulk writers
number objects `0 … n-1`, so at level 0 `row == id` and `object_ids` is simply
`arange(n)` (stamped `object_ids_sorted = true`).

Rows are dense; ids are whatever `object_ids` says. Three consequences:

- **Sparse or large ids are fine.** Writing ids `{7, 4_000_000_000_000}`
  creates two rows, not four trillion.
- **Appending** an id the level does not hold extends the index by one row and
  appends to `object_ids`; if the new ids no longer ascend, `object_ids_sorted`
  is cleared and lookups sort the table first.
- **Everything positional is positional in *rows*.** `object_attributes/<name>`
  row `r` belongs to manifest row `r`. Convert with
  `object_rows_for_ids` / `object_ids_for_rows`; do not write
  `for oid, m in enumerate(read_all_object_manifests(...))` unless ids are
  known to be dense — use `read_object_manifest_rows`, which returns the ids
  alongside.

```python
import numpy as np
from zarr_vectors.core.arrays import (
    object_ids_for_rows, object_rows_for_ids, read_object_manifest_rows,
)

ids, manifests = read_object_manifest_rows(level_group)       # row order
found, rows    = object_rows_for_ids(level_group, [3, 99])    # unknown ids dropped
```

#### Pyramid levels

For levels with `preserves_object_ids = true` — which is every level the core
coarsener (`coarsen_level` / `build_pyramid`) produces — `inherited_num_objects`
records the parent's id space and **every parent id keeps a row**. A dropped
object's row holds the 4-byte empty manifest (`B = 0`); its id is *preserved*,
not remapped. `num_objects` therefore equals the parent's, and `num_present`
counts the survivors. Presence is *the manifest being empty*: there is no
`present_mask` sidecar. Per-object attribute columns at such a level carry the
same rows, with dropped rows set to the array's `fill_value` (see
[Attributes](object_attributes.md#per-object-attributes-object_attributesname)).

`preserves_object_ids = false` is the default for levels written by other
means. The core library does not renumber surviving objects or write a mapping
back to level 0 anywhere. (The non-spatial `rechunk()` engine does renumber
objects with a running counter and writes no mapping; the spatial
`rechunk_spatial()` preserves ids.) Object ids are stable across reads of a
given store but may change when a store is rechunked by `rechunk()`.

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

**L1 (structural).** `object_index` is listed when present. The L1 validator
does not require it.

**What a conforming store satisfies.** Only the manifest checks marked "L3"
are enforced by the shipped validator; the rest are writer obligations.

- `object_index`'s attributes carry `num_objects` and `sid_ndim`, and
  `layout` is `"vlen_manifests_v1"` or `"vlen_manifests_v2"`.
- `manifests.shape == (num_objects,)`; under v2 `object_ids.shape ==
  (num_objects,)`, and `object_ids_sorted` is true only if the ids ascend.
- Each manifest blob decodes without error, with no trailing bytes.
- (L3) For every decoded block, `chunk_coords` names a chunk that exists at
  this level, and every fragment index — range blocks are checked through
  their expansion — is below that chunk's fragment count.
- When `shared_fragments == False` at this level, no `(chunk_coords,
  fragment_index)` pair is named by more than one manifest.

### Object index at coarser pyramid levels

`object_index/` is recomputed per level: each level has its own manifest list
referring to that level's `vertex_fragments/` blobs. At levels with
`shared_fragments=True`, manifests for closely spaced objects converge on
shared metavertices; reading either object yields the same vertex row from one
underlying chunk. Dropped objects have empty manifests and keep their id and
row; see [Object ids and rows](#object-ids-and-rows).
