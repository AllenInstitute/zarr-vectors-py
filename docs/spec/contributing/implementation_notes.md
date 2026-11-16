# Implementation notes

Why parts of `zarr_vectors` are built the way they are. These are notes for
maintainers; the docstrings say *what* each function does, this page keeps
the reasoning that would otherwise be lost.

## Writing and presence

**Shard transactions** (`core.shard_txn`, `building.shard_transaction`).
Without one, a task writes into the live store and each write is a
read-modify-write of the shard object. A retry then starts from its failed
attempt's half-written shards, and a link cell, which appends a row group per
call, doubles its records. A transaction stages every cell write (all writers
go through `Group.write_bytes` / `Group.write_cells`) and on exit encodes each
touched shard with zarr's own sharding codec against a staging store. Under
`replace` the target key is hidden, so a shard is rebuilt from the staged
cells alone and an array in scope that got no cells loses its shard; under
`merge` nothing is hidden. Each object is written as `<object>.<token>.partial`,
fsynced when `durable`, renamed into place, and the directories fsynced.
Re-running a task converges because zarr encodes a shard from its cells in a
fixed order: the same cells and codecs give the same bytes. With `io_threads`,
encode/write/fsync run on the shared writer pool and only the renames wait for
all of them; a failure on any thread publishes nothing and leaves no partial
behind. Non-local stores publish with one `set` (or delete) per object; on
icechunk always one at a time. Arrays must be allocated before the
transaction: several tasks creating one array at once is a race it cannot fix.

**Deferred presence** (`building.defer_presence`). Each presence stamp is a
read-modify-write of the whole `nonempty_chunks` list, so a build of N cells
costs O(N²) metadata bytes and disjoint-cell writers race. Arrays allocated
after the declaration are born without a manifest, and a write into an array
with no manifest stamps nothing, so no flag is threaded through the writers.
Readers then ask the store: `Group.chunk_exists` reads the cell,
`Group.list_chunks` lists the array (a sharded array shard by shard). A handle
opened before the declaration still allocates manifests; they stay correct
after the rebuild but their stamps race again. zarr-vectors < 0.9.3 reads a
missing manifest as empty; neuroglancer probes every cell when there is none.

**Claimed presence** (`building.set_presence`). The alternative,
`rebuild_presence`, lists and reads every shard object of every array. A cell
written by more than one worker (a canonical link cell) must appear in the
union of claims; `vertex_fragments`, `link_fragments` and `link_attributes/*`
must be named. `verify="objects"` reads a partly claimed shard's index with one
ranged read of its tail. `on_mismatch="derive"` derives only the failing arrays.

**Durability** (`core._durable`). A caller that journals "done" with fsync
needs its cells at least as durable as the journal line; after a power loss a
non-durable write that returned can be missing or empty. Windows skips the
directory fsync (a directory cannot be opened; NTFS journals renames).

**Array layout checks** (`core.arrays._array_matches_layout`). Emptiness is
asked of the store, not of `nonempty_chunks`: a writer with
`record_presence=False` leaves the manifest empty while payloads exist, and
trusting it deleted populated arrays. A populated array is never recreated
(that deletes the node); `sharding.reshard` re-lays it out.

## Object index

**Row reservation** (`core.object_rows`). `commit_object_index` sees unwritten
rows because a stored id table fills with `UNWRITTEN_ID` (-1) and a vlen row of
a partly written bucket reads `b""` (zarr fills the other rows of a chunk it
creates with the type's default, not the array's fill value). Rows of a bucket
nobody wrote read as empty objects either way. `at=` cannot be shared between
writers: the second writer's pad or truncation lands in the first's rows, and
shape metadata is read-modify-written by all.

**Dense append** (`core.dense_manifests._kept_block_end`). An append at
`start < n0` truncates `manifest_blocks` to the highest block end any kept row
`[0, start)` references. Fast path: when every row from `start - 1` on begins
where the previous one ended and the last ends at the block count (the layout
appends leave), the residue is exactly `[E, b0)` with `E` the end of row
`start - 1`, and no kept row's blocks lie in it, because no two rows share
blocks. Otherwise (a patch or placed range moved blocks out of row order, or
reserved blocks sit past the last row) the kept rows are scanned in bounded
pieces. A kept row emptied by a patch after the residue was written is counted
by the scan but not by the fast path; every row reads the same either way.
`write_object_manifests(layout="dense")` declares the dense capability even on
a store created without `manifest_layout="dense"`, so the store says so either way.

## Reading

**Cell keys** (`core.group._coord_to_index`). A rank mismatch is returned
unchanged rather than zipped: zip truncation aliased a rank-4 key onto a rank-3
cell and passed the bounds check. Icechunk lists only prefixes that name a
group or array, so cell listings start at the array prefix.

## Multiscale and metadata

**Leading grid axes** (`core.arrays._leading_grid_axes`). The attribute-bin
grid extent comes from the allocated `vertices` array's extra axes and from
`len(chunk_attribute_values)`: `rechunk` sets `chunk_dims` always but
`chunk_attribute_*` only for `attribute:` specs, so group or object-id rechunks
have a wider grid with no bin values. `chunk_dims` (names only, unvalidated) is
not consulted.

**Shard shape** (`core.metadata.normalise_shard_shape`). Stored verbatim and
not clipped: cells per shard per axis means the same at every level, and
clipping would make coarse levels disagree with the declared shape. A bin axis
gets a leading 1 because bins are queried independently.

**OME node ownership** (`core.ome.is_owned_node`). Ownership is keyed on the
exact `zv:level` type, not the `zv:` prefix, so a refresh owns only what it
builds. The level list is derived from disk; foreign children are preserved,
except one named like a level (RFC 8 requires unique child names). When levels
become `collection` nodes (0.10.0) this predicate needs a marker, which is why
the rule lives in one function.

**Chunk-rank validation** (`validate.structure._check_chunk_ranks`). Both checks
are needed: the consistency check compares fragment columns with `sid_ndim`,
which says nothing about cell addressing, and `derive_nonempty_chunks` rebuilds
presence at the array's own rank, so a mis-ranked array agrees with itself and
only the comparison with `vertices` catches it.
