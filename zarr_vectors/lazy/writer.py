"""ZVWriter — lazy / async mutation handle for a single ``ZVLevel``.

Adds the write-back surface the algorithms package needs:

* :meth:`add_attribute` (Tier A) — write a per-vertex attribute array
  aligned with the level's existing vertices, without rewriting the
  vertex data.
* :meth:`add_node_attribute`, :meth:`add_face_attribute`,
  :meth:`add_object_attribute` — siblings for the analogous result
  types.
* :meth:`append_vertices` — true incremental append (Step 7).
* :meth:`commit` / :meth:`compact` — pending-sidecar lifecycle.

Each public method has both an async and a sync mirror.  The async
methods drive the zarr store's async I/O directly; the sync mirrors run
them to completion with zarr's :func:`~zarr.core.sync.sync`, on this
module's own background event loop (:func:`_writer_loop`), not zarr's.
``asyncio.run`` is deliberately *not* used: it raises when called from
inside a running event loop, which rules out every embedded/browser host.
Same reasoning as :mod:`zarr_vectors.ops.relocate`.

Why not zarr's loop: the methods fan work out with
:func:`asyncio.to_thread`, which uses the running loop's default thread
pool, and every storage call inside that work is itself a zarr ``sync()``
that needs a thread from *zarr's* pool.  On zarr's loop those are one
pool, so once there are as many chunk keys as threads, every thread
waits on a storage call that no free thread can run: a permanent hang.
On a loop of our own the two pools differ, which is the same asymmetry
that keeps the async methods safe when a caller awaits them on theirs.

v1 is **single-writer-only**.  Concurrent writers against the same
level can race on object_index sidecar batch numbering; documented
loudly and not protected at runtime.
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from zarr.core.sync import sync

from zarr_vectors.constants import VERTEX_FRAGMENTS, VERTICES
from zarr_vectors.core.arrays import (
    _chunk_key,
    _maybe_batched_reads,
    list_chunk_keys,
    object_count,
    patch_object_manifests,
    read_all_object_manifests,
    read_chunk_vertices,
    vertices_dtype,
    write_chunk_attributes,
    write_chunk_vertices,
)
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.spatial.boundary import chunk_local_to_global_offsets
from zarr_vectors.spatial.chunking import assign_chunks
from zarr_vectors.typing import ChunkCoords, ObjectManifest

if TYPE_CHECKING:
    from zarr_vectors.lazy.level import ZVLevel


_LOOP: asyncio.AbstractEventLoop | None = None
_LOOP_LOCK = threading.Lock()


def _writer_loop() -> asyncio.AbstractEventLoop:
    """The event loop the ``*_sync`` mirrors run on, started on first use.

    Deliberately not zarr's (see the module docstring): the mirrors'
    ``to_thread`` fan-out must draw on a different thread pool from the
    one zarr's own ``sync()`` calls need, or it can take every thread
    those calls are waiting for.
    """
    global _LOOP
    if _LOOP is None:
        with _LOOP_LOCK:
            if _LOOP is None:
                loop = asyncio.new_event_loop()
                threading.Thread(
                    target=loop.run_forever, name="zv_writer_loop", daemon=True,
                ).start()
                _LOOP = loop
    return _LOOP


def _reset_writer_loop_after_fork() -> None:
    """A forked child inherits the loop object but not its thread."""
    global _LOOP, _LOOP_LOCK
    _LOOP = None
    _LOOP_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_writer_loop_after_fork)


def _run(coro):
    """Run ``coro`` to completion on :func:`_writer_loop`."""
    return sync(coro, loop=_writer_loop())


class ZVWriter:
    """Mutation handle for one :class:`ZVLevel`.

    Acquire one via ``zv[0].writer()``.  Holds a reference to the
    level's :class:`Group` so all mutations go through the same backend
    the reader uses.

    Usage::

        # Async — recommended for cloud stores
        async with zv[0].writer() as w:
            await w.add_attribute("normal", normals)

        # Sync — convenient for scripts
        with zv[0].writer() as w:
            w.add_attribute_sync("normal", normals)
    """

    def __init__(self, level: ZVLevel) -> None:
        import warnings

        warnings.warn(
            "ZVWriter is superseded by Dataset.add_* for creating geometry "
            "and Dataset.editing() for per-element edits. Note that "
            "add_node_attribute_sync / add_face_attribute_sync have NO "
            "replacement on Dataset -- a bulk per-vertex attribute column is "
            "not an edit -- so use "
            "zarr_vectors.building.write_vertex_attribute for those.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._level = level
        self._group = level._group
        self._committed = False
        # Manifests staged by ``append_vertices`` and flushed by
        # ``commit`` as a pending sidecar batch.
        self._pending_manifests: dict[int, ObjectManifest] = {}
        self._pending_sid_ndim: int | None = None

    # ---------------- context manager -----------------------------------

    async def __aenter__(self) -> ZVWriter:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if exc is None:
            await self.commit()

    def __enter__(self) -> ZVWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is None:
            self.commit_sync()

    # ---------------- Tier A: post-hoc attribute writes -----------------

    async def add_attribute(
        self,
        name: str,
        values: npt.NDArray,
        *,
        dtype: str | np.dtype | None = None,
    ) -> None:
        """Write a per-vertex attribute aligned with this level's vertices.

        Splits ``values`` by chunk using the existing vertex-count
        sidecars (Tier E) and writes one
        ``attributes/<name>/<chunk_key>`` per chunk.  No vertex data is
        re-encoded.

        Args:
            name: Attribute name.  Stored under ``attributes/<name>/``.
            values: ``(N,)`` or ``(N, C)`` array of length equal to the
                level's total vertex count.
            dtype: Override the on-disk dtype (default: ``values.dtype``).
        """
        await self._write_per_vertex_attribute(
            subpath="vertex_attributes", name=name, values=values, dtype=dtype,
        )

    async def add_node_attribute(
        self,
        name: str,
        values: npt.NDArray,
        *,
        dtype: str | np.dtype | None = None,
    ) -> None:
        """Per-node attribute on a graph / skeleton level.

        Identical semantics to :meth:`add_attribute` — nodes are the
        graph's vertices.  Provided as an ergonomic alias for code that
        reads more naturally with the graph terminology.
        """
        await self.add_attribute(name, values, dtype=dtype)

    async def add_face_attribute(
        self,
        name: str,
        values: npt.NDArray,
        *,
        dtype: str | np.dtype | None = None,
    ) -> None:
        """Per-face attribute on a mesh level.

        Stored under ``face_attributes/<name>/<chunk_key>``.  Faces are
        aligned 1:1 with the intra-chunk links array — values for a
        chunk's ``F_local`` faces appear in the same order as the
        decoded ``links/<chunk_key>``.

        Note: faces whose vertices span chunks are just ``link_width=3``
        records at non-zero offsets in the merged ``links/<delta>/<offsets>/``
        family; per-face attributes for those records use the parallel
        ``link_attributes/<name>/<delta>/<offsets>/`` array.
        """
        await self._write_per_face_attribute(
            name=name, values=values, dtype=dtype,
        )

    async def add_object_attribute(
        self,
        name: str,
        values: npt.NDArray,
        *,
        dtype: str | np.dtype | None = None,
    ) -> None:
        """Per-object attribute, length equal to ``num_objects``.

        Writes the dense ``(O,) | (O, C)`` array to
        ``object_attributes/<name>/data``.
        """
        from zarr_vectors.core.arrays import (
            create_object_attributes_array,
            write_object_attributes,
        )

        arr = np.asarray(values)
        if dtype is not None:
            arr = arr.astype(np.dtype(dtype), copy=False)
        await asyncio.to_thread(
            create_object_attributes_array, self._group, name,
        )
        await asyncio.to_thread(
            write_object_attributes, self._group, name, arr,
        )

    # ---------------- internal helpers ----------------------------------

    async def _write_per_vertex_attribute(
        self,
        *,
        subpath: str,
        name: str,
        values: npt.NDArray,
        dtype: str | np.dtype | None,
    ) -> None:
        arr = np.asarray(values)
        if dtype is not None:
            arr = arr.astype(np.dtype(dtype), copy=False)
        if arr.ndim < 1:
            raise ArrayError(
                f"attribute values must be at least 1D; got shape {arr.shape}"
            )

        ndim = self._level._root_meta.sid_ndim
        group = self._group

        def _layout():
            """The offset table and every chunk's fragment sizes.

            All the reading happens here, in one thread under one
            prefetch, before the fan-out below.  Reading inside the
            fan-out opened a one-cell prefetch per task, and those raced
            on the level's one prefetch slot.  The sizes are read at the
            dtype the level declares: a float64 cell read as float32
            decodes to twice its row count, and every attribute row after
            it lands against the wrong vertex.
            """
            keys = [_chunk_key(cc) for cc in list_chunk_keys(group, VERTICES)]
            vdtype = vertices_dtype(group)
            with _maybe_batched_reads(
                group, [(VERTICES, keys), (VERTEX_FRAGMENTS, keys)],
            ):
                offsets, chunk_keys, total = chunk_local_to_global_offsets(group)
                sizes = {
                    cc: [len(g) for g in read_chunk_vertices(group, cc, vdtype, ndim)]
                    for cc in chunk_keys
                }
            return offsets, chunk_keys, total, sizes

        offsets, chunk_keys, total, chunk_sizes = await asyncio.to_thread(_layout)
        if arr.shape[0] != total:
            raise ArrayError(
                f"add_attribute({name!r}): values length {arr.shape[0]} "
                f"!= level vertex count {total}"
            )

        # Schedule one per-chunk write in parallel.  Each task slices the
        # values array by the chunk's fragment sizes and emits the
        # attribute bytes.
        async def _write_one(cc: ChunkCoords) -> None:
            start = offsets[cc]
            sizes = chunk_sizes[cc]
            chunk_total = sum(sizes)
            if chunk_total == 0:
                return
            chunk_values = arr[start:start + chunk_total]
            # Split into groups aligned with the chunk's fragments.
            attr_groups: list[npt.NDArray] = []
            cursor = 0
            for s in sizes:
                attr_groups.append(chunk_values[cursor:cursor + s])
                cursor += s
            # If the writer is targeting non-default subpath (e.g.
            # face attributes), patch the array name; otherwise the
            # standard helper writes under "vertex_attributes/".
            # record_presence=False: these tasks run concurrently over
            # disjoint cells, but nonempty_chunks is array-wide, so
            # stamping it rewrites the shared zarr.json once per cell and
            # the tasks collide on it.  The manifest is rebuilt once after
            # the gather.
            if subpath == "vertex_attributes":
                await asyncio.to_thread(
                    write_chunk_attributes,
                    self._group, name, cc, attr_groups, arr.dtype,
                    record_presence=False,
                )
            else:
                await asyncio.to_thread(
                    _write_custom_subpath,
                    self._group, subpath, name, cc, attr_groups, arr.dtype,
                )

        # Allocate the array once, before the per-chunk fan-out below.
        #
        # This used to be a ``require_group(f"{subpath}/{name}")``, which
        # pre-created the per-array *group* the removed per-chunk-sub-array
        # layout wrote into — the parallel writes then raced to create it,
        # hence the pre-create.  Under the single-array layout the target is
        # one grid-shaped vlen array, so there is no group to race on;
        # pre-creating one instead left a Group where the array belongs and
        # every cell write failed against it.
        #
        # It still has to happen here rather than inside the per-chunk
        # write: ``_write_one`` is fanned out concurrently by the gather,
        # so allocating there would race for real.
        await asyncio.to_thread(
            _ensure_chunk_array, self._group, f"{subpath}/{name}",
        )

        await asyncio.gather(*(_write_one(cc) for cc in chunk_keys))

        # Rebuild the presence manifest the fanned-out writes deliberately
        # skipped.  One listing, after every task has landed.
        await asyncio.to_thread(
            self._group.derive_nonempty_chunks, f"{subpath}/{name}",
        )

        # Make sure the level metadata advertises the new array.  The
        # FAMILY, not "<family>/<name>": every other writer in core records
        # the family and every reader gates on it, so the per-name entry
        # this used to add was read by nothing at all.
        await asyncio.to_thread(self._touch_arrays_present, subpath)

    async def _write_per_face_attribute(
        self,
        *,
        name: str,
        values: npt.NDArray,
        dtype: str | np.dtype | None,
    ) -> None:
        # Face attributes are aligned with the intra-chunk links/faces.
        # The number of faces per chunk is the row count of links/<cc>.
        # We read each chunk's links via `read_chunk_links` to discover
        # the size, then slice and write.
        from zarr_vectors.core.arrays import (
            read_chunk_links,
        )

        arr = np.asarray(values)
        if dtype is not None:
            arr = arr.astype(np.dtype(dtype), copy=False)

        chunk_keys = await asyncio.to_thread(list_chunk_keys, self._group)

        # Phase 1: per-chunk face counts, read in one thread.  Fanned out,
        # each read opened its own prefetch, and those raced on the
        # level's one prefetch slot.
        def _counts() -> list[int]:
            counts: list[int] = []
            for cc in chunk_keys:
                try:
                    # delta=0: face counts come from intra-level links only.
                    groups = read_chunk_links(self._group, cc, np.int64, delta=0)
                except Exception:
                    counts.append(0)
                    continue
                counts.append(sum(int(g.shape[0]) for g in groups))
            return counts

        per_chunk_counts = await asyncio.to_thread(_counts)
        total_faces = sum(per_chunk_counts)
        if arr.shape[0] != total_faces:
            raise ArrayError(
                f"add_face_attribute({name!r}): values length "
                f"{arr.shape[0]} != total intra-chunk face count "
                f"{total_faces}"
            )

        # Phase 2: write slices in parallel.
        cursor = 0
        slices: list[tuple[ChunkCoords, npt.NDArray]] = []
        for cc, count in zip(chunk_keys, per_chunk_counts):
            if count == 0:
                continue
            slices.append((cc, arr[cursor:cursor + count]))
            cursor += count

        # Pre-create the parent face_attributes/<name> group to avoid
        # concurrent require_group races during parallel writes.
        await asyncio.to_thread(
            self._group.require_group, f"face_attributes/{name}"
        )

        async def _write_one(cc: ChunkCoords, sub: npt.NDArray) -> None:
            # One face attribute group per chunk; faces live in a single
            # logical group per chunk in mesh stores today.
            await asyncio.to_thread(
                _write_custom_subpath,
                self._group, "face_attributes", name, cc, [sub], arr.dtype,
            )

        await asyncio.gather(*(_write_one(cc, sub) for cc, sub in slices))

        await asyncio.to_thread(self._touch_arrays_present, "face_attributes")

    def _touch_arrays_present(self, entry: str) -> None:
        """Add ``entry`` to the level's ``arrays_present`` if missing."""
        attrs = self._group.attrs.to_dict()
        lv = attrs.get("zarr_vectors_level", {})
        ap = list(lv.get("arrays_present", []))
        if entry not in ap:
            ap.append(entry)
            lv["arrays_present"] = ap
            self._group.attrs.update({"zarr_vectors_level": lv})

    # ---------------- true append ---------------------------------------

    async def append_vertices(
        self,
        positions: npt.NDArray,
        *,
        object_ids: npt.NDArray | None = None,
        dtype: str | np.dtype | None = None,
    ) -> dict:
        """Append new vertices (and new objects) to this level.

        Routes each vertex to its spatial chunk, reads the existing
        chunk data, appends one fragment per new object, and
        rewrites the chunk.  Per-chunk RMW is parallelised over chunks
        via :func:`asyncio.gather`.

        Per-object manifest entries are staged in memory and flushed to
        a pending sidecar by :meth:`commit`.

        Args:
            positions: ``(N, D)`` array of new vertex positions.
            object_ids: ``(N,)`` integer object IDs for each new vertex.
                IDs must be ``>=`` the current ``num_objects`` (no
                conflict with existing objects).  Defaults to a
                contiguous range starting at the current count.
            dtype: Vertex dtype.  Defaults to the level's recorded dtype.

        Returns:
            Summary dict with ``vertices_added``, ``new_objects``,
            ``chunks_touched``.
        """
        positions = np.asarray(positions)
        if positions.ndim != 2:
            raise ArrayError(
                f"positions must be (N, D), got shape {positions.shape}"
            )
        n_new, ndim = positions.shape
        if n_new == 0:
            return {"vertices_added": 0, "new_objects": 0, "chunks_touched": 0}

        root_meta = self._level._root_meta
        if ndim != root_meta.sid_ndim:
            raise ArrayError(
                f"position ndim {ndim} != store sid_ndim {root_meta.sid_ndim}"
            )
        if dtype is None:
            try:
                vmeta = self._group.read_array_meta("vertices")
                dtype = np.dtype(vmeta.get("dtype", "float32"))
            except Exception:
                dtype = np.float32
        dtype = np.dtype(dtype)
        positions = positions.astype(dtype, copy=False)

        # Resolve object_ids; default = append after existing num_objects.
        existing_num = await asyncio.to_thread(self._current_num_objects)
        if object_ids is None:
            object_ids = np.arange(
                existing_num, existing_num + n_new, dtype=np.int64,
            )
        else:
            object_ids = np.asarray(object_ids, dtype=np.int64)
            if object_ids.shape != (n_new,):
                raise ArrayError(
                    f"object_ids shape {object_ids.shape} != (N,) = ({n_new},)"
                )
            if int(object_ids.min()) < existing_num:
                raise ArrayError(
                    f"object_ids overlap existing objects "
                    f"(min={int(object_ids.min())}, existing_num={existing_num})"
                )

        # Spatial assignment (per-vertex → chunk).
        chunk_assignments = await asyncio.to_thread(
            assign_chunks, positions, root_meta.chunk_shape,
        )

        # Read every touched chunk's existing groups first, in one thread
        # under one prefetch (a new chunk reads as empty).  Reading inside
        # the fan-out below opened a one-cell prefetch per task, and those
        # raced on the level's one prefetch slot.
        def _read_existing() -> dict[ChunkCoords, list[npt.NDArray]]:
            keys = [_chunk_key(cc) for cc in chunk_assignments]
            with _maybe_batched_reads(
                self._group, [(VERTICES, keys), (VERTEX_FRAGMENTS, keys)],
            ):
                return {
                    cc: _safe_read_chunk_vertices(self._group, cc, dtype, ndim)
                    for cc in chunk_assignments
                }

        existing = await asyncio.to_thread(_read_existing)

        # RMW per chunk in parallel.  Each chunk gets one new vertex
        # group per **unique** object id present in the chunk; an
        # object whose vertices span multiple chunks gets multiple
        # manifest entries.
        results: dict[ChunkCoords, dict[int, int]] = {}  # cc → {oid: fragment_idx_added}

        async def _rmw_chunk(cc: ChunkCoords, indices) -> None:
            sub_positions = positions[indices]
            sub_oids = object_ids[indices]

            existing_groups = existing[cc]
            existing_count = len(existing_groups)

            # Append one new fragment per unique object in this chunk.
            chunk_assignments_per_oid: dict[int, int] = {}
            new_groups: list[npt.NDArray] = []
            for new_oid in np.unique(sub_oids):
                mask = sub_oids == new_oid
                new_groups.append(sub_positions[mask])
                chunk_assignments_per_oid[int(new_oid)] = (
                    existing_count + len(new_groups) - 1
                )

            all_groups = existing_groups + new_groups
            # record_presence=False: these run concurrently over disjoint
            # cells, but nonempty_chunks is array-wide — stamping it
            # rewrites the shared zarr.json per cell and the tasks collide
            # on it.  Rebuilt once after the gather.
            await asyncio.to_thread(
                write_chunk_vertices, self._group, cc, all_groups, dtype,
                record_presence=False,
            )
            results[cc] = chunk_assignments_per_oid

        await asyncio.gather(*(
            _rmw_chunk(cc, idxs) for cc, idxs in chunk_assignments.items()
        ))

        for _arr in (VERTICES, VERTEX_FRAGMENTS):
            await asyncio.to_thread(self._group.derive_nonempty_chunks, _arr)

        # Build manifest entries per new object id.
        new_oids = set()
        for cc, oid_to_vg in results.items():
            for oid, fragment_idx in oid_to_vg.items():
                self._pending_manifests.setdefault(oid, []).append((cc, fragment_idx))
                new_oids.add(oid)

        # sid_ndim for the index encoding: include the +1 for attribute
        # chunking when the level is so chunked, else just ndim.
        try:
            existing_meta = self._group.read_array_meta("object_index")
            self._pending_sid_ndim = int(existing_meta.get("sid_ndim", ndim))
        except Exception:
            # No pre-existing object_index — fall back to chunk-key arity
            # discovered from results.
            if results:
                self._pending_sid_ndim = len(next(iter(results.keys())))
            else:
                self._pending_sid_ndim = ndim
        return {
            "vertices_added": n_new,
            "new_objects": len(new_oids),
            "chunks_touched": len(results),
        }

    def _current_num_objects(self) -> int:
        """Inspect existing object_index for total count."""
        # The slot count is stamped on the index; decoding every
        # manifest to take len() of the list read the whole level to
        # learn a number already written down.
        try:
            n_existing = object_count(self._group)
        except Exception:
            return 0
        existing_pending = self._pending_manifests
        if existing_pending:
            return max(
                n_existing,
                max(existing_pending.keys()) + 1,
            )
        return n_existing

    # ---------------- lifecycle -----------------------------------------

    async def commit(self) -> dict:
        """Flush pending appends into the main ``object_index/`` array.

        Reads the existing main index (if any), merges the staged
        manifests with last-write-wins on duplicate OIDs, and rewrites
        ``object_index/``.  Transactional backends (icechunk) make
        this cheap via copy-on-write; plain LocalStore rewrites the
        whole index on every commit.
        """
        out: dict[str, int] = {"committed": True}

        if not self._pending_manifests:
            self._committed = True
            return {**out, "objects_committed": 0}

        sid_ndim = self._pending_sid_ndim or self._level._root_meta.sid_ndim
        await asyncio.to_thread(self._merge_and_write_object_index, sid_ndim)

        # Update level vertex_count from the on-disk vertices blobs.
        await asyncio.to_thread(self._bump_level_vertex_count)

        committed = len(self._pending_manifests)
        self._pending_manifests = {}
        self._pending_sid_ndim = None
        self._committed = True
        return {
            **out,
            "objects_committed": committed,
        }

    async def compact(self) -> dict:
        """Compatibility shim: pending-sidecar staging was removed in
        0.6.0.  Calls :meth:`commit` (which now directly rewrites the
        main index) and reports the count for callers that used to
        rely on the explicit compaction step."""
        if self._pending_manifests:
            await self.commit()
        manifests = await asyncio.to_thread(
            read_all_object_manifests, self._group,
        )
        return {"compacted": True, "num_objects": len(manifests)}

    # ---------------- root-metadata mutators ----------------------------

    def _merge_and_write_object_index(self, sid_ndim: int) -> None:
        """Merge ``self._pending_manifests`` into the main ``object_index/``.

        Writes the staged OIDs over whatever those slots held, leaving
        every other row alone. Reading the whole index in order to write
        it back merged cost the object count on every flush, and an
        append-style writer flushes often; last-write-wins on the staged
        ids is the same outcome either way.
        """
        if not self._pending_manifests:
            return
        patch_object_manifests(
            self._group,
            {int(oid): list(m) for oid, m in self._pending_manifests.items()},
            sid_ndim,
        )

    def _bump_level_vertex_count(self) -> None:
        """Recompute the level's vertex_count from on-disk data."""
        offsets, _keys, total = chunk_local_to_global_offsets(self._group)
        attrs = self._group.attrs.to_dict()
        lv = attrs.get("zarr_vectors_level", {})
        lv["vertex_count"] = int(total)
        self._group.attrs.update({"zarr_vectors_level": lv})

    # ---------------- sync mirrors --------------------------------------

    def add_attribute_sync(
        self,
        name: str,
        values: npt.NDArray,
        *,
        dtype: str | np.dtype | None = None,
    ) -> None:
        _run(self.add_attribute(name, values, dtype=dtype))

    def add_node_attribute_sync(
        self,
        name: str,
        values: npt.NDArray,
        *,
        dtype: str | np.dtype | None = None,
    ) -> None:
        _run(self.add_node_attribute(name, values, dtype=dtype))

    def add_face_attribute_sync(
        self,
        name: str,
        values: npt.NDArray,
        *,
        dtype: str | np.dtype | None = None,
    ) -> None:
        _run(self.add_face_attribute(name, values, dtype=dtype))

    def add_object_attribute_sync(
        self,
        name: str,
        values: npt.NDArray,
        *,
        dtype: str | np.dtype | None = None,
    ) -> None:
        _run(self.add_object_attribute(name, values, dtype=dtype))

    def append_vertices_sync(
        self,
        positions: npt.NDArray,
        *,
        object_ids: npt.NDArray | None = None,
        dtype: str | np.dtype | None = None,
    ) -> dict:
        return _run(self.append_vertices(
            positions, object_ids=object_ids, dtype=dtype,
        ))

    def commit_sync(self) -> dict:
        return _run(self.commit())

    def compact_sync(self) -> dict:
        return _run(self.compact())


def _safe_read_chunk_vertices(
    level_group,
    cc: ChunkCoords,
    dtype: np.dtype,
    ndim: int,
) -> list[npt.NDArray]:
    """Read existing fragments; return ``[]`` if the chunk is missing."""
    if not level_group.chunk_exists("vertices", _chunk_key(cc)):
        return []
    try:
        return read_chunk_vertices(level_group, cc, dtype=dtype, ndim=ndim)
    except Exception:
        return []


def _ensure_chunk_array(level_group, full_name: str) -> None:
    """Allocate a per-chunk array node if it is not there yet.

    Call once before fanning per-chunk writes out concurrently: the write
    path no longer creates the container implicitly, and allocating from
    inside the fan-out would race.
    """
    from zarr_vectors.core.arrays import _ensure_array_dir

    if not level_group.array_exists(full_name):
        _ensure_array_dir(level_group, full_name)


def _write_custom_subpath(
    level_group,
    subpath: str,
    name: str,
    chunk_coords: ChunkCoords,
    attr_groups: list[npt.NDArray],
    dtype,
) -> None:
    """Write attribute bytes to ``<subpath>/<name>/<chunk_key>``.

    Mirrors :func:`write_chunk_attributes` but with a configurable
    top-level subpath (e.g. ``"face_attributes"``).  Per-group byte
    offsets are derived at read time from the parallel
    ``vertex_fragments`` table; no ``_offsets`` sibling is
    written.
    """
    from zarr_vectors.encoding.ragged import encode_ragged_floats

    dtype = np.dtype(dtype)
    key = _chunk_key(chunk_coords)
    full_name = f"{subpath}/{name}"
    # The array must already exist — see _ensure_chunk_array, which the
    # caller runs once before fanning these writes out.
    raw_bytes, _ = encode_ragged_floats(attr_groups, dtype)
    level_group.write_bytes(full_name, key, raw_bytes)
