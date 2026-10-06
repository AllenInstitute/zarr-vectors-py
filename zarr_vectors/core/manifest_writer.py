"""Write object manifests as a stream: :func:`object_manifest_writer`.

:func:`~zarr_vectors.building.write_object_manifests` is one call per
block of objects, and every call starts from nothing: it resolves the
index's layout and spatial rank from its metadata, checks the id table,
resizes every array it touches and rewrites the index's metadata. A
writer committing a large layer in bounded blocks -- ~11,600 calls for
1.9e8 objects -- pays that every time, and on a sharded object layer
each call that ends inside a shard makes the next one read the shard
back and write it again.

:class:`ObjectManifestWriter` makes the same writes as those calls, in
order, and stores the same bytes; but after its first write it resolves
nothing again. Rows are held until they fill whole storage objects (the
layer's shards when it is sharded, else its row chunks) and written
then, so a shard is written once, not once per call that reaches it.

The first write is exactly one ``write_object_manifests(mode="append",
at=at)`` call -- the one that creates the index, pads to ``at`` or
truncates the residue past it -- and the writer then holds the index's
state itself: the row and block counts, the id table's ends and its
sorted stamp. Anything it cannot hold that way (an id table chunked
otherwise than the writers chunk it, an offline or batched-write
session) sends every later write through the per-call path too, so the
result never depends on which path ran.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import OBJECT_INDEX
from zarr_vectors.exceptions import ArrayError

#: Rows a flush waits for before writing, rounded down to whole storage
#: objects of the arrays it writes.
FLUSH_ROWS = 1 << 17

#: The same for a dense index's blocks, which are written on their own
#: once this many are held (and always before the rows naming them).
FLUSH_BLOCKS = 1 << 21


def _outer_rows(node: Any) -> int:
    """Rows per storage object of ``node``: its shard, else its chunk."""
    outer = node.shards if getattr(node, "shards", None) is not None else node.chunks
    return int(outer[0])


class _Ids:
    """The id table's state, kept as the per-call writes would find it.

    Mirrors :func:`~zarr_vectors.core.arrays._ids_for_append` and
    :func:`~zarr_vectors.core.arrays._extend_object_id_table` for an
    append at the end: the ids a write gets, the errors it raises, and
    the sorted stamp it leaves.
    """

    def __init__(self, level_group: Any, meta: dict[str, Any], rows: int, stored: bool) -> None:
        from zarr_vectors.core.arrays import OBJECT_IDS_ARRAY, OBJECT_IDS_SORTED_ATTR

        self.path = f"{OBJECT_INDEX}/{OBJECT_IDS_ARRAY}"
        self.node = level_group.zarr_group[self.path] if stored else None
        self.stamp = bool(meta.get(OBJECT_IDS_SORTED_ATTR, False))
        self.persisted_stamp = self.stamp
        self.first: int | None = None
        self.last: int | None = None
        if self.node is not None and rows:
            self.first = int(self.node[0])
            self.last = int(self.node[rows - 1])
        self.buf: list[npt.NDArray[np.int64]] = []

    @property
    def stored(self) -> bool:
        return self.node is not None

    def for_append(self, start: int, n: int, ids: Any) -> npt.NDArray[np.int64] | None:
        new = None if ids is None else np.asarray(ids, dtype=np.int64).reshape(-1)
        if new is not None and new.size != n:
            raise ArrayError(f"{new.size} ids for {n} manifests")
        if not self.stored:
            if new is not None and not np.array_equal(new, np.arange(start, start + n)):
                raise ArrayError(
                    "this object index stores ids positionally (row i is object "
                    "i); ids= must be the rows being written"
                )
            return None
        identity = start == 0 or (
            self.stamp and self.first == 0 and self.last == start - 1
        )
        if new is not None:
            return new
        if not identity:
            raise ArrayError(
                "this object index stores its ids, and they are not simply the "
                "rows; pass ids= for the objects being appended"
            )
        return np.arange(start, start + n, dtype=np.int64)

    def push(self, new: npt.NDArray[np.int64]) -> None:
        self.stamp = self.stamp and bool(
            np.all(np.diff(new) > 0)
            and (self.last is None or not new.size or int(new[0]) > self.last)
        )
        if new.size:
            if self.first is None:
                self.first = int(new[0])
            self.last = int(new[-1])
            self.buf.append(new)

    def take(self, k: int) -> npt.NDArray[np.int64]:
        """The first ``k`` buffered ids, leaving the rest buffered."""
        allids = np.concatenate(self.buf) if self.buf else np.empty(0, np.int64)
        self.buf = [allids[k:]] if allids.size > k else []
        return allids[:k]


class ObjectManifestWriter:
    """Stream object manifests into a level's object index.

    Made by :func:`object_manifest_writer`; use it as a context::

        with object_manifest_writer(level, at=committed) as w:
            for block in blocks:
                w.write(chunk_coords=..., fragment_idx=..., manifest_offsets=...)

    :meth:`write` takes what ``write_object_manifests`` takes -- encoded
    ``manifest_blobs``, or ``chunk_coords`` / ``fragment_idx`` /
    ``manifest_offsets``, and ``ids`` -- and writes the objects after the
    previous write's, starting at ``at``. Leaving the block writes what
    is held and, with ``commit=True``, commits the index's count (see
    :func:`~zarr_vectors.core.arrays.commit_object_index`); leaving it by
    an exception writes nothing more, so the rows already written stay
    past the committed count as a torn flush's do, for the next append
    at that count to replace.
    """

    def __init__(
        self,
        level_group: Any,
        at: int | None = None,
        *,
        layout: str | None = None,
        commit: bool = False,
    ) -> None:
        if at is not None and int(at) < 0:
            raise ArrayError(f"at must be >= 0, got {at}")
        self._lg = level_group
        self._at = None if at is None else int(at)
        self._layout = layout
        self._commit = bool(commit)
        self._first: int | None = None
        self._cursor = 0
        self._closed = False
        self._direct = False
        self._dense = False
        self._pre_meta: dict[str, Any] = {}
        self._present = 0  # written rows holding at least one fragment
        self.flushes = 0

    # -- context ---------------------------------------------------------

    def __enter__(self) -> ObjectManifestWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self._abandon()

    # -- public ----------------------------------------------------------

    @property
    def first_row(self) -> int | None:
        """The row the first write started at (``None`` before it)."""
        return self._first

    @property
    def end(self) -> int:
        """The row the next write starts at."""
        return self._cursor

    def write(
        self,
        manifest_blobs: Any = None,
        *,
        chunk_coords: Any = None,
        fragment_idx: Any = None,
        manifest_offsets: Any = None,
        ids: Any = None,
    ) -> tuple[int, int]:
        """Write the next objects; returns ``(first_row, n)`` as
        ``write_object_manifests`` does."""
        if self._closed:
            raise ArrayError("this manifest writer is closed")
        as_arrays = chunk_coords is not None or fragment_idx is not None
        if as_arrays == (manifest_blobs is not None) or (
            as_arrays and (chunk_coords is None or fragment_idx is None)
        ):
            raise ArrayError(
                "give manifest_blobs, or chunk_coords and fragment_idx "
                "(optionally with manifest_offsets), not both"
            )
        if self._first is None:
            return self._write_first(
                manifest_blobs, chunk_coords, fragment_idx, manifest_offsets, ids,
            )
        if self._direct:
            return self._write_direct(
                manifest_blobs, chunk_coords, fragment_idx, manifest_offsets, ids,
            )
        if self._dense:
            return self._write_dense(
                manifest_blobs, chunk_coords, fragment_idx, manifest_offsets, ids,
            )
        return self._write_vlen(manifest_blobs, chunk_coords, fragment_idx, manifest_offsets, ids)

    def flush(self) -> None:
        """Write every held row now, whole storage objects or not."""
        if not self._closed and not self._direct and self._first is not None:
            self._flush(final=True)

    def close(self) -> tuple[int, int] | None:
        """Write what is held, then commit if asked. Returns
        ``(first_row, rows written)``, or ``None`` if nothing was written.
        """
        if self._closed:
            return None if self._first is None else (self._first, self._cursor - self._first)
        if self._first is None:
            self._closed = True
            return None
        if not self._direct:
            self._flush(final=True)
        self._closed = True
        if self._commit:
            self._commit_index()
        return self._first, self._cursor - self._first

    # -- the first write, and the per-call fallback ----------------------

    def _call(self, blobs: Any, cc: Any, fi: Any, off: Any, ids: Any, at: int | None,
              layout: str | None) -> tuple[int, int]:
        from zarr_vectors.building import write_object_manifests

        return write_object_manifests(
            self._lg, blobs, chunk_coords=cc, fragment_idx=fi, manifest_offsets=off,
            ids=ids, mode="append", at=at, layout=layout,
        )

    def _write_first(self, blobs: Any, cc: Any, fi: Any, off: Any, ids: Any) -> tuple[int, int]:
        if self._commit and self._lg.array_exists(OBJECT_INDEX):
            self._pre_meta = dict(self._lg.read_array_meta(OBJECT_INDEX) or {})
        first, n = self._call(blobs, cc, fi, off, ids, self._at, self._layout)
        self._first, self._cursor = int(first), int(first) + int(n)
        self._count_present(blobs, off, fi, n)
        self._open_state()
        return int(first), int(n)

    def _write_direct(self, blobs: Any, cc: Any, fi: Any, off: Any, ids: Any) -> tuple[int, int]:
        first, n = self._call(blobs, cc, fi, off, ids, self._cursor, None)
        self._cursor = int(first) + int(n)
        self._count_present(blobs, off, fi, n)
        return int(first), int(n)

    def _count_present(self, blobs: Any, off: Any, fi: Any, n: int) -> None:
        if not self._commit:
            return
        from zarr_vectors import _xp
        from zarr_vectors.core.arrays import _EMPTY_MANIFEST_BLOB

        if blobs is not None:
            self._present += sum(1 for b in blobs if bytes(b) != _EMPTY_MANIFEST_BLOB)
        elif off is None:
            self._present += int(n)  # one block each
        else:
            self._present += int(np.count_nonzero(np.diff(_xp.to_host(off, dtype=np.int64))))

    def _open_state(self) -> None:
        """Resolve the index once, after the first write made it exist."""
        from zarr_vectors.core import dense_manifests as dense
        from zarr_vectors.core.arrays import (
            _LAYOUT_DENSE,
            OBJECT_INDEX_LAYOUT_V2,
            OBJECT_INDEX_MANIFEST_BUCKET,
        )

        lg = self._lg
        if lg._offline is not None or lg._pending_array_metas is not None:
            self._direct = True
            return
        meta = dict(lg.read_array_meta(OBJECT_INDEX) or {}) if OBJECT_INDEX in lg else {}
        self._dense = dense.is_dense(lg, meta)
        rows = self._cursor
        if self._dense:
            self._sid = int(meta["sid_ndim"])
            self._spans = lg._require_array_node(dense.SPANS_PATH)
            self._blocks = lg._require_array_node(dense.BLOCKS_PATH)
            if int(self._spans.shape[0]) != rows:
                self._direct = True
                return
            self._rows_at = rows
            self._blocks_at = int(self._blocks.shape[0])
            self._blocks_end = self._blocks_at
            self._span_buf: list[npt.NDArray[np.int64]] = []
            self._block_buf: list[npt.NDArray[np.int64]] = []
            self._align = _outer_rows(self._spans)
            self._block_align = _outer_rows(self._blocks)
        else:
            path = f"{OBJECT_INDEX}/manifests"
            self._manifests = lg.zarr_group[path]
            if int(self._manifests.shape[0]) != rows:
                self._direct = True
                return
            try:
                self._vlen_sid = int(meta.get("sid_ndim"))
            except Exception:  # noqa: BLE001 - what the per-call encoder is given
                self._vlen_sid = None
            self._rows_at = rows
            self._blob_buf: list[np.ndarray] = []
            self._align = _outer_rows(self._manifests)
        table = f"{OBJECT_INDEX}/object_ids"
        stored = meta.get("layout") in (OBJECT_INDEX_LAYOUT_V2, _LAYOUT_DENSE) and (
            lg.array_exists(table)
        )
        if stored:
            node = lg.zarr_group[table]
            if int(node.shape[0]) != rows or int(node.chunks[0]) < OBJECT_INDEX_MANIFEST_BUCKET:
                # What the per-call path rewrites rather than extends.
                self._direct = True
                return
        self._ids = _Ids(lg, meta, rows, stored)

    # -- streamed writes -------------------------------------------------

    def _write_dense(self, blobs: Any, cc: Any, fi: Any, off: Any, ids: Any) -> tuple[int, int]:
        from zarr_vectors import _xp
        from zarr_vectors.core.arrays import _object_array
        from zarr_vectors.encoding.fragments import decode_object_manifests_csr

        sid = self._sid
        if blobs is not None:
            off, cc, fi = decode_object_manifests_csr(list(_object_array(blobs)), sid)
            off = np.asarray(off, dtype=np.int64)
            cc = np.asarray(cc, dtype=np.int64).reshape(-1, sid)
            fi = np.asarray(fi, dtype=np.int64).reshape(-1)
        else:
            cc = _xp.to_host(cc, dtype=np.int64)
            fi = _xp.to_host(fi, dtype=np.int64).reshape(-1)
            cc = cc.reshape(-1, sid)
            off = (
                np.arange(fi.size + 1, dtype=np.int64) if off is None
                else _xp.to_host(off, dtype=np.int64)
            )
            if off.size == 0 or off[0] != 0 or off[-1] != fi.size or np.any(np.diff(off) < 0):
                raise ArrayError(
                    f"manifest_offsets must start at 0, end at {fi.size} and not decrease"
                )
        n = int(off.size - 1)
        start = self._cursor
        new_ids = self._ids.for_append(start, n, ids)
        counts = np.diff(off)
        if cc.shape[0] != fi.size or int(off[-1]) != fi.size:
            raise ArrayError(
                f"{fi.size} fragments, {cc.shape[0]} coordinate rows and "
                f"offsets ending at {int(off[-1])} disagree"
            )
        if fi.size and int(fi.min()) < 0:
            raise ArrayError(f"fragment_index must be >= 0, got {int(fi.min())}")
        if n:
            spans = np.empty((n, 2), dtype=np.int64)
            spans[:, 0] = self._blocks_end + off[:-1]
            spans[:, 1] = counts
            self._span_buf.append(spans)
        if fi.size:
            self._block_buf.append(np.concatenate([cc, fi[:, None]], axis=1))
        self._blocks_end += int(fi.size)
        self._cursor += n
        if self._commit:
            self._present += int(np.count_nonzero(counts))
        if new_ids is not None:
            self._ids.push(new_ids)
        self._flush(final=False)
        if self._blocks_end - self._blocks_at >= FLUSH_BLOCKS:
            stop = (self._blocks_end // self._block_align) * self._block_align
            self._put_blocks(stop - self._blocks_at)
        return start, n

    def _write_vlen(self, blobs: Any, cc: Any, fi: Any, off: Any, ids: Any) -> tuple[int, int]:
        from zarr_vectors.core.arrays import _EMPTY_MANIFEST_BLOB, _object_array
        from zarr_vectors.encoding.fragments import encode_object_manifests_csr

        if blobs is None:
            blobs = encode_object_manifests_csr(cc, fi, off, sid_ndim=self._vlen_sid)
        blobs = _object_array(blobs)
        n = len(blobs)
        start = self._cursor
        new_ids = self._ids.for_append(start, n, ids)
        if n == 0:
            return start, 0
        self._blob_buf.append(blobs)
        self._cursor += n
        if self._commit:
            self._present += sum(1 for b in blobs if bytes(b) != _EMPTY_MANIFEST_BLOB)
        if new_ids is not None:
            self._ids.push(new_ids)
        self._flush(final=False)
        return start, n

    # -- flushing --------------------------------------------------------

    def _flush(self, final: bool) -> None:
        held = self._cursor - self._rows_at
        if final:
            stop = self._cursor
        else:
            if held < FLUSH_ROWS:
                return
            stop = (self._cursor // self._align) * self._align
        k = stop - self._rows_at
        if k <= 0:
            self._write_stamp()
            return
        if self._dense:
            self._flush_dense(k)
        else:
            self._flush_vlen(k)
        if self._ids.stored and k > 0:
            self._lg.extend_array(self._ids.path, self._ids.take(k), _node=self._ids.node)
        self._rows_at += max(k, 0)
        self._write_stamp()
        self.flushes += 1

    def _flush_dense(self, k: int) -> None:
        from zarr_vectors.core import dense_manifests as dense

        spans = np.concatenate(self._span_buf) if self._span_buf else np.empty((0, 2), np.int64)
        head, rest = spans[:k], spans[k:]
        self._span_buf = [rest] if rest.size else []
        # The blocks the flushed rows name, all of them first: a row on
        # disk never names a block that is not.
        need = int(head[-1, 0] + head[-1, 1]) if head.size else self._blocks_at
        if k == len(spans):
            need = self._blocks_end
        self._put_blocks(need - self._blocks_at)
        if head.size:
            self._lg.extend_array(dense.SPANS_PATH, head, _node=self._spans)

    def _put_blocks(self, take: int) -> None:
        """Write the first ``take`` held blocks."""
        from zarr_vectors.core import dense_manifests as dense

        if take <= 0:
            return
        blocks = np.concatenate(self._block_buf)
        self._block_buf = [blocks[take:]] if blocks.shape[0] > take else []
        self._lg.extend_array(dense.BLOCKS_PATH, blocks[:take], _node=self._blocks)
        self._blocks_at += take

    def _flush_vlen(self, k: int) -> None:
        from zarr.errors import UnstableSpecificationWarning

        blobs = np.concatenate(self._blob_buf)
        head = blobs[:k]
        self._blob_buf = [blobs[k:]] if blobs.size > k else []
        n0 = self._rows_at
        node = self._manifests
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnstableSpecificationWarning)
            node.resize((n0 + k,))
            node[n0:n0 + k] = head
        self._lg._invalidate_node(f"{OBJECT_INDEX}/manifests")

    def _write_stamp(self) -> None:
        from zarr_vectors.core.arrays import OBJECT_IDS_SORTED_ATTR

        if self._ids.stamp != self._ids.persisted_stamp:
            self._lg.write_array_meta(OBJECT_INDEX, {OBJECT_IDS_SORTED_ATTR: self._ids.stamp})
            self._ids.persisted_stamp = self._ids.stamp

    def _abandon(self) -> None:
        """Leave after an exception: nothing more is written."""
        self._closed = True
        if not self._direct and self._first is not None:
            if self._dense:
                self._span_buf, self._block_buf = [], []
            else:
                self._blob_buf = []
            self._ids.buf = []

    # -- commit ----------------------------------------------------------

    def _commit_index(self) -> None:
        from zarr_vectors.core.arrays import OBJECT_IDS_SORTED_ATTR, commit_object_index

        pre = self._pre_meta
        num_present = None
        if (
            self._first is not None
            and int(pre.get("num_objects", -1)) == self._first
            and pre.get("num_present") is not None
        ) or (self._first == 0 and "num_objects" not in pre):
            num_present = int(pre.get("num_present") or 0) + self._present
        ids_sorted = None
        if not self._direct and self._ids.stored and pre.get(OBJECT_IDS_SORTED_ATTR):
            # The stamp held over every row: exact, given it held before.
            ids_sorted = self._ids.stamp
        commit_object_index(
            self._lg, self._cursor, num_present=num_present, object_ids_sorted=ids_sorted,
        )


def object_manifest_writer(
    level_group: Any,
    at: int | None = None,
    *,
    layout: str | None = None,
    commit: bool = False,
) -> ObjectManifestWriter:
    """A context that writes object manifests as a stream.

    The streamed form of ``write_object_manifests(mode="append")``: the
    writes made through it store exactly the bytes the same sequence of
    calls would (each at the row the previous one ended at), but the
    index's layout, spatial rank and id-table state are resolved once,
    and rows are written in whole storage objects -- the object layer's
    shards when it is sharded, so no shard is rewritten per call.

    Args:
        level_group: The resolution level.
        at: The row the first write starts at, with ``at=``'s meaning
            for an append: a short index is padded to it, residue past it
            is truncated (by the first write -- an empty one included).
            ``None`` appends at the current end.
        layout: ``"vlen"`` or ``"dense"``, for an index the first write
            creates.
        commit: On a clean exit, commit the index's count (``num_objects``
            becomes the end of the last write) with
            :func:`~zarr_vectors.core.arrays.commit_object_index`,
            counting ``num_present`` from what was written when the index
            was committed at ``at`` before.

    Returns:
        An :class:`ObjectManifestWriter`; its ``write`` takes what
        ``write_object_manifests`` takes, less ``mode``/``at``/``layout``.
    """
    return ObjectManifestWriter(level_group, at, layout=layout, commit=commit)


__all__ = ["FLUSH_BLOCKS", "FLUSH_ROWS", "ObjectManifestWriter", "object_manifest_writer"]
