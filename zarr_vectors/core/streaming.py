"""Streaming writer for the object index at very large scale.

The full-rebuild :func:`zarr_vectors.core.arrays.write_object_index` takes a
dense ``{object_id: manifest}`` mapping and re-encodes every object — including
every fragment — on each call.  That is fine for moderate object counts but
infeasible when a pipeline emits **billions** of path objects: it can neither
hold the mapping in RAM nor afford to re-encode the fragment blobs Core 2
already wrote.

:class:`ObjectIndexAppender` streams new path objects onto an existing level
group in batches, holding only O(fragments) in memory:

* The ``object_index/manifests`` vlen-bytes array (the large one) is resized
  down to ``base_oid`` on open — keeping the fragment blobs ``[0, base_oid)``
  byte-intact, no re-encode — and new path manifests are appended in
  ``OBJECT_INDEX_MANIFEST_BUCKET``-aligned chunks.
* The per-object ``length`` attribute (small relative to manifests) is buffered
  and materialised once at :meth:`~ObjectIndexAppender.close`.
* Group 0 (fragments) is written explicitly; group 1 (the contiguous path
  range ``[base_oid, total_objects)``) is written via the implicit range
  encoding of :func:`zarr_vectors.core.arrays.write_groupings`, so it stays
  O(1) rather than an 8-byte-per-path int64 list.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from zarr.codecs import VLenBytesCodec
from zarr.errors import UnstableSpecificationWarning

from zarr_vectors.constants import OBJECT_INDEX
from zarr_vectors.core.arrays import (
    OBJECT_INDEX_LAYOUT_V1,
    OBJECT_INDEX_MANIFEST_BUCKET,
    read_object_attributes,
    write_groupings,
    write_object_attributes,
)
from zarr_vectors.core.store import FsGroup
from zarr_vectors.encoding.fragments import encode_object_manifest_blocks
from zarr_vectors.exceptions import StoreError

# One ObjectManifest is the same shape write_object_index accepts per object:
# an ordered list of (chunk_coords, fragment_index) references.
ObjectManifest = list[tuple[tuple[int, ...], int]]


class ObjectIndexAppender:
    """Stream path objects onto an existing level group in batches.

    See the module docstring for the why.  Lifecycle::

        app = ObjectIndexAppender(level_group, base_oid, sid_ndim, frag_oids)
        for manifests, lengths in batches:
            app.append(manifests, lengths)
        total_objects = app.close()

    or as a context manager (``close`` runs on exit)::

        with ObjectIndexAppender(level_group, base_oid, sid_ndim, frag_oids) as app:
            app.append(manifests, lengths)

    Args:
        level_group: Resolution-level group to append into.
        base_oid: Truncate the object index + length array to this many
            objects on open, keeping object blobs ``[0, base_oid)`` (the
            fragments) intact.  New path objects are appended at OIDs
            ``>= base_oid``.
        sid_ndim: Number of spatial index dimensions (manifest encoding +
            object_index meta).
        fragment_group_oids: Object IDs for group 0 (the fragments).
            Written at :meth:`close`.
        length_attr: Name of the per-object length attribute.
    """

    def __init__(
        self,
        level_group: FsGroup,
        base_oid: int,
        sid_ndim: int,
        fragment_group_oids: Sequence[int],
        length_attr: str = "length",
    ) -> None:
        self._level_group = level_group
        self._base_oid = int(base_oid)
        self._sid_ndim = int(sid_ndim)
        self._fragment_group_oids = list(fragment_group_oids)
        self._length_attr = length_attr

        self._blob_buf: list[bytes] = []
        self._length_buf: list[int] = []
        self._total_appended = 0
        self._closed = False

        self._arr = self._open_manifests_truncated()
        # Array length == next free OID; truncation leaves it at base_oid.
        self._len = self._base_oid

    def _open_manifests_truncated(self):
        """Open ``object_index/manifests`` resized to ``base_oid``.

        Keeps fragment blobs ``[0, base_oid)`` byte-intact (no re-encode);
        drops any prior-run path blobs ``[base_oid, L)``.  Creates the
        array if absent, filling ``[0, base_oid)`` with empty-manifest
        blobs so every slot is materialised (normal fresh runs pass
        ``base_oid == 0``).
        """
        oi_group = self._level_group.zarr_group.require_group(OBJECT_INDEX)
        if "manifests" in oi_group:
            arr = oi_group["manifests"]
            arr.resize((self._base_oid,))
            return arr

        # vlen-bytes lacks a finalised V3 spec — silence as
        # _write_object_index_manifests does.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnstableSpecificationWarning)
            arr = oi_group.create_array(
                "manifests",
                shape=(self._base_oid,),
                chunks=(OBJECT_INDEX_MANIFEST_BUCKET,),
                dtype="bytes",
                serializer=VLenBytesCodec(),
            )
            if self._base_oid > 0:
                empty = encode_object_manifest_blocks([], sid_ndim=self._sid_ndim)
                obj = np.empty(self._base_oid, dtype=object)
                obj[:] = [empty] * self._base_oid
                arr[:] = obj
        return arr

    def append(
        self,
        manifests: list[ObjectManifest],
        lengths: Sequence[int],
    ) -> None:
        """Append a batch of path objects.

        Each ``manifests[i]`` is one object's ordered list of
        ``(chunk_coords, fragment_index)`` references — the shape
        ``write_object_index`` accepts per object — encoded internally as
        mode-0 single-fragment blocks.  ``lengths[i]`` is that object's
        int length.  ``len(manifests) == len(lengths)``.
        """
        if self._closed:
            raise ValueError("append() called after close()")
        if len(manifests) != len(lengths):
            raise ValueError(
                f"len(manifests)={len(manifests)} != len(lengths)={len(lengths)}"
            )

        for manifest in manifests:
            blocks = [
                (tuple(int(c) for c in chunk_coords), int(fragment_index))
                for chunk_coords, fragment_index in manifest
            ]
            self._blob_buf.append(
                encode_object_manifest_blocks(blocks, sid_ndim=self._sid_ndim)
            )
        self._length_buf.extend(int(length) for length in lengths)
        self._total_appended += len(manifests)
        self._flush(final=False)

    def _flush(self, final: bool) -> None:
        """Write buffered manifest blobs, keeping writes chunk-aligned.

        Non-final flushes only write up to a ``OBJECT_INDEX_MANIFEST_BUCKET``
        boundary, so per-append partial read-modify-write is avoided.  A
        single partial-chunk write at the ``base_oid`` boundary (and one at
        the very end on ``final``) is acceptable.
        """
        start = self._len
        n = len(self._blob_buf)
        if n == 0:
            return
        if final:
            k = n
        else:
            bucket = OBJECT_INDEX_MANIFEST_BUCKET
            k = ((start + n) // bucket) * bucket - start
            if k <= 0:
                return  # buffer hasn't reached the next bucket boundary yet

        self._arr.resize((start + k,))
        obj = np.empty(k, dtype=object)
        obj[:] = self._blob_buf[:k]
        self._arr[start:start + k] = obj
        self._len = start + k
        del self._blob_buf[:k]

    def close(self) -> int:
        """Flush everything and write object_index meta, length, groupings.

        Returns:
            ``total_objects = base_oid + (total appended)``.
        """
        if self._closed:
            return self._len
        self._flush(final=True)
        total = self._len

        self._level_group.write_array_meta(OBJECT_INDEX, {
            "zv_array": "object_index",
            "num_objects": total,
            "sid_ndim": self._sid_ndim,
            "layout": OBJECT_INDEX_LAYOUT_V1,
        })

        self._write_length(total)

        write_groupings(self._level_group, {
            0: self._fragment_group_oids,
            1: range(self._base_oid, total),
        })

        self._closed = True
        return total

    def _write_length(self, total: int) -> None:
        """Write the full ``[0, total)`` int32 length array once.

        Keeps the existing fragment region ``[0, base_oid)`` (zeros if the
        attribute is absent) and concatenates the buffered path lengths.
        """
        try:
            existing = read_object_attributes(self._level_group, self._length_attr)
        except StoreError:
            existing = None

        frag = np.zeros(self._base_oid, dtype=np.int32)
        if existing is not None and existing.size:
            take = min(self._base_oid, existing.shape[0])
            frag[:take] = existing[:take].astype(np.int32, copy=False)

        full = np.concatenate([
            frag,
            np.asarray(self._length_buf, dtype=np.int32),
        ])
        write_object_attributes(
            self._level_group, self._length_attr, full, mode="replace",
        )

    def __enter__(self) -> "ObjectIndexAppender":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
