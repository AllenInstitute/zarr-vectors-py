"""Reserving the object layer's rows, and writing ranges of them in place.

The object layer's ``at=`` writes are a resume cursor: a write below the
current length truncates everything after it (a torn flush's residue),
one above it pads the gap, and every call resizes the arrays and rewrites
their shape metadata. That is right for one writer resuming after a crash
and wrong for several writers filling disjoint row ranges at once: the
second writer's pad or truncation lands in the first writer's rows, and
the shape metadata is read-modify-written by all of them.

So a parallel writer works in two steps:

1. The coordinator calls :func:`reserve_object_rows` once: every
   object-layer array it names is created or grown to the final row count
   (and, for a dense index, the final block count), so nothing is resized
   later.
2. Each worker writes its rows with ``mode="place"``
   (:func:`place_manifests`, :func:`place_attribute_columns`): the rows
   ``[at, at + n)`` only -- no resize, no padding, no metadata. Ranges of
   different workers must not share a storage object:
   :func:`concurrency_contract` gives the alignment.

Then :func:`~zarr_vectors.core.arrays.commit_object_index` commits the
count, once. Every reserved row below it must have been written -- an
object with no fragments is written as an empty manifest -- and the
commit refuses rows nobody wrote where it can see them: a stored id table
fills with ``-1``, and a vlen row of a bucket that was partly written
reads ``b""`` (zarr fills the other rows of a chunk it creates with the
type's default, not the array's fill value). Rows of a bucket nobody wrote
read as empty objects either way: a vlen index is created with the empty
manifest as its fill, a dense one's spans with ``(0, 0)``.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import OBJECT_ATTRIBUTES, OBJECT_INDEX
from zarr_vectors.exceptions import ArrayError

if TYPE_CHECKING:  # pragma: no cover
    from zarr_vectors.core.group import Group

#: The id-table fill: a row nobody wrote. Object ids are never negative.
UNWRITTEN_ID = -1


# --------------------------------------------------------------------
# The contract


def row_alignment(level_group: Group | None = None) -> tuple[int, int]:
    """``(rows, blocks)``: the alignment disjoint writers' ranges need.

    A range must start at a multiple and end at one (or at the reserved
    end), so no two writers touch one storage object: the object layer's
    shard when it is sharded, else the largest row bucket (65,536
    attribute rows; a dense index's blocks chunk at 16,384).
    """
    from zarr_vectors.core.arrays import (
        OBJECT_ATTRIBUTE_ROW_BUCKET,
        OBJECT_INDEX_MANIFEST_BUCKET,
        object_shard_rows,
    )

    rows = object_shard_rows(level_group) if level_group is not None else None
    if rows:
        return int(rows), int(rows)
    return OBJECT_ATTRIBUTE_ROW_BUCKET, OBJECT_INDEX_MANIFEST_BUCKET


def concurrency_contract(level_group: Group | None = None) -> dict[str, Any]:
    """What several processes may write into one store at once.

    - ``shard_is_unit``: per-chunk arrays (vertices, fragments, links and
      their attributes) may be written by many processes if each storage
      object -- a cell, or the shard holding it -- is written by one
      process at a time. Presence stamps are shared state: defer them
      (``defer_presence``) and record them once afterwards.
    - ``concurrent_at_ranges``: the object layer may be filled by many
      processes after one :func:`reserve_object_rows`, each writing its
      own ``[at, at + n)`` with ``mode="place"``, when ``at`` and
      ``at + n`` are multiples of ``at_alignment`` (the last range may end
      at the reserved end). Then one ``commit_object_index``.
    - ``at_alignment`` / ``block_alignment``: those multiples, for rows
      and (dense index) for ``block_at``; for ``level_group`` if given,
      else for an unsharded object layer.

    Not covered: the ``at=`` append mode (a resume cursor, one writer at
    a time), metadata writes (one coordinator), and two processes writing
    one link cell.
    """
    rows, blocks = row_alignment(level_group)
    return {
        "shard_is_unit": True,
        "concurrent_at_ranges": True,
        "at_alignment": rows,
        "block_alignment": blocks,
        "requires": ["reserve_object_rows", "mode='place'", "commit_object_index"],
    }


# --------------------------------------------------------------------
# Reserving


def _create_empty(
    level_group: Group,
    path: str,
    shape: tuple[int, ...],
    dtype: Any,
    chunks: tuple[int, ...],
    *,
    shards: tuple[int, ...] | None,
    fill_value: Any,
    attributes: dict[str, Any] | None = None,
    serializer: Any = None,
) -> None:
    """Create an array of ``shape`` without writing a byte of it.

    Every row reads as ``fill_value`` until written, which is what makes
    reserving a billion rows cost one ``zarr.json``.
    """
    from zarr.errors import UnstableSpecificationWarning

    parent_path, _, leaf = path.rpartition("/")
    parent = (
        level_group.zarr_group.require_group(parent_path) if parent_path
        else level_group.zarr_group
    )
    if leaf in parent:
        del parent[leaf]
    kwargs: dict[str, Any] = {
        "shape": shape, "chunks": chunks, "dtype": dtype, "fill_value": fill_value,
    }
    if shards is not None:
        kwargs["shards"] = shards
    if serializer is not None:
        kwargs["serializer"] = serializer
    resolved = level_group._resolve_codecs(None)
    if resolved is not None:
        from zarr_vectors.encoding.compression import codecs_for_create_array

        kwargs["compressors"] = codecs_for_create_array(resolved)
    if attributes:
        kwargs["attributes"] = attributes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnstableSpecificationWarning)
        parent.create_array(leaf, **kwargs)
    level_group._invalidate_node(path)


def _grow(level_group: Group, path: str, n: int) -> bool:
    """Grow ``path`` to ``n`` rows (never shrink); whether it grew."""
    node = level_group.zarr_group[path]
    n0 = int(node.shape[0])
    if n0 >= n:
        return False
    if "shape" in node.attrs:
        node.metadata.attributes["shape"] = [n, *node.shape[1:]]
    node.resize((n, *node.shape[1:]))
    level_group._invalidate_node(path)
    return True


def reserve_object_rows(
    level_group: Group,
    n_objects: int,
    *,
    sid_ndim: int | None = None,
    layout: str | None = None,
    n_blocks: int | None = None,
    columns: Mapping[str, Any] | None = None,
    shard_rows: int | None = None,
) -> dict[str, Any]:
    """Size the object layer for ``n_objects`` rows, once, before workers write.

    Creates, or grows (never shrinks), every array a parallel fill will
    write, so that no worker resizes anything:

    - the object index, in its existing layout or, new, in ``layout``
      (default: the store's ``manifest_layout``): a vlen index's
      ``manifests``, or a dense one's ``manifest_spans`` and -- when
      ``n_blocks`` is given -- ``manifest_blocks``; plus the id table,
      where the layout stores one;
    - each of ``columns``, ``{name: dtype}``, ``{name: (dtype,
      row_shape)}`` or ``{name: (dtype, row_shape, fill_value)}``, as
      :func:`~zarr_vectors.core.arrays.write_object_attributes` would
      create it.

    New arrays are created empty: nothing is written until a worker
    writes its rows. ``num_objects`` is not touched:
    :func:`~zarr_vectors.core.arrays.commit_object_index` commits it, and
    refuses reserved rows nobody wrote where it can tell (see the module
    docstring).

    Args:
        n_objects: Rows the layer will hold.
        sid_ndim: Spatial-index rank, for a new index (default: the level's).
        layout: ``"vlen"`` or ``"dense"``, for a new index.
        n_blocks: Dense only: rows of ``manifest_blocks`` to reserve --
            at least the fragments every worker will write, each worker's
            region aligned (see :func:`concurrency_contract`).
        columns: Object attribute columns to reserve.
        shard_rows: Create new arrays sharded at this many rows (a
            multiple of 65,536). ``None`` follows the layer's existing
            layout. Refused if the layer already has another.

    Returns:
        ``{"rows", "layout", "arrays", "at_alignment", "block_alignment"}``.
    """
    import zarr
    from zarr.codecs import VLenBytesCodec

    from zarr_vectors.core import dense_manifests as dense
    from zarr_vectors.core.arrays import (
        _EMPTY_MANIFEST_BLOB,
        _LAYOUT_DENSE,
        OBJECT_ATTRIBUTE_ROW_BUCKET,
        OBJECT_IDS_ARRAY,
        OBJECT_IDS_SORTED_ATTR,
        OBJECT_INDEX_LAYOUT_V1,
        OBJECT_INDEX_LAYOUT_V2,
        OBJECT_INDEX_MANIFEST_BUCKET,
        _index_layout_for_write,
        _index_sid_ndim,
        check_object_shard_rows,
        object_layer_arrays,
        object_shard_rows,
    )

    n = int(n_objects)
    if n < 0:
        raise ArrayError(f"n_objects {n} is negative")
    existing_rows = object_shard_rows(level_group)
    if shard_rows is not None:
        rows = check_object_shard_rows(shard_rows)
        if object_layer_arrays(level_group) and existing_rows != rows:
            raise ArrayError(
                f"the object layer is laid out with shard_rows={existing_rows}; "
                f"repack it with shard_object_layer before reserving at {rows}"
            )
    else:
        rows = existing_rows
    kind = _index_layout_for_write(level_group, layout)
    sid = int(sid_ndim) if sid_ndim is not None else _index_sid_ndim(level_group)
    meta = (
        level_group.read_array_meta(OBJECT_INDEX)
        if level_group.array_exists(OBJECT_INDEX) else {}
    )
    touched: list[str] = []

    def _shards(chunks: tuple[int, ...]) -> tuple[int, ...] | None:
        return None if rows is None else (rows, *chunks[1:])

    def _id_table() -> None:
        path = f"{OBJECT_INDEX}/{OBJECT_IDS_ARRAY}"
        chunks = (OBJECT_INDEX_MANIFEST_BUCKET,)
        if level_group.array_exists(path):
            if _grow(level_group, path, n):
                touched.append(path)
        else:
            _create_empty(
                level_group, path, (n,), np.int64, chunks,
                shards=_shards(chunks), fill_value=UNWRITTEN_ID,
            )
            touched.append(path)

    if kind == "dense":
        fresh = not level_group.array_exists(dense.SPANS_PATH)
        for path, width, count in (
            (dense.SPANS_PATH, 2, n),
            (dense.BLOCKS_PATH, sid + 1, n_blocks),
        ):
            if level_group.array_exists(path):
                if count is not None and _grow(level_group, path, int(count)):
                    touched.append(path)
            else:
                chunks = (dense._BUCKET, width)
                _create_empty(
                    level_group, path, (int(count or 0), width), np.int64, chunks,
                    shards=_shards(chunks), fill_value=0,
                )
                touched.append(path)
        _id_table()
        level_group.write_array_meta(OBJECT_INDEX, {
            "zv_array": "object_index", "sid_ndim": sid, "layout": _LAYOUT_DENSE,
            OBJECT_IDS_SORTED_ATTR: False,
        })
        if fresh:
            dense._declare_capability(level_group)
    else:
        path = f"{OBJECT_INDEX}/manifests"
        chunks = (OBJECT_INDEX_MANIFEST_BUCKET,)
        if level_group.array_exists(path):
            node = level_group.zarr_group[path]
            n0 = int(node.shape[0])
            if _grow(level_group, path, n):
                touched.append(path)
                if bytes(node.fill_value or b"") != _EMPTY_MANIFEST_BLOB:
                    # An index created before reservation filled with b"",
                    # which not every manifest reader accepts: give the new
                    # rows the empty manifest now, bucket by bucket.
                    node = level_group.zarr_group[path]
                    for lo in range(n0, n, OBJECT_INDEX_MANIFEST_BUCKET):
                        hi = min(lo + OBJECT_INDEX_MANIFEST_BUCKET, n)
                        block = np.empty(hi - lo, dtype=object)
                        block[:] = [_EMPTY_MANIFEST_BLOB] * (hi - lo)
                        node[lo:hi] = block
        else:
            _create_empty(
                level_group, path, (n,), "bytes", chunks, shards=_shards(chunks),
                fill_value=_EMPTY_MANIFEST_BLOB, serializer=VLenBytesCodec(),
            )
            touched.append(path)
        vlen_layout = meta.get("layout") or OBJECT_INDEX_LAYOUT_V1
        stamp = {"zv_array": "object_index", "sid_ndim": sid, "layout": vlen_layout}
        if vlen_layout == OBJECT_INDEX_LAYOUT_V2:
            _id_table()
            stamp[OBJECT_IDS_SORTED_ATTR] = False
        level_group.write_array_meta(OBJECT_INDEX, stamp)

    for name, spec in (columns or {}).items():
        dtype, row_shape, fill = _column_spec(spec)
        path = f"{OBJECT_ATTRIBUTES}/{name}"
        node = level_group._lookup_node(path)
        if isinstance(node, zarr.Array):
            if np.dtype(node.dtype) != dtype or tuple(node.shape[1:]) != row_shape:
                raise ArrayError(
                    f"{path} is {node.dtype} rows of {tuple(node.shape[1:])}; "
                    f"reserved as {dtype} rows of {row_shape}"
                )
            if _grow(level_group, path, n):
                touched.append(path)
            continue
        if fill is None:
            from zarr_vectors.core.arrays import _default_fill_value_for_dtype

            fill = _default_fill_value_for_dtype(dtype)
        chunks = (OBJECT_ATTRIBUTE_ROW_BUCKET, *row_shape)
        _create_empty(
            level_group, path, (n, *row_shape), dtype, chunks,
            shards=_shards(chunks), fill_value=fill,
            attributes={
                "zv_array": "object_attribute",
                "name": name,
                "dtype": str(dtype),
                "shape": [n, *row_shape],
                "fill_sentinel_meaning": "absent",
            },
        )
        touched.append(path)

    align_rows, align_blocks = row_alignment(level_group)
    return {
        "rows": n,
        "layout": kind,
        "arrays": touched,
        "at_alignment": align_rows,
        "block_alignment": align_blocks,
    }


def _column_spec(spec: Any) -> tuple[np.dtype, tuple[int, ...], Any]:
    if isinstance(spec, tuple):
        dtype = np.dtype(spec[0])
        tail = spec[1] if len(spec) > 1 and spec[1] is not None else ()
        row_shape = tuple(int(d) for d in tail)
        fill = spec[2] if len(spec) > 2 else None
        return dtype, row_shape, fill
    return np.dtype(spec), (), None


# --------------------------------------------------------------------
# Writing in place


def _reserved(level_group: Group, path: str, end: int, what: str) -> Any:
    import zarr

    node = level_group._lookup_node(path)
    if not isinstance(node, zarr.Array):
        raise ArrayError(f"{what}: {path} is not reserved; call reserve_object_rows first")
    if int(node.shape[0]) < end:
        raise ArrayError(
            f"{what}: rows up to {end} requested, {path} reserves "
            f"{int(node.shape[0])}; reserve_object_rows for the full count first"
        )
    return node


def place_manifests(
    level_group: Group,
    at: int,
    *,
    manifest_blobs: Any = None,
    offsets: npt.NDArray[np.int64] | None = None,
    coords: npt.NDArray[np.int64] | None = None,
    frags: npt.NDArray[np.int64] | None = None,
    ids: Any = None,
    block_at: int | None = None,
) -> tuple[int, int]:
    """Write manifests into reserved rows ``[at, at + n)``, and nothing else.

    No resize, no padding, no metadata write -- so disjoint, aligned
    ranges may be written by several processes at once. Give encoded
    ``manifest_blobs`` (vlen) or CSR ``offsets`` / ``coords`` / ``frags``
    (either layout). A dense index also needs ``block_at``: the first row
    of ``manifest_blocks`` this call's fragments occupy, inside the
    reservation and aligned like ``at``. Where the layout stores ids,
    ``ids`` names the rows' objects (default: the rows themselves).

    Returns:
        ``(at, n)``.
    """
    from zarr_vectors.core import dense_manifests as dense
    from zarr_vectors.core.arrays import (
        _LAYOUT_DENSE,
        OBJECT_IDS_ARRAY,
        OBJECT_INDEX_LAYOUT_V2,
        _index_sid_ndim,
    )
    from zarr_vectors.encoding.fragments import encode_object_manifests_csr

    at = int(at)
    if at < 0:
        raise ArrayError(f"at must be >= 0, got {at}")
    meta = level_group.read_array_meta(OBJECT_INDEX) or {}
    layout = meta.get("layout")
    sid = _index_sid_ndim(level_group)
    if layout == _LAYOUT_DENSE:
        if offsets is None or coords is None or frags is None:
            raise ArrayError("a dense index is placed from offsets, coords and frags")
        offsets = np.asarray(offsets, dtype=np.int64)
        frags = np.asarray(frags, dtype=np.int64).reshape(-1)
        coords = np.asarray(coords, dtype=np.int64).reshape(-1, sid)
        n, k = int(offsets.size - 1), int(frags.size)
        if k and block_at is None:
            raise ArrayError("placing into a dense index needs block_at")
        b0 = int(block_at or 0)
        spans = _reserved(level_group, dense.SPANS_PATH, at + n, "place_manifests")
        blocks = _reserved(level_group, dense.BLOCKS_PATH, b0 + k, "place_manifests")
        counts = np.diff(offsets)
        spans[at:at + n] = np.stack([b0 + offsets[:-1], counts], axis=1)
        if k:
            blocks[b0:b0 + k] = np.concatenate([coords, frags[:, None]], axis=1)
        stores_ids = True
    else:
        if manifest_blobs is None:
            if offsets is None or coords is None or frags is None:
                raise ArrayError("give manifest_blobs, or offsets, coords and frags")
            manifest_blobs = encode_object_manifests_csr(coords, frags, offsets, sid_ndim=sid)
        blobs = np.empty(len(manifest_blobs), dtype=object)
        blobs[:] = list(manifest_blobs)
        n = int(blobs.size)
        node = _reserved(level_group, f"{OBJECT_INDEX}/manifests", at + n, "place_manifests")
        node[at:at + n] = blobs
        stores_ids = layout == OBJECT_INDEX_LAYOUT_V2
    rows = np.arange(at, at + n, dtype=np.int64)
    new_ids = rows if ids is None else np.asarray(ids, dtype=np.int64).reshape(-1)
    if new_ids.size != n:
        raise ArrayError(f"{new_ids.size} ids for {n} manifests")
    if stores_ids:
        table = _reserved(
            level_group, f"{OBJECT_INDEX}/{OBJECT_IDS_ARRAY}", at + n, "place_manifests",
        )
        table[at:at + n] = new_ids
    elif not np.array_equal(new_ids, rows):
        raise ArrayError(
            "this object index stores ids positionally (row i is object i); "
            "ids= must be the rows being written"
        )
    return at, n


def place_attribute_columns(
    level_group: Group, columns: Mapping[str, Any], at: int,
) -> None:
    """Write attribute rows into reserved rows ``[at, at + n)``, and nothing else.

    The in-place twin of the append mode: no resize, no gap fill, no
    metadata. Every column must have been reserved
    (:func:`reserve_object_rows`) to at least ``at + n`` rows.
    """
    from zarr_vectors import _xp

    at = int(at)
    if at < 0:
        raise ArrayError(f"at must be >= 0, got {at}")
    for name, data in columns.items():
        rows = np.asarray(_xp.to_host(data))
        path = f"{OBJECT_ATTRIBUTES}/{name}"
        node = _reserved(level_group, path, at + rows.shape[0], "place_attribute_columns")
        if tuple(node.shape[1:]) != tuple(rows.shape[1:]):
            raise ArrayError(
                f"{path} holds rows of {tuple(node.shape[1:])}; got {tuple(rows.shape[1:])}"
            )
        node[at:at + rows.shape[0]] = rows.astype(node.dtype, copy=False)


def aligned_regions(counts: Any, alignment: int) -> npt.NDArray[np.int64]:
    """Start of each of several writers' regions, each aligned.

    For a coordinator assigning dense-index block regions (or row ranges
    of uneven size): writer ``i`` gets ``[starts[i], starts[i] +
    counts[i])``, and the total to reserve is ``starts[-1]``.
    """
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    if counts.size and int(counts.min()) < 0:
        raise ArrayError("counts must not be negative")
    padded = -(-counts // alignment) * alignment
    return np.concatenate([[0], np.cumsum(padded)]).astype(np.int64)


__all__ = [
    "UNWRITTEN_ID",
    "aligned_regions",
    "concurrency_contract",
    "place_attribute_columns",
    "place_manifests",
    "reserve_object_rows",
    "row_alignment",
]
