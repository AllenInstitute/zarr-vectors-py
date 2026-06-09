"""Create, write, and read ZV per-level arrays.

Each "array" is a subdirectory within a resolution-level group.  Chunk
data is stored as raw binary files named by their spatial chunk
coordinates (e.g. ``vertices/0.0.0``).  Array metadata is in
``<array>/.zattrs``.

This module is the single point of contact for all array I/O — type
modules (``types/*.py``) call these functions rather than touching
the store or encoding modules directly.
"""

from __future__ import annotations

import math
import warnings
from contextlib import contextmanager
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import zarr
from zarr.codecs import VLenBytesCodec
from zarr.errors import UnstableSpecificationWarning

from zarr_vectors.constants import (
    CROSS_CHUNK_LINK_ATTRIBUTES,
    CROSS_CHUNK_LINKS,
    FRAGMENT_ATTRIBUTES,
    GROUP_ATTRIBUTES,
    GROUPS,
    LINK_ATTRIBUTES,
    LINK_FRAGMENTS,
    LINKS,
    OBJECT_ATTRIBUTES,
    OBJECT_INDEX,
    VERTEX_ATTRIBUTES,
    VERTEX_FRAGMENTS,
    VERTICES,
)
from zarr_vectors.core.paths import (
    cross_chunk_link_attributes_path,
    cross_chunk_links_path,
    format_delta,
    leaf_index_key,
    link_attributes_path,
    links_path,
    parse_delta,
    parse_leaf_index_key,
)
from zarr_vectors.core.metadata import (
    LevelMetadata,
    RootMetadata,
    get_level_chunk_shape,
)
from zarr_vectors.core.store import FsGroup
from zarr_vectors.encoding.fragments import (
    ChunkFragmentIndex,
    decode_fragments,
    decode_object_manifest_blocks,
    encode_fragments,
    encode_object_manifest_blocks,
)
from zarr_vectors.encoding.ragged import (
    decode_ragged_blob,
    decode_ragged_floats,
    decode_ragged_ints,
    encode_ragged_blob,
    encode_ragged_floats,
    encode_ragged_ints,
)
from zarr_vectors.exceptions import ArrayError, StoreError
from zarr_vectors.typing import (
    ChunkCoords,
    CrossChunkLink,
    ObjectManifest,
)


# Sentinel for read-side ``default=...`` soft-fail kwargs.  Lets callers
# distinguish "did not pass default" (caller wants the raise) from
# "passed default=None" (caller wants None back on miss).
_UNSET: Any = object()


# Discriminator written into ``object_index`` group attrs to identify
# the ragged-array layout (one vlen-bytes zarr array named ``manifests``,
# one entry per object_id).  Absent on legacy stores, which used a pair
# of single-chunk ``data`` + ``offsets`` byte blobs.
OBJECT_INDEX_LAYOUT_V1 = "vlen_manifests_v1"

# Objects per zarr chunk of ``object_index/manifests``.  A single-object
# read fetches only the chunk containing the requested oid, so this sets
# the read amplification ceiling (~16K manifest blobs per fetch).
OBJECT_INDEX_MANIFEST_BUCKET = 16_384


# ===================================================================
# Helpers
# ===================================================================

def _chunk_key(coords: ChunkCoords) -> str:
    """Convert chunk coordinates to a dot-separated key string.

    ``(0, 1, 2)`` → ``"0.1.2"``
    """
    return ".".join(str(c) for c in coords)


def _parse_chunk_key(key: str) -> ChunkCoords:
    """Parse a dot-separated chunk key back to coordinates.

    ``"0.1.2"`` → ``(0, 1, 2)``
    """
    return tuple(int(x) for x in key.split("."))


@contextmanager
def _maybe_batched_reads(
    level_group: FsGroup,
    plan: list[tuple[str, list[str]]],
):
    """Open a single-chunk prefetch unless one is already active.

    Wraps :meth:`FsGroup.batched_reads` for the common "I'm about to
    read N>1 sibling arrays for one chunk" case: it fans the reads out
    in one ``asyncio.gather`` instead of paying N sequential
    round-trips.  When the caller has already entered an outer
    :meth:`batched_reads` context (i.e. a multi-chunk loop), this is a
    no-op so the outer plan stays in charge.
    """
    if level_group._prefetch_cache is not None:
        yield
        return
    with level_group.batched_reads(plan):
        yield


def _short_circuit_existing(
    level_group: FsGroup,
    full_name: str,
    exist_ok: bool,
) -> bool:
    """Return True when ``create_*_array`` should no-op because the array
    already exists and ``exist_ok=True``.  Raises :class:`ArrayError` when
    the array exists and ``exist_ok=False``.
    """
    if not level_group.array_exists(full_name):
        return False
    if exist_ok:
        return True
    raise ArrayError(
        f"{full_name!r} already exists; pass exist_ok=True to ignore"
    )


def _ensure_array_dir(level_group: FsGroup, array_name: str) -> None:
    """Ensure an array subdirectory exists within a level group.

    Inside :meth:`Group.batched_writes` we skip the sync ``require_group``
    round-trip — the metadata flush will PUT the parent ``zarr.json``
    directly, including the right attributes, in the same gather as the
    chunk PUTs.  Outside batched mode we still call ``require_group`` so
    the parent group exists before any subsequent ``write_array_meta``
    call (which only ``attrs.update``s — it does not create the group).
    """
    if level_group._pending_array_metas is not None:
        return
    level_group.require_group(array_name)


def read_zv_array_tag(meta: dict) -> str | None:
    """Return the discriminator string from an array's ``.zattrs`` dict.

    Looks under the new key ``zv_array`` first and falls back to the
    legacy ``zvf_array`` key for stores written before the rename.  Use
    this helper instead of indexing the dict directly so that conformance
    / validation code keeps working against existing-on-disk stores.
    """
    return meta.get("zv_array", meta.get("zvf_array"))


# ===================================================================
# Array creation (set up directory + metadata)
# ===================================================================

def create_vertices_array(
    level_group: FsGroup,
    dtype: str = "float32",
    encoding: str = "raw",
    *,
    exist_ok: bool = True,
) -> None:
    """Create the ``vertices/`` array within a resolution level.

    Args:
        level_group: The resolution level FsGroup.
        dtype: Numpy dtype string for vertex positions.
        encoding: ``"raw"`` or ``"draco"``.
        exist_ok: When True (default), no-op if the array already exists.
            When False, raise :class:`ArrayError` on conflict.
    """
    if _short_circuit_existing(level_group, VERTICES, exist_ok):
        return
    _ensure_array_dir(level_group, VERTICES)
    _ensure_array_dir(level_group, VERTEX_FRAGMENTS)
    level_group.write_array_meta(VERTICES, {
        "zv_array": "vertices",
        "dtype": dtype,
        "encoding": encoding,
    })
    level_group.write_array_meta(VERTEX_FRAGMENTS, {
        "zv_array": VERTEX_FRAGMENTS,
        "encoding": "fragment_index_v1",
    })


def create_links_array(
    level_group: FsGroup,
    link_width: int,
    dtype: str = "int64",
    *,
    delta: int = 0,
    exist_ok: bool = True,
) -> None:
    """Create a ``links/<delta>/`` array.

    Under the 0.4 multiscale links layout each ``<delta>`` segment is a
    distinct array; ``delta=0`` is the intra-level array (the only one
    written pre-0.4) and non-zero deltas hold edges that point ``delta``
    pyramid levels away (positive = coarser, negative = finer).

    Args:
        level_group: The resolution level FsGroup.
        link_width: Number of vertex indices per link entry (L).
            1 for skeleton parents, 2 for edges, 3 for triangle faces.
        dtype: Integer dtype.
        delta: Level delta; see :mod:`zarr_vectors.core.paths`.
        exist_ok: When True (default), no-op if the array already exists.
            When False, raise :class:`ArrayError` on conflict.
    """
    full_name = links_path(delta)
    if _short_circuit_existing(level_group, full_name, exist_ok):
        return
    _ensure_array_dir(level_group, full_name)
    level_group.write_array_meta(full_name, {
        "zv_array": "links",
        "dtype": dtype,
        "link_width": link_width,
        "level_delta": int(delta),
    })


def create_attribute_array(
    level_group: FsGroup,
    name: str,
    dtype: str = "float32",
    channel_names: list[str] | None = None,
    extra_meta: dict[str, Any] | None = None,
    *,
    exist_ok: bool = True,
) -> None:
    """Create a vertex attribute array ``attributes/<name>/``.

    Args:
        level_group: The resolution level FsGroup.
        name: Attribute name (e.g. ``"radius"``, ``"gene_expression"``).
        dtype: Numpy dtype string.
        channel_names: Optional list of channel names.
        extra_meta: Additional JSON-serialisable fields merged into the
            array metadata.  Used for the dictionary-encoding
            convention (``encoding="dictionary"``, ``categories``,
            ``ordered``, ``_FillValue``) and other userspace
            extensions.  Keys collide-check against the core fields
            (``zv_array``, ``name``, ``dtype``, ``channel_names``).
    """
    full_name = f"{VERTEX_ATTRIBUTES}/{name}"
    if _short_circuit_existing(level_group, full_name, exist_ok):
        return
    _ensure_array_dir(level_group, full_name)
    meta: dict[str, Any] = {
        "zv_array": "attribute",
        "name": name,
        "dtype": dtype,
    }
    if channel_names is not None:
        meta["channel_names"] = channel_names
    if extra_meta:
        reserved = {"zv_array", "name", "dtype", "channel_names"}
        clobber = reserved & set(extra_meta)
        if clobber:
            raise ArrayError(
                f"extra_meta cannot override core attribute fields: {sorted(clobber)}"
            )
        meta.update(extra_meta)
    level_group.write_array_meta(full_name, meta)


def create_fragment_attribute_array(
    level_group: FsGroup,
    name: str,
    dtype: str = "float32",
    channel_names: list[str] | None = None,
    extra_meta: dict[str, Any] | None = None,
    *,
    exist_ok: bool = True,
) -> None:
    """Create a fragment attribute array ``fragment_attributes/<name>/``.

    Per-chunk dense byte blob storing one row per fragment in the chunk;
    row count is derived from ``vertex_fragments/<chunk>`` at read time.
    Optional storage layer — the common opt-in use case is materializing
    parent-IDs as attributes (e.g. the OID owning each fragment as a
    fragment attribute ``object_id``).

    Args:
        level_group: The resolution level FsGroup.
        name: Attribute name (e.g. ``"object_id"``).
        dtype: Numpy dtype string.
        channel_names: Optional list of channel names.  When provided,
            row shape becomes ``(num_fragments, len(channel_names))``;
            otherwise rows are scalar.
        extra_meta: Additional JSON-serialisable fields merged into the
            array metadata.  Same collision rules as
            :func:`create_attribute_array`.
        exist_ok: When True (default), no-op if the array already exists.
            When False, raise :class:`ArrayError` on conflict.
    """
    full_name = f"{FRAGMENT_ATTRIBUTES}/{name}"
    if _short_circuit_existing(level_group, full_name, exist_ok):
        return
    _ensure_array_dir(level_group, full_name)
    meta: dict[str, Any] = {
        "zv_array": "fragment_attribute",
        "name": name,
        "dtype": dtype,
    }
    if channel_names is not None:
        meta["channel_names"] = channel_names
    if extra_meta:
        reserved = {"zv_array", "name", "dtype", "channel_names"}
        clobber = reserved & set(extra_meta)
        if clobber:
            raise ArrayError(
                f"extra_meta cannot override core fragment_attribute fields: "
                f"{sorted(clobber)}"
            )
        meta.update(extra_meta)
    level_group.write_array_meta(full_name, meta)


def create_object_index_array(
    level_group: FsGroup,
    *,
    exist_ok: bool = True,
) -> None:
    """Create the ``object_index/`` array.

    Args:
        level_group: Resolution level group.
        exist_ok: When True (default), no-op if the array already exists.
            When False, raise :class:`ArrayError` on conflict.
    """
    if _short_circuit_existing(level_group, OBJECT_INDEX, exist_ok):
        return
    _ensure_array_dir(level_group, OBJECT_INDEX)
    level_group.write_array_meta(OBJECT_INDEX, {
        "zv_array": "object_index",
    })


def create_object_attributes_array(
    level_group: FsGroup,
    name: str,
    dtype: str = "float32",
    num_channels: int = 1,
    *,
    exist_ok: bool = True,
) -> None:
    """Create an object attribute array ``object_attributes/<name>/``.

    Args:
        level_group: The resolution level FsGroup.
        name: Attribute name.
        dtype: Numpy dtype string.
        num_channels: Number of channels (C dimension).
        exist_ok: When True (default), no-op if the array already exists.
            When False, raise :class:`ArrayError` on conflict.
    """
    full_name = f"{OBJECT_ATTRIBUTES}/{name}"
    if _short_circuit_existing(level_group, full_name, exist_ok):
        return
    _ensure_array_dir(level_group, full_name)
    level_group.write_array_meta(full_name, {
        "zv_array": "object_attribute",
        "name": name,
        "dtype": dtype,
        "num_channels": num_channels,
    })


def create_groupings_array(
    level_group: FsGroup,
    *,
    exist_ok: bool = True,
) -> None:
    """Create the ``groupings/`` array.

    Args:
        level_group: Resolution level group.
        exist_ok: When True (default), no-op if the array already exists.
            When False, raise :class:`ArrayError` on conflict.
    """
    if _short_circuit_existing(level_group, GROUPS, exist_ok):
        return
    _ensure_array_dir(level_group, GROUPS)
    level_group.write_array_meta(GROUPS, {
        "zv_array": "groups",
    })


def create_groupings_attributes_array(
    level_group: FsGroup,
    name: str,
    dtype: str = "float32",
    num_channels: int = 1,
    *,
    exist_ok: bool = True,
) -> None:
    """Create a groupings attribute array ``groupings_attributes/<name>/``.

    Args:
        level_group: Resolution level group.
        name: Attribute name.
        dtype: Numpy dtype string.
        num_channels: Number of channels (C dimension).
        exist_ok: When True (default), no-op if the array already exists.
            When False, raise :class:`ArrayError` on conflict.
    """
    full_name = f"{GROUP_ATTRIBUTES}/{name}"
    if _short_circuit_existing(level_group, full_name, exist_ok):
        return
    _ensure_array_dir(level_group, full_name)
    level_group.write_array_meta(full_name, {
        "zv_array": "groupings_attribute",
        "name": name,
        "dtype": dtype,
        "num_channels": num_channels,
    })


def create_cross_chunk_links_array(
    level_group: FsGroup,
    *,
    delta: int = 0,
    link_width: int = 2,
    sid_ndim: int | None = None,
    exist_ok: bool = True,
) -> None:
    """Create a ``cross_chunk_links/<delta>/`` array (v0.8 vlen-bytes layout).

    Records are partitioned by the sorted unique chunks each record
    touches and stored as one cell of a 1-D vlen-bytes zarr Array.  A
    sidecar ``leaf_index`` in the group's ``.zattrs`` maps the sorted
    chunks tuple to the cell ordinal so readers can look up "records
    between A and B" with a single sort + dict lookup.  See
    :func:`write_cross_chunk_links` for the cell payload format.

    Source-side endpoints (endpoint 0) live at the owning resolution
    level; target-side endpoints (1..L-1) live at ``this_level + delta``.

    Args:
        level_group: Resolution level group.
        delta: Level delta (0 for intra-level, ±N for cross-level).
        link_width: Number of vertex refs per record.  2 for edges,
            3 for triangle faces, 1 for parent→child metanode refs.
        sid_ndim: Spatial-index dimension arity, stamped on group meta.
            Defaulted by writers when omitted here.
        exist_ok: When True (default), no-op if the group already exists.
            When False, raise :class:`ArrayError` on conflict.
    """
    full_name = cross_chunk_links_path(delta)
    if _short_circuit_existing(level_group, full_name, exist_ok):
        return
    _ensure_array_dir(level_group, full_name)
    meta = {
        "zv_array": "cross_chunk_links",
        "level_delta": int(delta),
        "link_width": int(link_width),
    }
    if sid_ndim is not None:
        meta["sid_ndim"] = int(sid_ndim)
    level_group.write_array_meta(full_name, meta)


def create_link_attributes_array(
    level_group: FsGroup,
    name: str,
    dtype: str = "float32",
    *,
    delta: int = 0,
    exist_ok: bool = True,
) -> None:
    """Create a ``link_attributes/<name>/<delta>/`` array (parallel to
    the matching ``links/<delta>/`` array).

    ``exist_ok=True`` (default) makes the call idempotent; pass
    ``exist_ok=False`` to raise :class:`ArrayError` on conflict.
    """
    full_name = link_attributes_path(name, delta)
    if _short_circuit_existing(level_group, full_name, exist_ok):
        return
    _ensure_array_dir(level_group, full_name)
    level_group.write_array_meta(full_name, {
        "zv_array": "link_attribute",
        "name": name,
        "dtype": dtype,
        "level_delta": int(delta),
    })


def create_cross_chunk_link_attributes_array(
    level_group: FsGroup,
    name: str,
    dtype: str = "float32",
    *,
    delta: int = 0,
    exist_ok: bool = True,
) -> None:
    """Create a ``cross_chunk_link_attributes/<name>/<delta>/`` array (v0.8 vlen-bytes).

    Parallel attribute storage for the matching ``cross_chunk_links/<delta>/``
    array.  Attribute rows are partitioned in lockstep with the link
    cells: each cell holds the rows for the corresponding link leaf.
    The leaf-ordering and ``leaf_index`` mirror the parent CCL array's
    sidecar.

    ``exist_ok=True`` (default) makes the call idempotent; pass
    ``exist_ok=False`` to raise :class:`ArrayError` on conflict.
    """
    full_name = cross_chunk_link_attributes_path(name, delta)
    if _short_circuit_existing(level_group, full_name, exist_ok):
        return
    _ensure_array_dir(level_group, full_name)
    level_group.write_array_meta(full_name, {
        "zv_array": "cross_chunk_link_attribute",
        "name": name,
        "dtype": dtype,
        "level_delta": int(delta),
    })


# ===================================================================
# Writing data
# ===================================================================

def write_chunk_vertices(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
    groups: list[npt.NDArray[np.floating]],
    dtype: np.dtype | str = np.float32,
) -> npt.NDArray[np.int64]:
    """Write fragments to a spatial chunk.

    Encodes the groups as a contiguous byte buffer in ``vertices/`` and
    writes a v0.6 fragment-index to ``vertex_fragments/`` describing each
    group as a contiguous range of vertex rows in source order.

    Args:
        level_group: Resolution level group.
        chunk_coords: Spatial chunk coordinates.
        groups: List of arrays, each ``(N_k, D)``.
        dtype: Numpy dtype for serialisation.

    Returns:
        ``(K,)`` int64 array of vertex byte offsets (kept for backwards-
        compatible signature; callers that need the v0.6 fragment-index
        should use :func:`read_vertex_fragment_index`).
    """
    dtype = np.dtype(dtype)
    key = _chunk_key(chunk_coords)

    raw_bytes, vertex_byte_offsets = encode_ragged_floats(groups, dtype)
    level_group.write_bytes(VERTICES, key, raw_bytes)

    # Express each group as a contiguous (start_row, count) fragment.
    if len(groups) == 0:
        fragments: list[tuple[int, int]] = []
    else:
        per_group_counts = [int(np.asarray(g).shape[0]) for g in groups]
        cumulative = 0
        fragments = []
        for n in per_group_counts:
            fragments.append((cumulative, n))
            cumulative += n
    level_group.write_bytes(
        VERTEX_FRAGMENTS, key, encode_fragments(fragments),
    )
    return vertex_byte_offsets


def write_chunk_links(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
    link_groups: list[npt.NDArray[np.integer]],
    dtype: np.dtype | str = np.int64,
    *,
    delta: int = 0,
) -> npt.NDArray[np.int64]:
    """Write link groups to a spatial chunk under ``links/<delta>/``.

    For ``delta=0`` link groups are 1:1 aligned with the chunk's
    fragments; readers derive per-group link byte offsets from the
    cumulative sizes of each group's link bytes (see
    :func:`read_chunk_links`).

    For ``delta != 0`` (cross-pyramid-level links) the source vertex
    groups and link groups live at different levels and there is
    typically one link group spanning the chunk.

    Args:
        level_group: Resolution level group.
        chunk_coords: Spatial chunk coordinates.
        link_groups: List of arrays, each ``(M_k, L)``.
        dtype: Integer dtype.
        delta: Level delta; see :mod:`zarr_vectors.core.paths`.

    Returns:
        ``(K,)`` int64 array of link byte offsets.
    """
    dtype = np.dtype(dtype)
    key = _chunk_key(chunk_coords)
    full_name = links_path(delta)

    if delta == 0 and level_group.chunk_exists(VERTEX_FRAGMENTS, key):
        existing_fi = decode_fragments(
            level_group.read_bytes(VERTEX_FRAGMENTS, key),
        )
        if existing_fi.num_fragments != len(link_groups):
            raise ArrayError(
                f"Link group count ({len(link_groups)}) != "
                f"vertex fragment count ({existing_fi.num_fragments}) in chunk {key}"
            )

    if delta == 0:
        # v0.6 intra-level: flat concatenated link data + sibling
        # link_fragments/ describing per-group row ranges.
        data_bytes, link_byte_offsets = encode_ragged_ints(link_groups, dtype)
        level_group.write_bytes(full_name, key, data_bytes)

        link_row_size = dtype.itemsize * (
            int(np.asarray(link_groups[0]).shape[1]) if (
                link_groups and np.asarray(link_groups[0]).ndim == 2
            ) else 1
        )
        # Fragment per group as a contiguous range of link rows.
        if len(link_groups) == 0:
            link_fragments: list[tuple[int, int]] = []
        else:
            cumulative = 0
            link_fragments = []
            for g in link_groups:
                n = int(np.asarray(g).shape[0]) if np.asarray(g).ndim >= 1 else 0
                link_fragments.append((cumulative, n))
                cumulative += n
        # Ensure the sibling array group exists.
        if not level_group.chunk_exists(LINK_FRAGMENTS, key):
            level_group.require_group(LINK_FRAGMENTS)
            try:
                level_group.read_array_meta(LINK_FRAGMENTS)
            except Exception:
                level_group.write_array_meta(LINK_FRAGMENTS, {
                    "zv_array": LINK_FRAGMENTS,
                    "encoding": "fragment_index_v1",
                })
        level_group.write_bytes(
            LINK_FRAGMENTS, key, encode_fragments(link_fragments),
        )
        del link_row_size  # silence unused-variable warning
        return link_byte_offsets

    # delta != 0: cross-level links keep the v0.5 inline self-describing
    # layout (out of scope for the v0.6 fragment-index refactor).
    blob = encode_ragged_blob(link_groups, dtype)
    level_group.write_bytes(full_name, key, blob)
    _, link_byte_offsets = encode_ragged_ints(link_groups, dtype)
    return link_byte_offsets


def write_chunk_fragments(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
    new_fragments: list,
    *,
    target: Literal["vertex", "link"] = "vertex",
    mode: Literal["replace", "append"] = "replace",
) -> list[int]:
    """Write fragment-index entries to a chunk's vertex_fragments/<chunk>
    or link_fragments/<chunk> blob.

    Args:
        level_group: Resolution level group.
        chunk_coords: Spatial chunk coordinates.
        new_fragments: Mix of ``(start, count)`` range tuples and
            ``np.ndarray[int64]`` explicit index arrays. Order is
            preserved in the output index.
        target: ``"vertex"`` writes to ``vertex_fragments/<chunk>``;
            ``"link"`` writes to ``link_fragments/<chunk>``.
        mode: ``"replace"`` writes ``new_fragments`` as the whole blob.
            ``"append"`` reads the existing blob, decodes its fragments,
            concatenates ``new_fragments``, re-encodes, writes back.
            If the blob does not yet exist, ``"append"`` behaves like a
            first-time ``"replace"``.

    Returns:
        Fragment-index values assigned to the newly-written entries.
        ``"replace"`` returns ``list(range(len(new_fragments)))``.
        ``"append"`` returns
        ``list(range(n_existing, n_existing + len(new_fragments)))``;
        existing fragment_index values are stable. An empty
        ``new_fragments`` in append mode returns ``[]`` and does not
        write the blob.

    Raises:
        ArrayError: If ``target`` or ``mode`` is invalid.

    Concurrency:
        Read-modify-write. ``write_bytes`` is delete-then-create — last
        writer wins. This is NOT cross-writer-safe; callers must
        serialise concurrent appends to the same chunk's fragment-index
        (per-chunk sharding satisfies this).
    """
    if target == "vertex":
        constant = VERTEX_FRAGMENTS
    elif target == "link":
        constant = LINK_FRAGMENTS
    else:
        raise ArrayError(
            f"target must be 'vertex' or 'link', got {target!r}"
        )
    if mode not in ("replace", "append"):
        raise ArrayError(
            f"mode must be 'replace' or 'append', got {mode!r}"
        )

    key = _chunk_key(chunk_coords)
    new_list = list(new_fragments)

    if mode == "replace":
        level_group.write_bytes(constant, key, encode_fragments(new_list))
        return list(range(len(new_list)))

    # append
    if not new_list:
        return []   # no-op append — don't touch the blob

    def _fi_to_list(raw: bytes) -> list:
        fi = decode_fragments(raw)
        return [
            fi.range(i) if fi.is_range(i) else fi.indices(i)
            for i in range(fi.num_fragments)
        ]

    _, existing = _read_modify_write_blob(
        level_group, constant, key,
        decode_fn=_fi_to_list,
        merge_fn=lambda ex: ex + new_list,
        encode_fn=encode_fragments,
        initial=[],
    )
    n_before = len(existing)
    return list(range(n_before, n_before + len(new_list)))


def write_chunk_attributes(
    level_group: FsGroup,
    attr_name: str,
    chunk_coords: ChunkCoords,
    attr_groups: list[npt.NDArray],
    dtype: np.dtype | str = np.float32,
) -> None:
    """Write vertex attribute data for groups in a spatial chunk.

    Attribute groups align 1:1 with fragments, so per-group byte
    offsets are derived at read time from ``vertex_fragments`` and
    the attribute dtype/ncols.  No sibling ``_offsets`` blob is written.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name (e.g. ``"radius"``).
        chunk_coords: Spatial chunk coordinates.
        attr_groups: List of arrays aligned with fragments.
            Each array is ``(N_k,)`` for scalar or ``(N_k, C)`` for
            multi-channel attributes.
        dtype: Numpy dtype.
    """
    dtype = np.dtype(dtype)
    key = _chunk_key(chunk_coords)
    full_name = f"{VERTEX_ATTRIBUTES}/{attr_name}"
    raw_bytes, _ = encode_ragged_floats(attr_groups, dtype)
    level_group.write_bytes(full_name, key, raw_bytes)


def write_chunk_fragment_attributes(
    level_group: FsGroup,
    attr_name: str,
    chunk_coords: ChunkCoords,
    data: npt.NDArray,
    dtype: np.dtype | str = np.float32,
) -> None:
    """Write per-fragment attribute data for a spatial chunk.

    The on-disk layout is a single dense byte blob whose row count
    equals ``num_fragments_in_chunk``.  Fragment count is not validated
    against ``vertex_fragments/<chunk>`` at write time — the read side
    fails loudly on byte-length mismatch, so callers are trusted to
    pass a correctly-sized array.  Replace-only at the chunk level.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name (e.g. ``"object_id"``).
        chunk_coords: Spatial chunk coordinates.
        data: ``(F,)`` for scalar or ``(F, C)`` for multi-channel,
            where ``F`` is the number of fragments in this chunk.
        dtype: Numpy dtype to cast ``data`` to before writing.
    """
    dtype = np.dtype(dtype)
    key = _chunk_key(chunk_coords)
    full_name = f"{FRAGMENT_ATTRIBUTES}/{attr_name}"
    arr = np.ascontiguousarray(np.asarray(data).astype(dtype, copy=False))
    level_group.write_bytes(full_name, key, arr.tobytes())


def write_chunk_link_attributes(
    level_group: FsGroup,
    attr_name: str,
    chunk_coords: ChunkCoords,
    attr_groups: list[npt.NDArray],
    dtype: np.dtype | str = np.float32,
    *,
    delta: int = 0,
) -> None:
    """Write per-edge attribute data parallel to ``links/<delta>/``.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name (e.g. ``"weight"``).
        chunk_coords: Spatial chunk coordinates.
        attr_groups: List of arrays, each ``(M_k,)`` or ``(M_k, C)``,
            aligned with link groups in the ``links/<delta>/`` array.
        dtype: Numpy dtype.
        delta: Level delta; see :mod:`zarr_vectors.core.paths`.
    """
    dtype = np.dtype(dtype)
    key = _chunk_key(chunk_coords)
    full_name = link_attributes_path(attr_name, delta)
    raw_bytes, _ = encode_ragged_floats(attr_groups, dtype)
    level_group.write_bytes(full_name, key, raw_bytes)


def write_object_index(
    level_group: FsGroup,
    manifests: dict[int, ObjectManifest],
    sid_ndim: int,
    *,
    total_objects: int | None = None,
) -> None:
    """Write object index: object_id → ordered fragment references.

    Args:
        level_group: Resolution level group.
        manifests: ``{object_id: [(chunk_coords, fragment_index), ...], ...}``.
            Sparse — OIDs absent from the dict get empty manifests.
        sid_ndim: Number of spatial index dimensions.
        total_objects: Number of OID slots to write.  When provided,
            the dense manifest list spans ``range(total_objects)`` even
            if the largest OID present is smaller — used by the
            ID-preserving pyramid regime, where surviving OIDs are a
            sparse subset of the parent's OID space.  When ``None``
            (default), the size is ``max(manifests.keys()) + 1``
            (legacy behaviour).
    """
    if not manifests and total_objects is None:
        return

    if total_objects is not None:
        size = int(total_objects)
    else:
        size = max(manifests.keys()) + 1
    # Build a dense list, filling gaps with empty manifests
    manifest_list: list[list[tuple[tuple[int, ...], int]]] = []
    for oid in range(size):
        manifest_list.append(manifests.get(oid, []))

    # v0.6 manifest-block encoding.  Each old (chunk, fragment_index) tuple
    # becomes one mode-0 (single fragment) block.  Range / explicit
    # short-circuits are reserved for writers that know they produce
    # ranges or fragment-list shapes — to be plumbed through the
    # higher-level write APIs in a future change.
    manifest_blobs: list[bytes] = []
    for manifest in manifest_list:
        blocks = [
            (tuple(int(c) for c in chunk_coords), int(fragment_index))
            for chunk_coords, fragment_index in manifest
        ]
        manifest_blobs.append(
            encode_object_manifest_blocks(blocks, sid_ndim=sid_ndim)
        )

    _write_object_index_manifests(level_group, manifest_blobs)
    level_group.write_array_meta(OBJECT_INDEX, {
        "zv_array": "object_index",
        "num_objects": size,
        "sid_ndim": sid_ndim,
        "layout": OBJECT_INDEX_LAYOUT_V1,
    })


def _write_object_index_manifests(
    level_group: FsGroup,
    manifest_blobs: list[bytes],
) -> None:
    """Write ``object_index/manifests`` as a single ragged vlen-bytes array.

    One zarr chunk holds ``OBJECT_INDEX_MANIFEST_BUCKET`` consecutive
    objects, so a single-object read fetches at most one chunk regardless
    of total ``num_objects`` — fixing the legacy O(num_objects) read
    amplification.  Drops legacy ``object_index/{data,offsets}`` arrays
    if they exist from a prior write.
    """
    n = len(manifest_blobs)
    oi_group = level_group.zarr_group.require_group(OBJECT_INDEX)
    for legacy in ("manifests", "data", "offsets"):
        if legacy in oi_group:
            del oi_group[legacy]

    if n == 0:
        return

    chunk_size = min(OBJECT_INDEX_MANIFEST_BUCKET, n)
    # zarr 3.x's variable-length bytes dtype lacks a finalised V3 spec
    # (zarr-extensions tracks it); the warning is informational and ZVF
    # is alpha — accept it and silence at the call site so writes stay
    # quiet.  Revisit if the spec lands incompatibly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnstableSpecificationWarning)
        arr = oi_group.create_array(
            "manifests",
            shape=(n,),
            chunks=(chunk_size,),
            dtype="bytes",
            serializer=VLenBytesCodec(),
        )
        obj = np.empty(n, dtype=object)
        for i, blob in enumerate(manifest_blobs):
            obj[i] = blob
        arr[:] = obj


def write_object_attributes(
    level_group: FsGroup,
    attr_name: str,
    data: npt.NDArray,
    *,
    present_mask: npt.NDArray | None = None,
    mode: Literal["replace", "append"] = "replace",
) -> None:
    """Write dense O×C object attribute data.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name.
        data: ``(O,)`` or ``(O, C)`` array.  In ``mode="append"`` this is
            interpreted as the NEW rows to append.
        present_mask: Optional ``(O,)`` byte array (``0``/``1`` per
            object) marking which rows are real.  Required for levels
            that use ID-preserving sparsification — rows for dropped
            objects have ``mask[i] == 0`` and the corresponding
            ``data[i]`` row is dtype-zero padding.  When omitted, every
            row is assumed real (backwards compatible).  In append mode,
            this is the mask for the NEW rows; existing rows get
            ``1`` backfilled if the existing array had no mask.
        mode: ``"replace"`` (default, current behaviour) writes ``data``
            as the full array.  ``"append"`` reads the existing array,
            concatenates ``data`` along axis 0, writes back.  Existing
            dtype wins on dtype mismatch (new rows are cast).  If the
            attribute does not exist yet, the first append behaves like
            a ``"replace"``.

    Raises:
        ArrayError: If ``mode`` is invalid, or if the appended row shape
            (everything beyond axis 0) does not match the existing array.

    Concurrency:
        ``mode="append"`` is read-modify-write and NOT cross-writer-safe.
        Callers must serialise concurrent appends to the same attribute.
    """
    if mode == "replace":
        full_name = f"{OBJECT_ATTRIBUTES}/{attr_name}"
        _ensure_array_dir(level_group, full_name)
        level_group.write_bytes(full_name, "data", data.tobytes())
        if present_mask is not None:
            mask = np.asarray(present_mask, dtype=np.uint8)
            if mask.shape[0] != data.shape[0]:
                raise ArrayError(
                    f"present_mask length {mask.shape[0]} != data row count "
                    f"{data.shape[0]}"
                )
            level_group.write_bytes(full_name, "present_mask", mask.tobytes())
        level_group.write_array_meta(full_name, {
            "zv_array": "object_attribute",
            "name": attr_name,
            "dtype": str(data.dtype),
            "shape": list(data.shape),
            "has_present_mask": bool(present_mask is not None),
        })
        return

    if mode != "append":
        raise ArrayError(
            f"mode must be 'replace' or 'append', got {mode!r}"
        )

    # Append branch ------------------------------------------------------
    full_name = f"{OBJECT_ATTRIBUTES}/{attr_name}"
    data = np.asarray(data)

    meta = level_group.read_array_meta(full_name)
    if meta and "shape" in meta and level_group.chunk_exists(full_name, "data"):
        existing = read_object_attributes(level_group, attr_name)
        had_existing_mask = bool(meta.get("has_present_mask"))
    else:
        existing = None
        had_existing_mask = False

    if existing is None:
        combined = data
    else:
        # Tail-shape (everything beyond axis 0) must match.
        if existing.shape[1:] != data.shape[1:]:
            raise ArrayError(
                f"append shape mismatch: existing {existing.shape} vs "
                f"new {data.shape} — tail dimensions must match"
            )
        new_cast = data.astype(existing.dtype, copy=False)
        combined = np.concatenate([existing, new_cast], axis=0)

    _ensure_array_dir(level_group, full_name)
    level_group.write_bytes(
        full_name, "data", np.ascontiguousarray(combined).tobytes(),
    )

    has_mask = had_existing_mask or (present_mask is not None)
    if present_mask is not None:
        new_mask = np.asarray(present_mask, dtype=np.uint8)
        if new_mask.shape[0] != data.shape[0]:
            raise ArrayError(
                f"present_mask length {new_mask.shape[0]} != appended row "
                f"count {data.shape[0]}"
            )
        if had_existing_mask:
            old_mask = np.frombuffer(
                level_group.read_bytes(full_name, "present_mask"),
                dtype=np.uint8,
            )
        elif existing is not None:
            # First-time mask introduction — backfill existing rows.
            old_mask = np.ones(existing.shape[0], dtype=np.uint8)
        else:
            old_mask = np.empty(0, dtype=np.uint8)
        combined_mask = np.concatenate([old_mask, new_mask])
        level_group.write_bytes(
            full_name, "present_mask", combined_mask.tobytes(),
        )

    level_group.write_array_meta(full_name, {
        "zv_array": "object_attribute",
        "name": attr_name,
        "dtype": str(combined.dtype),
        "shape": list(combined.shape),
        "has_present_mask": has_mask,
    })


def read_object_attribute_present_mask(
    level_group: FsGroup,
    attr_name: str,
) -> npt.NDArray[np.uint8] | None:
    """Read the optional ``present_mask`` byte sidecar for an attribute.

    Returns ``None`` when the level was written without a mask (every
    row real) or the array is missing.
    """
    full_name = f"{OBJECT_ATTRIBUTES}/{attr_name}"
    try:
        meta = level_group.read_array_meta(full_name)
    except Exception:
        return None
    if not meta.get("has_present_mask"):
        return None
    if not level_group.chunk_exists(full_name, "present_mask"):
        return None
    raw = level_group.read_bytes(full_name, "present_mask")
    return np.frombuffer(raw, dtype=np.uint8)


def write_groupings(
    level_group: FsGroup,
    groups: dict[int, list[int]],
) -> None:
    """Write group memberships: group_id → list of object_ids.

    Args:
        level_group: Resolution level group.
        groups: ``{group_id: [object_id, ...], ...}``.
            Group IDs must be contiguous starting from 0.
    """
    if not groups:
        return

    max_gid = max(groups.keys())
    group_list: list[npt.NDArray] = []
    for gid in range(max_gid + 1):
        members = groups.get(gid, [])
        group_list.append(np.array(members, dtype=np.int64))

    raw_bytes, offsets = encode_ragged_ints(group_list, dtype=np.dtype(np.int64))
    level_group.write_bytes(GROUPS, "data", raw_bytes)
    level_group.write_bytes(GROUPS, "offsets", offsets.tobytes())
    level_group.write_array_meta(GROUPS, {
        "zv_array": "groups",
        "num_groups": max_gid + 1,
    })


def write_groupings_attributes(
    level_group: FsGroup,
    attr_name: str,
    data: npt.NDArray,
) -> None:
    """Write dense G×C groupings attribute data.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name.
        data: ``(G,)`` or ``(G, C)`` array.
    """
    full_name = f"{GROUP_ATTRIBUTES}/{attr_name}"
    _ensure_array_dir(level_group, full_name)
    level_group.write_bytes(full_name, "data", data.tobytes())
    level_group.write_array_meta(full_name, {
        "zv_array": "groupings_attribute",
        "name": attr_name,
        "dtype": str(data.dtype),
        "shape": list(data.shape),
    })


def write_cross_chunk_links(
    level_group: FsGroup,
    links: list[list[tuple[ChunkCoords, int]]] | list[CrossChunkLink],
    sid_ndim: int,
    *,
    delta: int = 0,
    link_width: int | None = None,
    mode: Literal["replace", "append"] = "replace",
) -> int:
    """Write cross-chunk link records under ``cross_chunk_links/<delta>/``.

    Each record is ``link_width`` ``(chunk_coords, vertex_idx)``
    endpoints.  ``link_width=2`` (the default) encodes the classic
    cross-chunk edge ``((chunk_A, vi_A), (chunk_B, vi_B))``;
    ``link_width=3`` encodes a triangle face spanning chunks;
    ``link_width=1`` encodes a single parent→child reference used by
    pyramid metanode drill-down.

    **v0.8 partitioned layout:** records are filed into K-deep leaves
    keyed by the sorted unique set of chunks each record touches
    (1 ≤ K ≤ link_width).  Each leaf stores ``L * uint8 ci`` + ``L *
    int64 vi`` per record (``9 * link_width`` bytes per record); chunk
    coords come from the leaf path's K sorted segments, not the
    payload.  For ``delta=0, link_width=2`` (undirected edges) records
    are canonicalized so ``ci = [0, 1]`` — both orientations of an
    edge collapse into one.

    Records may be passed either as legacy 2-tuples (compatibility
    with the pre-0.6.0 edge-only API) or as a list of endpoint lists
    when ``link_width`` is supplied explicitly.

    Endpoint 0 is at the owning resolution level; endpoint k (k>0)
    is at ``this_level + delta``.

    Args:
        level_group: Resolution level group.
        links: List of records; each record is a list of
            ``(chunk_coords, vertex_idx)`` tuples of length
            ``link_width``.  Legacy 2-tuple form is accepted when
            ``link_width`` is 2 (or omitted).
        sid_ndim: Number of spatial index dimensions.
        delta: Level delta; see :mod:`zarr_vectors.core.paths`.
        link_width: Endpoints per record.  Defaults to 2 (or to the
            arity of the first record if it's a list).
        mode: ``"replace"`` (default) overwrites every leaf under
            ``cross_chunk_links/<delta>/``.  ``"append"`` reads every
            existing leaf, concatenates ``links``, writes back.
            ``link_width`` of the appended records must match the
            existing ``link_width``.

    Returns:
        ``"replace"`` returns ``0``.  ``"append"`` returns the
        pre-append total record count (i.e. the index of the first
        newly-appended record had records been concatenated in
        canonical-leaf-walk order).  This matches the legacy
        single-blob API's return semantic for source compatibility,
        though the actual on-disk layout no longer has a single linear
        row index.

    Concurrency:
        ``mode="append"`` is read-modify-write across every leaf and
        NOT cross-writer-safe.  Callers must serialise concurrent
        appends to the same ``cross_chunk_links/<delta>/`` group.
    """
    if mode not in ("replace", "append"):
        raise ArrayError(
            f"mode must be 'replace' or 'append', got {mode!r}"
        )
    if not links:
        return 0

    # Normalise input to a list-of-lists shape; resolve link_width.
    normalised: list[list[tuple[ChunkCoords, int]]] = []
    for rec in links:
        if (
            isinstance(rec, tuple)
            and len(rec) == 2
            and isinstance(rec[0], tuple)
            and not isinstance(rec[0][0], tuple)
        ):
            # Legacy CrossChunkLink: ((chunk_a, vi_a), (chunk_b, vi_b))
            normalised.append([rec[0], rec[1]])
        else:
            normalised.append(list(rec))

    if link_width is None:
        link_width = len(normalised[0])
    for rec in normalised:
        if len(rec) != link_width:
            raise ArrayError(
                f"cross_chunk_links/{format_delta(delta)}: record arity "
                f"{len(rec)} != link_width {link_width}"
            )
        for chunk, _vi in rec:
            if len(chunk) != sid_ndim:
                raise ArrayError(
                    f"chunk coords arity mismatch in cross_chunk_links/"
                    f"{format_delta(delta)}: sid_ndim={sid_ndim}, "
                    f"got len(chunk)={len(chunk)}"
                )

    if mode == "append":
        existing = read_cross_chunk_links(level_group, delta=delta)
        if existing:
            existing_meta = level_group.read_array_meta(
                cross_chunk_links_path(delta),
            )
            existing_link_width = int(existing_meta.get("link_width", 2))
            if existing_link_width != link_width:
                raise ArrayError(
                    f"cross_chunk_links/{format_delta(delta)}: cannot "
                    f"append records of link_width {link_width} onto "
                    f"existing array of link_width {existing_link_width}"
                )
        first_new = len(existing)
        combined = [list(rec) for rec in existing] + normalised
    else:
        first_new = 0
        # Replace mode: clear every kN sub-array (writers may end up
        # with fewer K-buckets than the previous write).
        _clear_kN_arrays(level_group, delta=delta)
        combined = normalised

    # Apply canonicalization for delta=0, link_width=2 undirected
    # edges: endpoint 0 must be at the lex-smaller chunk.
    if delta == 0 and link_width == 2:
        combined = [_canonicalize_l2_delta0(rec) for rec in combined]

    # Re-stamp the group meta with sid_ndim, link_width, layout,
    # shard_shape.  Drop num_links + leaf_index; per-K arrays carry
    # their own meta.
    full_name = cross_chunk_links_path(delta)
    _ensure_array_dir(level_group, full_name)
    level_group.write_array_meta(full_name, {
        "zv_array": "cross_chunk_links",
        "sid_ndim": int(sid_ndim),
        "level_delta": int(delta),
        "link_width": int(link_width),
    })

    # Partition records by K = number of distinct chunks → group by
    # sorted-unique-chunks key → per-cell ci/vi payload.  The kN array
    # shape is derived purely from the actual record extent (no bounds
    # lookup) so the writer works on stores that haven't fully
    # initialized root metadata yet.
    cells_by_K: dict[int, dict[tuple[ChunkCoords, ...], tuple[list[list[int]], list[list[int]]]]] = {}
    max_seen = [0] * sid_ndim
    min_seen = [0] * sid_ndim
    saw_any = False
    for rec in combined:
        record_chunks: list[ChunkCoords] = [
            tuple(int(c) for c in ep[0]) for ep in rec
        ]
        unique_chunks: list[ChunkCoords] = []
        for ch in record_chunks:
            if ch not in unique_chunks:
                unique_chunks.append(ch)
        unique_chunks.sort()
        sorted_unique = tuple(unique_chunks)
        K = len(sorted_unique)
        ci_row = [sorted_unique.index(ch) for ch in record_chunks]
        vi_row = [int(ep[1]) for ep in rec]
        by_K = cells_by_K.setdefault(K, {})
        bucket = by_K.get(sorted_unique)
        if bucket is None:
            bucket = ([], [])
            by_K[sorted_unique] = bucket
        bucket[0].append(ci_row)
        bucket[1].append(vi_row)
        for ch in sorted_unique:
            for a in range(sid_ndim):
                if not saw_any:
                    min_seen[a] = ch[a]
                    max_seen[a] = ch[a]
                else:
                    if ch[a] < min_seen[a]:
                        min_seen[a] = ch[a]
                    if ch[a] > max_seen[a]:
                        max_seen[a] = ch[a]
            saw_any = True

    # Chunk grids may include negative coords (stores with negative
    # min_corner).  Pick an origin per axis equal to min(0, min_seen)
    # so cell indices are always non-negative for zarr; readers add
    # the offset back.
    chunk_origin = tuple(min(0, min_seen[a]) for a in range(sid_ndim))
    chunk_grid_shape = tuple(
        max(1, max_seen[a] - chunk_origin[a] + 1) for a in range(sid_ndim)
    )

    # For each K-bucket, lazily create kN array + batch-write cells
    # by shard so multiple cells in one shard are a single
    # read-modify-write.
    for K, cells in cells_by_K.items():
        arr = _open_or_create_kN_array(
            level_group,
            delta=delta,
            K=K,
            sid_ndim=sid_ndim,
            link_width=link_width,
            chunk_grid_shape=chunk_grid_shape,
            chunk_origin=chunk_origin,
        )
        encoded: dict[tuple[ChunkCoords, ...], bytes] = {}
        for sorted_chunks, (ci_rows, vi_rows) in cells.items():
            encoded[sorted_chunks] = _encode_cell_payload(
                ci_rows, vi_rows, link_width=link_width,
            )
        _write_cells_batched(arr, encoded, chunk_origin=chunk_origin)

    return first_new


def _canonicalize_l2_delta0(
    rec: list[tuple[ChunkCoords, int]],
) -> list[tuple[ChunkCoords, int]]:
    """Normalize a ``delta=0, link_width=2`` undirected edge so endpoint 0
    is at the lex-smaller chunk.  Same-chunk records are left as-is.
    """
    (ca, va), (cb, vb) = rec
    ca_t = tuple(int(x) for x in ca)
    cb_t = tuple(int(x) for x in cb)
    if ca_t <= cb_t:
        return [(ca_t, int(va)), (cb_t, int(vb))]
    return [(cb_t, int(vb)), (ca_t, int(va))]


def create_cross_chunk_link_kN_arrays(
    level_group: FsGroup,
    *,
    sid_ndim: int,
    link_width: int,
    chunk_grid_shape: tuple[int, ...],
    chunk_origin: tuple[int, ...],
    max_K: int | None = None,
) -> None:
    """Pre-create the ``kN`` sharded cross-chunk-link arrays (``k1..kK``).

    Decentralized writers (:func:`write_cross_chunk_link_cells`) must NOT race to
    create the same array, and every writer must agree on the array's shape +
    ``chunk_origin``.  A single coordinator calls this once with the **level-wide**
    ``chunk_grid_shape`` / ``chunk_origin`` (from the target level's chunk bounds);
    workers then only *write cells* into the pre-created arrays.
    """
    create_cross_chunk_links_array(
        level_group, delta=0, link_width=link_width, sid_ndim=sid_ndim,
    )
    for K in range(1, (max_K or link_width) + 1):
        _open_or_create_kN_array(
            level_group, delta=0, K=K, sid_ndim=sid_ndim, link_width=link_width,
            chunk_grid_shape=chunk_grid_shape, chunk_origin=chunk_origin,
        )


def write_cross_chunk_link_cells(
    level_group: FsGroup,
    links: list[list[tuple[ChunkCoords, int]]],
    *,
    sid_ndim: int,
    chunk_grid_shape: tuple[int, ...],
    chunk_origin: tuple[int, ...],
    delta: int = 0,
    link_width: int = 2,
) -> int:
    """Write a batch of cross-chunk-link records' cells, leaving other cells intact.

    Unlike :func:`write_cross_chunk_links` (whole-level replace/append), this writes
    ONLY the cells the given ``links`` fall into — so independent workers can each
    write their own records concurrently **provided their cells lie in disjoint
    outer shards** (``_write_cells_batched`` does a per-shard read-modify-write).
    The ``kN`` arrays must already exist with a matching ``chunk_grid_shape`` /
    ``chunk_origin`` (see :func:`create_cross_chunk_link_kN_arrays`).  Returns the
    number of records written.
    """
    records = [list(r) for r in links]
    if not records:
        return 0
    if delta == 0 and link_width == 2:
        records = [_canonicalize_l2_delta0(rec) for rec in records]
    cells_by_K: dict[int, dict[tuple[ChunkCoords, ...], tuple[list, list]]] = {}
    for rec in records:
        record_chunks = [tuple(int(c) for c in ep[0]) for ep in rec]
        uniq: list[ChunkCoords] = []
        for ch in record_chunks:
            if ch not in uniq:
                uniq.append(ch)
        uniq.sort()
        sorted_unique = tuple(uniq)
        K = len(sorted_unique)
        ci_row = [sorted_unique.index(ch) for ch in record_chunks]
        vi_row = [int(ep[1]) for ep in rec]
        bucket = cells_by_K.setdefault(K, {}).setdefault(sorted_unique, ([], []))
        bucket[0].append(ci_row)
        bucket[1].append(vi_row)
    for K, cells in cells_by_K.items():
        arr = _open_or_create_kN_array(
            level_group, delta=delta, K=K, sid_ndim=sid_ndim,
            link_width=link_width, chunk_grid_shape=chunk_grid_shape,
            chunk_origin=chunk_origin,
        )
        encoded = {
            sc: _encode_cell_payload(ci, vi, link_width=link_width)
            for sc, (ci, vi) in cells.items()
        }
        _write_cells_batched(arr, encoded, chunk_origin=chunk_origin)
    return len(records)


# Default shard-shape axis for kN sharded vlen-bytes arrays.  Each
# shard holds shard_size^(sid_ndim*K) cells.  Tuned for concurrent
# writers touching different spatial regions — different writers
# typically hit different shard files; same-region edits do
# read-modify-write at the shard level (cheap for typical record
# counts but not cross-writer-safe).
CROSS_CHUNK_LINK_SHARD_AXIS = 4


def _level_chunk_grid_shape(level_group: FsGroup) -> tuple[int, ...]:
    """Return the per-axis chunk-grid extent for the level group's parent
    store, computed from root bounds + effective chunk_shape.

    Used by the v0.8 sharded CCL writer to size each ``kN`` array's
    chunk grid: shape = ``chunk_grid_shape * K``.  Inherits the
    per-level ``chunk_shape`` override (v0.7) when present.
    """
    level_zg = level_group.zarr_group
    # Navigate to the root zarr Group via the underlying store.  zarr 3.x
    # doesn't expose a ``.parent`` on Group, so we open the same store at
    # path "" to fetch the root.
    root_zg = zarr.open_group(store=level_zg.store_path.store, path="")
    root_meta = RootMetadata.from_dict(dict(root_zg.attrs))
    try:
        level_meta = LevelMetadata.from_dict(dict(level_zg.attrs))
    except Exception:
        level_meta = None
    chunk_shape = get_level_chunk_shape(root_meta, level_meta)
    min_corner, max_corner = root_meta.bounds
    extents = [
        float(max_corner[i]) - float(min_corner[i])
        for i in range(len(chunk_shape))
    ]
    return tuple(
        max(1, int(math.ceil(extents[i] / chunk_shape[i])))
        for i in range(len(chunk_shape))
    )


def _kN_array_path(delta: int, K: int) -> str:
    """On-disk path for the ``cross_chunk_links/<delta>/k{K}`` sub-array."""
    return f"{cross_chunk_links_path(delta)}/k{K}"


def _kN_attr_array_path(name: str, delta: int, K: int) -> str:
    """Path for the parallel attribute sub-array
    ``cross_chunk_link_attributes/<name>/<delta>/k{K}``.
    """
    return f"{cross_chunk_link_attributes_path(name, delta)}/k{K}"


def _clear_kN_arrays(level_group: FsGroup, *, delta: int) -> None:
    """Delete every ``kN`` sub-array under ``cross_chunk_links/<delta>/``."""
    parent = cross_chunk_links_path(delta)
    if not level_group.array_exists(parent):
        return
    for sub in level_group.list_subgroups(parent):
        if sub.startswith("k"):
            level_group.delete_subtree(f"{parent}/{sub}")


def _open_or_create_kN_array(
    level_group: FsGroup,
    *,
    delta: int,
    K: int,
    sid_ndim: int,
    link_width: int,
    chunk_grid_shape: tuple[int, ...],
    chunk_origin: tuple[int, ...],
):
    """Lazily create (or open) the ``kN`` sharded vlen-bytes Array.

    Shape is ``chunk_grid_shape * K`` — each axis-group represents one
    of the K sorted-unique chunks the leaf records touch.  Inner chunks
    are ``(1,) * (sid_ndim * K)``; outer shards pack
    ``CROSS_CHUNK_LINK_SHARD_AXIS``-wide blocks per axis.

    ``chunk_origin`` is the per-axis offset applied to record chunk
    coords when computing cell index: cell index = ``chunk - origin``.
    Stored on the array meta so readers can invert.
    """
    ndim = sid_ndim * K
    zg = level_group.zarr_group
    parent_path = cross_chunk_links_path(delta)
    parent_group = zg.require_group(parent_path)
    child_name = f"k{K}"
    if child_name in parent_group:
        node = parent_group[child_name]
        try:
            shape_ok = (
                tuple(int(s) for s in node.shape) == tuple(chunk_grid_shape) * K
            )
            existing_origin = tuple(
                int(x) for x in node.attrs.get("chunk_origin", (0,) * sid_ndim)
            )
            origin_ok = existing_origin == tuple(chunk_origin)
        except AttributeError:
            shape_ok = origin_ok = False
        if shape_ok and origin_ok:
            return node
        del parent_group[child_name]

    shape = tuple(chunk_grid_shape) * K
    chunks = (1,) * ndim
    shards = tuple(
        min(CROSS_CHUNK_LINK_SHARD_AXIS, shape[i]) for i in range(ndim)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnstableSpecificationWarning)
        arr = parent_group.create_array(
            child_name,
            shape=shape,
            chunks=chunks,
            shards=shards,
            dtype="bytes",
            serializer=VLenBytesCodec(),
        )
        arr.attrs.update({
            "zv_array": "cross_chunk_links_k",
            "level_delta": int(delta),
            "K": int(K),
            "sid_ndim": int(sid_ndim),
            "link_width": int(link_width),
            "shard_shape": list(shards),
            "chunk_origin": list(chunk_origin),
        })
    return arr


def _encode_cell_payload(
    ci_rows: list[list[int]],
    vi_rows: list[list[int]],
    *,
    link_width: int,
) -> bytes:
    """Encode N records into one cell payload.

    Layout per record:
        ``L * uint8 ci`` (L bytes) + ``L * int64 vi`` (8L bytes) =
        ``9 * L`` bytes per record.  Records concatenated.
    """
    n = len(ci_rows)
    if n == 0:
        return b""
    ci_arr = np.asarray(ci_rows, dtype=np.uint8)
    vi_arr = np.asarray(vi_rows, dtype=np.int64)
    vi_bytes = vi_arr.astype("<i8", copy=False).view(np.uint8).reshape(
        n, 8 * link_width,
    )
    return np.concatenate([ci_arr, vi_bytes], axis=1).tobytes()


def _write_cells_batched(
    arr,
    encoded: dict[tuple[ChunkCoords, ...], bytes],
    *,
    chunk_origin: tuple[int, ...],
) -> None:
    """Write multiple cells of a kN sharded vlen-bytes Array, batching
    by shard so cells in one shard are a single read-modify-write.

    ``encoded`` maps sorted-chunks tuples to payload bytes.
    """
    if not encoded:
        return
    ndim = arr.ndim
    shard_shape = tuple(int(s) for s in (arr.shards or arr.chunks))
    # Per-cell absolute cell index + payload, grouped by shard origin.
    by_shard: dict[
        tuple[int, ...], list[tuple[tuple[int, ...], bytes]]
    ] = {}
    for sorted_chunks, payload in encoded.items():
        # Compute the flat cell index.
        cell_index: list[int] = []
        for ch in sorted_chunks:
            for a, c in enumerate(ch):
                cell_index.append(int(c) - chunk_origin[a])
        cell_t = tuple(cell_index)
        if any(c < 0 or c >= arr.shape[i] for i, c in enumerate(cell_t)):
            raise ArrayError(
                f"cell write: cell index {cell_t} out of array shape "
                f"{arr.shape} (chunk_origin={chunk_origin})"
            )
        shard_origin = tuple(
            (cell_t[i] // shard_shape[i]) * shard_shape[i]
            for i in range(ndim)
        )
        by_shard.setdefault(shard_origin, []).append((cell_t, payload))

    for shard_origin, items in by_shard.items():
        # Build a (shard_shape) object slab populated with empty bytes,
        # then fill in our cells.  We need to preserve cells from this
        # shard that already have data (read-modify-write semantics) —
        # so we read the existing slab first.
        slab_slices = tuple(
            slice(shard_origin[i], shard_origin[i] + shard_shape[i])
            for i in range(ndim)
        )
        existing = arr[slab_slices]
        # Ensure object dtype for vlen assignment.
        if existing.dtype != object:
            slab = np.empty(shard_shape, dtype=object)
            for idx in np.ndindex(*shard_shape):
                v = existing[idx]
                slab[idx] = bytes(v) if v else b""
        else:
            slab = existing.copy()
        for cell_t, payload in items:
            local = tuple(cell_t[i] - shard_origin[i] for i in range(ndim))
            slab[local] = payload
        arr[slab_slices] = slab


def _write_one_cell(
    arr,
    sorted_chunks: tuple[ChunkCoords, ...],
    payload: bytes,
    *,
    chunk_origin: tuple[int, ...] | None = None,
) -> None:
    """Assign one cell of a kN sharded vlen-bytes Array.

    ``sorted_chunks`` is the K-tuple of chunk-coord tuples; cell index
    is the flat concatenation of those K coords minus ``chunk_origin``
    (offset that the array meta declares).  When ``chunk_origin`` is
    ``None``, it's read from ``arr.attrs.chunk_origin`` (default all-0).
    """
    K = len(sorted_chunks)
    ndim = arr.ndim
    sid_ndim = ndim // K
    if chunk_origin is None:
        chunk_origin = tuple(
            int(x) for x in arr.attrs.get("chunk_origin", (0,) * sid_ndim)
        )
    cell_index: list[int] = []
    for ch in sorted_chunks:
        if len(ch) != sid_ndim:
            raise ArrayError(
                f"cell write: chunk arity {len(ch)} != sid_ndim {sid_ndim}"
            )
        cell_index.extend(int(c) - chunk_origin[a] for a, c in enumerate(ch))
    for i, c in enumerate(cell_index):
        if c < 0 or c >= arr.shape[i]:
            raise ArrayError(
                f"cell write: cell index {cell_index} dim {i} out of "
                f"range [0, {arr.shape[i]}) (chunk_origin={chunk_origin})"
            )
    slab_idx = tuple(slice(c, c + 1) for c in cell_index)
    obj = np.empty((1,) * ndim, dtype=object)
    obj[(0,) * ndim] = payload
    arr[slab_idx] = obj


def write_cross_chunk_link_attributes(
    level_group: FsGroup,
    attr_name: str,
    attr_data: npt.NDArray,
    *,
    num_links: int,
    delta: int = 0,
    mode: Literal["replace", "append"] = "replace",
) -> None:
    """Write per-edge attribute data parallel to ``cross_chunk_links/<delta>/``.

    **v0.8 partitioned layout:** attribute rows are partitioned into
    K-deep leaves under
    ``cross_chunk_link_attributes/<name>/<delta>/<chunk_sorted_0>/.../<chunk_sorted_{K-1}>/data``,
    in lockstep with the link leaves at
    ``cross_chunk_links/<delta>/<same path>/data``.  Each attribute
    leaf has one row per link record in the matching link leaf.  The
    writer reads the existing link leaves in canonical (sorted-chunks)
    lex order to know how many rows go into each attribute leaf, and
    slices ``attr_data`` accordingly.

    Length is runtime-checked: the post-write total row count must
    equal ``num_links`` (the total record count returned by
    :func:`write_cross_chunk_links`).  A desynchronized write fails
    loudly instead of producing silent corruption.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name.
        attr_data: ``(num_links,)`` or ``(num_links, C)`` array.  In
            ``mode="append"`` this is interpreted as the NEW rows to
            append; the post-append total must equal ``num_links``.
        num_links: Expected post-write total row count.
        delta: Level delta; see :mod:`zarr_vectors.core.paths`.
        mode: ``"replace"`` (default) overwrites every attribute leaf.
            ``"append"`` reads existing rows in canonical order,
            concatenates ``attr_data``, then redistributes across
            leaves.  Tail dimensions must match the existing array.

    Raises:
        ArrayError: If ``mode`` is invalid; if the post-write length
            does not equal ``num_links``; if the appended row shape
            does not match the existing array; or if the canonical
            walk of link leaves shows row counts that don't sum to
            ``num_links`` (i.e. the link table the attribute attaches
            to was modified mid-write).

    Concurrency:
        ``mode="append"`` is read-modify-write and NOT cross-writer-safe.
        Callers must serialise concurrent appends to the same attribute.
    """
    if mode not in ("replace", "append"):
        raise ArrayError(
            f"mode must be 'replace' or 'append', got {mode!r}"
        )

    full_name = cross_chunk_link_attributes_path(attr_name, delta)
    new_arr = np.asarray(attr_data)

    if mode == "append":
        meta = level_group.read_array_meta(full_name)
        # kN attribute sub-nodes are zarr Arrays, so list_chunks (which
        # returns Array children) is the right primitive — list_subgroups
        # would always return [].
        existing_present = (
            bool(meta) and level_group.array_exists(full_name)
            and any(
                child.startswith("k")
                for child in level_group.list_chunks(full_name)
            )
        )
        if existing_present:
            existing = read_cross_chunk_link_attributes(
                level_group, attr_name, delta=delta,
            )
            if existing.shape[1:] != new_arr.shape[1:]:
                raise ArrayError(
                    f"cross_chunk_link_attributes[{attr_name}] append "
                    f"shape mismatch: existing {existing.shape} vs new "
                    f"{new_arr.shape} — tail dimensions must match"
                )
            new_cast = new_arr.astype(existing.dtype, copy=False)
            combined = np.concatenate([existing, new_cast], axis=0)
        else:
            combined = new_arr
    else:
        combined = new_arr

    if combined.shape[0] != num_links:
        raise ArrayError(
            f"cross_chunk_link_attributes[{attr_name}] row count "
            f"{combined.shape[0]} != num_links {num_links} "
            f"(delta={format_delta(delta)})"
        )

    # Look up the parallel link array's per-cell record counts so we can
    # slice the flat attribute array.
    links_name = cross_chunk_links_path(delta)
    if not level_group.array_exists(links_name):
        raise ArrayError(
            f"cross_chunk_link_attributes[{attr_name}] (delta={format_delta(delta)}): "
            f"no parallel link array at {links_name} — write the link "
            f"records via write_cross_chunk_links() first"
        )
    link_meta = level_group.read_array_meta(links_name)
    if "link_width" not in link_meta:
        raise ArrayError(
            f"cross_chunk_link_attributes[{attr_name}] (delta={format_delta(delta)}): "
            f"parallel link array {links_name} has no link_width meta"
        )
    _check_not_legacy_ccl_blob(level_group, full_name=links_name)
    link_width = int(link_meta["link_width"])
    sid_ndim = int(link_meta.get("sid_ndim", 0))

    # Collect per-cell record counts across all kN link arrays in the
    # canonical (K, lex(sorted-chunks)) walk order that
    # read_cross_chunk_links uses.
    per_cell_plan: list[tuple[int, tuple[ChunkCoords, ...], int]] = []
    # Each entry: (K, sorted_chunks_tuple, n_records).
    for K, link_arr in _list_kN_arrays(
        level_group, delta=delta, link_width=link_width,
    ):
        shard_shape = tuple(int(s) for s in (link_arr.shards or link_arr.chunks))
        cells: list[tuple[tuple[int, ...], bytes]] = []
        for shard_coord in _walk_populated_shards(link_arr):
            shard_origin = tuple(
                shard_coord[i] * shard_shape[i] for i in range(link_arr.ndim)
            )
            cells.extend(_iter_populated_cells_in_shard(
                link_arr, shard_origin, shard_shape,
            ))
        cells.sort(key=lambda p: p[0])
        for cell_idx, payload in cells:
            sorted_chunks = tuple(
                tuple(int(x) for x in cell_idx[k * sid_ndim : (k + 1) * sid_ndim])
                for k in range(K)
            )
            n_records = len(payload) // (9 * link_width)
            per_cell_plan.append((K, sorted_chunks, n_records))

    total_link_records = sum(n for _, _, n in per_cell_plan)
    if total_link_records != num_links:
        raise ArrayError(
            f"cross_chunk_link_attributes[{attr_name}] (delta={format_delta(delta)}): "
            f"parallel link array has {total_link_records} records but "
            f"caller passed num_links={num_links}"
        )

    # Clear stale attribute kN arrays under replace mode.
    if mode == "replace":
        parent_path = full_name
        if level_group.array_exists(parent_path):
            for sub in level_group.list_subgroups(parent_path):
                if sub.startswith("k"):
                    level_group.delete_subtree(f"{parent_path}/{sub}")

    # Stamp the parent group meta.
    _ensure_array_dir(level_group, full_name)
    level_group.write_array_meta(full_name, {
        "zv_array": "cross_chunk_link_attribute",
        "name": attr_name,
        "dtype": str(combined.dtype),
        "level_delta": int(delta),
        "link_width": link_width,
        "sid_ndim": sid_ndim,
        "shape": list(combined.shape),
    })

    # Slice the flat attribute array and write into per-K attribute
    # arrays, mirroring the link arrays' shapes (incl. chunk_origin).
    combined_c = np.ascontiguousarray(combined)
    cursor = 0
    by_K: dict[int, list[tuple[tuple[ChunkCoords, ...], bytes]]] = {}
    for K, sorted_chunks, n_records in per_cell_plan:
        slab = combined_c[cursor : cursor + n_records]
        cursor += n_records
        payload = slab.tobytes()
        by_K.setdefault(K, []).append((sorted_chunks, payload))
    # Open each parallel link kN array to inherit its shape +
    # chunk_origin (so the attr-kN array has matching cell indices).
    parent_link_group = level_group.zarr_group[cross_chunk_links_path(delta)]
    for K, cells in by_K.items():
        link_arr = parent_link_group[f"k{K}"]
        link_shape = tuple(int(s) for s in link_arr.shape)
        link_grid_shape = link_shape[:sid_ndim]
        link_origin = tuple(
            int(x) for x in link_arr.attrs.get("chunk_origin", (0,) * sid_ndim)
        )
        attr_arr = _open_or_create_kN_attr_array(
            level_group,
            attr_name=attr_name,
            delta=delta,
            K=K,
            sid_ndim=sid_ndim,
            link_width=link_width,
            chunk_grid_shape=link_grid_shape,
            chunk_origin=link_origin,
        )
        encoded = {sc: payload for sc, payload in cells}
        _write_cells_batched(attr_arr, encoded, chunk_origin=link_origin)


def _open_or_create_kN_attr_array(
    level_group: FsGroup,
    *,
    attr_name: str,
    delta: int,
    K: int,
    sid_ndim: int,
    link_width: int,
    chunk_grid_shape: tuple[int, ...],
    chunk_origin: tuple[int, ...] = (),
):
    """Lazily create the parallel attribute ``kN`` sharded vlen-bytes Array
    under ``cross_chunk_link_attributes/<name>/<delta>/``.
    """
    ndim = sid_ndim * K
    parent_path = cross_chunk_link_attributes_path(attr_name, delta)
    zg = level_group.zarr_group
    parent_group = zg.require_group(parent_path)
    child_name = f"k{K}"
    shape = tuple(chunk_grid_shape) * K
    if child_name in parent_group:
        node = parent_group[child_name]
        try:
            shape_ok = tuple(int(s) for s in node.shape) == shape
        except AttributeError:
            shape_ok = False
        if shape_ok:
            return node
        del parent_group[child_name]

    chunks = (1,) * ndim
    shards = tuple(
        min(CROSS_CHUNK_LINK_SHARD_AXIS, shape[i]) for i in range(ndim)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnstableSpecificationWarning)
        arr = parent_group.create_array(
            child_name,
            shape=shape,
            chunks=chunks,
            shards=shards,
            dtype="bytes",
            serializer=VLenBytesCodec(),
        )
        attrs_meta = {
            "zv_array": "cross_chunk_link_attribute_k",
            "name": attr_name,
            "level_delta": int(delta),
            "K": int(K),
            "sid_ndim": int(sid_ndim),
            "link_width": int(link_width),
            "shard_shape": list(shards),
        }
        if chunk_origin:
            attrs_meta["chunk_origin"] = list(chunk_origin)
        arr.attrs.update(attrs_meta)
    return arr


# ===================================================================
# Reading data
# ===================================================================

def read_chunk_vertices(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
    dtype: np.dtype | str = np.float32,
    ndim: int = 3,
) -> list[npt.NDArray[np.floating]]:
    """Read all fragments from a spatial chunk.

    Handles both range and explicit fragments: the full chunk buffer is read
    once and dispatched per fragment based on the ``vertex_fragments/<chunk>``
    index — range fragments are returned as contiguous slices, explicit
    fragments via ``fi.indices(f)`` gather.

    Args:
        level_group: Resolution level group.
        chunk_coords: Spatial chunk coordinates.
        dtype: Numpy dtype.
        ndim: Number of coordinate dimensions (D).

    Returns:
        List of arrays, each ``(N_k, D)``.

    Raises:
        ArrayError: If the chunk does not exist or data is malformed.
    """
    key = _chunk_key(chunk_coords)
    dtype = np.dtype(dtype)

    with _maybe_batched_reads(level_group, [
        (VERTICES, [key]),
        (VERTEX_FRAGMENTS, [key]),
    ]):
        try:
            raw = level_group.read_bytes(VERTICES, key)
        except Exception as e:
            raise ArrayError(f"Cannot read vertices chunk {key}: {e}") from e

        fi = read_vertex_fragment_index(level_group, chunk_coords)

    if fi.num_fragments == 0:
        return []
    full = _reshape_vertex_buffer(raw, dtype, ndim)
    groups: list[npt.NDArray[np.floating]] = []
    for f in range(fi.num_fragments):
        if fi.is_range(f):
            start, count = fi.range(f)
            groups.append(full[start : start + count])
        else:
            groups.append(full[fi.indices(f)])
    return groups


def read_fragment(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
    fragment_index: int,
    dtype: np.dtype | str = np.float32,
    ndim: int = 3,
    *,
    default: Any = _UNSET,
) -> npt.NDArray[np.floating] | Any:
    """Read a single vertex fragment from a chunk.

    Dispatches on the ``vertex_fragments/<chunk>`` index: range fragments
    use a byte-slice fast path; explicit fragments reshape the full chunk
    buffer and gather rows via ``fi.indices(fragment_index)``.

    Args:
        level_group: Resolution level group.
        chunk_coords: Spatial chunk coordinates.
        fragment_index: Index of the fragment within the chunk.
        dtype: Numpy dtype.
        ndim: Number of coordinate dimensions.
        default: When supplied, returned on read failure (missing chunk
            or out-of-range ``fragment_index``) instead of raising.  Pass
            ``None`` for the common "soft-fail with None" pattern.  Only
            :class:`ArrayError` and :class:`StoreError` are caught;
            programming errors propagate.

    Returns:
        Array of shape ``(N, D)`` (or ``(N,)`` when ``ndim == 1``), or
        ``default`` when supplied and the read fails.
    """
    key = _chunk_key(chunk_coords)
    dtype = np.dtype(dtype)

    try:
        with _maybe_batched_reads(level_group, [
            (VERTICES, [key]),
            (VERTEX_FRAGMENTS, [key]),
        ]):
            raw = level_group.read_bytes(VERTICES, key)
            fi = read_vertex_fragment_index(level_group, chunk_coords)

        if fragment_index < 0 or fragment_index >= fi.num_fragments:
            raise ArrayError(
                f"Fragment index {fragment_index} out of range "
                f"(chunk {key} has {fi.num_fragments} groups)"
            )

        if fi.is_range(fragment_index):
            start, count = fi.range(fragment_index)
            return _slice_vertex_range(raw, start, count, dtype, ndim)

        full = _reshape_vertex_buffer(raw, dtype, ndim)
        return full[fi.indices(fragment_index)]
    except (ArrayError, StoreError):
        if default is _UNSET:
            raise
        return default


def read_chunk_links(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
    dtype: np.dtype | str = np.int64,
    link_width: int | None = None,
    *,
    delta: int = 0,
) -> list[npt.NDArray[np.integer]]:
    """Read all link groups from a spatial chunk's ``links/<delta>/`` array.

    Args:
        level_group: Resolution level group.
        chunk_coords: Spatial chunk coordinates.
        dtype: Integer dtype.
        link_width: Number of columns per link (L). If None, read from
            array metadata.
        delta: Level delta; ``0`` is the intra-level array.

    Returns:
        List of arrays, each ``(M_k, L)``.
    """
    key = _chunk_key(chunk_coords)
    dtype = np.dtype(dtype)
    full_name = links_path(delta)

    if link_width is None:
        meta = level_group.read_array_meta(full_name)
        link_width = meta.get("link_width", 2)

    # delta == 0 needs both the link bytes and the fragment-index sibling;
    # delta != 0 keeps the v0.5 inline self-describing layout (one blob).
    plan: list[tuple[str, list[str]]] = [(full_name, [key])]
    if delta == 0:
        plan.append((LINK_FRAGMENTS, [key]))

    with _maybe_batched_reads(level_group, plan):
        try:
            raw = level_group.read_bytes(full_name, key)
        except Exception as e:
            raise ArrayError(
                f"Cannot read links chunk {key} (delta={format_delta(delta)}): {e}"
            ) from e

        if delta == 0:
            # v0.6 intra-level layout: raw is the flat concatenated link
            # data; per-group row counts live in link_fragments/<chunk>.
            # Handles both range and explicit fragments by reshaping once
            # and dispatching per fragment.
            fi = read_link_fragment_index(level_group, chunk_coords)
            if fi.num_fragments == 0:
                return []
            full = _reshape_link_buffer(raw, dtype, link_width)
            groups: list[npt.NDArray[np.integer]] = []
            for f in range(fi.num_fragments):
                if fi.is_range(f):
                    start, count = fi.range(f)
                    groups.append(full[start : start + count])
                else:
                    groups.append(full[fi.indices(f)])
            return groups

        # Cross-level delta != 0: v0.5 inline self-describing layout.
        return decode_ragged_blob(raw, dtype, ncols=link_width)


def read_chunk_link_fragment(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
    fragment_index: int,
    dtype: np.dtype | str = np.int64,
    link_width: int | None = None,
    *,
    default: Any = _UNSET,
) -> npt.NDArray[np.integer] | Any:
    """Read a single link fragment from a chunk's ``links/0/<chunk>`` array.

    Intra-level only (``delta == 0``). Dispatches on the
    ``link_fragments/<chunk>`` index: range fragments use a byte-slice
    fast path; explicit fragments reshape the full chunk buffer and gather
    rows via ``fi.indices(fragment_index)``.

    Args:
        level_group: Resolution level group.
        chunk_coords: Spatial chunk coordinates.
        fragment_index: Index of the fragment within the chunk.
        dtype: Integer dtype.
        link_width: Number of columns per link (L). If None, read from
            array metadata.
        default: When supplied, returned on read failure (missing chunk
            or out-of-range ``fragment_index``) instead of raising.  Pass
            ``None`` for the common "soft-fail with None" pattern.  Only
            :class:`ArrayError` and :class:`StoreError` are caught;
            programming errors propagate.

    Returns:
        Array of shape ``(M, L)`` (or ``(M,)`` when ``link_width == 1``),
        or ``default`` when supplied and the read fails.

    Raises:
        ArrayError: If the chunk does not exist or ``fragment_index`` is
            out of range, AND ``default`` was not supplied.
    """
    key = _chunk_key(chunk_coords)
    dtype = np.dtype(dtype)
    full_name = links_path(0)

    try:
        if link_width is None:
            meta = level_group.read_array_meta(full_name)
            link_width = meta.get("link_width", 2)

        with _maybe_batched_reads(level_group, [
            (full_name, [key]),
            (LINK_FRAGMENTS, [key]),
        ]):
            try:
                raw = level_group.read_bytes(full_name, key)
            except Exception as e:
                raise ArrayError(
                    f"Cannot read links chunk {key} (delta=0): {e}"
                ) from e
            fi = read_link_fragment_index(level_group, chunk_coords)

        if fragment_index < 0 or fragment_index >= fi.num_fragments:
            raise ArrayError(
                f"Fragment index {fragment_index} out of range "
                f"(chunk {key} has {fi.num_fragments} groups)"
            )

        if fi.is_range(fragment_index):
            start, count = fi.range(fragment_index)
            return _slice_link_range(raw, start, count, dtype, link_width)

        full = _reshape_link_buffer(raw, dtype, link_width)
        return full[fi.indices(fragment_index)]
    except (ArrayError, StoreError):
        if default is _UNSET:
            raise
        return default


def read_chunk_attributes(
    level_group: FsGroup,
    attr_name: str,
    chunk_coords: ChunkCoords,
    dtype: np.dtype | str = np.float32,
    ncols: int = 1,
    *,
    vert_dtype: np.dtype | str | None = None,
    vert_ndim: int | None = None,
) -> list[npt.NDArray]:
    """Read vertex attribute data for a chunk.

    Per-group byte offsets are derived from ``vertex_fragments``:
    group ``k`` has ``n_k = (vert_offsets[k+1] - vert_offsets[k]) /
    (vert_dtype.itemsize * vert_ndim)`` vertices, so its attribute
    byte offset is ``cumsum(n_k) * dtype.itemsize * ncols``.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name.
        chunk_coords: Spatial chunk coordinates.
        dtype: Numpy dtype of the attribute.
        ncols: Number of columns (channels). Use 1 for scalars.
        vert_dtype: Vertex dtype (needed to derive per-group sizes).
            When ``None`` (default) it is read from the ``vertices/``
            array metadata.
        vert_ndim: Vertex coordinate dimensionality.  When ``None``
            (default) it is read from root metadata via NGFF axes; on
            failure falls back to 3.

    Returns:
        List of arrays aligned with fragments.
    """
    key = _chunk_key(chunk_coords)
    dtype = np.dtype(dtype)
    full_name = f"{VERTEX_ATTRIBUTES}/{attr_name}"

    if vert_dtype is None:
        try:
            vmeta = level_group.read_array_meta(VERTICES)
            vert_dtype = np.dtype(vmeta.get("dtype", "float32"))
        except Exception:
            vert_dtype = np.dtype(np.float32)
    else:
        vert_dtype = np.dtype(vert_dtype)
    if vert_ndim is None:
        vert_ndim = _infer_vert_ndim(level_group)

    with _maybe_batched_reads(level_group, [
        (full_name, [key]),
        (VERTEX_FRAGMENTS, [key]),
    ]):
        try:
            raw = level_group.read_bytes(full_name, key)
        except Exception as e:
            raise ArrayError(
                f"Cannot read attribute '{attr_name}' chunk {key}: {e}"
            ) from e

        attr_offsets = _derive_attribute_offsets(
            level_group, chunk_coords,
            vert_dtype=vert_dtype, vert_ndim=vert_ndim,
            attr_dtype=dtype, attr_ncols=ncols,
            total_attr_bytes=len(raw),
        )
    return decode_ragged_floats(raw, attr_offsets, dtype, ncols)


def read_chunk_fragment_attributes(
    level_group: FsGroup,
    attr_name: str,
    chunk_coords: ChunkCoords,
    dtype: np.dtype | str = np.float32,
    ncols: int = 1,
    *,
    default: Any = _UNSET,
) -> npt.NDArray | Any:
    """Read per-fragment attribute data for a spatial chunk.

    The on-disk layout is a dense per-chunk byte blob; ``F`` (number of
    fragments) is derived from the byte length and the row stride
    (``dtype.itemsize * ncols``).  No round-trip to ``vertex_fragments``
    is needed.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name.
        chunk_coords: Spatial chunk coordinates.
        dtype: Numpy dtype of the attribute.
        ncols: Number of columns (channels).  Use 1 for scalars.
        default: When supplied, returned on read failure (missing chunk,
            byte-length mismatch) instead of raising.  Pass ``None`` for
            the common "soft-fail with None" pattern.  Only
            :class:`ArrayError` and :class:`StoreError` are caught;
            programming errors propagate.

    Returns:
        Array of shape ``(F,)`` (when ``ncols == 1``) or ``(F, ncols)``
        (when ``ncols > 1``).  Empty 1-D array when the blob is empty.
    """
    key = _chunk_key(chunk_coords)
    dtype = np.dtype(dtype)
    full_name = f"{FRAGMENT_ATTRIBUTES}/{attr_name}"
    row_bytes = dtype.itemsize * ncols

    try:
        try:
            raw = level_group.read_bytes(full_name, key)
        except Exception as e:
            raise ArrayError(
                f"Cannot read fragment_attribute '{attr_name}' chunk {key}: {e}"
            ) from e

        if row_bytes <= 0 or len(raw) % row_bytes != 0:
            raise ArrayError(
                f"fragment_attribute '{attr_name}' chunk {key}: byte length "
                f"{len(raw)} is not a multiple of row stride "
                f"{row_bytes} (dtype={dtype}, ncols={ncols})"
            )

        flat = np.frombuffer(raw, dtype=dtype)
        if ncols == 1:
            return flat.copy()
        return flat.reshape(-1, ncols).copy()
    except (ArrayError, StoreError):
        if default is _UNSET:
            raise
        return default


def read_chunk_link_attributes(
    level_group: FsGroup,
    attr_name: str,
    chunk_coords: ChunkCoords,
    dtype: np.dtype | str = np.float32,
    ncols: int = 1,
    *,
    delta: int = 0,
) -> list[npt.NDArray]:
    """Read per-link attribute data for a chunk.

    Mirrors :func:`read_chunk_attributes` for the per-link case: the
    ragged bytes live under ``link_attributes/<name>/<delta>/<chunk>``
    and align 1:1 with the link fragments under ``link_fragments/<chunk>``
    (intra-level only, ``delta == 0``).  Per-link group ``k`` has the
    same row count as link group ``k`` in ``links/<delta>/<chunk>``.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name (e.g. ``"weight"``).
        chunk_coords: Spatial chunk coordinates.
        dtype: Numpy dtype of the attribute.
        ncols: Number of columns (channels).  Use 1 for scalars.
        delta: Level delta; cross-level link attributes are stored
            differently and must be read via the global
            ``cross_chunk_link_attributes`` path — this helper handles
            only the per-chunk ``delta == 0`` case.

    Returns:
        List of arrays aligned with the link fragments in the chunk.
    """
    if delta != 0:
        raise ArrayError(
            f"read_chunk_link_attributes only supports delta=0 "
            f"(per-chunk intra-level); got delta={delta}.  Use "
            f"read_cross_chunk_link_attributes for cross-level "
            f"link attributes.",
        )
    key = _chunk_key(chunk_coords)
    dtype = np.dtype(dtype)
    full_name = link_attributes_path(attr_name, delta)

    try:
        raw = level_group.read_bytes(full_name, key)
    except Exception as e:
        raise ArrayError(
            f"Cannot read link attribute '{attr_name}' chunk {key} "
            f"(delta={format_delta(delta)}): {e}"
        ) from e

    # Per-link group `k` has N_k rows where N_k = link_fragments[k].count.
    # Derive byte offsets from the link-fragment sidecar.
    fi = read_link_fragment_index(level_group, chunk_coords)
    if fi.num_fragments == 0:
        return []
    row_bytes = int(dtype.itemsize) * int(ncols)
    cursor = 0
    out: list[npt.NDArray] = []
    for f in range(fi.num_fragments):
        if not fi.is_range(f):
            raise ArrayError(
                f"link_fragments/{key} fragment {f} is non-contiguous; "
                "read_chunk_link_attributes requires every fragment to be "
                "a contiguous range of link rows.",
            )
        _start, count = fi.range(f)
        seg = raw[cursor : cursor + int(count) * row_bytes]
        cursor += int(count) * row_bytes
        arr = np.frombuffer(seg, dtype=dtype)
        if ncols > 1:
            arr = arr.reshape(-1, ncols)
        out.append(arr.copy())
    return out


def _infer_vert_ndim(level_group: FsGroup) -> int:
    """Best-effort lookup of the spatial-index dimensionality.

    Reads NGFF ``multiscales[0].axes`` length from root attrs.  Falls
    back to 3 when unavailable.
    """
    try:
        # Level groups don't carry root attrs; walk up to root via the
        # backend.  Most levels have an ``_backend`` handle that owns
        # the root path.
        from zarr_vectors.core.group import Group
        root_handle = Group._from_backend(level_group._backend, "")
        ms = root_handle.attrs.to_dict().get("multiscales") or []
        if ms and isinstance(ms, list):
            axes = ms[0].get("axes") or []
            if axes:
                return len(axes)
    except Exception:
        pass
    return 3


def _derive_attribute_offsets(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
    *,
    vert_dtype: np.dtype,
    vert_ndim: int,
    attr_dtype: np.dtype,
    attr_ncols: int,
    total_attr_bytes: int,
) -> npt.NDArray[np.int64]:
    """Compute per-group attribute byte offsets from vertex offsets.

    Attribute groups align 1:1 with fragments.  The k-th vertex
    group spans ``vert_offsets[k+1] - vert_offsets[k]`` bytes of
    vertex data, which corresponds to ``n_k`` vertices (and therefore
    ``n_k`` attribute rows).
    """
    vert_row_size = vert_dtype.itemsize * vert_ndim
    if vert_row_size <= 0:
        return np.empty(0, dtype=np.int64)
    fi = read_vertex_fragment_index(level_group, chunk_coords)
    if fi.num_fragments == 0:
        return np.empty(0, dtype=np.int64)
    # Per-fragment vertex row count.  Today's writers always emit
    # range fragments; non-contiguous shapes will need a richer
    # attribute-alignment story (out of scope for this change).
    n_per_group = np.empty(fi.num_fragments, dtype=np.int64)
    for f in range(fi.num_fragments):
        if not fi.is_range(f):
            raise ArrayError(
                f"vertex_fragments fragment {f} is non-contiguous; "
                "attribute alignment requires every fragment to be a "
                "contiguous range of vertex rows.",
            )
        _start, count = fi.range(f)
        n_per_group[f] = int(count)
    attr_row_size = attr_dtype.itemsize * attr_ncols
    attr_byte_lengths = n_per_group * int(attr_row_size)
    attr_offsets = np.empty_like(attr_byte_lengths)
    attr_offsets[0] = 0
    np.cumsum(attr_byte_lengths[:-1], out=attr_offsets[1:])
    del total_attr_bytes  # signature retained for caller compat
    return attr_offsets


def read_object_manifest(
    level_group: FsGroup,
    object_id: int,
) -> ObjectManifest:
    """Read the ordered fragment reference list for one object.

    Args:
        level_group: Resolution level group.
        object_id: Object ID.

    Returns:
        List of ``(chunk_coords, fragment_index)`` tuples.
    """
    meta = level_group.read_array_meta(OBJECT_INDEX)
    sid_ndim = meta["sid_ndim"]
    num_objects = meta["num_objects"]

    if object_id < 0 or object_id >= num_objects:
        raise ArrayError(
            f"Object ID {object_id} out of range [0, {num_objects})"
        )

    if meta.get("layout") == OBJECT_INDEX_LAYOUT_V1:
        manifests_arr = level_group.zarr_group[OBJECT_INDEX]["manifests"]
        # Slice (then index) instead of scalar indexing: zarr 3.x vlen-bytes
        # returns a 0-d object ndarray under ``arr[i]``, whose ``bytes()``
        # is the array header — not the payload.  ``arr[i:i+1][0]`` is the
        # actual bytes object and still fetches only the chunk holding i.
        blob = manifests_arr[object_id:object_id + 1][0]
    else:
        blob = _legacy_read_object_blob(level_group, object_id, num_objects)

    blocks = decode_object_manifest_blocks(blob, sid_ndim=sid_ndim)
    return _expand_blocks(blocks)


def read_all_object_manifests(
    level_group: FsGroup,
) -> list[ObjectManifest]:
    """Read all object manifests at once.

    Returns:
        List indexed by object_id, each a list of ``(chunk_coords, fragment_index)``.
    """
    meta = level_group.read_array_meta(OBJECT_INDEX)
    sid_ndim = meta["sid_ndim"]
    num_objects = int(meta.get("num_objects", 0))

    if meta.get("layout") == OBJECT_INDEX_LAYOUT_V1:
        if num_objects == 0:
            return []
        manifests_arr = level_group.zarr_group[OBJECT_INDEX]["manifests"]
        # Slicing yields a 1-D object ndarray whose elements are bytes
        # directly (unlike scalar indexing — see read_object_manifest).
        blobs = manifests_arr[:]
        return [
            _expand_blocks(decode_object_manifest_blocks(b, sid_ndim=sid_ndim))
            for b in blobs
        ]

    # Legacy layout: single-chunk data + offsets byte blobs.
    with _maybe_batched_reads(level_group, [
        (OBJECT_INDEX, ["data", "offsets"]),
    ]):
        raw = level_group.read_bytes(OBJECT_INDEX, "data")
        offsets = np.frombuffer(
            level_group.read_bytes(OBJECT_INDEX, "offsets"),
            dtype=np.int64,
        )
    return [
        _expand_blocks(
            decode_object_manifest_blocks(
                _slice_legacy_blob(raw, offsets, i, num_objects),
                sid_ndim=sid_ndim,
            ),
        )
        for i in range(num_objects)
    ]


def _legacy_read_object_blob(
    level_group: FsGroup,
    object_id: int,
    num_objects: int,
) -> bytes:
    """Load one object's encoded manifest blob from the legacy
    ``object_index/{data,offsets}`` byte-blob layout.

    Reads the full ``data`` and ``offsets`` arrays (each a single-chunk
    blob) and slices to the one object's byte range.  This is the cost
    the vlen-bytes ``manifests`` layout was introduced to eliminate;
    kept for backwards-compatible reads of pre-vlen stores.
    """
    with _maybe_batched_reads(level_group, [
        (OBJECT_INDEX, ["data", "offsets"]),
    ]):
        raw = level_group.read_bytes(OBJECT_INDEX, "data")
        offsets = np.frombuffer(
            level_group.read_bytes(OBJECT_INDEX, "offsets"),
            dtype=np.int64,
        )
    return _slice_legacy_blob(raw, offsets, object_id, num_objects)


def _slice_legacy_blob(
    data: bytes,
    offsets: npt.NDArray[np.int64],
    object_id: int,
    num_objects: int,
) -> bytes:
    start = int(offsets[object_id])
    end = (
        int(offsets[object_id + 1])
        if object_id + 1 < num_objects
        else len(data)
    )
    return data[start:end]


def _expand_blocks(
    blocks: list[tuple[ChunkCoords, Any]],
) -> ObjectManifest:
    """Expand v0.6 manifest blocks to the legacy
    ``[(chunk_coords, fragment_index), ...]`` tuple list.

    Mode-1 (range) and mode-2 (explicit list) blocks expand to one
    tuple per fragment so existing call sites that iterate
    ``(chunk_coords, fragment_index)`` keep working unchanged.  Callers that
    want the raw block representation can use
    :func:`zarr_vectors.encoding.fragments.decode_object_manifest_blocks`
    directly.
    """
    out: ObjectManifest = []
    for chunk_coords, frag_ref in blocks:
        if isinstance(frag_ref, int):
            out.append((chunk_coords, int(frag_ref)))
        elif isinstance(frag_ref, tuple):
            r_start, r_count = frag_ref
            for k in range(int(r_count)):
                out.append((chunk_coords, int(r_start) + k))
        else:
            # np.ndarray of explicit indices
            for idx in frag_ref:
                out.append((chunk_coords, int(idx)))
    return out


def read_object_vertices(
    level_group: FsGroup,
    object_id: int,
    dtype: np.dtype | str = np.float32,
    ndim: int = 3,
) -> list[npt.NDArray[np.floating]]:
    """Read all vertex data for an object by following its manifest.

    Args:
        level_group: Resolution level group.
        object_id: Object ID.
        dtype: Numpy dtype.
        ndim: Number of coordinate dimensions.

    Returns:
        List of fragment arrays in reconstruction order.
    """
    manifest = read_object_manifest(level_group, object_id)
    if not manifest:
        return []

    # Prefetch every chunk this object touches in one async gather, so
    # the per-fragment read_fragment calls below hit the cache
    # instead of paying one round-trip per fragment.  Distinct chunks
    # appear once in the plan; multiple fragments inside the same chunk
    # share the same cache entry.
    chunk_keys = sorted({_chunk_key(cc) for cc, _ in manifest})
    with _maybe_batched_reads(level_group, [
        (VERTICES, chunk_keys),
        (VERTEX_FRAGMENTS, chunk_keys),
    ]):
        groups: list[npt.NDArray] = []
        for chunk_coords, fragment_index in manifest:
            fragment = read_fragment(
                level_group, chunk_coords, fragment_index,
                dtype=dtype, ndim=ndim,
            )
            groups.append(fragment)
    return groups


def read_object_attributes(
    level_group: FsGroup,
    attr_name: str,
    dtype: np.dtype | str | None = None,
) -> npt.NDArray:
    """Read dense O×C object attribute data.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name.
        dtype: Override dtype. If None, read from metadata.

    Returns:
        Array of shape ``(O,)`` or ``(O, C)``.
    """
    full_name = f"{OBJECT_ATTRIBUTES}/{attr_name}"
    meta = level_group.read_array_meta(full_name)
    if dtype is None:
        dtype = np.dtype(meta["dtype"])
    else:
        dtype = np.dtype(dtype)
    shape = tuple(meta["shape"])

    raw = level_group.read_bytes(full_name, "data")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def read_group_object_ids(
    level_group: FsGroup,
    group_id: int,
) -> list[int]:
    """Read the list of object IDs belonging to a group.

    Args:
        level_group: Resolution level group.
        group_id: Group ID.

    Returns:
        List of object ID integers.
    """
    meta = level_group.read_array_meta(GROUPS)
    num_groups = meta["num_groups"]

    if group_id < 0 or group_id >= num_groups:
        raise ArrayError(
            f"Group ID {group_id} out of range [0, {num_groups})"
        )

    raw = level_group.read_bytes(GROUPS, "data")
    offsets = np.frombuffer(
        level_group.read_bytes(GROUPS, "offsets"),
        dtype=np.int64,
    )

    all_groups = decode_ragged_ints(raw, offsets, dtype=np.dtype(np.int64), ncols=1)
    return all_groups[group_id].tolist()


def read_all_groupings(
    level_group: FsGroup,
) -> list[list[int]]:
    """Read all group memberships.

    Returns:
        List indexed by group_id, each a list of object_id ints.
    """
    meta = level_group.read_array_meta(GROUPS)

    raw = level_group.read_bytes(GROUPS, "data")
    offsets = np.frombuffer(
        level_group.read_bytes(GROUPS, "offsets"),
        dtype=np.int64,
    )

    all_groups = decode_ragged_ints(raw, offsets, dtype=np.dtype(np.int64), ncols=1)
    return [g.tolist() for g in all_groups]


def read_groupings_attributes(
    level_group: FsGroup,
    attr_name: str,
    dtype: np.dtype | str | None = None,
) -> npt.NDArray:
    """Read dense G×C groupings attribute data."""
    full_name = f"{GROUP_ATTRIBUTES}/{attr_name}"
    meta = level_group.read_array_meta(full_name)
    if dtype is None:
        dtype = np.dtype(meta["dtype"])
    else:
        dtype = np.dtype(dtype)
    shape = tuple(meta["shape"])

    raw = level_group.read_bytes(full_name, "data")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def _check_not_legacy_ccl_blob(
    level_group: FsGroup, *, full_name: str,
) -> None:
    """Refuse pre-v0.8 monolithic CCL blobs.

    The v0.8 partitioning scheme stores records in ``kK`` sub-arrays
    under the ``cross_chunk_links/<delta>/`` (or attribute) parent
    group; pre-v0.8 stores instead held a single ``data`` byte blob
    directly under the parent group.  Detect the legacy shape
    structurally — by the presence of that ``data`` zarr Array child
    — and raise with a pointer at the migration helper.

    Codec choice (sharded vs unsharded vlen-bytes) is intentionally
    NOT a check here; that's a writer-side performance/operational
    decision encapsulated by zarr itself.
    """
    try:
        parent_node = level_group.zarr_group[full_name]
    except KeyError:
        return
    if not isinstance(parent_node, zarr.Group):
        return
    if "data" not in parent_node:
        return
    data_child = parent_node["data"]
    if isinstance(data_child, zarr.Array):
        raise ArrayError(
            f"{full_name}: found a legacy monolithic ``data`` blob "
            f"under the parent group; v0.8 readers expect ``kK`` "
            f"sub-arrays instead.  Run "
            f"``zarr_vectors.migration.partition_legacy_cross_chunk_links"
            f"(store_path)`` to convert in place."
        )


def _decode_ccl_cell_payload(
    raw: bytes,
    sorted_chunks: tuple[ChunkCoords, ...],
    *,
    link_width: int,
) -> list[tuple[tuple[ChunkCoords, int], ...]]:
    """Decode one cell's bytes into legacy-shape records.

    Per record: ``L * uint8 ci`` + ``L * int64 vi`` = ``9 * L`` bytes.
    """
    record_size = 9 * link_width
    if len(raw) == 0:
        return []
    if len(raw) % record_size != 0:
        raise ArrayError(
            f"cross_chunk_links cell payload length {len(raw)} not a "
            f"multiple of 9 * link_width = {record_size}"
        )
    n_records = len(raw) // record_size
    out: list[tuple[tuple[ChunkCoords, int], ...]] = []
    for r in range(n_records):
        base = r * record_size
        ci = np.frombuffer(raw[base : base + link_width], dtype=np.uint8)
        vi = np.frombuffer(
            raw[base + link_width : base + record_size], dtype=np.int64,
        )
        endpoints: list[tuple[ChunkCoords, int]] = []
        for j in range(link_width):
            ci_j = int(ci[j])
            if ci_j < 0 or ci_j >= len(sorted_chunks):
                raise ArrayError(
                    f"cross_chunk_links cell record {r} has ci[{j}]="
                    f"{ci_j} out of range [0, {len(sorted_chunks)})"
                )
            endpoints.append((sorted_chunks[ci_j], int(vi[j])))
        out.append(tuple(endpoints))
    return out


def _walk_populated_shards(arr) -> list[tuple[int, ...]]:
    """Return outer-shard coords that have on-disk data for ``arr``.

    Walks the array's underlying zarr Store for ``c/...`` keys whose
    coord-segments exist (sharded layout: one file per populated
    outer shard).  Returns coords in lex order so the read order is
    deterministic.
    """
    try:
        store_path = arr.store_path
    except AttributeError:
        return []
    # The async-array's store iterates keys under the array's own
    # path; filter to chunk-data keys and parse the coord suffix.
    try:
        store = store_path.store
    except AttributeError:
        return []
    prefix = (store_path.path or "").rstrip("/")
    seek = f"{prefix}/c/" if prefix else "c/"

    import asyncio

    async def _gather() -> list[str]:
        keys: list[str] = []
        async for k in store.list_prefix(seek):
            keys.append(k)
        return keys

    try:
        loop = asyncio.new_event_loop()
        try:
            keys = loop.run_until_complete(_gather())
        finally:
            loop.close()
    except Exception:
        return []

    shard_coords: list[tuple[int, ...]] = []
    for k in keys:
        # Strip the prefix; split remaining segments.
        rel = k[len(seek):]
        if not rel:
            continue
        segs = rel.split("/")
        try:
            coord = tuple(int(s) for s in segs)
        except ValueError:
            continue
        shard_coords.append(coord)
    return sorted(set(shard_coords))


def _iter_populated_cells_in_shard(
    arr,
    shard_origin: tuple[int, ...],
    shard_shape: tuple[int, ...],
) -> list[tuple[tuple[int, ...], bytes]]:
    """Read one outer shard slab and yield ``(cell_index, payload)`` for
    every non-empty cell in it.

    The cell_index is the absolute coord (origin + local).
    """
    slab_slices = tuple(
        slice(shard_origin[i], shard_origin[i] + shard_shape[i])
        for i in range(len(shard_shape))
    )
    slab = arr[slab_slices]
    out: list[tuple[tuple[int, ...], bytes]] = []
    for local in np.ndindex(*slab.shape):
        val = slab[local]
        if val is None or len(val) == 0:
            continue
        coord = tuple(int(shard_origin[i] + local[i]) for i in range(len(local)))
        out.append((coord, bytes(val)))
    return out


def _list_kN_arrays(
    level_group: FsGroup,
    *,
    delta: int,
    link_width: int,
) -> list[tuple[int, "zarr.Array"]]:
    """Return ``[(K, kN_array), ...]`` for every ``kN`` sub-array that
    exists under ``cross_chunk_links/<delta>/``.

    K ranges from 1 to ``link_width``.  Missing sub-arrays are skipped
    (lazy allocation — only K-buckets with records exist on disk).
    """
    parent_name = cross_chunk_links_path(delta)
    if not level_group.array_exists(parent_name):
        return []
    out: list[tuple[int, zarr.Array]] = []
    parent_group = level_group.zarr_group[parent_name]
    for K in range(1, link_width + 1):
        child = f"k{K}"
        if child not in parent_group:
            continue
        node = parent_group[child]
        if isinstance(node, zarr.Array):
            out.append((K, node))
    return out


def list_cross_chunk_link_leaves(
    level_group: FsGroup,
    *,
    delta: int = 0,
    involves: ChunkCoords | None = None,
) -> list[tuple[ChunkCoords, ...]]:
    """Enumerate every populated cell across all ``kN`` arrays under
    ``cross_chunk_links/<delta>/``.

    Returns each cell's K-tuple of chunk-coord tuples (the sorted
    unique chunks the record touches), in (K, lex(chunks)) order.
    ``involves``, if given, filters to cells containing that chunk
    anywhere — useful for "all records touching chunk X" neighbourhood
    queries.
    """
    parent_name = cross_chunk_links_path(delta)
    if not level_group.array_exists(parent_name):
        return []
    _check_not_legacy_ccl_blob(level_group, full_name=parent_name)
    meta = level_group.read_array_meta(parent_name)
    if not meta or "link_width" not in meta:
        return []
    sid_ndim = int(meta.get("sid_ndim", 0))
    link_width = int(meta["link_width"])
    target = tuple(int(c) for c in involves) if involves is not None else None

    out: list[tuple[ChunkCoords, ...]] = []
    for K, arr in _list_kN_arrays(
        level_group, delta=delta, link_width=link_width,
    ):
        shard_shape = tuple(int(s) for s in (arr.shards or arr.chunks))
        origin_attr = tuple(
            int(x) for x in arr.attrs.get("chunk_origin", (0,) * sid_ndim)
        )
        for shard_coord in _walk_populated_shards(arr):
            shard_origin = tuple(
                shard_coord[i] * shard_shape[i] for i in range(arr.ndim)
            )
            for cell_idx, _payload in _iter_populated_cells_in_shard(
                arr, shard_origin, shard_shape,
            ):
                # Split cell_idx into K chunk-coord tuples of arity sid_ndim,
                # adding back the per-axis origin offset.
                chunks_tuple = tuple(
                    tuple(
                        int(cell_idx[k * sid_ndim + a]) + origin_attr[a]
                        for a in range(sid_ndim)
                    )
                    for k in range(K)
                )
                if target is not None and target not in chunks_tuple:
                    continue
                out.append(chunks_tuple)
    return sorted(out, key=lambda t: (len(t), t))


def read_cross_chunk_link_leaf(
    level_group: FsGroup,
    chunks: tuple[ChunkCoords, ...],
    *,
    delta: int = 0,
) -> list[tuple[tuple[ChunkCoords, int], ...]]:
    """Read every record stored in a single cell (kN array lookup).

    ``chunks`` is the K-tuple of chunk-coord tuples the cell represents
    — does NOT need to be pre-sorted; the helper sorts + dedupes
    internally.  Returns ``[]`` when no such cell exists.
    """
    parent_name = cross_chunk_links_path(delta)
    if not level_group.array_exists(parent_name):
        return []
    _check_not_legacy_ccl_blob(level_group, full_name=parent_name)
    meta = level_group.read_array_meta(parent_name)
    if not meta or "link_width" not in meta:
        return []
    link_width = int(meta["link_width"])

    # Dedupe + lex-sort the caller's chunks.
    seen: set[ChunkCoords] = set()
    deduped: list[ChunkCoords] = []
    for ch in chunks:
        ch_t = tuple(int(c) for c in ch)
        if ch_t in seen:
            continue
        seen.add(ch_t)
        deduped.append(ch_t)
    deduped.sort()
    sorted_unique = tuple(deduped)
    K = len(sorted_unique)
    if K == 0:
        return []
    # The kN node is a zarr Array (not Group), so FsGroup.array_exists
    # (which checks "is this a Group?") would return False — go through
    # the underlying zarr Group directly.
    zg = level_group.zarr_group
    if parent_name not in zg:
        return []
    parent_group = zg[parent_name]
    child = f"k{K}"
    if child not in parent_group:
        return []
    arr = parent_group[child]
    if not isinstance(arr, zarr.Array):
        return []
    sid_ndim = int(meta.get("sid_ndim", 0))
    origin_attr = tuple(
        int(x) for x in arr.attrs.get("chunk_origin", (0,) * sid_ndim)
    )
    cell_index = tuple(
        int(c) - origin_attr[a]
        for ch in sorted_unique
        for a, c in enumerate(ch)
    )
    for i, c in enumerate(cell_index):
        if c < 0 or c >= arr.shape[i]:
            return []
    # Slice a 1-cell slab so the result is an object array (not a
    # 0-D scalar) — extract the single payload from local index (0,)*ndim.
    slab = arr[tuple(slice(c, c + 1) for c in cell_index)]
    payload = slab[(0,) * arr.ndim]
    if payload is None or len(payload) == 0:
        return []
    return _decode_ccl_cell_payload(
        bytes(payload), sorted_unique, link_width=link_width,
    )


def read_cross_chunk_links(
    level_group: FsGroup,
    *,
    delta: int = 0,
) -> list[tuple[tuple[ChunkCoords, int], ...]]:
    """Read every cross-chunk-link record under ``cross_chunk_links/<delta>/``.

    **v0.8 sharded layout:** walks each ``kN`` sub-array's populated
    shards, decodes each non-empty cell's records, and concatenates
    them across all K-buckets.  Each record comes back as
    ``((chunk_0, vi_0), (chunk_1, vi_1), …)`` (length ``link_width``);
    endpoint chunks are recovered from the cell index via the
    per-endpoint ``ci`` byte.

    Endpoint 0 lives at the owning resolution level; endpoints k (k>0)
    live at ``this_level + delta``.

    Returns ``[]`` when the ``<delta>`` group is absent or has no
    populated cells.  Raises :class:`ArrayError` for pre-v0.8 layouts
    (run the migration helper).

    Returns:
        List of records.  Order: K ascending; lex over sorted-chunks
        within each K; record-order within each cell.
    """
    parent_name = cross_chunk_links_path(delta)
    if not level_group.array_exists(parent_name):
        return []
    _check_not_legacy_ccl_blob(level_group, full_name=parent_name)
    meta = level_group.read_array_meta(parent_name)
    if not meta or "link_width" not in meta:
        return []
    sid_ndim = int(meta.get("sid_ndim", 0))
    link_width = int(meta["link_width"])

    out: list[tuple[tuple[ChunkCoords, int], ...]] = []
    for K, arr in _list_kN_arrays(
        level_group, delta=delta, link_width=link_width,
    ):
        shard_shape = tuple(int(s) for s in (arr.shards or arr.chunks))
        origin_attr = tuple(
            int(x) for x in arr.attrs.get("chunk_origin", (0,) * sid_ndim)
        )
        # Collect (cell_index, payload) pairs across all populated shards.
        cells: list[tuple[tuple[int, ...], bytes]] = []
        for shard_coord in _walk_populated_shards(arr):
            shard_origin = tuple(
                shard_coord[i] * shard_shape[i] for i in range(arr.ndim)
            )
            cells.extend(_iter_populated_cells_in_shard(
                arr, shard_origin, shard_shape,
            ))
        cells.sort(key=lambda p: p[0])
        for cell_idx, payload in cells:
            sorted_chunks = tuple(
                tuple(
                    int(cell_idx[k * sid_ndim + a]) + origin_attr[a]
                    for a in range(sid_ndim)
                )
                for k in range(K)
            )
            out.extend(_decode_ccl_cell_payload(
                payload, sorted_chunks, link_width=link_width,
            ))
    return out


def read_cross_chunk_link_attributes(
    level_group: FsGroup,
    attr_name: str,
    dtype: np.dtype | str | None = None,
    *,
    delta: int = 0,
) -> npt.NDArray:
    """Read per-link attribute data parallel to ``cross_chunk_links/<delta>/``.

    **v0.8 sharded layout:** walks the attribute kN sub-arrays in the
    same (K, lex(sorted-chunks)) order as :func:`read_cross_chunk_links`
    and concatenates each cell's row block into a single flat
    ``(num_records_total,)`` or ``(num_records_total, C)`` array.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name (e.g. ``"weight"``).
        dtype: Override the on-disk dtype.  ``None`` (default) reads it
            from the array group's ``.zattrs``.
        delta: Level delta.
    """
    parent_name = cross_chunk_link_attributes_path(attr_name, delta)
    fallback_dtype = np.float32 if dtype is None else np.dtype(dtype)
    if not level_group.array_exists(parent_name):
        return np.empty(0, dtype=fallback_dtype)
    _check_not_legacy_ccl_blob(level_group, full_name=parent_name)
    meta = level_group.read_array_meta(parent_name)
    if not meta:
        return np.empty(0, dtype=fallback_dtype)
    if dtype is None:
        dtype = np.dtype(meta["dtype"])
    else:
        dtype = np.dtype(dtype)
    sid_ndim = int(meta.get("sid_ndim", 0))
    link_width = int(meta.get("link_width", 0))
    if link_width <= 0:
        # Fall back to parallel link-group link_width.
        link_meta = level_group.read_array_meta(cross_chunk_links_path(delta))
        link_width = int(link_meta.get("link_width", 0))

    out_rows: list[npt.NDArray] = []
    parent_group = level_group.zarr_group[parent_name]
    for K in range(1, link_width + 1):
        child = f"k{K}"
        if child not in parent_group:
            continue
        attr_arr = parent_group[child]
        if not isinstance(attr_arr, zarr.Array):
            continue
        shard_shape = tuple(int(s) for s in (attr_arr.shards or attr_arr.chunks))
        cells: list[tuple[tuple[int, ...], bytes]] = []
        for shard_coord in _walk_populated_shards(attr_arr):
            shard_origin = tuple(
                shard_coord[i] * shard_shape[i] for i in range(attr_arr.ndim)
            )
            cells.extend(_iter_populated_cells_in_shard(
                attr_arr, shard_origin, shard_shape,
            ))
        cells.sort(key=lambda p: p[0])
        for _idx, payload in cells:
            out_rows.append(np.frombuffer(payload, dtype=dtype))

    if not out_rows:
        return np.empty(0, dtype=dtype)
    flat = np.concatenate(out_rows, axis=0).copy()
    shape_meta = meta.get("shape")
    if shape_meta is not None and len(shape_meta) == 2:
        channels = int(shape_meta[1])
        if flat.size % channels != 0:
            raise ArrayError(
                f"cross_chunk_link_attributes[{attr_name}] (delta="
                f"{format_delta(delta)}): flat size {flat.size} not "
                f"divisible by channels {channels}"
            )
        flat = flat.reshape(-1, channels)
    return flat


# ===================================================================
# Listing / introspection
# ===================================================================

def list_chunk_keys(
    level_group: FsGroup,
    array_name: str = VERTICES,
) -> list[ChunkCoords]:
    """List all chunk coordinates that have data for an array.

    Args:
        level_group: Resolution level group.
        array_name: Array name (default: ``"vertices"``).

    Returns:
        Sorted list of chunk coordinate tuples.
    """
    keys = level_group.list_chunks(array_name)
    coords: list[ChunkCoords] = []
    for k in keys:
        try:
            coords.append(_parse_chunk_key(k))
        except ValueError:
            continue  # skip non-chunk files (e.g. .zattrs)
    return sorted(coords)


def _list_deltas_under(level_group: FsGroup, group_path: str) -> list[int]:
    """List signed level-delta segments present under a group path.

    Returns the sorted list of integers parsed from immediate child
    names that look like delta segments (``"0"``, ``"+N"``, ``"-N"``).
    Returns an empty list when the parent group is absent.  Used by the
    public ``list_link_deltas`` / ``list_cross_link_deltas`` helpers
    (and indirectly by readers and validators that walk the multiscale
    link layout).
    """
    if not level_group.array_exists(group_path):
        return []
    try:
        sub = level_group[group_path]
    except Exception:
        return []
    deltas: list[int] = []
    for name in sub:
        try:
            deltas.append(parse_delta(name))
        except ValueError:
            continue
    return sorted(deltas)


def list_link_deltas(level_group: FsGroup) -> list[int]:
    """Sorted list of ``<delta>`` values present under ``links/`` in a level."""
    return _list_deltas_under(level_group, LINKS)


def list_cross_link_deltas(level_group: FsGroup) -> list[int]:
    """Sorted list of ``<delta>`` values present under ``cross_chunk_links/``."""
    return _list_deltas_under(level_group, CROSS_CHUNK_LINKS)


def list_link_attribute_deltas(level_group: FsGroup, name: str) -> list[int]:
    """Sorted list of ``<delta>`` values present under ``link_attributes/<name>/``."""
    return _list_deltas_under(level_group, f"{LINK_ATTRIBUTES}/{name}")


def list_cross_chunk_link_attribute_deltas(
    level_group: FsGroup, name: str,
) -> list[int]:
    """Sorted list of ``<delta>`` values under ``cross_chunk_link_attributes/<name>/``."""
    return _list_deltas_under(level_group, f"{CROSS_CHUNK_LINK_ATTRIBUTES}/{name}")


def resolve_chunk_keys(
    level_group: FsGroup,
    chunk_shape: tuple[float, ...],
    *,
    bbox: tuple[npt.NDArray, npt.NDArray] | None = None,
    chunks: list[ChunkCoords] | None = None,
    array_name: str = VERTICES,
) -> list[ChunkCoords]:
    """Resolve the chunk_keys present in a level, intersected with the
    bbox-implied set and the explicit ``chunks`` whitelist.

    Combination of filters is AND: a chunk must be physically present
    *and* satisfy every supplied constraint.

    Args:
        level_group: Resolution level group.
        chunk_shape: Physical chunk size per spatial dimension.
        bbox: Optional ``(min_corner, max_corner)``. Intersected with the
            stored keys via :func:`chunks_intersecting_bbox`.
        chunks: Optional explicit whitelist of chunk coordinate tuples.
            Pass ``[]`` for "no chunks" (yields an empty result). Pass
            ``None`` (the default) for "no filter".
        array_name: Array whose chunk keys to enumerate.

    Returns:
        Sorted list of chunk coordinate tuples.

    Raises:
        ValueError: If a tuple in ``chunks`` has the wrong arity for
            this store.
    """
    from zarr_vectors.spatial.chunking import chunks_intersecting_bbox

    present = list_chunk_keys(level_group, array_name=array_name)
    keys: set[ChunkCoords] = set(present)

    if bbox is not None:
        target = set(chunks_intersecting_bbox(
            np.asarray(bbox[0]), np.asarray(bbox[1]), tuple(chunk_shape),
        ))
        keys &= target

    if chunks is not None:
        expected_arity = len(chunk_shape)
        normalised: set[ChunkCoords] = set()
        for c in chunks:
            t = tuple(int(x) for x in c)
            if len(t) != expected_arity:
                # Some stores (e.g. attribute-binned points / graphs) prefix
                # spatial chunk coords with an extra binning axis, giving
                # keys of length ``expected_arity + 1``. Accept those too.
                if len(t) != expected_arity + 1:
                    raise ValueError(
                        f"chunks tuple {c!r} has arity {len(t)}; "
                        f"expected {expected_arity} (or {expected_arity + 1} "
                        f"for attribute-binned stores)"
                    )
            normalised.add(t)
        keys &= normalised

    return sorted(keys)


def count_fragments(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
) -> int:
    """Count fragments in a chunk by reading the fragment-index header."""
    return len(read_vertex_fragment_index(level_group, chunk_coords))


# ===================================================================
# Internal helpers
# ===================================================================

def _read_modify_write_blob(
    level_group: FsGroup,
    full_name: str,
    key: str,
    *,
    decode_fn,
    merge_fn,
    encode_fn,
    initial,
) -> tuple[Any, Any]:
    """Read-modify-write a single blob with caller-supplied codec hooks.

    Returns ``(combined, existing)`` — callers need ``existing`` to compute
    the row-index of the newly-appended portion.

    Atomicity: ``write_bytes`` is delete-then-create, not strictly atomic;
    treat it as last-writer-wins.  The pattern is NOT cross-writer-safe.
    Callers MUST serialise concurrent writes to the same blob.
    """
    try:
        raw = level_group.read_bytes(full_name, key)
        existing = decode_fn(raw)
    except StoreError:
        existing = initial
    combined = merge_fn(existing)
    level_group.write_bytes(full_name, key, encode_fn(combined))
    return combined, existing


def read_vertex_fragment_index(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
) -> ChunkFragmentIndex:
    """Read and decode the ``vertex_fragments/<chunk>`` blob.

    Returns the v0.6 :class:`ChunkFragmentIndex` view describing how rows of
    ``vertices/<chunk>`` partition into fragments.
    """
    key = _chunk_key(chunk_coords)
    raw = level_group.read_bytes(VERTEX_FRAGMENTS, key)
    return decode_fragments(raw)


def read_link_fragment_index(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
) -> ChunkFragmentIndex:
    """Read and decode the ``link_fragments/<chunk>`` blob (delta=0 only)."""
    key = _chunk_key(chunk_coords)
    raw = level_group.read_bytes(LINK_FRAGMENTS, key)
    return decode_fragments(raw)


def _reshape_link_buffer(
    raw: bytes,
    dtype: np.dtype,
    link_width: int,
) -> npt.NDArray[np.integer]:
    """Return a ``(M_total, link_width)`` view onto a ``links/0/<chunk>`` blob.

    Falls back to a 1-D array when ``link_width == 1``.
    """
    arr = np.frombuffer(raw, dtype=dtype)
    return arr.reshape(-1, link_width) if link_width > 1 else arr


def _slice_link_range(
    raw: bytes,
    start: int,
    count: int,
    dtype: np.dtype,
    link_width: int,
) -> npt.NDArray[np.integer]:
    """Return rows ``[start, start+count)`` from a ``links/0/<chunk>`` blob."""
    row_size = int(dtype.itemsize) * int(link_width)
    seg = raw[int(start) * row_size : (int(start) + int(count)) * row_size]
    arr = np.frombuffer(seg, dtype=dtype)
    return arr.reshape(-1, link_width) if link_width > 1 else arr


def _reshape_vertex_buffer(
    raw: bytes,
    dtype: np.dtype,
    ndim: int,
) -> npt.NDArray[np.floating]:
    """Return a ``(N_total, ndim)`` view onto a ``vertices/<chunk>`` blob.

    Falls back to a 1-D array when ``ndim == 1``.
    """
    arr = np.frombuffer(raw, dtype=dtype)
    return arr.reshape(-1, ndim) if ndim > 1 else arr


def _slice_vertex_range(
    raw: bytes,
    start: int,
    count: int,
    dtype: np.dtype,
    ndim: int,
) -> npt.NDArray[np.floating]:
    """Return rows ``[start, start+count)`` from a ``vertices/<chunk>`` blob."""
    bytes_per_vertex = int(dtype.itemsize) * int(ndim)
    seg = raw[int(start) * bytes_per_vertex : (int(start) + int(count)) * bytes_per_vertex]
    arr = np.frombuffer(seg, dtype=dtype)
    return arr.reshape(-1, ndim) if ndim > 1 else arr


def _read_vertex_offsets(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
    *,
    bytes_per_vertex: int | None = None,
) -> npt.NDArray[np.int64]:
    """Read the ``(K,)`` int64 vertex byte offsets for a chunk.

    Computed from the v0.6 ``vertex_fragments/<chunk>`` index.  Every
    fragment must be a contiguous range over rows of ``vertices/<chunk>``
    — the only shape the existing writer produces.  Stores written by
    future writers that materialise non-contiguous / shared-row
    fragments must use the higher-level fragment-index API directly;
    this helper raises rather than silently lying about a byte offset
    that doesn't exist.

    Args:
        level_group: Resolution level group.
        chunk_coords: Spatial chunk coordinates.
        bytes_per_vertex: Bytes per vertex row.  When omitted it is
            inferred from the ``vertices/`` array's ``dtype`` metadata
            and the root NGFF axes count.
    """
    if bytes_per_vertex is None:
        vmeta = level_group.read_array_meta(VERTICES)
        vdtype = np.dtype(vmeta.get("dtype", "float32"))
        ndim = _infer_vert_ndim(level_group)
        bytes_per_vertex = int(vdtype.itemsize) * int(ndim)
    fi = read_vertex_fragment_index(level_group, chunk_coords)
    if fi.num_fragments == 0:
        return np.empty(0, dtype=np.int64)
    offsets = np.empty(fi.num_fragments, dtype=np.int64)
    for i in range(fi.num_fragments):
        if not fi.is_range(i):
            raise ArrayError(
                f"vertex_fragments/{_chunk_key(chunk_coords)} fragment {i} "
                "is non-contiguous; byte-offset access requires every "
                "fragment to be a contiguous range over rows of "
                "vertices/<chunk>.  Use read_vertex_fragment_index() "
                "directly for non-contiguous fragments.",
            )
        start, _count = fi.range(i)
        offsets[i] = int(start) * int(bytes_per_vertex)
    return offsets


