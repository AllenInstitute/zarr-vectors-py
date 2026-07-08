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

import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, Sequence

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
    format_cell_key,
    format_delta,
    link_attributes_path,
    links_path,
    parse_cell_key,
    parse_delta,
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

    Detects both layouts: the legacy Option-G group-with-chunk-arrays
    (``array_exists``) and the 0.8.1 single-standard-Zarr-v3-array layout
    (``standalone_array_exists``).

    When the caller is running inside
    :meth:`Group.native_sharded_arrays` and the existing node is a
    legacy Option-G group at a per-chunk-array path, *do not*
    short-circuit — the create call needs to replace the legacy group
    with a native-sharded Zarr array.  This matters when the store
    was warmed via :func:`create_store` (which writes an empty
    ``vertices/`` group) and the caller is the first writer to request
    sharded layout.
    """
    exists_as_array = level_group.standalone_array_exists(full_name)
    exists_as_group = (
        level_group.array_exists(full_name) and not exists_as_array
    )
    if not (exists_as_array or exists_as_group):
        return False

    # Legacy group at a per-chunk-array path, inside a sharded
    # writer — fall through so ``_ensure_array_dir`` replaces it.
    if (
        exists_as_group
        and level_group._native_sharded_config is not None
        and _is_per_chunk_array(full_name)
    ):
        return False

    if exist_ok:
        return True
    raise ArrayError(
        f"{full_name!r} already exists; pass exist_ok=True to ignore"
    )


def _default_fill_value_for_dtype(dtype: np.dtype) -> Any:
    """Pick the conventional 'absent' sentinel for a numpy dtype.

    Returns a Python-native value used both as the in-array marker for
    rows excluded by a caller-supplied ``present_mask`` and as the Zarr
    array's persisted ``fill_value`` (so unwritten chunks read back as
    the same value).

    Conventions:
        * float dtypes → ``NaN``
        * signed-int dtypes → the dtype's minimum value
        * unsigned-int dtypes → the dtype's maximum value

    Bool and other dtypes have no in-band sentinel; callers must
    promote (e.g. ``bool`` → ``int8`` with explicit ``fill_value=-1``)
    or pass an explicit ``fill_value=`` to the writer.
    """
    if dtype.kind == "f":
        return float("nan")
    if dtype.kind == "i":
        return int(np.iinfo(dtype).min)
    if dtype.kind == "u":
        return int(np.iinfo(dtype).max)
    if dtype.kind in ("U", "S"):
        # Empty string as the absence marker.  Documented caveat:
        # collides with a legitimate empty value; callers that care
        # must pass ``fill_value=`` explicitly.
        return ""
    raise ArrayError(
        f"no default 'absent' fill_value for dtype {dtype!r}; "
        f"pass fill_value= explicitly (or promote bool → int8)"
    )


def _ensure_array_dir(level_group: FsGroup, array_name: str) -> None:
    """Ensure an array subdirectory exists within a level group.

    Three layouts are produced here depending on context:

    * Inside :meth:`Group.native_sharded_arrays` (per-chunk arrays only)
      — allocate a single multidim vlen-bytes Zarr array at the path
      using the ``sharding_indexed`` codec.  Subsequent ``write_bytes``
      calls land in this array's grid-coord cells.
    * Inside :meth:`Group.batched_writes` — skip the sync
      ``require_group`` round-trip; the metadata flush will PUT the
      parent ``zarr.json`` directly, including the right attributes,
      in the same gather as the chunk PUTs.
    * Otherwise — create an empty Zarr group at the path so subsequent
      ``write_array_meta`` (which only ``attrs.update``s, not create)
      has a target.
    """
    cfg = level_group._native_sharded_config
    if cfg is not None and _is_per_chunk_array(array_name):
        # Idempotent: if the sharded array already exists at this path
        # (e.g. a second writer pass on the same level), leave it alone.
        if not level_group.standalone_array_exists(array_name):
            level_group.create_sharded_chunk_array(
                array_name,
                grid_shape=cfg["grid_shape"],
                shard_shape=cfg["shard_shape"],
            )
        return
    if level_group._pending_array_metas is not None:
        return
    level_group.require_group(array_name)


@contextmanager
def open_write_session(
    level_group: FsGroup,
    *,
    compressor: Any = None,
    shard_shape: int | tuple[int, ...] | None = None,
    bounds: tuple[list[float], list[float]] | None = None,
    chunk_shape: tuple[float, ...] | None = None,
):
    """Open the batched-write context (and native-sharded context when
    ``shard_shape`` is set) used by every type writer.

    ``shard_shape`` accepts an int (broadcast to every axis) or an
    explicit per-axis tuple.  When provided, ``bounds`` and
    ``chunk_shape`` are required so the grid shape can be computed
    upfront — that's the size of the multidim vlen-bytes array each
    per-chunk path is allocated as.

    Args:
        level_group: The resolution-level group.
        compressor: Forwarded to :meth:`Group.batched_writes`.
        shard_shape: When set, activates
            :meth:`Group.native_sharded_arrays` so per-chunk-array
            creations route through the ``sharding_indexed`` codec
            and per-chunk writes land directly in array cells.
        bounds: ``(min_corner, max_corner)`` for the level — used to
            compute the chunk grid extent.
        chunk_shape: Physical chunk size per axis — paired with
            ``bounds`` for the grid extent.
    """
    from contextlib import ExitStack

    stack = ExitStack()
    with stack:
        stack.enter_context(level_group.batched_writes(compressor=compressor))
        if shard_shape is not None:
            if bounds is None or chunk_shape is None:
                raise ArrayError(
                    "shard_shape requires `bounds` and `chunk_shape` so "
                    "the per-axis chunk-grid extent can be computed"
                )
            # Chunk coords are origin-anchored: ``floor(position /
            # chunk_shape)``.  The grid must therefore size to
            # ``ceil(max_corner / chunk_shape)``, *not* the extent
            # ``ceil((max - min) / chunk_shape)`` — bounds that don't
            # start at zero would otherwise yield coords past the end.
            max_corner = np.asarray(bounds[1], dtype=np.float64)
            cs = np.asarray(chunk_shape, dtype=np.float64)
            grid_shape = tuple(
                max(1, int(np.ceil(m / c))) for m, c in zip(max_corner, cs)
            )
            ss = (
                (int(shard_shape),) * len(grid_shape)
                if isinstance(shard_shape, int)
                else tuple(int(x) for x in shard_shape)
            )
            stack.enter_context(
                level_group.native_sharded_arrays(ss, grid_shape)
            )
        yield


def _is_per_chunk_array(name: str) -> bool:
    """Whether ``name`` points to a per-spatial-chunk array container.

    These are the Option-G groups whose children are chunk-key-named
    byte blobs — the arrays that benefit from native sharding because
    each spatial chunk is one tiny storage object without it.  Object-
    level arrays (``object_index``, ``object_attributes/...``, groups,
    ``group_attributes/...``) use the single-array layout already and
    do **not** get rewritten.

    ``cross_chunk_links/<delta>`` is excluded for now — its "cell keys"
    are canonical-sorted endpoint chunk-pair tuples, not single-chunk
    spatial coords, so the grid shape doesn't match the spatial
    chunk grid.  Callers who need those sharded can apply
    :func:`zarr_vectors.sharding.shard_store` post-hoc.
    """
    if name in {VERTICES, VERTEX_FRAGMENTS, LINK_FRAGMENTS}:
        return True
    prefixes = (
        VERTEX_ATTRIBUTES + "/",
        FRAGMENT_ATTRIBUTES + "/",
        LINKS + "/",
        LINK_ATTRIBUTES + "/",
    )
    return any(name.startswith(p) for p in prefixes)


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
    """Reserve the ``object_attributes/<name>/`` slot.

    Under the 0.8.1 layout :func:`write_object_attributes` creates the
    Zarr array on first write (it needs the shape and dtype of the data
    in hand), so this function only performs the ``exist_ok`` conflict
    check — ``dtype`` and ``num_channels`` are accepted for API
    compatibility but not persisted at create-time.

    Args:
        level_group: The resolution level FsGroup.
        name: Attribute name.
        dtype: Numpy dtype string (informational only in 0.8.1).
        num_channels: Number of channels (informational only in 0.8.1).
        exist_ok: When True (default), no-op if the array already exists.
            When False, raise :class:`ArrayError` on conflict.
    """
    del dtype, num_channels  # informational only; see docstring.
    full_name = f"{OBJECT_ATTRIBUTES}/{name}"
    if _short_circuit_existing(level_group, full_name, exist_ok):
        return
    # Reserve the slot as an empty Zarr group; :func:`write_object_attributes`
    # replaces it with the real chunked array on first write.  The
    # placeholder lets a subsequent ``exist_ok=False`` create call detect
    # the conflict.
    _ensure_array_dir(level_group, full_name)


def create_groupings_array(
    level_group: FsGroup,
    *,
    exist_ok: bool = True,
) -> None:
    """Reserve the ``groups/`` slot.

    Under the 0.8.1 layout :func:`write_groupings` creates the vlen
    Zarr array on first write, so this function only performs the
    ``exist_ok`` conflict check.

    Args:
        level_group: Resolution level group.
        exist_ok: When True (default), no-op if the array already exists.
            When False, raise :class:`ArrayError` on conflict.
    """
    if _short_circuit_existing(level_group, GROUPS, exist_ok):
        return
    _ensure_array_dir(level_group, GROUPS)


def create_groupings_attributes_array(
    level_group: FsGroup,
    name: str,
    dtype: str = "float32",
    num_channels: int = 1,
    *,
    exist_ok: bool = True,
) -> None:
    """Reserve the ``group_attributes/<name>/`` slot.

    Under the 0.8.1 layout :func:`write_groupings_attributes` creates
    the Zarr array on first write, so this function only performs the
    ``exist_ok`` conflict check.

    Args:
        level_group: Resolution level group.
        name: Attribute name.
        dtype: Informational only in 0.8.1.
        num_channels: Informational only in 0.8.1.
        exist_ok: When True (default), no-op if the array already exists.
            When False, raise :class:`ArrayError` on conflict.
    """
    del dtype, num_channels
    full_name = f"{GROUP_ATTRIBUTES}/{name}"
    if _short_circuit_existing(level_group, full_name, exist_ok):
        return
    _ensure_array_dir(level_group, full_name)


def create_cross_chunk_links_array(
    level_group: FsGroup,
    *,
    delta: int = 0,
    link_width: int = 2,
    sid_ndim: int | None = None,
    directed: bool = False,
    store: str = "canonical",
    exist_ok: bool = True,
) -> None:
    """Create a ``cross_chunk_links/<delta>/`` array.

    Source-side endpoints live at the owning resolution level;
    target-side endpoints live at ``this_level + delta``.

    The whole-level :func:`write_cross_chunk_links` re-stamps every field
    on write, so the ``directed`` / ``store`` / ``sid_ndim`` args here
    matter mainly for the decentralized per-cell writer flow, where a
    coordinator pre-creates the array and workers only write cells.

    Args:
        level_group: Resolution level group.
        delta: Level delta (0 for intra-level, ±N for cross-level).
        link_width: Number of vertex refs per record.  2 for edges
            (the default — chunk pairs straddling a boundary), 3 for
            triangle faces, 1 for parent→child metanode references.
        sid_ndim: Spatial index dims; stamped into the meta when given
            (per-cell writers and readers need it).
        directed: Endpoint order is meaningful; see
            :func:`write_cross_chunk_links`.
        store: ``"canonical"`` or ``"duplicate"``; see
            :func:`write_cross_chunk_links`.
        exist_ok: When True (default), no-op if the array already exists.
            When False, raise :class:`ArrayError` on conflict.
    """
    if store not in ("canonical", "duplicate"):
        raise ArrayError(
            f"store must be 'canonical' or 'duplicate', got {store!r}"
        )
    full_name = cross_chunk_links_path(delta)
    if _short_circuit_existing(level_group, full_name, exist_ok):
        return
    _ensure_array_dir(level_group, full_name)
    meta = {
        "zv_array": "cross_chunk_links",
        "level_delta": int(delta),
        "link_width": int(link_width),
        "directed": bool(directed),
        "store": str(store),
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
    """Create a ``cross_chunk_link_attributes/<name>/<delta>/`` array.

    Parallel attribute storage for the matching
    ``cross_chunk_links/<delta>/`` array; one value (or one ``C``-vector
    row) per cross-chunk link in path order.

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

    For ``delta=0`` readers derive per-group link byte offsets from the
    cumulative sizes of each group's link bytes (see
    :func:`read_chunk_links`); link groups need not be 1:1 with the
    chunk's vertex fragments.

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

    # NOTE: link groups are no longer required to be 1:1 with the chunk's
    # vertex fragments. BRIDGE stores streamlines as vertex-fragments and the
    # node graph as link-fragments, so a chunk legitimately has many vertex
    # fragments (Core-1 range + polyline twins) but few link groups (the node
    # graph). Readers derive per-group link ranges from ``link_fragments/``
    # (not from ``vertex_fragments``), so no write-time 1:1 guard is needed.

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
        # Ensure the sibling array container exists.  Routes through
        # ``_ensure_array_dir`` so that native-sharded writers allocate
        # a multidim vlen-bytes array at this path instead of the
        # legacy Option-G group.
        if not level_group.array_exists(LINK_FRAGMENTS):
            _ensure_array_dir(level_group, LINK_FRAGMENTS)
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
    fill_value: Any = None,
    mode: Literal["replace", "append"] = "replace",
) -> None:
    """Write dense O×C object attribute data as a single Zarr v3 array.

    The array is stored at ``object_attributes/<attr_name>`` as a standard
    chunked Zarr v3 array.  Absent rows (selected by ``present_mask``)
    are encoded in-band using ``fill_value`` — no sibling
    ``present_mask`` child array is written.  Stock Zarr v3 tooling can
    read the array without library-specific decoding; absent positions
    are visible as the array's ``fill_value`` (also returned for any
    chunk that was never written).

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name.
        data: ``(O,)`` or ``(O, C)`` array.  In ``mode="append"`` this is
            the NEW rows to append.
        present_mask: Optional ``(O,)`` byte array (``0``/``1`` per
            object) marking real rows.  When provided, absent rows
            (``mask[i] == 0``) are overwritten with ``fill_value`` before
            write; callers that already encoded sentinels in ``data``
            should leave this ``None``.
        fill_value: Sentinel for absent positions.  Defaults to NaN for
            floats, dtype-min for signed ints, dtype-max for unsigned
            ints (see :func:`_default_fill_value_for_dtype`).  Used both
            as the in-array sentinel for masked rows and as the array's
            persisted ``fill_value``.
        mode: ``"replace"`` (default) writes ``data`` as the full array.
            ``"append"`` reads the existing array, concatenates ``data``
            along axis 0, writes back.  Existing dtype wins on dtype
            mismatch (new rows are cast).  ``"append"`` against a
            missing attribute behaves like ``"replace"``.

    Raises:
        ArrayError: If ``mode`` is invalid, ``present_mask`` length
            mismatches ``data``, the dtype has no default sentinel and
            no ``fill_value`` is given, or the appended row shape
            (everything beyond axis 0) does not match the existing array.

    Concurrency:
        ``mode="append"`` is read-modify-write and NOT cross-writer-safe.
        Callers must serialise concurrent appends to the same attribute.
    """
    if mode not in ("replace", "append"):
        raise ArrayError(
            f"mode must be 'replace' or 'append', got {mode!r}"
        )

    data = np.asarray(data)
    full_name = f"{OBJECT_ATTRIBUTES}/{attr_name}"

    appending = (
        mode == "append"
        and level_group.standalone_array_exists(full_name)
    )
    if appending:
        existing = level_group.read_array(full_name)
        if existing.shape[1:] != data.shape[1:]:
            raise ArrayError(
                f"append shape mismatch: existing {existing.shape} vs "
                f"new {data.shape} — tail dimensions must match"
            )
        # Existing dtype wins on conflict; new rows are cast.
        new_rows = data.astype(existing.dtype, copy=False)
        target_dtype = existing.dtype
    else:
        existing = None
        new_rows = data
        target_dtype = data.dtype

    if fill_value is None:
        fill_value = _default_fill_value_for_dtype(target_dtype)

    if present_mask is not None:
        mask_arr = np.asarray(present_mask, dtype=bool)
        if mask_arr.shape[0] != data.shape[0]:
            raise ArrayError(
                f"present_mask length {mask_arr.shape[0]} != data row "
                f"count {data.shape[0]}"
            )
        new_rows = new_rows.copy()
        new_rows[~mask_arr] = fill_value

    if appending:
        write_data = np.concatenate([existing, new_rows], axis=0)
    else:
        write_data = new_rows

    level_group.write_array(
        full_name, write_data,
        fill_value=fill_value,
        attributes={
            "zv_array": "object_attribute",
            "name": attr_name,
            "dtype": str(write_data.dtype),
            "shape": list(write_data.shape),
            "fill_sentinel_meaning": "absent",
        },
    )


def read_object_attribute_present_mask(
    level_group: FsGroup,
    attr_name: str,
) -> npt.NDArray[np.uint8] | None:
    """Reconstruct the ``(O,)`` byte present-mask for an object attribute.

    The 0.8.1 layout stores absence in-band via ``fill_value`` rather
    than as a sibling ``present_mask`` array, so this reader compares
    each row against the stored fill on the fly.  A row counts as
    "absent" only if every channel equals the sentinel.

    Returns:
        ``uint8`` mask (``1`` = present, ``0`` = absent), or ``None``
        when every row is present (so callers can skip allocating a
        full mask).  Also returns ``None`` when the attribute is
        missing.
    """
    full_name = f"{OBJECT_ATTRIBUTES}/{attr_name}"
    if not level_group.standalone_array_exists(full_name):
        return None

    data = level_group.read_array(full_name)
    fill = level_group.read_array_fill_value(full_name)

    # NaN ≠ NaN, so detect via isnan for any float-like fill (Python
    # float or numpy float scalar — zarr can return either).
    fill_is_nan = False
    if data.dtype.kind == "f":
        try:
            fill_is_nan = bool(np.isnan(fill))
        except (TypeError, ValueError):
            fill_is_nan = False
    if fill_is_nan:
        absent_per_elem = np.isnan(data)
    else:
        absent_per_elem = (data == fill)

    if data.ndim == 1:
        absent = absent_per_elem
    else:
        # Row "absent" iff every channel equals the sentinel.
        absent = absent_per_elem.reshape(data.shape[0], -1).all(axis=1)

    if not absent.any():
        return None
    return (~absent).astype(np.uint8)


def write_groupings(
    level_group: FsGroup,
    groups: dict[int, list[int] | range],
) -> None:
    """Write group memberships as a single vlen-bytes Zarr v3 array.

    Each row of the array at ``groups/`` is the byte-serialised int64
    member list of one group, addressable by integer index.  This
    replaces the legacy ``groups/data`` + ``groups/offsets`` CSR pair
    with the same vlen layout already used by ``object_index/manifests``.

    A group whose members form a contiguous ``range(start, stop)`` may be
    passed as a ``range`` object.  It is stored implicitly as a
    ``[start, stop)`` descriptor in the array's ``group_ranges`` attribute
    (with an empty placeholder row), so a billion-member contiguous group
    costs O(1) on disk and in memory instead of an 8-byte-per-member int64
    list.  Explicit (non-range) groups keep their exact byte layout, so
    older stores remain readable.

    Args:
        level_group: Resolution level group.
        groups: ``{group_id: [object_id, ...] | range, ...}``.
            Group IDs must be contiguous starting from 0.  A ``range``
            value must have ``step == 1``.
    """
    if not groups:
        return

    max_gid = max(groups.keys())
    blobs: list[bytes] = []
    group_ranges: dict[str, list[int]] = {}
    for gid in range(max_gid + 1):
        members = groups.get(gid, [])
        if isinstance(members, range) and len(members) > 0:
            if members.step != 1:
                raise ArrayError(
                    f"range group {gid} must have step 1, got {members.step}"
                )
            group_ranges[str(gid)] = [members.start, members.stop]
            blobs.append(b"")  # implicit — members live in group_ranges
        else:
            arr = np.array(list(members), dtype=np.int64)
            blobs.append(arr.tobytes())

    attributes: dict[str, Any] = {
        "zv_array": "groups",
        "num_groups": max_gid + 1,
    }
    if group_ranges:
        attributes["group_ranges"] = group_ranges

    level_group.write_vlen_array(GROUPS, blobs, attributes=attributes)


def write_groupings_attributes(
    level_group: FsGroup,
    attr_name: str,
    data: npt.NDArray,
) -> None:
    """Write dense G×C groupings attribute data as a single Zarr v3 array.

    Stored at ``group_attributes/<attr_name>`` as a standard chunked
    Zarr v3 array.  Unlike :func:`write_object_attributes`, no
    sentinel / present-mask handling — groupings attributes are always
    fully populated.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name.
        data: ``(G,)`` or ``(G, C)`` array.
    """
    data = np.asarray(data)
    full_name = f"{GROUP_ATTRIBUTES}/{attr_name}"
    level_group.write_array(
        full_name, data,
        attributes={
            "zv_array": "groupings_attribute",
            "name": attr_name,
            "dtype": str(data.dtype),
            "shape": list(data.shape),
        },
    )


@dataclass
class CrossChunkLinkPartition:
    """Per-input-record cell mapping returned by
    :func:`write_cross_chunk_links`.

    Pass to :func:`write_cross_chunk_link_attributes` to align
    per-record attribute data with the per-cell layout used by the
    link writer.

    Attributes:
        cell_indices: Maps ``cell_key`` → list of indices into the
            input ``links`` list (in input order).  Within each
            bucket, indices preserve input ordering, so attribute
            data passed in input order partitions deterministically.
        num_links: Total record count after the write (==
            ``sum(len(v) for v in cell_indices.values())``).  For
            ``append`` mode this includes pre-existing records.
        first_new: Backward-compat — the input-order row index of the
            first newly-appended record.  ``0`` for ``replace`` mode,
            ``len(existing)`` for ``append`` mode.
    """

    cell_indices: dict[str, list[int]]
    num_links: int
    first_new: int = 0

    def __int__(self) -> int:
        # Lets legacy callers that did
        # ``first_new = int(write_cross_chunk_links(...))`` still work.
        return self.first_new


def _normalise_cross_records(
    links: list[list[tuple[ChunkCoords, int]]] | list[CrossChunkLink],
    link_width: int | None,
    sid_ndim: int,
    delta: int,
) -> tuple[list[list[tuple[ChunkCoords, int]]], int]:
    """Normalise cross-chunk input to list-of-lists and resolve link_width.

    Accepts the legacy ``((chunk_a, vi_a), (chunk_b, vi_b))`` 2-tuple form
    or a list of ``(chunk_coords, vi)`` endpoint lists.  Validates record
    arity against ``link_width`` (inferred from the first record if omitted)
    and every chunk-coord arity against ``sid_ndim``.
    """
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
    return normalised, link_width


def write_cross_chunk_links(
    level_group: FsGroup,
    links: list[list[tuple[ChunkCoords, int]]] | list[CrossChunkLink],
    sid_ndim: int,
    *,
    delta: int = 0,
    link_width: int | None = None,
    mode: Literal["replace", "append"] = "replace",
    directed: bool = False,
    store: Literal["canonical", "duplicate"] = "canonical",
) -> CrossChunkLinkPartition:
    """Write cross-chunk link records under ``cross_chunk_links/<delta>/<cell_key>``.

    Each record is ``link_width`` ``(chunk_coords, vertex_idx)``
    endpoints.  ``link_width=2`` (the default) encodes the classic
    cross-chunk edge ``((chunk_A, vi_A), (chunk_B, vi_B))``;
    ``link_width=3`` encodes a triangle face spanning chunks;
    ``link_width=1`` encodes a single parent→child reference used by
    pyramid metanode drill-down.

    Each record's endpoints are canonical-sorted by ``chunk_coords``
    (tie-break ``vi``) and the record is stored under the cell
    keyed by the dotted concatenation of those L sorted chunks (see
    :func:`zarr_vectors.core.paths.format_cell_key`).  Original
    endpoint order is preserved via a Lehmer-coded ``perm_idx``
    stored alongside the canonical vertex indices, so readers can
    recover the input ordering (mesh-face winding, directed-edge
    direction).

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
        mode: ``"replace"`` (default) clears the existing
            ``cross_chunk_links/<delta>/`` family and writes
            ``links`` afresh.  ``"append"`` reads existing records
            cell-by-cell, appends ``links`` (canonical-sorted to
            their cells), and writes the merged cells back.
            ``link_width`` of the appended records must match the
            existing ``link_width``.

    Returns:
        :class:`CrossChunkLinkPartition` describing where each input
        record landed.  ``int(partition)`` recovers the legacy
        ``first_new`` int for callers that only cared about the
        append offset.

    Other args:
        directed: When ``True``, endpoint order is meaningful (streamline
            predecessor→successor, skeleton child→parent): records are NOT
            canonical-sorted, the cell key follows input order, and
            ``perm_idx`` is 0.  ``A→B`` and ``B→A`` file under distinct
            cells.  Stamped in the family ``.zattrs`` so readers/validators
            don't guess.  A family is uniformly directed or undirected —
            appending with a different ``directed`` than the existing
            family raises.
        store: ``"canonical"`` (default) files each record in one cell;
            ``"duplicate"`` files it under one cell per distinct incident
            chunk (independent physical copies) so incidence reads become a
            prefix scan.  Duplicated links return multiple times from
            :func:`read_cross_chunk_links`; the parallel attribute family
            replicates identically via the returned partition.

    Concurrency:
        ``mode="append"`` is read-modify-write per cell — safe
        across disjoint cells, unsafe within a single cell.
    """
    if mode not in ("replace", "append"):
        raise ArrayError(
            f"mode must be 'replace' or 'append', got {mode!r}"
        )
    if store not in ("canonical", "duplicate"):
        raise ArrayError(
            f"store must be 'canonical' or 'duplicate', got {store!r}"
        )

    # Local import to avoid circular import at module load.
    from zarr_vectors.encoding.ragged import (
        decode_ragged_blob,
        encode_ragged_blob,
    )
    from zarr_vectors.spatial.boundary import partition_cross_records_by_tuple

    if not links:
        # Honour the empty-input fast-exit; emit an empty partition
        # so callers don't crash on attribute reflection.
        return CrossChunkLinkPartition(
            cell_indices={}, num_links=0, first_new=0,
        )

    normalised, link_width = _normalise_cross_records(
        links, link_width, sid_ndim, delta,
    )

    full_name = cross_chunk_links_path(delta)

    # Cross-link_width check on existing array before destructive write.
    if mode == "append" and level_group.array_exists(full_name):
        existing_meta = level_group.read_array_meta(full_name)
        if existing_meta:
            existing_link_width = int(
                existing_meta.get("link_width", link_width)
            )
            if existing_link_width != link_width:
                raise ArrayError(
                    f"cross_chunk_links/{format_delta(delta)}: cannot "
                    f"append records of link_width {link_width} onto "
                    f"existing array of link_width {existing_link_width}"
                )
            existing_sid = int(
                existing_meta.get("sid_ndim", sid_ndim)
            )
            if existing_sid != sid_ndim:
                raise ArrayError(
                    f"cross_chunk_links/{format_delta(delta)}: cannot "
                    f"append with sid_ndim={sid_ndim} onto existing "
                    f"sid_ndim={existing_sid}"
                )
            existing_directed = bool(existing_meta.get("directed", False))
            if existing_directed != bool(directed):
                raise ArrayError(
                    f"cross_chunk_links/{format_delta(delta)}: cannot "
                    f"append directed={directed} onto existing "
                    f"directed={existing_directed} (a family is uniformly "
                    f"directed or undirected)"
                )
            existing_store = str(existing_meta.get("store", "canonical"))
            if existing_store != store:
                raise ArrayError(
                    f"cross_chunk_links/{format_delta(delta)}: cannot "
                    f"append store={store!r} onto existing "
                    f"store={existing_store!r}"
                )

    # Build the per-cell partition (input order preserved within each
    # cell).  ``partition_cross_records_by_tuple`` applies the directed /
    # store policy: canonical sort for undirected, input-order keys for
    # directed, and one cell per distinct incident chunk for duplicate.
    partitioned = partition_cross_records_by_tuple(
        normalised, link_width, sid_ndim, directed=directed, store=store,
    )
    bucket_entries: dict[str, list[tuple[list[int], int]]] = {
        key: [(vi, perm) for vi, perm, _ in entries]
        for key, entries in partitioned.items()
    }
    cell_indices: dict[str, list[int]] = {
        key: [idx for _, _, idx in entries]
        for key, entries in partitioned.items()
    }
    # Physical rows this write adds (== len(normalised) for canonical, more
    # when ``store="duplicate"`` fans a record across incident-chunk cells).
    new_physical = sum(len(rows) for rows in bucket_entries.values())

    record_len_int64 = 1 + link_width  # [perm_idx, vi_0, ..., vi_{L-1}]

    if mode == "replace":
        # Drop the whole family — clears stale cells, .zattrs,
        # everything — and re-create with the new contents.
        if level_group.array_exists(full_name):
            level_group.delete_subtree(full_name)
        first_new = 0
        existing_total = 0
        existing_physical = 0

        # Write each new cell.
        for cell_key, rows in bucket_entries.items():
            blob_groups: list[npt.NDArray] = []
            for vi_cell, perm_idx in rows:
                row = np.empty(record_len_int64, dtype=np.int64)
                row[0] = perm_idx
                row[1:] = vi_cell
                blob_groups.append(row)
            blob = encode_ragged_blob(blob_groups, np.dtype(np.int64))
            level_group.write_bytes(full_name, cell_key, blob)
    else:
        # Append: per-cell RMW.  Cells not touched by ``links`` keep
        # their existing contents.
        existing_total = 0
        existing_physical = 0
        if level_group.array_exists(full_name):
            # Existing counts come from family meta in O(1); fall back to a
            # full cell scan only for older stores predating the meta.
            existing_meta = level_group.read_array_meta(full_name) or {}
            if "num_links" in existing_meta:
                existing_total = int(existing_meta["num_links"])
                existing_physical = int(
                    existing_meta.get("num_physical_records", existing_total)
                )
            else:
                for existing_key in level_group.list_chunks(full_name):
                    blob = level_group.read_bytes(full_name, existing_key)
                    existing_rows = decode_ragged_blob(
                        blob, np.dtype(np.int64), ncols=record_len_int64,
                    )
                    existing_physical += len(existing_rows)
                existing_total = existing_physical
        first_new = existing_total

        for cell_key, rows in bucket_entries.items():
            # Read existing cell rows (if any).
            existing_rows: list[npt.NDArray] = []
            if level_group.chunk_exists(full_name, cell_key):
                blob = level_group.read_bytes(full_name, cell_key)
                existing_rows = decode_ragged_blob(
                    blob, np.dtype(np.int64), ncols=record_len_int64,
                )
            new_rows: list[npt.NDArray] = []
            for vi_cell, perm_idx in rows:
                row = np.empty(record_len_int64, dtype=np.int64)
                row[0] = perm_idx
                row[1:] = vi_cell
                new_rows.append(row)
            combined = existing_rows + new_rows
            blob = encode_ragged_blob(combined, np.dtype(np.int64))
            level_group.write_bytes(full_name, cell_key, blob)

    # Family .zattrs on the parent group.  ``num_links`` is the *logical*
    # record count (one per input record); ``num_physical_records`` is the
    # on-disk row count (larger than ``num_links`` under ``duplicate``).
    # ``directed`` / ``store`` let readers and the validator interpret the
    # cell keys without inspecting the data.
    new_total = existing_total + len(normalised)
    new_physical_total = existing_physical + new_physical
    _ensure_array_dir(level_group, full_name)
    level_group.write_array_meta(full_name, {
        "zv_array": "cross_chunk_links",
        "sid_ndim": int(sid_ndim),
        "level_delta": int(delta),
        "link_width": int(link_width),
        "num_links": int(new_total),
        "num_physical_records": int(new_physical_total),
        "directed": bool(directed),
        "store": str(store),
    })

    return CrossChunkLinkPartition(
        cell_indices=cell_indices,
        num_links=new_total,
        first_new=first_new,
    )


def write_cross_chunk_link_attributes(
    level_group: FsGroup,
    attr_name: str,
    attr_data: npt.NDArray,
    *,
    num_links: int,
    delta: int = 0,
    mode: Literal["replace", "append"] = "replace",
    partition: CrossChunkLinkPartition | None = None,
) -> None:
    """Write per-edge attribute data parallel to
    ``cross_chunk_links/<delta>/<cell_key>``.

    Attribute rows are stored per-cell, one row per record, matching
    the row order of the parallel links cell.

    ``attr_data`` is expected in **input order** — the same ordering
    used for the corresponding ``write_cross_chunk_links`` call.  The
    ``partition`` returned by that call provides the cell mapping; if
    omitted, the caller is responsible for ensuring ``attr_data`` is
    in input order and the writer falls back to re-deriving the
    partition by reading back the link records.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name.
        attr_data: ``(num_links,)`` or ``(num_links, C)`` array in
            **input order**.  In ``mode="append"`` this represents
            the NEW rows; the post-append total must equal
            ``num_links``.
        num_links: Expected post-write total record count (matches
            ``CrossChunkLinkPartition.num_links``).
        delta: Level delta.
        mode: ``"replace"`` (default) — wipes existing cells and
            writes the full attribute family afresh.  ``"append"`` —
            per-cell RMW: existing attribute rows are kept, new rows
            are appended to their respective cells in the same per-
            cell order as the link writer used.
        partition: Optional :class:`CrossChunkLinkPartition` returned
            by the matching :func:`write_cross_chunk_links` call.
            Supplying it avoids reading the link records back.

    Raises:
        ArrayError: If ``mode`` is invalid, if the row count of
            ``attr_data`` does not match the expected ``num_links``
            (replace) or the new-row count derived from
            ``partition`` (append), or if the appended row shape
            does not match an existing array.

    Concurrency:
        ``mode="append"`` is read-modify-write per cell — safe
        across disjoint cells, unsafe within a single cell.
    """
    if mode not in ("replace", "append"):
        raise ArrayError(
            f"mode must be 'replace' or 'append', got {mode!r}"
        )

    full_name = cross_chunk_link_attributes_path(attr_name, delta)
    links_family = cross_chunk_links_path(delta)
    arr = np.ascontiguousarray(np.asarray(attr_data))

    # Resolve the partition.  Append mode requires it explicitly —
    # only the matching write_cross_chunk_links call knows which
    # cell each new attribute row belongs to.  Replace mode can
    # fall back to re-deriving from disk (assumes ``attr_data`` is
    # in cell-key-sorted on-disk order).
    if partition is None:
        if mode == "append":
            raise ArrayError(
                f"cross_chunk_link_attributes[{attr_name}] append mode "
                f"requires partition=... from the matching "
                f"write_cross_chunk_links(mode='append') call so the "
                f"writer knows which cell each new row belongs to"
            )
        # A duplicate family fans one input record across several cells;
        # only the link writer's partition records that fan-out, so the
        # re-derived (physical, positional) partition would misalign
        # attribute rows.  Require the partition explicitly.
        links_meta = level_group.read_array_meta(links_family) or {}
        if str(links_meta.get("store", "canonical")) == "duplicate":
            raise ArrayError(
                f"cross_chunk_link_attributes[{attr_name}]: store='duplicate' "
                f"requires partition=... from the matching "
                f"write_cross_chunk_links call so replicated rows align"
            )
        partition = _derive_partition_from_links(
            level_group, links_family, delta,
        )

    # Validate counts.
    if mode == "replace":
        if arr.shape[0] != num_links:
            raise ArrayError(
                f"cross_chunk_link_attributes[{attr_name}] row count "
                f"{arr.shape[0]} != num_links {num_links} "
                f"(delta={format_delta(delta)})"
            )
        if arr.shape[0] != partition.num_links:
            raise ArrayError(
                f"cross_chunk_link_attributes[{attr_name}] attr rows "
                f"{arr.shape[0]} != link records {partition.num_links} "
                f"derived from cross_chunk_links/{format_delta(delta)}"
            )
        # Wipe + rewrite the whole family.
        if level_group.array_exists(full_name):
            level_group.delete_subtree(full_name)

        for cell_key, input_idxs in partition.cell_indices.items():
            cell_attrs = arr[np.asarray(input_idxs, dtype=np.int64)]
            level_group.write_bytes(
                full_name, cell_key,
                np.ascontiguousarray(cell_attrs).tobytes(),
            )
    else:
        # Append mode.  The post-append total record count must equal
        # num_links; arr.shape[0] is the new-row count.
        new_count = sum(len(v) for v in partition.cell_indices.values())
        if arr.shape[0] != new_count:
            raise ArrayError(
                f"cross_chunk_link_attributes[{attr_name}] append: "
                f"attr_data row count {arr.shape[0]} != new link "
                f"record count {new_count} "
                f"(delta={format_delta(delta)})"
            )
        if partition.num_links != num_links:
            raise ArrayError(
                f"cross_chunk_link_attributes[{attr_name}] row count "
                f"{partition.num_links} != num_links {num_links} "
                f"(delta={format_delta(delta)})"
            )

        for cell_key, input_idxs in partition.cell_indices.items():
            cell_attrs_new = arr[np.asarray(input_idxs, dtype=np.int64)]
            if level_group.chunk_exists(full_name, cell_key):
                existing_meta = level_group.read_array_meta(full_name)
                existing_blob = level_group.read_bytes(full_name, cell_key)
                if existing_meta and "dtype" in existing_meta:
                    existing_dtype = np.dtype(existing_meta["dtype"])
                else:
                    existing_dtype = arr.dtype
                tail_shape = tuple(existing_meta.get(
                    "row_shape", arr.shape[1:],
                ))
                if tail_shape != arr.shape[1:]:
                    raise ArrayError(
                        f"cross_chunk_link_attributes[{attr_name}] "
                        f"append shape mismatch: existing row shape "
                        f"{tail_shape} vs new {arr.shape[1:]}"
                    )
                row_size = int(np.prod(tail_shape)) if tail_shape else 1
                row_bytes = existing_dtype.itemsize * row_size
                existing_count = len(existing_blob) // row_bytes
                existing_arr = np.frombuffer(
                    existing_blob, dtype=existing_dtype,
                ).reshape((existing_count, *tail_shape)).copy()
                new_cast = cell_attrs_new.astype(existing_dtype, copy=False)
                combined = np.concatenate([existing_arr, new_cast], axis=0)
            else:
                combined = cell_attrs_new
            level_group.write_bytes(
                full_name, cell_key,
                np.ascontiguousarray(combined).tobytes(),
            )

    # Family .zattrs.  Stores ``row_shape`` (the tail dimensions per
    # row, ``()`` for 1-D arrays) so readers can reconstruct shape
    # from a per-cell byte blob without ambiguity.
    _ensure_array_dir(level_group, full_name)
    level_group.write_array_meta(full_name, {
        "zv_array": "cross_chunk_link_attribute",
        "name": attr_name,
        "dtype": str(arr.dtype),
        "level_delta": int(delta),
        "num_links": int(num_links),
        "row_shape": list(arr.shape[1:]),
    })


def _derive_partition_from_links(
    level_group: FsGroup,
    links_family: str,
    delta: int,
) -> CrossChunkLinkPartition:
    """Re-derive a :class:`CrossChunkLinkPartition` by reading the
    on-disk records under ``links_family``.

    Returns a partition whose ``cell_indices`` is keyed in the same
    iteration order as the on-disk cell layout (cell-key-sorted),
    with input indices ``0..num_links-1`` assigned in that read
    order.  Callers passing ``attr_data`` to
    :func:`write_cross_chunk_link_attributes` without an explicit
    partition must supply rows in this same cell-key-sorted order.
    """
    cell_indices: dict[str, list[int]] = {}
    next_idx = 0
    if level_group.array_exists(links_family):
        meta = level_group.read_array_meta(links_family) or {}
        link_width = int(meta.get("link_width", 2))
        record_len = 1 + link_width
        # Local import to avoid pulling decode_ragged_blob at module load.
        from zarr_vectors.encoding.ragged import decode_ragged_blob

        for cell_key in sorted(level_group.list_chunks(links_family)):
            blob = level_group.read_bytes(links_family, cell_key)
            rows = decode_ragged_blob(
                blob, np.dtype(np.int64), ncols=record_len,
            )
            n = len(rows)
            cell_indices[cell_key] = list(range(next_idx, next_idx + n))
            next_idx += n
    return CrossChunkLinkPartition(
        cell_indices=cell_indices,
        num_links=next_idx,
        first_new=0,
    )


# ===================================================================
# Decentralized (per-cell) cross-chunk-link writers
# ===================================================================
#
# ``write_cross_chunk_links`` is a whole-family replace/append that also
# maintains the family-wide ``num_links`` / ``num_physical_records`` meta.
# The functions below instead let many independent workers each append a
# batch of records into ONLY the cells those records touch, deferring the
# global bookkeeping to a single ``finalize_cross_chunk_links`` pass.
#
# Race-freedom rests on the flat layout: each cell is its own object
# (sharding is a later, coordinator-run pass), so workers writing DISJOINT
# cells never touch the same file.  Give each worker ownership of the
# records whose canonical-first chunk (or, when ``directed``, source chunk)
# it is responsible for, so every cell is written by exactly one worker.
# Per-cell RMW is NOT safe for two workers hitting the SAME cell.


def write_cross_chunk_link_cells(
    level_group: FsGroup,
    links: list[list[tuple[ChunkCoords, int]]] | list[CrossChunkLink],
    sid_ndim: int,
    *,
    delta: int = 0,
    link_width: int | None = None,
    directed: bool = False,
    store: Literal["canonical", "duplicate"] = "canonical",
) -> CrossChunkLinkPartition:
    """Append a batch of cross-chunk records into only the cells they touch.

    Unlike :func:`write_cross_chunk_links`, this decentralized writer leaves
    every other cell untouched and does **not** update the family-wide
    ``num_links`` / ``num_physical_records`` counts — call
    :func:`finalize_cross_chunk_links` once, after all workers finish, to
    reconcile them (and shard separately if desired).

    A coordinator should pre-create the family with matching ``directed`` /
    ``store`` / ``sid_ndim`` via :func:`create_cross_chunk_links_array` so
    workers agree on the policy and don't race to create it; this function
    also creates it idempotently if absent and rejects a policy mismatch.

    Returns the :class:`CrossChunkLinkPartition` for THIS batch, for use
    with :func:`write_cross_chunk_link_attribute_cells`.
    """
    if store not in ("canonical", "duplicate"):
        raise ArrayError(
            f"store must be 'canonical' or 'duplicate', got {store!r}"
        )
    if not links:
        return CrossChunkLinkPartition(
            cell_indices={}, num_links=0, first_new=0,
        )

    from zarr_vectors.encoding.ragged import (
        decode_ragged_blob,
        encode_ragged_blob,
    )
    from zarr_vectors.spatial.boundary import partition_cross_records_by_tuple

    normalised, link_width = _normalise_cross_records(
        links, link_width, sid_ndim, delta,
    )
    full_name = cross_chunk_links_path(delta)

    # Ensure the family exists with the requested policy (idempotent), then
    # guard against a coordinator that created it with different flags.
    create_cross_chunk_links_array(
        level_group, delta=delta, link_width=link_width, sid_ndim=sid_ndim,
        directed=directed, store=store, exist_ok=True,
    )
    fam_meta = level_group.read_array_meta(full_name) or {}
    if bool(fam_meta.get("directed", directed)) != directed:
        raise ArrayError(
            f"cross_chunk_links/{format_delta(delta)}: family directed="
            f"{fam_meta.get('directed')} != requested {directed}"
        )
    if str(fam_meta.get("store", store)) != store:
        raise ArrayError(
            f"cross_chunk_links/{format_delta(delta)}: family store="
            f"{fam_meta.get('store')!r} != requested {store!r}"
        )
    if int(fam_meta.get("link_width", link_width)) != link_width:
        raise ArrayError(
            f"cross_chunk_links/{format_delta(delta)}: family link_width="
            f"{fam_meta.get('link_width')} != requested {link_width}"
        )

    partitioned = partition_cross_records_by_tuple(
        normalised, link_width, sid_ndim, directed=directed, store=store,
    )
    record_len_int64 = 1 + link_width
    for cell_key, entries in partitioned.items():
        existing_rows: list[npt.NDArray] = []
        if level_group.chunk_exists(full_name, cell_key):
            blob = level_group.read_bytes(full_name, cell_key)
            existing_rows = decode_ragged_blob(
                blob, np.dtype(np.int64), ncols=record_len_int64,
            )
        new_rows: list[npt.NDArray] = []
        for vi_cell, perm_idx, _ in entries:
            row = np.empty(record_len_int64, dtype=np.int64)
            row[0] = perm_idx
            row[1:] = vi_cell
            new_rows.append(row)
        combined = list(existing_rows) + new_rows
        blob = encode_ragged_blob(combined, np.dtype(np.int64))
        level_group.write_bytes(full_name, cell_key, blob)

    cell_indices = {
        key: [idx for _, _, idx in entries]
        for key, entries in partitioned.items()
    }
    return CrossChunkLinkPartition(
        cell_indices=cell_indices,
        num_links=len(normalised),
        first_new=0,
    )


def write_cross_chunk_link_attribute_cells(
    level_group: FsGroup,
    attr_name: str,
    attr_data: npt.NDArray,
    *,
    partition: CrossChunkLinkPartition,
    delta: int = 0,
) -> None:
    """Append attribute rows for the cells one batch wrote.

    ``attr_data`` holds one row per logical record in the SAME batch, in the
    input order used for the matching :func:`write_cross_chunk_link_cells`
    call; ``partition`` is that call's return value.  Rows are appended per
    cell (replicating automatically in ``duplicate`` mode, where an input
    index appears under several cells).  Global counts are reconciled by
    :func:`finalize_cross_chunk_links`; only ``dtype`` / ``row_shape`` are
    stamped here (idempotent across workers).
    """
    full_name = cross_chunk_link_attributes_path(attr_name, delta)
    arr = np.ascontiguousarray(np.asarray(attr_data))
    tail_shape = arr.shape[1:]
    for cell_key, input_idxs in partition.cell_indices.items():
        new_rows = arr[np.asarray(input_idxs, dtype=np.int64)]
        if level_group.chunk_exists(full_name, cell_key):
            existing_blob = level_group.read_bytes(full_name, cell_key)
            existing_arr = np.frombuffer(
                existing_blob, dtype=arr.dtype,
            ).reshape((-1, *tail_shape))
            combined = np.concatenate([existing_arr, new_rows], axis=0)
        else:
            combined = new_rows
        level_group.write_bytes(
            full_name, cell_key,
            np.ascontiguousarray(combined).tobytes(),
        )

    _ensure_array_dir(level_group, full_name)
    meta = level_group.read_array_meta(full_name) or {}
    meta.update({
        "zv_array": "cross_chunk_link_attribute",
        "name": attr_name,
        "dtype": str(arr.dtype),
        "row_shape": list(tail_shape),
        "level_delta": int(delta),
    })
    level_group.write_array_meta(full_name, meta)


def finalize_cross_chunk_links(
    level_group: FsGroup,
    *,
    delta: int = 0,
) -> CrossChunkLinkPartition:
    """Reconcile a ``cross_chunk_links/<delta>/`` family's counts after
    decentralized per-cell writes.

    Scans every cell to recompute ``num_physical_records`` (total on-disk
    rows) and ``num_links`` (logical record count).  For ``canonical``
    families the two are equal; for ``duplicate`` families the logical
    count is recovered by deduplicating decoded records (every copy of a
    record decodes to the same input-order endpoints).

    Sharding is a separate coordinator step — run
    :func:`zarr_vectors.sharding.shard_store` after finalizing, once all
    cells are on disk.

    Returns the re-derived :class:`CrossChunkLinkPartition` (cell-key order).
    """
    from zarr_vectors.encoding.ragged import decode_ragged_blob

    full_name = cross_chunk_links_path(delta)
    if not level_group.array_exists(full_name):
        return CrossChunkLinkPartition(
            cell_indices={}, num_links=0, first_new=0,
        )
    meta = level_group.read_array_meta(full_name) or {}
    link_width = int(meta.get("link_width", 2))
    store = str(meta.get("store", "canonical"))
    record_len = 1 + link_width

    physical = 0
    cell_indices: dict[str, list[int]] = {}
    next_idx = 0
    for cell_key in sorted(level_group.list_chunks(full_name)):
        blob = level_group.read_bytes(full_name, cell_key)
        rows = decode_ragged_blob(
            blob, np.dtype(np.int64), ncols=record_len,
        )
        n = len(rows)
        cell_indices[cell_key] = list(range(next_idx, next_idx + n))
        next_idx += n
        physical += n

    if store == "duplicate":
        # Every physical copy of a logical record decodes to the same
        # input-order endpoints, so distinct decoded records == logical.
        records = read_cross_chunk_links(level_group, delta=delta)
        logical = len({tuple(r) for r in records})
    else:
        logical = physical

    meta["num_links"] = int(logical)
    meta["num_physical_records"] = int(physical)
    _ensure_array_dir(level_group, full_name)
    level_group.write_array_meta(full_name, meta)

    return CrossChunkLinkPartition(
        cell_indices=cell_indices,
        num_links=physical,
        first_new=0,
    )


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
            # In the native-sharded layout, ``links/0`` is a single
            # multidim Zarr array; cells for chunks with no intra-edges
            # return ``b""`` (the vlen fill value) instead of "chunk not
            # found".  Short-circuit that case before trying to read
            # the (likely absent) ``link_fragments`` array.
            if not raw:
                return []
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

    _require_object_index_v1(meta)
    manifests_arr = level_group.zarr_group[OBJECT_INDEX]["manifests"]
    # Slice (then index) instead of scalar indexing: zarr 3.x vlen-bytes
    # returns a 0-d object ndarray under ``arr[i]``, whose ``bytes()``
    # is the array header — not the payload.  ``arr[i:i+1][0]`` is the
    # actual bytes object and still fetches only the chunk holding i.
    blob = manifests_arr[object_id:object_id + 1][0]
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

    _require_object_index_v1(meta)
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


def _require_object_index_v1(meta: dict[str, Any]) -> None:
    """Raise if ``object_index`` is not in the 0.8.1 vlen-manifests layout.

    Pre-0.6 stores wrote a single-chunk ``data`` + ``offsets`` pair; the
    reader for that layout was removed in 0.8.1 alongside the broader
    flattening migration.  Any store reaching this point with a
    non-V1 ``layout`` field is therefore older than what this build
    supports and must be rewritten from source — surfacing that as a
    clear error here beats decoding garbage out of legacy bytes.
    """
    layout = meta.get("layout")
    if layout != OBJECT_INDEX_LAYOUT_V1:
        raise ArrayError(
            f"object_index layout {layout!r} is not the 0.8.1 vlen-manifests "
            f"layout ({OBJECT_INDEX_LAYOUT_V1!r}); pre-0.6 ``data``+``offsets`` "
            f"stores are not readable by this build — rewrite from source."
        )


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

    Returns the raw array values, including any sentinel positions for
    absent rows.  Use :func:`read_object_attribute_present_mask` to
    reconstruct a 0/1 mask, or compare directly against
    ``level_group.read_array_fill_value(...)``.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name.
        dtype: Optional override applied via ``astype`` after read.

    Returns:
        Array of shape ``(O,)`` or ``(O, C)``.
    """
    full_name = f"{OBJECT_ATTRIBUTES}/{attr_name}"
    out = level_group.read_array(full_name)
    if dtype is not None:
        out = out.astype(np.dtype(dtype), copy=False)
    return out


def read_group_object_ids(
    level_group: FsGroup,
    group_id: int,
) -> Sequence[int]:
    """Read the object IDs belonging to a group.

    Args:
        level_group: Resolution level group.
        group_id: Group ID.

    Returns:
        The member object IDs.  A group written from a ``range`` (see
        :func:`write_groupings`) is returned as a ``range`` — O(1) rather
        than materialising up to a billion ints.  Explicit groups are
        returned as a ``list[int]``.  Both support ``len()``, indexing
        and iteration.
    """
    blobs = level_group.read_vlen_array(GROUPS)
    if group_id < 0 or group_id >= len(blobs):
        raise ArrayError(
            f"Group ID {group_id} out of range [0, {len(blobs)})"
        )
    group_ranges = level_group.read_array_meta(GROUPS).get("group_ranges", {})
    rng = group_ranges.get(str(group_id))
    if rng is not None:
        return range(int(rng[0]), int(rng[1]))
    return np.frombuffer(blobs[group_id], dtype=np.int64).tolist()


def read_all_groupings(
    level_group: FsGroup,
) -> list[Sequence[int]]:
    """Read all group memberships.

    Returns:
        List indexed by group_id.  Each entry is a ``list[int]`` for an
        explicit group, or a ``range`` for an implicitly-stored contiguous
        group (see :func:`write_groupings`).
    """
    blobs = level_group.read_vlen_array(GROUPS)
    group_ranges = level_group.read_array_meta(GROUPS).get("group_ranges", {})
    out: list[Sequence[int]] = []
    for gid, b in enumerate(blobs):
        rng = group_ranges.get(str(gid))
        if rng is not None:
            out.append(range(int(rng[0]), int(rng[1])))
        else:
            out.append(np.frombuffer(b, dtype=np.int64).tolist())
    return out




def read_groupings_attributes(
    level_group: FsGroup,
    attr_name: str,
    dtype: np.dtype | str | None = None,
) -> npt.NDArray:
    """Read dense G×C groupings attribute data."""
    full_name = f"{GROUP_ATTRIBUTES}/{attr_name}"
    out = level_group.read_array(full_name)
    if dtype is not None:
        out = out.astype(np.dtype(dtype), copy=False)
    return out


def read_cross_chunk_links(
    level_group: FsGroup,
    *,
    delta: int = 0,
) -> list[tuple[tuple[ChunkCoords, int], ...]]:
    """Read all cross-chunk link records under ``cross_chunk_links/<delta>/``.

    Records are returned in cell-key-sorted order (and within each
    cell in write order).  Each record is a tuple of
    ``(chunk_coords, vi)`` endpoints in **original input order** —
    canonical sorting and ``perm_idx`` encoding are reversed here so
    callers see the same record shape they wrote.  This works uniformly
    for directed families (``perm_idx=0`` makes the reversal a no-op).

    For a ``store="duplicate"`` family each logical record was written to
    several cells, so it is returned **once per copy** — callers wanting
    unique links should dedupe, or query with
    :func:`read_cross_chunk_links_for_tuple`.

    Returns ``[]`` when the ``<delta>`` family is absent or empty.
    """
    full_name = cross_chunk_links_path(delta)
    if not level_group.array_exists(full_name):
        return []
    meta = level_group.read_array_meta(full_name) or {}
    if "link_width" not in meta or "sid_ndim" not in meta:
        return []
    link_width = int(meta["link_width"])
    sid_ndim = int(meta["sid_ndim"])

    from zarr_vectors.encoding.ragged import decode_ragged_blob
    from zarr_vectors.spatial.boundary import apply_perm_inverse

    record_len = 1 + link_width  # [perm_idx, vi_0, ..., vi_{L-1}]
    out: list[tuple[tuple[ChunkCoords, int], ...]] = []
    for cell_key in sorted(level_group.list_chunks(full_name)):
        canonical_chunks = parse_cell_key(
            cell_key, sid_ndim=sid_ndim, link_width=link_width,
        )
        blob = level_group.read_bytes(full_name, cell_key)
        rows = decode_ragged_blob(
            blob, np.dtype(np.int64), ncols=record_len,
        )
        if not rows:
            continue
        # Decode the whole cell once into an (R, record_len) array and pull
        # the perm column + vi columns out in two C-level conversions,
        # instead of ``np.asarray(row).reshape(-1)`` + per-element int casts
        # per record.
        rows_arr = np.asarray(rows, dtype=np.int64).reshape(len(rows), record_len)
        perm_list = rows_arr[:, 0].tolist()
        vi_list = rows_arr[:, 1:1 + link_width].tolist()
        if link_width == 2:
            # The only 2-endpoint perms are identity (0) and swap (1);
            # avoid the per-row Lehmer decode in apply_perm_inverse.
            cc0, cc1 = canonical_chunks[0], canonical_chunks[1]
            for perm_idx, (v0, v1) in zip(perm_list, vi_list):
                ep0 = (cc0, v0)
                ep1 = (cc1, v1)
                out.append((ep0, ep1) if perm_idx == 0 else (ep1, ep0))
        else:
            for perm_idx, vi_canonical in zip(perm_list, vi_list):
                canonical_endpoints = list(zip(canonical_chunks, vi_canonical))
                out.append(tuple(apply_perm_inverse(
                    canonical_endpoints, perm_idx, link_width,
                )))
    return out


def read_cross_chunk_links_for_tuple(
    level_group: FsGroup,
    chunk_tuple: Sequence[ChunkCoords],
    *,
    delta: int = 0,
) -> list[tuple[tuple[ChunkCoords, int], ...]]:
    """Read records that span exactly the L chunks in ``chunk_tuple``.

    For an **undirected** family the input chunk-tuple is canonical-sorted
    internally, so callers may pass the chunks in any order.  For a
    **directed** family the tuple selects a specific directional cell —
    ``(A, B)`` and ``(B, A)`` are different cells — so the order matters.
    ``len(chunk_tuple)`` must equal the family's ``link_width``.

    Returns ``[]`` if no records exist for that exact L-tuple.
    Records are returned in write order, each in original input
    endpoint order (``perm_idx`` is reversed for the caller).
    """
    full_name = cross_chunk_links_path(delta)
    if not level_group.array_exists(full_name):
        return []
    meta = level_group.read_array_meta(full_name) or {}
    if "link_width" not in meta or "sid_ndim" not in meta:
        return []
    link_width = int(meta["link_width"])
    sid_ndim = int(meta["sid_ndim"])
    directed = bool(meta.get("directed", False))
    if len(chunk_tuple) != link_width:
        raise ArrayError(
            f"chunk_tuple has {len(chunk_tuple)} chunks; expected "
            f"link_width={link_width}"
        )
    for c in chunk_tuple:
        if len(c) != sid_ndim:
            raise ArrayError(
                f"chunk_tuple element {c} has arity {len(c)}; "
                f"expected sid_ndim={sid_ndim}"
            )

    # Directed cells key on input order; undirected canonical-sort so any
    # caller ordering resolves to the one canonical cell.
    if directed:
        query_chunks = tuple(tuple(c) for c in chunk_tuple)
    else:
        query_chunks = tuple(sorted(tuple(c) for c in chunk_tuple))
    cell_key = format_cell_key(query_chunks)
    if not level_group.chunk_exists(full_name, cell_key):
        return []

    from zarr_vectors.encoding.ragged import decode_ragged_blob
    from zarr_vectors.spatial.boundary import apply_perm_inverse

    record_len = 1 + link_width
    blob = level_group.read_bytes(full_name, cell_key)
    rows = decode_ragged_blob(
        blob, np.dtype(np.int64), ncols=record_len,
    )
    out: list[tuple[tuple[ChunkCoords, int], ...]] = []
    for row in rows:
        row_arr = np.asarray(row).reshape(-1)
        perm_idx = int(row_arr[0])
        vi_cell = [int(v) for v in row_arr[1:1 + link_width]]
        cell_endpoints = list(zip(query_chunks, vi_cell))
        input_order = apply_perm_inverse(
            cell_endpoints, perm_idx, link_width,
        )
        out.append(tuple(input_order))
    return out


def read_cross_chunk_link_attributes(
    level_group: FsGroup,
    attr_name: str,
    dtype: np.dtype | str | None = None,
    *,
    delta: int = 0,
) -> npt.NDArray:
    """Read per-link attribute data under ``cross_chunk_link_attributes/<name>/<delta>/``.

    Returns rows in cell-key-sorted order — the same order as
    :func:`read_cross_chunk_links` returns records.

    Returns:
        Array of shape ``(num_links,)`` or ``(num_links, *row_shape)``.
    """
    full_name = cross_chunk_link_attributes_path(attr_name, delta)
    if not level_group.array_exists(full_name):
        return np.array([], dtype=dtype or np.float32)
    meta = level_group.read_array_meta(full_name) or {}
    if dtype is None:
        dtype = np.dtype(meta["dtype"])
    else:
        dtype = np.dtype(dtype)
    row_shape = tuple(meta.get("row_shape", ()))
    row_size = int(np.prod(row_shape)) if row_shape else 1
    row_bytes = dtype.itemsize * row_size

    chunks: list[npt.NDArray] = []
    for cell_key in sorted(level_group.list_chunks(full_name)):
        blob = level_group.read_bytes(full_name, cell_key)
        if not blob:
            continue
        n = len(blob) // row_bytes
        arr = np.frombuffer(blob, dtype=dtype).reshape(
            (n, *row_shape) if row_shape else (n,),
        )
        chunks.append(arr.copy())
    if not chunks:
        return np.empty((0, *row_shape), dtype=dtype) if row_shape else np.empty((0,), dtype=dtype)
    return np.concatenate(chunks, axis=0)


def read_cross_chunk_link_attributes_for_tuple(
    level_group: FsGroup,
    attr_name: str,
    chunk_tuple: Sequence[ChunkCoords],
    dtype: np.dtype | str | None = None,
    *,
    delta: int = 0,
) -> npt.NDArray:
    """Read per-link attribute rows for the cell spanning ``chunk_tuple``.

    Returns rows in the same order as
    :func:`read_cross_chunk_links_for_tuple` returns records for the
    same chunk-tuple (cell write order).
    """
    full_name = cross_chunk_link_attributes_path(attr_name, delta)
    if not level_group.array_exists(full_name):
        return np.array([], dtype=dtype or np.float32)
    meta = level_group.read_array_meta(full_name) or {}
    if dtype is None:
        dtype = np.dtype(meta["dtype"])
    else:
        dtype = np.dtype(dtype)
    row_shape = tuple(meta.get("row_shape", ()))
    row_size = int(np.prod(row_shape)) if row_shape else 1
    row_bytes = dtype.itemsize * row_size

    # Cell keys must match the parallel links family — read its `directed`
    # flag (attribute meta doesn't carry it) to pick the same ordering.
    links_meta = level_group.read_array_meta(
        cross_chunk_links_path(delta)
    ) or {}
    if bool(links_meta.get("directed", False)):
        query_chunks = tuple(tuple(c) for c in chunk_tuple)
    else:
        query_chunks = tuple(sorted(tuple(c) for c in chunk_tuple))
    cell_key = format_cell_key(query_chunks)
    if not level_group.chunk_exists(full_name, cell_key):
        return np.empty((0, *row_shape), dtype=dtype) if row_shape else np.empty((0,), dtype=dtype)
    blob = level_group.read_bytes(full_name, cell_key)
    n = len(blob) // row_bytes
    return np.frombuffer(blob, dtype=dtype).reshape(
        (n, *row_shape) if row_shape else (n,),
    ).copy()


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


