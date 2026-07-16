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
from typing import Any, Iterator, Literal, Sequence

import numpy as np
import numpy.typing as npt
import zarr
from zarr.codecs import VLenBytesCodec
from zarr.errors import UnstableSpecificationWarning

from zarr_vectors.constants import (
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
    format_delta,
    format_offsets,
    intra_offsets,
    is_intra,
    link_attributes_group_path,
    link_attributes_path,
    links_group_path,
    links_path,
    parse_delta,
    parse_offsets,
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

    Detects both layouts: the legacy group-with-chunk-arrays form
    (``array_exists``) and the single-standard-Zarr-v3-array layout
    (``standalone_array_exists``).

    When the caller is running inside
    :meth:`Group.native_sharded_arrays` and the existing node is a
    legacy group at a per-chunk-array path, *do not*
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

    # A single vlen array already exists but an explicit session wants a
    # different layout (e.g. the warm-created unsharded ``vertices`` is
    # now being written with ``shard_shape=``).  Fall through so
    # ``_ensure_array_dir`` recreates it to match the requested layout.
    if (
        exists_as_array
        and level_group._native_sharded_config is not None
        and _is_per_chunk_array(full_name)
        and not _array_matches_layout(
            level_group, full_name, level_group._native_sharded_config
        )
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


_UNSET = object()


def _derive_native_config(level_group: FsGroup) -> dict[str, Any] | None:
    """Best-effort single-array grid config from the store's metadata.

    Lets writers that don't open an explicit
    :func:`open_write_session` (multiresolution coarsen, rechunk, lazy
    append, in-place edits that add a new array) still produce the
    single vlen-array layout.  Derives ``(origin, grid_shape)`` from the
    root ``bounds`` and the level's effective ``chunk_shape``.

    Returns ``None`` when that metadata can't be read — e.g. a bare
    ``Group`` with no root ``.zattrs``.  There is no longer a per-cell
    group primitive to fall back to: :meth:`Group.write_bytes` requires an
    allocated chunk array and raises otherwise.  A ``None`` here therefore
    means a grid-shaped array cannot be allocated at all, which is
    intrinsic rather than incidental — the grid is precisely what the
    missing metadata would have supplied.

    Cached on the ``level_group`` instance for the object's lifetime.
    """
    cached = level_group.__dict__.get("_derived_native_config", _UNSET)
    if cached is not _UNSET:
        return cached

    cfg: dict[str, Any] | None
    try:
        import zarr

        from zarr_vectors.core.metadata import (
            LevelMetadata,
            get_level_chunk_shape,
        )
        from zarr_vectors.core.store import read_root_metadata

        store = level_group._zarr.store
        root_zarr = zarr.open_group(store, path="/", mode="r")
        root_group = type(level_group)._from_zarr(root_zarr)
        root_meta = read_root_metadata(root_group)
        try:
            level_meta = LevelMetadata.from_dict(level_group.attrs.to_dict())
        except Exception:
            level_meta = None
        chunk_shape = get_level_chunk_shape(root_meta, level_meta)
        origin, grid_shape = level_grid_layout(root_meta.bounds, chunk_shape)
        cfg = {
            "origin": origin,
            "grid_shape": grid_shape,
            "shard_shape": None,
        }
    except Exception:
        cfg = None

    level_group.__dict__["_derived_native_config"] = cfg
    return cfg


def _derive_level_scales(
    level_group: FsGroup, delta: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ``(scale_src, scale_trg)`` for records spanning ``delta`` levels.

    Each is the per-axis integer multiple of the root ``chunk_shape`` that
    the relevant level's ``chunk_shape`` represents — ``r_src`` for the
    owning level, ``r_trg`` for ``owning + delta``.
    :func:`zarr_vectors.spatial.boundary.anchor_chunk` needs both to make
    a cross-level chunk offset well-defined: when the two levels carry
    different ``chunk_shape`` (a per-level override written by
    ``coarsen`` under ``chunk_scale_factor > 1``) their chunk coords index
    grids of different cell sizes and cannot be differenced directly.

    Returns all-ones for both when the metadata cannot be read, or when
    the target level does not exist yet.  All-ones is exactly right for a
    default pyramid (``chunk_scale_factor=1``), where every level inherits
    the root ``chunk_shape`` — and it makes the anchor a no-op, which is
    the pre-merge behaviour.  It is *wrong* for a scaled pyramid, so this
    mirrors :func:`_derive_native_config`'s best-effort contract: callers
    on a scaled store must be reachable from real metadata.
    """
    ndim_fallback: tuple[int, ...]
    try:
        import zarr

        from zarr_vectors.core.metadata import (
            LevelMetadata,
            chunk_scale_factor,
        )
        from zarr_vectors.core.store import read_root_metadata

        store = level_group._zarr.store
        root_zarr = zarr.open_group(store, path="/", mode="r")
        root_group = type(level_group)._from_zarr(root_zarr)
        root_meta = read_root_metadata(root_group)
        ndim_fallback = tuple(1 for _ in root_meta.chunk_shape)

        src_meta = LevelMetadata.from_dict(level_group.attrs.to_dict())
        scale_src = chunk_scale_factor(root_meta, src_meta)
        if delta == 0:
            return scale_src, scale_src

        from zarr_vectors.core.store import (
            get_resolution_level,
            read_level_metadata,
        )

        target_idx = int(src_meta.level) + int(delta)
        try:
            get_resolution_level(root_group, target_idx)
            trg_meta = read_level_metadata(root_group, target_idx)
        except Exception:
            # Target level not written yet (pyramid still being built):
            # fall back to the source's own scale, which is correct
            # whenever the pyramid is uniform.
            return scale_src, scale_src
        return scale_src, chunk_scale_factor(root_meta, trg_meta)
    except Exception:
        try:
            ndim_fallback
        except NameError:
            ndim_fallback = (1, 1, 1)
        return ndim_fallback, ndim_fallback


def _ensure_array_dir(
    level_group: FsGroup,
    array_name: str,
    *,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Ensure an array node exists within a level group.

    For per-spatial-chunk arrays (vertices, fragments,
    ``links/<delta>/<offsets>``, attribute arrays) this allocates a single
    multidim vlen-bytes Zarr array (one cell per spatial chunk) — the
    default single-array layout.  The grid comes from the active
    :func:`open_write_session` when one is open, else it is derived from
    the store's metadata (so coarsen / rechunk / edit paths get
    single-array too).

    Non-per-chunk names (``object_index``, the ``links/<delta>`` family
    group, attribute *namespace* groups) become plain Zarr groups.  Which
    is which is decided by :func:`_is_per_chunk_array`, and it is
    depth-aware: ``links/0`` is a group, ``links/0/0.0.+1`` an array.
    Under an
    active :meth:`Group.batched_writes`, the sync ``require_group``
    round-trip is skipped — the metadata flush PUTs the parent
    ``zarr.json`` directly.

    ``attributes`` is the array's own metadata block.  Passing it here
    rather than calling :meth:`Group.write_array_meta` afterwards is what
    keeps allocation to a single store write: the attrs ride along with
    the ``create_array`` that writes ``zarr.json`` anyway, instead of
    rewriting the whole object a second time.  A writer that allocates one
    array per offsets segment pays that saving per segment.  When the
    array already exists it is *not* recreated, so the metadata is applied
    with a normal write — same end state either way.
    """
    explicit_cfg = level_group._native_sharded_config
    cfg = explicit_cfg
    if cfg is None and _is_per_chunk_array(array_name):
        # No explicit session — derive the grid from metadata so this
        # write path still produces a single vlen array.
        cfg = _derive_native_config(level_group)
    if cfg is not None and _is_per_chunk_array(array_name):
        if level_group.standalone_array_exists(array_name):
            # An array is already here.  Under the derived path (no
            # explicit session) always reuse it — recreating would drop
            # existing cells (e.g. an edit adding a chunk).  Under an
            # explicit session, reuse only when the on-disk layout
            # already matches the requested grid/sharding; otherwise
            # recreate to honor the writer's intent (e.g. a store whose
            # ``vertices`` was warm-created unsharded but is now being
            # written with ``shard_shape=``).
            if explicit_cfg is None or _array_matches_layout(
                level_group, array_name, explicit_cfg
            ):
                # Reused as-is: the attrs can't ride a create_array that
                # isn't happening, so apply them directly.
                if attributes:
                    level_group.write_array_meta(array_name, attributes)
                return
        # Honor the session compressor so the array's codec pipeline
        # matches the batched default (no compression) unless the
        # caller asked for zstd/blosc.
        compressors = None
        if level_group._active_codecs is not None:
            from zarr_vectors.encoding.compression import (
                codecs_for_create_array,
            )
            compressors = codecs_for_create_array(
                level_group._active_codecs
            )
        level_group.create_sharded_chunk_array(
            array_name,
            grid_shape=cfg["grid_shape"],
            shard_shape=cfg["shard_shape"],
            origin=cfg.get("origin"),
            compressors=compressors,
            attributes=attributes,
        )
        return
    # A group node, not a chunk array: nothing to ride along with, so any
    # metadata is written normally rather than dropped.
    if attributes:
        level_group.write_array_meta(array_name, attributes)
        return
    if level_group._pending_array_metas is not None:
        return
    level_group.require_group(array_name)


def _array_matches_layout(
    level_group: FsGroup, array_name: str, cfg: dict[str, Any],
) -> bool:
    """Whether an existing single vlen array can be reused as-is under an
    explicit session, rather than recreated to match ``cfg``.

    An **empty** array (no cells written) is never a match: it is
    recreated so it picks up the session's full config — grid shape,
    sharding, *and* codec pipeline.  This is what lets a store whose
    ``vertices`` was warm-created (unsharded, uncompressed) by
    :func:`create_store` be rewritten with ``shard_shape=`` or
    ``compressor=``.  A non-empty array is reused only when its grid and
    sharded-ness already match (a legit second writer pass), so its data
    is preserved.
    """
    existing = level_group._sharded_chunk_array(array_name)
    if existing is None:
        return False
    if not existing.attrs.get("nonempty_chunks"):
        return False
    desired_sharded = cfg.get("shard_shape") is not None
    existing_sharded = getattr(existing, "shards", None) is not None
    return (
        tuple(existing.shape) == tuple(cfg["grid_shape"])
        and existing_sharded == desired_sharded
    )


def level_grid_layout(
    bounds: tuple[list[float], list[float]],
    chunk_shape: tuple[float, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ``(origin, grid_shape)`` for a level's single vlen arrays.

    :func:`zarr_vectors.spatial.chunking.assign_chunks` maps a position
    to the *absolute* coord ``floor(pos / chunk_shape)``.  When the data
    extends below the origin (negative positions), those coords are
    negative — which a 0-indexed Zarr array cannot hold directly.  We
    therefore anchor the array at ``origin = floor(min_corner /
    chunk_shape)`` (per axis) and store that offset on the array, so
    cell ``index = coord - origin`` and the array only needs to span
    ``floor(max_corner / chunk_shape) - origin + 1`` cells.  Using
    ``ceil(max_corner / chunk_shape)`` would also be one short whenever a
    point lands exactly on a chunk boundary — ``floor(..) + 1`` is exact.
    """
    min_corner = np.asarray(bounds[0], dtype=np.float64)
    max_corner = np.asarray(bounds[1], dtype=np.float64)
    cs = np.asarray(chunk_shape, dtype=np.float64)
    origin = tuple(int(np.floor(mn / c)) for mn, c in zip(min_corner, cs))
    grid_shape = tuple(
        max(1, int(np.floor(mx / c)) - o + 1)
        for mx, c, o in zip(max_corner, cs, origin)
    )
    return origin, grid_shape


@contextmanager
def open_write_session(
    level_group: FsGroup,
    *,
    compressor: Any = None,
    shard_shape: int | tuple[int, ...] | None = None,
    bounds: tuple[list[float], list[float]] | None = None,
    chunk_shape: tuple[float, ...] | None = None,
    bin_count: int | None = None,
):
    """Open the write session used by every type writer.

    Always activates the single-array layout: each per-spatial-chunk
    array (vertices, vertex_fragments, links/<delta>, link_fragments,
    vertex_attributes/<name>, …) is one multidim vlen-bytes Zarr array
    whose cells are spatial chunks (files at ``<array>/c/i/j/k``), and a
    batched-write context flushes every cell of an array in one
    concurrent :meth:`~zarr.Array.set_coordinate_selection`.

    ``bounds`` and ``chunk_shape`` are required so the chunk-grid extent
    (and origin offset, for data with negative coords) can be sized
    upfront — that's the shape of each vlen array.

    Args:
        level_group: The resolution-level group.
        compressor: Forwarded to :meth:`Group.batched_writes`; also
            determines each array's on-disk codec pipeline (``None`` →
            no compression, the default).
        shard_shape: ``None`` (default) → unsharded, one storage object
            per spatial chunk.  An int (broadcast to every axis) or a
            per-axis tuple wraps the cells in the ``sharding_indexed``
            codec so many chunks pack into one storage object.
        bounds: ``(min_corner, max_corner)`` for the level — used to
            compute the chunk grid extent and origin.
        chunk_shape: Physical chunk size per axis — paired with
            ``bounds`` for the grid extent.
        bin_count: When the writer chunks by an attribute, every chunk
            key gains a leading attr-bin axis; pass the number of bins
            so the vlen array grid gets that extra leading axis
            (origin 0, extent ``bin_count``).
    """
    from contextlib import ExitStack

    if bounds is None or chunk_shape is None:
        raise ArrayError(
            "open_write_session requires `bounds` and `chunk_shape` so "
            "the per-axis chunk-grid extent can be computed"
        )

    origin, grid_shape = level_grid_layout(bounds, chunk_shape)
    if bin_count is not None:
        # chunk_by_attribute prepends a 0-based bin axis to every key.
        origin = (0, *origin)
        grid_shape = (int(bin_count), *grid_shape)

    ss: tuple[int, ...] | None
    if shard_shape is None:
        ss = None
    elif isinstance(shard_shape, int):
        ss = (int(shard_shape),) * len(grid_shape)
    else:
        ss = tuple(int(x) for x in shard_shape)

    stack = ExitStack()
    with stack:
        stack.enter_context(level_group.batched_writes(compressor=compressor))
        stack.enter_context(
            level_group.native_sharded_arrays(ss, grid_shape, origin=origin)
        )
        yield


def _is_per_chunk_array(name: str) -> bool:
    """Whether ``name`` is a per-spatial-chunk array.

    These become a single vlen-bytes Zarr array whose shape is the
    level's chunk grid — one cell per spatial chunk (file at
    ``<name>/c/i/j/k``).  Object-level arrays (``object_index``,
    ``object_attributes/...``, ``groups``, ``group_attributes/...``) are
    plain single arrays that do **not** use this grid layout.

    The test is **depth-aware**, not a prefix match, because the link
    families nest a group above their arrays: ``links/<delta>`` is a
    *group* whose children are one array per relative-offset segment
    (``links/<delta>/<offsets>``).  A prefix test would match both and
    :func:`_ensure_array_dir` would clobber the group with an array.

        vertices                                    array
        links/0                                     GROUP
        links/0/0.0.+1                              array
        link_attributes/weight/0                    GROUP
        link_attributes/weight/0/0.0.+1             array

    Note this predicate does double duty: :mod:`zarr_vectors.sharding.io`
    imports it to decide what ``shard_store`` may migrate.  Both link
    families are rank-D grids now, so they shard like any other.
    """
    if name in {VERTICES, VERTEX_FRAGMENTS, LINK_FRAGMENTS}:
        return True
    parts = name.split("/")
    if len(parts) == 2 and parts[0] in (VERTEX_ATTRIBUTES, FRAGMENT_ATTRIBUTES):
        return True
    # links/<delta>/<offsets>
    if len(parts) == 3 and parts[0] == LINKS:
        return True
    # link_attributes/<name>/<delta>/<offsets>
    if len(parts) == 4 and parts[0] == LINK_ATTRIBUTES:
        return True
    return False


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


def links_has_perm(
    offsets: Sequence[ChunkCoords],
    *,
    delta: int,
    directed: bool,
    store: str,
) -> bool:
    """Whether ``links/<delta>/<offsets>/`` rows carry a ``perm_idx`` column.

    ``perm_idx`` exists only to undo a canonical sort, so it is needed
    exactly when :func:`zarr_vectors.spatial.boundary._cell_placements`
    may return a non-identity permutation.  Storing it unconditionally
    would add 8 bytes to every intra-chunk link — the overwhelming
    majority of rows — for a value that is always 0.

    Keep this in lockstep with ``_cell_placements``: it is the single
    definition the writer and reader both consult to agree on the record
    width (``L`` columns, or ``1 + L`` when this returns True).
    """
    if is_intra(offsets):
        # Every endpoint in the source chunk: identity placement, input
        # order preserved.  This is what keeps the all-zero-offsets array
        # byte-identical to the pre-merge ``links/<delta>/``.
        return False
    if delta != 0:
        # Cross-level: endpoints are distinguished by level, so the source
        # is always input endpoint 0 and the placement is the identity.
        return False
    if store == "duplicate":
        # Each copy leads with a different endpoint, so sigma varies.
        return True
    return not directed


def create_links_family(
    level_group: FsGroup,
    *,
    delta: int = 0,
    link_width: int = 2,
    sid_ndim: int | None = None,
    directed: bool = False,
    store: str = "canonical",
) -> None:
    """Stamp the ``links/<delta>/`` family group's policy, nothing else.

    ``directed`` / ``store`` / ``link_width`` / ``sid_ndim`` are
    family-wide: every offsets array under the delta decodes against them,
    which is why they live on the group and why
    :func:`write_links` refuses to flip them under surviving siblings.

    Use this when the policy must exist before any offsets array does —
    principally the decentralized flow, where a coordinator fixes the
    policy up front and workers then create only the offsets arrays their
    own records land in.  :func:`create_links_array` would otherwise force
    a family to materialise an arbitrary (usually intra) array purely to
    carry the group stamp, which a cross-only family does not want.

    Idempotent, and merges rather than overwrites: an existing group keeps
    any counts :func:`finalize_links` left on it.

    Raises:
        ArrayError: If ``store`` is not ``"canonical"`` or ``"duplicate"``,
            or if the family already exists with a conflicting policy —
            silently re-stamping would strand every array already written
            under the old one.
    """
    if store not in ("canonical", "duplicate"):
        raise ArrayError(
            f"store must be 'canonical' or 'duplicate', got {store!r}"
        )
    family = links_group_path(delta)
    existing = (
        level_group.read_array_meta(family) or {}
        if level_group.array_exists(family) else {}
    )
    if existing:
        _check_link_family_policy(
            existing, delta=delta, link_width=link_width, sid_ndim=sid_ndim,
            directed=directed, store=store, action="re-stamp",
        )
    meta = dict(existing)
    meta.update({
        "zv_array": "links_family",
        "level_delta": int(delta),
        "link_width": int(link_width),
        "directed": bool(directed),
        "store": str(store),
    })
    if sid_ndim is not None:
        meta["sid_ndim"] = int(sid_ndim)
    _ensure_array_dir(level_group, family)
    level_group.write_array_meta(family, meta)


def create_links_array(
    level_group: FsGroup,
    link_width: int,
    dtype: str = "int64",
    *,
    delta: int = 0,
    sid_ndim: int | None = None,
    offsets: Sequence[ChunkCoords] | None = None,
    directed: bool = False,
    store: str = "canonical",
    exist_ok: bool = True,
) -> None:
    """Create a ``links/<delta>/<offsets>/`` array and its family group.

    To stamp the family policy *without* materialising any offsets array
    — a cross-only family, or a coordinator fixing policy before workers
    run — use :func:`create_links_family` instead.

    Each ``<delta>`` segment is a group; its children are one rank-D vlen
    array per distinct relative-offset segment, each cell holding the
    records whose **source** chunk is that cell.  ``offsets`` all-zero is
    the intra-chunk array (what was a standalone ``links/<delta>/`` array
    before the offset layout); non-zero offsets hold records whose other
    endpoints sit that far away — what used to be ``cross_chunk_links/``.

    Family-wide policy (``directed``, ``store``, ``sid_ndim``,
    ``link_width``) is stamped on the ``<delta>`` **group**, since several
    arrays now live under it and must agree.  Per-array meta carries only
    what is needed to decode that array's own cells.

    Args:
        level_group: The resolution level FsGroup.
        link_width: Number of vertex indices per link entry (L).
            1 for skeleton parents, 2 for edges, 3 for triangle faces.
        dtype: Integer dtype.
        delta: Level delta; see :mod:`zarr_vectors.core.paths`.
        sid_ndim: Spatial index dims.  Required when ``offsets`` is None
            so the intra segment can be derived.
        offsets: The ``link_width - 1`` relative offsets naming this
            array.  Defaults to the all-zero (intra-chunk) offsets.
        directed: Endpoint order is data; see
            :func:`zarr_vectors.spatial.boundary._cell_placements`.
        store: ``"canonical"`` or ``"duplicate"``.
        exist_ok: When True (default), no-op if the array already exists.
            When False, raise :class:`ArrayError` on conflict.
    """
    if store not in ("canonical", "duplicate"):
        raise ArrayError(
            f"store must be 'canonical' or 'duplicate', got {store!r}"
        )
    if offsets is None:
        if sid_ndim is None:
            raise ArrayError(
                "create_links_array requires sid_ndim when offsets is None "
                "(needed to derive the intra-chunk offsets segment)"
            )
        offsets = intra_offsets(sid_ndim, link_width)
    if sid_ndim is None and offsets:
        sid_ndim = len(offsets[0])

    family = links_group_path(delta)
    full_name = links_path(delta, offsets)
    if _short_circuit_existing(level_group, full_name, exist_ok):
        return
    # Hand the array's metadata to the allocation itself: one offsets
    # array per distinct offset means this runs once per segment, and
    # writing the meta separately would double the store writes for it.
    _ensure_array_dir(level_group, full_name, attributes={
        "zv_array": "links",
        "dtype": dtype,
        "offsets": [list(int(c) for c in o) for o in offsets],
        "has_perm": links_has_perm(
            offsets, delta=delta, directed=directed, store=store,
        ),
        "link_width": link_width,
        "level_delta": int(delta),
    })
    family_meta: dict[str, Any] = {
        "zv_array": "links_family",
        "level_delta": int(delta),
        "link_width": int(link_width),
        "directed": bool(directed),
        "store": str(store),
    }
    if sid_ndim is not None:
        family_meta["sid_ndim"] = int(sid_ndim)
    level_group.write_array_meta(family, family_meta)


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


def create_link_attributes_array(
    level_group: FsGroup,
    name: str,
    dtype: str = "float32",
    *,
    delta: int = 0,
    sid_ndim: int | None = None,
    link_width: int = 2,
    offsets: Sequence[ChunkCoords] | None = None,
    exist_ok: bool = True,
) -> None:
    """Create a ``link_attributes/<name>/<delta>/<offsets>/`` array.

    Mirrors the matching ``links/<delta>/<offsets>/`` array exactly —
    same delta, same offsets segment, same cells, same per-cell row
    order — so attribute rows align 1:1 with link records without
    storing a row id.  There is no separate cross-chunk attribute
    family: an intra-chunk link's attributes live under the all-zero
    offsets segment, just like the links themselves.

    ``exist_ok=True`` (default) makes the call idempotent; pass
    ``exist_ok=False`` to raise :class:`ArrayError` on conflict.
    """
    if offsets is None:
        if sid_ndim is None:
            raise ArrayError(
                "create_link_attributes_array requires sid_ndim when "
                "offsets is None (needed to derive the intra segment)"
            )
        offsets = intra_offsets(sid_ndim, link_width)
    full_name = link_attributes_path(name, delta, offsets)
    if _short_circuit_existing(level_group, full_name, exist_ok):
        return
    _ensure_array_dir(level_group, full_name)
    level_group.write_array_meta(link_attributes_group_path(name, delta), {
        "zv_array": "link_attribute_family",
        "name": name,
        "level_delta": int(delta),
    })
    level_group.write_array_meta(full_name, {
        "zv_array": "link_attribute",
        "name": name,
        "dtype": dtype,
        "offsets": [list(int(c) for c in o) for o in offsets],
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
    *,
    record_presence: bool = True,
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
    # record_presence: see write_chunk_attributes.  nonempty_chunks is
    # array-wide, so stamping it rewrites the shared zarr.json once per
    # cell; concurrent writers over disjoint cells still collide there.
    level_group.write_bytes(
        VERTICES, key, raw_bytes, record_presence=record_presence,
    )

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
        record_presence=record_presence,
    )
    return vertex_byte_offsets


def _infer_link_width(link_groups: Sequence[npt.NDArray[np.integer]]) -> int:
    """Best-effort ``L`` from a list of ``(M_k, L)`` link groups."""
    for g in link_groups:
        arr = np.asarray(g)
        if arr.ndim == 2:
            return int(arr.shape[1])
    return 2


def write_chunk_links(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
    link_groups: list[npt.NDArray[np.integer]],
    dtype: np.dtype | str = np.int64,
    *,
    delta: int = 0,
    offsets: Sequence[ChunkCoords] | None = None,
    link_width: int | None = None,
    record_presence: bool = True,
) -> npt.NDArray[np.int64]:
    """Write link rows to one cell of ``links/<delta>/<offsets>/``.

    ``chunk_coords`` is the **source** chunk — the cell — and ``offsets``
    names which array under ``links/<delta>/`` receives them.  Defaults to
    the all-zero (intra-chunk) offsets, which is the array that holds
    records whose endpoints all share the source chunk.

    This is the low-level per-cell writer: ``link_groups`` rows are
    written as given.  Choosing the source chunk, computing the offsets
    and prepending any ``perm_idx`` column is the caller's job — see
    :func:`zarr_vectors.spatial.boundary.partition_records_by_offset`.

    Two encodings, selected by ``delta == 0 and is_intra(offsets)``:

    - **flat + ``link_fragments/`` sidecar** for intra-chunk links at
      delta 0.  Readers derive per-group byte offsets from the cumulative
      group sizes (see :func:`read_chunk_links`); link groups need not be
      1:1 with the chunk's vertex fragments.
    - **inline self-describing ragged blob** otherwise — cross-offset
      records at any delta, and every cross-level record.  These carry no
      fragment sidecar.

    That condition reproduces both pre-merge layouts exactly: the intra
    array is byte-identical to the old ``links/<delta>/``, and every other
    offset array matches the old ``cross_chunk_links/<delta>/`` cells.

    Args:
        level_group: Resolution level group.
        chunk_coords: **Source** chunk coordinates (the array cell).
        link_groups: List of arrays, each ``(M_k, L)``.
        dtype: Integer dtype.
        delta: Level delta; see :mod:`zarr_vectors.core.paths`.
        offsets: Relative offsets naming the target array.  ``None``
            (default) means the all-zero intra-chunk offsets.
        link_width: Only used to derive the intra offsets segment when
            ``offsets`` is None and ``link_groups`` is empty.
        record_presence: Whether to stamp this cell into the array's
            ``nonempty_chunks`` manifest.  Decentralized workers pass
            False: the manifest is array-wide state, so stamping it is a
            read-modify-write that two workers racing on *disjoint* cells
            still lose keys to.  They leave it to a coordinator's
            :func:`finalize_links`, which rebuilds it from the store
            listing once every worker has finished.

    Returns:
        ``(K,)`` int64 array of link byte offsets.
    """
    dtype = np.dtype(dtype)
    key = _chunk_key(chunk_coords)
    if offsets is None:
        if link_width is None:
            link_width = _infer_link_width(link_groups)
        offsets = intra_offsets(len(chunk_coords), link_width)
    full_name = links_path(delta, offsets)

    # NOTE: link groups are no longer required to be 1:1 with the chunk's
    # vertex fragments. BRIDGE stores streamlines as vertex-fragments and the
    # node graph as link-fragments, so a chunk legitimately has many vertex
    # fragments (Core-1 range + polyline twins) but few link groups (the node
    # graph). Readers derive per-group link ranges from ``link_fragments/``
    # (not from ``vertex_fragments``), so no write-time 1:1 guard is needed.

    if delta == 0 and is_intra(offsets):
        # v0.6 intra-level: flat concatenated link data + sibling
        # link_fragments/ describing per-group row ranges.
        data_bytes, link_byte_offsets = encode_ragged_ints(link_groups, dtype)
        level_group.write_bytes(
            full_name, key, data_bytes, record_presence=record_presence,
        )

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
        # legacy per-chunk-array group.
        #
        # ``link_fragments/<chunk>`` is keyed by chunk ALONE — it carries
        # no delta and no offsets segment — and this write is an
        # unconditional replace.  It is therefore only correct because the
        # enclosing branch admits exactly one array per chunk: the intra
        # one at delta 0.  If a non-intra offset ever reached here it would
        # silently clobber the intra array's fragment index.  Keep the
        # branch condition and this write together.
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

    # Cross-offset and/or cross-level: inline self-describing ragged blob,
    # no fragment sidecar.  Matches the pre-merge ``cross_chunk_links/``
    # cell encoding and the old ``links/<delta!=0>/`` layout.
    blob = encode_ragged_blob(link_groups, dtype)
    level_group.write_bytes(full_name, key, blob, record_presence=record_presence)
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
    *,
    record_presence: bool = True,
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
        record_presence: Whether to stamp this cell into the array's
            ``nonempty_chunks`` manifest.  Pass False when fanning
            per-chunk writes out concurrently, then rebuild the manifest
            once with :meth:`Group.derive_nonempty_chunks`.
    """
    dtype = np.dtype(dtype)
    key = _chunk_key(chunk_coords)
    full_name = f"{VERTEX_ATTRIBUTES}/{attr_name}"
    # The array must already exist: this is a per-cell write, and callers
    # fan it out across chunks (sometimes concurrently), so allocating
    # here would both race and re-create the array on every chunk —
    # ``_ensure_array_dir`` drops and rebuilds an array whose
    # ``nonempty_chunks`` is still empty.  Allocate once up front instead.
    #
    # ``record_presence=False`` is for exactly that concurrent fan-out:
    # ``nonempty_chunks`` lives on the array, so stamping it rewrites the
    # shared ``zarr.json`` once per cell.  Two tasks writing *disjoint*
    # cells still collide there — on Windows as a rename race over
    # ``zarr.json``, elsewhere as a silently dropped key.  Callers that
    # fan out pass False and rebuild once via
    # :meth:`Group.derive_nonempty_chunks`.
    raw_bytes, _ = encode_ragged_floats(attr_groups, dtype)
    level_group.write_bytes(
        full_name, key, raw_bytes, record_presence=record_presence,
    )


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
    offsets: Sequence[ChunkCoords] | None = None,
    link_width: int = 2,
) -> None:
    """Write one cell of ``link_attributes/<name>/<delta>/<offsets>/``.

    Mirrors :func:`write_chunk_links` cell-for-cell: ``chunk_coords`` is
    the **source** chunk and ``offsets`` names the same array under
    ``link_attributes/<name>/<delta>/`` that ``links/<delta>/`` receives
    the parallel records in.  Rows are written as given, in the same
    order — that positional alignment is the whole contract, so there is
    no row id.

    Unlike the link cell there is no encoding branch: the payload is
    always a flat dense blob of ``sum(M_k)`` rows.  Per-group boundaries
    are recovered at read time from the parallel link cell — the
    ``link_fragments/<chunk>`` sidecar for the intra array at delta 0,
    the links blob's own inline header otherwise (see
    :func:`read_chunk_link_attributes`).

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name (e.g. ``"weight"``).
        chunk_coords: **Source** chunk coordinates (the array cell).
        attr_groups: List of arrays, each ``(M_k,)`` or ``(M_k, C)``,
            aligned with the link groups in the parallel links cell.
        dtype: Numpy dtype.
        delta: Level delta; see :mod:`zarr_vectors.core.paths`.
        offsets: Relative offsets naming the target array.  ``None``
            (default) means the all-zero intra-chunk offsets.
        link_width: Only used to derive the intra offsets segment when
            ``offsets`` is None.
    """
    dtype = np.dtype(dtype)
    key = _chunk_key(chunk_coords)
    if offsets is None:
        offsets = intra_offsets(len(chunk_coords), link_width)
    full_name = link_attributes_path(attr_name, delta, offsets)
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
class LinkPartition:
    """Per-input-record placement map returned by :func:`write_links`.

    Pass to :func:`write_link_attributes` to align per-record attribute
    data with the ``(offsets, cell)`` layout the link writer chose.

    Attributes:
        cell_indices: Maps ``(offsets_segment, source_chunk)`` → list of
            indices into the input ``links`` list.  Within each bucket
            indices preserve input ordering, so attribute data passed in
            input order partitions deterministically.  Under
            ``store="duplicate"`` one input index appears under several
            buckets — that is what makes the parallel attribute family
            replicate identically.
        num_links: **Logical** record count after the write — one per
            input record, regardless of how many physical copies
            ``store="duplicate"`` filed.  For ``append`` mode this
            includes pre-existing records.  This is the count
            :func:`write_link_attributes` validates ``attr_data``
            against, since ``attr_data`` is in input order.
        num_physical_records: On-disk row count after the write.  Equal
            to ``num_links`` for a ``canonical`` family; larger for a
            ``duplicate`` one.
        first_new: The input-order row index of the first newly-appended
            record.  ``0`` for ``replace`` mode, the pre-existing logical
            count for ``append`` mode.
    """

    cell_indices: dict[tuple[str, ChunkCoords], list[int]]
    num_links: int
    num_physical_records: int = 0
    first_new: int = 0

    def __int__(self) -> int:
        # Lets callers that did ``first_new = int(write_links(...))``
        # keep working.
        return self.first_new


def _normalise_link_records(
    links: list[list[tuple[ChunkCoords, int]]] | list[CrossChunkLink],
    link_width: int | None,
    sid_ndim: int,
    delta: int,
) -> tuple[list[list[tuple[ChunkCoords, int]]], int]:
    """Normalise record input to list-of-lists and resolve link_width.

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
                f"links/{format_delta(delta)}: record arity "
                f"{len(rec)} != link_width {link_width}"
            )
        for chunk, _vi in rec:
            if len(chunk) != sid_ndim:
                raise ArrayError(
                    f"chunk coords arity mismatch in links/"
                    f"{format_delta(delta)}: sid_ndim={sid_ndim}, "
                    f"got len(chunk)={len(chunk)}"
                )
    return normalised, link_width


def _pad_scale(scale: Sequence[int], sid_ndim: int) -> tuple[int, ...]:
    """Fit a per-axis chunk-scale factor to ``sid_ndim`` axes.

    :func:`_derive_level_scales` derives its rank from the root
    ``chunk_shape``, i.e. the *spatial* rank.  Writers that chunk by an
    attribute prepend a bin axis to every chunk key (see
    :func:`open_write_session`'s ``bin_count``), so their ``sid_ndim`` is
    one larger.  That axis indexes bins, not space, and therefore never
    rescales across pyramid levels — pad with 1s at the front, which
    makes :func:`~zarr_vectors.spatial.boundary.anchor_chunk` a no-op on
    it.  Without this the rank check in ``partition_records_by_offset``
    rejects every attribute-binned store.
    """
    scale = tuple(int(s) for s in scale)
    if len(scale) == sid_ndim:
        return scale
    if len(scale) < sid_ndim:
        return (1,) * (sid_ndim - len(scale)) + scale
    return scale[len(scale) - sid_ndim:]


def _link_scales(
    level_group: FsGroup, delta: int, sid_ndim: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(scale_src, scale_trg)`` fitted to ``sid_ndim`` — see
    :func:`_derive_level_scales` and :func:`_pad_scale`."""
    scale_src, scale_trg = _derive_level_scales(level_group, delta)
    return _pad_scale(scale_src, sid_ndim), _pad_scale(scale_trg, sid_ndim)


def _partition_links(
    level_group: FsGroup,
    records: Sequence[Sequence[tuple[ChunkCoords, int]]],
    link_width: int,
    sid_ndim: int,
    *,
    delta: int,
    directed: bool,
    store: str,
) -> dict[tuple[str, ChunkCoords], list[tuple[list[int], int, int]]]:
    """Bucket records into ``(offsets_segment, source_chunk)`` cells.

    The single choke point deciding where a record lands.  ``delta != 0``
    forces ``cross_level``, which keeps the source at the owning level so
    the anchor's ``scale_src`` is that level's — see
    :func:`~zarr_vectors.spatial.boundary.partition_records_by_offset`.
    """
    from zarr_vectors.spatial.boundary import partition_records_by_offset

    scale_src, scale_trg = _link_scales(level_group, delta, sid_ndim)
    return partition_records_by_offset(
        records, link_width, sid_ndim,
        scale_src=scale_src, scale_trg=scale_trg,
        directed=directed, store=store, cross_level=(delta != 0),
    )


def _check_link_family_policy(
    fam_meta: dict[str, Any],
    *,
    delta: int,
    link_width: int,
    sid_ndim: int,
    directed: bool,
    store: str,
    action: str,
) -> None:
    """Reject a write whose policy contradicts the existing family.

    ``directed`` / ``store`` / ``link_width`` / ``sid_ndim`` live on the
    ``links/<delta>/`` **group** precisely because several offset arrays
    sit under it and must agree — a per-array override would let two
    arrays in one family decode differently.  Absent keys mean "no
    opinion yet" and pass.
    """
    fam = f"links/{format_delta(delta)}"
    checks = (
        ("link_width", int(fam_meta.get("link_width", link_width)), link_width),
        ("sid_ndim", int(fam_meta.get("sid_ndim", sid_ndim)), sid_ndim),
        ("directed", bool(fam_meta.get("directed", directed)), bool(directed)),
        ("store", str(fam_meta.get("store", store)), str(store)),
    )
    for field, existing, requested in checks:
        if existing != requested:
            raise ArrayError(
                f"{fam}: cannot {action} it with {field}={requested!r} onto an "
                f"existing family with {field}={existing!r} — this policy is "
                f"family-wide (every offsets array under the delta shares it). "
                f"Drop the family first to change it."
            )


def _pack_link_rows(
    entries: Sequence[tuple[list[int], int, int]],
    *,
    has_perm: bool,
    link_width: int,
    dtype: np.dtype,
) -> npt.NDArray[np.integer]:
    """Build the ``(M, W)`` row block for one cell.

    ``W`` is ``link_width`` normally and ``1 + link_width`` when
    ``has_perm`` — see :func:`links_has_perm`, the single definition
    writer and reader both consult.
    """
    width = link_width + (1 if has_perm else 0)
    out = np.empty((len(entries), width), dtype=dtype)
    for r, (vi, perm_idx, _input_idx) in enumerate(entries):
        if has_perm:
            out[r, 0] = perm_idx
            out[r, 1:] = vi
        else:
            out[r, :] = vi
    return out


def write_links(
    level_group: FsGroup,
    links: list[list[tuple[ChunkCoords, int]]] | list[CrossChunkLink],
    sid_ndim: int,
    *,
    delta: int = 0,
    link_width: int | None = None,
    dtype: np.dtype | str = np.int64,
    mode: Literal["replace", "append"] = "replace",
    directed: bool = False,
    store: Literal["canonical", "duplicate"] = "canonical",
) -> LinkPartition:
    """Write whole link records into ``links/<delta>/<offsets>/``.

    The whole-family counterpart to :func:`write_chunk_links`: it takes
    records in *global* ``(chunk_coords, vertex_idx)`` form, decides where
    each one belongs, and fans them out across the offset arrays.  There
    is no separate cross-chunk family — a record whose endpoints all
    share one chunk lands in the all-zero-offsets array, one whose
    endpoints straddle chunks lands in the array named by the offsets
    between them.

    Each record is ``link_width`` endpoints.  ``link_width=2`` (the
    default) encodes an edge, ``3`` a triangle face, ``1`` a parent→child
    reference used by pyramid metanode drill-down.  Records may be passed
    as legacy 2-tuples or as endpoint lists.

    For ``delta != 0``, endpoint 0 is at the owning resolution level and
    endpoints k>0 are at ``this_level + delta``; the source is always
    endpoint 0 so offsets anchor against the owning level's chunk grid
    (see :func:`~zarr_vectors.spatial.boundary.anchor_chunk`).

    Args:
        level_group: Resolution level group.
        links: List of records; each a list of ``(chunk_coords,
            vertex_idx)`` tuples of length ``link_width``.
        sid_ndim: Number of spatial index dimensions.
        delta: Level delta; see :mod:`zarr_vectors.core.paths`.
        link_width: Endpoints per record.  Defaults to the arity of the
            first record.
        dtype: Integer dtype for the stored rows.
        mode: ``"replace"`` (default) rewrites, from scratch, exactly the
            offset arrays this call's records land in — see the warning
            below.  ``"append"`` adds the records as a new group in each
            target cell, leaving existing rows in place.
        directed: When ``True`` endpoint order is data (streamline
            predecessor→successor, skeleton child→parent): records are
            NOT canonical-sorted, so ``A→B`` and ``B→A`` file under
            opposite offsets (``0.0.+1`` vs ``0.0.-1``) and carry no
            ``perm_idx``.
        store: ``"canonical"`` (default) files each record once, under
            the lexicographically-positive offset.  ``"duplicate"`` files
            it once per distinct incident chunk, so every record incident
            to a chunk can be found by scanning that chunk's cell alone.
            Duplicated records are returned once per copy by
            :func:`read_links`; the parallel attribute family replicates
            identically via the returned partition.

    Returns:
        :class:`LinkPartition` describing where each input record landed.

    Warning:
        ``mode="replace"`` scopes its delete to the offset arrays this
        call's records actually target.  It deliberately does **not**
        clear the whole ``links/<delta>/`` family: intra-chunk links live
        under it now, so a family-wide wipe would silently destroy
        everything :func:`write_chunk_links` wrote.  The corollary is
        that an offset array populated by an *earlier* call that this
        call's records do not reach survives untouched — enumerate with
        :func:`list_link_offsets` and delete explicitly if you need a
        broader wipe.

    Concurrency:
        ``mode="append"`` is read-modify-write per cell — safe across
        disjoint cells, unsafe within a single cell.
    """
    if mode not in ("replace", "append"):
        raise ArrayError(
            f"mode must be 'replace' or 'append', got {mode!r}"
        )
    if store not in ("canonical", "duplicate"):
        raise ArrayError(
            f"store must be 'canonical' or 'duplicate', got {store!r}"
        )
    if not links:
        # Honour the empty-input fast-exit; emit an empty partition so
        # callers don't crash on attribute reflection.  NOTE this means
        # an empty `links` does NOT clear anything — callers wanting that
        # must delete the offset arrays themselves (list_link_offsets).
        return LinkPartition(
            cell_indices={}, num_links=0, num_physical_records=0, first_new=0,
        )

    dtype = np.dtype(dtype)
    normalised, link_width = _normalise_link_records(
        links, link_width, sid_ndim, delta,
    )
    family = links_group_path(delta)
    fam_meta = level_group.read_array_meta(family) or {}

    buckets = _partition_links(
        level_group, normalised, link_width, sid_ndim,
        delta=delta, directed=directed, store=store,
    )
    targeted: list[str] = sorted({seg for seg, _chunk in buckets})
    surviving = [
        seg for seg in list_link_offsets(level_group, delta)
        if seg not in targeted
    ]

    # An append never wipes, so it must agree with the family it joins.
    # A replace only has to agree when it leaves siblings behind: those
    # arrays keep decoding against the family group's policy, so letting
    # this call flip it would silently corrupt them.
    if fam_meta and (mode == "append" or surviving):
        _check_link_family_policy(
            fam_meta, delta=delta, link_width=link_width, sid_ndim=sid_ndim,
            directed=directed, store=store,
            action="append to" if mode == "append" else "replace part of",
        )

    if mode == "replace":
        first_new = 0
        existing_total = 0
        existing_physical = 0
        for seg in targeted:
            path = f"{family}/{seg}"
            if level_group.array_exists(path):
                level_group.delete_subtree(path)
    else:
        # Existing counts come from family meta in O(1); a family written
        # only by the decentralized per-cell writers has none until
        # ``finalize_links`` runs, so fall back to a scan.
        if "num_links" in fam_meta:
            existing_total = int(fam_meta["num_links"])
            existing_physical = int(
                fam_meta.get("num_physical_records", existing_total)
            )
        elif level_group._native_sharded_config is not None:
            # An open write session is mid-build: the scan below reads the
            # store, which does not see cells this session has queued but
            # not flushed, so it would count 0 and stamp that as the
            # family's total — and a later ``write_link_attributes``
            # validating ``len(attr_data) == num_links`` would then reject
            # correct data against a fabricated 0.  A family with no counts
            # and a session open has nothing durable to count; leave the
            # reconciliation to ``finalize_links`` after the session
            # closes, which is when the cells are actually readable.
            existing_total = 0
            existing_physical = 0
        else:
            counted = finalize_links(level_group, delta=delta)
            existing_total = counted.num_links
            existing_physical = counted.num_physical_records
        first_new = existing_total

    new_physical = 0
    for seg in targeted:
        offsets = parse_offsets(seg, sid_ndim=sid_ndim, link_width=link_width)
        create_links_array(
            level_group, link_width, dtype=str(dtype), delta=delta,
            sid_ndim=sid_ndim, offsets=offsets, directed=directed,
            store=store, exist_ok=True,
        )
        has_perm = links_has_perm(
            offsets, delta=delta, directed=directed, store=store,
        )
        for (bucket_seg, src_chunk), entries in buckets.items():
            if bucket_seg != seg:
                continue
            rows = _pack_link_rows(
                entries, has_perm=has_perm, link_width=link_width, dtype=dtype,
            )
            new_physical += rows.shape[0]
            groups: list[npt.NDArray] = []
            if mode == "append":
                groups = list(_decode_link_cell(
                    level_group, src_chunk, delta=delta, offsets=offsets,
                    dtype=dtype, width=rows.shape[1], default=[],
                ))
            groups.append(rows)
            # Route through the per-cell writer so the flat+sidecar vs
            # inline-blob choice has exactly one definition.
            write_chunk_links(
                level_group, src_chunk, groups, dtype,
                delta=delta, offsets=offsets, link_width=link_width,
            )

    if surviving:
        # This call did not own the whole family, so the totals it can
        # see are partial.  Recount from disk — the same scan finalize
        # does — rather than stamp a number that is wrong.
        counted = finalize_links(level_group, delta=delta)
        new_total = counted.num_links
        new_physical_total = counted.num_physical_records
    else:
        new_total = existing_total + len(normalised)
        new_physical_total = existing_physical + new_physical

    _stamp_link_family_meta(
        level_group, delta=delta, link_width=link_width, sid_ndim=sid_ndim,
        directed=directed, store=store, num_links=new_total,
        num_physical_records=new_physical_total,
    )

    return LinkPartition(
        cell_indices={k: [idx for _, _, idx in v] for k, v in buckets.items()},
        num_links=new_total,
        num_physical_records=new_physical_total,
        first_new=first_new,
    )


def _stamp_link_family_meta(
    level_group: FsGroup,
    *,
    delta: int,
    link_width: int,
    sid_ndim: int,
    directed: bool,
    store: str,
    num_links: int | None = None,
    num_physical_records: int | None = None,
) -> None:
    """Write the ``links/<delta>/`` group's family-wide ``.zattrs``.

    Policy (``link_width`` / ``sid_ndim`` / ``directed`` / ``store``)
    lives here rather than on the arrays because every offsets array
    under the delta must agree on it.  The counts are family-wide totals
    across all of them; the decentralized per-cell writers leave them
    absent for :func:`finalize_links` to fill in.
    """
    meta: dict[str, Any] = {
        "zv_array": "links_family",
        "level_delta": int(delta),
        "link_width": int(link_width),
        "sid_ndim": int(sid_ndim),
        "directed": bool(directed),
        "store": str(store),
    }
    if num_links is not None:
        meta["num_links"] = int(num_links)
    if num_physical_records is not None:
        meta["num_physical_records"] = int(num_physical_records)
    _ensure_array_dir(level_group, links_group_path(delta))
    level_group.write_array_meta(links_group_path(delta), meta)


def write_link_attributes(
    level_group: FsGroup,
    attr_name: str,
    attr_data: npt.NDArray,
    *,
    num_links: int,
    delta: int = 0,
    mode: Literal["replace", "append"] = "replace",
    partition: LinkPartition | None = None,
) -> None:
    """Write per-link attribute data parallel to ``links/<delta>/``.

    The whole-family counterpart to :func:`write_chunk_link_attributes`.
    Attribute rows are stored per ``(offsets, cell)``, one row per link
    record, in the same order the link writer used — that positional
    alignment is the whole contract.

    ``attr_data`` is in **input order** — the same ordering passed to the
    matching :func:`write_links` call, whose returned ``partition``
    supplies the mapping.  Without a ``partition`` the writer re-derives
    one by reading the link records back, which only recovers *on-disk*
    order, so ``attr_data`` must then already be in the enumeration order
    :func:`read_links` returns (each offsets segment sorted, then each
    cell sorted).

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name.
        attr_data: ``(num_links,)`` or ``(num_links, C)`` array in
            **input order**.  In ``mode="append"`` this is the NEW rows;
            the post-append total must equal ``num_links``.
        num_links: Expected post-write logical record count (matches
            :attr:`LinkPartition.num_links`).
        delta: Level delta.
        mode: ``"replace"`` (default) rewrites exactly the offset arrays
            the partition targets — same scoping rule, and same caveat,
            as :func:`write_links`.  ``"append"`` per-cell RMW.
        partition: :class:`LinkPartition` from the matching
            :func:`write_links` call.  Required for ``append`` and for
            any ``store="duplicate"`` family.

    Raises:
        ArrayError: If ``mode`` is invalid, if the row count of
            ``attr_data`` does not match ``num_links`` (replace) or the
            new-row count derived from ``partition`` (append), or if the
            appended row shape does not match an existing array.

    Concurrency:
        ``mode="append"`` is read-modify-write per cell — safe across
        disjoint cells, unsafe within a single cell.
    """
    if mode not in ("replace", "append"):
        raise ArrayError(
            f"mode must be 'replace' or 'append', got {mode!r}"
        )

    arr = np.ascontiguousarray(np.asarray(attr_data))
    fam_meta = level_group.read_array_meta(links_group_path(delta)) or {}

    # Resolve the partition.  Append mode requires it explicitly — only
    # the matching write_links call knows which cell each new attribute
    # row belongs to.  Replace mode can fall back to re-deriving from
    # disk (assumes ``attr_data`` is in on-disk enumeration order).
    if partition is None:
        if mode == "append":
            raise ArrayError(
                f"link_attributes[{attr_name}] append mode requires "
                f"partition=... from the matching write_links(mode='append') "
                f"call so the writer knows which cell each new row belongs to"
            )
        # A duplicate family fans one input record across several cells;
        # only the link writer's partition records that fan-out, so the
        # re-derived (physical, positional) partition would misalign
        # attribute rows.  Require the partition explicitly.
        if str(fam_meta.get("store", "canonical")) == "duplicate":
            raise ArrayError(
                f"link_attributes[{attr_name}]: store='duplicate' requires "
                f"partition=... from the matching write_links call so "
                f"replicated rows align"
            )
        partition = _derive_partition_from_links(level_group, delta)

    link_width = int(fam_meta.get("link_width", 2))
    targeted = sorted({seg for seg, _chunk in partition.cell_indices})

    if mode == "replace":
        if arr.shape[0] != num_links:
            raise ArrayError(
                f"link_attributes[{attr_name}] row count {arr.shape[0]} != "
                f"num_links {num_links} (delta={format_delta(delta)})"
            )
        if arr.shape[0] != partition.num_links:
            raise ArrayError(
                f"link_attributes[{attr_name}] attr rows {arr.shape[0]} != "
                f"link records {partition.num_links} derived from "
                f"links/{format_delta(delta)}"
            )
        # Scoped exactly like write_links: only the offset arrays this
        # partition targets, never the whole family.
        for seg in targeted:
            path = f"{link_attributes_group_path(attr_name, delta)}/{seg}"
            if level_group.array_exists(path):
                level_group.delete_subtree(path)
    else:
        # Append mode.  The post-append total record count must equal
        # num_links; arr.shape[0] is the new-row count.
        new_count = sum(len(v) for v in partition.cell_indices.values())
        if arr.shape[0] != new_count:
            raise ArrayError(
                f"link_attributes[{attr_name}] append: attr_data row count "
                f"{arr.shape[0]} != new link record count {new_count} "
                f"(delta={format_delta(delta)})"
            )
        if partition.num_links != num_links:
            raise ArrayError(
                f"link_attributes[{attr_name}] row count "
                f"{partition.num_links} != num_links {num_links} "
                f"(delta={format_delta(delta)})"
            )

    sid_ndim = int(fam_meta.get("sid_ndim", 0)) or None
    for seg in targeted:
        offsets = _parse_offsets_for_family(
            seg, sid_ndim=sid_ndim, link_width=link_width, partition=partition,
        )
        create_link_attributes_array(
            level_group, attr_name, dtype=str(arr.dtype), delta=delta,
            sid_ndim=sid_ndim, link_width=link_width, offsets=offsets,
            exist_ok=True,
        )
        full_name = link_attributes_path(attr_name, delta, offsets)
        for (bucket_seg, src_chunk), input_idxs in partition.cell_indices.items():
            if bucket_seg != seg:
                continue
            key = _chunk_key(src_chunk)
            new_rows = arr[np.asarray(input_idxs, dtype=np.int64)]
            if mode == "append" and level_group.chunk_exists(full_name, key):
                combined = np.concatenate(
                    [_read_attr_cell(level_group, full_name, key, arr), new_rows],
                    axis=0,
                )
            else:
                combined = new_rows
            level_group.write_bytes(
                full_name, key, np.ascontiguousarray(combined).tobytes(),
            )
        # ``row_shape`` (the tail dims per row, ``()`` for 1-D) lets the
        # reader reconstruct shape from a bare byte blob.
        level_group.write_array_meta(full_name, {
            "zv_array": "link_attribute",
            "name": attr_name,
            "dtype": str(arr.dtype),
            "row_shape": list(arr.shape[1:]),
            "offsets": [list(int(c) for c in o) for o in offsets],
            "level_delta": int(delta),
        })

    _ensure_array_dir(level_group, link_attributes_group_path(attr_name, delta))
    level_group.write_array_meta(link_attributes_group_path(attr_name, delta), {
        "zv_array": "link_attribute_family",
        "name": attr_name,
        "level_delta": int(delta),
        "num_links": int(num_links),
    })


def _read_attr_cell(
    level_group: FsGroup, full_name: str, key: str, like: npt.NDArray,
) -> npt.NDArray:
    """Decode one attribute cell as ``(M, *row_shape)``, matching ``like``.

    Row shape comes from the array's own meta and must agree with the
    incoming rows — an append that changes the tail dims would make the
    blob undecodable, so it raises instead.
    """
    meta = level_group.read_array_meta(full_name) or {}
    dtype = np.dtype(meta["dtype"]) if "dtype" in meta else like.dtype
    tail = tuple(meta.get("row_shape", like.shape[1:]))
    if tail != like.shape[1:]:
        raise ArrayError(
            f"{full_name}: append shape mismatch — existing row shape "
            f"{tail} vs new {like.shape[1:]}"
        )
    blob = level_group.read_bytes(full_name, key)
    return np.frombuffer(blob, dtype=dtype).reshape((-1, *tail)).astype(
        like.dtype, copy=False,
    )


def _parse_offsets_for_family(
    seg: str,
    *,
    sid_ndim: int | None,
    link_width: int,
    partition: LinkPartition,
) -> tuple[ChunkCoords, ...]:
    """Parse an offsets segment, recovering ``sid_ndim`` if the family
    group meta didn't carry it.

    The source chunks in the partition have the family's true arity, so
    they are an exact fallback for a family whose group ``.zattrs``
    predates ``sid_ndim`` (or was written by the per-cell writers alone).
    """
    if sid_ndim is None:
        for _seg, src_chunk in partition.cell_indices:
            sid_ndim = len(src_chunk)
            break
    if sid_ndim is None:
        raise ArrayError(
            f"cannot parse offsets segment {seg!r}: no sid_ndim on the "
            f"links family meta and no cells in the partition"
        )
    return parse_offsets(seg, sid_ndim=sid_ndim, link_width=link_width)


def list_link_offsets(level_group: FsGroup, delta: int = 0) -> list[str]:
    """Sorted offsets segments present under ``links/<delta>/``.

    ``links/<delta>`` is a group whose children are one array per
    distinct relative-offset segment, so this enumerates the whole
    family — the all-zero (intra-chunk) array included, since it is not
    a separate family any more.

    Uses :meth:`FsGroup.children`, not iteration: ``__iter__`` yields
    sub-*groups* only and these children are arrays.
    """
    group_path = links_group_path(delta)
    if not level_group.array_exists(group_path):
        return []
    try:
        return sorted(level_group[group_path].children())
    except Exception:
        return []


def list_link_attribute_offsets(
    level_group: FsGroup, name: str, delta: int = 0,
) -> list[str]:
    """Sorted offsets segments under ``link_attributes/<name>/<delta>/``.

    Mirrors :func:`list_link_offsets`; the attribute family is required
    to carry the same segments as the link family it parallels.
    """
    group_path = link_attributes_group_path(name, delta)
    if not level_group.array_exists(group_path):
        return []
    try:
        return sorted(level_group[group_path].children())
    except Exception:
        return []


def _decode_link_cell(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
    *,
    delta: int,
    offsets: Sequence[ChunkCoords],
    dtype: np.dtype | str,
    width: int,
    default: Any = _UNSET,
) -> list[npt.NDArray[np.integer]] | Any:
    """Decode one ``links/<delta>/<offsets>/`` cell into its row groups.

    ``width`` is the **physical** row width — ``link_width``, or
    ``1 + link_width`` when the array carries a ``perm_idx`` column.
    :func:`links_has_perm` is the single definition of which; this
    decoder is told, it never guesses.

    The encoding branch mirrors :func:`write_chunk_links` exactly and
    must stay in lockstep with it: a cell written under one branch is
    undecodable under the other.
    """
    dtype = np.dtype(dtype)
    full_name = links_path(delta, offsets)
    key = _chunk_key(chunk_coords)
    try:
        if not level_group.chunk_exists(full_name, key):
            raise ArrayError(f"{full_name}: no cell {key}")
        raw = level_group.read_bytes(full_name, key)
    except (ArrayError, StoreError):
        if default is _UNSET:
            raise
        return default

    if delta == 0 and is_intra(offsets):
        # Flat blob + link_fragments/<chunk> sidecar.  An empty cell is
        # the vlen fill value under the single-array layout, which is not
        # the same as "no sidecar" — short-circuit before reading it.
        if not raw:
            return []
        fi = read_link_fragment_index(level_group, chunk_coords)
        if fi.num_fragments == 0:
            return []
        full = _reshape_link_buffer(raw, dtype, width)
        groups: list[npt.NDArray[np.integer]] = []
        for f in range(fi.num_fragments):
            if fi.is_range(f):
                start, count = fi.range(f)
                groups.append(full[start : start + count])
            else:
                groups.append(full[fi.indices(f)])
        return groups

    # Inline self-describing blob; no sidecar.
    return decode_ragged_blob(raw, dtype, ncols=width)


def link_family_policy(
    level_group: FsGroup, delta: int,
) -> tuple[int, int | None, bool, str] | None:
    """``(link_width, sid_ndim, directed, store)`` from the family group.

    Returns ``None`` when the family carries no policy — an absent
    family, or one whose group ``.zattrs`` was never stamped — which
    callers treat as "nothing to read".
    """
    family = links_group_path(delta)
    if not level_group.array_exists(family):
        return None
    fam_meta = level_group.read_array_meta(family) or {}
    if "link_width" not in fam_meta:
        return None
    return (
        int(fam_meta["link_width"]),
        int(fam_meta["sid_ndim"]) if "sid_ndim" in fam_meta else None,
        bool(fam_meta.get("directed", False)),
        str(fam_meta.get("store", "canonical")),
    )


def iter_link_cells(
    level_group: FsGroup, delta: int,
) -> "Iterator[tuple[str, tuple[ChunkCoords, ...], ChunkCoords, list[npt.NDArray[np.integer]]]]":
    """Yield ``(segment, offsets, source_chunk, groups)`` for every cell.

    Enumeration order is the canonical one every reader shares: offsets
    segments sorted, then cells sorted within each.  ``groups`` are the
    decoded **physical** row blocks — still carrying any ``perm_idx``
    column, since callers differ on whether they want it.

    Offsets come from each array's own ``offsets`` meta when present,
    falling back to parsing the path segment; the two agree, but the
    meta needs no ``sid_ndim`` to decode.
    """
    policy = link_family_policy(level_group, delta)
    if policy is None:
        return
    link_width, sid_ndim, directed, store = policy
    family = links_group_path(delta)
    for seg in list_link_offsets(level_group, delta):
        full_name = f"{family}/{seg}"
        meta = level_group.read_array_meta(full_name) or {}
        raw_offsets = meta.get("offsets")
        if raw_offsets is not None:
            offsets = tuple(tuple(int(c) for c in o) for o in raw_offsets)
        elif sid_ndim is not None:
            offsets = parse_offsets(
                seg, sid_ndim=sid_ndim, link_width=link_width,
            )
        else:
            continue
        dtype = np.dtype(meta.get("dtype", "int64"))
        has_perm = links_has_perm(
            offsets, delta=delta, directed=directed, store=store,
        )
        width = link_width + (1 if has_perm else 0)
        for key in sorted(level_group.list_chunks(full_name)):
            try:
                chunk = _parse_chunk_key(key)
            except ValueError:
                continue  # skip non-chunk entries
            groups = _decode_link_cell(
                level_group, chunk, delta=delta, offsets=offsets,
                dtype=dtype, width=width, default=[],
            )
            if not groups:
                continue
            yield seg, offsets, chunk, groups


def _derive_partition_from_links(
    level_group: FsGroup,
    delta: int,
) -> LinkPartition:
    """Re-derive a :class:`LinkPartition` from the on-disk link records.

    ``cell_indices`` is keyed in :func:`read_links` enumeration order —
    each offsets segment sorted, then each cell sorted — with indices
    ``0..num_links-1`` assigned in that read order.  Callers passing
    ``attr_data`` to :func:`write_link_attributes` without an explicit
    partition must supply rows in that same order.

    Only meaningful for a ``canonical`` family: under ``duplicate`` the
    on-disk rows are physical copies, so positional indices cannot
    express the fan-out that maps them back to input records.
    """
    cell_indices: dict[tuple[str, ChunkCoords], list[int]] = {}
    next_idx = 0
    for seg, offsets, chunk, rows in iter_link_cells(level_group, delta):
        del offsets
        n = sum(int(np.asarray(g).shape[0]) for g in rows)
        cell_indices[(seg, chunk)] = list(range(next_idx, next_idx + n))
        next_idx += n
    return LinkPartition(
        cell_indices=cell_indices,
        num_links=next_idx,
        num_physical_records=next_idx,
        first_new=0,
    )


# ===================================================================
# Decentralized (per-cell) link writers
# ===================================================================
#
# ``write_links`` is a whole-family replace/append that also maintains the
# family-wide ``num_links`` / ``num_physical_records`` meta.  The functions
# below instead let many independent workers each write a batch of records
# into ONLY the cells those records touch, deferring the global bookkeeping
# to a single ``finalize_links`` pass.
#
# Race-freedom rests on the flat layout: each cell is its own object
# (sharding is a later, coordinator-run pass), so workers writing DISJOINT
# cells never touch the same file.  A cell IS a source chunk, so giving
# each worker ownership of a disjoint set of source chunks makes every
# cell the property of exactly one worker.  Per-cell RMW is NOT safe for
# two workers hitting the SAME cell.


def write_link_cells(
    level_group: FsGroup,
    links: list[list[tuple[ChunkCoords, int]]] | list[CrossChunkLink],
    sid_ndim: int,
    *,
    delta: int = 0,
    link_width: int | None = None,
    dtype: np.dtype | str = np.int64,
    directed: bool = False,
    store: Literal["canonical", "duplicate"] = "canonical",
) -> LinkPartition:
    """Write a batch of records into only the cells they touch.

    The decentralized counterpart to :func:`write_links`: it leaves every
    other cell untouched and does **not** maintain the family-wide
    ``num_links`` / ``num_physical_records`` counts — call
    :func:`finalize_links` once, after all workers finish, to reconcile
    them (and shard separately if desired).

    Placement routes through :func:`_partition_links`, the same choke
    point :func:`write_links` uses, so a batch lands in exactly the cells
    the whole-family writer would have chosen.  That is what makes
    disjoint per-cell writes plus one ``finalize_links`` equivalent to a
    single ``write_links`` over the union of the batches.

    A coordinator may pre-create the family with matching ``directed`` /
    ``store`` / ``sid_ndim`` via :func:`create_links_array` so workers
    agree on the policy and don't race to create it; this function also
    creates it idempotently and rejects a policy mismatch.

    Returns the :class:`LinkPartition` for THIS batch — its ``num_links``
    is the batch's logical record count, not the family's — for use with
    :func:`write_link_attribute_cells`.

    Concurrency:
        Per-cell read-modify-write.  Safe only while workers own disjoint
        source chunks; two workers appending to one cell lose rows.
    """
    if store not in ("canonical", "duplicate"):
        raise ArrayError(
            f"store must be 'canonical' or 'duplicate', got {store!r}"
        )
    if not links:
        return LinkPartition(
            cell_indices={}, num_links=0, num_physical_records=0, first_new=0,
        )

    dtype = np.dtype(dtype)
    normalised, link_width = _normalise_link_records(
        links, link_width, sid_ndim, delta,
    )

    # Guard before any write: a worker joining a family stamped with other
    # flags would file rows the family-wide reader decodes at the wrong
    # width.  Absent meta means the family is this call's to create.
    fam_meta = level_group.read_array_meta(links_group_path(delta)) or {}
    if fam_meta:
        _check_link_family_policy(
            fam_meta, delta=delta, link_width=link_width, sid_ndim=sid_ndim,
            directed=directed, store=store, action="append to",
        )

    buckets = _partition_links(
        level_group, normalised, link_width, sid_ndim,
        delta=delta, directed=directed, store=store,
    )

    physical = 0
    for (seg, src_chunk), entries in buckets.items():
        offsets = parse_offsets(seg, sid_ndim=sid_ndim, link_width=link_width)
        create_links_array(
            level_group, link_width, dtype=str(dtype), delta=delta,
            sid_ndim=sid_ndim, offsets=offsets, directed=directed,
            store=store, exist_ok=True,
        )
        has_perm = links_has_perm(
            offsets, delta=delta, directed=directed, store=store,
        )
        rows = _pack_link_rows(
            entries, has_perm=has_perm, link_width=link_width, dtype=dtype,
        )
        physical += rows.shape[0]
        # RMW: this batch is a new group appended after whatever the cell
        # already holds, so a worker may call this repeatedly.
        groups = list(_decode_link_cell(
            level_group, src_chunk, delta=delta, offsets=offsets,
            dtype=dtype, width=rows.shape[1], default=[],
        ))
        groups.append(rows)
        # Route through the per-cell writer so the flat+sidecar vs
        # inline-blob choice has exactly one definition.
        #
        # record_presence=False: this is the worker half of the
        # decentralized protocol.  Workers own disjoint source chunks, so
        # their *cells* never collide — but ``nonempty_chunks`` is
        # array-wide, and stamping it is a read-modify-write that two
        # workers writing disjoint cells still race on, silently dropping
        # the loser's key while its payload sits on disk.  The manifest is
        # rebuilt from the store listing by :func:`finalize_links`, which
        # the coordinator must run before any ``shard_store``.
        write_chunk_links(
            level_group, src_chunk, groups, dtype,
            delta=delta, offsets=offsets, link_width=link_width,
            record_presence=False,
        )

    return LinkPartition(
        cell_indices={k: [idx for _, _, idx in v] for k, v in buckets.items()},
        num_links=len(normalised),
        num_physical_records=physical,
        first_new=0,
    )


def write_link_attribute_cells(
    level_group: FsGroup,
    attr_name: str,
    attr_data: npt.NDArray,
    *,
    partition: LinkPartition,
    delta: int = 0,
) -> None:
    """Append attribute rows for the cells one batch wrote.

    ``attr_data`` holds one row per logical record in the SAME batch, in
    the input order used for the matching :func:`write_link_cells` call;
    ``partition`` is that call's return value.  Rows are appended per
    ``(offsets, cell)``, replicating automatically in ``duplicate`` mode
    where one input index appears under several cells.

    Only ``dtype`` / ``row_shape`` are stamped here (idempotent across
    workers).  The attribute family's ``num_links`` is deliberately left
    unstamped: a worker sees only its own batch, and :func:`read_link_attributes`
    derives the row count from the cells themselves.
    """
    arr = np.ascontiguousarray(np.asarray(attr_data))
    fam_meta = level_group.read_array_meta(links_group_path(delta)) or {}
    link_width = int(fam_meta.get("link_width", 2))
    sid_ndim = int(fam_meta.get("sid_ndim", 0)) or None

    for seg in sorted({seg for seg, _chunk in partition.cell_indices}):
        offsets = _parse_offsets_for_family(
            seg, sid_ndim=sid_ndim, link_width=link_width, partition=partition,
        )
        create_link_attributes_array(
            level_group, attr_name, dtype=str(arr.dtype), delta=delta,
            sid_ndim=sid_ndim, link_width=link_width, offsets=offsets,
            exist_ok=True,
        )
        full_name = link_attributes_path(attr_name, delta, offsets)
        for (bucket_seg, src_chunk), input_idxs in partition.cell_indices.items():
            if bucket_seg != seg:
                continue
            key = _chunk_key(src_chunk)
            new_rows = arr[np.asarray(input_idxs, dtype=np.int64)]
            if level_group.chunk_exists(full_name, key):
                combined = np.concatenate(
                    [_read_attr_cell(level_group, full_name, key, arr), new_rows],
                    axis=0,
                )
            else:
                combined = new_rows
            level_group.write_bytes(
                full_name, key, np.ascontiguousarray(combined).tobytes(),
            )
        # ``row_shape`` (the tail dims per row, ``()`` for 1-D) lets the
        # reader reconstruct shape from a bare byte blob.
        level_group.write_array_meta(full_name, {
            "zv_array": "link_attribute",
            "name": attr_name,
            "dtype": str(arr.dtype),
            "row_shape": list(arr.shape[1:]),
            "offsets": [list(int(c) for c in o) for o in offsets],
            "level_delta": int(delta),
        })


def finalize_links(
    level_group: FsGroup,
    *,
    delta: int = 0,
) -> LinkPartition:
    """Reconcile a ``links/<delta>/`` family's counts after decentralized
    per-cell writes.

    Scans every offsets array and every cell under the family to recompute
    ``num_physical_records`` (total on-disk rows) and ``num_links`` (the
    logical record count).  For a ``canonical`` family the two are equal;
    for a ``duplicate`` one the logical count is recovered by deduplicating
    decoded records, since every physical copy of a record decodes to the
    same input-order endpoints.

    Sharding is a separate coordinator step — run
    :func:`zarr_vectors.sharding.shard_store` after finalizing, once all
    cells are on disk.

    Returns the re-derived :class:`LinkPartition` in :func:`read_links`
    enumeration order (each offsets segment sorted, then each cell
    sorted).  ``num_links`` is the **logical** count and
    ``num_physical_records`` the **physical** one — the same two numbers
    stamped on the family group, which the returned partition must agree
    with.  ``cell_indices`` is positional over the physical rows, so under
    ``store="duplicate"`` its indices are row positions, not input
    indices, and must not be fed to :func:`write_link_attributes`.
    """
    policy = link_family_policy(level_group, delta)
    if policy is None:
        return LinkPartition(
            cell_indices={}, num_links=0, num_physical_records=0, first_new=0,
        )
    store = policy[3]  # (link_width, sid_ndim, directed, store)

    # Rebuild each offset array's ``nonempty_chunks`` manifest from the
    # store listing BEFORE enumerating.  This is the coordinator half of
    # the decentralized-write protocol: workers pass
    # ``record_presence=False`` to skip the manifest, because stamping it
    # is a read-modify-write of state shared by every cell in the array —
    # two workers writing *disjoint* cells still race on it, and the
    # loser's key vanishes even though its payload landed.  Without this
    # rebuild, ``iter_link_cells`` (→ ``list_chunks`` → the manifest)
    # would enumerate nothing and this pass would count 0.
    #
    # ``list_link_offsets`` discovers segments via ``children()`` — a
    # store listing — so segment discovery is already race-free.
    #
    # Only valid unsharded: a shard packs many cells into one object whose
    # inner index is not derivable from key names.  Hence the ordering
    # requirement that ``shard_store`` runs *after* finalize.
    family_group = links_group_path(delta)
    for seg in list_link_offsets(level_group, delta):
        try:
            level_group.derive_nonempty_chunks(f"{family_group}/{seg}")
        except Exception:
            # An already-stamped array (the whole-family writer path)
            # needs no rebuild; never let that mask the counts below.
            pass

    physical = 0
    cell_indices: dict[tuple[str, ChunkCoords], list[int]] = {}
    for seg, offsets, chunk, groups in iter_link_cells(level_group, delta):
        del offsets
        n = sum(int(np.asarray(g).shape[0]) for g in groups)
        cell_indices[(seg, chunk)] = list(range(physical, physical + n))
        physical += n

    if store == "duplicate":
        # Every physical copy of a logical record decodes to the same
        # input-order endpoints, so distinct decoded records == logical.
        records = read_links(level_group, delta=delta)
        logical = len({tuple(r) for r in records})
    else:
        logical = physical

    # Merge into the existing family meta rather than restamping policy:
    # this pass counts rows, it has no opinion on directed/store, and a
    # family group written without ``sid_ndim`` must not gain a made-up one.
    family = links_group_path(delta)
    meta = level_group.read_array_meta(family) or {}
    meta["num_links"] = int(logical)
    meta["num_physical_records"] = int(physical)
    _ensure_array_dir(level_group, family)
    level_group.write_array_meta(family, meta)

    return LinkPartition(
        cell_indices=cell_indices,
        num_links=logical,
        num_physical_records=physical,
        first_new=0,
    )


# ===================================================================
# Reading data
# ===================================================================

def vertices_dtype(level_group: FsGroup) -> np.dtype:
    """The element dtype ``vertices/`` declares.

    A cell is a flat buffer with no inline header, so the Zarr
    ``data_type`` (``variable_length_bytes``) describes the *container*
    and says nothing about the payload.  The element type lives in the
    array's ``dtype`` **attribute**, which :func:`create_vertices_array`
    stamps — this is the only place it is recorded.

    Falls back to ``float32`` when the attribute is unreadable, matching
    the writer's own default.
    """
    try:
        vmeta = level_group.read_array_meta(VERTICES) or {}
        return np.dtype(vmeta.get("dtype", "float32"))
    except Exception:
        return np.dtype(np.float32)


def read_chunk_vertices(
    level_group: FsGroup,
    chunk_coords: ChunkCoords,
    dtype: np.dtype | str | None = None,
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
        dtype: Numpy dtype.  ``None`` (the default) reads the dtype the
            store declares — see :func:`vertices_dtype`.  Pass one only to
            override, and only knowing that a wrong value does not raise:
            a ``float64`` cell read as ``float32`` decodes to garbage at
            twice the row count, silently.
        ndim: Number of coordinate dimensions (D).

    Returns:
        List of arrays, each ``(N_k, D)``.

    Raises:
        ArrayError: If the chunk does not exist or data is malformed.
    """
    key = _chunk_key(chunk_coords)
    # Default to the declared dtype rather than float32.  Nothing in the
    # blob records the element type, so an assumed dtype is not checkable
    # — it just produces wrong numbers.  The store already knows; ask it.
    dtype = vertices_dtype(level_group) if dtype is None else np.dtype(dtype)

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
    offsets: Sequence[ChunkCoords] | None = None,
) -> list[npt.NDArray[np.integer]]:
    """Read link groups from one cell of ``links/<delta>/<offsets>/``.

    The per-cell counterpart to :func:`write_chunk_links`, and it mirrors
    its encoding rule exactly: ``chunk_coords`` is the **source** chunk
    (the cell) and ``offsets`` names the array.  Defaults to the all-zero
    (intra-chunk) offsets — the array holding records whose endpoints all
    share the source chunk.

    Rows come back **as stored**, in placement order: this reader does not
    reverse a canonical sort.  Where the array carries a ``perm_idx``
    column (see :func:`links_has_perm`) the returned rows are ``1 + L``
    wide, with ``perm_idx`` in column 0 and the ``L`` vertex indices after
    it — and those indices are in the placement's endpoint order, not the
    order they were written in.  Use :func:`read_links` or
    :func:`read_links_for_tuple` for whole records in input order.

    Args:
        level_group: Resolution level group.
        chunk_coords: **Source** chunk coordinates (the array cell).
        dtype: Integer dtype.
        link_width: The **logical** width L. If None, read from the family
            group's metadata.  Note the *stored* width may be ``1 + L``;
            this argument never changes how the bytes are decoded, which
            is governed by the array's own ``has_perm`` stamp.
        delta: Level delta; ``0`` is the intra-level family.
        offsets: Relative offsets naming the array.  ``None`` (default)
            means the all-zero intra-chunk offsets.

    Returns:
        List of arrays, each ``(M_k, L)`` — or ``(M_k, 1 + L)`` when the
        array carries ``perm_idx``.
    """
    key = _chunk_key(chunk_coords)
    dtype = np.dtype(dtype)

    # Read L from the <delta> family group, not from the array: the
    # offsets segment naming the array is itself a function of L, so
    # reading it from the array would be circular.
    fam_meta = level_group.read_array_meta(links_group_path(delta)) or {}
    if link_width is None:
        link_width = int(fam_meta.get("link_width", 2))
    if offsets is None:
        offsets = intra_offsets(len(chunk_coords), link_width)
    full_name = links_path(delta, offsets)

    # Decode at the PHYSICAL row width.  ``link_width`` is the logical
    # width; a links_has_perm array stores 1 + L per row.  Decoding
    # physical bytes at the logical width does not reliably raise: with N
    # records the cell holds (1 + L)*N elements, and (1 + L)*N ≡ N (mod L),
    # so it silently yields (1 + L)*N/L fabricated rows whenever
    # N % L == 0 — every even N at L=2.  Take the width from the array's
    # own stamp; never infer it from L.
    arr_meta = level_group.read_array_meta(full_name) or {}
    has_perm = bool(arr_meta.get(
        "has_perm",
        links_has_perm(
            offsets,
            delta=delta,
            directed=bool(fam_meta.get("directed", False)),
            store=str(fam_meta.get("store", "canonical")),
        ),
    ))
    ncols = (1 + link_width) if has_perm else link_width

    # Only the intra array at delta 0 is flat-with-sidecar; every other
    # offset array is an inline self-describing blob.  Same condition as
    # write_chunk_links — keep them in lockstep.
    flat = delta == 0 and is_intra(offsets)
    plan: list[tuple[str, list[str]]] = [(full_name, [key])]
    if flat:
        plan.append((LINK_FRAGMENTS, [key]))

    with _maybe_batched_reads(level_group, plan):
        try:
            raw = level_group.read_bytes(full_name, key)
        except Exception as e:
            raise ArrayError(
                f"Cannot read links chunk {key} (delta={format_delta(delta)}, "
                f"offsets={format_offsets(offsets)}): {e}"
            ) from e

        if flat:
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
            full = _reshape_link_buffer(raw, dtype, ncols)
            groups: list[npt.NDArray[np.integer]] = []
            for f in range(fi.num_fragments):
                if fi.is_range(f):
                    start, count = fi.range(f)
                    groups.append(full[start : start + count])
                else:
                    groups.append(full[fi.indices(f)])
            return groups

        # Cross-offset and/or cross-level: inline self-describing blob.
        return decode_ragged_blob(raw, dtype, ncols=ncols)


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

    try:
        if link_width is None:
            # From the family group — the offsets segment naming the
            # array depends on L, so reading L from the array is circular.
            fam_meta = level_group.read_array_meta(links_group_path(0)) or {}
            link_width = int(fam_meta.get("link_width", 2))
        # link_fragments/ only ever partitions the intra array at delta 0
        # (it is keyed by chunk alone, with no delta or offsets segment),
        # so this reader is intra-only by construction.
        full_name = links_path(0, intra_offsets(len(chunk_coords), link_width))

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

    Per-vertex attributes are stored one row per vertex, aligned 1:1
    with the ``vertices`` buffer (Core-1's fragment 0).  Each fragment
    is gathered by vertex index from the shared attribute buffer via the
    ``vertex_fragments/<chunk>`` index — range fragments as contiguous
    slices, explicit fragments via ``fi.indices(f)`` — exactly as
    :func:`read_chunk_vertices` does.  There is no contiguity
    requirement, so explicit path-fragment vertex-fragments (e.g.
    BRIDGE's polyline twins appended on top of Core-1's range) read
    correctly.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name.
        chunk_coords: Spatial chunk coordinates.
        dtype: Numpy dtype of the attribute.
        ncols: Number of columns (channels). Use 1 for scalars.
        vert_dtype: Retained for backward compatibility; ignored.
            Per-fragment sizes now come from the fragment index directly.
        vert_ndim: Retained for backward compatibility; ignored.

    Returns:
        List of arrays aligned with fragments, each ``(N_k, ncols)`` (or
        ``(N_k,)`` when ``ncols == 1``).  ``groups[0]`` is Core-1's
        fragment 0 — the full per-vertex buffer for the chunk.
    """
    del vert_dtype, vert_ndim  # retained for signature compat; unused
    key = _chunk_key(chunk_coords)
    dtype = np.dtype(dtype)
    full_name = f"{VERTEX_ATTRIBUTES}/{attr_name}"

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
        fi = read_vertex_fragment_index(level_group, chunk_coords)

    if fi.num_fragments == 0:
        return []
    # Per-vertex attributes are one row per underlying vertex, aligned 1:1
    # with the ``vertices`` buffer (Core-1's fragment 0).  Reshape the flat
    # buffer to ``(N_vertices, width)`` where ``width`` is the attribute's
    # true per-vertex column count (derived from the buffer size and the
    # vertex extent — independent of the requested ``ncols``), then gather
    # each fragment by vertex index exactly as read_chunk_vertices does:
    # range fragments as contiguous slices, explicit fragments (e.g.
    # BRIDGE's appended path-fragment twins) via ``fi.indices(f)``.  Each
    # gathered group is finally flattened and re-shaped to honour the
    # caller's ``ncols`` (ncols=1 yields a flat 1-D array), matching the
    # historical decode_ragged_floats contract.
    itemsize = dtype.itemsize
    total_elements = len(raw) // itemsize if itemsize else 0
    n_vertices = _fragment_vertex_extent(fi)
    if n_vertices <= 0 or total_elements == 0:
        empty = np.empty((0,) if ncols == 1 else (0, ncols), dtype=dtype)
        return [empty for _ in range(fi.num_fragments)]
    if total_elements % n_vertices != 0:
        raise ArrayError(
            f"Attribute '{attr_name}' chunk {key} has {total_elements} "
            f"elements, not a multiple of its {n_vertices} vertices; "
            "per-vertex attributes must align 1:1 with the vertices array."
        )
    width = total_elements // n_vertices
    full = np.frombuffer(raw, dtype=dtype).reshape(n_vertices, width)
    groups: list[npt.NDArray] = []
    for f in range(fi.num_fragments):
        if fi.is_range(f):
            start, count = fi.range(f)
            rows = full[start : start + count]
        else:
            rows = full[fi.indices(f)]
        flat = np.ascontiguousarray(rows).reshape(-1)
        groups.append(flat if ncols == 1 else flat.reshape(-1, ncols))
    return groups


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
        # An unwritten cell of a vlen array reads back b"" rather than
        # raising, and b"" passes the stride check below — so without this
        # guard a missing chunk would quietly decode to an empty array and
        # ``default`` would never be honoured.  ``chunk_exists`` consults
        # the presence manifest, which is what distinguishes "absent" from
        # "empty" now that the two are byte-identical.
        if not level_group.chunk_exists(full_name, key):
            raise ArrayError(
                f"fragment_attribute '{attr_name}' chunk {key} not present"
            )
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
    """Read per-link attribute data for a chunk's intra-chunk links.

    Mirrors :func:`read_chunk_attributes` for the per-link case: the
    ragged bytes live under
    ``link_attributes/<name>/<delta>/<all-zero offsets>/<chunk>`` and
    align 1:1 with the link fragments under ``link_fragments/<chunk>``.

    Intra-chunk at ``delta == 0`` only, and not by omission: the fragment
    sidecar is keyed by chunk alone, so the all-zero-offsets array is the
    only one it partitions.  Attributes on any other offsets array have no
    fragment structure to align to — read those with
    :func:`read_link_attributes`.  Per-link group ``k`` has the
    same row count as link group ``k`` in ``links/<delta>/<chunk>``.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name (e.g. ``"weight"``).
        chunk_coords: Spatial chunk coordinates.
        dtype: Numpy dtype of the attribute.
        ncols: Number of columns (channels).  Use 1 for scalars.
        delta: Level delta; cross-level link attributes are stored
            differently and must be read via the whole-family
            :func:`read_link_attributes` — this helper handles only the
            per-chunk ``delta == 0`` case.

    Returns:
        List of arrays aligned with the link fragments in the chunk.
    """
    if delta != 0:
        raise ArrayError(
            f"read_chunk_link_attributes only supports delta=0 "
            f"(per-chunk intra-level); got delta={delta}.  Use "
            f"read_link_attributes for cross-level link attributes.",
        )
    key = _chunk_key(chunk_coords)
    dtype = np.dtype(dtype)
    # Fragment-aligned attributes exist only for the intra array: it is
    # the one link_fragments/<chunk> partitions.
    fam_meta = level_group.read_array_meta(links_group_path(delta)) or {}
    link_width = int(fam_meta.get("link_width", 2))
    full_name = link_attributes_path(
        attr_name, delta, intra_offsets(len(chunk_coords), link_width),
    )

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


def cell_endpoint_chunks(
    src: ChunkCoords,
    offsets: Sequence[ChunkCoords],
    scale_src: Sequence[int],
    scale_trg: Sequence[int],
) -> tuple[ChunkCoords, ...]:
    """Reconstruct a cell's L endpoint chunks from its source and offsets.

    The inverse of the placement arithmetic in
    :func:`~zarr_vectors.spatial.boundary.partition_records_by_offset`:
    endpoint 0 *is* the source (its offset is the implicit zero and is
    never encoded), and endpoint k>0 is ``anchor(src) + o_k``.  The
    anchor makes this exact across levels whose chunk grids differ; it
    is the identity when they do not.
    """
    from zarr_vectors.spatial.boundary import anchor_chunk

    anchor = anchor_chunk(src, scale_src, scale_trg)
    return (tuple(src),) + tuple(
        tuple(int(a) + int(o) for a, o in zip(anchor, off)) for off in offsets
    )


def _link_cell_rows(
    blob: bytes, *, ncols: int, flat: bool,
) -> npt.NDArray[np.int64]:
    """Decode one link cell's bytes into its ``(M, ncols)`` physical rows.

    The single decode :func:`read_links` and :func:`read_links_for_tuple`
    share, so a whole-family read and a single-cell read of the same bytes
    cannot disagree.  ``flat`` selects the branch
    :func:`write_chunk_links` wrote under (``delta == 0 and
    is_intra(offsets)``) — the two encodings are not interchangeable.

    Rows come back in write order: groups in append order, rows in order
    within each group.  That matches the enumeration
    :func:`iter_link_cells` counts against.

    NOTE :func:`~zarr_vectors.encoding.ragged.decode_ragged_blob` yields
    one entry per *group* — an ``(M_k, ncols)`` block — not one per row.
    A whole-family write files a cell's rows as a single group and each
    append adds another, so the groups must be concatenated; treating the
    group count as a row count mis-parses every multi-record cell.
    """
    if flat:
        # Intra-chunk at delta 0: flat concatenation, group bounds in the
        # link_fragments sidecar.  A cell-wide read wants every row, so
        # the sidecar is not consulted.
        return np.frombuffer(blob, dtype=np.int64).reshape(-1, ncols)
    groups = decode_ragged_blob(blob, np.dtype(np.int64), ncols=ncols)
    if not groups:
        return np.empty((0, ncols), dtype=np.int64)
    return np.concatenate(
        [np.asarray(g, dtype=np.int64).reshape(-1, ncols) for g in groups],
        axis=0,
    )


def read_links(
    level_group: FsGroup,
    *,
    delta: int = 0,
) -> list[tuple[tuple[ChunkCoords, int], ...]]:
    """Read every link record under ``links/<delta>/``.

    The whole-family counterpart to :func:`write_links`, and the inverse
    of its placement: each cell's source chunk plus the array's offsets
    segment reconstruct the record's endpoint chunks, and ``perm_idx``
    (where present) reverses the canonical sort — so callers see the same
    record shape they wrote.  Intra-chunk links are included; they are
    the all-zero-offsets array, not a separate family.

    Records are returned in **(offsets segment, cell) sorted order**, and
    within a cell in write order.  :func:`read_link_attributes`
    enumerates identically — that shared order is the only thing aligning
    attribute rows to link records, so the two must not drift.

    For a ``store="duplicate"`` family each logical record was filed in
    several cells, so it is returned **once per copy** — dedupe, or query
    a single location with :func:`read_links_for_tuple`.

    Returns ``[]`` when the ``<delta>`` family is absent or empty.
    """
    family = links_group_path(delta)
    if not level_group.array_exists(family):
        return []
    fam_meta = level_group.read_array_meta(family) or {}
    if "link_width" not in fam_meta or "sid_ndim" not in fam_meta:
        return []
    link_width = int(fam_meta["link_width"])
    sid_ndim = int(fam_meta["sid_ndim"])
    directed = bool(fam_meta.get("directed", False))
    store = str(fam_meta.get("store", "canonical"))

    from zarr_vectors.encoding.ragged import decode_ragged_blob
    from zarr_vectors.spatial.boundary import apply_perm_inverse

    # Must be _link_scales, not _derive_level_scales: the latter derives
    # its rank from the root chunk_shape, but an attribute-binned store
    # prepends a bin axis to every chunk key, so the scales need padding
    # to sid_ndim.  The write path anchors through _link_scales — reading
    # with a differently-ranked scale would silently anchor elsewhere.
    scale_src, scale_trg = _link_scales(level_group, delta, sid_ndim)

    out: list[tuple[tuple[ChunkCoords, int], ...]] = []
    for seg in list_link_offsets(level_group, delta):
        arr_name = f"{family}/{seg}"
        try:
            offsets = parse_offsets(
                seg, sid_ndim=sid_ndim, link_width=link_width,
            )
        except ValueError as e:
            raise ArrayError(
                f"{arr_name}: offsets segment {seg!r} does not match the "
                f"family's sid_ndim={sid_ndim} link_width={link_width}: {e}"
            ) from e
        arr_meta = level_group.read_array_meta(arr_name) or {}
        # Trust the stored width; fall back to recomputing it from the
        # family policy for an array written before it was stamped.
        # Guessing here would silently mis-parse every row in the cell.
        has_perm = bool(arr_meta.get(
            "has_perm",
            links_has_perm(
                offsets, delta=delta, directed=directed, store=store,
            ),
        ))
        ncols = (1 + link_width) if has_perm else link_width
        flat = delta == 0 and is_intra(offsets)
        chunks_cache: dict[ChunkCoords, tuple[ChunkCoords, ...]] = {}

        for cell_key in sorted(level_group.list_chunks(arr_name)):
            blob = level_group.read_bytes(arr_name, cell_key)
            if not blob:
                continue
            src = _parse_chunk_key(cell_key)
            rows_arr = _link_cell_rows(blob, ncols=ncols, flat=flat)
            if rows_arr.size == 0:
                continue

            if src not in chunks_cache:
                chunks_cache[src] = cell_endpoint_chunks(
                    src, offsets, scale_src, scale_trg,
                )
            chunks = chunks_cache[src]

            if has_perm:
                perm_list = rows_arr[:, 0].tolist()
                vi_list = rows_arr[:, 1:1 + link_width].tolist()
            else:
                # Identity placement only: rows are already input order.
                perm_list = None
                vi_list = rows_arr[:, :link_width].tolist()

            if perm_list is None:
                for vis in vi_list:
                    out.append(tuple(zip(chunks, vis)))
            elif link_width == 2:
                # The only 2-endpoint perms are identity (0) and swap (1);
                # avoid the per-row Lehmer decode in apply_perm_inverse.
                cc0, cc1 = chunks[0], chunks[1]
                for perm_idx, (v0, v1) in zip(perm_list, vi_list):
                    ep0 = (cc0, v0)
                    ep1 = (cc1, v1)
                    out.append((ep0, ep1) if perm_idx == 0 else (ep1, ep0))
            else:
                for perm_idx, vi_placed in zip(perm_list, vi_list):
                    placed_endpoints = list(zip(chunks, vi_placed))
                    out.append(tuple(apply_perm_inverse(
                        placed_endpoints, perm_idx, link_width,
                    )))
    return out


def _link_tuple_cell(
    level_group: FsGroup,
    chunk_tuple: Sequence[ChunkCoords],
    delta: int,
) -> tuple[
    tuple[ChunkCoords, ...],
    ChunkCoords,
    tuple[ChunkCoords, ...],
    tuple[int, int, bool, str],
] | None:
    """Resolve an L-chunk tuple to the one ``(offsets, source, chunks)``
    cell that can hold its records, plus the family policy.

    The single definition :func:`read_links_for_tuple` and
    :func:`read_link_attributes_for_tuple` both consult, so a tuple cannot
    resolve to one cell in the links family and a different one in the
    attribute family that parallels it.

    Inverts :func:`~zarr_vectors.spatial.boundary._cell_placements` for a
    query that knows chunks but not vertex indices:

    - An **undirected canonical** family at delta 0 sorts its endpoints by
      ``(chunk, vi)``.  A ``vi`` only breaks ties between endpoints already
      in the same chunk, so it never reorders the chunk sequence — sorting
      the chunks alone reproduces the placement exactly, and callers may
      pass any order.
    - Every other family leads with input endpoint 0: ``directed`` and
      cross-level (``delta != 0``) force the identity placement, and
      ``duplicate`` files one copy per distinct incident chunk of which
      the identity is always one.  The tuple is taken as given, so its
      order is meaningful.

    Offsets are anchor-projected, which is what makes a ``delta != 0``
    query resolve against the owning level's chunk grid rather than
    differencing two incommensurable grids.

    Returns ``None`` when the family is absent or carries no policy.
    """
    policy = link_family_policy(level_group, delta)
    if policy is None:
        return None
    link_width, sid_ndim, directed, store = policy
    if len(chunk_tuple) != link_width:
        raise ArrayError(
            f"chunk_tuple has {len(chunk_tuple)} chunks; expected "
            f"link_width={link_width}"
        )
    chunks = tuple(tuple(int(c) for c in ch) for ch in chunk_tuple)
    if sid_ndim is None:
        # Family group predating the sid_ndim stamp: the query's own arity
        # is the only evidence, and the arity check below is then vacuous.
        sid_ndim = len(chunks[0]) if chunks else 0
    for c in chunks:
        if len(c) != sid_ndim:
            raise ArrayError(
                f"chunk_tuple element {c} has arity {len(c)}; "
                f"expected sid_ndim={sid_ndim}"
            )

    if delta == 0 and not directed and store == "canonical":
        chunks = tuple(sorted(chunks))
    src = chunks[0]

    from zarr_vectors.spatial.boundary import anchor_chunk

    scale_src, scale_trg = _link_scales(level_group, delta, sid_ndim)
    anchor = anchor_chunk(src, scale_src, scale_trg)
    offsets = tuple(
        tuple(int(c) - int(a) for c, a in zip(ch, anchor)) for ch in chunks[1:]
    )
    return offsets, src, chunks, (link_width, sid_ndim, directed, store)


def read_links_for_tuple(
    level_group: FsGroup,
    chunk_tuple: Sequence[ChunkCoords],
    *,
    delta: int = 0,
) -> list[tuple[tuple[ChunkCoords, int], ...]]:
    """Read records that span exactly the L chunks in ``chunk_tuple``.

    The single-cell counterpart to :func:`read_links`: the tuple resolves
    to exactly one ``(offsets, source)`` cell, so only that cell is
    fetched.  See :func:`_link_tuple_cell` for how the tuple maps to it —
    in particular an **undirected canonical** family sorts the tuple
    internally (pass the chunks in any order), while a **directed** or
    **duplicate** family reads it as input order (``(A, B)`` and ``(B, A)``
    are different cells).  An all-equal tuple reads that chunk's
    intra-chunk links, which are no longer a separate family.
    ``len(chunk_tuple)`` must equal the family's ``link_width``.

    Returns ``[]`` if no records exist for that exact L-tuple.
    Records are returned in write order, each in original input
    endpoint order (``perm_idx`` is reversed for the caller).
    """
    resolved = _link_tuple_cell(level_group, chunk_tuple, delta)
    if resolved is None:
        return []
    offsets, src, chunks, (link_width, _sid_ndim, directed, store) = resolved

    full_name = links_path(delta, offsets)
    key = _chunk_key(src)
    if not level_group.chunk_exists(full_name, key):
        return []
    blob = level_group.read_bytes(full_name, key)
    if not blob:
        return []

    arr_meta = level_group.read_array_meta(full_name) or {}
    # Trust the stored width; fall back to recomputing it from the family
    # policy for an array written before it was stamped.  Guessing here
    # would silently mis-parse every row in the cell.
    has_perm = bool(arr_meta.get(
        "has_perm",
        links_has_perm(offsets, delta=delta, directed=directed, store=store),
    ))
    ncols = (1 + link_width) if has_perm else link_width
    rows_arr = _link_cell_rows(
        blob, ncols=ncols, flat=(delta == 0 and is_intra(offsets)),
    )
    if rows_arr.size == 0:
        return []

    from zarr_vectors.spatial.boundary import apply_perm_inverse

    out: list[tuple[tuple[ChunkCoords, int], ...]] = []
    if not has_perm:
        # Identity placement only: rows are already input order.
        for vis in rows_arr[:, :link_width].tolist():
            out.append(tuple(zip(chunks, vis)))
        return out
    for row in rows_arr.tolist():
        perm_idx = int(row[0])
        placed_endpoints = list(zip(chunks, row[1:1 + link_width]))
        out.append(tuple(apply_perm_inverse(
            placed_endpoints, perm_idx, link_width,
        )))
    return out


def read_link_attributes(
    level_group: FsGroup,
    attr_name: str,
    dtype: np.dtype | str | None = None,
    *,
    delta: int = 0,
) -> npt.NDArray:
    """Read every per-link attribute row under ``link_attributes/<name>/<delta>/``.

    The whole-family counterpart to :func:`write_link_attributes`, and the
    attribute-side mirror of :func:`read_links`: rows come back in
    **(offsets segment, cell) sorted order**, which is exactly the order
    :func:`read_links` returns records, so attribute row ``i`` belongs to
    link record ``i``.  That shared enumeration is the ONLY thing aligning
    the two — nothing on disk ties a row to a record — so the two
    functions must not drift.

    Args:
        level_group: Resolution level group.
        attr_name: Attribute name.
        dtype: Parse the stored bytes as this dtype instead of the one on
            the array's meta.
        delta: Level delta.

    Returns:
        Array of shape ``(num_links,)`` or ``(num_links, *row_shape)``;
        empty when the family is absent or holds no rows.
    """
    group_path = link_attributes_group_path(attr_name, delta)
    if not level_group.array_exists(group_path):
        return np.array([], dtype=dtype or np.float32)

    blocks: list[npt.NDArray] = []
    row_shape: tuple[int, ...] = ()
    out_dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float32)
    for seg in list_link_attribute_offsets(level_group, attr_name, delta):
        full_name = f"{group_path}/{seg}"
        meta = level_group.read_array_meta(full_name) or {}
        # ``dtype`` / ``row_shape`` come from each array's own meta — a
        # bare byte blob is undecodable without them.
        out_dtype = (
            np.dtype(dtype) if dtype is not None else np.dtype(meta["dtype"])
        )
        row_shape = tuple(meta.get("row_shape", ()))
        row_size = int(np.prod(row_shape)) if row_shape else 1
        row_bytes = out_dtype.itemsize * row_size
        for cell_key in sorted(level_group.list_chunks(full_name)):
            blob = level_group.read_bytes(full_name, cell_key)
            if not blob:
                continue
            n = len(blob) // row_bytes
            arr = np.frombuffer(blob, dtype=out_dtype).reshape(
                (n, *row_shape) if row_shape else (n,),
            )
            blocks.append(arr.copy())
    if not blocks:
        return (
            np.empty((0, *row_shape), dtype=out_dtype) if row_shape
            else np.empty((0,), dtype=out_dtype)
        )
    return np.concatenate(blocks, axis=0)


def read_link_attributes_for_tuple(
    level_group: FsGroup,
    attr_name: str,
    chunk_tuple: Sequence[ChunkCoords],
    dtype: np.dtype | str | None = None,
    *,
    delta: int = 0,
) -> npt.NDArray:
    """Read per-link attribute rows for the cell spanning ``chunk_tuple``.

    Returns rows in the same order :func:`read_links_for_tuple` returns
    records for the same tuple (cell write order).  The cell is resolved
    by :func:`_link_tuple_cell` against the **links** family's policy —
    attribute meta carries no ``directed`` / ``store`` — so both readers
    land on the same cell by construction.
    """
    resolved = _link_tuple_cell(level_group, chunk_tuple, delta)
    if resolved is None:
        return np.array([], dtype=dtype or np.float32)
    offsets, src, _chunks, _policy = resolved

    full_name = link_attributes_path(attr_name, delta, offsets)
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

    key = _chunk_key(src)
    if not level_group.chunk_exists(full_name, key):
        return (
            np.empty((0, *row_shape), dtype=dtype) if row_shape
            else np.empty((0,), dtype=dtype)
        )
    blob = level_group.read_bytes(full_name, key)
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
    public ``list_link_deltas`` / ``list_link_attribute_deltas`` helpers
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


def list_link_attribute_deltas(level_group: FsGroup, name: str) -> list[int]:
    """Sorted list of ``<delta>`` values present under ``link_attributes/<name>/``."""
    return _list_deltas_under(level_group, f"{LINK_ATTRIBUTES}/{name}")


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
    except StoreError:
        raw = b""
    # An allocated-but-never-written cell reads back b"" — the vlen fill
    # value — which means "nothing here yet", not "corrupt".  Only StoreError
    # used to signal absence; since every per-chunk array became a single
    # vlen array, absence usually arrives as empty bytes instead, and
    # decode_fn raises ArrayError on those.  Treat both as absent.
    existing = decode_fn(raw) if raw else initial
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


def _fragment_vertex_extent(fi: ChunkFragmentIndex) -> int:
    """Return the number of underlying vertices a fragment index spans.

    This is ``max(referenced vertex index) + 1`` across every fragment —
    equal to the length of the ``vertices`` array the chunk's per-vertex
    attributes align 1:1 with (Core-1's fragment 0 is a range covering
    all vertices, so it sets this extent; explicit twins only reference a
    subset).  Used by :func:`read_chunk_attributes` to recover the
    attribute's true per-vertex column width.
    """
    extent = 0
    for f in range(fi.num_fragments):
        if fi.is_range(f):
            start, count = fi.range(f)
            extent = max(extent, int(start) + int(count))
        else:
            idx = fi.indices(f)
            if idx.size:
                extent = max(extent, int(idx.max()) + 1)
    return extent


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


