"""ZV store creation, opening, and management.

Naming: the on-disk format is referred to as **ZV** (Zarr Vectors).  The
older ``ZVF`` initialism may still appear in archived doc text but is
not used in the wire format.

All storage I/O routes through a :class:`zarr.abc.store.Store` wrapped
by the :class:`Group` abstraction in :mod:`zarr_vectors.core.group`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

import zarr
from zarr.storage import LocalStore

if TYPE_CHECKING:
    from zarr.storage import StoreLike

from zarr_vectors.constants import (
    DEFAULT_AXES_NAMES,
    DEFAULT_BOUNDS_SIDE,
    DEFAULT_OOB_POLICY,
    FORMAT_VERSION,  # noqa: F401  (re-exported for callers)
    PARAMETRIC_GROUP,
    RESOLUTION_PREFIX,
    VALID_OOB_POLICIES,
    VERTICES,
)
from zarr_vectors.core.group import Group, _BackendShim
from zarr_vectors.core.metadata import (
    LevelMetadata,
    NgffAxis,
    ParametricTypeDef,
    RootMetadata,
    deserialise_parametric_types,
    serialise_parametric_types,
)
from zarr_vectors.exceptions import MetadataError, StoreError


# ===================================================================
# Path / URL → Zarr store
# ===================================================================


def _resolve_local_path(path: StoreLike) -> Path:
    """Coerce a path-or-``file://``-URL to a local :class:`Path`.

    Only ``str`` and :class:`pathlib.Path` inputs are meaningful here;
    other ``StoreLike`` variants (``Store``, ``StorePath``, ``FSMap``,
    ``dict``) should be handled by the caller before reaching this
    helper and will raise :class:`TypeError`.
    """
    if isinstance(path, Path):
        return path
    if isinstance(path, str):
        if path.startswith("file://"):
            parsed = urlparse(path)
            p = unquote(parsed.path)
            if os.name == "nt" and len(p) > 2 and p[0] == "/" and p[2] == ":":
                p = p[1:]
            return Path(p)
        return Path(path)
    raise TypeError(
        f"_resolve_local_path expected str | Path, got {type(path).__name__}"
    )


def _detect_scheme(url: StoreLike) -> str:
    """Return the URL scheme of ``url`` lowercased; ``""`` for anything
    that is not a string with an explicit scheme.

    Tolerates the wider :data:`zarr.storage.StoreLike` type so callers
    can pass through pre-built ``Store`` / ``StorePath`` / ``FSMap``
    instances and get ``""`` back (signalling "no URL scheme — treat as
    a non-URL input"); the caller is expected to handle those branches
    separately.
    """
    if isinstance(url, Path):
        return ""
    if not isinstance(url, str):
        return ""
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if len(scheme) <= 1:
        return ""
    return scheme


def _make_obstore_zarr_store(
    url: str,
    *,
    mode: str = "r+",
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Build a :class:`zarr.storage.ObjectStore` for ``url`` via obstore.

    Dispatches on URL scheme to the appropriate ``obstore.store.*``
    backend (S3Store, GCSStore, AzureStore, HTTPStore, LocalStore) and
    wraps it in Zarr's ``ObjectStore`` adapter.  ``kwargs`` are forwarded
    verbatim to the obstore constructor (e.g. ``access_key_id``,
    ``region``, ``anonymous``).
    """
    try:
        from zarr.storage import ObjectStore
    except ImportError as e:
        raise StoreError(
            f"backend='obstore' requires zarr>=3 (have {zarr.__version__})"
        ) from e
    try:
        from obstore.store import (
            AzureStore,
            GCSStore,
            HTTPStore,
            S3Store,
        )
        from obstore.store import LocalStore as ObsLocalStore
    except ImportError as e:
        raise StoreError(
            "obstore is not installed. Install: pip install zarr-vectors[obstore]"
        ) from e

    scheme = _detect_scheme(url)
    read_only = mode == "r"
    if scheme == "s3":
        obs = S3Store.from_url(url, **kwargs)
    elif scheme in ("gs", "gcs"):
        obs = GCSStore.from_url(url, **kwargs)
    elif scheme in ("az", "azure", "abfs"):
        obs = AzureStore.from_url(url, **kwargs)
    elif scheme in ("http", "https"):
        obs = HTTPStore.from_url(url, **kwargs)
    elif scheme in ("", "file"):
        local = _resolve_local_path(url)
        if not read_only:
            local.mkdir(parents=True, exist_ok=True)
        obs = ObsLocalStore(prefix=str(local), **kwargs)
    else:
        raise StoreError(
            f"obstore: unsupported URL scheme {scheme!r} in {url!r}"
        )
    return ObjectStore(obs, read_only=read_only), None


def _make_fsspec_zarr_store(
    url: str,
    *,
    mode: str = "r+",
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Build a :class:`zarr.storage.FsspecStore` for ``url``.

    ``kwargs`` are forwarded as ``storage_options`` to the underlying
    fsspec filesystem (e.g. ``key`` / ``secret`` for s3fs,
    ``token`` for gcsfs).
    """
    try:
        from zarr.storage import FsspecStore
    except ImportError as e:
        raise StoreError(
            f"backend='fsspec' requires zarr>=3 (have {zarr.__version__})"
        ) from e
    return (
        FsspecStore.from_url(
            str(url),
            storage_options=(kwargs or None),
            read_only=(mode == "r"),
        ),
        None,
    )


def _make_zarr_store_with_session(
    path: StoreLike,
    *,
    mode: str = "r+",
    backend: str | None = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Construct a Zarr store for ``path``.

    Returns ``(store, session)`` where ``session`` is non-``None`` only
    for transactional backends (currently just ``icechunk``).  The
    session must be kept alive for the lifetime of the store and is
    flushed via :func:`commit`.

    Dispatch order:

    1. If ``path`` is already a :class:`zarr.abc.store.Store`, return it
       as-is (ome-zarr-py-style pass-through for caller-built stores).
       Honors ``mode="r"`` by switching the store to read-only when
       supported.
    2. If ``path`` is a bare :class:`obstore.store.ObjectStore` instance,
       wrap it in :class:`zarr.storage.ObjectStore`.
    3. ``backend="icechunk"`` routes through
       :func:`zarr_vectors.core.backends.icechunk_backend.make_icechunk_session`.
    4. Otherwise, resolve the byte-level backend name via
       :func:`zarr_vectors.core.backends.resolve_backend_name` (explicit
       kwarg → ``ZARR_VECTORS_BACKEND`` env → URL-scheme auto-detect)
       and dispatch to ``LocalStore``, :func:`_make_obstore_zarr_store`,
       or :func:`_make_fsspec_zarr_store`.
    """
    from zarr.abc.store import Store as _ZStore

    # 1. Pre-built Zarr store: pass through.
    if isinstance(path, _ZStore):
        if mode == "r" and not path.read_only and hasattr(path, "with_read_only"):
            path = path.with_read_only(True)
        return path, None

    # 2. Pre-built obstore object: wrap in ObjectStore.
    try:
        import obstore.store as _obs_store_mod
    except ImportError:
        _obs_store_mod = None
    if _obs_store_mod is not None and isinstance(path, _obs_store_mod.ObjectStore):
        from zarr.storage import ObjectStore as _ZarrObjStore

        return _ZarrObjStore(path, read_only=(mode == "r")), None

    # 3. icechunk — transactional, explicit only.
    if backend == "icechunk":
        from zarr_vectors.core.backends.icechunk_backend import (
            make_icechunk_session,
        )

        return make_icechunk_session(str(path), mode=mode, **kwargs)

    # After the Store / obstore short-circuits, the remaining supported
    # StoreLike variants are `str` (URL or local path) and `Path`.
    # `StorePath`, `FSMap`, and `dict[str, Buffer]` aren't routed here.
    if not isinstance(path, (str, Path)):
        raise StoreError(
            f"Unsupported StoreLike variant for backend dispatch: "
            f"{type(path).__name__}. Pass a URL string, a pathlib.Path, "
            f"or a pre-built zarr.abc.store.Store / obstore.store.ObjectStore."
        )

    scheme = _detect_scheme(path)
    if scheme in {"", "file"}:
        return LocalStore(_resolve_local_path(path)), None

    # Cloud scheme — route through fsspec or obstore.
    from zarr_vectors.core.backends import resolve_backend_name

    name = resolve_backend_name(str(path), backend)
    if name == "local":
        return LocalStore(_resolve_local_path(path)), None
    if name == "obstore":
        return _make_obstore_zarr_store(str(path), mode=mode, **kwargs)
    if name == "fsspec":
        return _make_fsspec_zarr_store(str(path), mode=mode, **kwargs)
    raise StoreError(f"Unknown backend: {name!r}")


def _make_zarr_store(
    path: str | Path,
    *,
    backend: str | None = None,
    **kwargs: Any,
) -> Any:
    """Back-compat wrapper that discards the session.

    Existing call sites that don't care about transactional backends
    can keep calling this; callers that need to keep a session alive
    (``create_store``, ``open_store``) use
    :func:`_make_zarr_store_with_session` directly.
    """
    store, _ = _make_zarr_store_with_session(path, backend=backend, **kwargs)
    return store


# ===================================================================
# FsGroup — backwards-compatible shim
# ===================================================================


class FsGroup(Group):
    """Backwards-compatible subclass of :class:`Group` rooted at a local path.

    .. deprecated::
        Direct use of ``FsGroup`` is deprecated.  New code should call
        :func:`create_store` / :func:`open_store`.

    Args:
        path: Filesystem path or :class:`pathlib.Path`.  A ``file://``
            URL is also accepted.
        create: If True, create the directory if it does not already
            exist.  If False, raise :class:`StoreError` if the directory
            is missing (matching the legacy behaviour).
    """

    def __init__(self, path: str | Path, *, create: bool = False) -> None:
        root = _resolve_local_path(path)
        if create:
            root.mkdir(parents=True, exist_ok=True)
        elif not root.is_dir():
            raise StoreError(f"Store path does not exist: {root}")
        store = LocalStore(root)
        # Use mode="a" so the store is read-write but does not clobber
        # any existing root group.
        zg = zarr.open_group(store, mode="a")
        super().__init__(zg)


# ===================================================================
# Store API
# ===================================================================


def create_store(
    path: StoreLike,
    *,
    bounds: tuple[list[float], list[float]] | None = None,
    chunk_shape: tuple[float, ...] | None = None,
    axes: list[NgffAxis] | None = None,
    geometry_types: list[str] | None = None,
    crs: dict[str, Any] | None = None,
    ndim: int | None = None,
    vertex_dtype: str = "float32",
    vertex_encoding: str = "raw",
    links_convention: str | None = None,
    object_index_convention: str | None = None,
    cross_chunk_strategy: str | None = None,
    cross_level_depth: int | None = None,
    cross_level_storage: str | None = None,
    reduction_factor: int | None = None,
    base_bin_shape: tuple[float, ...] | None = None,
    format_capabilities: list[str] | None = None,
    backend: str | None = None,
    **backend_kwargs: Any,
) -> Group:
    """Create a new ZV store.

    ``create_store(path)`` produces a structurally valid "warm" shell:
    root group with format markers + default ``bounds``, a
    ``0/`` sub-group, and the empty ragged-vertex pair
    ``0/vertices/`` + ``0/vertex_fragments/``.

    ``bounds`` is **mandatory** for every ZV store; when the caller does
    not pass one, the default ``([0,...,0], [128,...,128])`` is stamped
    (using ``ndim`` if given, otherwise inferred from ``axes`` /
    ``chunk_shape`` / ``bounds``, defaulting to 3D).  Subsequent writes
    must fit within these bounds unless the caller passes
    ``out_of_bounds="expand"`` (which grows the bounds in-place) or
    calls :func:`set_bounds` first.

    The ``/parametric`` sub-group is **not** created here; it is created
    lazily on first use via :func:`get_parametric_group`.

    Args:
        path: URL or filesystem path for the new store.
        bounds: ``(min_corner, max_corner)``.  When omitted, defaults to
            ``([0]*ndim, [128]*ndim)``.
        chunk_shape: Spatial chunk size per dimension.  When omitted,
            defaults to a single chunk covering ``bounds``.
        axes: OME-Zarr-style axes list.  When omitted, generated from
            :data:`DEFAULT_AXES_NAMES`.
        geometry_types: List of geometry types the store will contain.
            Defaults to ``[]``.
        crs: Coordinate reference system dict.
        ndim: Number of spatial index dimensions.  Useful when no other
            ndim-bearing kwarg is supplied (``axes`` / ``bounds`` /
            ``chunk_shape``).  Defaults to 3.
        vertex_dtype: dtype for the level-0 vertices array.
        vertex_encoding: ``"raw"`` or ``"draco"``.
        links_convention: How edges are encoded
            (``"explicit"`` / ``"implicit_sequential"`` /
            ``"implicit_sequential_with_branches"``).  When omitted the
            store has no convention stamped at create time; the first
            type writer (``write_graph``, ``write_polyline``, ...)
            fills it in via ``_ensure_root_metadata_for_write``.
        object_index_convention: How ``object_index/`` is encoded
            (``"standard"`` / ``"identity"``).  Same lazy-fill rule as
            ``links_convention``.
        cross_chunk_strategy: Cross-chunk connectivity strategy
            (``"boundary_deduplication"`` / ``"explicit_links"`` /
            ``"both"``).  Lazy-filled by type writers.
        cross_level_depth: Maximum ``|delta|`` materialised by
            ``build_pyramid``.  ``0`` disables cross-level link
            arrays.
        cross_level_storage: ``"none"`` / ``"implicit"`` /
            ``"explicit"`` — see
            :data:`zarr_vectors.constants.VALID_XLEVEL_STORAGE`.
        reduction_factor: Default vertex-count fold per pyramid step.
        base_bin_shape: Level-0 supervoxel bin edge lengths.  When
            omitted, defaults to ``chunk_shape`` (one bin per chunk).
        format_capabilities: Optional capability tokens to stamp on
            the root.  See :mod:`zarr_vectors.constants` ``CAP_*``.
        backend: Force a particular backend (``"local"`` / ``"icechunk"``).
        **backend_kwargs: Forwarded to the backend constructor.

    Returns:
        The root :class:`Group`.

    Raises:
        StoreError: If a store already exists at ``path``.
        MetadataError: If kwargs are inconsistent (mismatched ndim).
    """
    resolved_ndim = _resolve_ndim(
        ndim=ndim, axes=axes, chunk_shape=chunk_shape, bounds=bounds,
    )
    if axes is None:
        if resolved_ndim > len(DEFAULT_AXES_NAMES):
            raise MetadataError(
                f"Cannot auto-generate axes for ndim={resolved_ndim}; "
                f"pass an explicit `axes` kwarg."
            )
        axes = [
            {"name": n, "type": "space"}
            for n in DEFAULT_AXES_NAMES[:resolved_ndim]
        ]
    if bounds is None:
        bounds = (
            [0.0] * resolved_ndim,
            [DEFAULT_BOUNDS_SIDE] * resolved_ndim,
        )
    if chunk_shape is None:
        chunk_shape = tuple(
            float(hi - lo) for lo, hi in zip(bounds[0], bounds[1])
        )
    if geometry_types is None:
        geometry_types = []

    from zarr.abc.store import Store as _ZStore
    local_root: Path | None = None
    if (
        backend != "icechunk"
        and not isinstance(path, _ZStore)
        and isinstance(path, (str, Path))
    ):
        scheme = _detect_scheme(path)
        if scheme in {"", "file"}:
            local_root = _resolve_local_path(path)
            if local_root.exists() and local_root.is_dir() and any(local_root.iterdir()):
                raise StoreError(f"Store already exists at {local_root}")
            local_root.mkdir(parents=True, exist_ok=True)

    store, session = _make_zarr_store_with_session(
        path, mode="w", backend=backend, **backend_kwargs,
    )
    if session is not None:
        store._zv_icechunk_session = session
    zg = zarr.open_group(store, mode="w")
    root = FsGroup.__new__(FsGroup) if isinstance(store, LocalStore) else Group.__new__(Group)
    root._zarr = zg

    _write_root_attrs(
        root,
        chunk_shape=chunk_shape,
        bounds=bounds,
        axes=axes,
        geometry_types=geometry_types,
        crs=crs,
        links_convention=links_convention,
        object_index_convention=object_index_convention,
        cross_chunk_strategy=cross_chunk_strategy,
        cross_level_depth=cross_level_depth,
        cross_level_storage=cross_level_storage,
        reduction_factor=reduction_factor,
        base_bin_shape=base_bin_shape,
        format_capabilities=format_capabilities,
    )

    # 0/ + empty vertices pair — the "warm" payload.
    level0 = create_resolution_level(
        root, 0,
        LevelMetadata(
            level=0,
            vertex_count=0,
            arrays_present=[VERTICES],
        ),
    )
    # Defer import: arrays.py imports from store.py (FsGroup).
    from zarr_vectors.core.arrays import create_vertices_array
    create_vertices_array(level0, dtype=vertex_dtype, encoding=vertex_encoding)
    return root


def _resolve_ndim(
    *,
    ndim: int | None,
    axes: list[dict[str, str]] | None,
    chunk_shape: tuple[float, ...] | None,
    bounds: tuple[list[float], list[float]] | None,
) -> int:
    """Resolve a single ``sid_ndim`` from the kwargs that bear it.

    Mismatched values raise.  ``None`` falls back to 3D.
    """
    candidates: list[tuple[str, int]] = []
    if ndim is not None:
        candidates.append(("ndim", ndim))
    if axes is not None:
        candidates.append(("axes", len(axes)))
    if chunk_shape is not None:
        candidates.append(("chunk_shape", len(chunk_shape)))
    if bounds is not None:
        candidates.append(("bounds[0]", len(bounds[0])))
        candidates.append(("bounds[1]", len(bounds[1])))
    if not candidates:
        return 3
    base_name, base = candidates[0]
    for name, value in candidates[1:]:
        if value != base:
            raise MetadataError(
                f"Inconsistent dimensionality: {base_name}={base}, "
                f"{name}={value}"
            )
    return base


def _write_root_attrs(
    root: Group,
    *,
    chunk_shape: tuple[float, ...],
    bounds: tuple[list[float], list[float]],
    axes: list[dict[str, str]],
    geometry_types: list[str],
    crs: dict[str, Any] | None = None,
    links_convention: str | None = None,
    object_index_convention: str | None = None,
    cross_chunk_strategy: str | None = None,
    cross_level_depth: int | None = None,
    cross_level_storage: str | None = None,
    reduction_factor: int | None = None,
    base_bin_shape: tuple[float, ...] | None = None,
    format_capabilities: list[str] | None = None,
) -> None:
    """Write the ``zarr_vectors`` root-attrs block plus the eager NGFF
    ``multiscales`` block (axes only — ``datasets`` are filled in by
    :func:`zarr_vectors.core.multiscale.write_multiscale_metadata`
    when the pyramid is materialised).

    Used by :func:`create_store` (initial) and helpers that update
    structural fields after create (e.g. :func:`set_bounds`).
    """
    full_attrs = root.attrs.to_dict()
    existing = full_attrs.get("zarr_vectors", {})
    zv: dict[str, Any] = dict(existing)
    zv["zv_version"] = FORMAT_VERSION
    if format_capabilities is not None:
        zv["format_capabilities"] = list(format_capabilities)
    else:
        zv.setdefault("format_capabilities", list(zv.get("format_capabilities") or []))
    zv["chunk_shape"] = list(chunk_shape)
    zv["bounds"] = [list(bounds[0]), list(bounds[1])]
    zv["geometry_types"] = list(geometry_types)
    if crs is not None:
        zv["crs"] = crs
    if links_convention is not None:
        zv["links_convention"] = links_convention
    if object_index_convention is not None:
        zv["object_index_convention"] = object_index_convention
    if cross_chunk_strategy is not None:
        zv["cross_chunk_strategy"] = cross_chunk_strategy
    if cross_level_depth is not None:
        zv["cross_level_depth"] = int(cross_level_depth)
    if cross_level_storage is not None:
        zv["cross_level_storage"] = cross_level_storage
    if reduction_factor is not None:
        zv["reduction_factor"] = int(reduction_factor)
    if base_bin_shape is not None:
        zv["base_bin_shape"] = list(base_bin_shape)

    # Eager NGFF ``multiscales`` block — axes are the canonical axis
    # store from 0.5.0 on.  We seed datasets with level 0 only; the
    # multiscale module rewrites datasets when more levels appear.
    existing_ms = full_attrs.get("multiscales") or []
    ms_entry: dict[str, Any] = (
        dict(existing_ms[0]) if existing_ms and isinstance(existing_ms, list)
        else {}
    )
    ms_entry["version"] = "0.4"
    ms_entry.setdefault("name", "default")
    ms_entry["axes"] = list(axes)
    ms_entry.setdefault(
        "datasets",
        [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [1.0] * len(axes)},
        ]}],
    )
    md = dict(ms_entry.get("metadata") or {})
    md["format"] = "zarr_vectors"
    ms_entry["metadata"] = md
    multiscales = [ms_entry]

    root.attrs.update({"zarr_vectors": zv, "multiscales": multiscales})


def _ensure_root_metadata_for_write(
    root: Group,
    *,
    inferred_ndim: int,
    geometry_type: str | None = None,
    base_bin_shape: tuple[float, ...] | None = None,
    links_convention: str | None = None,
    object_index_convention: str | None = None,
    cross_chunk_strategy: str | None = None,
) -> RootMetadata:
    """Stamp any writer-specific fields (``geometry_type``, conventions,
    ``base_bin_shape``) onto root attrs and return the parsed metadata.

    Structural fields (``axes`` / ``bounds`` / ``chunk_shape``) are set
    at :func:`create_store` time and are NOT modified here — bounds
    growth is the responsibility of :func:`_apply_out_of_bounds_policy`
    (per-write) and :func:`set_bounds` (explicit).
    """
    full_attrs = root.attrs.to_dict()
    existing = full_attrs.get("zarr_vectors", {})
    zv: dict[str, Any] = dict(existing)
    zv.setdefault("zv_version", FORMAT_VERSION)
    zv.setdefault("format_capabilities", list(zv.get("format_capabilities") or []))

    # Axes live in NGFF ``multiscales[0].axes`` (0.5.0+).
    ms = full_attrs.get("multiscales") or []
    axes_existing = (
        ms[0].get("axes") if ms and isinstance(ms, list) else None
    ) or []
    if not axes_existing or len(axes_existing) != inferred_ndim:
        raise MetadataError(
            f"Store ndim={len(axes_existing)} does not match data "
            f"ndim={inferred_ndim}.  Re-create the store with the right "
            f"`ndim`/`axes`/`bounds` shape."
        )
    if not zv.get("bounds") or not zv.get("chunk_shape"):
        raise MetadataError(
            "Store missing required `bounds` / `chunk_shape`. "
            "This should never happen for stores created by the current "
            "`create_store` — the store may have been written by an older "
            "build; re-create it."
        )

    if geometry_type is not None:
        gts = list(zv.get("geometry_types") or [])
        if geometry_type not in gts:
            gts.append(geometry_type)
        zv["geometry_types"] = gts

    if base_bin_shape is not None:
        zv["base_bin_shape"] = list(base_bin_shape)

    if links_convention is not None:
        zv.setdefault("links_convention", links_convention)
    if object_index_convention is not None:
        zv.setdefault("object_index_convention", object_index_convention)
    if cross_chunk_strategy is not None:
        zv.setdefault("cross_chunk_strategy", cross_chunk_strategy)

    root.attrs.update({"zarr_vectors": zv})
    # Pass the full attrs (including the NGFF ``multiscales`` block
    # written by :func:`_write_root_attrs`) so strict parsing finds the
    # axes block under ``multiscales[0].axes``.
    return RootMetadata.from_dict(root.attrs.to_dict())


def _apply_out_of_bounds_policy(
    root: Group,
    positions: Any,  # npt.NDArray[np.floating]
    *,
    policy: str,
) -> tuple[Any, Any]:
    """Apply the ``out_of_bounds`` policy to ``positions`` against the
    store's current bounds.

    Returns ``(filtered_positions, kept_mask)``.

    - ``"raise"`` — raise :class:`MetadataError` if any point is out of
      bounds (no filtering).
    - ``"ignore"`` — drop out-of-bounds points; ``kept_mask`` marks the
      survivors so callers can drop aligned attribute arrays / object
      ids in lock-step.
    - ``"expand"`` — grow the store's ``bounds`` (min/max union) to
      include the new points, persist the new bounds, and return all
      points untouched.
    """
    import numpy as np

    if policy not in VALID_OOB_POLICIES:
        raise MetadataError(
            f"unknown out_of_bounds policy {policy!r}; "
            f"valid: {sorted(VALID_OOB_POLICIES)}"
        )
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.size == 0:
        # Nothing to check — empty input or non-(N, D) shape (caller
        # validates downstream).  Return as-is with an all-True mask.
        return positions, np.ones(len(positions), dtype=bool)

    zv = root.attrs.to_dict().get("zarr_vectors", {})
    cur_bounds = zv.get("bounds")
    if not cur_bounds:
        raise MetadataError(
            "Store is missing `bounds`; cannot apply out_of_bounds policy."
        )
    bmin = np.asarray(cur_bounds[0], dtype=positions.dtype)
    bmax = np.asarray(cur_bounds[1], dtype=positions.dtype)
    in_bounds = np.all(
        (positions >= bmin) & (positions <= bmax), axis=1,
    )
    if in_bounds.all():
        return positions, in_bounds

    n_out = int((~in_bounds).sum())
    if policy == "raise":
        raise MetadataError(
            f"{n_out} of {len(positions)} points are outside store bounds "
            f"{cur_bounds!r}; pass out_of_bounds='ignore' to drop them, "
            f"'expand' to grow bounds, or call set_bounds() first."
        )
    if policy == "ignore":
        return positions[in_bounds], in_bounds
    # policy == "expand": union the bounds in-place
    pt_min = positions.min(axis=0).tolist()
    pt_max = positions.max(axis=0).tolist()
    new_min = [min(a, b) for a, b in zip(cur_bounds[0], pt_min)]
    new_max = [max(a, b) for a, b in zip(cur_bounds[1], pt_max)]
    zv["bounds"] = [new_min, new_max]
    root.attrs.update({"zarr_vectors": zv})
    return positions, in_bounds


def set_bounds(
    root: Group,
    new_bounds: tuple[list[float], list[float]],
    *,
    force: bool = False,
) -> None:
    """Update the store's ``bounds`` after creation.

    - **Expanding** in any dimension (``new_min <= cur_min`` and
      ``new_max >= cur_max``) is always allowed.
    - **Contracting** in any dimension requires ``force=True``.  When
      forced, any vertices that fall outside the new bounds are pruned
      per-vertex from every level's ``vertices/`` array.  Auxiliary
      arrays (``object_index``, attributes) are NOT rewritten — re-run
      the type writer or :mod:`rechunk` if you need a fully consistent
      store after a contract.

    Args:
        root: Store root group returned by :func:`create_store` /
            :func:`open_store`.
        new_bounds: ``(min_corner, max_corner)`` for the new bounds.
        force: Required when any dimension is contracting.
    """
    import numpy as np

    zv = root.attrs.to_dict().get("zarr_vectors", {})
    cur = zv.get("bounds")
    if not cur:
        raise MetadataError("Store is missing `bounds`; cannot update.")
    cur_min = np.asarray(cur[0], dtype=float)
    cur_max = np.asarray(cur[1], dtype=float)
    new_min = np.asarray(new_bounds[0], dtype=float)
    new_max = np.asarray(new_bounds[1], dtype=float)
    if new_min.shape != cur_min.shape or new_max.shape != cur_max.shape:
        raise MetadataError(
            f"new_bounds shape {new_min.shape}/{new_max.shape} does not "
            f"match store ndim={len(cur_min)}"
        )
    if np.any(new_max < new_min):
        raise MetadataError(
            f"new_bounds inverted: max={new_max.tolist()} < min={new_min.tolist()}"
        )

    contracting = bool(np.any(new_min > cur_min) or np.any(new_max < cur_max))
    if contracting and not force:
        raise StoreError(
            f"new_bounds {new_bounds!r} contract the current bounds "
            f"{cur!r}; pass force=True to remove out-of-bounds vertices."
        )

    if contracting:
        _prune_vertices_outside_bounds(root, new_min, new_max)

    zv["bounds"] = [new_min.tolist(), new_max.tolist()]
    root.attrs.update({"zarr_vectors": zv})


def _prune_vertices_outside_bounds(
    root: Group,
    new_min: Any,  # np.ndarray
    new_max: Any,
) -> None:
    """Per-vertex prune every level's ``vertices/`` array to the new
    bounds.  See :func:`set_bounds`.
    """
    import numpy as np

    from zarr_vectors.core.arrays import (
        list_chunk_keys,
        read_chunk_vertices,
        write_chunk_vertices,
    )

    ndim = len(new_min)
    for level_idx in list_resolution_levels(root):
        level = get_resolution_level(root, level_idx)
        try:
            vmeta = level.read_array_meta(VERTICES)
        except Exception:
            continue
        dtype = np.dtype(vmeta.get("dtype", "float32"))
        for chunk_key in list_chunk_keys(level):
            try:
                vert_groups = read_chunk_vertices(
                    level, chunk_key, dtype=dtype, ndim=ndim,
                )
            except Exception:
                continue
            new_groups: list[Any] = []
            changed = False
            for fragment in vert_groups:
                if len(fragment) == 0:
                    new_groups.append(fragment)
                    continue
                in_b = np.all(
                    (fragment >= new_min) & (fragment <= new_max), axis=1,
                )
                if in_b.all():
                    new_groups.append(fragment)
                else:
                    changed = True
                    new_groups.append(fragment[in_b])
            if changed:
                write_chunk_vertices(level, chunk_key, new_groups, dtype=dtype)


def _create_or_open_store(
    path: StoreLike | Group,
    *,
    backend: str | None = None,
    bounds: tuple[list[float], list[float]] | None = None,
    chunk_shape: tuple[float, ...] | None = None,
    axes: list[dict[str, str]] | None = None,
    geometry_types: list[str] | None = None,
    ndim: int | None = None,
    **backend_kwargs: Any,
) -> Group:
    """Warm the store with :func:`create_store` if no store exists at
    ``path``, else open it read-write.  Writers route through this so
    callers can either ``create_store(path)`` first and then write, or
    write directly against a fresh path.

    Accepts an already-opened :class:`Group` (e.g. the return value of
    :func:`create_store` / :func:`open_store`) and returns it unchanged
    — this lets callers write through a handle they already hold.

    The create-only kwargs (``bounds`` / ``chunk_shape`` / ``axes`` /
    ``geometry_types`` / ``ndim``) are forwarded to :func:`create_store`
    on a fresh path and ignored when opening an existing store — the
    existing store's structural metadata stays authoritative.
    """
    # Pass-through for an already-opened Group handle.
    if isinstance(path, Group):
        return path

    creator_kwargs: dict[str, Any] = {}
    if bounds is not None:
        creator_kwargs["bounds"] = bounds
    if chunk_shape is not None:
        creator_kwargs["chunk_shape"] = chunk_shape
    if axes is not None:
        creator_kwargs["axes"] = axes
    if geometry_types is not None:
        creator_kwargs["geometry_types"] = geometry_types
    if ndim is not None:
        creator_kwargs["ndim"] = ndim

    if backend == "icechunk":
        try:
            return create_store(
                path, backend=backend, **creator_kwargs, **backend_kwargs,
            )
        except StoreError:
            return open_store(path, mode="r+", backend=backend, **backend_kwargs)
    from zarr.abc.store import Store as _ZStore
    if isinstance(path, _ZStore):
        # Pre-built stores can't be cheaply probed for emptiness; try to
        # open first, fall back to create.  Mirrors the icechunk pattern.
        try:
            return open_store(path, mode="r+", backend=backend, **backend_kwargs)
        except StoreError:
            return create_store(path, backend=backend, **backend_kwargs)
    if isinstance(path, (str, Path)):
        scheme = _detect_scheme(path)
        if scheme in {"", "file"}:
            local_root = _resolve_local_path(path)
            if local_root.exists() and local_root.is_dir() and any(local_root.iterdir()):
                return open_store(path, mode="r+", backend=backend, **backend_kwargs)
        else:
            # Non-local URL (gs://, s3://, http(s)://, ...): we cannot
            # cheaply probe for existence, so try ``open_store`` first
            # and fall back to ``create_store`` on StoreError.  Mirrors
            # the icechunk / pre-built-Store paths above.  Without this,
            # ``write_points`` (and other type writers) against an
            # existing remote store would silently re-run
            # ``create_store(mode="w")`` and clobber the NGFF
            # ``multiscales[0].axes`` block — the writer does not
            # forward the user's ``axes=`` kwarg, so axis units written
            # at create time would be replaced by unit-less defaults.
            try:
                return open_store(
                    path, mode="r+", backend=backend, **backend_kwargs,
                )
            except StoreError:
                pass
    return create_store(
        path, backend=backend, **creator_kwargs, **backend_kwargs,
    )


def _finalize_write(root: Group, message: str) -> str | None:
    """Commit the in-flight write transaction if the store is icechunk-backed.

    This is the auto-commit hook the high-level type writers
    (``write_points``, ``write_graph``, etc.) call just before
    returning their summary dict.  For non-transactional backends it
    is a no-op; for icechunk-backed stores it flushes the writable
    session opened by :func:`_create_or_open_store`, otherwise all
    writes would be discarded when the session is GC'd.

    Returns the new snapshot id on icechunk-backed stores, ``None``
    otherwise.
    """
    return commit(root, message)

def open_store(
    path: StoreLike,
    mode: str = "r",
    *,
    backend: str | None = None,
    **backend_kwargs: Any,
) -> Group:
    """Open an existing ZV store.

    Args:
        path: URL or filesystem path to the store.
        mode: ``"r"`` (read-only — writes will raise), ``"r+"``
            (read-write), ``"a"`` (append).  For ``mode="r"`` the
            underlying Zarr store is wrapped via
            ``store.with_read_only(True)`` when the store implementation
            supports it (the LocalStore and FsspecStore do; icechunk
            readonly sessions enforce read-only at the transaction
            layer).  Callers that need to mutate must open with
            ``mode="r+"``.
        backend: Force a particular backend (auto-detect by default).
        **backend_kwargs: Forwarded to the backend constructor.

    Returns:
        The root :class:`Group`.

    Raises:
        StoreError: If the store does not exist or is structurally invalid.
        MetadataError: If root metadata cannot be parsed.
    """
    # Local-FS existence check; transactional backends (icechunk) verify
    # repository existence inside their own session factory.  Pre-built
    # Store objects and cloud schemes skip the local check and rely on
    # ``zarr.open_group`` below to raise ``GroupNotFoundError``.
    from zarr.abc.store import Store as _ZStore
    if (
        backend != "icechunk"
        and not isinstance(path, _ZStore)
        and isinstance(path, (str, Path))
    ):
        scheme = _detect_scheme(path)
        if scheme in {"", "file"}:
            local_root = _resolve_local_path(path)
            if not local_root.is_dir():
                raise StoreError(f"Store not found at {local_root}")

    # Pass mode through so the icechunk backend can pick readonly_session
    # vs writable_session.  For non-icechunk backends the byte-level
    # store is wrapped read-only below when mode="r".
    store, session = _make_zarr_store_with_session(
        path, mode=mode, backend=backend, **backend_kwargs,
    )

    # Enforce read-only when mode="r": wrap the underlying Zarr store
    # via ``with_read_only(True)`` if the store supports it.  Callers in
    # ``zarr_vectors.lazy`` that resurrect a Group via
    # :meth:`Group._from_backend` already know how to unwrap this; the
    # standard path through ``open_store`` honours the requested mode.
    # Skip this for transactional backends (icechunk): they returned a
    # session and already picked the right readonly/writable session
    # inside their factory — wrapping here would create a new store
    # object and break the session linkage below.
    if session is None and mode == "r" and hasattr(store, "with_read_only"):
        try:
            store = store.with_read_only(True)
        except Exception:
            # Some Zarr store implementations don't support runtime
            # mode flips; fall back to opening read-only at the group
            # level only.
            pass
    if session is not None:
        store._zv_icechunk_session = session
    zarr_open_mode = "r" if mode == "r" else "r+"
    try:
        zg = zarr.open_group(store, mode=zarr_open_mode)
    except zarr.errors.GroupNotFoundError as e:
        raise StoreError(f"Not a valid ZV store at {path}: {e}") from None
    root = FsGroup.__new__(FsGroup) if isinstance(store, LocalStore) else Group.__new__(Group)
    root._zarr = zg

    attrs = root.attrs.to_dict()
    if "zarr_vectors" not in attrs:
        raise StoreError(
            f"Not a valid ZV store: missing 'zarr_vectors' in root attrs "
            f"at {root.url}"
        )
    # Permissive parse: a freshly-warmed store has no dims/bounds/chunk_
    # _shape yet.  The required `format_version` key is still enforced.
    RootMetadata.from_dict(attrs, strict=False)
    return root


# ===================================================================
# Transactional helpers (icechunk)
# ===================================================================


def session_for(group: Group) -> Any | None:
    """Return the underlying ``icechunk.Session`` if any, else ``None``.

    Looks up the session stashed on the group's Zarr store at
    construction time (see :func:`create_store` / :func:`open_store`).
    Works for root groups *and* any sub-group reached via
    ``root.create_group(...)`` / ``root[name]`` — they all share the
    same underlying Zarr store.

    Useful when callers need branch / snapshot operations that aren't
    surfaced by the zarr-vectors API (creating branches, tagging,
    listing snapshots, etc.).
    """
    zg = getattr(group, "_zarr", None)
    if zg is None:
        return None
    return getattr(zg.store, "_zv_icechunk_session", None)


def commit(group: Group, message: str = "zarr-vectors write") -> str | None:
    """Commit pending changes when the store is backed by a transactional
    backend (currently ``icechunk``).

    For non-transactional backends this is a no-op and returns ``None``;
    writes are durable as soon as they hit the store.

    For icechunk-backed stores this calls ``session.commit(message)``
    and returns the new snapshot id (a hex string).  The same session
    continues to be writable after the commit — subsequent ZV writes
    accumulate uncommitted state until the next ``commit(group, ...)``
    call.

    Args:
        group: A :class:`Group` returned by :func:`create_store` or
            :func:`open_store` (or any sub-group of one).
        message: Commit message; defaults to a placeholder.  Empty
            strings are rejected by icechunk.

    Returns:
        Snapshot id (``str``) for icechunk-backed groups, else ``None``.
    """
    session = session_for(group)
    if session is None:
        return None
    return session.commit(message)


def discard_changes(group: Group) -> None:
    """Drop uncommitted changes on a transactional backend.

    No-op for non-transactional backends.
    """
    session = session_for(group)
    if session is None:
        return
    discard = getattr(session, "discard_changes", None)
    if discard is not None:
        discard()


# ===================================================================
# icechunk branch / rebase / merge wrappers (Iteration 2)
# ===================================================================

def branch(group: Group, name: str, *, from_snapshot_id: str | None = None) -> str:
    """Create a new icechunk branch off the current session's tip.

    Returns the snapshot id the branch was anchored at.  Raises
    :class:`StoreError` on non-icechunk backends.
    """
    session = session_for(group)
    if session is None:
        raise StoreError(
            "branch() requires an icechunk-backed store; this store "
            "has no icechunk session."
        )
    from zarr_vectors.core.backends.icechunk_backend import create_branch_wrapper
    return create_branch_wrapper(session, name, from_snapshot_id=from_snapshot_id)


def switch_branch(group: Group, name: str) -> None:
    """Swap the group's underlying session to a writable session on
    branch ``name``.

    Existing references to ``group`` continue to work; the swap is
    transparent.  Pending uncommitted edits on the previous session are
    discarded — caller should commit() first if they want to keep them.
    """
    session = session_for(group)
    if session is None:
        raise StoreError(
            "switch_branch() requires an icechunk-backed store."
        )
    from zarr_vectors.core.backends.icechunk_backend import switch_branch_wrapper
    new_session = switch_branch_wrapper(session, name)
    # Replace the session on the underlying Zarr store + rewire the
    # zarr.Group so future writes go through the new session's store.
    store = new_session.store
    store._zv_icechunk_session = new_session  # type: ignore[attr-defined]
    new_zg = zarr.open_group(store, mode="r+")
    group._zarr = new_zg


def rebase(group: Group, base: str = "main") -> None:
    """Rebase the group's session's branch onto ``base``."""
    session = session_for(group)
    if session is None:
        raise StoreError("rebase() requires an icechunk-backed store.")
    from zarr_vectors.core.backends.icechunk_backend import rebase_wrapper
    rebase_wrapper(session, base)


def merge_branch(
    group: Group,
    name: str,
    *,
    message: str = "merge branch",
) -> str | None:
    """Merge branch ``name`` into the group's current branch."""
    session = session_for(group)
    if session is None:
        raise StoreError("merge_branch() requires an icechunk-backed store.")
    from zarr_vectors.core.backends.icechunk_backend import merge_branch_wrapper
    return merge_branch_wrapper(session, name, message=message)


def rebind(
    group: Group,
    backend: str | Any,
    **backend_kwargs: Any,
) -> Group:
    """Re-open the underlying store with a different driver (no data move).

    Under the Zarr-native layer, ``rebind`` opens a new Zarr store at
    the same URL and swaps it in.  For phases 1-3 only the local Zarr
    store is supported, so this is effectively a no-op unless the
    caller explicitly passes a different store.
    """
    old_url = group.url
    if isinstance(backend, str):
        new_store = _make_zarr_store(old_url, backend=backend, **backend_kwargs)
        new_url = old_url
    elif hasattr(backend, "set"):  # zarr.abc.store.Store duck-type
        new_store = backend
        new_url = group.url  # accept as-is
    else:
        # Legacy StorageBackend-shaped object (LocalBackend etc.) —
        # require its declared URL match the current one so we catch
        # programming mistakes rather than silently no-op'ing.
        new_url = getattr(backend, "url", None)
        if new_url is None:
            return group
        if _canonical(new_url) != _canonical(old_url):
            raise StoreError(
                f"rebind requires matching URLs; current store is at "
                f"{old_url!r}, new backend is at {new_url!r}. Use "
                f"open_store to point at a different location."
            )
        return group

    if _canonical(new_url) != _canonical(old_url):
        raise StoreError(
            f"rebind requires matching URLs; current store is at "
            f"{old_url!r}, new backend is at {new_url!r}. Use "
            f"open_store to point at a different location."
        )
    zg = zarr.open_group(new_store, mode="r+")
    group._zarr = zg
    return group


def _canonical(url: str) -> str:
    """Normalise a URL for cross-backend equality checks."""
    return url.rstrip("/").lower()


# ===================================================================
# Resolution levels
# ===================================================================


def create_resolution_level(
    root: Group,
    level: int,
    level_metadata: LevelMetadata,
) -> Group:
    """Create a new resolution level group within the store.

    The level's spatial transform (``bin_ratio`` → NGFF ``scale``,
    ``bin_shape`` → NGFF ``translation = bin_shape / 2``) is written to
    the NGFF ``multiscales[0].datasets`` block via
    :func:`zarr_vectors.core.multiscale.upsert_level_transform` so the
    NGFF block stays the **single source of truth** for spatial
    geometry.  ``bin_shape`` and ``bin_ratio`` are intentionally
    omitted from the level's own ``zarr_vectors_level`` attrs; readers
    derive them from the NGFF block (see :func:`read_level_metadata`).
    """
    level_metadata.validate()
    group_name = f"{RESOLUTION_PREFIX}{level}"
    level_group = root.require_group(group_name)
    payload = level_metadata.to_dict()
    payload["zarr_vectors_level"].pop("bin_shape", None)
    payload["zarr_vectors_level"].pop("bin_ratio", None)
    level_group.attrs.update(payload)

    # Mirror the spatial transform into the NGFF block.
    from zarr_vectors.core.multiscale import upsert_level_transform
    if level == 0:
        # Level 0: scale = 1.0 per-axis; translation seeded by base_bin/2 if known.
        try:
            root_meta = read_root_metadata(root)
            ndim = root_meta.sid_ndim
            base_bin = root_meta.effective_bin_shape
            scale = [1.0] * ndim
            translation = [bs / 2.0 for bs in base_bin]
        except Exception:
            scale = [1.0]
            translation = None
        upsert_level_transform(
            root, level, scale=scale, translation=translation,
        )
    elif level_metadata.bin_ratio is not None or level_metadata.bin_shape is not None:
        try:
            root_meta = read_root_metadata(root)
            ndim = root_meta.sid_ndim
        except Exception:
            ndim = (
                len(level_metadata.bin_ratio) if level_metadata.bin_ratio
                else (len(level_metadata.bin_shape) if level_metadata.bin_shape else 1)
            )
        if level_metadata.bin_ratio is not None:
            scale = [float(r) for r in level_metadata.bin_ratio]
        else:
            scale = [1.0] * ndim
        translation = (
            [bs / 2.0 for bs in level_metadata.bin_shape]
            if level_metadata.bin_shape is not None
            else None
        )
        upsert_level_transform(root, level, scale=scale, translation=translation)
    return level_group


def get_resolution_level(root: Group, level: int) -> Group:
    """Get an existing resolution level group.

    Raises:
        StoreError: If the level does not exist.
    """
    group_name = f"{RESOLUTION_PREFIX}{level}"
    if group_name not in root:
        raise StoreError(f"Resolution level {level} not found in store")
    return root[group_name]


def list_resolution_levels(root: Group) -> list[int]:
    """Return sorted level indices present in the store.

    Resolution-level group names are bare integers (``0``, ``1``, ...)
    under the 0.4.1+ layout (formerly ``resolution_0`` /
    ``resolution_1``).  Top-level groups whose name doesn't parse as
    ``int`` are some other entity (e.g. ``parametric/``) and are
    silently skipped.
    """
    levels: list[int] = []
    for name in root:
        # Tolerate the legacy ``RESOLUTION_PREFIX`` slice in case it is
        # ever re-introduced; the empty prefix in 0.4.1+ makes this a
        # no-op.
        candidate = name[len(RESOLUTION_PREFIX):] if RESOLUTION_PREFIX else name
        try:
            levels.append(int(candidate))
        except ValueError:
            continue
    return sorted(levels)


def get_parametric_group(root: Group) -> Group:
    """Get the ``/parametric/`` group, creating it if needed."""
    return root.require_group(PARAMETRIC_GROUP)


def read_root_metadata(root: Group) -> RootMetadata:
    """Read and parse root metadata from the store."""
    return RootMetadata.from_dict(root.attrs.to_dict())


def read_level_metadata(root: Group, level: int) -> LevelMetadata:
    """Read and parse level metadata.

    ``bin_shape`` and ``bin_ratio`` are read from the NGFF
    ``multiscales[0].datasets`` block (the authoritative source under
    0.5+).  Legacy stores that still carry them under
    ``zarr_vectors_level`` are honoured as a fallback.

    Raises:
        StoreError: If the level does not exist.
        MetadataError: If metadata is malformed.
    """
    level_group = get_resolution_level(root, level)
    lm = LevelMetadata.from_dict(level_group.attrs.to_dict())

    # NGFF takes precedence over the legacy fields when present.
    from zarr_vectors.core.multiscale import read_level_transform
    scale, translation = read_level_transform(root, level)
    if scale is not None:
        if level == 0:
            # Level 0 has scale == 1.0 by convention; bin_ratio stays None.
            lm.bin_ratio = None
            lm.bin_shape = None
        else:
            lm.bin_ratio = tuple(int(round(s)) for s in scale)
            if translation is not None:
                lm.bin_shape = tuple(2.0 * t for t in translation)
    return lm


def write_parametric_types(
    root: Group,
    types: list[ParametricTypeDef],
) -> None:
    """Write parametric type registry to ``/parametric/.zattrs``."""
    para = get_parametric_group(root)
    para.attrs.update(serialise_parametric_types(types))


def read_parametric_types(root: Group) -> list[ParametricTypeDef]:
    """Read parametric type registry from ``/parametric/.zattrs``."""
    para = get_parametric_group(root)
    return deserialise_parametric_types(para.attrs.to_dict())


def store_info(root: Group) -> dict[str, Any]:
    """Return summary information about a ZV store."""
    meta = read_root_metadata(root)
    levels = list_resolution_levels(root)

    level_summaries: list[dict[str, Any]] = []
    for lvl in levels:
        try:
            lm = read_level_metadata(root, lvl)
            level_summaries.append({
                "level": lm.level,
                "vertex_count": lm.vertex_count,
                "bin_shape": list(lm.bin_shape) if lm.bin_shape else None,
                "bin_ratio": list(lm.bin_ratio) if lm.bin_ratio else None,
                "object_sparsity": lm.object_sparsity,
                "coarsening_method": lm.coarsening_method,
                "arrays_present": lm.arrays_present,
            })
        except (MetadataError, StoreError):
            level_summaries.append({"level": lvl, "error": "unreadable"})

    ptypes = read_parametric_types(root)

    return {
        "zv_version": meta.zv_version,
        "geometry_types": meta.geometry_types,
        "spatial_index_dims": meta.spatial_index_dims,
        "chunk_shape": list(meta.chunk_shape),
        "base_bin_shape": list(meta.base_bin_shape) if meta.base_bin_shape else None,
        "bins_per_chunk": list(meta.bins_per_chunk),
        "bounds": [list(meta.bounds[0]), list(meta.bounds[1])],
        "links_convention": meta.links_convention,
        "object_index_convention": meta.object_index_convention,
        "cross_chunk_strategy": meta.cross_chunk_strategy,
        "reduction_factor": meta.reduction_factor,
        "levels": level_summaries,
        "parametric_types": [t.to_dict() for t in ptypes],
    }


# ===================================================================
# Manual level management
# ===================================================================


def add_resolution_level(
    root: Group,
    level_index: int,
    bin_ratio: tuple[int, ...],
    *,
    object_sparsity: float = 1.0,
    coarsening_method: str = "manual",
    parent_level: int | None = None,
) -> Group:
    """Create a new resolution level with a specified bin ratio."""
    from zarr_vectors.core.metadata import (
        compute_bin_shape,
        validate_bin_shape_divides_chunk,
    )

    meta = read_root_metadata(root)
    base_bin = meta.effective_bin_shape
    chunk_shape = meta.chunk_shape

    bin_shape = compute_bin_shape(base_bin, bin_ratio)
    validate_bin_shape_divides_chunk(chunk_shape, bin_shape)

    if parent_level is None:
        parent_level = max(0, level_index - 1)

    group_name = f"{RESOLUTION_PREFIX}{level_index}"
    if group_name in root:
        raise StoreError(f"Resolution level {level_index} already exists")

    level_meta = LevelMetadata(
        level=level_index,
        vertex_count=0,
        arrays_present=[],
        bin_shape=bin_shape,
        bin_ratio=bin_ratio,
        object_sparsity=object_sparsity,
        coarsening_method=coarsening_method,
        parent_level=parent_level,
    )
    level_meta.validate()
    # Route through create_resolution_level so the NGFF block is updated
    # via :func:`zarr_vectors.core.multiscale.upsert_level_transform` in
    # the same way as a writer-driven create.
    return create_resolution_level(root, level_index, level_meta)


def remove_resolution_level(root: Group, level_index: int) -> None:
    """Remove a resolution level from the store.

    Level 0 cannot be removed.

    Raises:
        StoreError: If the level does not exist or is level 0.
    """
    if level_index == 0:
        raise StoreError("Cannot remove level 0 (full resolution)")

    group_name = f"{RESOLUTION_PREFIX}{level_index}"
    if group_name not in root:
        raise StoreError(f"Resolution level {level_index} not found")

    root.delete_subtree(group_name)


def list_available_ratios(root: Group) -> list[tuple[int, ...]]:
    """Return bin ratios for all existing resolution levels."""
    from zarr_vectors.core.metadata import compute_bin_ratio

    meta = read_root_metadata(root)
    base_bin = meta.effective_bin_shape
    ndim = meta.sid_ndim
    levels = list_resolution_levels(root)
    ratios: list[tuple[int, ...]] = []

    for lvl in levels:
        if lvl == 0:
            ratios.append(tuple(1 for _ in range(ndim)))
            continue
        try:
            lm = read_level_metadata(root, lvl)
            if lm.bin_ratio is not None:
                ratios.append(lm.bin_ratio)
            elif lm.bin_shape is not None:
                ratios.append(compute_bin_ratio(base_bin, lm.bin_shape))
            else:
                ratios.append(tuple(1 for _ in range(ndim)))
        except Exception:
            ratios.append(tuple(1 for _ in range(ndim)))

    return ratios
