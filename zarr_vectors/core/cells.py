"""Many cells of many arrays in one read, returned as flat arrays.

The per-cell readers answer "this chunk's vertices"; a pipeline that works
on a chunk and its neighbours asks that 27 times per array, one round trip
each. :func:`read_cells` reads any set of cells of any set of per-chunk
arrays with one batched prefetch, and returns each array as one flat
``data`` array plus CSR ``offsets`` over the requested cells -- the shape
an array program wants, and the one that crosses to a device in a single
copy (``device="cuda"``).

The rows are each cell's *stored* rows, as :func:`read_chunk_vertex_buffer`
returns them: not filtered by fragment, so per-vertex attribute rows line
up 1:1 with vertex rows.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from zarr_vectors import _xp
from zarr_vectors.constants import (
    FRAGMENT_ATTRIBUTES,
    LINK_ATTRIBUTES,
    LINK_FRAGMENTS,
    LINKS,
    VERTEX_ATTRIBUTES,
    VERTEX_FRAGMENTS,
    VERTICES,
)
from zarr_vectors.exceptions import ArrayError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zarr_vectors.core.group import Group


@dataclass(frozen=True)
class CellReadError:
    """A cell that could not be read; the rest of the batch still was."""

    array: str
    chunk_key: str
    message: str


@dataclass(frozen=True)
class ArrayRead:
    """How one array's cells reached a :class:`CellBatch`.

    ``path`` is one of:

    - ``gds``: stored bytes read with kvikio, GPUDirect Storage on;
    - ``kvikio-compat``: read with kvikio through cuFile's bounce buffer;
    - ``pinned-host``: plain reads into pinned host memory, one copy up;
    - ``store``: byte-range gets from a store that is not a local
      directory, one copy up;
    - ``host``: read and decoded on the host (a host read, or an array
      a device read left to the host).

    The first four decode on the device. ``stored_bytes`` is what they
    read from storage, before decompression; ``None`` for ``host``, where
    zarr does not say.
    """

    array: str
    path: str
    cells: int
    stored_bytes: int | None


@dataclass(frozen=True)
class CellColumn:
    """One array's rows across the requested cells.

    ``data[offsets[i]:offsets[i + 1]]`` are cell ``i``'s rows. Both live
    on the batch's device.
    """

    data: Any
    offsets: Any
    _host_offsets: npt.NDArray[np.int64] = field(repr=False, compare=False)

    @property
    def num_cells(self) -> int:
        return len(self._host_offsets) - 1

    def rows(self, i: int) -> Any:
        """Cell ``i``'s rows (a view of ``data``)."""
        lo, hi = self._host_offsets[i], self._host_offsets[i + 1]
        return self.data[int(lo):int(hi)]

    @cached_property
    def cell(self) -> Any:
        """The cell index of every row, on ``data``'s device (int32)."""
        counts = np.diff(self._host_offsets)
        host = np.repeat(np.arange(self.num_cells, dtype=np.int32), counts)
        return host if not _xp.is_device_array(self.data) else _xp.to_device(host, "cuda")


@dataclass(frozen=True)
class FragmentColumn:
    """The vertex fragments of every requested cell, as flat arrays.

    Cell ``i``'s fragments are ``cell_offsets[i]:cell_offsets[i + 1]``,
    in the order its fragment index lists them. Fragment ``f`` holds
    ``counts[f]`` of the cell's vertex rows, numbered from the cell's
    first row in the ``vertices`` column: rows ``starts[f]`` to
    ``starts[f] + counts[f]`` when it is a range, and otherwise
    (``starts[f] == -1``) rows ``indices[index_offsets[f]:index_offsets[f + 1]]``.
    Range fragments have no entries in ``indices``. All on the batch's
    device.

    A store written bin by bin (a point cloud, say) holds fragments that
    tile each cell's rows in bin order, empty bins included, so cell
    ``i``'s bin boundaries are ``0`` followed by the running sum of
    ``counts[a:b]``, with ``a, b = cell_offsets[i], cell_offsets[i + 1]``.
    """

    cell_offsets: Any
    starts: Any
    counts: Any
    index_offsets: Any
    indices: Any

    def to_device(self, device: str) -> FragmentColumn:
        return FragmentColumn(**{
            name: _xp.to_device(getattr(self, name), device)
            for name in ("cell_offsets", "starts", "counts", "index_offsets", "indices")
        })


@dataclass(frozen=True)
class CellBatch(Mapping[str, CellColumn]):
    """The result of :func:`read_cells`: one :class:`CellColumn` per array.

    ``chunk_coords`` is the requested cells in request order, always on
    the host; the columns are on ``device``.

    ``fragments`` is the vertices' fragment index when it was asked for
    (``read_cells(..., fragments=True)``), else None.

    ``io`` says how each array read was served (:class:`ArrayRead`), and
    ``io_seconds`` where the time went: ``fetch`` and ``decode`` for the
    arrays decoded on the device, ``host`` for the rest (reading and
    decoding together). Neither takes part in equality.
    """

    chunk_coords: npt.NDArray[np.int64]
    columns: dict[str, CellColumn]
    errors: tuple[CellReadError, ...] = ()
    device: str = "cpu"
    fragments: FragmentColumn | None = None
    io: tuple[ArrayRead, ...] = field(default=(), compare=False)
    io_seconds: dict[str, float] = field(default_factory=dict, compare=False)

    def __getitem__(self, name: str) -> CellColumn:
        return self.columns[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.columns)

    def __len__(self) -> int:
        return len(self.columns)

    def to_device(self, device: str) -> CellBatch:
        """The same batch on ``device``: one copy per data/offsets array."""
        device = _xp.resolve_device(device)
        moved = {
            name: replace(
                col,
                data=_xp.to_device(col.data, device),
                offsets=_xp.to_device(col.offsets, device),
            )
            for name, col in self.columns.items()
        }
        fragments = None if self.fragments is None else self.fragments.to_device(device)
        return replace(self, columns=moved, fragments=fragments, device=device)


# --------------------------------------------------------------------


_Kind = Literal["vertices", "vertex_attr", "fragment_attr", "links", "link_attr"]


def _kind_of(name: str) -> _Kind:
    parts = name.split("/")
    if name == VERTICES:
        return "vertices"
    if name in (VERTEX_FRAGMENTS, LINK_FRAGMENTS):
        raise ArrayError(
            f"read_cells does not decode the fragment index {name!r}; read "
            f"it with read_vertex_fragment_index / read_link_fragment_index"
        )
    if parts[0] == VERTEX_ATTRIBUTES and len(parts) == 2:
        return "vertex_attr"
    if parts[0] == FRAGMENT_ATTRIBUTES and len(parts) == 2:
        return "fragment_attr"
    if parts[0] == LINKS and len(parts) == 3:
        return "links"
    if parts[0] == LINK_ATTRIBUTES and len(parts) == 4:
        return "link_attr"
    raise ArrayError(
        f"read_cells cannot read {name!r}: expected vertices, "
        f"vertex_attributes/<name>, fragment_attributes/<name>, "
        f"links/<delta>/<offsets> or link_attributes/<name>/<delta>/<offsets>"
    )


@dataclass
class _Layout:
    """How to turn one array's cell bytes into rows."""

    kind: _Kind
    dtype: np.dtype
    row_shape: tuple[int, ...] | None  # None: width inferred per cell
    ncols: int = 0          # links: physical columns
    flat: bool = False      # links: flat (intra, delta 0) vs ragged


def _layout(level_group: Group, name: str, kind: _Kind, ndim: int) -> _Layout:
    from zarr_vectors.core.arrays import (
        _resolve_attribute_layout,
        links_has_perm,
        vertices_dtype,
    )
    from zarr_vectors.core.paths import (
        is_intra,
        links_group_path,
        parse_delta,
        parse_offsets,
    )

    if kind == "vertices":
        return _Layout(kind, vertices_dtype(level_group), (ndim,) if ndim > 1 else ())
    meta = level_group.read_array_meta(name) or {}
    if kind in ("vertex_attr", "fragment_attr"):
        stamped = meta.get("row_shape") is not None or meta.get("channel_names")
        dtype, ncols = _resolve_attribute_layout(level_group, name, None, None)
        if kind == "vertex_attr" and not stamped:
            # Width is whatever divides the cell against its vertex rows.
            return _Layout(kind, dtype, None)
        return _Layout(kind, dtype, (ncols,) if ncols > 1 else ())
    if kind == "link_attr":
        return _Layout(
            kind,
            np.dtype(meta.get("dtype", "float32")),
            tuple(int(x) for x in meta.get("row_shape", ())),
        )
    # links/<delta>/<offsets>
    _, delta_seg, seg = name.split("/")
    delta = parse_delta(delta_seg)
    fam = level_group.read_array_meta(links_group_path(delta)) or {}
    if "link_width" not in fam or "sid_ndim" not in fam:
        raise ArrayError(f"{name!r}: its family records no link_width/sid_ndim")
    link_width, sid_ndim = int(fam["link_width"]), int(fam["sid_ndim"])
    offsets = parse_offsets(seg, sid_ndim=sid_ndim, link_width=link_width)
    has_perm = bool(meta.get("has_perm", links_has_perm(
        offsets, delta=delta, directed=bool(fam.get("directed", False)),
        store=str(fam.get("store", "canonical")),
    )))
    ncols = link_width + (1 if has_perm else 0)
    return _Layout(
        kind, np.dtype(meta.get("dtype", "int64")), (ncols,),
        ncols=ncols, flat=delta == 0 and is_intra(offsets),
    )


def _decode(raw: bytes, lay: _Layout, vertex_rows: int | None) -> np.ndarray:
    from zarr_vectors.core.arrays import _link_cell_rows

    if lay.kind == "links":
        if not raw:
            return np.empty((0, lay.ncols), dtype=np.int64)
        return _link_cell_rows(raw, ncols=lay.ncols, flat=lay.flat, dtype=lay.dtype)
    flat = np.frombuffer(raw, dtype=lay.dtype)
    if lay.row_shape is not None:
        return flat.reshape(-1, *lay.row_shape)
    # A vertex attribute with no stamped width: one row per vertex row.
    if flat.size == 0:
        return flat.reshape(0, 1)
    if not vertex_rows or flat.size % vertex_rows:
        raise ArrayError(
            f"{flat.size} attribute elements do not divide into this cell's "
            f"{vertex_rows or 0} vertex rows"
        )
    return flat.reshape(vertex_rows, flat.size // vertex_rows)


@contextlib.contextmanager
def _prefetch(level_group: Group, plan: list[tuple[str, list[str]]]):
    """One tolerant prefetch, unless an outer one (or a replay) serves us."""
    if not plan or level_group._prefetch_cache is not None:
        yield
        return
    with level_group.batched_reads(plan, tolerant=True):
        yield


def read_cells(
    level_group: Group,
    chunk_coords: Any,
    arrays: Sequence[str] = (VERTICES,),
    *,
    ndim: int | None = None,
    missing_arrays: Literal["raise", "skip"] = "raise",
    on_error: Literal["record", "raise"] = "record",
    device: str | None = None,
    decode: Literal["auto", "host", "device"] = "auto",
    io: Literal["auto", "kvikio", "host"] = "auto",
    fragments: bool = False,
) -> CellBatch:
    """Read the cells ``chunk_coords`` of every array in ``arrays`` at once.

    Args:
        level_group: Resolution level group.
        chunk_coords: ``(C, K)`` integer chunk coordinates (a list of
            tuples is fine; a device array is copied off once). The
            result keeps this order, duplicates included.
        arrays: Per-chunk arrays to read: ``vertices``,
            ``vertex_attributes/<name>``, ``fragment_attributes/<name>``,
            ``links/<delta>/<offsets>`` (physical rows, perm column
            included) or ``link_attributes/<name>/<delta>/<offsets>``.
        ndim: Vertex coordinate width; ``None`` reads it from the store.
        missing_arrays: ``"skip"`` leaves an array that does not exist
            out of the result instead of raising.
        on_error: ``"record"`` puts a cell that cannot be read in
            ``errors`` and gives it no rows; ``"raise"`` re-raises.
        device: ``"cpu"``, ``"cuda"`` or ``None`` (where ``chunk_coords``
            lives). Each returned array crosses to the device once.
        decode: Where a ``device="cuda"`` read decodes its cells; the
            result is the same either way. ``"auto"`` decodes on the
            device every uncompressed array (vlen-bytes cells, sharded or
            not) and the rest on the host. ``"device"`` also decodes zstd
            arrays there, through nvCOMP, and raises for an array it
            cannot decode instead of falling back. nvCOMP trusts its
            input: a zstd cell corrupted inside a block can return wrong
            bytes, hang, or crash the CUDA context where the host decoder
            raises, so use it only on stores you trust. ``"host"``
            decodes everything on the host and uploads the result.
            Ignored for a host read.
        io: How a device decode reads a local store's files:
            ``"kvikio"`` straight into device memory, ``"host"`` into
            pinned host memory and one copy up. ``"auto"`` takes
            ``$ZARR_VECTORS_GPU_IO`` if set, else ``kvikio`` only where
            cuFile reports GPUDirect Storage available
            (:func:`zarr_vectors._gds.choose_io`). The result is the same
            either way. Ignored for a host read and for other stores.
        fragments: Also return the vertices' fragment index, as
            ``batch.fragments`` (:class:`FragmentColumn`): how each cell's
            vertex rows split into fragments -- a point cloud's bins, for
            instance, without working them out again from coordinates.
            Read in the same prefetch and decoded on the host (a range
            fragment is 16 bytes), then moved to ``device``.

    Returns:
        A :class:`CellBatch`. A cell nobody wrote, or outside an array's
        grid, has no rows.
    """
    from zarr_vectors._gds import check_io
    from zarr_vectors.core.arrays import _chunk_key, _infer_vert_ndim

    check_io(io)
    device = _xp.resolve_device(device, chunk_coords)
    cc = _xp.to_host(chunk_coords, dtype=np.int64)
    if cc.ndim == 1 and cc.size == 0:
        cc = cc.reshape(0, 0)
    if cc.ndim != 2:
        raise ArrayError(f"chunk_coords must be (C, K); got shape {cc.shape}")
    keys = [_chunk_key(tuple(row)) for row in cc.tolist()]
    if ndim is None:
        ndim = _infer_vert_ndim(level_group)

    layouts: dict[str, _Layout] = {}
    for name in dict.fromkeys(arrays):
        kind = _kind_of(name)
        if level_group._sharded_chunk_array(name) is None:
            if missing_arrays == "skip":
                continue
            raise ArrayError(f"{name!r} does not exist at this level")
        layouts[name] = _layout(level_group, name, kind, ndim)
    needs_vertices = any(lay.row_shape is None for lay in layouts.values())

    # Which requested cells each array can hold; others read as empty.
    def _in_grid(name: str) -> list[bool]:
        bounds = level_group.chunk_grid_bounds(name)
        if bounds is None:
            return [False] * len(keys)
        origin, shape = bounds
        origin = origin or (0,) * len(shape)
        if cc.shape[1] != len(shape):
            return [False] * len(keys)
        rel = cc - np.asarray(origin, dtype=np.int64)
        return list(np.all((rel >= 0) & (rel < np.asarray(shape)), axis=1))

    read_names = list(layouts)
    if needs_vertices and VERTICES not in layouts:
        read_names.append(VERTICES)
    in_grid = {name: _in_grid(name) for name in read_names}
    plan = [
        (name, sorted({k for k, ok in zip(keys, in_grid[name]) if ok}))
        for name in read_names
    ]
    fragment_plan: tuple[str, list[str]] | None = None
    if fragments:
        if level_group._sharded_chunk_array(VERTEX_FRAGMENTS) is not None:
            ok = _in_grid(VERTEX_FRAGMENTS)
            fragment_plan = (VERTEX_FRAGMENTS, sorted({k for k, o in zip(keys, ok) if o}))
        elif missing_arrays != "skip":
            raise ArrayError(f"{VERTEX_FRAGMENTS!r} does not exist at this level")

    errors: list[CellReadError] = []
    seconds: dict[str, float] = {}
    on_device = _device_payloads(
        level_group, plan, layouts, keys, cc, decode if device == "cuda" else "host",
        on_error, errors, io, seconds,
    )
    raw: dict[tuple[str, str], bytes] = {}
    host_plan = [p for p in plan if p[0] not in on_device]
    if fragment_plan is not None:
        host_plan.append(fragment_plan)
    t_host = time.perf_counter()
    with _prefetch(level_group, [p for p in host_plan if p[1]]):
        for name, cell_keys in host_plan:
            for key in cell_keys:
                try:
                    raw[(name, key)] = level_group.read_bytes(name, key)
                except Exception as exc:  # noqa: BLE001 - recorded or re-raised
                    if on_error == "raise":
                        raise
                    errors.append(CellReadError(name, key, f"{type(exc).__name__}: {exc}"))

    vertex_rows: dict[str, int] = {}
    if needs_vertices:
        itemsize = np.dtype(_layout(level_group, VERTICES, "vertices", ndim).dtype).itemsize
        for key in keys:
            stored = (
                on_device[VERTICES].length(key) if VERTICES in on_device
                else len(raw.get((VERTICES, key), b""))
            )
            vertex_rows[key] = stored // (itemsize * ndim)

    host_seconds = time.perf_counter() - t_host
    columns: dict[str, CellColumn] = {}
    for name, lay in layouts.items():
        if name in on_device:
            from zarr_vectors.gpu import _read as device_read

            t0 = time.perf_counter()
            columns[name] = device_read.column(
                name, on_device[name], keys, lay, vertex_rows, on_error, errors,
            )
            seconds["decode"] = seconds.get("decode", 0.0) + time.perf_counter() - t0
            continue
        t0 = time.perf_counter()
        parts: list[np.ndarray] = []
        for key in keys:
            try:
                parts.append(_decode(raw.get((name, key), b""), lay, vertex_rows.get(key)))
            except Exception as exc:  # noqa: BLE001
                if on_error == "raise":
                    raise
                errors.append(CellReadError(name, key, f"{type(exc).__name__}: {exc}"))
                parts.append(None)  # type: ignore[arg-type]
        columns[name] = _column(parts, lay, device)
        host_seconds += time.perf_counter() - t0
    fragment_column = None
    if fragment_plan is not None:
        t0 = time.perf_counter()
        fragment_column = _fragment_column(keys, raw, on_error, errors, device)
        host_seconds += time.perf_counter() - t0

    if on_device:
        # The last gathers are still running; count them as decode time.
        t0 = time.perf_counter()
        _xp._gpu().xp.cuda.get_current_stream().synchronize()
        seconds["decode"] = seconds.get("decode", 0.0) + time.perf_counter() - t0
    if host_plan:
        seconds["host"] = host_seconds
    io_report = tuple(
        ArrayRead(name, on_device[name].path, len(cell_keys), on_device[name].stored_bytes)
        if name in on_device else ArrayRead(name, "host", len(cell_keys), None)
        for name, cell_keys in plan + ([fragment_plan] if fragment_plan else [])
    )
    return CellBatch(
        chunk_coords=cc, columns=columns, errors=tuple(errors), device=device,
        fragments=fragment_column, io=io_report, io_seconds=seconds,
    )


def _fragment_column(
    keys: list[str],
    raw: dict[tuple[str, str], bytes],
    on_error: str,
    errors: list[CellReadError],
    device: str,
) -> FragmentColumn:
    """The requested cells' fragment indices, decoded, as one column."""
    from zarr_vectors.encoding.fragments import decode_fragments

    empty = np.zeros(0, dtype=np.int64)
    decoded: dict[str, tuple[np.ndarray, ...]] = {}
    for key in dict.fromkeys(keys):
        blob = raw.get((VERTEX_FRAGMENTS, key), b"")
        if not blob:
            decoded[key] = (empty,) * 4
            continue
        try:
            decoded[key] = decode_fragments(blob).flat()
        except Exception as exc:  # noqa: BLE001 - recorded or re-raised
            if on_error == "raise":
                raise
            errors.append(CellReadError(VERTEX_FRAGMENTS, key, f"{type(exc).__name__}: {exc}"))
            decoded[key] = (empty,) * 4
    parts = [decoded[k] for k in keys]
    per_cell = np.array([len(p[0]) for p in parts], dtype=np.int64)
    starts, counts, index_counts, indices = (
        np.concatenate([p[j] for p in parts]) if parts else empty for j in range(4)
    )
    return FragmentColumn(
        cell_offsets=np.concatenate([[0], np.cumsum(per_cell)]).astype(np.int64),
        starts=starts,
        counts=counts,
        index_offsets=np.concatenate([[0], np.cumsum(index_counts)]).astype(np.int64),
        indices=indices,
    ).to_device(device)


def _device_payloads(
    level_group: Group,
    plan: list[tuple[str, list[str]]],
    layouts: dict[str, _Layout],
    keys: list[str],
    cc: np.ndarray,
    decode: str,
    on_error: str,
    errors: list[CellReadError],
    io: str = "auto",
    seconds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Fetch and unframe on the device every array it can decode.

    Returns ``{array: Payloads}`` for the arrays read there; the others
    are left to the host path. Empty for a host read.
    """
    if decode == "host":
        return {}
    if decode not in ("auto", "device"):
        raise ArrayError(f"decode={decode!r}; expected 'auto', 'host' or 'device'")
    _xp._gpu()  # the extension, or the install hint
    from zarr_vectors.gpu import _read as device_read

    coords_of = dict(zip(keys, cc.tolist()))
    items, names = [], []
    for name, cell_keys in plan:
        lay = layouts.get(name)
        src = device_read.supports(level_group, name, lay, zstd=decode == "device")
        if src is None:
            if decode == "device":
                raise ArrayError(
                    f"{name!r} cannot be decoded on the device (its codecs or "
                    f"store are not supported there, or nvCOMP is missing); "
                    f"use decode='auto' to decode it on the host"
                )
            continue
        items.append((
            src, cell_keys,
            np.asarray([coords_of[k] for k in cell_keys], dtype=np.int64),
            lay is not None and lay.kind == "links" and not lay.flat,
        ))
        names.append(name)
    out: dict[str, Any] = {}
    if not items:
        return out
    for name, payloads in zip(names, device_read.fetch_payloads(items, io=io, seconds=seconds)):
        for key, message in payloads.errors.items():
            if on_error == "raise":
                raise ArrayError(f"{name} cell {key}: {message}")
            errors.append(CellReadError(name, key, f"ArrayError: {message}"))
        out[name] = payloads
    return out


def _column(parts: list[np.ndarray | None], lay: _Layout, device: str) -> CellColumn:
    good = [p for p in parts if p is not None and p.size]
    if lay.kind == "links":
        tail, dtype = (lay.ncols,), np.dtype(np.int64)
    elif lay.row_shape is None:
        widths = {p.shape[1] for p in good}
        if len(widths) > 1:
            raise ArrayError(f"cells disagree on attribute width: {sorted(widths)}")
        tail, dtype = (widths.pop() if widths else 1,), lay.dtype
    else:
        tail, dtype = lay.row_shape, lay.dtype
    counts = np.array(
        [0 if p is None else p.shape[0] for p in parts], dtype=np.int64,
    )
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    data = (
        np.concatenate([p.reshape(-1, *tail) for p in good], axis=0)
        if good else np.empty((0, *tail), dtype=dtype)
    )
    return CellColumn(
        data=_xp.to_device(data, device),
        offsets=_xp.to_device(offsets, device),
        _host_offsets=offsets,
    )


def read_neighbourhood(
    level_group: Group,
    chunk_coords: Sequence[int],
    arrays: Sequence[str] = (VERTICES,),
    *,
    halo: int = 1,
    present_only: bool = True,
    present_array: str = VERTICES,
    **read_cells_kw: Any,
) -> CellBatch:
    """A cell and its neighbours within ``halo``, in one :func:`read_cells`.

    The centre comes first, then the neighbours in sorted order. With
    ``present_only`` (the default), neighbours ``present_array`` holds
    no data for are left out rather than read as empty. Filtering rows to
    an overlap band is the caller's business.
    """
    from zarr_vectors.spatial.chunking import neighbouring_chunk_keys

    centre = tuple(int(c) for c in chunk_coords)
    occupied = level_group.chunk_present_set(present_array) if present_only else None
    around = neighbouring_chunk_keys(centre, halo=halo, occupied_keys=occupied)
    cells = np.asarray([centre, *around], dtype=np.int64).reshape(-1, len(centre))
    return read_cells(level_group, cells, arrays, **read_cells_kw)
