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
class CellBatch(Mapping[str, CellColumn]):
    """The result of :func:`read_cells`: one :class:`CellColumn` per array.

    ``chunk_coords`` is the requested cells in request order, always on
    the host; the columns are on ``device``.
    """

    chunk_coords: npt.NDArray[np.int64]
    columns: dict[str, CellColumn]
    errors: tuple[CellReadError, ...] = ()
    device: str = "cpu"

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
        return replace(self, columns=moved, device=device)


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

    Returns:
        A :class:`CellBatch`. A cell nobody wrote, or outside an array's
        grid, has no rows.
    """
    from zarr_vectors.core.arrays import _chunk_key, _infer_vert_ndim

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

    errors: list[CellReadError] = []
    raw: dict[tuple[str, str], bytes] = {}
    with _prefetch(level_group, [p for p in plan if p[1]]):
        for name, cell_keys in plan:
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
            vertex_rows[key] = len(raw.get((VERTICES, key), b"")) // (itemsize * ndim)

    columns: dict[str, CellColumn] = {}
    for name, lay in layouts.items():
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

    return CellBatch(
        chunk_coords=cc, columns=columns, errors=tuple(errors), device=device,
    )


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
