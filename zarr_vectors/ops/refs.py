"""Reference dataclasses for the edit API.

A "ref" is a lightweight handle that uniquely identifies one element
(vertex / link / fragment / object / attribute) inside a ZV store so
the edit engine knows exactly which bytes to mutate.

Refs are *physical* by default — ``VertexRef(level, chunk, fragment,
local)`` is a direct address.  Each ref class also exposes alternate
constructors (``VertexRef.from_object``, ``VertexRef.from_position``)
that resolve from higher-level addressing schemes by reading the store.

Note on the name ``FragmentRef``: this is a dataclass in
:mod:`zarr_vectors.ops.refs` that adds a ``level`` field.  It is
distinct from the older :data:`zarr_vectors.typing.FragmentRef` tuple
type alias ``(chunk_coords, fragment_index)``; the two are namespaced
separately and serve different layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

from zarr_vectors.exceptions import EditError
from zarr_vectors.typing import ChunkCoords

if TYPE_CHECKING:
    from zarr_vectors.core.group import Group


@dataclass(frozen=True)
class VertexRef:
    """Physical address of one vertex row inside the store.

    ``(level, chunk, fragment, local)``:

    - ``level``: resolution-level index (``0`` for full resolution).
    - ``chunk``: spatial chunk coordinates, e.g. ``(0, 1, 2)``.
    - ``fragment``: fragment index *within* the chunk (0-based).
    - ``local``: row index *within* the fragment (0-based).
    """

    level: int
    chunk: ChunkCoords
    fragment: int
    local: int

    def __post_init__(self) -> None:
        if self.level < 0:
            raise EditError(f"VertexRef.level must be >= 0, got {self.level}")
        if self.fragment < 0:
            raise EditError(
                f"VertexRef.fragment must be >= 0, got {self.fragment}"
            )
        if self.local < 0:
            raise EditError(f"VertexRef.local must be >= 0, got {self.local}")

    @classmethod
    def from_object(
        cls,
        root: Group,
        *,
        level: int,
        object_id: int,
        vertex_index: int,
    ) -> VertexRef:
        """Resolve a vertex by ``(object_id, vertex_index_within_object)``.

        Reads the object's manifest from ``object_index/manifests`` and
        walks fragments in manifest order until ``vertex_index`` is
        located.  Raises :class:`EditError` if the object has fewer
        vertices than requested.
        """
        from zarr_vectors.core.arrays import (
            read_chunk_vertices,
            read_object_manifest,
        )
        from zarr_vectors.core.metadata import RootMetadata
        from zarr_vectors.core.store import get_resolution_level

        if vertex_index < 0:
            raise EditError(
                f"vertex_index must be >= 0, got {vertex_index}"
            )
        level_group = get_resolution_level(root, level)
        manifest = read_object_manifest(level_group, object_id)

        meta = RootMetadata.from_dict(root.attrs.to_dict())
        ndim = meta.sid_ndim
        vmeta = level_group.read_array_meta("vertices")
        vdtype = np.dtype(vmeta.get("dtype", "float32"))

        cursor = 0
        for chunk_coords, frag_idx in manifest:
            groups = read_chunk_vertices(
                level_group, chunk_coords, dtype=vdtype, ndim=ndim,
            )
            if frag_idx >= len(groups):
                raise EditError(
                    f"Object {object_id} manifest references fragment "
                    f"{frag_idx} in chunk {chunk_coords}, but the chunk has "
                    f"only {len(groups)} fragments."
                )
            n = int(groups[frag_idx].shape[0])
            if vertex_index < cursor + n:
                local = vertex_index - cursor
                return cls(
                    level=level,
                    chunk=tuple(int(c) for c in chunk_coords),
                    fragment=int(frag_idx),
                    local=int(local),
                )
            cursor += n
        raise EditError(
            f"Object {object_id} has {cursor} vertices; vertex_index="
            f"{vertex_index} is out of range."
        )

    @classmethod
    def from_position(
        cls,
        root: Group,
        *,
        level: int,
        pos: npt.ArrayLike,
        tol: float = 0.0,
    ) -> VertexRef:
        """Resolve the vertex nearest to ``pos`` within the chunk that
        owns ``pos``.

        Looks up the chunk via ``floor(pos / chunk_shape)``, decodes
        every fragment in that chunk, and returns the ref of the closest
        vertex (Euclidean L2).  ``tol > 0`` enforces a maximum distance;
        raises :class:`EditError` if no candidate is within ``tol``.
        """
        from zarr_vectors.core.arrays import read_chunk_vertices
        from zarr_vectors.core.metadata import RootMetadata
        from zarr_vectors.core.store import get_resolution_level
        from zarr_vectors.spatial.chunking import compute_chunk_coords

        meta = RootMetadata.from_dict(root.attrs.to_dict())
        chunk_shape = tuple(meta.chunk_shape)
        ndim = meta.sid_ndim

        pos_arr = np.asarray(pos, dtype=np.float64).reshape(-1)
        if pos_arr.shape[0] != ndim:
            raise EditError(
                f"pos has arity {pos_arr.shape[0]} but store ndim={ndim}"
            )

        chunk_coords = compute_chunk_coords(pos_arr, chunk_shape)
        level_group = get_resolution_level(root, level)
        vmeta = level_group.read_array_meta("vertices")
        vdtype = np.dtype(vmeta.get("dtype", "float32"))

        try:
            groups = read_chunk_vertices(
                level_group, chunk_coords, dtype=vdtype, ndim=ndim,
            )
        except Exception as e:
            raise EditError(
                f"No vertex data in chunk {chunk_coords}: {e}"
            ) from None

        best: tuple[float, int, int] | None = None  # (d2, frag, local)
        for f_idx, group in enumerate(groups):
            if group.shape[0] == 0:
                continue
            diff = group.astype(np.float64) - pos_arr
            d2 = np.einsum("ij,ij->i", diff, diff)
            local = int(np.argmin(d2))
            d = float(d2[local])
            if best is None or d < best[0]:
                best = (d, f_idx, local)

        if best is None:
            raise EditError(
                f"No vertices in chunk {chunk_coords}; cannot resolve "
                f"VertexRef.from_position."
            )
        if tol > 0.0 and best[0] > tol * tol:
            raise EditError(
                f"Nearest vertex is {best[0] ** 0.5:.6g} away, > tol={tol}"
            )
        return cls(
            level=level,
            chunk=tuple(int(c) for c in chunk_coords),
            fragment=best[1],
            local=best[2],
        )


@dataclass(frozen=True)
class LinkRef:
    """Physical address of one link row.

    ``(level, chunk, fragment, row, delta)``:

    - ``delta == 0`` → intra-level link inside ``links/0/<chunk>``.
    - ``delta != 0`` → cross-level link inside ``links/<delta>/<chunk>``.

    ``row`` is the row index inside that link fragment group.
    """

    level: int
    chunk: ChunkCoords
    fragment: int
    row: int
    delta: int = 0


@dataclass(frozen=True)
class CrossChunkLinkRef:
    """Address of one row in the global
    ``cross_chunk_links/<delta>/data`` array.
    """

    level: int
    row: int
    delta: int = 0


@dataclass(frozen=True)
class FragmentRef:
    """Address of one fragment (group of vertices) inside a chunk.

    Distinct from :data:`zarr_vectors.typing.FragmentRef` (tuple alias):
    this dataclass adds the ``level`` field needed by the edit engine.
    """

    level: int
    chunk: ChunkCoords
    fragment: int


@dataclass(frozen=True)
class ObjectRef:
    """Address of one object by ``(level, object_id)``."""

    level: int
    object_id: int


# Attribute scope literals: identify which of the three ragged
# attribute kinds the edit targets.
AttrScope = Literal["vertex", "object", "link"]


@dataclass(frozen=True)
class AttributeRef:
    """Address of one attribute value.

    ``scope`` is one of:

    - ``"vertex"``: per-vertex attribute.  ``target`` is a
      :class:`VertexRef`.  Indexed inside
      ``attributes/<name>/<chunk>`` at the row aligned with the vertex.
    - ``"object"``: per-object attribute.  ``target`` is an
      :class:`ObjectRef`.  Indexed inside
      ``object_attributes/<name>/data`` at row ``object_id``.
    - ``"link"``: per-link attribute.  ``target`` is a :class:`LinkRef`.
      Indexed inside ``link_attributes/<name>/<delta>/<chunk>``.
    """

    scope: AttrScope
    name: str
    target: VertexRef | ObjectRef | LinkRef
