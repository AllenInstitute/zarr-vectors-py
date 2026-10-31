"""Ownership-based raw mesh reads for out-of-core renderers.

Face ownership follows the stored canonical source cell, not the centroid.
Selecting disjoint owners partitions the faces without losing boundary faces.
The existing data API's bbox/vertex filtering remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from zarr_vectors.core.arrays import (
    cell_endpoint_chunks,
    chunk_vertex_count,
    link_endpoints_to_rows,
    list_link_offsets,
    read_link_arrays,
)
from zarr_vectors.core.metadata import get_level_chunk_shape
from zarr_vectors.core.paths import links_group_path, parse_offsets
from zarr_vectors.core.store import (
    get_resolution_level,
    open_store,
    read_level_metadata,
    read_root_metadata,
)
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.typing import ChunkCoords


@dataclass(frozen=True)
class MeshChunkInfo:
    """An owner cell and conservative bounds of all its face endpoints."""

    key: ChunkCoords
    bounds: tuple[tuple[float, ...], tuple[float, ...]]


class MeshReader:
    """Read a raw mesh by ownership cells, fetching referenced vertices.

    Construct one reader per stored surface level. Metadata discovery lists
    the connectivity cells once; reads fetch only selected connectivity and
    referenced vertex chunks. Missing dependencies and invalid indices raise
    rather than returning a surface with silently missing triangles.

    Draco is explicitly unsupported: its intra-cell faces require a separate
    decode path and boundary indices need validation against decoder ordering.
    A reader is not thread safe; serialize calls or construct independent ones.
    """

    def __init__(self, source: Any, *, level: int = 0) -> None:
        self.root = open_store(source)
        self.group = get_resolution_level(self.root, level)
        root_meta = read_root_metadata(self.root)
        if root_meta.sid_ndim != 3:
            raise NotImplementedError("MeshReader currently supports spatial 3-D meshes")
        vertex_meta = self.group.read_array_meta("vertices")
        if vertex_meta.get("encoding", "raw") != "raw":
            raise NotImplementedError("MeshReader currently supports raw meshes, not Draco")
        self.dtype = np.dtype(vertex_meta.get("dtype", "float32"))
        if not self.group.array_exists("links/0"):
            raise ValueError(f"level {level} has no renderable surface connectivity")
        family = self.group.read_array_meta("links/0") or {}
        self.width = int(family.get("link_width", 0))
        if self.width < 3:
            raise ValueError(f"level {level} has no renderable surface connectivity")
        if family.get("store", "canonical") != "canonical":
            raise NotImplementedError("MeshReader requires canonical face ownership")
        shape = np.asarray(
            get_level_chunk_shape(
                root_meta,
                read_level_metadata(self.root, level),
            ),
            dtype=float,
        )
        self._cells: dict[str, set[str]] = {}
        endpoints: dict[ChunkCoords, set[ChunkCoords]] = {}
        for segment in list_link_offsets(self.group, 0):
            path = f"{links_group_path(0)}/{segment}"
            offsets = parse_offsets(segment, sid_ndim=3, link_width=self.width)
            keys = set(self.group.list_chunks(path))
            self._cells[path] = keys
            for key in keys:
                owner = tuple(int(c) for c in key.split("."))
                endpoints.setdefault(owner, set()).update(
                    cell_endpoint_chunks(owner, offsets, (1, 1, 1), (1, 1, 1))
                )
        self.chunks = tuple(
            MeshChunkInfo(
                key,
                (
                    tuple(np.min(np.asarray(sorted(cells)) * shape, axis=0)),
                    tuple(np.max((np.asarray(sorted(cells)) + 1) * shape, axis=0)),
                ),
            )
            for key, cells in sorted(endpoints.items())
        )
        self._keys = {chunk.key for chunk in self.chunks}

    def read(self, chunks: list[ChunkCoords], *, max_bytes: int | None = None) -> dict[str, Any]:
        """Return complete faces owned by ``chunks`` and their vertices.

        The result uses the same four keys as ``read_mesh``. Vertex indices
        are local to this result. Vertices shared by independently read owners
        may be duplicated; each face belongs to exactly one owner.

        ``max_bytes`` rejects reads whose decoded dependency vertices plus
        output faces exceed the limit, before fetching vertex payloads. Owned
        connectivity and fragment indices are read to discover that size;
        their I/O/decode scratch is additional to the limit.
        """
        if max_bytes is not None and max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        owners = sorted(set(tuple(c) for c in chunks))
        if not set(owners) <= self._keys:
            raise KeyError("unknown mesh ownership cell")
        owner_keys = {".".join(str(c) for c in owner) for owner in owners}
        plan = [(path, sorted(keys & owner_keys)) for path, keys in self._cells.items()]
        with self.root.cached_nodes(), self.group.batched_reads(plan):
            face_chunks, face_vi = read_link_arrays(self.group, source_chunks=owners)
        if face_vi.size == 0:
            return {
                "vertices": np.empty((0, 3), dtype=self.dtype),
                "faces": np.empty((0, self.width), dtype=np.int64),
                "vertex_count": 0,
                "face_count": 0,
            }
        dependencies = sorted(set(map(tuple, face_chunks.reshape(-1, 3))))
        vertex_keys = [".".join(str(c) for c in cc) for cc in dependencies]
        if max_bytes is not None:
            with self.group.batched_reads([("vertex_fragments", vertex_keys)]):
                counts = [chunk_vertex_count(self.group, cc, default=-1) for cc in dependencies]
            if any(count < 0 for count in counts):
                raise ArrayError("mesh dependency has no vertex fragment index")
            required = sum(counts) * 3 * self.dtype.itemsize + face_vi.nbytes
            if required > max_bytes:
                raise MemoryError(f"mesh dependencies need {required} bytes; limit is {max_bytes}")
        positions = []
        offsets = {}
        sizes = {}
        count = 0
        with self.group.batched_reads([("vertices", vertex_keys)]):
            for cc in dependencies:
                blob = self.group.read_bytes("vertices", ".".join(str(c) for c in cc))
                if not blob:
                    raise ArrayError(f"mesh face references missing vertex chunk {cc}")
                # Indices address physical rows, not concatenated fragment views:
                # explicit/shared fragments may overlap or reorder those rows.
                block = np.frombuffer(blob, dtype=self.dtype).reshape(-1, 3)
                offsets[cc] = count
                sizes[cc] = len(block)
                count += len(block)
                positions.append(block)
        for cc in dependencies:
            indices = face_vi[np.all(face_chunks == cc, axis=-1)]
            if np.any(indices < 0) or np.any(indices >= sizes[cc]):
                raise ArrayError(f"mesh face has invalid vertex index in chunk {cc}")
        vertices = np.concatenate(positions, axis=0)
        faces = link_endpoints_to_rows(face_chunks, face_vi, offsets)
        # Return only referenced vertices, preserving each face's winding.
        used, inverse = np.unique(faces, return_inverse=True)
        vertices = vertices[used]
        faces = inverse.reshape(faces.shape).astype(np.int64)
        return {
            "vertices": vertices,
            "faces": faces,
            "vertex_count": len(vertices),
            "face_count": len(faces),
        }
