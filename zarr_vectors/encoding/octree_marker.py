"""The ``zarr_vectors_octree_chunks`` codec, carried by an octree level's per-chunk arrays.

An identity bytes-to-bytes codec: a zarr reader that does not know octree
chunks fails to open the arrays (``UnknownCodecError``) instead of reading
leaves as uniform cells.  Importing :mod:`zarr_vectors` registers it.  See
``docs/spec/chunking/octree_chunks.md`` (Compatibility).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from zarr.abc.codec import BytesBytesCodec
from zarr.core.common import parse_named_configuration
from zarr.registry import register_codec
from zarr_vectors.constants import OCTREE_CODEC_NAMES

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer

#: The codec's name in ``zarr.json``.
OCTREE_MARKER_CODEC = "zarr_vectors_octree_chunks"


@dataclass(frozen=True)
class OctreeChunksMarker(BytesBytesCodec):
    """Identity bytes-to-bytes codec marking an octree level's arrays."""

    is_fixed_size = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OctreeChunksMarker:
        parse_named_configuration(data, OCTREE_MARKER_CODEC, require_configuration=False)
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {"name": OCTREE_MARKER_CODEC}

    def _decode_sync(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return chunk_bytes

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return chunk_bytes

    def _encode_sync(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer | None:
        return chunk_bytes

    async def _encode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer | None:
        return chunk_bytes

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length


def is_marker(codec: Any) -> bool:
    """Whether ``codec`` (a codec object, or its ``zarr.json`` dict) is the marker."""
    if isinstance(codec, dict):
        return codec.get("name") == OCTREE_MARKER_CODEC
    return isinstance(codec, OctreeChunksMarker)


def cell_codecs(array: Any) -> list[Any]:
    """The codec objects ``array``'s cells pass through (inside its sharding codec, if any)."""
    codecs = list(array.metadata.codecs)
    if codecs and codecs[0].to_dict().get("name") in ("sharding_indexed", *OCTREE_CODEC_NAMES):
        codecs = list(getattr(codecs[0], "codecs", ()))
    return codecs


def has_marker(array: Any) -> bool:
    """Whether ``array``'s cell codecs hold the marker."""
    return any(is_marker(c) for c in cell_codecs(array))


register_codec(OCTREE_MARKER_CODEC, OctreeChunksMarker)

__all__ = ["OCTREE_MARKER_CODEC", "OctreeChunksMarker", "cell_codecs", "has_marker", "is_marker"]
