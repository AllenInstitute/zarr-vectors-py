"""Compression codec configuration for ZV arrays.

Provides sensible default codec pipelines for each array type in the
Zarr Vectors (ZV) format.  All returned configurations are dicts
compatible with Zarr v3's codec pipeline specification.
"""

from __future__ import annotations

from typing import Any

from zarr_vectors.constants import (
    CROSS_CHUNK_LINKS,
    DEFAULT_COMPRESSOR_OPTS,
    GROUP_ATTRIBUTES,
    GROUPS,
    LINK_ATTRIBUTES,
    LINK_FRAGMENTS,
    FRAGMENT_ATTRIBUTES,
    LINKS,
    OBJECT_ATTRIBUTES,
    OBJECT_INDEX,
    VERTEX_ATTRIBUTES,
    VERTEX_FRAGMENTS,
    VERTICES,
)


# ---------------------------------------------------------------------------
# Public codec-selection API
# ---------------------------------------------------------------------------

# Bytes serializer that every numeric inner array uses as its first codec
# (zarr-vectors stores chunks as 1D uint8 arrays).
_BYTES_SERIALIZER: dict[str, Any] = {"name": "bytes"}

# zarr 3.2.1's default compressor for numeric arrays, matching
# ``default_compressors_v3`` in zarr.core.array — a Zstd codec at level 0
# with checksum disabled.  Kept as a module-level constant so the value is
# trivial to inspect from tests and the benchmark notebook.
ZARR_V3_DEFAULT_ZSTD_CODEC: dict[str, Any] = {
    "name": "zstd",
    "configuration": {"level": 0, "checksum": False},
}

# Spec-aspirational default ("blosc" shorthand).  Matches the Blosc(Zstd,
# BitShuffle, l5) pipeline described in
# ``docs/spec/foundations/codec_pipeline.md``.
_BLOSC_BITSHUFFLE_L5_CODEC: dict[str, Any] = {
    "name": "blosc",
    "configuration": {
        "cname": "zstd",
        "clevel": 5,
        "shuffle": "bitshuffle",
        "typesize": 4,
        "blocksize": 0,
    },
}


def resolve_compressor(
    value: Any,
) -> list[dict[str, Any]]:
    """Resolve a user-supplied ``compressor=`` value to a full codecs list.

    The returned list is the canonical Zarr V3 ``codecs`` JSON shape used
    inside per-chunk ``zarr.json`` files — i.e. the BytesCodec serializer
    plus zero or more BytesBytes compressors.  Pass-through for already-
    formed lists; if the user omits the BytesCodec serializer it is
    prepended automatically.

    Args:
        value: One of:
            - ``None`` *(default)*, ``"none"``, or ``False`` — no
              compression (``bytes`` only).  Keeps the fast async PUT
              path active.
            - ``"zstd"`` — zarr v3's default compressor (``bytes`` +
              ``zstd``, level 0).  Forces the sync codec-encoding path.
            - ``"blosc"`` — Blosc(Zstd, BitShuffle, l5) shorthand
              matching ``codec_pipeline.md``.
            - ``list[dict]`` — explicit codec specs; the BytesCodec
              serializer is prepended unless already present.

    Returns:
        Codec list suitable to splice into an inner-array ``zarr.json``
        ``"codecs"`` field, or to strip-and-pass to
        ``zarr.Group.create_array(compressors=...)``.

    Raises:
        ValueError: For any other value shape.
    """
    if value is None or value is False or value == "none":
        return [dict(_BYTES_SERIALIZER)]
    if value == "zstd":
        return [dict(_BYTES_SERIALIZER), dict(ZARR_V3_DEFAULT_ZSTD_CODEC)]
    if value == "blosc":
        return [dict(_BYTES_SERIALIZER), dict(_BLOSC_BITSHUFFLE_L5_CODEC)]
    if isinstance(value, list):
        if not all(isinstance(c, dict) and "name" in c for c in value):
            raise ValueError(
                f"compressor list entries must be dicts with a 'name' key; got {value!r}"
            )
        if value and value[0].get("name") == "bytes":
            return [dict(c) for c in value]
        return [dict(_BYTES_SERIALIZER), *(dict(c) for c in value)]
    raise ValueError(
        f"compressor must be None, 'none'/False, 'zstd', 'blosc', or list[dict]; "
        f"got {value!r}"
    )


def codecs_for_create_array(codecs_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip the BytesCodec serializer from a full codecs list.

    ``zarr.Group.create_array`` takes the BytesBytes compressors separately
    from the serializer (it adds the BytesCodec itself).  Use this to
    convert a :func:`resolve_compressor` result into the
    ``compressors=...`` kwarg form.
    """
    return [
        dict(c) for c in codecs_list if c.get("name") != "bytes"
    ]


def get_default_compressor(array_type: str) -> dict[str, object]:
    """Return default compressor configuration for a given array type.

    Args:
        array_type: One of the canonical array name constants (e.g.
            ``"vertices"``, ``"links"``, ``"vertex_attributes"``).

    Returns:
        Dict with compressor settings suitable for ``numcodecs`` or
        Zarr v3 codec configuration.
    """
    # Links with sequential parents benefit from delta encoding
    if array_type == LINKS:
        return {
            "id": "blosc",
            "cname": "zstd",
            "clevel": 5,
            "shuffle": 2,  # SHUFFLE_BITSHUFFLE — good for correlated ints
        }

    # Fragment-index blobs (v0.6) and the object_index blob mix int64
    # range tables with uint32 CSR offsets — byte-shuffle decorrelates
    # the high zero bytes well across the heterogeneous payload.
    if array_type in (VERTEX_FRAGMENTS, LINK_FRAGMENTS, OBJECT_INDEX):
        return {
            "id": "blosc",
            "cname": "zstd",
            "clevel": 5,
            "shuffle": 1,  # SHUFFLE_BYTE
        }

    # Vertex positions and attributes — byte shuffle works well on floats
    if array_type in (VERTICES, VERTEX_ATTRIBUTES, FRAGMENT_ATTRIBUTES):
        return {
            "id": "blosc",
            "cname": "zstd",
            "clevel": 5,
            "shuffle": 1,
        }

    # Dense arrays (object attributes, group attributes)
    if array_type in (OBJECT_ATTRIBUTES, GROUP_ATTRIBUTES):
        return {
            "id": "blosc",
            "cname": "zstd",
            "clevel": 5,
            "shuffle": 1,
        }

    # Everything else — safe default
    return dict(DEFAULT_COMPRESSOR_OPTS)


def get_codec_pipeline(
    array_type: str,
    encoding: str = "raw",
    compression: str | None = None,
    compression_opts: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Build a Zarr v3 codec pipeline for an array.

    The pipeline is a list of codec specifications applied in order.
    For raw encoding, this is typically just a compressor.  For Draco
    encoding, the raw bytes are already compressed so we use no
    additional compression (or minimal).

    Args:
        array_type: Canonical array name constant.
        encoding: ``"raw"`` (default) or ``"draco"``.  When draco, the
            data is already compressed and minimal additional compression
            is applied.
        compression: Override compressor name (``"blosc"``, ``"zstd"``,
            ``"gzip"``, ``None`` for no compression).
        compression_opts: Override compressor options dict.

    Returns:
        List of codec config dicts for the Zarr v3 codec pipeline.
    """
    pipeline: list[dict[str, object]] = []

    if encoding == "draco":
        # Draco output is already compressed — store as raw bytes
        # Optionally wrap in a light pass-through codec
        if compression is not None and compression != "none":
            pipeline.append(
                _build_compressor(compression, compression_opts or {"clevel": 1})
            )
        return pipeline

    # Raw encoding — apply compressor
    if compression == "none" or compression is None and encoding == "raw":
        # Use defaults
        if compression is None:
            compressor = get_default_compressor(array_type)
            pipeline.append(compressor)
        # compression == "none" → no compression
        return pipeline

    if compression is not None:
        pipeline.append(
            _build_compressor(compression, compression_opts or {})
        )
        return pipeline

    return pipeline


def _build_compressor(
    name: str,
    opts: dict[str, object],
) -> dict[str, object]:
    """Build a compressor config dict.

    Args:
        name: Compressor name (``"blosc"``, ``"zstd"``, ``"gzip"``).
        opts: Additional options.

    Returns:
        Compressor configuration dict.
    """
    config: dict[str, object] = {"id": name}
    config.update(opts)
    return config
