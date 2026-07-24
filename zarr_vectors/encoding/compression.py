"""Compression codec configuration for ZV arrays.

Provides sensible default codec pipelines for each array type in the
Zarr Vectors (ZV) format.  All returned configurations are dicts
compatible with Zarr v3's codec pipeline specification.
"""

from __future__ import annotations

from typing import Any


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

