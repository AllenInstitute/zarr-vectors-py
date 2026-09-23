"""Frozen copies of write paths the array-form writers replaced.

Each is the definition a rewrite is compared against, byte for byte. They
are copied here, not imported, so the comparison target cannot drift with
the code it checks.
"""

from __future__ import annotations

from zarr_vectors.constants import LINK_FRAGMENTS, VERTEX_FRAGMENTS
from zarr_vectors.core.arrays import _chunk_key
from zarr_vectors.encoding.fragments import decode_fragments, encode_fragments
from zarr_vectors.exceptions import StoreError


def write_chunk_fragments_append_by_list(level_group, chunk_coords, new_list, *, target="vertex"):
    """``write_chunk_fragments(mode="append")`` before the CSR rewrite:
    decode every existing fragment to a Python object and re-encode all."""
    constant = VERTEX_FRAGMENTS if target == "vertex" else LINK_FRAGMENTS
    key = _chunk_key(chunk_coords)
    if not new_list:
        return []
    try:
        raw = level_group.read_bytes(constant, key)
    except StoreError:
        raw = b""
    if raw:
        fi = decode_fragments(raw)
        existing = [
            fi.range(i) if fi.is_range(i) else fi.indices(i)
            for i in range(fi.num_fragments)
        ]
    else:
        existing = []
    level_group.write_bytes(constant, key, encode_fragments(existing + list(new_list)))
    return list(range(len(existing), len(existing) + len(new_list)))
