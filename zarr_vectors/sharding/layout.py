"""Back-compat surface for the legacy ``ShardLayout`` / ``ShardCodec`` API.

Older versions exposed an enum of shard layouts (FLAT, OCTREE, SNAKE,
INDEX_TABLE) plus a ``ShardCodec`` that mapped chunk coordinates to
shard ids via Morton / Hilbert space-filling curves.  Sharding now
goes through Zarr v3's built-in ``sharding_indexed`` codec, which
clusters spatially-adjacent inner chunks into the same outer chunk
automatically via the C-order outer grid — no custom curve needed.

This module remains only to keep the import line in older callers
working.  Treat ``OCTREE`` / ``SNAKE`` / ``INDEX_TABLE`` as aliases
for the single native-sharded mode; ``FLAT`` is the unsharded layout.
"""

from __future__ import annotations

from enum import Enum


class ShardLayout(str, Enum):
    """Layout selector for back-compat with older shard APIs.

    ``FLAT`` keeps one storage object per ZVF chunk (no sharding).
    All other values request native ``sharding_indexed`` sharding;
    the curve distinction is no longer load-bearing.
    """

    FLAT = "flat"
    OCTREE = "octree"
    SNAKE = "snake"
    INDEX_TABLE = "index_table"

    @property
    def is_sharded(self) -> bool:
        return self is not ShardLayout.FLAT
