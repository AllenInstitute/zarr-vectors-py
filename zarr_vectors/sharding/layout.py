"""``ShardLayout``: a sharded / unsharded selector.

Sharding goes through Zarr v3's built-in ``sharding_indexed`` codec,
which clusters spatially-adjacent inner chunks into the same outer chunk
via the C-order outer grid, so there is a single sharded mode.
``OCTREE`` / ``SNAKE`` / ``INDEX_TABLE`` all name it; ``FLAT`` is the
unsharded layout.
"""

from __future__ import annotations

from enum import StrEnum


class ShardLayout(StrEnum):
    """Shard layout selector.

    ``FLAT`` keeps one storage object per Zarr Vectors chunk (no sharding).
    All other values request native ``sharding_indexed`` sharding, and
    are equivalent to one another.
    """

    FLAT = "flat"
    OCTREE = "octree"
    SNAKE = "snake"
    INDEX_TABLE = "index_table"

    @property
    def is_sharded(self) -> bool:
        return self is not ShardLayout.FLAT
