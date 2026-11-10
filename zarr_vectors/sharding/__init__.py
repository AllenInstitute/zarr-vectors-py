"""Native-codec sharding for ZV stores.

Sharding packs many Zarr Vectors per-chunk byte blobs into a single storage
object via Zarr v3's built-in ``sharding_indexed`` codec.  The result
is a fully spec-compliant Zarr store readable by any standards-compliant
Zarr v3 implementation.

Public API:

    from zarr_vectors.sharding import shard_store, unshard_store, reshard

    shard_store("scan.zv", shard_shape=8)        # 8x8x8 = 512 chunks/shard
    info = get_shard_info("scan.zv")
    unshard_store("scan.zv")                     # back to flat
    set_octree("scan.zv")                        # optional: zarr-octree's codec (needs zarr-octree)
    set_octree("scan.zv", False)                 # and back to sharding_indexed

See :doc:`/spec/chunking/sharding` for the full design rationale.
"""

from zarr_vectors.sharding.io import (
    get_shard_info,
    is_sharded,
    reshard,
    set_octree,
    shard_store,
    unshard_store,
)
from zarr_vectors.sharding.layout import ShardLayout

__all__ = [
    "ShardLayout",
    "get_shard_info",
    "is_sharded",
    "reshard",
    "set_octree",
    "shard_store",
    "unshard_store",
]
