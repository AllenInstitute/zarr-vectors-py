"""Opt-in read-path experiments, off unless named in the environment.

``ZARR_VECTORS_EXPERIMENTS`` is a comma-separated list of names.  Each
experiment changes only how a read fetches, never what it returns, so a
benchmark can run the same queries with and without it.

==================  =========================================================
``level_handle``    a query resolves its level group once, and a level
                    lookup is one request rather than two
``prime_nodes``     a query resolves its level and every array it can
                    touch in one gather (:mod:`zarr_vectors._query_nodes`)
``gather_indexes``  a store-backed read fetches every array's shard
                    indexes in one round, not one array after another
``batch_cells``     the optimised sharded reader parses keys, locates cells
                    and plans its reads as arrays, and decompresses each
                    cell straight out of its read, when a read averages 32
                    or more cells an array; the same reads either way
==================  =========================================================
"""

from __future__ import annotations

import os


def enabled(name: str) -> bool:
    """Whether experiment ``name`` is switched on."""
    raw = os.environ.get("ZARR_VECTORS_EXPERIMENTS", "")
    return name in {n.strip() for n in raw.split(",") if n.strip()}
