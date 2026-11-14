"""Opt-in read-path experiments, off unless named in the environment.

``ZARR_VECTORS_EXPERIMENTS`` is a comma-separated list of names.  Each
experiment changes only how a read fetches, never what it returns, so a
benchmark can run the same queries with and without it.

====================  =======================================================
``read_cache``        one :meth:`Group.cached_nodes` block per query: each
                      node and array spec is resolved once per read
``octree_read_once``  a level with octree chunks is read by its reader
                      alone, without the engine's prefetch of uniform
                      cells and its replays: one pass, the leaf table
                      read once; a graph's link segments are resolved in
                      one gather
====================  =======================================================
"""

from __future__ import annotations

import os


def enabled(name: str) -> bool:
    """Whether experiment ``name`` is switched on."""
    raw = os.environ.get("ZARR_VECTORS_EXPERIMENTS", "")
    return name in {n.strip() for n in raw.split(",") if n.strip()}
