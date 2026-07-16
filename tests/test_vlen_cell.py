"""The vlen-cell readers must all decode identically.

There used to be four hand-rolled copies of "read one vlen cell" (sync,
async, sync-fallback, manifests). They all had to agree on a zarr 3.x
subtlety — scalar-indexing a vlen array returns the array *header*, not
the payload — and a drift between copies would be silent corruption, not
a crash. They now share zarr_vectors.core._vlen; these tests guard against
a future re-divergence.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from zarr_vectors.constants import VERTICES
from zarr_vectors.core._vlen import cell_region, region_to_bytes
from zarr_vectors.core.arrays import list_chunk_keys
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.types.points import write_points


def test_region_to_bytes_handles_fill_and_empty() -> None:
    # A fill value (None) and an empty region both decode to b"".
    assert region_to_bytes(np.array([None], dtype=object)) == b""
    assert region_to_bytes(np.empty((0,), dtype=object)) == b""
    assert region_to_bytes(np.array([b"payload"], dtype=object)) == b"payload"


def test_cell_region_is_a_slice_not_a_scalar() -> None:
    # Every component must be a slice — a scalar index would trigger the
    # zarr 3.x header-not-payload hazard.
    region = cell_region((2, 0, 5))
    assert region == (slice(2, 3), slice(0, 1), slice(5, 6))
    assert all(isinstance(s, slice) for s in region)


def test_batched_and_unbatched_reads_are_byte_identical() -> None:
    """The sync reader and the batched (async) reader must agree.

    A multi-chunk store guarantees some empty cells, which is exactly
    where a header-vs-payload drift would show up.
    """
    path = os.path.join(tempfile.mkdtemp(), "p.zv")
    pos = np.random.default_rng(0).uniform(0, 100, (300, 3)).astype("f4")
    write_points(
        path, pos, chunk_shape=(30.0, 30.0, 30.0),
        bounds=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
    )
    lg = get_resolution_level(open_store(path), 0)
    keys = list_chunk_keys(lg)
    str_keys = [".".join(map(str, k)) for k in keys]

    unbatched = {k: lg.read_bytes(VERTICES, k) for k in str_keys}
    with lg.batched_reads([(VERTICES, str_keys)]):
        batched = {k: lg.read_bytes(VERTICES, k) for k in str_keys}

    assert unbatched == batched
    # And at least one non-trivial payload exists, so the test isn't
    # vacuously comparing all-empty.
    assert any(v for v in unbatched.values())
