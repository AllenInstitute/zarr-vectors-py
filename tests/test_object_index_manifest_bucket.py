"""``object_index/manifests`` must be chunked at the bucket, whoever creates it.

The chunk shape of that array is decided once, by whichever write brings it
into existence, and is then immutable: every later append only ``resize()``s,
and zarr does not let a resize change ``chunk_shape``.

That made the layout a function of write *ordering*. In a per-spatial-chunk
parallel build the first flush is whichever worker won the store lock —
typically a sparse edge chunk emitting a few dozen objects. Clamping the chunk
to that first write (``min(BUCKET, n)``) therefore sized a whole-brain index
from a rounding error: two stores built by identical code over comparable data
came out at 26 and 1128 blobs per chunk, a 20x difference in file count with no
cause but timing. At 19.5M objects the 26-blob store would have occupied
~750,000 files in a single directory instead of ~1,200, and the run that hit
it died on ``OSError: [Errno 122] Disk quota exceeded`` against a 1,000,000
inode limit — with the volume 35% full by bytes.

So the property under test is not "the bucket is 16384". It is **the bucket
does not depend on who writes first**, which is the only version of the
statement that a parallel build can rely on. The file-count assertions are
here because ``chunks`` metadata being right is not the same as the inodes
being right, and inodes were what ran out.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ``building`` throughout, not ``core``: it is the supported surface, and a
# regression test that reaches past it would keep passing while the public API
# it is meant to protect broke.  The bucket constant has no ``building``
# spelling, so it is the one import from ``core``.
from zarr_vectors.building import (
    create_store,
    encode_object_manifest_blocks,
    get_resolution_level,
    read_all_object_manifests,
    write_object_manifests,
)
from zarr_vectors.constants import OBJECT_INDEX
from zarr_vectors.core.arrays import (
    OBJECT_INDEX_LAYOUT_V1,
    OBJECT_INDEX_MANIFEST_BUCKET,
)

SID_NDIM = 3


def _level(prefix: str):
    """A fresh store's level-0 group, with no object index yet."""
    path = Path(tempfile.mkdtemp(prefix=f"oib_{prefix}_")) / "store.zarrvectors"
    root = create_store(
        path,
        bounds=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
        chunk_shape=(50.0, 50.0, 50.0),
        geometry_types=["polyline"],
    )
    return get_resolution_level(root, 0), path


def _blobs(n: int, *, base: int = 0) -> list[bytes]:
    """``n`` distinct one-fragment manifests, so a mix-up is detectable."""
    return [
        encode_object_manifest_blocks([((1, 2, 3), base + i)], sid_ndim=SID_NDIM)
        for i in range(n)
    ]


def _manifest_array(level):
    return level.zarr_group[OBJECT_INDEX]["manifests"]


def _commit(level, n: int) -> None:
    """Stamp the ``object_index`` group meta a reader needs.

    ``write_object_manifests`` writes the array only -- the count and
    ``sid_ndim`` are committed separately, which is exactly how a per-chunk
    parallel writer does it (the commit is deferred to the end of a flush so a
    torn write leaves rows past the count rather than live ids with no
    payload).  These tests drive the same split, so they have to close it too.
    """
    level.write_array_meta(OBJECT_INDEX, {
        "zv_array": "object_index",
        "num_objects": n,
        "num_present": n,
        "sid_ndim": SID_NDIM,
        "layout": OBJECT_INDEX_LAYOUT_V1,
    })


def _chunk_files(store_path: Path) -> int:
    """Actual data objects on disk for the manifests array."""
    cdir = store_path / "0" / OBJECT_INDEX / "manifests" / "c"
    return sum(1 for p in cdir.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------
# The regression
# ---------------------------------------------------------------------------

def test_bucket_is_independent_of_first_write_size() -> None:
    """A sparse first append must not size the chunk for the whole store.

    This reproduces the incident exactly: one small append at ``at=0``
    (the edge chunk that won the lock), then many larger ones. Before the
    fix the array reports ``chunks[0] == 26`` forever.
    """
    level, _ = _level("first_write")

    write_object_manifests(level, _blobs(26), mode="append", at=0)
    cursor = 26
    for i in range(1, 40):
        batch = _blobs(500, base=cursor)
        write_object_manifests(level, batch, mode="append", at=cursor)
        cursor += 500

    arr = _manifest_array(level)
    assert arr.shape == (cursor,)
    assert arr.chunks[0] == OBJECT_INDEX_MANIFEST_BUCKET, (
        f"chunk sized from the first write ({arr.chunks[0]}) rather than the "
        f"bucket — a {cursor}-object index would materialise "
        f"{math.ceil(cursor / arr.chunks[0])} files instead of "
        f"{math.ceil(cursor / OBJECT_INDEX_MANIFEST_BUCKET)}"
    )


@pytest.mark.parametrize("first", [1, 26, 1128, 4096, 16_384, 40_000])
def test_bucket_is_stable_across_write_orderings(first: int) -> None:
    """Whatever the first writer emits, the layout that results is the same.

    Parametrised over the two real observed values (26, 1128) plus the
    boundaries, because "it works for the size I happened to test" is the
    bug, not the fix.
    """
    level, _ = _level(f"order_{first}")

    write_object_manifests(level, _blobs(first), mode="append", at=0)
    write_object_manifests(
        level, _blobs(5_000, base=first), mode="append", at=first,
    )

    arr = _manifest_array(level)
    assert arr.chunks[0] == OBJECT_INDEX_MANIFEST_BUCKET
    assert arr.shape == (first + 5_000,)


# ---------------------------------------------------------------------------
# Inodes, not metadata
# ---------------------------------------------------------------------------

def test_file_count_is_ceil_objects_over_bucket() -> None:
    """Count the objects on disk, not the number zarr claims it would use.

    ``chunks`` being correct and the directory being correct are separate
    facts, and it was the second one that exhausted the quota.
    """
    level, path = _level("filecount")

    n = 3 * OBJECT_INDEX_MANIFEST_BUCKET + 17
    write_object_manifests(level, _blobs(n), mode="append", at=0)

    expected = math.ceil(n / OBJECT_INDEX_MANIFEST_BUCKET)
    assert _chunk_files(path) == expected == 4


def test_a_tiny_store_still_costs_one_file() -> None:
    """The unconditional bucket must not over-allocate for small stores.

    zarr materialises only chunks that intersect the shape, so a 26-blob
    array in a 16384-slot chunk is one file — the same as before the fix.
    This is what makes the change free rather than a trade.
    """
    level, path = _level("tiny")
    write_object_manifests(level, _blobs(26), mode="append", at=0)

    assert _manifest_array(level).chunks[0] == OBJECT_INDEX_MANIFEST_BUCKET
    assert _chunk_files(path) == 1


# ---------------------------------------------------------------------------
# The payload has to survive all of it
# ---------------------------------------------------------------------------

def test_blobs_round_trip_across_the_appends() -> None:
    """Re-chunking is only safe if the manifests still read back in order."""
    level, _ = _level("roundtrip")

    write_object_manifests(level, _blobs(26), mode="append", at=0)
    write_object_manifests(level, _blobs(2_000, base=26), mode="append", at=26)
    _commit(level, 2_026)

    got = read_all_object_manifests(level)
    assert len(got) == 2_026
    # Fragment index was seeded from the object's position, so this checks
    # ordering and content together rather than merely that N blobs exist.
    for oid, manifest in enumerate(got):
        assert manifest == [((1, 2, 3), oid)], f"wrong manifest at oid={oid}"
