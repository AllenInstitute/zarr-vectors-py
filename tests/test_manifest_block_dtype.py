"""An int32 ``manifest_blocks``: opt-in, kept for the index's life, range-checked on every write.

``create_store(manifest_block_dtype="int32")`` makes it the default for a
store's new dense indexes, and ``block_dtype=`` asks for it per write. The
properties:

- every writer path stores the same manifests in an int32 index as in an
  int64 one, and every reader returns the same int64 arrays from both;
- every write into an int32 index refuses a value that does not fit
  (``ArrayError``, and nothing of the write lands) -- zarr itself would
  silently wrap it, which is the hazard this exists to close;
- an index keeps the dtype it was created with: appends, reservations and
  placements refuse another, a full rewrite may change it, and an int64
  index stays int64 under an int32 store default.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import zarr_vectors as zv
import zarr_vectors.building as zb
from zarr_vectors.constants import OBJECT_INDEX
from zarr_vectors.core import dense_manifests as dense
from zarr_vectors.core.arrays import (
    object_present_mask,
    read_object_manifest,
    read_object_manifest_rows,
    read_object_manifests,
)
from zarr_vectors.core.metadata import RootMetadata
from zarr_vectors.core.store import read_root_metadata
from zarr_vectors.core.streaming import ObjectIndexAppender
from zarr_vectors.exceptions import ArrayError, MetadataError

pytestmark = pytest.mark.vlen_only  # these pick their layouts themselves

I32_MAX = 2**31 - 1


def _root(tmp_path, name, *, store_dtype=None, layout="dense"):
    return zb.create_store(
        tmp_path / f"{name}.zarrvectors", bounds=([0.0] * 3, [100.0] * 3),
        chunk_shape=(50.0,) * 3, geometry_types=["polyline"], manifest_layout=layout,
        manifest_block_dtype=store_dtype,
    )


def _level(tmp_path, name, **kw):
    return zb.get_resolution_level(_root(tmp_path, name, **kw), 0)


def _blocks(lg):
    return lg.zarr_group[dense.BLOCKS_PATH]


def _csr(rng, n, *, max_k=5, top=I32_MAX):
    """Manifests whose values reach the int32 edges: coordinates down to
    -2**31, fragment indices up to ``top``."""
    counts = rng.integers(0, max_k + 1, n)
    off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    k = int(off[-1])
    cc = rng.integers(-(2**31), 2**31, (k, 3)).astype(np.int64)
    fi = rng.integers(0, top, k, endpoint=True).astype(np.int64)
    if k:
        cc[0] = (-(2**31), I32_MAX, 0)
        fi[0] = top
    return off, cc, fi


def _too_big(kind="fragment"):
    """One object whose single block overflows int32."""
    if kind == "fragment":
        return np.array([0, 1]), np.zeros((1, 3), np.int64), np.array([2**40 + 5])
    return np.array([0, 1]), np.array([[-(2**31) - 1, 0, 0]]), np.array([3])


def _write(lg, off, cc, fi, **kw):
    return zb.write_object_manifests(
        lg, chunk_coords=cc, fragment_idx=fi, manifest_offsets=off, **kw,
    )


def _as_dict(off, cc, fi, ids=None):
    m = dense.manifests_from_csr(off, cc, fi)
    ids = range(len(m)) if ids is None else ids
    return {int(o): v for o, v in zip(ids, m)}


def _reads(lg):
    """What every manifest reader returns, with dtypes."""
    csr = zb.read_all_object_manifests_csr(lg)
    ids, rows = read_object_manifest_rows(lg)
    some = [int(i) for i in ids[:: max(1, len(ids) // 7)]]
    by_id = zb.read_object_manifests_csr(lg, some)
    offsets, coords, frags = dense.read_csr(lg)
    for arr in (csr.offsets, csr.chunk_coords, csr.fragment_idx, by_id.chunk_coords,
                by_id.fragment_idx, coords, frags):
        assert arr.dtype == np.int64, arr.dtype
    return {
        "csr": (csr.offsets.tolist(), csr.chunk_coords.tolist(), csr.fragment_idx.tolist(),
                None if csr.object_ids is None else csr.object_ids.tolist()),
        "all": zb.read_all_object_manifests(lg),
        "rows": (ids.tolist(), rows),
        "by_id": (by_id.object_ids.tolist(), by_id.offsets.tolist(),
                  by_id.chunk_coords.tolist(), by_id.fragment_idx.tolist()),
        "dict": read_object_manifests(lg),
        "one": [read_object_manifest(lg, o) for o in some],
        "present": object_present_mask(lg).tolist(),
        "dense": (offsets.tolist(), coords.tolist(), frags.tolist()),
    }


def _pair(tmp_path, name):
    """An int32 level and an int64 twin, under a store default each."""
    return (_level(tmp_path, f"{name}32", store_dtype="int32"),
            _level(tmp_path, f"{name}64"))


# --------------------------------------------------------------------
# The store default and the per-call choice


def test_the_store_records_the_default(tmp_path):
    root = _root(tmp_path, "a", store_dtype="int32")
    assert root.attrs.to_dict()["zarr_vectors"]["manifest_block_dtype"] == "int32"
    assert read_root_metadata(root).manifest_block_dtype == "int32"
    for absent in (None, "int64"):
        r = _root(tmp_path, f"b{absent}", store_dtype=absent)
        assert "manifest_block_dtype" not in r.attrs.to_dict()["zarr_vectors"]
    with pytest.raises(MetadataError, match="manifest_block_dtype"):
        _root(tmp_path, "c", store_dtype="int16")
    meta = RootMetadata.from_dict(root.attrs.to_dict())
    assert meta.to_dict()["zarr_vectors"]["manifest_block_dtype"] == "int32"


def test_a_bad_block_dtype_is_refused(tmp_path):
    lg = _level(tmp_path, "a")
    off, cc, fi = _csr(np.random.default_rng(0), 4)
    for bad in ("int16", "uint32", "float64", "nope"):
        with pytest.raises(ArrayError, match="block dtype"):
            _write(lg, off, cc, fi, mode="append", block_dtype=bad)
        with pytest.raises(ArrayError, match="block dtype"):
            zb.reserve_object_rows(lg, 4, block_dtype=bad)
    assert not lg.array_exists(dense.BLOCKS_PATH)


@pytest.mark.parametrize("how", ["store", "call"])
def test_the_index_is_created_int32_and_stamped(tmp_path, how):
    lg = _level(tmp_path, "a", store_dtype="int32" if how == "store" else None)
    off, cc, fi = _csr(np.random.default_rng(1), 20)
    _write(lg, off, cc, fi, mode="append", at=0,
           block_dtype="int32" if how == "call" else None)
    assert _blocks(lg).dtype == np.int32
    assert lg.zarr_group[dense.SPANS_PATH].dtype == np.int64
    assert lg.zarr_group[f"{OBJECT_INDEX}/object_ids"].dtype == np.int64
    assert lg.read_array_meta(OBJECT_INDEX)[dense.BLOCK_DTYPE_ATTR] == "int32"
    zb.commit_object_index(lg, 20, sid_ndim=3)
    assert lg.read_array_meta(OBJECT_INDEX)[dense.BLOCK_DTYPE_ATTR] == "int32"


def test_an_int64_index_carries_no_stamp(tmp_path):
    lg = _level(tmp_path, "a")
    off, cc, fi = _csr(np.random.default_rng(2), 20)
    _write(lg, off, cc, fi, mode="append", at=0)
    zb.commit_object_index(lg, 20, sid_ndim=3)
    assert _blocks(lg).dtype == np.int64
    assert dense.BLOCK_DTYPE_ATTR not in lg.read_array_meta(OBJECT_INDEX)


def test_a_call_overrides_the_store_default(tmp_path):
    lg = _level(tmp_path, "a", store_dtype="int32")
    off, cc, fi = _csr(np.random.default_rng(3), 5)
    _write(lg, off, cc, fi, mode="append", block_dtype="int64")
    assert _blocks(lg).dtype == np.int64


# --------------------------------------------------------------------
# Round trips: every writer, every reader


def _by_append(lg, rng):
    for at in (0, 300, 650):
        off, cc, fi = _csr(rng, 300 if at < 650 else 50)
        _write(lg, off, cc, fi, mode="append", at=at)
    zb.commit_object_index(lg, 700, sid_ndim=3)


def _by_blobs(lg, rng):
    off, cc, fi = _csr(rng, 200)
    blobs = zb.encode_object_manifests_csr(cc, fi, off, sid_ndim=3)
    zb.write_object_manifests(lg, blobs, mode="append", at=0, layout="dense")
    zb.write_object_manifests(lg, blobs[:50], mode="append", at=200)
    zb.commit_object_index(lg, 250, sid_ndim=3)


def _by_index(lg, rng):
    off, cc, fi = _csr(rng, 300)
    ids = np.sort(rng.choice(10**12, 300, replace=False))
    zb.write_object_index(lg, _as_dict(off, cc, fi, ids.tolist()), 3)


def _by_replace(lg, rng):
    off, cc, fi = _csr(rng, 100)
    _write(lg, off, cc, fi, mode="replace")
    zb.commit_object_index(lg, 100, sid_ndim=3)


def _by_place(lg, rng):
    parts = [_csr(rng, n) for n in (65_536, 1_000)]
    zb.reserve_object_rows(lg, 66_536, sid_ndim=3, n_blocks=65_536 * 2)
    starts = zb.aligned_regions([int(p[0][-1]) for p in parts],
                                zv.concurrency_contract(lg)["block_alignment"])

    def place(i):
        off, cc, fi = parts[i]
        _write(lg, off, cc, fi, mode="place", at=65_536 * i, block_at=int(starts[i]))

    zb.reserve_object_rows(lg, 66_536, n_blocks=int(starts[-1]))
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(place, range(2)))
    zb.commit_object_index(lg, 66_536, sid_ndim=3)


def _by_stream(lg, rng):
    off, cc, fi = _csr(rng, 3_000)
    with zb.object_manifest_writer(lg, at=0, commit=True) as w:
        for lo in range(0, 3_000, 400):
            hi = min(lo + 400, 3_000)
            w.write(chunk_coords=cc[off[lo]:off[hi]], fragment_idx=fi[off[lo]:off[hi]],
                    manifest_offsets=off[lo:hi + 1] - off[lo])


def _by_appender(lg, rng):
    off, cc, fi = _csr(rng, 500)
    with ObjectIndexAppender(lg, 0, 3, []) as app:
        app.append(dense.manifests_from_csr(off, cc, fi), [1] * 500)


def _by_patch(lg, rng):
    _by_append(lg, rng)
    off, cc, fi = _csr(rng, 30)
    zb.patch_object_manifests(lg, _as_dict(off, cc, fi, range(0, 900, 30)), 3)


def _by_torn_flush(lg, rng):
    a, b, c = _csr(rng, 400), _csr(rng, 400), _csr(rng, 100)
    _write(lg, *a, mode="append", at=0)
    _write(lg, *b, mode="append", at=400)
    _write(lg, *c, mode="append", at=250)      # rows 250.. were residue
    zb.commit_object_index(lg, 350, sid_ndim=3)


def _by_vlen_rewrite(lg, rng):
    """A vlen index rewritten as dense takes the block dtype then."""
    off, cc, fi = _csr(rng, 200)
    blobs = zb.encode_object_manifests_csr(cc, fi, off, sid_ndim=3)
    zb.write_object_manifests(lg, blobs, mode="replace", layout="vlen")
    zb.commit_object_index(lg, 200, sid_ndim=3)
    assert not zb.is_dense_index(lg)
    zb.write_object_manifests(lg, blobs, mode="replace", layout="dense")
    zb.commit_object_index(lg, 200, sid_ndim=3)


PATHS = {
    "append": _by_append, "blobs": _by_blobs, "write_object_index": _by_index,
    "replace": _by_replace, "place": _by_place, "object_manifest_writer": _by_stream,
    "appender": _by_appender, "patch": _by_patch, "torn_flush": _by_torn_flush,
    "vlen_rewrite": _by_vlen_rewrite,
}


@pytest.mark.parametrize("path", sorted(PATHS))
def test_every_writer_round_trips_int32_as_int64(tmp_path, path):
    lg32, lg64 = _pair(tmp_path, path)
    PATHS[path](lg32, np.random.default_rng(10))
    PATHS[path](lg64, np.random.default_rng(10))
    assert _blocks(lg32).dtype == np.int32 and _blocks(lg64).dtype == np.int64
    np.testing.assert_array_equal(_blocks(lg32)[:], _blocks(lg64)[:])
    assert _reads(lg32) == _reads(lg64)


def test_vacuum_keeps_the_dtype(tmp_path):
    root = _root(tmp_path, "v", store_dtype="int32")
    lg = zb.get_resolution_level(root, 0)
    off, cc, fi = _csr(np.random.default_rng(11), 60)
    off = np.concatenate([[0], np.cumsum(np.maximum(np.diff(off), 1))]).astype(np.int64)
    k = int(off[-1])
    cc, fi = np.zeros((k, 3), np.int64), np.arange(k, dtype=np.int64) + (I32_MAX - k)
    zb.write_object_index(lg, _as_dict(off, cc, fi), 3)
    zb.patch_object_manifests(lg, {o: [] for o in range(0, 60, 3)}, 3)
    want = [m for m in zb.read_all_object_manifests(lg) if m]
    from zarr_vectors.ops import vacuum

    report = vacuum(root)
    assert len(report.oid_remap) == 40
    assert _blocks(lg).dtype == np.int32
    assert zb.read_all_object_manifests(lg) == want


def test_a_full_rewrite_may_change_the_dtype(tmp_path):
    lg = _level(tmp_path, "a")
    off, cc, fi = _csr(np.random.default_rng(12), 50)
    zb.write_object_index(lg, _as_dict(off, cc, fi), 3)
    want = zb.read_all_object_manifests(lg)
    zb.write_object_index(lg, _as_dict(off, cc, fi), 3, block_dtype="int32")
    assert _blocks(lg).dtype == np.int32 and zb.read_all_object_manifests(lg) == want
    assert lg.read_array_meta(OBJECT_INDEX)[dense.BLOCK_DTYPE_ATTR] == "int32"
    zb.write_object_index(lg, _as_dict(off, cc, fi), 3)          # keeps int32
    assert _blocks(lg).dtype == np.int32
    _write(lg, off, cc, fi, mode="replace", block_dtype="int64")
    assert _blocks(lg).dtype == np.int64
    assert lg.read_array_meta(OBJECT_INDEX)[dense.BLOCK_DTYPE_ATTR] == "int64"


def test_a_shard_repack_keeps_int32(tmp_path):
    lg = _level(tmp_path, "a", store_dtype="int32")
    _by_append(lg, np.random.default_rng(13))
    want = _reads(lg)
    zb.shard_object_layer(lg, 1 << 20)
    assert _blocks(lg).dtype == np.int32 and _blocks(lg).shards[0] == 1 << 20
    assert _reads(lg) == want
    off, cc, fi = _csr(np.random.default_rng(14), 10)
    _write(lg, off, cc, fi, mode="append", at=700)          # appends stay checked
    with pytest.raises(ArrayError, match="does not fit"):
        _write(lg, *_too_big(), mode="append", at=710)


def test_a_sharded_reservation_is_int32(tmp_path):
    lg = _level(tmp_path, "a", store_dtype="int32")
    zb.reserve_object_rows(lg, 0, sid_ndim=3, n_blocks=0, shard_rows=1 << 20)
    assert _blocks(lg).dtype == np.int32 and _blocks(lg).shards[0] == 1 << 20
    assert lg.read_array_meta(OBJECT_INDEX)[dense.BLOCK_DTYPE_ATTR] == "int32"


# --------------------------------------------------------------------
# Overflow raises, on every path, and writes nothing


def _int32_index(tmp_path, name="a", n=40):
    lg = _level(tmp_path, name, store_dtype="int32")
    off, cc, fi = _csr(np.random.default_rng(20), n)
    _write(lg, off, cc, fi, mode="append", at=0)
    zb.commit_object_index(lg, n, sid_ndim=3)
    return lg


def _state(lg):
    return (_blocks(lg)[:].tolist(), lg.zarr_group[dense.SPANS_PATH][:].tolist())


@pytest.mark.parametrize("kind", ["fragment", "coordinate"])
def test_overflow_raises_on_append_and_writes_nothing(tmp_path, kind):
    lg = _int32_index(tmp_path)
    before = _state(lg)
    with pytest.raises(ArrayError, match="does not fit object_index/manifest_blocks"):
        _write(lg, *_too_big(kind), mode="append", at=40)
    off, cc, fi = _too_big(kind)
    blobs = zb.encode_object_manifests_csr(cc, fi, off, sid_ndim=3)
    with pytest.raises(ArrayError, match="does not fit"):
        zb.write_object_manifests(lg, blobs, mode="append", at=40)
    assert _state(lg) == before


def test_overflow_names_the_value(tmp_path):
    lg = _int32_index(tmp_path)
    with pytest.raises(ArrayError, match=r"fragment index 1099511627781 "):
        _write(lg, *_too_big("fragment"), mode="append", at=40)
    with pytest.raises(ArrayError, match=r"chunk coordinate 0 -2147483649 "):
        _write(lg, *_too_big("coordinate"), mode="append", at=40)


def test_overflow_raises_on_a_torn_flush_resume(tmp_path):
    lg = _int32_index(tmp_path)
    before = _state(lg)
    with pytest.raises(ArrayError, match="does not fit"):
        _write(lg, *_too_big(), mode="append", at=10)
    assert _state(lg) == before          # nothing truncated either


def test_overflow_raises_on_replace_and_write_object_index(tmp_path):
    lg = _int32_index(tmp_path)
    before = _state(lg)
    with pytest.raises(ArrayError, match="does not fit"):
        _write(lg, *_too_big(), mode="replace")
    with pytest.raises(ArrayError, match="does not fit"):
        zb.write_object_index(lg, {0: [((0, 0, 0), 2**33)]}, 3)
    assert _state(lg) == before


def test_overflow_raises_on_patch(tmp_path):
    lg = _int32_index(tmp_path)
    before = _state(lg)
    with pytest.raises(ArrayError, match="does not fit"):
        zb.patch_object_manifests(lg, {3: [((0, 0, 0), 2**31)]}, 3)
    assert _state(lg) == before


def test_overflow_raises_on_place(tmp_path):
    lg = _level(tmp_path, "a", store_dtype="int32")
    zb.reserve_object_rows(lg, 10, sid_ndim=3, n_blocks=10)
    before = _state(lg)
    with pytest.raises(ArrayError, match="does not fit"):
        _write(lg, *_too_big(), mode="place", at=0, block_at=0)
    assert _state(lg) == before


def test_overflow_raises_on_the_streaming_writer(tmp_path):
    lg = _int32_index(tmp_path)
    off, cc, fi = _csr(np.random.default_rng(21), 10)
    with pytest.raises(ArrayError, match="object_manifest_writer: fragment index"):
        with zb.object_manifest_writer(lg, at=40, commit=True) as w:
            w.write(chunk_coords=cc, fragment_idx=fi, manifest_offsets=off)
            w.write(chunk_coords=_too_big()[1], fragment_idx=_too_big()[2])
    # The writer left by the exception: its first write landed, uncommitted.
    assert lg.read_array_meta(OBJECT_INDEX)["num_objects"] == 40
    # A first write that overflows is refused by the per-call path.
    with pytest.raises(ArrayError, match="does not fit"):
        with zb.object_manifest_writer(lg, at=40) as w:
            w.write(chunk_coords=_too_big()[1], fragment_idx=_too_big()[2])


def test_overflow_raises_on_the_appender(tmp_path):
    lg = _int32_index(tmp_path)
    with pytest.raises(ArrayError, match="does not fit"):
        with ObjectIndexAppender(lg, 40, 3, []) as app:
            app.append([[((0, 0, 0), 2**31)]], [1])


def test_overflow_raises_on_a_vlen_rewrite(tmp_path):
    lg = _level(tmp_path, "a", store_dtype="int32", layout="vlen")
    zb.write_object_index(lg, {0: [((0, 0, 0), 2**35)]}, 3)
    assert not zb.is_dense_index(lg)
    blobs = zb.encode_object_manifests_csr(
        np.zeros((1, 3), np.int64), np.array([2**35]), np.array([0, 1]), sid_ndim=3,
    )
    with pytest.raises(ArrayError, match="does not fit"):
        zb.write_object_manifests(lg, blobs, mode="replace", layout="dense")
    # Refused before the vlen index went.
    assert not zb.is_dense_index(lg)
    assert zb.read_all_object_manifests(lg) == [[((0, 0, 0), 2**35)]]
    zb.write_object_manifests(lg, blobs, mode="replace", layout="dense", block_dtype="int64")
    assert _blocks(lg).dtype == np.int64


# --------------------------------------------------------------------
# An index keeps its dtype


def test_an_int64_index_stays_int64_under_an_int32_default(tmp_path, monkeypatch):
    lg = _level(tmp_path, "a")
    _by_append(lg, np.random.default_rng(30))
    assert _blocks(lg).dtype == np.int64
    # The store's default changes to int32 after the index was created.
    monkeypatch.setattr(dense, "store_block_dtype", lambda _lg: np.dtype(np.int32))
    off, cc, fi = _csr(np.random.default_rng(31), 10, top=2**40)
    _write(lg, off, cc, fi, mode="append", at=700)
    with zb.object_manifest_writer(lg, at=710, commit=True) as w:
        w.write(chunk_coords=cc, fragment_idx=fi, manifest_offsets=off)
    zb.patch_object_manifests(lg, {5: [((0, 0, 0), 2**40)]}, 3)
    zb.reserve_object_rows(lg, 2_000, n_blocks=200_000)
    zb.write_object_index(lg, {0: [((0, 0, 0), 2**40)]}, 3)         # a full rewrite
    assert _blocks(lg).dtype == np.int64
    assert zb.read_all_object_manifests(lg) == [[((0, 0, 0), 2**40)]]


def test_a_reservation_grows_int32_and_refuses_another_dtype(tmp_path):
    lg = _level(tmp_path, "a", store_dtype="int32")
    zb.reserve_object_rows(lg, 10, sid_ndim=3, n_blocks=10)
    zb.reserve_object_rows(lg, 100, n_blocks=200)
    zb.reserve_object_rows(lg, 200, n_blocks=400, block_dtype="int32")
    assert _blocks(lg).dtype == np.int32 and _blocks(lg).shape[0] == 400
    with pytest.raises(ArrayError, match="keeps the dtype it was created with"):
        zb.reserve_object_rows(lg, 300, n_blocks=500, block_dtype="int64")
    assert _blocks(lg).shape[0] == 400
    with pytest.raises(ArrayError, match="keeps the dtype"):
        _write(lg, *_csr(np.random.default_rng(32), 2), mode="place", at=0, block_at=0,
               block_dtype="int64")
    with pytest.raises(ArrayError, match="keeps the dtype"):
        with zb.object_manifest_writer(lg, at=0, block_dtype="int64") as w:
            w.write(chunk_coords=np.zeros((1, 3), np.int64), fragment_idx=np.array([1]))


def test_an_int64_reservation_refuses_int32(tmp_path):
    lg = _level(tmp_path, "a")
    zb.reserve_object_rows(lg, 10, sid_ndim=3, n_blocks=10)
    with pytest.raises(ArrayError, match="keeps the dtype"):
        zb.reserve_object_rows(lg, 20, n_blocks=20, block_dtype="int32")
    with pytest.raises(ArrayError, match="keeps the dtype"):
        _write(lg, *_csr(np.random.default_rng(33), 2), mode="append", block_dtype="int32")


# --------------------------------------------------------------------
# Reads off the store's bytes: offline, on the device; validation


def test_offline_reads_return_int64(tmp_path):
    from zarr_vectors.core.aio import read_async
    from zarr_vectors.types.polylines import read_polylines, write_polylines

    rng = np.random.default_rng(40)
    lines = [rng.uniform(5, 95, (int(rng.integers(2, 9)), 3)).astype("float32")
             for _ in range(30)]
    got = {}
    for dt in ("int32", "int64"):
        root = _root(tmp_path, dt, store_dtype=dt)
        write_polylines(root, lines, chunk_shape=(50.0,) * 3)
        lg = zb.get_resolution_level(root, 0)
        assert _blocks(lg).dtype == np.dtype(dt)

        def reader(r):
            return zb.read_all_object_manifests_csr(zb.get_resolution_level(r, 0))

        csr = asyncio.run(read_async(reader, str(tmp_path / f"{dt}.zarrvectors")))
        assert csr.chunk_coords.dtype == csr.fragment_idx.dtype == np.int64
        sync = zb.read_all_object_manifests_csr(lg)
        np.testing.assert_array_equal(csr.fragment_idx, sync.fragment_idx)
        np.testing.assert_array_equal(csr.chunk_coords, sync.chunk_coords)
        got[dt] = (asyncio.run(read_async(read_polylines, str(tmp_path / f"{dt}.zarrvectors"))),
                   csr)
    assert len(got["int32"][0]["polylines"]) == len(got["int64"][0]["polylines"]) == 30
    for a, b in zip(got["int32"][0]["polylines"], got["int64"][0]["polylines"]):
        assert [np.asarray(x).tolist() for x in a] == [np.asarray(x).tolist() for x in b]
    np.testing.assert_array_equal(got["int32"][1].fragment_idx, got["int64"][1].fragment_idx)


def test_device_reads_return_int64(tmp_path):
    if not zv.runtime_capabilities().get("device_arrays"):
        pytest.skip("no device arrays in this install")
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)
    except Exception as exc:  # noqa: BLE001 - no usable device
        pytest.skip(f"no usable device: {exc}")
    lg = _level(tmp_path, "a", store_dtype="int32")
    _by_append(lg, np.random.default_rng(41))
    host = zb.read_all_object_manifests_csr(lg)
    dev = zb.read_all_object_manifests_csr(lg, device="cuda")
    assert dev.chunk_coords.dtype == dev.fragment_idx.dtype == np.int64
    np.testing.assert_array_equal(cp.asnumpy(dev.fragment_idx), host.fragment_idx)
    np.testing.assert_array_equal(cp.asnumpy(dev.chunk_coords), host.chunk_coords)


def test_the_validator_accepts_int32_and_catches_a_wrong_stamp(tmp_path):
    from zarr_vectors.types.polylines import write_polylines
    from zarr_vectors.validate import validate

    root = _root(tmp_path, "a", store_dtype="int32")
    rng = np.random.default_rng(42)
    write_polylines(root, [rng.uniform(5, 95, (6, 3)).astype("float32") for _ in range(8)],
                    chunk_shape=(50.0,) * 3)
    path = str(tmp_path / "a.zarrvectors")
    lg = zb.get_resolution_level(zb.open_store(path, mode="r+"), 0)
    assert _blocks(lg).dtype == np.int32
    assert validate(path, level=5).ok
    lg.write_array_meta(OBJECT_INDEX, {dense.BLOCK_DTYPE_ATTR: "int64"})
    result = validate(path, level=3)
    assert not result.ok and "block_dtype" in result.summary()


def test_the_stored_array_is_int32(tmp_path):
    """What is stored, not only what is read: the chunks hold 4-byte values."""
    lg32, lg64 = _pair(tmp_path, "b")
    off = np.arange(0, 300_001, 3, dtype=np.int64)
    cc = np.repeat(np.arange(100_000) // 2_000, 3)[:, None].repeat(3, 1).astype(np.int64)
    fi = np.random.default_rng(43).integers(0, 1 << 17, 300_000).astype(np.int64)
    for lg in (lg32, lg64):
        _write(lg, off, cc, fi, mode="append", at=0)
    meta = json.loads((lg32.path / dense.BLOCKS_PATH / "zarr.json").read_text())
    assert meta["data_type"] == "int32"
    raw = {
        dt: np.asarray(lg.zarr_group[dense.BLOCKS_PATH][: dense.BLOCKS_BUCKET]).nbytes
        for dt, lg in (("int32", lg32), ("int64", lg64))
    }
    assert 2 * raw["int32"] == raw["int64"]


def test_the_feature_and_the_api_version():
    assert "manifest-block-dtype" in zv.FEATURES
    from zarr_vectors._api_version import __api_version__

    assert __api_version__ >= (1, 5)
    zv.require_api(">=1.5", features=["manifest-block-dtype", "manifest-blocks-64k",
                                      "manifest-codec"])
