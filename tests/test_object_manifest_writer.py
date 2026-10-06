"""building.object_manifest_writer stores what per-call writes store.

The reference is the same sequence of ``write_object_manifests(mode=
"append", at=<end of the previous write>)`` calls: every file of the
store must come out byte for byte the same, vlen and dense, sharded and
not, from a fresh index, at the end of a committed one, after a pad and
over a torn flush's residue, with ids given or not. Flushes are made
small so that rows cross storage-object boundaries many times; one run
keeps the default.

A stream interrupted by an exception leaves only rows past the committed
count -- residue -- and the next stream from that count stores what an
uninterrupted one would have.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors import building as zb
from zarr_vectors.constants import OBJECT_INDEX
from zarr_vectors.core import dense_manifests as dense
from zarr_vectors.core import manifest_writer as mw
from zarr_vectors.core.arrays import write_object_index
from zarr_vectors.exceptions import ArrayError

from ._store_compare import assert_stores_identical

pytestmark = pytest.mark.vlen_only  # the layouts are chosen here

ROWS = zb.OBJECT_SHARD_ROW_MULTIPLE


def _store(root, layout):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "s.zv"
    zb.create_store(
        str(path), bounds=([0.0] * 3, [100.0] * 3), chunk_shape=(50.0,) * 3,
        geometry_types=["polyline"], manifest_layout=layout,
    )
    return path


def _level(path):
    return zb.get_resolution_level(zb.open_store(str(path), mode="r+"), 0)


def _writes(seed, total, *, max_n=3_000, ids_base=None, blobs_every=0):
    """A sequence of write() argument dicts covering ``total`` objects."""
    rng = np.random.default_rng(seed)
    out, done, i = [], 0, 0
    while done < total:
        n = min(int(rng.integers(0, max_n)), total - done)
        if i % 7 == 3:
            n = 0  # empty writes are writes too
        counts = np.where(rng.random(n) < 0.2, 0, rng.integers(1, 4, n))
        off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        m = int(off[-1])
        args = dict(
            chunk_coords=rng.integers(0, 2, (m, 3)),
            fragment_idx=rng.integers(0, 10_000, m),
            manifest_offsets=off,
        )
        if blobs_every and i % blobs_every == 1:
            args = {"manifest_blobs": zb.encode_object_manifests_csr(
                args["chunk_coords"], args["fragment_idx"], off, sid_ndim=3,
            )}
        if ids_base is not None:
            args["ids"] = ids_base + done + np.arange(n, dtype=np.int64)
        out.append(args)
        done += n
        i += 1
    return out


def _base(level, layout, scenario, n0):
    """The index the stream starts on, and the ``at`` it starts at."""
    if scenario == "fresh":
        return 0
    rng = np.random.default_rng(1)
    write_object_index(level, {
        o: [((0, int(rng.integers(0, 2)), 1), int(rng.integers(0, 99)))] for o in range(n0)
    }, 3)
    if scenario == "end":
        return None
    if scenario == "pad":
        return n0 + 37 if layout == "dense" else None  # a stored-id vlen index cannot pad
    # "resume": residue past the committed count
    zb.write_object_manifests(
        level, chunk_coords=np.ones((50, 3), np.int64), fragment_idx=np.arange(50),
        mode="append", at=n0,
    )
    return n0


def _per_call(level, writes, at, layout):
    pos = at
    for k, args in enumerate(writes):
        first, n = zb.write_object_manifests(
            level, **args, mode="append", at=pos, layout=layout if k == 0 else None,
        )
        pos = first + n
    return pos


def _streamed(level, writes, at, layout, **kw):
    with zb.object_manifest_writer(level, at=at, layout=layout, **kw) as w:
        for args in writes:
            w.write(**args)
    # Streamed, in several flushes -- not the per-call fallback.
    assert not w._direct and w.flushes > 1
    return w.end


CASES = [
    ("vlen", "fresh", None), ("vlen", "end", "ids"), ("vlen", "resume", None),
    ("vlen", "pad", None),
    ("dense", "fresh", None), ("dense", "end", "ids"), ("dense", "resume", None),
    ("dense", "resume", "ids"), ("dense", "pad", None),
]


@pytest.mark.parametrize("sharded", [False, True], ids=["unsharded", "sharded"])
@pytest.mark.parametrize(("layout", "scenario", "ids"), CASES)
def test_the_stream_stores_what_per_call_writes_store(
    tmp_path, monkeypatch, layout, scenario, ids, sharded,
):
    monkeypatch.setattr(mw, "FLUSH_ROWS", 1_000)
    monkeypatch.setattr(mw, "FLUSH_BLOCKS", 2_500)
    total = (2 * ROWS + 5_000) if sharded else 40_000
    n0 = 1_000

    def build(root):
        path = _store(root, layout)
        lg = _level(path)
        at = _base(lg, layout, scenario, n0)
        if sharded:
            zb.shard_object_layer(lg, ROWS) if scenario != "fresh" else None
        return path, lg, at

    a, la, at = build(tmp_path / "calls")
    b, lb, _ = build(tmp_path / "stream")
    if sharded and scenario == "fresh":
        # A fresh index takes the layer's layout from what is there.
        for lg in (la, lb):
            zb.reserve_object_rows(lg, 0, sid_ndim=3, layout=layout, shard_rows=ROWS)
            zb.commit_object_index(lg, 0, sid_ndim=3)
    start = at if at is not None else n0
    writes = _writes(7, total, ids_base=(10**9 if ids else None), blobs_every=5)
    end_a = _per_call(la, writes, at, layout)
    end_b = _streamed(lb, writes, at, layout)
    assert end_a == end_b == start + total
    assert_stores_identical(a, b)


def test_the_default_flush_sizes(tmp_path):
    a, b = _store(tmp_path / "calls", "dense"), _store(tmp_path / "stream", "dense")
    writes = _writes(3, 300_000, max_n=20_000)
    _per_call(_level(a), writes, 0, None)
    lb = _level(b)
    with zb.object_manifest_writer(lb, at=0) as w:
        for args in writes:
            w.write(**args)
    assert w.flushes >= 2
    assert_stores_identical(a, b)


@pytest.mark.parametrize("layout", ["vlen", "dense"])
@pytest.mark.parametrize("sharded", [False, True], ids=["unsharded", "sharded"])
def test_an_interrupted_stream_leaves_residue_the_next_one_replaces(
    tmp_path, monkeypatch, layout, sharded,
):
    monkeypatch.setattr(mw, "FLUSH_ROWS", 1_000)
    monkeypatch.setattr(mw, "FLUSH_BLOCKS", 2_500)
    n0 = 500
    total = (ROWS + 9_000) if sharded else 40_000
    writes = _writes(11, total)

    def base(root):
        path = _store(root, layout)
        lg = _level(path)
        write_object_index(lg, {o: [((1, 0, 1), o)] for o in range(n0)}, 3)
        if sharded:
            zb.shard_object_layer(lg, ROWS)
        return path, lg

    clean, lc = base(tmp_path / "clean")
    _streamed(lc, writes, n0, None)
    zb.commit_object_index(lc, n0 + total, sid_ndim=3)

    torn, lt = base(tmp_path / "torn")
    with pytest.raises(RuntimeError, match="worker died"):
        with zb.object_manifest_writer(lt, at=n0) as w:
            for k, args in enumerate(writes):
                w.write(**args)
                if k == len(writes) * 2 // 3:
                    raise RuntimeError("worker died")
    lt = _level(torn)
    meta = lt.read_array_meta(OBJECT_INDEX)
    assert meta["num_objects"] == n0  # nothing committed
    rows = dense.num_rows(lt) if layout == "dense" else int(lt.zarr_group[f"{OBJECT_INDEX}/manifests"].shape[0])
    assert n0 < rows < n0 + total  # residue, short of the whole stream
    if layout == "dense":
        # A row on disk never names a block that is not.
        offsets, _, _ = dense.read_csr(lt, stop=rows)
        assert int(offsets[-1]) <= dense.num_blocks(lt)

    _streamed(lt, writes, n0, None)
    zb.commit_object_index(lt, n0 + total, sid_ndim=3)
    assert_stores_identical(clean, torn)


@pytest.mark.parametrize("layout", ["vlen", "dense"])
def test_commit_counts_what_was_written(tmp_path, layout):
    a, b = _store(tmp_path / "calls", layout), _store(tmp_path / "stream", layout)
    writes = _writes(5, 30_000)
    la, lb = _level(a), _level(b)
    for lg in (la, lb):
        write_object_index(lg, {o: ([((0, 0, 0), o)] if o % 3 else []) for o in range(200)}, 3)
    end = _per_call(la, writes, 200, None)
    zb.commit_object_index(la, end, sid_ndim=3)
    with zb.object_manifest_writer(lb, at=200, commit=True) as w:
        for args in writes:
            w.write(**args)
    assert_stores_identical(a, b)
    assert lb.read_array_meta(OBJECT_INDEX)["num_objects"] == end


@pytest.mark.parametrize("layout", ["vlen", "dense"])
def test_errors_are_the_per_call_errors(tmp_path, layout):
    path = _store(tmp_path / "s", layout)
    lg = _level(path)
    write_object_index(lg, {o: [((0, 0, 0), o)] for o in range(10)}, 3)
    with zb.object_manifest_writer(lg, at=10) as w:
        w.write(chunk_coords=np.zeros((2, 3), np.int64), fragment_idx=[1, 2])
        with pytest.raises(ArrayError, match="not both"):
            w.write(manifest_blobs=[b"x"], chunk_coords=np.zeros((1, 3), np.int64), fragment_idx=[1])
        if layout == "dense":
            with pytest.raises(ArrayError, match="manifest_offsets"):
                w.write(chunk_coords=np.zeros((2, 3), np.int64), fragment_idx=[1, 2], manifest_offsets=[0, 3])
        with pytest.raises(ArrayError, match="ids for"):
            w.write(chunk_coords=np.zeros((2, 3), np.int64), fragment_idx=[1, 2], ids=[5])
        # Ids that are not the rows, then none: the table is no longer the
        # identity, so the per-call path refuses, and so does the stream.
        w.write(chunk_coords=np.zeros((1, 3), np.int64), fragment_idx=[3], ids=[10**6])
        with pytest.raises(ArrayError, match="pass ids="):
            w.write(chunk_coords=np.zeros((1, 3), np.int64), fragment_idx=[4])
    with pytest.raises(ArrayError, match="closed"):
        w.write(chunk_coords=np.zeros((1, 3), np.int64), fragment_idx=[4])


def test_a_stream_that_writes_nothing_changes_nothing(tmp_path):
    a, b = _store(tmp_path / "a", "dense"), _store(tmp_path / "b", "dense")
    for p in (a, b):
        write_object_index(_level(p), {0: [((0, 0, 0), 1)]}, 3)
    with zb.object_manifest_writer(_level(b), at=0, commit=True) as w:
        pass
    assert w.close() is None
    assert_stores_identical(a, b)


def test_the_stream_resolves_the_index_once(tmp_path, monkeypatch):
    path = _store(tmp_path / "s", "dense")
    lg = _level(path)
    calls = {"meta": 0}
    real = type(lg).read_array_meta

    def counting(self, name):
        if name == OBJECT_INDEX:
            calls["meta"] += 1
        return real(self, name)

    writes = _writes(2, 20_000, max_n=500)
    monkeypatch.setattr(type(lg), "read_array_meta", counting)
    with zb.object_manifest_writer(lg, at=0) as w:
        w.write(**writes[0])
        after_first = calls["meta"]
        for args in writes[1:]:
            w.write(**args)
    assert len(writes) > 30
    assert calls["meta"] - after_first <= 2
