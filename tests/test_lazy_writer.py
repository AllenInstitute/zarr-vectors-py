"""Tests for the ZVWriter (Tier A + append_vertices)."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from zarr_vectors.core.arrays import read_all_object_manifests
from zarr_vectors.core.store import (
    get_resolution_level,
    open_store,
    read_root_metadata,
)
from zarr_vectors.lazy.store import open_zv
from zarr_vectors.types.points import read_points, write_points


def _run(coro):
    return asyncio.run(coro)


def _make_store(tmp_path, n=200):
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 100, (n, 3)).astype("f4")
    store = tmp_path / "p.zv"
    write_points(
        str(store), pos,
        chunk_shape=(50.0, 50.0, 50.0),
        object_ids=np.arange(n, dtype=np.int64),
    )
    return store, pos


# ===================================================================
# Tier A — add_attribute
# ===================================================================


def test_add_attribute_round_trip(tmp_path):
    store, pos = _make_store(tmp_path, n=200)
    normals = np.random.default_rng(1).normal(size=(200, 3)).astype("f4")

    async def go():
        zv = open_zv(str(store))
        async with zv[0].writer() as w:
            await w.add_attribute("normal", normals)

    _run(go())

    out = read_points(str(store), attribute_names=["normal"])
    assert "normal" in out["vertex_attributes"]
    # Data flattens via read_points's ncols=1 path; total count matches.
    assert out["vertex_attributes"]["normal"].size == 200 * 3


def test_add_attribute_sync_mirror(tmp_path):
    store, _ = _make_store(tmp_path, n=120)
    rng = np.random.default_rng(2)
    intensities = rng.uniform(0, 1, 120).astype("f4")

    zv = open_zv(str(store))
    with zv[0].writer() as w:
        w.add_attribute_sync("intensity", intensities)

    out = read_points(str(store), attribute_names=["intensity"])
    assert out["vertex_attributes"]["intensity"].size == 120


def test_add_attribute_length_mismatch_raises(tmp_path):
    store, _ = _make_store(tmp_path, n=50)
    bad = np.zeros(51, dtype="f4")  # one too many

    async def go():
        zv = open_zv(str(store))
        async with zv[0].writer() as w:
            await w.add_attribute("bad", bad)

    from zarr_vectors.exceptions import ArrayError
    with pytest.raises(ArrayError, match="!= level vertex count"):
        _run(go())


def test_add_object_attribute(tmp_path):
    from zarr_vectors.core.arrays import read_object_attributes
    store, _ = _make_store(tmp_path, n=80)

    async def go():
        zv = open_zv(str(store))
        async with zv[0].writer() as w:
            await w.add_object_attribute("score", np.arange(80, dtype="f4"))

    _run(go())

    root = open_store(str(store))
    lvl = get_resolution_level(root, 0)
    scores = read_object_attributes(lvl, "score")
    assert scores.shape == (80,)
    assert float(scores[5]) == 5.0


# ===================================================================
# append_vertices (commits directly into object_index/ in 0.6.0+)
# ===================================================================


def test_append_vertices_grows_store(tmp_path):
    store, _ = _make_store(tmp_path, n=100)
    new_pos = np.random.default_rng(3).uniform(0, 100, (40, 3)).astype("f4")

    async def go():
        zv = open_zv(str(store))
        async with zv[0].writer() as w:
            result = await w.append_vertices(new_pos)
            return result

    summary = _run(go())
    assert summary["vertices_added"] == 40
    assert summary["new_objects"] == 40

    out = read_points(str(store))
    assert out["vertex_count"] == 140


def test_append_then_compact_is_a_no_op(tmp_path):
    """0.6.0+: compact() is a compatibility shim that just reports counts.

    Pending-sidecar staging was removed; every append commits directly
    into ``object_index/``.
    """
    store, _ = _make_store(tmp_path, n=60)

    async def go():
        zv = open_zv(str(store))
        async with zv[0].writer() as w:
            await w.append_vertices(
                np.random.default_rng(4).uniform(0, 100, (10, 3)).astype("f4")
            )

    _run(go())

    async def do_compact():
        zv = open_zv(str(store))
        async with zv[0].writer() as w:
            return await w.compact()

    result = _run(do_compact())
    assert result["compacted"] is True
    assert result["num_objects"] == 70

    assert read_points(str(store))["vertex_count"] == 70


def test_two_sequential_appends_merge_into_object_index(tmp_path):
    store, _ = _make_store(tmp_path, n=30)

    async def go():
        zv = open_zv(str(store))
        async with zv[0].writer() as w:
            await w.append_vertices(
                np.random.default_rng(5).uniform(0, 100, (5, 3)).astype("f4")
            )
        zv = open_zv(str(store))
        async with zv[0].writer() as w:
            await w.append_vertices(
                np.random.default_rng(6).uniform(0, 100, (7, 3)).astype("f4")
            )

    _run(go())

    root = open_store(str(store))
    lvl = get_resolution_level(root, 0)
    manifests = read_all_object_manifests(lvl)
    assert len(manifests) == 42


def test_append_vertices_overlap_oid_raises(tmp_path):
    store, _ = _make_store(tmp_path, n=20)
    overlap = np.array([5, 6, 7], dtype=np.int64)  # collide with existing

    async def go():
        zv = open_zv(str(store))
        async with zv[0].writer() as w:
            await w.append_vertices(
                np.zeros((3, 3), dtype="f4"),
                object_ids=overlap,
            )

    from zarr_vectors.exceptions import ArrayError
    with pytest.raises(ArrayError, match="overlap existing"):
        _run(go())


# ===================================================================
# Sync mirrors: no deadlock, one prefetch, the declared dtype
# ===================================================================

# Run in a child process: a deadlock here is permanent, and inside the
# test process it would also leave zarr's global thread pool blocked for
# every test after it.  ZARR_THREADING__MAX_WORKERS pins zarr's pool at 4
# threads, so the hang threshold does not depend on the machine's cores.
_DEADLOCK_CHILD = """
import sys, warnings
import numpy as np
warnings.simplefilter("ignore")
from zarr.core import sync as zsync
from zarr_vectors.lazy.store import open_zv
from zarr_vectors.types.points import read_points, write_points

path, mode = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)
n = 400
pos = rng.uniform(0, 100, (n, 3)).astype("f4")
kw = {}
if mode == "attr":
    genes = np.array(["A", "B"])[rng.integers(0, 2, n)]
    kw = dict(vertex_attributes={"gene": genes}, chunk_by_attribute="gene")
write_points(path, pos, chunk_shape=(25.0, 25.0, 25.0),
             bounds=[[0, 0, 0], [100, 100, 100]], object_ids=np.arange(n), **kw)
assert zsync._executor._max_workers == 4, zsync._executor._max_workers
lg = open_zv(path)[0]._group
n_keys = len(lg.list_chunks("vertices"))
assert n_keys >= 8, n_keys  # well past the pool, or the test proves nothing
w = open_zv(path)[0].writer()
if mode == "append":
    w.append_vertices_sync(rng.uniform(0, 100, (64, 3)).astype("f4"))
    w.commit_sync()
    assert read_points(path)["vertex_count"] == n + 64
else:
    w.add_attribute_sync("x", np.arange(n, dtype="f4"))
    assert len(lg.list_chunks("vertex_attributes/x")) == n_keys
print("DONE")
"""


@pytest.mark.parametrize("mode", ["attr", "plain", "append"])
def test_sync_mirrors_do_not_deadlock_past_zarrs_thread_pool(tmp_path, mode):
    """Past as many chunk keys as zarr has threads, these hung forever."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "ZARR_THREADING__MAX_WORKERS": "4"}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _DEADLOCK_CHILD, str(tmp_path / "s.zv"), mode],
            env=env, capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{mode}: the sync mirror deadlocked")
    assert proc.returncode == 0, proc.stderr
    assert "DONE" in proc.stdout


def test_the_writer_loop_is_its_own_and_is_reused():
    from zarr.core.sync import _get_loop

    from zarr_vectors.lazy.writer import _writer_loop

    loop = _writer_loop()
    assert loop is not _get_loop()
    assert _writer_loop() is loop
    assert loop.is_running()


def test_add_attribute_prefetches_the_level_once(tmp_path, monkeypatch):
    """One prefetch for the level, not one per chunk racing for the slot."""
    from zarr_vectors.core.group import Group

    store, _ = _make_store(tmp_path, n=200)
    lg = get_resolution_level(open_store(str(store)), 0)
    assert len(lg.list_chunks("vertices")) == 8
    zv = open_zv(str(store))
    w = zv[0].writer()

    calls = []
    real = Group.batched_reads

    def _counting(self, *args, **kw):
        calls.append(1)
        return real(self, *args, **kw)

    monkeypatch.setattr(Group, "batched_reads", _counting)
    w.add_attribute_sync("x", np.zeros(200, dtype="f4"))

    assert len(calls) == 1


@pytest.mark.parametrize("path", ["sync", "async"])
def test_add_attribute_on_a_float64_store_lines_up_with_the_vertices(tmp_path, path):
    """A guard, not a regression: the writer read cells as float32.

    The row counts it needs come from the fragment index, so the float32
    read counted them right even on a float64 level; the decoded values
    were garbage but unused.  It now reads at the declared dtype, and this
    pins that attribute rows line up with a float64 level's vertices.
    """
    from zarr_vectors.core.arrays import (
        list_chunk_keys,
        read_chunk_attributes,
        read_chunk_vertices,
    )

    rng = np.random.default_rng(4)
    store = str(tmp_path / "f64.zv")
    write_points(
        store, rng.uniform(0, 100, (120, 3)), dtype="float64",
        chunk_shape=(50.0, 50.0, 50.0), object_ids=np.arange(120),
    )
    lg = get_resolution_level(open_store(store), 0)
    keys = list_chunk_keys(lg)
    cells = {
        cc: np.concatenate(read_chunk_vertices(lg, cc, dtype="float64", ndim=3))
        for cc in keys
    }
    # The x column, in the order the level stores its vertices.
    xs = np.concatenate([cells[cc][:, 0] for cc in keys])

    w = open_zv(store)[0].writer()
    if path == "sync":
        w.add_attribute_sync("x", xs)
    else:
        _run(w.add_attribute("x", xs))

    lg = get_resolution_level(open_store(store), 0)
    for cc in keys:
        got = np.concatenate(
            read_chunk_attributes(lg, "x", cc, dtype="float64", ncols=1),
        ).ravel()
        np.testing.assert_array_equal(got, cells[cc][:, 0])
