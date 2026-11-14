"""A read's round-trip savings change how it fetches, never what it returns.

A query holds its level group (``Level._held_store``), resolves the level
and its arrays in one gather (:mod:`zarr_vectors._query_nodes`) and reads
every array's shard indexes in one round
(``_cells_on_disk._prefetch_indexes``).  Each is patched out here, alone
and together, and the answers compared on point clouds, meshes and
graphs, local and through a store wrapper (the object-store path),
sharded and not.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
from zarr.storage import LocalStore, WrapperStore

import zarr_vectors as zv
from zarr_vectors import _query_nodes
from zarr_vectors.api.level import Level
from zarr_vectors.core import _cells_on_disk

BOX = 100.0
SAVINGS = ["held_level", "primed_nodes", "gathered_indexes"]


def _without(monkeypatch, names):
    """Patch out each saving in ``names``."""
    if "held_level" in names:
        monkeypatch.setattr(Level, "_held_store", lambda self: contextlib.nullcontext())
    if "primed_nodes" in names:
        monkeypatch.setattr(_query_nodes, "primed", lambda root, level: contextlib.nullcontext())
    if "gathered_indexes" in names:
        monkeypatch.setattr(_cells_on_disk, "_prefetch_indexes", lambda requests: {})


class _Remote(WrapperStore):
    """A local store read as a remote one: no local path to read files from."""


def _data(kind, n=2_000, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.uniform(20, 80, (3, 3))
    pts = centres[rng.integers(0, 3, n)] + rng.standard_t(3, (n, 3)) * 6
    pts = np.clip(pts, 0, BOX - 1e-3).astype(np.float32)
    ids = np.arange(n) // 50
    if kind == "mesh":
        # Triangles within each object, between neighbouring rows.
        tri = np.stack([np.arange(0, n - 2), np.arange(1, n - 1), np.arange(2, n)], 1)
        return pts, ids, tri[ids[tri[:, 0]] == ids[tri[:, 2]]]
    if kind == "graph":
        e = np.stack([np.arange(n - 1), np.arange(1, n)], 1)
        return pts, ids, e[ids[e[:, 0]] == ids[e[:, 1]]]
    return pts, ids, None


def _store(tmp_path, kind, shard):
    pts, ids, conn = _data(kind)
    path = tmp_path / f"{kind}-{shard}.zv"
    ds = zv.create(str(path), schema=zv.Schema(
        bounds=([0, 0, 0], [BOX] * 3), kind=kind,
        layout=zv.Layout(cells=8, shard_cells=shard, compression="zstd")))
    if kind == "point_cloud":
        ds.add_points(pts, object_ids=ids)
    elif kind == "mesh":
        ds.add_mesh(pts, conn, object_ids=ids)
    else:
        ds.add_graph(pts, conn, object_ids=ids)
    return path


QUERIES = [
    {},
    {"bbox": ([30.0, 30.0, 30.0], [60.0, 60.0, 60.0])},
    {"bbox": ([0.0, 0.0, 0.0], [5.0, 5.0, 5.0])},
    {"objects": [3]},
    {"objects": [0, 7, 19]},
]


def _answer(path, sel, remote):
    store = _Remote(LocalStore(str(path), read_only=True)) if remote else str(path)
    r = zv.open(store).select(**sel).read()
    out = {"positions": np.sort(np.asarray(r.positions), axis=0)}
    for name in ("faces", "edges"):
        a = getattr(r, name, None)
        if a is not None:
            out[name] = len(a)
    return out


@pytest.mark.parametrize("kind", ["point_cloud", "mesh", "graph"])
@pytest.mark.parametrize("shard", [None, 4])
@pytest.mark.parametrize("remote", [False, True], ids=["local", "remote"])
@pytest.mark.parametrize("reader", ["standard", "optimised"])
def test_savings_read_what_a_read_without_them_reads(
    tmp_path, monkeypatch, kind, shard, remote, reader,
):
    path = _store(tmp_path, kind, shard)
    monkeypatch.setenv("ZARR_VECTORS_SHARD_READER", reader)
    want = [_answer(path, sel, remote) for sel in QUERIES]
    for off in [[name] for name in SAVINGS] + [SAVINGS]:
        with monkeypatch.context() as m:
            _without(m, off)
            for sel, ref in zip(QUERIES, want):
                got = _answer(path, sel, remote)
                assert got.keys() == ref.keys(), (off, sel)
                assert np.array_equal(got["positions"], ref["positions"]), (off, sel)
                for name in ("faces", "edges"):
                    assert got.get(name) == ref.get(name), (off, sel, name)


def test_primed_nodes_end_with_the_query(tmp_path):
    path = _store(tmp_path, "point_cloud", 4)
    zv.open(str(path)).select(bbox=([30.0] * 3, [60.0] * 3)).read()
    assert not _query_nodes._nodes and not _query_nodes._active


class _Gets(WrapperStore):
    """A remote-looking store that counts the cells it serves per array."""

    def __init__(self, store, log):
        super().__init__(store)
        self.log = log

    def _with_store(self, store):
        return type(self)(store, self.log)

    async def get(self, key, prototype, byte_range=None):
        if "/c/" in key or "/c" == key[-2:]:
            self.log.append(key.split("/c/")[0])
        return await super().get(key, prototype, byte_range)


@pytest.mark.parametrize("shard", [None, 4])
def test_a_whole_tiled_point_level_reads_one_fragment_cell(tmp_path, shard):
    """A whole-level read of a point level claiming ``fragments_tile``
    takes the buffers as they are, re-checking the claim on one cell: one
    ``vertex_fragments`` cell is fetched, not every one."""
    path = _store(tmp_path, "point_cloud", shard)
    ds = zv.open(str(path))
    assert ds.level(0)._metadata().fragments_tile
    log: list[str] = []
    got = zv.open(_Gets(LocalStore(str(path), read_only=True), log)).select().read()
    np.testing.assert_array_equal(
        np.sort(np.asarray(got.positions), axis=0), np.sort(_data("point_cloud")[0], axis=0),
    )
    frag = [k for k in log if k.endswith("vertex_fragments")]
    assert log.count("0/vertices") > 4 and len(frag) <= 2, (len(frag), log.count("0/vertices"))
    # Narrowed or by object, the fragments are read as before.
    for sel in ({"bbox": ([30.0] * 3, [60.0] * 3)}, {"objects": [3]}):
        log.clear()
        zv.open(_Gets(LocalStore(str(path), read_only=True), log)).select(**sel).read()
        assert any(k.endswith("vertex_fragments") for k in log), sel
