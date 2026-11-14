"""The read experiments return exactly what the default read returns.

``ZARR_VECTORS_EXPERIMENTS`` switches on read-path changes that only
alter how a read fetches (see ``zarr_vectors/_experiments.py``).  Each is
run here against the default on point clouds, meshes and graphs, local
and through a store wrapper (the object-store path), sharded and not.
"""

from __future__ import annotations

import numpy as np
import pytest
from zarr.storage import LocalStore, WrapperStore

import zarr_vectors as zv

BOX = 100.0
EXPERIMENTS = ["level_handle", "prime_nodes", "gather_indexes",
               "level_handle,prime_nodes,gather_indexes"]


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
def test_experiments_read_what_the_default_reads(tmp_path, monkeypatch, kind, shard, remote, reader):
    path = _store(tmp_path, kind, shard)
    monkeypatch.setenv("ZARR_VECTORS_SHARD_READER", reader)
    monkeypatch.delenv("ZARR_VECTORS_EXPERIMENTS", raising=False)
    want = [_answer(path, sel, remote) for sel in QUERIES]
    for experiments in EXPERIMENTS:
        monkeypatch.setenv("ZARR_VECTORS_EXPERIMENTS", experiments)
        for sel, ref in zip(QUERIES, want):
            got = _answer(path, sel, remote)
            assert got.keys() == ref.keys(), (experiments, sel)
            assert np.array_equal(got["positions"], ref["positions"]), (experiments, sel)
            for name in ("faces", "edges"):
                assert got.get(name) == ref.get(name), (experiments, sel, name)


def test_primed_nodes_end_with_the_query(tmp_path, monkeypatch):
    from zarr_vectors import _query_nodes

    path = _store(tmp_path, "point_cloud", 4)
    monkeypatch.setenv("ZARR_VECTORS_EXPERIMENTS", "prime_nodes")
    zv.open(str(path)).select(bbox=([30.0] * 3, [60.0] * 3)).read()
    assert not _query_nodes._nodes and not _query_nodes._active
