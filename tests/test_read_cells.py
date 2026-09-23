"""``read_cells`` / ``read_neighbourhood``: many cells, one prefetch, CSR out."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from tests._fake_device import FakeDeviceArray
from zarr_vectors import _xp
from zarr_vectors.core.arrays import (
    _link_cell_rows,
    list_chunk_keys,
    list_link_offsets,
    read_chunk_attribute_rows,
    read_chunk_vertex_buffer,
)
from zarr_vectors.core.cells import read_cells, read_neighbourhood
from zarr_vectors.core.group import Group
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.types.graphs import write_graph
from zarr_vectors.types.points import write_points

_CHUNK = (25.0, 25.0, 25.0)


@pytest.fixture
def points(tmp_path):
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 100, (600, 3)).astype("float32")
    path = str(tmp_path / "p.zarrvectors")
    write_points(
        path, pos, chunk_shape=_CHUNK, bounds=[[0, 0, 0], [100, 100, 100]],
        vertex_attributes={
            "intensity": rng.uniform(0, 1, 600).astype("float32"),
            "rgb": rng.integers(0, 255, (600, 3)).astype("uint8"),
        },
    )
    return path


def _level(path):
    return get_resolution_level(open_store(path), 0)


def _cells(lg, extra=()):
    present = [tuple(k) for k in list_chunk_keys(lg)]
    return np.asarray(present + list(extra), dtype=np.int64)


def test_vertices_and_attributes_match_the_per_cell_readers(points):
    lg = _level(points)
    cells = _cells(lg, extra=[(0, 0, 0), (9, 9, 9)])  # a repeat, and off-grid
    batch = read_cells(
        lg, cells, ["vertices", "vertex_attributes/intensity", "vertex_attributes/rgb"],
    )
    assert not batch.errors
    np.testing.assert_array_equal(batch.chunk_coords, cells)
    for i, cc in enumerate(map(tuple, cells.tolist())):
        on_grid = all(0 <= c < 4 for c in cc)
        want_v = read_chunk_vertex_buffer(lg, cc) if on_grid else np.empty((0, 3))
        np.testing.assert_array_equal(batch["vertices"].rows(i), want_v)
        for name in ("intensity", "rgb"):
            got = batch[f"vertex_attributes/{name}"].rows(i)
            if on_grid:
                want = read_chunk_attribute_rows(lg, name, cc)
                np.testing.assert_array_equal(
                    got.reshape(len(want), -1), want.reshape(len(want), -1),
                )
            else:
                assert got.shape[0] == 0
    col = batch["vertices"]
    assert col.offsets[-1] == len(col.data) == 600 + len(read_chunk_vertex_buffer(lg, (0, 0, 0)))
    np.testing.assert_array_equal(np.bincount(col.cell, minlength=len(cells)), np.diff(col.offsets))


def test_one_prefetch_serves_every_array(points, monkeypatch):
    lg = _level(points)
    calls = []
    real = Group.batched_reads

    def _count(self, plan, **kw):
        calls.append(sorted(name for name, _ in plan))
        return real(self, plan, **kw)

    monkeypatch.setattr(Group, "batched_reads", _count)
    read_cells(lg, _cells(lg), ["vertices", "vertex_attributes/intensity"])
    assert calls == [["vertex_attributes/intensity", "vertices"]]


def test_a_missing_array_raises_or_is_skipped(points):
    lg = _level(points)
    with pytest.raises(ArrayError, match="does not exist"):
        read_cells(lg, _cells(lg), ["vertices", "vertex_attributes/nope"])
    batch = read_cells(
        lg, _cells(lg), ["vertices", "vertex_attributes/nope"], missing_arrays="skip",
    )
    assert list(batch) == ["vertices"]


def test_a_fragment_index_is_refused(points):
    with pytest.raises(ArrayError, match="fragment index"):
        read_cells(_level(points), [(0, 0, 0)], ["vertex_fragments"])


def test_one_bad_cell_is_recorded_and_the_rest_are_read(points):
    lg = _level(points)
    cells = _cells(lg)
    bad = tuple(cells[3].tolist())
    cell_file = Path(points, "0", "vertices", "c", *map(str, bad))
    assert cell_file.exists()
    cell_file.write_bytes(b"\xff\x01")  # not a valid encoded cell

    batch = read_cells(_level(points), cells, ["vertices"])
    assert [(e.array, e.chunk_key) for e in batch.errors] == [
        ("vertices", ".".join(map(str, bad)))
    ]
    assert batch["vertices"].rows(3).shape == (0, 3)
    np.testing.assert_array_equal(
        batch["vertices"].rows(4), read_chunk_vertex_buffer(lg, tuple(cells[4].tolist())),
    )
    with pytest.raises(Exception):
        read_cells(_level(points), cells, ["vertices"], on_error="raise")


def test_link_segments_read_as_physical_rows(tmp_path):
    rng = np.random.default_rng(1)
    n = 120
    pos = rng.uniform(0, 100, (n, 3)).astype("float32")
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    path = str(tmp_path / "g.zarrvectors")
    write_graph(
        path, positions=pos, edges=edges, object_ids=np.zeros(n, dtype=np.int64),
        chunk_shape=(40.0, 40.0, 40.0), bounds=([0, 0, 0], [100, 100, 100]),
    )
    lg = _level(path)
    segs = [f"links/0/{s}" for s in list_link_offsets(lg, 0)]
    cells = np.asarray(sorted({tuple(k) for s in segs for k in lg.list_chunk_coords(s)}))
    batch = read_cells(lg, cells, segs)
    total = 0
    for s in segs:
        meta = lg.read_array_meta(s)
        col = batch[s]
        for i, cc in enumerate(map(tuple, cells.tolist())):
            key = ".".join(map(str, cc))
            raw = lg.read_bytes(s, key) if key in set(lg.list_chunks(s)) else b""
            if not raw:
                assert col.rows(i).shape[0] == 0
                continue
            want = _link_cell_rows(
                raw, ncols=col.data.shape[1], flat=s.endswith("/0.0.0"),
                dtype=meta.get("dtype", "int64"),
            )
            np.testing.assert_array_equal(col.rows(i), want)
        total += len(col.data)
    assert total == n - 1  # every edge, intra or cross-chunk, exactly once


def test_read_neighbourhood_is_the_centre_then_present_neighbours(points):
    lg = _level(points)
    batch = read_neighbourhood(lg, (1, 1, 1), ["vertices"])
    present = {tuple(k) for k in list_chunk_keys(lg)}
    want = [(1, 1, 1)] + sorted(
        c for c in present if c != (1, 1, 1) and max(abs(a - 1) for a in c) <= 1
    )
    assert [tuple(c) for c in batch.chunk_coords.tolist()] == want


def test_device_coords_are_copied_off_once_and_cuda_needs_the_extension(points, monkeypatch):
    import sys

    lg = _level(points)
    cells = _cells(lg)
    with _xp.count_transfers() as stats:
        batch = read_cells(lg, cells, ["vertices"], device="cpu")
        read_cells(lg, FakeDeviceArray(cells), ["vertices"], device="cpu")
    assert stats.d2h_calls == 1 and batch.device == "cpu"
    monkeypatch.setitem(sys.modules, "zarr_vectors.gpu", None)
    from zarr_vectors.exceptions import ZVError

    with pytest.raises(ZVError, match="gpu"):
        read_cells(lg, cells, ["vertices"], device="cuda")


def test_it_replays_offline(points):
    """Under the async prime-and-replay path the planned cells are fetched
    up front, and the replayed pass reads them from the snapshot."""
    from zarr_vectors.core.aio import read_async

    cells = _cells(_level(points))

    def reader(root):
        return read_cells(get_resolution_level(root, 0), cells, ["vertices"])

    got = asyncio.run(read_async(reader, points))
    want = read_cells(_level(points), cells, ["vertices"])
    assert not got.errors
    np.testing.assert_array_equal(got["vertices"].data, want["vertices"].data)
