"""``dense_manifests.read_csr`` reads what it read before, faster.

The 100 um export's chain read spent ~17 s here: two ``np.unique`` calls,
a ``unique``/``searchsorted`` for block rows that were already ascending,
and 147.6M rows fetched through zarr's orthogonal indexer when they were
one contiguous run. The reference below is the code as it was; every
input -- all rows, a prefix, ascending, gapped, unsorted, duplicated,
empty, after a patch has scattered the blocks -- must give the same
arrays, and ascending ids must not reach the orthogonal indexer at all.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from zarr_vectors.building import (
    create_store,
    get_resolution_level,
    read_object_manifests_csr,
)
from zarr_vectors.core import dense_manifests as dense
from zarr_vectors.core.arrays import patch_object_manifests, write_object_index

pytestmark = pytest.mark.vlen_only  # dense by construction
N = 600
BUCKET = 32  # several zarr chunks per array, so runs cross chunk edges


def _reference_rows(level_group, path, rows):
    node = dense._node(level_group, path)
    if isinstance(rows, slice):
        return np.asarray(node[rows])
    if rows.size == 0:
        return np.empty((0, *node.shape[1:]), dtype=node.dtype)
    return np.asarray(node.get_orthogonal_selection((rows,)))


def _reference_read_csr(level_group, rows=None, *, stop=None):
    """read_csr as of 218c633."""
    sel = slice(0, stop) if rows is None else np.asarray(rows, dtype=np.int64)
    spans = _reference_rows(level_group, dense.SPANS_PATH, sel).astype(np.int64).reshape(-1, 2)
    offsets, block_rows = dense._gather(spans)
    n_blocks = dense.num_blocks(level_group)
    contiguous = (
        rows is None and block_rows.size == n_blocks
        and np.array_equal(block_rows, np.arange(n_blocks))
    )
    blocks = (
        _reference_rows(level_group, dense.BLOCKS_PATH, slice(None)) if contiguous
        else _reference_rows(level_group, dense.BLOCKS_PATH, np.unique(block_rows))
    )
    blocks = np.asarray(blocks, dtype=np.int64)
    if not contiguous and block_rows.size:
        blocks = blocks[np.searchsorted(np.unique(block_rows), block_rows)]
    if blocks.size == 0:
        sid = dense._sid(level_group)
        return offsets, np.empty((0, sid), np.int64), np.empty(0, np.int64)
    return offsets, blocks[:, :-1], blocks[:, -1]


def _manifest(rng):
    k = 0 if rng.random() < 0.2 else int(rng.integers(1, 5))
    return [
        (tuple(int(c) for c in rng.integers(0, 3, 3)), int(rng.integers(0, 2**40)))
        for _ in range(k)
    ]


@pytest.fixture(params=["written", "patched"])
def level(request, tmp_path, monkeypatch):
    monkeypatch.setattr(dense, "_BUCKET", BUCKET)
    rng = np.random.default_rng(7)
    root = create_store(
        tmp_path / "s.zarrvectors", bounds=([0.0] * 3, [150.0] * 3),
        chunk_shape=(50.0,) * 3, geometry_types=["polyline"], manifest_layout="dense",
    )
    lg = get_resolution_level(root, 0)
    write_object_index(lg, {o: _manifest(rng) for o in range(N)}, 3)
    if request.param == "patched":
        # Patched objects own blocks appended at the end, so ascending
        # object rows no longer own ascending block rows.
        ids = rng.choice(N, 40, replace=False)
        patch_object_manifests(lg, {int(o): _manifest(rng) for o in ids}, 3)
    return lg


def _same(got, want):
    assert len(got) == len(want) == 3
    for g, w in zip(got, want):
        assert g.dtype == w.dtype and g.shape == w.shape
        np.testing.assert_array_equal(g, w)


def _inputs(rng):
    asc = np.arange(N, dtype=np.int64)
    return {
        "all-ascending": asc,
        "run": asc[100:400],
        "gapped": np.concatenate([asc[:50], asc[120:300], asc[301:302], asc[500:]]),
        "scattered": np.sort(rng.choice(N, 150, replace=False)),
        "unsorted": rng.permutation(N)[:300],
        "duplicates": rng.integers(0, N, 400),
        "descending": asc[::-1].copy(),
        "single": asc[17:18],
        "empty": asc[:0],
    }


@pytest.mark.parametrize("path", ["default", "slices", "orthogonal"])
def test_every_input_reads_what_the_reference_reads(level, monkeypatch, path):
    if path == "slices":
        monkeypatch.setattr(dense, "_SLICE_RUNS_MAX", 10**9)
    elif path == "orthogonal":
        monkeypatch.setattr(dense, "_SLICE_RUNS_MAX", 0)  # unless runs >= a chunk
    _same(dense.read_csr(level), _reference_read_csr(level))
    _same(dense.read_csr(level, stop=250), _reference_read_csr(level, stop=250))
    _same(dense.read_csr(level, stop=0), _reference_read_csr(level, stop=0))
    for name, rows in _inputs(np.random.default_rng(3)).items():
        try:
            _same(dense.read_csr(level, rows), _reference_read_csr(level, rows))
        except AssertionError as exc:
            raise AssertionError(f"{name}: {exc}") from None


def test_rows_of_an_array_match_zarrs_indexer(level, monkeypatch):
    monkeypatch.setattr(dense, "_SLICE_RUNS_MAX", 10**9)
    for rows in _inputs(np.random.default_rng(5)).values():
        for path in (dense.SPANS_PATH, dense.BLOCKS_PATH):
            n = dense._node(level, path).shape[0]
            rows = rows[rows < n]
            np.testing.assert_array_equal(
                dense._rows(level, path, rows), _reference_rows(level, path, rows),
            )


def test_out_of_bounds_rows_still_raise(level):
    with pytest.raises(IndexError):
        dense._rows(level, dense.SPANS_PATH, np.array([0, N + 5], dtype=np.int64))


def test_ascending_ids_never_reach_the_orthogonal_indexer(tmp_path, monkeypatch):
    monkeypatch.setattr(dense, "_BUCKET", BUCKET)
    rng = np.random.default_rng(11)
    root = create_store(
        tmp_path / "s.zarrvectors", bounds=([0.0] * 3, [150.0] * 3),
        chunk_shape=(50.0,) * 3, geometry_types=["polyline"], manifest_layout="dense",
    )
    lg = get_resolution_level(root, 0)
    write_object_index(lg, {o: _manifest(rng) for o in range(N)}, 3)
    ids = np.concatenate([np.arange(0, 200), np.arange(260, N)])
    want = read_object_manifests_csr(lg, ids.tolist())

    def refuse(*a, **kw):
        raise AssertionError("ascending ids went through get_orthogonal_selection")

    monkeypatch.setattr(zarr.Array, "get_orthogonal_selection", refuse)
    monkeypatch.setattr(np, "unique", refuse)
    got = read_object_manifests_csr(lg, ids)
    np.testing.assert_array_equal(got.object_ids, want.object_ids)
    np.testing.assert_array_equal(got.offsets, want.offsets)
    np.testing.assert_array_equal(got.chunk_coords, want.chunk_coords)
    np.testing.assert_array_equal(got.fragment_idx, want.fragment_idx)
