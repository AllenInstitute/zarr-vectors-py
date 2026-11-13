"""Resolving object ids to rows without holding (or sorting) the id table.

The lookup used to read all of ``object_index/object_ids`` and, unless
the sorted stamp said otherwise, argsort it: 8 + 8 bytes a row, and
``reserve_object_rows`` stamps unsorted. An identity table -- every
dense or appended index whose ids are its rows -- is now proved so
without being held: from two ids under the stamp, else read in pieces.
Answers are checked against the old algorithm on identity, sorted,
permuted, reserved and residue-bearing tables; the commit's id check,
now read in pieces too, against whole-table arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors import building as zb
from zarr_vectors.constants import OBJECT_INDEX
from zarr_vectors.core import arrays as A
from zarr_vectors.exceptions import ArrayError

pytestmark = pytest.mark.vlen_only  # layouts chosen here


def _level(tmp_path, layout, name="s.zv"):
    root = tmp_path / layout / name
    root.parent.mkdir(parents=True, exist_ok=True)
    zb.create_store(
        str(root), bounds=([0.0] * 3, [100.0] * 3), chunk_shape=(50.0,) * 3,
        geometry_types=["polyline"], manifest_layout=layout,
    )
    return zb.get_resolution_level(zb.open_store(str(root), mode="r+"), 0)


def _reference(level, wanted):
    """The lookup as it was: the whole table, argsorted unless stamped."""
    table = A.read_object_id_table(level)
    wanted = np.asarray(wanted, dtype=np.int64)
    if table is None:
        n = A.object_row_count(level)
        keep = (wanted >= 0) & (wanted < n)
        return wanted[keep], wanted[keep]
    meta = level.read_array_meta(OBJECT_INDEX)
    if meta.get(A.OBJECT_IDS_SORTED_ATTR):
        sorted_ids, rows = table, np.arange(table.size)
    else:
        order = np.argsort(table, kind="stable")
        sorted_ids, rows = table[order], order
    pos = np.clip(np.searchsorted(sorted_ids, wanted), 0, max(sorted_ids.size - 1, 0))
    found = sorted_ids[pos] == wanted if sorted_ids.size else np.zeros_like(wanted, bool)
    return wanted[found], rows[pos[found]]


def _append(level, n, at, ids=None):
    zb.write_object_manifests(
        level, chunk_coords=np.zeros((n, 3), np.int64), fragment_idx=np.arange(n),
        mode="append", at=at, ids=ids,
    )


def _check(level, *, expect):
    level._object_id_lookup_cache = None
    rng = np.random.default_rng(0)
    wanted = np.concatenate([
        rng.integers(-5, 3 * 10**6, 300), np.arange(-2, 40), [10**9, 10**9 + 3],
    ])
    got = A.object_rows_for_ids(level, wanted)
    want = _reference(level, wanted)
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])
    kind = A._object_id_lookup(level)
    assert (kind[0] if kind else None) == expect


@pytest.mark.parametrize("layout", ["vlen", "dense"])
def test_tables_of_every_shape_answer_as_before(tmp_path, layout, monkeypatch):
    monkeypatch.setattr(A, "_ID_SCAN_ROWS", 1_000)  # many pieces

    # An identity table, stamped sorted.
    lg = _level(tmp_path, layout, "identity.zv")
    A.write_object_index(lg, {o: [((0, 0, 0), o)] for o in range(5_000)}, 3)
    _check(lg, expect="identity")
    # The same stamped unsorted, as reserve_object_rows leaves it.
    lg.write_array_meta(OBJECT_INDEX, {A.OBJECT_IDS_SORTED_ATTR: False})
    _check(lg, expect="identity")

    # Rows past the committed count that are not the identity: residue
    # with other ids, which a commit's stamp does not speak for.
    lg = _level(tmp_path, layout, "residue.zv")
    A.write_object_index(lg, {o: [((0, 0, 0), o)] for o in range(3_000)}, 3)
    _append(lg, 10, 3_000, ids=np.arange(10) + 10**9)
    lg.write_array_meta(OBJECT_INDEX, {A.OBJECT_IDS_SORTED_ATTR: True})
    _check(lg, expect="sorted")

    # Sparse ascending ids: sorted, not the identity.
    lg = _level(tmp_path, layout, "sparse.zv")
    A.write_object_index(lg, {3 * o + 1: [((0, 0, 0), o)] for o in range(2_000)}, 3)
    _check(lg, expect="sorted")

    # Ids out of order: sorted once, as before.
    lg = _level(tmp_path, layout, "permuted.zv")
    A.write_object_index(lg, {o: [((0, 0, 0), o)] for o in range(100)}, 3)
    _append(lg, 50, 100, ids=np.arange(50)[::-1] + 500)
    _check(lg, expect="permuted")

    # Reserved rows nobody wrote (-1) past the commit.
    lg = _level(tmp_path, layout, "reserved.zv")
    A.write_object_index(lg, {o: [((0, 0, 0), o)] for o in range(1_500)}, 3)
    zb.reserve_object_rows(lg, 4_000, sid_ndim=3, n_blocks=8_000 if layout == "dense" else None)
    _check(lg, expect="permuted")


def test_an_identity_table_is_never_read_whole(tmp_path, monkeypatch):
    lg = _level(tmp_path, "dense")
    _append(lg, 70_000, 0)
    zb.commit_object_index(lg, 70_000, sid_ndim=3)

    def refuse(_level):
        raise AssertionError("read the whole id table")

    monkeypatch.setattr(A, "read_object_id_table", refuse)
    found, rows = A.object_rows_for_ids(lg, [5, 69_999, 70_000, -1])
    assert found.tolist() == rows.tolist() == [5, 69_999]
    # Unstamped: read in pieces, none longer than the scan.
    lg.write_array_meta(OBJECT_INDEX, {A.OBJECT_IDS_SORTED_ATTR: False})
    lg._object_id_lookup_cache = None
    monkeypatch.setattr(A, "_ID_SCAN_ROWS", 4_096)
    sizes = []
    real = A._id_table_pieces

    def spy(node, lo, hi):
        for a, piece in real(node, lo, hi):
            sizes.append(piece.size)
            yield a, piece

    monkeypatch.setattr(A, "_id_table_pieces", spy)
    found, rows = A.object_rows_for_ids(lg, [0, 12_345])
    assert rows.tolist() == [0, 12_345]
    assert sizes and max(sizes) <= 4_096 and sum(sizes) == 70_000


@pytest.mark.parametrize("layout", ["vlen", "dense"])
def test_ids_for_rows_on_an_identity_table(tmp_path, layout):
    lg = _level(tmp_path, layout)
    A.write_object_index(lg, {o: [((0, 0, 0), o)] for o in range(300)}, 3)
    table = A.read_object_id_table(lg)
    rows = [0, 299, -1, -300, 17]
    assert A.object_ids_for_rows(lg, rows).tolist() == table[rows].tolist()
    for bad in (300, -301):
        with pytest.raises(IndexError):
            table[[bad]]
        with pytest.raises(IndexError):
            A.object_ids_for_rows(lg, [bad])
    assert A.object_ids_for_rows(lg).tolist() == table.tolist()


@pytest.mark.parametrize(
    ("ids", "sorted_", "unwritten"),
    [
        (np.arange(3_000), True, 0),
        (np.r_[np.arange(1_000), np.arange(999, 3_000)], False, 0),  # tie at a piece edge
        (np.r_[np.arange(1_000), np.arange(1_000, 3_000)[::-1]], False, 0),
        (np.r_[np.arange(1_500), -np.ones(1_500, np.int64)], None, 1_500),
    ],
)
def test_the_commit_checks_ids_in_pieces(tmp_path, monkeypatch, ids, sorted_, unwritten):
    monkeypatch.setattr(A, "_ID_SCAN_ROWS", 1_000)
    lg = _level(tmp_path, "dense")
    n = ids.size
    zb.reserve_object_rows(lg, n, sid_ndim=3, n_blocks=n)
    lg.zarr_group[f"{OBJECT_INDEX}/object_ids"][:] = ids
    lg.zarr_group["object_index/manifest_spans"][:] = np.c_[np.arange(n), np.ones(n, np.int64)]
    if unwritten:
        with pytest.raises(ArrayError, match=f"{unwritten} of the {n} rows have no object id"):
            zb.commit_object_index(lg, n, sid_ndim=3)
        return
    out = zb.commit_object_index(lg, n, sid_ndim=3)
    assert out[A.OBJECT_IDS_SORTED_ATTR] is sorted_
    assert sorted_ == bool(np.all(np.diff(ids) > 0))
