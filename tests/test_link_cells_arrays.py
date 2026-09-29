"""``write_link_cells(chunks=, vids=, attributes=)``: the store the list form leaves.

The list form plus one ``write_link_attribute_cells`` per attribute is the
definition. The array form must leave the same store -- links, their
fragment sidecar, every attribute -- over repeated batches into the same
cells, which is how a per-chunk worker writes seams.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._store_compare import assert_stores_identical
from zarr_vectors.building import (
    create_links_array,
    create_store,
    finalize_links,
    get_resolution_level,
    read_link_arrays,
    write_link_attribute_cells,
    write_link_cells,
)
from zarr_vectors.core.paths import intra_offsets


def _level(tmp_path, name):
    (tmp_path / name).mkdir()
    path = tmp_path / name / "s.zarrvectors"
    root = create_store(
        path, bounds=([0.0] * 3, [200.0] * 3), chunk_shape=(50.0,) * 3,
    )
    return path, get_resolution_level(root, 0)


def _batch(rng, n, L, *, intra_share=0.4):
    """Records on a 4^3 grid: some inside one chunk, the rest neighbours."""
    base = rng.integers(0, 4, (n, 1, 3))
    step = rng.integers(-1, 2, (n, L, 3))
    step[rng.random(n) < intra_share] = 0
    chunks = np.clip(base + step, 0, 3).astype(np.int64)
    vids = rng.integers(0, 60, (n, L)).astype(np.int64)
    attrs = {
        "w": rng.uniform(0, 1, n).astype(np.float32),
        "kind": rng.integers(0, 5, (n, 2)).astype(np.int16),
    }
    return chunks, vids, attrs


def _as_records(chunks, vids):
    return [
        [(tuple(chunks[r, k].tolist()), int(vids[r, k])) for k in range(chunks.shape[1])]
        for r in range(chunks.shape[0])
    ]


@pytest.mark.parametrize("L,directed,store,int32", [
    (2, False, "canonical", False),   # BRIDGE seams
    (2, False, "canonical", True),    # an int32-stamped family
    (3, False, "canonical", False),
    (2, True, "canonical", False),
    (2, False, "duplicate", False),   # the record-placement fallback
])
def test_arrays_leave_the_store_records_do(tmp_path, L, directed, store, int32):
    path_a, lg_a = _level(tmp_path, "records")
    path_b, lg_b = _level(tmp_path, "arrays")
    if int32:
        for lg in (lg_a, lg_b):
            create_links_array(lg, L, dtype="int32", delta=0, sid_ndim=3,
                               offsets=intra_offsets(3, L))
    rng = np.random.default_rng(L + 10 * int(directed))
    for _ in range(4):  # repeated batches land in the same cells
        chunks, vids, attrs = _batch(rng, int(rng.integers(1, 60)), L)
        part = write_link_cells(
            lg_a, _as_records(chunks, vids), 3, directed=directed, store=store,
        )
        for name, values in attrs.items():
            write_link_attribute_cells(lg_a, name, values, partition=part)
        write_link_cells(
            lg_b, chunks=chunks, vids=vids, attributes=attrs,
            directed=directed, store=store,
        )
    for lg in (lg_a, lg_b):
        finalize_links(lg, delta=0)
    assert_stores_identical(path_a, path_b)


def test_the_partition_matches_the_list_forms(tmp_path):
    _, lg_a = _level(tmp_path, "a")
    _, lg_b = _level(tmp_path, "b")
    chunks, vids, _ = _batch(np.random.default_rng(3), 50, 2)
    want = write_link_cells(lg_a, _as_records(chunks, vids), 3)
    got = write_link_cells(lg_b, chunks=chunks, vids=vids)
    assert set(got.cell_indices) == set(want.cell_indices)
    for key, idx in want.cell_indices.items():
        assert list(got.cell_indices[key]) == list(idx)
    assert got.num_physical_records == want.num_physical_records


def test_one_prefetch_serves_links_sidecar_and_attributes(tmp_path, monkeypatch):
    from zarr_vectors.core import _batch_reader

    _, lg = _level(tmp_path, "p")
    chunks, vids, attrs = _batch(np.random.default_rng(4), 40, 2)
    calls = []
    real = _batch_reader.flush_prefetch

    def _count(*args, **kw):
        calls.append(len(args[1]))
        return real(*args, **kw)

    monkeypatch.setattr(_batch_reader, "flush_prefetch", _count)
    write_link_cells(lg, chunks=chunks, vids=vids, attributes=attrs)
    assert len(calls) == 1 and calls[0] > 3


def test_device_arrays_are_copied_off_once(tmp_path):
    from tests._fake_device import FakeDeviceArray
    from zarr_vectors import _xp

    _, lg = _level(tmp_path, "d")
    chunks, vids, attrs = _batch(np.random.default_rng(5), 30, 2)
    with _xp.count_transfers() as stats:
        write_link_cells(
            lg, chunks=FakeDeviceArray(chunks), vids=FakeDeviceArray(vids),
            attributes={k: FakeDeviceArray(v) for k, v in attrs.items()},
        )
    assert stats.d2h_calls == 2 + len(attrs)
    finalize_links(lg, delta=0)
    assert read_link_arrays(lg)[1].shape == (30, 2)


def test_both_forms_at_once_is_an_error(tmp_path):
    from zarr_vectors.exceptions import ArrayError

    _, lg = _level(tmp_path, "e")
    with pytest.raises(ArrayError, match="not both"):
        write_link_cells(lg, [[((0, 0, 0), 1), ((0, 0, 0), 2)]], 3,
                         chunks=np.zeros((1, 2, 3)), vids=np.zeros((1, 2)))


class TestAStampedDtypeSkipsTheListing:
    """A segment already stamped with the incoming dtype is not listed.

    "Does this segment hold data?" decides between the stamped dtype and
    the data's, so when the two are equal the answer cannot matter. In a
    deferred level the listing that asks it reads every shard, once per
    attribute segment per seam write.
    """

    @staticmethod
    def _listings(monkeypatch) -> list[str]:
        from zarr_vectors.core.group import Group

        asked: list[str] = []
        real = Group.list_chunks

        def spy(self, name, *a, **kw):
            asked.append(name)
            return real(self, name, *a, **kw)

        monkeypatch.setattr(Group, "list_chunks", spy)
        return asked

    @staticmethod
    def _one_intra_record(cell, vids):
        return (
            np.array([[cell, cell]], dtype=np.int64),
            np.array([vids], dtype=np.int64),
        )

    def test_the_array_form_does_not_list_a_matching_segment(self, tmp_path, monkeypatch):
        w = {"w": np.array([0.5], dtype=np.float32)}
        first = self._one_intra_record((0, 0, 0), (1, 2))
        # A cell of the same segment that holds nothing yet.
        second = self._one_intra_record((1, 1, 1), (3, 4))

        ref_path, ref = _level(tmp_path, "ref")
        for chunks, vids in (first, second):
            write_link_cells(ref, chunks=chunks, vids=vids, attributes=w)

        path, lg = _level(tmp_path, "a")
        write_link_cells(lg, chunks=first[0], vids=first[1], attributes=w)
        asked = self._listings(monkeypatch)
        write_link_cells(lg, chunks=second[0], vids=second[1], attributes=w)
        assert not [n for n in asked if n.startswith("link_attributes/")]
        assert_stores_identical(ref_path, path)

    def test_the_list_form_does_not_list_a_matching_segment(self, tmp_path, monkeypatch):
        _, lg = _level(tmp_path, "l")
        create_links_array(lg, link_width=2, delta=0, sid_ndim=3)
        p1 = write_link_cells(lg, [[((0, 0, 0), 1), ((0, 0, 0), 2)]], sid_ndim=3)
        p2 = write_link_cells(lg, [[((1, 1, 1), 3), ((1, 1, 1), 4)]], sid_ndim=3)
        write_link_attribute_cells(lg, "w", np.array([0.1], dtype=np.float32), partition=p1)
        asked = self._listings(monkeypatch)
        write_link_attribute_cells(lg, "w", np.array([0.2], dtype=np.float32), partition=p2)
        assert not [n for n in asked if n.startswith("link_attributes/")]

    def test_a_placeholder_of_another_dtype_still_asks_and_yields(self, tmp_path, monkeypatch):
        from zarr_vectors.building import create_link_attributes_array

        _, lg = _level(tmp_path, "p")
        create_links_array(lg, link_width=2, delta=0, sid_ndim=3)
        p = write_link_cells(lg, [[((0, 0, 0), 1), ((0, 0, 0), 2)]], sid_ndim=3)
        # A pre-created, empty segment stamped float64: the data's float32 wins.
        create_link_attributes_array(
            lg, "w", "float64", delta=0, sid_ndim=3, offsets=intra_offsets(3, 2),
        )
        asked = self._listings(monkeypatch)
        write_link_attribute_cells(lg, "w", np.array([0.25], dtype=np.float32), partition=p)
        assert [n for n in asked if n.startswith("link_attributes/")]
        meta = lg.read_array_meta(asked[-1])
        assert np.dtype(meta["dtype"]) == np.float32
