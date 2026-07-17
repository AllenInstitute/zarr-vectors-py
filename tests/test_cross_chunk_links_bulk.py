"""Tests for :func:`write_cross_chunk_links_bulk`, the numpy-native fast
path for ``cross_chunk_links`` writes restricted to
``link_width=2``/``directed=True``/``store="canonical"``/
``layout="packed_sharded"``/``mode="replace"``.

It must produce byte-for-byte-decodable-identical output to the general
``write_cross_chunk_links`` call with the equivalent list-of-tuples input
for that same policy -- these tests pin that equivalence, order
preservation within a cell, and the empty/singleton edge cases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.core.arrays import (
    read_cross_chunk_link_manifest,
    read_cross_chunk_links_for_tuple,
    write_cross_chunk_links,
    write_cross_chunk_links_bulk,
)
from zarr_vectors.core.store import create_store, get_resolution_level


def _new_lg(tmp_path: Path, name: str):
    root = create_store(
        str(tmp_path / name),
        bounds=([0.0, 0.0, 0.0], [10000.0, 10000.0, 10000.0]),
        chunk_shape=(100.0, 100.0, 100.0),
        geometry_types=["graph"],
        ndim=3,
    )
    return get_resolution_level(root, 0)


def _records_to_links(records: np.ndarray, sid_ndim: int) -> list:
    D = sid_ndim
    return [
        [
            (tuple(int(x) for x in r[:D]), int(r[D])),
            (tuple(int(x) for x in r[D + 1:2 * D + 1]), int(r[2 * D + 1])),
        ]
        for r in records
    ]


def _all_query_pairs(records: np.ndarray, sid_ndim: int) -> set:
    D = sid_ndim
    pairs = set()
    for r in records:
        a = tuple(int(x) for x in r[:D])
        b = tuple(int(x) for x in r[D + 1:2 * D + 1])
        pairs.add((a, b))
    return pairs


class TestEquivalenceToGeneralWriter:
    def test_random_records_byte_identical_readback(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        D = 3
        N = 5000
        chunks_a = rng.integers(0, 6, size=(N, D))
        chunks_b = rng.integers(0, 6, size=(N, D))
        vi_a = rng.integers(0, 100, size=N)
        vi_b = rng.integers(0, 100, size=N)
        records = np.concatenate(
            [chunks_a, vi_a[:, None], chunks_b, vi_b[:, None]], axis=1
        ).astype(np.int64)

        lg_general = _new_lg(tmp_path, "general.zv")
        lg_bulk = _new_lg(tmp_path, "bulk.zv")

        part_general = write_cross_chunk_links(
            lg_general, _records_to_links(records, D), sid_ndim=D, delta=0,
            mode="replace", directed=True, store="canonical",
            layout="packed_sharded",
        )
        part_bulk = write_cross_chunk_links_bulk(lg_bulk, records, sid_ndim=D, delta=0)

        assert part_general.num_links == part_bulk.num_links == N

        man_general = read_cross_chunk_link_manifest(lg_general, delta=0)
        man_bulk = read_cross_chunk_link_manifest(lg_bulk, delta=0)
        assert set(man_general["cell_keys"]) == set(man_bulk["cell_keys"])

        for a, b in _all_query_pairs(records, D):
            r_general = read_cross_chunk_links_for_tuple(lg_general, (a, b), delta=0)
            r_bulk = read_cross_chunk_links_for_tuple(lg_bulk, (a, b), delta=0)
            assert r_general == r_bulk

    def test_preserves_input_order_within_a_duplicate_cell(self, tmp_path: Path) -> None:
        # Three records land in the SAME cell (0,0,0)->(0,0,1), interleaved
        # with an unrelated cell -- order within the shared cell must match
        # input order, since callers may align parallel attribute arrays.
        records = np.array([
            [0, 0, 0, 1, 0, 0, 1, 2],
            [0, 0, 0, 3, 0, 0, 1, 4],
            [5, 5, 5, 9, 5, 5, 6, 8],
            [0, 0, 0, 5, 0, 0, 1, 6],
        ], dtype=np.int64)
        lg = _new_lg(tmp_path, "order.zv")
        part = write_cross_chunk_links_bulk(lg, records, sid_ndim=3, delta=0)

        result = read_cross_chunk_links_for_tuple(lg, ((0, 0, 0), (0, 0, 1)), delta=0)
        assert result == [
            (((0, 0, 0), 1), ((0, 0, 1), 2)),
            (((0, 0, 0), 3), ((0, 0, 1), 4)),
            (((0, 0, 0), 5), ((0, 0, 1), 6)),
        ]
        cell_key = [k for k in part.cell_indices if k.startswith("0.0.0.0.0.1")][0]
        assert part.cell_indices[cell_key] == [0, 1, 3]


class TestEdgeCases:
    def test_empty_input_is_noop(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path, "empty.zv")
        part = write_cross_chunk_links_bulk(
            lg, np.empty((0, 8), dtype=np.int64), sid_ndim=3, delta=0,
        )
        assert part.num_links == 0
        assert part.cell_indices == {}

    def test_single_record(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path, "single.zv")
        records = np.array([[1, 2, 3, 7, 1, 2, 4, 9]], dtype=np.int64)
        part = write_cross_chunk_links_bulk(lg, records, sid_ndim=3, delta=0)
        assert part.num_links == 1
        result = read_cross_chunk_links_for_tuple(lg, ((1, 2, 3), (1, 2, 4)), delta=0)
        assert result == [(((1, 2, 3), 7), ((1, 2, 4), 9))]

    def test_a_to_b_and_b_to_a_are_distinct_directed_cells(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path, "directed.zv")
        records = np.array([
            [0, 0, 0, 1, 1, 0, 0, 2],
            [1, 0, 0, 9, 0, 0, 0, 8],
        ], dtype=np.int64)
        write_cross_chunk_links_bulk(lg, records, sid_ndim=3, delta=0)
        ab = read_cross_chunk_links_for_tuple(lg, ((0, 0, 0), (1, 0, 0)), delta=0)
        ba = read_cross_chunk_links_for_tuple(lg, ((1, 0, 0), (0, 0, 0)), delta=0)
        assert ab == [(((0, 0, 0), 1), ((1, 0, 0), 2))]
        assert ba == [(((1, 0, 0), 9), ((0, 0, 0), 8))]
