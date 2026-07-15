"""Tests for the decentralized (per-cell) cross-chunk-link writers.

``write_cross_chunk_link_cells`` lets independent workers each append a
batch of records into only the cells they touch; ``finalize_cross_chunk_links``
reconciles the family-wide counts afterward.  The core guarantee is that a
sequence of disjoint per-cell writes + finalize is equivalent to a single
whole-family ``write_cross_chunk_links``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.core.arrays import (
    create_cross_chunk_link_attributes_array,
    create_cross_chunk_links_array,
    finalize_cross_chunk_links,
    read_cross_chunk_link_attributes,
    read_cross_chunk_links,
    write_cross_chunk_link_attribute_cells,
    write_cross_chunk_link_cells,
    write_cross_chunk_links,
)
from zarr_vectors.core.paths import cross_chunk_links_path
from zarr_vectors.core.store import create_store, get_resolution_level
from zarr_vectors.exceptions import ArrayError


def _new_lg(tmp_path: Path, name: str = "store.zv"):
    root = create_store(
        str(tmp_path / name),
        bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
        chunk_shape=(100.0, 100.0, 100.0),
        geometry_types=["graph"],
        ndim=3,
    )
    return get_resolution_level(root, 0)


def _as_set(records):
    return {tuple(r) for r in records}


class TestEquivalenceToWholeWrite:
    """Disjoint per-cell writes + finalize == one whole-family write."""

    def test_canonical(self, tmp_path: Path) -> None:
        all_records = [
            [((0, 0, 0), 1), ((1, 0, 0), 2)],   # cell {0,1}
            [((0, 0, 0), 3), ((1, 0, 0), 4)],   # cell {0,1}
            [((5, 0, 0), 5), ((6, 0, 0), 6)],   # cell {5,6}  (disjoint)
        ]
        # Whole-family reference.
        lg_whole = _new_lg(tmp_path, "whole.zv")
        write_cross_chunk_links(lg_whole, all_records, sid_ndim=3, delta=0)
        ref = _as_set(read_cross_chunk_links(lg_whole, delta=0))
        ref_meta = lg_whole.read_array_meta(cross_chunk_links_path(0))

        # Two workers over disjoint cells.
        lg = _new_lg(tmp_path, "cells.zv")
        create_cross_chunk_links_array(lg, delta=0, sid_ndim=3)
        write_cross_chunk_link_cells(lg, all_records[:2], sid_ndim=3, delta=0)
        write_cross_chunk_link_cells(lg, all_records[2:], sid_ndim=3, delta=0)
        finalize_cross_chunk_links(lg, delta=0)

        assert _as_set(read_cross_chunk_links(lg, delta=0)) == ref
        meta = lg.read_array_meta(cross_chunk_links_path(0))
        assert meta["num_links"] == ref_meta["num_links"] == 3
        assert meta["num_physical_records"] == ref_meta["num_physical_records"] == 3

    def test_duplicate(self, tmp_path: Path) -> None:
        all_records = [
            [((0, 0, 0), 1), ((1, 0, 0), 2)],
            [((5, 0, 0), 5), ((6, 0, 0), 6)],
        ]
        lg_whole = _new_lg(tmp_path, "whole.zv")
        write_cross_chunk_links(
            lg_whole, all_records, sid_ndim=3, delta=0, store="duplicate",
        )
        ref = _as_set(read_cross_chunk_links(lg_whole, delta=0))
        ref_meta = lg_whole.read_array_meta(cross_chunk_links_path(0))

        lg = _new_lg(tmp_path, "cells.zv")
        create_cross_chunk_links_array(
            lg, delta=0, sid_ndim=3, store="duplicate",
        )
        write_cross_chunk_link_cells(
            lg, all_records[:1], sid_ndim=3, delta=0, store="duplicate",
        )
        write_cross_chunk_link_cells(
            lg, all_records[1:], sid_ndim=3, delta=0, store="duplicate",
        )
        finalize_cross_chunk_links(lg, delta=0)

        assert _as_set(read_cross_chunk_links(lg, delta=0)) == ref
        meta = lg.read_array_meta(cross_chunk_links_path(0))
        assert meta["num_links"] == ref_meta["num_links"] == 2
        assert (
            meta["num_physical_records"]
            == ref_meta["num_physical_records"]
            == 4
        )

    def test_directed(self, tmp_path: Path) -> None:
        recs = [
            [((1, 0, 0), 9), ((0, 0, 0), 8)],   # cell 1.0.0.0.0.0
            [((5, 0, 0), 1), ((6, 0, 0), 2)],   # cell 5.0.0.6.0.0
        ]
        lg = _new_lg(tmp_path, "cells.zv")
        create_cross_chunk_links_array(lg, delta=0, sid_ndim=3, directed=True)
        write_cross_chunk_link_cells(
            lg, recs[:1], sid_ndim=3, delta=0, directed=True,
        )
        write_cross_chunk_link_cells(
            lg, recs[1:], sid_ndim=3, delta=0, directed=True,
        )
        finalize_cross_chunk_links(lg, delta=0)
        out = _as_set(read_cross_chunk_links(lg, delta=0))
        assert out == {
            (((1, 0, 0), 9), ((0, 0, 0), 8)),
            (((5, 0, 0), 1), ((6, 0, 0), 2)),
        }
        meta = lg.read_array_meta(cross_chunk_links_path(0))
        assert meta["directed"] is True
        assert meta["num_links"] == 2


class TestAttributeCells:
    """Per-cell attribute writes stay row-aligned with per-cell link writes."""

    def test_round_trip(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        create_cross_chunk_links_array(lg, delta=0, sid_ndim=3)
        create_cross_chunk_link_attributes_array(lg, "weight", delta=0)

        batch1 = [[((0, 0, 0), 1), ((1, 0, 0), 2)]]
        batch2 = [[((5, 0, 0), 5), ((6, 0, 0), 6)]]
        p1 = write_cross_chunk_link_cells(lg, batch1, sid_ndim=3, delta=0)
        write_cross_chunk_link_attribute_cells(
            lg, "weight", np.array([0.1], dtype=np.float32),
            partition=p1, delta=0,
        )
        p2 = write_cross_chunk_link_cells(lg, batch2, sid_ndim=3, delta=0)
        write_cross_chunk_link_attribute_cells(
            lg, "weight", np.array([0.9], dtype=np.float32),
            partition=p2, delta=0,
        )
        finalize_cross_chunk_links(lg, delta=0)

        links = read_cross_chunk_links(lg, delta=0)
        attrs = read_cross_chunk_link_attributes(lg, "weight", delta=0)
        by_rec = dict(zip((tuple(r) for r in links), attrs))
        assert np.isclose(by_rec[(((0, 0, 0), 1), ((1, 0, 0), 2))], 0.1)
        assert np.isclose(by_rec[(((5, 0, 0), 5), ((6, 0, 0), 6))], 0.9)

    def test_duplicate_attributes_replicate(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        create_cross_chunk_links_array(
            lg, delta=0, sid_ndim=3, store="duplicate",
        )
        create_cross_chunk_link_attributes_array(lg, "weight", delta=0)
        batch = [[((0, 0, 0), 1), ((4, 4, 4), 9)]]
        p = write_cross_chunk_link_cells(
            lg, batch, sid_ndim=3, delta=0, store="duplicate",
        )
        write_cross_chunk_link_attribute_cells(
            lg, "weight", np.array([0.5], dtype=np.float32),
            partition=p, delta=0,
        )
        finalize_cross_chunk_links(lg, delta=0)
        links = read_cross_chunk_links(lg, delta=0)
        attrs = read_cross_chunk_link_attributes(lg, "weight", delta=0)
        assert len(links) == len(attrs) == 2
        assert all(np.isclose(a, 0.5) for a in attrs)


class TestPolicyGuards:
    def test_store_mismatch_raises(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        create_cross_chunk_links_array(
            lg, delta=0, sid_ndim=3, store="canonical",
        )
        with pytest.raises(ArrayError, match="store"):
            write_cross_chunk_link_cells(
                lg, [[((0, 0, 0), 1), ((1, 0, 0), 2)]],
                sid_ndim=3, delta=0, store="duplicate",
            )

    def test_directed_mismatch_raises(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        create_cross_chunk_links_array(lg, delta=0, sid_ndim=3, directed=True)
        with pytest.raises(ArrayError, match="directed"):
            write_cross_chunk_link_cells(
                lg, [[((0, 0, 0), 1), ((1, 0, 0), 2)]],
                sid_ndim=3, delta=0, directed=False,
            )


class TestFinalizeThenShard:
    """After finalize, a shard pass must preserve the records."""

    def test_shard_round_trip(self, tmp_path: Path) -> None:
        from zarr_vectors.core.store import open_store
        from zarr_vectors.sharding import shard_store

        store_path = tmp_path / "cells.zv"
        root = create_store(
            str(store_path),
            bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
            chunk_shape=(100.0, 100.0, 100.0),
            geometry_types=["graph"],
            ndim=3,
        )
        lg = get_resolution_level(root, 0)
        create_cross_chunk_links_array(lg, delta=0, sid_ndim=3)
        recs = [
            [((0, 0, 0), 1), ((1, 0, 0), 2)],
            [((5, 0, 0), 5), ((6, 0, 0), 6)],
        ]
        write_cross_chunk_link_cells(lg, recs, sid_ndim=3, delta=0)
        finalize_cross_chunk_links(lg, delta=0)
        before = _as_set(read_cross_chunk_links(lg, delta=0))

        shard_store(str(store_path))

        # Re-open after the (destructive) shard pass and re-read.
        lg2 = get_resolution_level(open_store(str(store_path), mode="r"), 0)
        after = _as_set(read_cross_chunk_links(lg2, delta=0))
        assert after == before
