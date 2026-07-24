"""Tests for the decentralized (per-cell) link writers.

``write_link_cells`` lets independent workers each append a batch of
records into only the cells they touch; ``finalize_links`` reconciles the
family-wide counts afterward.  The core guarantee is that a sequence of
disjoint per-cell writes + finalize is equivalent to a single
whole-family ``write_links``.

Placement routes through the same choke point for both writers, so a
batch lands in exactly the cells the whole-family writer would pick —
including the all-zero (intra-chunk) offsets array, which is no longer a
separate family.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.core.arrays import (
    create_link_attributes_array,
    create_links_array,
    finalize_links,
    list_link_offsets,
    read_link_attributes,
    read_links,
    write_link_attribute_cells,
    write_link_cells,
    write_links,
)
from zarr_vectors.core.paths import links_group_path, links_path
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
            [((0, 0, 0), 1), ((1, 0, 0), 2)],   # cell (0,0,0) @ "+1.0.0"
            [((0, 0, 0), 3), ((1, 0, 0), 4)],   # cell (0,0,0) @ "+1.0.0"
            [((5, 0, 0), 5), ((6, 0, 0), 6)],   # cell (5,0,0) @ "+1.0.0"
        ]
        # Whole-family reference.
        lg_whole = _new_lg(tmp_path, "whole.zv")
        write_links(lg_whole, all_records, sid_ndim=3, delta=0)
        ref = _as_set(read_links(lg_whole, delta=0))
        ref_meta = lg_whole.read_array_meta(links_group_path(0))

        # Two workers over disjoint cells.
        lg = _new_lg(tmp_path, "cells.zv")
        create_links_array(lg, link_width=2, delta=0, sid_ndim=3)
        write_link_cells(lg, all_records[:2], sid_ndim=3, delta=0)
        write_link_cells(lg, all_records[2:], sid_ndim=3, delta=0)
        finalize_links(lg, delta=0)

        assert _as_set(read_links(lg, delta=0)) == ref
        meta = lg.read_array_meta(links_group_path(0))
        assert meta["num_links"] == ref_meta["num_links"] == 3
        assert meta["num_physical_records"] == ref_meta["num_physical_records"] == 3

    def test_canonical_with_intra_records(self, tmp_path: Path) -> None:
        # Intra-chunk records are the all-zero offsets array, so they go
        # through the same per-cell writer — the equivalence has to hold
        # across a batch that mixes them with cross-chunk records.
        all_records = [
            [((0, 0, 0), 1), ((1, 0, 0), 2)],   # cross → "+1.0.0"
            [((0, 0, 0), 7), ((0, 0, 0), 8)],   # intra → "0.0.0"
            [((5, 0, 0), 5), ((5, 0, 0), 6)],   # intra → "0.0.0"
        ]
        lg_whole = _new_lg(tmp_path, "whole.zv")
        write_links(lg_whole, all_records, sid_ndim=3, delta=0)
        ref = _as_set(read_links(lg_whole, delta=0))
        ref_meta = lg_whole.read_array_meta(links_group_path(0))

        lg = _new_lg(tmp_path, "cells.zv")
        create_links_array(lg, link_width=2, delta=0, sid_ndim=3)
        write_link_cells(lg, all_records[:2], sid_ndim=3, delta=0)
        write_link_cells(lg, all_records[2:], sid_ndim=3, delta=0)
        finalize_links(lg, delta=0)

        assert _as_set(read_links(lg, delta=0)) == ref
        assert list_link_offsets(lg, 0) == \
            list_link_offsets(lg_whole, 0) == ["+1.0.0", "0.0.0"]
        meta = lg.read_array_meta(links_group_path(0))
        assert meta["num_links"] == ref_meta["num_links"] == 3
        assert meta["num_physical_records"] == ref_meta["num_physical_records"] == 3

    def test_duplicate(self, tmp_path: Path) -> None:
        all_records = [
            [((0, 0, 0), 1), ((1, 0, 0), 2)],
            [((5, 0, 0), 5), ((6, 0, 0), 6)],
        ]
        lg_whole = _new_lg(tmp_path, "whole.zv")
        write_links(
            lg_whole, all_records, sid_ndim=3, delta=0, store="duplicate",
        )
        ref = _as_set(read_links(lg_whole, delta=0))
        ref_meta = lg_whole.read_array_meta(links_group_path(0))

        lg = _new_lg(tmp_path, "cells.zv")
        create_links_array(
            lg, link_width=2, delta=0, sid_ndim=3, store="duplicate",
        )
        write_link_cells(
            lg, all_records[:1], sid_ndim=3, delta=0, store="duplicate",
        )
        write_link_cells(
            lg, all_records[1:], sid_ndim=3, delta=0, store="duplicate",
        )
        finalize_links(lg, delta=0)

        assert _as_set(read_links(lg, delta=0)) == ref
        meta = lg.read_array_meta(links_group_path(0))
        assert meta["num_links"] == ref_meta["num_links"] == 2
        assert (
            meta["num_physical_records"]
            == ref_meta["num_physical_records"]
            == 4
        )

    def test_directed(self, tmp_path: Path) -> None:
        recs = [
            [((1, 0, 0), 9), ((0, 0, 0), 8)],   # cell (1,0,0) @ "-1.0.0"
            [((5, 0, 0), 1), ((6, 0, 0), 2)],   # cell (5,0,0) @ "+1.0.0"
        ]
        # Whole-family reference, so the equivalence is checked for
        # directed too — not just the literal records.  It pre-creates the
        # family exactly as the per-cell coordinator does, so the two
        # differ only in how the records got written.
        lg_whole = _new_lg(tmp_path, "whole.zv")
        create_links_array(
            lg_whole, link_width=2, delta=0, sid_ndim=3, directed=True,
        )
        write_links(lg_whole, recs, sid_ndim=3, delta=0, directed=True)
        ref = _as_set(read_links(lg_whole, delta=0))
        ref_meta = lg_whole.read_array_meta(links_group_path(0))

        lg = _new_lg(tmp_path, "cells.zv")
        create_links_array(
            lg, link_width=2, delta=0, sid_ndim=3, directed=True,
        )
        write_link_cells(
            lg, recs[:1], sid_ndim=3, delta=0, directed=True,
        )
        write_link_cells(
            lg, recs[1:], sid_ndim=3, delta=0, directed=True,
        )
        finalize_links(lg, delta=0)
        out = _as_set(read_links(lg, delta=0))
        assert out == ref == {
            (((1, 0, 0), 9), ((0, 0, 0), 8)),
            (((5, 0, 0), 1), ((6, 0, 0), 2)),
        }
        # Direction survives as the offset sign: a canonical family would
        # have filed the first record at cell (0,0,0) under "+1.0.0".
        assert "-1.0.0" in list_link_offsets(lg, 0)
        assert list_link_offsets(lg, 0) == list_link_offsets(lg_whole, 0)
        meta = lg.read_array_meta(links_group_path(0))
        assert meta["directed"] is True
        assert meta["num_links"] == ref_meta["num_links"] == 2


class TestAttributeCells:
    """Per-cell attribute writes stay row-aligned with per-cell link writes."""

    def test_round_trip(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        create_links_array(lg, link_width=2, delta=0, sid_ndim=3)
        create_link_attributes_array(lg, "weight", delta=0, sid_ndim=3)

        batch1 = [[((0, 0, 0), 1), ((1, 0, 0), 2)]]
        batch2 = [[((5, 0, 0), 5), ((6, 0, 0), 6)]]
        p1 = write_link_cells(lg, batch1, sid_ndim=3, delta=0)
        write_link_attribute_cells(
            lg, "weight", np.array([0.1], dtype=np.float32),
            partition=p1, delta=0,
        )
        p2 = write_link_cells(lg, batch2, sid_ndim=3, delta=0)
        write_link_attribute_cells(
            lg, "weight", np.array([0.9], dtype=np.float32),
            partition=p2, delta=0,
        )
        finalize_links(lg, delta=0)

        links = read_links(lg, delta=0)
        attrs = read_link_attributes(lg, "weight", delta=0)
        by_rec = dict(zip((tuple(r) for r in links), attrs))
        assert np.isclose(by_rec[(((0, 0, 0), 1), ((1, 0, 0), 2))], 0.1)
        assert np.isclose(by_rec[(((5, 0, 0), 5), ((6, 0, 0), 6))], 0.9)

    def test_duplicate_attributes_replicate(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        create_links_array(
            lg, link_width=2, delta=0, sid_ndim=3, store="duplicate",
        )
        create_link_attributes_array(lg, "weight", delta=0, sid_ndim=3)
        batch = [[((0, 0, 0), 1), ((4, 4, 4), 9)]]
        p = write_link_cells(
            lg, batch, sid_ndim=3, delta=0, store="duplicate",
        )
        write_link_attribute_cells(
            lg, "weight", np.array([0.5], dtype=np.float32),
            partition=p, delta=0,
        )
        finalize_links(lg, delta=0)
        links = read_links(lg, delta=0)
        attrs = read_link_attributes(lg, "weight", delta=0)
        assert len(links) == len(attrs) == 2
        assert all(np.isclose(a, 0.5) for a in attrs)


class TestPolicyGuards:
    def test_store_mismatch_raises(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        create_links_array(
            lg, link_width=2, delta=0, sid_ndim=3, store="canonical",
        )
        with pytest.raises(ArrayError, match="store"):
            write_link_cells(
                lg, [[((0, 0, 0), 1), ((1, 0, 0), 2)]],
                sid_ndim=3, delta=0, store="duplicate",
            )

    def test_directed_mismatch_raises(self, tmp_path: Path) -> None:
        lg = _new_lg(tmp_path)
        create_links_array(
            lg, link_width=2, delta=0, sid_ndim=3, directed=True,
        )
        with pytest.raises(ArrayError, match="directed"):
            write_link_cells(
                lg, [[((0, 0, 0), 1), ((1, 0, 0), 2)]],
                sid_ndim=3, delta=0, directed=False,
            )


class TestFinalizeThenShard:
    """After finalize, a shard pass must preserve the records."""

    def test_shard_round_trip(self, tmp_path: Path) -> None:
        from zarr_vectors.core.store import open_store
        from zarr_vectors.sharding import shard_store
        from zarr_vectors.sharding.io import _is_native_sharded

        store_path = tmp_path / "cells.zv"
        root = create_store(
            str(store_path),
            bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
            chunk_shape=(100.0, 100.0, 100.0),
            geometry_types=["graph"],
            ndim=3,
        )
        lg = get_resolution_level(root, 0)
        create_links_array(lg, link_width=2, delta=0, sid_ndim=3)
        recs = [
            [((0, 0, 0), 1), ((1, 0, 0), 2)],   # cross  → "+1.0.0"
            [((5, 0, 0), 5), ((6, 0, 0), 6)],   # cross  → "+1.0.0"
            [((2, 0, 0), 3), ((2, 0, 0), 4)],   # intra  → "0.0.0"
        ]
        write_link_cells(lg, recs, sid_ndim=3, delta=0)
        finalize_links(lg, delta=0)
        before = _as_set(read_links(lg, delta=0))
        assert len(before) == 3

        shard_store(str(store_path))

        # Re-open after the (destructive) shard pass and re-read.
        root2 = open_store(str(store_path), mode="r")
        lg2 = get_resolution_level(root2, 0)
        after = _as_set(read_links(lg2, delta=0))
        assert after == before

        # Each offsets array is a rank-D grid, so shard_store must
        # actually have migrated it.  Without this the test would pass
        # even if sharding skipped the links family entirely — which is
        # exactly what made the pre-merge version of it vacuous.
        segments = list_link_offsets(lg2, 0)
        assert segments == ["+1.0.0", "0.0.0"]
        for seg in segments:
            node = lg2.zarr_group[f"{links_group_path(0)}/{seg}"]
            assert _is_native_sharded(node), (
                f"{links_path(0, ((0, 0, 0),)).rsplit('/', 1)[0]}/{seg} "
                f"was not sharded by shard_store"
            )
