"""Tests for the opt-in ``layout="packed_sharded"`` cross-chunk-links layout.

``packed_sharded`` stores ALL of a ``cross_chunk_links/<delta>/`` family's
populated cells in ONE native-sharded 1-D vlen-bytes array (shape
``(N,)``), with the sorted cell-key list carried in the array meta as
``cell_keys``.  It must read back byte-for-byte identically to the default
``flat_cells`` layout — these tests pin that equivalence, the on-disk file
collapse, append semantics, and the empty-input edge case.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.core.arrays import (
    read_cross_chunk_links,
    read_cross_chunk_links_for_tuple,
    write_cross_chunk_links,
)
from zarr_vectors.core.paths import cross_chunk_links_path
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


def _write_both(tmp_path, records, **kwargs):
    """Write the SAME records to a flat store and a packed store."""
    lg_flat = _new_lg(tmp_path, "flat.zv")
    lg_packed = _new_lg(tmp_path, "packed.zv")
    p_flat = write_cross_chunk_links(lg_flat, records, **kwargs)
    p_packed = write_cross_chunk_links(
        lg_packed, records, layout="packed_sharded", **kwargs
    )
    return (lg_flat, p_flat), (lg_packed, p_packed)


# ------------------------------------------------------------------
# 1. Round-trip equality: packed reads identical to flat.
# ------------------------------------------------------------------

class TestRoundTripEquality:
    def test_directed_streamline_l2(self, tmp_path: Path) -> None:
        # directed edges (link_width=2), A->B and B->A distinct cells.
        records = [
            [((0, 0, 0), 1), ((1, 0, 0), 2)],
            [((1, 0, 0), 9), ((0, 0, 0), 8)],
            [((2, 0, 0), 8), ((0, 0, 0), 3)],  # canonical would reorder
            [((5, 0, 0), 4), ((5, 1, 0), 5)],
        ]
        (lg_f, _), (lg_p, _) = _write_both(
            tmp_path, records, sid_ndim=3, delta=0, directed=True,
        )
        assert read_cross_chunk_links(lg_f, delta=0) == \
            read_cross_chunk_links(lg_p, delta=0)

    def test_undirected_canonical_l2(self, tmp_path: Path) -> None:
        records = [
            [((0, 0, 0), 1), ((4, 4, 4), 9)],
            [((4, 4, 4), 2), ((0, 0, 0), 7)],
            [((1, 0, 0), 3), ((2, 0, 0), 6)],
            [((2, 0, 0), 8), ((0, 0, 0), 3)],
        ]
        (lg_f, _), (lg_p, _) = _write_both(
            tmp_path, records, sid_ndim=3, delta=0,
        )
        assert read_cross_chunk_links(lg_f, delta=0) == \
            read_cross_chunk_links(lg_p, delta=0)

    def test_triangle_faces_l3(self, tmp_path: Path) -> None:
        faces = [
            [((0, 0, 0), 5), ((1, 0, 0), 12), ((1, 1, 1), 7)],
            [((1, 1, 1), 7), ((1, 0, 0), 12), ((0, 0, 0), 5)],  # opp winding
            [((2, 0, 0), 1), ((3, 0, 0), 2), ((4, 0, 0), 3)],
        ]
        (lg_f, _), (lg_p, _) = _write_both(
            tmp_path, faces, sid_ndim=3, delta=0, link_width=3,
        )
        assert read_cross_chunk_links(lg_f, delta=0) == \
            read_cross_chunk_links(lg_p, delta=0)

    def test_many_cells_equal(self, tmp_path: Path) -> None:
        records = [
            [((i, 0, 0), i * 2), ((i + 1, 0, 0), i * 2 + 1)]
            for i in range(200)
        ]
        (lg_f, _), (lg_p, _) = _write_both(
            tmp_path, records, sid_ndim=3, delta=0,
        )
        rf = read_cross_chunk_links(lg_f, delta=0)
        rp = read_cross_chunk_links(lg_p, delta=0)
        assert rf == rp
        assert len(rf) == 200


# ------------------------------------------------------------------
# 2. Per-tuple reads: packed == flat, present and absent.
# ------------------------------------------------------------------

class TestForTupleEquality:
    def test_present_and_absent(self, tmp_path: Path) -> None:
        records = [
            [((0, 0, 0), 1), ((4, 4, 4), 9)],
            [((4, 4, 4), 2), ((0, 0, 0), 7)],
            [((1, 0, 0), 3), ((2, 0, 0), 6)],
        ]
        (lg_f, _), (lg_p, _) = _write_both(
            tmp_path, records, sid_ndim=3, delta=0,
        )
        queries = [
            [(0, 0, 0), (4, 4, 4)],   # present, either order resolves same
            [(4, 4, 4), (0, 0, 0)],
            [(1, 0, 0), (2, 0, 0)],   # present
            [(5, 5, 5), (6, 6, 6)],   # absent
        ]
        for q in queries:
            assert read_cross_chunk_links_for_tuple(lg_f, q, delta=0) == \
                read_cross_chunk_links_for_tuple(lg_p, q, delta=0)

    def test_directed_tuple_direction(self, tmp_path: Path) -> None:
        records = [
            [((0, 0, 0), 1), ((1, 0, 0), 2)],
            [((1, 0, 0), 9), ((0, 0, 0), 8)],
        ]
        (lg_f, _), (lg_p, _) = _write_both(
            tmp_path, records, sid_ndim=3, delta=0, directed=True,
        )
        for q in ([(0, 0, 0), (1, 0, 0)], [(1, 0, 0), (0, 0, 0)]):
            assert read_cross_chunk_links_for_tuple(lg_f, q, delta=0) == \
                read_cross_chunk_links_for_tuple(lg_p, q, delta=0)


# ------------------------------------------------------------------
# 3. On-disk: packed is a single sharded array (few files).
# ------------------------------------------------------------------

class TestOnDiskFileCount:
    def test_packed_collapses_files(self, tmp_path: Path) -> None:
        # 200 records producing ~200 distinct cells in flat mode.
        records = [
            [((i, 0, 0), i * 2), ((i + 1, 0, 0), i * 2 + 1)]
            for i in range(200)
        ]
        lg_flat = _new_lg(tmp_path, "flat.zv")
        lg_packed = _new_lg(tmp_path, "packed.zv")
        write_cross_chunk_links(lg_flat, records, sid_ndim=3, delta=0)
        write_cross_chunk_links(
            lg_packed, records, sid_ndim=3, delta=0,
            layout="packed_sharded",
        )

        family = cross_chunk_links_path(0)
        flat_dir = lg_flat.path / family
        packed_dir = lg_packed.path / family
        flat_files = [p for p in flat_dir.rglob("*") if p.is_file()]
        packed_files = [p for p in packed_dir.rglob("*") if p.is_file()]

        # Flat writes one object per cell (hundreds); packed collapses to
        # a handful (zarr.json + ceil(N/512) shard files).
        assert len(flat_files) > 100
        assert len(packed_files) <= 5, (
            f"expected a handful of packed files, got {len(packed_files)}"
        )

        # The packed family is a single standalone Zarr array, not a group.
        assert lg_packed.standalone_array_exists(family)
        assert not lg_flat.standalone_array_exists(family)


# ------------------------------------------------------------------
# 4. Append mode: packed append == flat append.
# ------------------------------------------------------------------

class TestAppendEquality:
    def test_append_matches_flat(self, tmp_path: Path) -> None:
        batch1 = [
            [((i, 0, 0), i), ((i + 1, 0, 0), i + 1)] for i in range(30)
        ]
        batch2 = [
            [((i, 0, 0), i * 3), ((i + 1, 0, 0), i * 3 + 1)]
            for i in range(15, 45)
        ]
        lg_flat = _new_lg(tmp_path, "flat.zv")
        lg_packed = _new_lg(tmp_path, "packed.zv")

        write_cross_chunk_links(lg_flat, batch1, sid_ndim=3, delta=0)
        pf = write_cross_chunk_links(
            lg_flat, batch2, sid_ndim=3, delta=0, mode="append",
        )
        write_cross_chunk_links(
            lg_packed, batch1, sid_ndim=3, delta=0, layout="packed_sharded",
        )
        pp = write_cross_chunk_links(
            lg_packed, batch2, sid_ndim=3, delta=0, mode="append",
            layout="packed_sharded",
        )

        assert read_cross_chunk_links(lg_flat, delta=0) == \
            read_cross_chunk_links(lg_packed, delta=0)
        assert pf.first_new == pp.first_new == 30
        assert pf.num_links == pp.num_links

        # Family counts match too.
        mf = lg_flat.read_array_meta(cross_chunk_links_path(0))
        mp = lg_packed.read_array_meta(cross_chunk_links_path(0))
        assert mf["num_links"] == mp["num_links"]
        assert mf["num_physical_records"] == mp["num_physical_records"]

    def test_append_from_empty(self, tmp_path: Path) -> None:
        # Append onto a never-written family behaves like a first write.
        records = [[((0, 0, 0), 1), ((1, 0, 0), 2)]]
        lg_packed = _new_lg(tmp_path, "packed.zv")
        write_cross_chunk_links(
            lg_packed, records, sid_ndim=3, delta=0, mode="append",
            layout="packed_sharded",
        )
        assert read_cross_chunk_links(lg_packed, delta=0) == [
            (((0, 0, 0), 1), ((1, 0, 0), 2)),
        ]


# ------------------------------------------------------------------
# 5. Empty input.
# ------------------------------------------------------------------

class TestEmptyInput:
    def test_empty_no_crash_reads_empty(self, tmp_path: Path) -> None:
        lg_packed = _new_lg(tmp_path, "packed.zv")
        p = write_cross_chunk_links(
            lg_packed, [], sid_ndim=3, delta=0, layout="packed_sharded",
        )
        assert p.num_links == 0
        assert p.cell_indices == {}
        assert read_cross_chunk_links(lg_packed, delta=0) == []
        assert read_cross_chunk_links_for_tuple(
            lg_packed, [(0, 0, 0), (1, 0, 0)], delta=0,
        ) == []


# ------------------------------------------------------------------
# Meta: packed stamps layout + cell_keys; flat stays layout-free.
# ------------------------------------------------------------------

class TestLayoutMeta:
    def test_packed_meta_fields(self, tmp_path: Path) -> None:
        records = [
            [((0, 0, 0), 1), ((1, 0, 0), 2)],
            [((3, 0, 0), 5), ((4, 0, 0), 6)],
        ]
        lg_packed = _new_lg(tmp_path, "packed.zv")
        write_cross_chunk_links(
            lg_packed, records, sid_ndim=3, delta=0,
            layout="packed_sharded",
        )
        meta = lg_packed.read_array_meta(cross_chunk_links_path(0))
        assert meta["layout"] == "packed_sharded"
        assert meta["link_width"] == 2
        assert meta["sid_ndim"] == 3
        assert meta["num_links"] == 2
        assert meta["num_physical_records"] == 2
        # cell_keys is sorted and its length is N (populated cells).
        assert meta["cell_keys"] == sorted(meta["cell_keys"])
        assert len(meta["cell_keys"]) == 2

    def test_flat_default_has_no_layout_key(self, tmp_path: Path) -> None:
        # Zero-regression guard: the default flat path must not stamp a
        # ``layout`` attribute onto the family meta.
        lg_flat = _new_lg(tmp_path, "flat.zv")
        write_cross_chunk_links(
            lg_flat, [[((0, 0, 0), 1), ((1, 0, 0), 2)]], sid_ndim=3, delta=0,
        )
        meta = lg_flat.read_array_meta(cross_chunk_links_path(0))
        assert "layout" not in meta


# ------------------------------------------------------------------
# 5. Hybrid read helpers: manifest (O(cells)) + cells-by-index.
# ------------------------------------------------------------------

class TestHybridReadHelpers:
    _RECORDS = [
        [((0, 0, 0), 1), ((1, 0, 0), 2)],
        [((1, 0, 0), 9), ((0, 0, 0), 8)],
        [((2, 0, 0), 8), ((0, 0, 0), 3)],
        [((5, 0, 0), 4), ((5, 1, 0), 5)],
        [((0, 0, 0), 7), ((1, 0, 0), 7)],
    ]

    def test_manifest_reports_cells_and_policy(self, tmp_path: Path) -> None:
        from zarr_vectors.core.arrays import read_cross_chunk_link_manifest
        lg = _new_lg(tmp_path, "m.zv")
        write_cross_chunk_links(
            lg, self._RECORDS, sid_ndim=3, delta=0, directed=True,
            layout="packed_sharded",
        )
        man = read_cross_chunk_link_manifest(lg, delta=0)
        assert man is not None
        assert man["layout"] == "packed_sharded"
        assert man["link_width"] == 2
        assert man["sid_ndim"] == 3
        assert man["directed"] is True
        # cell_keys sorted, flat index i <-> cell_keys[i].
        assert man["cell_keys"] == sorted(man["cell_keys"])
        assert len(man["cell_keys"]) >= 1

    def test_manifest_none_when_absent(self, tmp_path: Path) -> None:
        from zarr_vectors.core.arrays import read_cross_chunk_link_manifest
        lg = _new_lg(tmp_path, "empty.zv")
        assert read_cross_chunk_link_manifest(lg, delta=0) is None

    def test_all_cells_by_index_equals_full_read(self, tmp_path: Path) -> None:
        from zarr_vectors.core.arrays import (
            read_cross_chunk_link_cells_by_index,
            read_cross_chunk_link_manifest,
        )
        lg = _new_lg(tmp_path, "byidx.zv")
        write_cross_chunk_links(
            lg, self._RECORDS, sid_ndim=3, delta=0, directed=True,
            layout="packed_sharded",
        )
        man = read_cross_chunk_link_manifest(lg, delta=0)
        specs = list(enumerate(man["cell_keys"]))
        by_idx = read_cross_chunk_link_cells_by_index(
            lg, specs, delta=0, link_width=man["link_width"],
            sid_ndim=man["sid_ndim"], layout=man["layout"],
        )
        full = read_cross_chunk_links(lg, delta=0)
        # Same multiset of records (by_index iterates cells in manifest order,
        # which IS the full-read order).
        assert by_idx == full

    def test_subset_by_index_returns_only_those_cells(self, tmp_path: Path) -> None:
        from zarr_vectors.core.arrays import (
            read_cross_chunk_link_cells_by_index,
            read_cross_chunk_link_manifest,
        )
        from zarr_vectors.core.paths import parse_cell_key
        lg = _new_lg(tmp_path, "sub.zv")
        write_cross_chunk_links(
            lg, self._RECORDS, sid_ndim=3, delta=0, directed=True,
            layout="packed_sharded",
        )
        man = read_cross_chunk_link_manifest(lg, delta=0)
        # Read just cell 0.
        one = read_cross_chunk_link_cells_by_index(
            lg, [(0, man["cell_keys"][0])], delta=0,
            link_width=man["link_width"], sid_ndim=man["sid_ndim"],
            layout=man["layout"],
        )
        # Every returned record's endpoints must belong to cell 0's chunks.
        cell0_chunks = {
            tuple(c) for c in parse_cell_key(
                man["cell_keys"][0], sid_ndim=3, link_width=2,
            )
        }
        assert one, "cell 0 should have >=1 record"
        for rec in one:
            for chunk, _vi in rec:
                assert tuple(chunk) in cell0_chunks

    def test_empty_specs(self, tmp_path: Path) -> None:
        from zarr_vectors.core.arrays import read_cross_chunk_link_cells_by_index
        lg = _new_lg(tmp_path, "es.zv")
        write_cross_chunk_links(
            lg, self._RECORDS, sid_ndim=3, delta=0, directed=True,
            layout="packed_sharded",
        )
        assert read_cross_chunk_link_cells_by_index(
            lg, [], delta=0, link_width=2, sid_ndim=3, layout="packed_sharded",
        ) == []
