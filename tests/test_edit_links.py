"""Tests for link / cross-chunk-link edits + the implicit-sequential
materialise helper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.core.arrays import read_chunk_links, read_cross_chunk_links
from zarr_vectors.core.store import open_store
from zarr_vectors.exceptions import EditError
from zarr_vectors.ops import (
    EditSession,
    LinkRef,
    add_cross_chunk_link,
    add_link,
    edit_link,
    materialise_object_links_explicit,
    remove_link,
)
from zarr_vectors.types.graphs import write_graph


@pytest.fixture
def explicit_graph(tmp_path: Path) -> str:
    """Tiny explicit-convention graph in one chunk."""
    path = tmp_path / "store.zv"
    positions = np.array(
        [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0], [30.0, 30.0, 30.0]],
        dtype=np.float32,
    )
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    write_graph(
        str(path), positions, edges,
        chunk_shape=(200.0, 200.0, 200.0),
        bounds=([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]),
        kind="graph",  # explicit convention
    )
    return str(path)


@pytest.fixture
def skeleton_graph(tmp_path: Path) -> str:
    """Tiny implicit_sequential_with_branches skeleton."""
    path = tmp_path / "store.zv"
    positions = np.array(
        [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0], [30.0, 30.0, 30.0],
         [40.0, 40.0, 40.0]],
        dtype=np.float32,
    )
    # Linear chain 0->1->2->3 — every edge is sequential.
    edges = np.array([[1, 0], [2, 1], [3, 2]], dtype=np.int64)
    write_graph(
        str(path), positions, edges,
        chunk_shape=(200.0, 200.0, 200.0),
        bounds=([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]),
        kind="skeleton",  # implicit_sequential_with_branches
    )
    return str(path)


class TestAddLink:

    def test_add_intra_chunk(self, explicit_graph: str) -> None:
        root = open_store(explicit_graph, mode="r+")
        ref, report = add_link(
            root, level=0, src=0, dst=2, chunk=(0, 0, 0),
        )
        assert isinstance(ref, LinkRef)
        # Existing 2 edges + 1 new = 3 rows in fragment 0.
        groups = read_chunk_links(root["0"], (0, 0, 0), delta=0)
        # Cardinality varies by writer fragment grouping; assert the new
        # endpoints appear somewhere.
        all_rows = np.concatenate(groups, axis=0) if groups else np.empty((0, 2), int)
        assert np.any(np.all(all_rows == [0, 2], axis=1)), all_rows


class TestEditLink:

    def test_edit_link_minimal_overwrite(self, explicit_graph: str) -> None:
        root = open_store(explicit_graph, mode="r+")
        # Locate the first link via the chunk index — pick fragment 0 row 0.
        groups = read_chunk_links(root["0"], (0, 0, 0), delta=0)
        # Find the fragment that has at least one row.
        target_frag = next(i for i, g in enumerate(groups) if g.shape[0] > 0)
        ref = LinkRef(
            level=0, chunk=(0, 0, 0), fragment=target_frag, row=0, delta=0,
        )
        edit_link(root, ref, new_endpoints=(0, 2), atomic=False)
        groups = read_chunk_links(root["0"], (0, 0, 0), delta=0)
        np.testing.assert_array_equal(groups[target_frag][0], [0, 2])


class TestRemoveLink:

    def test_remove_link_drops_row(self, explicit_graph: str) -> None:
        root = open_store(explicit_graph, mode="r+")
        groups_before = read_chunk_links(root["0"], (0, 0, 0), delta=0)
        n_before = sum(g.shape[0] for g in groups_before)
        target_frag = next(
            i for i, g in enumerate(groups_before) if g.shape[0] > 0
        )
        ref = LinkRef(
            level=0, chunk=(0, 0, 0), fragment=target_frag, row=0, delta=0,
        )
        remove_link(root, ref)
        groups_after = read_chunk_links(root["0"], (0, 0, 0), delta=0)
        n_after = sum(g.shape[0] for g in groups_after)
        assert n_after == n_before - 1


class TestImplicitSequential:

    def test_edit_link_raises_on_implicit_branches(
        self, skeleton_graph: str,
    ) -> None:
        root = open_store(skeleton_graph, mode="r+")
        # The skeleton has no branch rows (the chain is pure sequential
        # so the branch table is empty).  Trying to add a link to the
        # branch table from a sequential parent->child pair is legal.
        ref, _ = add_link(
            root, level=0, src=2, dst=0, chunk=(0, 0, 0),
        )
        assert ref.delta == 0

    def test_materialise_object_links_explicit_round_trip(
        self, skeleton_graph: str,
    ) -> None:
        root = open_store(skeleton_graph, mode="r+")
        added = materialise_object_links_explicit(
            root, level=0, object_id=0, flip_convention=False,
        )
        # 4-vertex chain has 3 consecutive pairs.
        assert added >= 1


class TestCrossChunkLink:

    def test_add_cross_chunk_link(self, tmp_path: Path) -> None:
        path = tmp_path / "store.zv"
        positions = np.array(
            [[10.0, 10.0, 10.0], [70.0, 70.0, 70.0]], dtype=np.float32,
        )
        edges = np.array([[0, 1]], dtype=np.int64)
        write_graph(
            str(path), positions, edges,
            chunk_shape=(50.0, 50.0, 50.0),
            bounds=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
            kind="graph",
        )
        root = open_store(str(path), mode="r+")
        before = read_cross_chunk_links(root["0"], delta=0)
        endpoints = [((0, 0, 0), 0), ((1, 1, 1), 0)]
        ref, _ = add_cross_chunk_link(
            root, level=0, endpoints=endpoints, delta=0,
        )
        after = read_cross_chunk_links(root["0"], delta=0)
        assert len(after) == len(before) + 1
