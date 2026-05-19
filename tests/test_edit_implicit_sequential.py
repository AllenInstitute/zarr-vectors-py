"""Tests for the implicit-sequential workflows.

Two paths are supported:

- ``add_vertex(manifest_position=K)`` splices a new vertex at a
  specific position in an object's manifest.  Under
  ``implicit_sequential*`` the topology updates naturally with no
  branch-table writes.
- ``materialise_object_links_explicit`` expands the implicit chain
  into branch-table rows so subsequent ``edit_link`` calls can
  address each edge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.core.arrays import read_object_manifest
from zarr_vectors.core.store import open_store
from zarr_vectors.exceptions import EditError
from zarr_vectors.ops import (
    EditSession,
    materialise_object_links_explicit,
)
from zarr_vectors.types.graphs import write_graph
from zarr_vectors.types.points import write_points


@pytest.fixture
def linear_skeleton(tmp_path: Path) -> str:
    """6-node linear skeleton with default writer layout (single fragment).

    ``materialise_object_links_explicit`` uses this — fragment-level
    layout doesn't matter for it, only the vertex sequence does.
    """
    path = tmp_path / "store.zv"
    positions = np.array(
        [
            [10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0],
            [30.0, 30.0, 30.0],
            [40.0, 40.0, 40.0],
            [50.0, 50.0, 50.0],
            [60.0, 60.0, 60.0],
        ],
        dtype=np.float32,
    )
    edges = np.array([[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]], dtype=np.int64)
    write_graph(
        str(path), positions, edges,
        chunk_shape=(200.0, 200.0, 200.0),
        bounds=([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]),
        kind="skeleton",
    )
    return str(path)


@pytest.fixture
def fine_grained_object(tmp_path: Path) -> str:
    """6-point points store where each vertex is in its own chunk (so
    each lands in a separate fragment).  All six belong to ``object_id=0``,
    so the manifest has 6 entries — the granularity needed by the
    splicing tests.
    """
    path = tmp_path / "store.zv"
    # Spread the points across different chunks (chunk side = 50, so
    # each point is in a unique cell).
    positions = np.array(
        [
            [10.0, 10.0, 10.0],   # chunk (0,0,0)
            [60.0, 10.0, 10.0],   # chunk (1,0,0)
            [10.0, 60.0, 10.0],   # chunk (0,1,0)
            [60.0, 60.0, 10.0],   # chunk (1,1,0)
            [10.0, 10.0, 60.0],   # chunk (0,0,1)
            [60.0, 10.0, 60.0],   # chunk (1,0,1)
        ],
        dtype=np.float32,
    )
    write_points(
        str(path), positions,
        chunk_shape=(50.0, 50.0, 50.0),
        bounds=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
        object_ids=np.zeros(6, dtype=np.int64),
    )
    return str(path)


class TestManifestSplicing:

    def test_splice_inserts_at_position(
        self, fine_grained_object: str,
    ) -> None:
        """Insert a new vertex H between C (idx 2) and D (idx 3) of the
        sequence [A,B,C,D,E,F]."""
        root = open_store(fine_grained_object, mode="r+")
        before = read_object_manifest(root["0"], 0)
        with EditSession(root, atomic=False) as ed:
            new_ref = ed.add_vertex(
                level=0,
                pos=[35.0, 35.0, 35.0],
                object_id=0,
                manifest_position=3,  # between C and D
            )
        after = read_object_manifest(root["0"], 0)
        # Manifest length grew by 1.
        assert len(after) == len(before) + 1
        # The new fragment ref lands at position 3 of the manifest.
        assert tuple(after[3][0]) == new_ref.chunk
        assert after[3][1] == new_ref.fragment

    def test_splice_requires_object_id(
        self, fine_grained_object: str,
    ) -> None:
        root = open_store(fine_grained_object, mode="r+")
        with EditSession(root) as ed:
            with pytest.raises(EditError):
                ed.add_vertex(
                    level=0,
                    pos=[35.0, 35.0, 35.0],
                    manifest_position=3,
                )

    def test_splice_out_of_range(self, fine_grained_object: str) -> None:
        root = open_store(fine_grained_object, mode="r+")
        manifest_len_before = len(read_object_manifest(root["0"], 0))
        with EditSession(root) as ed:
            with pytest.raises(EditError):
                ed.add_vertex(
                    level=0,
                    pos=[35.0, 35.0, 35.0],
                    object_id=0,
                    manifest_position=manifest_len_before + 5,
                )


class TestMaterialise:

    def test_materialise_emits_branch_rows(self, linear_skeleton: str) -> None:
        root = open_store(linear_skeleton, mode="r+")
        added = materialise_object_links_explicit(
            root, level=0, object_id=0, flip_convention=False,
        )
        # 6 vertices -> 5 consecutive pairs.
        assert added == 5

    def test_materialise_with_flip_changes_convention(
        self, linear_skeleton: str,
    ) -> None:
        from zarr_vectors.core.metadata import RootMetadata
        root = open_store(linear_skeleton, mode="r+")
        materialise_object_links_explicit(
            root, level=0, object_id=0, flip_convention=True,
        )
        meta = RootMetadata.from_dict(root.attrs.to_dict())
        assert meta.links_convention == "explicit"
