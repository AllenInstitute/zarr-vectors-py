"""Tests for per-vertex / per-object / per-link attribute edits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.core.arrays import (
    read_chunk_attributes,
    read_object_attributes,
)
from zarr_vectors.core.store import open_store
from zarr_vectors.ops import (
    AttributeRef,
    ObjectRef,
    VertexRef,
    add_attribute,
    edit_attribute,
    remove_attribute,
)
from zarr_vectors.types.points import write_points


@pytest.fixture
def attr_store(tmp_path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    """3-point store with both per-vertex and per-object attributes."""
    path = tmp_path / "store.zv"
    positions = np.array(
        [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0], [30.0, 30.0, 30.0]],
        dtype=np.float32,
    )
    intensity = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    label = np.array([10, 20, 30], dtype=np.int64)
    write_points(
        str(path), positions,
        chunk_shape=(200.0, 200.0, 200.0),
        bounds=([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]),
        object_ids=np.arange(3, dtype=np.int64),
        vertex_attributes={"intensity": intensity},
        object_attributes={"label": label},
    )
    return str(path), intensity, label


class TestVertexAttribute:

    def test_edit_per_vertex_in_place(
        self, attr_store: tuple[str, np.ndarray, np.ndarray],
    ) -> None:
        path, intensity, _ = attr_store
        root = open_store(path, mode="r+")
        ref = VertexRef.from_object(root, level=0, object_id=1, vertex_index=0)
        edit_attribute(
            root,
            AttributeRef(scope="vertex", name="intensity", target=ref),
            value=9.0,
        )
        groups = read_chunk_attributes(
            root["0"], "intensity", ref.chunk, dtype=np.float32,
        )
        np.testing.assert_allclose(groups[ref.fragment][ref.local], 9.0, atol=1e-5)
        # Other rows untouched.
        other = VertexRef.from_object(root, level=0, object_id=0, vertex_index=0)
        np.testing.assert_allclose(
            read_chunk_attributes(
                root["0"], "intensity", other.chunk, dtype=np.float32,
            )[other.fragment][other.local],
            intensity[0],
            atol=1e-5,
        )


class TestObjectAttribute:

    def test_edit_per_object_preserves_siblings(
        self, attr_store: tuple[str, np.ndarray, np.ndarray],
    ) -> None:
        path, _, label = attr_store
        root = open_store(path, mode="r+")
        edit_attribute(
            root,
            AttributeRef(
                scope="object", name="label",
                target=ObjectRef(level=0, object_id=1),
            ),
            value=99,
        )
        arr = read_object_attributes(root["0"], "label")
        assert arr[1] == 99
        # Other OIDs unchanged.
        assert arr[0] == label[0]
        assert arr[2] == label[2]

    def test_remove_per_object_writes_zero(
        self, attr_store: tuple[str, np.ndarray, np.ndarray],
    ) -> None:
        path, _, _ = attr_store
        root = open_store(path, mode="r+")
        remove_attribute(
            root,
            AttributeRef(
                scope="object", name="label",
                target=ObjectRef(level=0, object_id=1),
            ),
        )
        arr = read_object_attributes(root["0"], "label")
        assert arr[1] == 0

    def test_add_new_per_object_attribute_allocates(
        self, attr_store: tuple[str, np.ndarray, np.ndarray],
    ) -> None:
        path, _, _ = attr_store
        root = open_store(path, mode="r+")
        add_attribute(
            root,
            AttributeRef(
                scope="object", name="confidence",
                target=ObjectRef(level=0, object_id=1),
            ),
            value=0.42,
        )
        arr = read_object_attributes(root["0"], "confidence")
        assert arr.shape == (3,)
        np.testing.assert_allclose(arr[1], 0.42, atol=1e-5)
        # Other rows are zero-filled defaults.
        assert arr[0] == 0
        assert arr[2] == 0
