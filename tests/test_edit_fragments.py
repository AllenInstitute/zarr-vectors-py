"""Tests for fragment-level edits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.core.arrays import read_chunk_vertices, read_object_manifest
from zarr_vectors.core.store import open_store
from zarr_vectors.exceptions import EditError
from zarr_vectors.ops import (
    FragmentRef,
    add_fragment,
    edit_fragment,
    remove_fragment,
    VertexRef,
)
from zarr_vectors.types.points import write_points


@pytest.fixture
def store_with_fragments(tmp_path: Path) -> str:
    path = tmp_path / "store.zv"
    positions = np.array(
        [[10.0, 10.0, 10.0], [11.0, 11.0, 11.0]], dtype=np.float32,
    )
    write_points(
        str(path), positions,
        chunk_shape=(200.0, 200.0, 200.0),
        bounds=([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]),
        object_ids=np.array([0, 1], dtype=np.int64),
    )
    return str(path)


class TestAddFragment:

    def test_add_fragment_appends(self, store_with_fragments: str) -> None:
        root = open_store(store_with_fragments, mode="r+")
        ref, report = add_fragment(
            root, level=0, chunk=(0, 0, 0),
            vertices=np.array(
                [[20.0, 20.0, 20.0], [21.0, 21.0, 21.0]], dtype=np.float32,
            ),
        )
        assert isinstance(ref, FragmentRef)
        groups = read_chunk_vertices(
            root["0"], ref.chunk, dtype=np.float32, ndim=3,
        )
        np.testing.assert_allclose(
            groups[ref.fragment],
            [[20.0, 20.0, 20.0], [21.0, 21.0, 21.0]],
            atol=1e-5,
        )
        assert report.n_edits == 1


class TestEditFragment:

    def test_edit_fragment_minimal_replace_rows(
        self, store_with_fragments: str,
    ) -> None:
        root = open_store(store_with_fragments, mode="r+")
        ref0 = VertexRef.from_object(root, level=0, object_id=0, vertex_index=0)
        frag_ref = FragmentRef(
            level=0, chunk=ref0.chunk, fragment=ref0.fragment,
        )
        new_rows = np.array([[50.0, 50.0, 50.0]], dtype=np.float32)
        report = edit_fragment(
            root, frag_ref, new_vertices=new_rows, atomic=False,
        )
        groups = read_chunk_vertices(
            root["0"], frag_ref.chunk, dtype=np.float32, ndim=3,
        )
        np.testing.assert_allclose(
            groups[frag_ref.fragment], new_rows, atol=1e-5,
        )
        # Manifest is unchanged under atomic=False.
        assert report.oid_remap == {}
        m = read_object_manifest(root["0"], 0)
        assert tuple(m[0][0]) == frag_ref.chunk
        assert m[0][1] == frag_ref.fragment

    def test_edit_fragment_atomic_remaps_referring_oids(
        self, store_with_fragments: str,
    ) -> None:
        root = open_store(store_with_fragments, mode="r+")
        ref0 = VertexRef.from_object(root, level=0, object_id=0, vertex_index=0)
        frag_ref = FragmentRef(
            level=0, chunk=ref0.chunk, fragment=ref0.fragment,
        )
        new_rows = np.array([[55.0, 55.0, 55.0]], dtype=np.float32)
        report = edit_fragment(
            root, frag_ref, new_vertices=new_rows, atomic=True,
        )
        assert 0 in report.oid_remap
        new_oid = report.oid_remap[0]
        # New OID's manifest points at the new fragment; old OID's
        # manifest still points at the original.
        new_m = read_object_manifest(root["0"], new_oid)
        assert new_m and new_m[0][1] != frag_ref.fragment


class TestRemoveFragment:

    def test_remove_fragment_atomic_tombstones(
        self, store_with_fragments: str,
    ) -> None:
        root = open_store(store_with_fragments, mode="r+")
        ref0 = VertexRef.from_object(root, level=0, object_id=0, vertex_index=0)
        frag_ref = FragmentRef(
            level=0, chunk=ref0.chunk, fragment=ref0.fragment,
        )
        report = remove_fragment(root, frag_ref, atomic=True)
        # Fragment slot is empty but still present.
        groups = read_chunk_vertices(
            root["0"], frag_ref.chunk, dtype=np.float32, ndim=3,
        )
        assert frag_ref.fragment < len(groups)
        assert groups[frag_ref.fragment].shape[0] == 0
        # New OID drops the reference; original OID still points at it
        # (which is fine — atomic guarantees old readers keep working).
        new_oid = report.oid_remap[0]
        new_m = read_object_manifest(root["0"], new_oid)
        assert all(
            not (tuple(cc) == tuple(frag_ref.chunk) and fi == frag_ref.fragment)
            for cc, fi in new_m
        )

    def test_remove_fragment_minimal_rejected(
        self, store_with_fragments: str,
    ) -> None:
        root = open_store(store_with_fragments, mode="r+")
        ref0 = VertexRef.from_object(root, level=0, object_id=0, vertex_index=0)
        frag_ref = FragmentRef(
            level=0, chunk=ref0.chunk, fragment=ref0.fragment,
        )
        with pytest.raises(EditError):
            remove_fragment(root, frag_ref, atomic=False)
