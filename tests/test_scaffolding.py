"""Step 01 tests: verify scaffolding imports and fixtures produce valid data."""

from __future__ import annotations

import numpy as np


class TestImports:
    """Verify all scaffolding modules import without error."""

    def test_import_package(self) -> None:
        import zarr_vectors
        assert hasattr(zarr_vectors, "__version__")

    def test_import_exceptions(self) -> None:
        from zarr_vectors.exceptions import (
            ZVFError,
            StoreError,
            MetadataError,
            ArrayError,
            ChunkingError,
            ConventionError,
            ValidationError,
            IngestError,
            ExportError,
            CoarseningError,
            DracoError,
        )
        # All should be subclasses of ZVFError
        for exc_class in [
            StoreError, MetadataError, ArrayError, ChunkingError,
            ConventionError, ValidationError, IngestError, ExportError,
            CoarseningError, DracoError,
        ]:
            assert issubclass(exc_class, ZVFError)

    def test_import_typing(self) -> None:
        from zarr_vectors.typing import (
            Vertices, Faces, Edges, ParentArray,
            ChunkCoords, ChunkShape, BoundingBox,
            FragmentRef, ObjectManifest, CrossChunkLink,
        )

    def test_import_constants(self) -> None:
        from zarr_vectors.constants import (
            FORMAT_VERSION, VERTICES, LINKS, OBJECT_INDEX,
            LINKS_EXPLICIT, LINKS_IMPLICIT_SEQUENTIAL,
            OBJIDX_STANDARD, OBJIDX_IDENTITY,
            VALID_GEOMETRY_TYPES,
        )
        assert FORMAT_VERSION.startswith("0.9")
        assert len(VALID_GEOMETRY_TYPES) == 7


class TestExceptionHierarchy:
    """Verify exception hierarchy is correct."""

    def test_zvf_error_is_exception(self) -> None:
        from zarr_vectors.exceptions import ZVFError
        assert issubclass(ZVFError, Exception)

    def test_can_catch_specific(self) -> None:
        from zarr_vectors.exceptions import StoreError, ZVFError
        try:
            raise StoreError("test")
        except ZVFError:
            pass  # should be caught

    def test_specific_not_caught_by_sibling(self) -> None:
        from zarr_vectors.exceptions import StoreError, MetadataError
        try:
            raise StoreError("test")
        except MetadataError:
            assert False, "StoreError should not be caught by MetadataError"
        except StoreError:
            pass


class TestFixtures:
    """Verify test fixtures produce valid numpy arrays."""

    def test_simple_points_3d(self, simple_points_3d: dict) -> None:
        pos = simple_points_3d["positions"]
        assert pos.shape == (100, 3)
        assert pos.dtype == np.float32
        assert np.all(pos >= 0) and np.all(pos < 1000)

        intensity = simple_points_3d["vertex_attributes"]["intensity"]
        assert intensity.shape == (100,)
        assert intensity.dtype == np.float32

    def test_two_chunk_points(self, two_chunk_points: dict) -> None:
        pos = two_chunk_points["positions"]
        assert pos.shape == (50, 3)
        # First 25 should have x < 50, last 25 x >= 50
        assert np.all(pos[:25, 0] < 50)
        assert np.all(pos[25:, 0] >= 50)

    def test_simple_streamlines(self, simple_streamlines: dict) -> None:
        polylines = simple_streamlines["polylines"]
        assert len(polylines) == 10
        for pl in polylines:
            assert pl.ndim == 2
            assert pl.shape[1] == 3
            assert 20 <= pl.shape[0] <= 51

    def test_simple_skeleton(self, simple_skeleton: dict) -> None:
        pos = simple_skeleton["positions"]
        parents = simple_skeleton["parents"]
        assert pos.shape == (50, 3)
        assert parents.shape == (50,)
        assert parents[0] == -1  # root

        # All non-root parents should be valid indices
        for i in range(1, 50):
            assert 0 <= parents[i] < 50

    def test_simple_graph(self, simple_graph: dict) -> None:
        pos = simple_graph["positions"]
        edges = simple_graph["edges"]
        assert pos.shape == (20, 3)
        assert edges.ndim == 2
        assert edges.shape[1] == 2
        # No self-loops
        assert np.all(edges[:, 0] != edges[:, 1])

    def test_simple_mesh(self, simple_mesh: dict) -> None:
        verts = simple_mesh["vertices"]
        faces = simple_mesh["faces"]
        assert verts.ndim == 2
        assert verts.shape[1] == 3
        assert faces.ndim == 2
        assert faces.shape[1] == 3
        # All face indices should be valid vertex indices
        assert np.all(faces >= 0)
        assert np.all(faces < len(verts))
