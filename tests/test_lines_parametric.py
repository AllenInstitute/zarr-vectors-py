"""Step 07 tests: finite lines and parametric objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.types.lines import write_lines, read_lines
from zarr_vectors.types.parametric import write_parametric_objects, read_parametric_objects
from zarr_vectors.exceptions import ArrayError


# ===================================================================
# Finite lines
# ===================================================================

class TestLinesSingleChunk:

    def test_single_line(self, tmp_path: Path) -> None:
        endpoints = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.float32)
        store = str(tmp_path / "line.zarrvectors")

        summary = write_lines(store, endpoints, chunk_shape=(100.0, 100.0, 100.0))
        assert summary["line_count"] == 1
        assert summary["cross_chunk_count"] == 0

        result = read_lines(store)
        assert result["line_count"] == 1
        np.testing.assert_allclose(result["endpoints"][0, 0], [10, 20, 30], atol=1e-5)
        np.testing.assert_allclose(result["endpoints"][0, 1], [40, 50, 60], atol=1e-5)

    def test_multiple_lines_same_chunk(self, tmp_path: Path) -> None:
        endpoints = np.array([
            [[10, 10, 10], [20, 20, 20]],
            [[30, 30, 30], [40, 40, 40]],
            [[50, 50, 50], [60, 60, 60]],
        ], dtype=np.float32)
        store = str(tmp_path / "lines.zarrvectors")

        summary = write_lines(store, endpoints, chunk_shape=(100.0, 100.0, 100.0))
        assert summary["line_count"] == 3
        assert summary["chunk_count"] == 1
        assert summary["cross_chunk_count"] == 0

        result = read_lines(store)
        assert result["line_count"] == 3

    def test_read_by_object_id(self, tmp_path: Path) -> None:
        endpoints = np.array([
            [[10, 10, 10], [20, 20, 20]],
            [[30, 30, 30], [40, 40, 40]],
        ], dtype=np.float32)
        store = str(tmp_path / "byid.zarrvectors")

        write_lines(store, endpoints, chunk_shape=(100.0, 100.0, 100.0))

        result = read_lines(store, object_ids=[1])
        assert result["line_count"] == 1
        np.testing.assert_allclose(result["endpoints"][0, 0], [30, 30, 30], atol=1e-5)


class TestLinesCrossChunk:

    def test_line_crossing_boundary(self, tmp_path: Path) -> None:
        # Point A in chunk (0,0,0), point B in chunk (1,0,0)
        endpoints = np.array([[[10, 50, 50], [110, 50, 50]]], dtype=np.float32)
        store = str(tmp_path / "cross.zarrvectors")

        summary = write_lines(store, endpoints, chunk_shape=(100.0, 100.0, 100.0))
        assert summary["line_count"] == 1
        assert summary["cross_chunk_count"] == 1
        assert summary["chunk_count"] == 2

        result = read_lines(store)
        assert result["line_count"] == 1
        ep = result["endpoints"][0]
        np.testing.assert_allclose(ep[0], [10, 50, 50], atol=1e-5)
        np.testing.assert_allclose(ep[1], [110, 50, 50], atol=1e-5)

    def test_mixed_same_and_cross(self, tmp_path: Path) -> None:
        endpoints = np.array([
            [[10, 50, 50], [20, 50, 50]],     # same chunk
            [[10, 50, 50], [110, 50, 50]],     # cross chunk
            [[50, 50, 50], [60, 50, 50]],      # same chunk
        ], dtype=np.float32)
        store = str(tmp_path / "mixed.zarrvectors")

        summary = write_lines(store, endpoints, chunk_shape=(100.0, 100.0, 100.0))
        assert summary["line_count"] == 3
        assert summary["cross_chunk_count"] == 1

        result = read_lines(store)
        assert result["line_count"] == 3


class TestLinesBbox:

    def test_bbox_filter(self, tmp_path: Path) -> None:
        endpoints = np.array([
            [[10, 10, 10], [20, 20, 20]],      # inside
            [[50, 50, 50], [60, 60, 60]],       # inside
            [[200, 200, 200], [210, 210, 210]],  # outside
        ], dtype=np.float32)
        store = str(tmp_path / "bbox.zarrvectors")

        write_lines(store, endpoints, chunk_shape=(300.0, 300.0, 300.0))

        result = read_lines(
            store,
            bbox=(np.array([0, 0, 0]), np.array([100, 100, 100])),
        )
        assert result["line_count"] == 2


class TestLinesAttributes:

    def test_line_attributes(self, tmp_path: Path) -> None:
        endpoints = np.array([
            [[10, 10, 10], [20, 20, 20]],
            [[30, 30, 30], [40, 40, 40]],
        ], dtype=np.float32)
        lengths = np.array([17.32, 17.32], dtype=np.float32)
        store = str(tmp_path / "attrs.zarrvectors")

        write_lines(
            store, endpoints,
            chunk_shape=(100.0, 100.0, 100.0),
            object_attributes={"length": lengths},
        )

        # Verify store was created (attribute reading tested in full integration)
        result = read_lines(store)
        assert result["line_count"] == 2

    def test_per_endpoint_attributes_round_trip(self, tmp_path: Path) -> None:
        """``vertex_attributes`` were accepted and then silently dropped.

        ``write_lines`` imported ``create_attribute_array`` and
        ``write_chunk_attributes`` and called neither, so two values per
        line went in and nothing came back out -- with no warning and no
        ``vertex_attributes/`` directory on disk.
        """
        endpoints = np.array([
            [[10, 10, 10], [20, 20, 20]],
            [[30, 30, 30], [40, 40, 40]],
        ], dtype=np.float32)
        width = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        colour = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
        store = str(tmp_path / "vattrs.zarrvectors")

        write_lines(
            store, endpoints,
            chunk_shape=(100.0, 100.0, 100.0),
            vertex_attributes={"width": width, "colour": colour},
        )
        result = read_lines(store)

        # One row per endpoint, in the order ``endpoints`` flattens.
        np.testing.assert_allclose(
            result["vertex_attributes"]["width"], width.reshape(-1),
        )
        np.testing.assert_allclose(
            result["vertex_attributes"]["colour"], colour.reshape(-1, 3),
        )

    def test_attributes_follow_a_line_split_across_chunks(
        self, tmp_path: Path,
    ) -> None:
        """A split line puts one endpoint in each chunk; so must its rows.

        The two endpoints become single-vertex fragments in different
        chunks, so an attribute column assembled level-by-level rather
        than per object would pair each value with the wrong endpoint --
        and, because the counts still match, would do it silently.
        """
        endpoints = np.array([
            [[10, 10, 10], [310, 10, 10]],
            [[320, 20, 20], [20, 20, 20]],
        ], dtype=np.float32)
        width = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        store = str(tmp_path / "split.zarrvectors")

        summary = write_lines(
            store, endpoints,
            chunk_shape=(200.0, 200.0, 200.0),
            vertex_attributes={"width": width},
        )
        assert summary["cross_chunk_count"] == 2

        result = read_lines(store)
        np.testing.assert_allclose(result["endpoints"], endpoints)
        np.testing.assert_allclose(
            result["vertex_attributes"]["width"], width.reshape(-1),
        )

    def test_attributes_survive_a_bbox_filter(self, tmp_path: Path) -> None:
        """The bbox drops whole lines, so it must drop two rows per line."""
        endpoints = np.array([
            [[10, 10, 10], [20, 20, 20]],
            [[80, 80, 80], [90, 90, 90]],
        ], dtype=np.float32)
        width = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        store = str(tmp_path / "bbox.zarrvectors")

        write_lines(
            store, endpoints,
            chunk_shape=(100.0, 100.0, 100.0),
            vertex_attributes={"width": width},
        )
        result = read_lines(
            store, bbox=(np.array([0, 0, 0]), np.array([25, 25, 25])),
        )
        assert result["line_count"] == 1
        np.testing.assert_allclose(
            result["vertex_attributes"]["width"], [1.0, 2.0],
        )

    def test_per_endpoint_attributes_reach_the_api(self, tmp_path: Path) -> None:
        """The data surface sees them, and knows it looked.

        ``ReadResult.attributes_read`` has to be True even when the store
        carries none, or a caller cannot tell "no attributes" from "not
        attempted" -- which is the distinction the flag exists for.
        """
        import zarr_vectors as zv

        endpoints = np.array([
            [[10, 10, 10], [310, 10, 10]],
        ], dtype=np.float32)
        width = np.array([[1.0, 2.0]], dtype=np.float32)
        store = str(tmp_path / "api.zarrvectors")
        write_lines(
            store, endpoints,
            chunk_shape=(200.0, 200.0, 200.0),
            vertex_attributes={"width": width},
        )

        level = zv.open(store).level(0)
        assert level.attribute_names("vertex") == ("width",)

        result = level.read()
        assert result.attributes_read is True
        np.testing.assert_allclose(result.attributes["width"], [1.0, 2.0])
        np.testing.assert_allclose(
            result.positions, endpoints.reshape(-1, 3),
        )

        bare = str(tmp_path / "bare.zarrvectors")
        write_lines(bare, endpoints, chunk_shape=(200.0, 200.0, 200.0))
        empty = zv.open(bare).level(0).read()
        assert empty.attributes_read is True
        assert empty.attributes.names() == ()

    def test_wrong_per_endpoint_shape_is_rejected(self, tmp_path: Path) -> None:
        """Rejected before the store exists, not after."""
        endpoints = np.array([
            [[10, 10, 10], [20, 20, 20]],
        ], dtype=np.float32)
        store = tmp_path / "wrong.zarrvectors"
        try:
            write_lines(
                str(store), endpoints,
                chunk_shape=(100.0, 100.0, 100.0),
                # One value per line, not one per endpoint.
                vertex_attributes={"width": np.array([1.0], dtype=np.float32)},
            )
            assert False, "expected ArrayError"
        except ArrayError as e:
            assert "(N, 2)" in str(e)
        assert not store.exists()


class TestLinesEdgeCases:

    def test_bad_shape_raises(self, tmp_path: Path) -> None:
        bad = np.zeros((5, 3), dtype=np.float32)  # not (N, 2, D)
        try:
            write_lines(str(tmp_path / "bad.zarrvectors"), bad, chunk_shape=(100.0, 100.0, 100.0))
            assert False
        except ArrayError:
            pass

    def test_2d_lines(self, tmp_path: Path) -> None:
        endpoints = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
        store = str(tmp_path / "2d.zarrvectors")

        write_lines(store, endpoints, chunk_shape=(10.0, 10.0))
        result = read_lines(store)
        assert result["line_count"] == 1
        assert result["endpoints"].shape == (1, 2, 2)


# ===================================================================
# Parametric objects
# ===================================================================

class TestParametricPlane:

    def test_single_plane(self, tmp_path: Path) -> None:
        store = str(tmp_path / "plane.zarrvectors")

        summary = write_parametric_objects(
            store,
            [{"type": "plane", "coefficients": [0, 1, 0, -50], "name": "coronal_y50"}],
            create_new_store=True,
        )
        assert summary["object_count"] == 1
        assert summary["type_counts"]["plane"] == 1

        objects = read_parametric_objects(store)
        assert len(objects) == 1
        assert objects[0]["type"] == "plane"
        assert objects[0]["coefficients"] == [0.0, 1.0, 0.0, -50.0]
        assert objects[0]["coefficient_names"] == ["A", "B", "C", "D"]
        assert objects[0]["name"] == "coronal_y50"


class TestParametricLine:

    def test_single_line(self, tmp_path: Path) -> None:
        store = str(tmp_path / "pline.zarrvectors")

        summary = write_parametric_objects(
            store,
            [{"type": "line", "coefficients": [10, 20, 30, 0, 0, 1], "name": "z_probe"}],
            create_new_store=True,
        )
        assert summary["object_count"] == 1

        objects = read_parametric_objects(store)
        assert objects[0]["type"] == "line"
        assert objects[0]["coefficients"] == [10.0, 20.0, 30.0, 0.0, 0.0, 1.0]
        assert objects[0]["coefficient_names"] == ["x0", "y0", "z0", "dx", "dy", "dz"]


class TestParametricSphere:

    def test_single_sphere(self, tmp_path: Path) -> None:
        store = str(tmp_path / "sphere.zarrvectors")

        summary = write_parametric_objects(
            store,
            [{"type": "sphere", "coefficients": [0, 0, 0, 100], "name": "bounding_sphere"}],
            create_new_store=True,
        )
        assert summary["object_count"] == 1

        objects = read_parametric_objects(store)
        assert objects[0]["type"] == "sphere"
        assert objects[0]["coefficients"] == [0.0, 0.0, 0.0, 100.0]


class TestParametricMultiple:

    def test_mixed_types(self, tmp_path: Path) -> None:
        store = str(tmp_path / "mixed.zarrvectors")

        summary = write_parametric_objects(
            store,
            [
                {"type": "plane", "coefficients": [0, 1, 0, -50]},
                {"type": "line", "coefficients": [10, 20, 30, 0, 0, 1]},
                {"type": "sphere", "coefficients": [0, 0, 0, 100]},
                {"type": "plane", "coefficients": [1, 0, 0, -25]},
            ],
            create_new_store=True,
        )
        assert summary["object_count"] == 4
        assert summary["type_counts"]["plane"] == 2
        assert summary["type_counts"]["line"] == 1
        assert summary["type_counts"]["sphere"] == 1

        objects = read_parametric_objects(store)
        assert len(objects) == 4
        assert objects[0]["type"] == "plane"
        assert objects[1]["type"] == "line"
        assert objects[2]["type"] == "sphere"
        assert objects[3]["type"] == "plane"

    def test_type_by_id(self, tmp_path: Path) -> None:
        store = str(tmp_path / "byid.zarrvectors")

        write_parametric_objects(
            store,
            [{"type": 0, "coefficients": [0, 0, 1, -10]}],  # plane by ID
            create_new_store=True,
        )
        objects = read_parametric_objects(store)
        assert objects[0]["type"] == "plane"


class TestParametricAttributes:

    def test_with_object_attributes(self, tmp_path: Path) -> None:
        store = str(tmp_path / "attrs.zarrvectors")

        write_parametric_objects(
            store,
            [
                {"type": "plane", "coefficients": [0, 1, 0, -50]},
                {"type": "plane", "coefficients": [1, 0, 0, -25]},
            ],
            object_attributes={
                "opacity": np.array([0.5, 0.8], dtype=np.float32),
            },
            create_new_store=True,
        )
        # Verify store was created with attributes
        objects = read_parametric_objects(store)
        assert len(objects) == 2

    def test_with_groups(self, tmp_path: Path) -> None:
        store = str(tmp_path / "grouped.zarrvectors")

        write_parametric_objects(
            store,
            [
                {"type": "plane", "coefficients": [0, 1, 0, -50]},
                {"type": "plane", "coefficients": [1, 0, 0, -25]},
                {"type": "plane", "coefficients": [0, 0, 1, -75]},
            ],
            groups={
                0: [0, 1],   # anatomical planes
                1: [2],      # clipping plane
            },
            create_new_store=True,
        )
        objects = read_parametric_objects(store)
        assert len(objects) == 3


class TestParametricErrors:

    def test_unknown_type_name(self, tmp_path: Path) -> None:
        store = str(tmp_path / "bad.zarrvectors")
        try:
            write_parametric_objects(
                store,
                [{"type": "torus", "coefficients": [1, 2, 3]}],
                create_new_store=True,
            )
            assert False
        except ArrayError:
            pass

    def test_wrong_coefficient_count(self, tmp_path: Path) -> None:
        store = str(tmp_path / "bad2.zarrvectors")
        try:
            write_parametric_objects(
                store,
                [{"type": "plane", "coefficients": [1, 2]}],  # needs 4
                create_new_store=True,
            )
            assert False
        except ArrayError:
            pass

    def test_empty_store(self, tmp_path: Path) -> None:
        store = str(tmp_path / "empty.zarrvectors")
        write_parametric_objects(store, [], create_new_store=True)
        objects = read_parametric_objects(store)
        assert objects == []
