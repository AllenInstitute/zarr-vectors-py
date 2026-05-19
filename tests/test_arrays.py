"""Step 04 tests: core array creation, write, and read round-trips."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.constants import LINK_FRAGMENTS, VERTEX_FRAGMENTS, VERTICES
from zarr_vectors.core.arrays import (
    count_fragments,
    create_attribute_array,
    create_cross_chunk_link_attributes_array,
    create_cross_chunk_links_array,
    create_groupings_array,
    create_groupings_attributes_array,
    create_link_attributes_array,
    create_links_array,
    create_object_attributes_array,
    create_object_index_array,
    create_vertices_array,
    list_chunk_keys,
    read_all_groupings,
    read_all_object_manifests,
    read_chunk_attributes,
    read_chunk_link_fragment,
    read_chunk_links,
    read_chunk_vertices,
    read_cross_chunk_link_attributes,
    read_cross_chunk_links,
    read_fragment,
    read_group_object_ids,
    read_groupings_attributes,
    read_link_fragment_index,
    read_object_attributes,
    read_object_manifest,
    read_object_vertices,
    read_vertex_fragment_index,
    write_chunk_attributes,
    write_chunk_fragments,
    write_chunk_link_attributes,
    write_chunk_links,
    write_chunk_vertices,
    write_cross_chunk_link_attributes,
    write_cross_chunk_links,
    write_groupings,
    write_groupings_attributes,
    write_object_attributes,
    write_object_index,
)
from zarr_vectors.core.store import FsGroup
from zarr_vectors.encoding.fragments import encode_fragments
from zarr_vectors.exceptions import ArrayError


def _make_level_group(tmp_path: Path, name: str = "0") -> FsGroup:
    root = FsGroup(tmp_path / "store.zarr", create=True)
    return root.create_group(name)


# ===================================================================
# Vertex write/read round-trips
# ===================================================================

class TestVertexArrays:

    def test_single_chunk_single_group(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)

        pts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        write_chunk_vertices(lg, (0, 0, 0), [pts])

        groups = read_chunk_vertices(lg, (0, 0, 0), dtype=np.float32, ndim=3)
        assert len(groups) == 1
        np.testing.assert_array_equal(groups[0], pts)

    def test_single_chunk_multiple_groups(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)

        g0 = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        g1 = np.array([[10, 10, 10]], dtype=np.float32)
        g2 = np.array([[20, 20, 20], [21, 21, 21], [22, 22, 22]], dtype=np.float32)
        write_chunk_vertices(lg, (0, 0, 0), [g0, g1, g2])

        groups = read_chunk_vertices(lg, (0, 0, 0), dtype=np.float32, ndim=3)
        assert len(groups) == 3
        np.testing.assert_array_equal(groups[0], g0)
        np.testing.assert_array_equal(groups[1], g1)
        np.testing.assert_array_equal(groups[2], g2)

    def test_multiple_chunks(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)

        pts_a = [np.array([[1, 2, 3]], dtype=np.float32)]
        pts_b = [np.array([[4, 5, 6], [7, 8, 9]], dtype=np.float32)]
        write_chunk_vertices(lg, (0, 0, 0), pts_a)
        write_chunk_vertices(lg, (1, 0, 0), pts_b)

        keys = list_chunk_keys(lg)
        assert keys == [(0, 0, 0), (1, 0, 0)]

        g_a = read_chunk_vertices(lg, (0, 0, 0), dtype=np.float32, ndim=3)
        g_b = read_chunk_vertices(lg, (1, 0, 0), dtype=np.float32, ndim=3)
        np.testing.assert_array_equal(g_a[0], pts_a[0])
        np.testing.assert_array_equal(g_b[0], pts_b[0])

    def test_read_single_fragment(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)

        g0 = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        g1 = np.array([[10, 10, 10], [11, 11, 11], [12, 12, 12]], dtype=np.float32)
        write_chunk_vertices(lg, (0, 0, 0), [g0, g1])

        fragment0 = read_fragment(lg, (0, 0, 0), 0, dtype=np.float32, ndim=3)
        fragment1 = read_fragment(lg, (0, 0, 0), 1, dtype=np.float32, ndim=3)
        np.testing.assert_array_equal(fragment0, g0)
        np.testing.assert_array_equal(fragment1, g1)

    def test_read_fragment_out_of_range(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        write_chunk_vertices(lg, (0, 0, 0), [np.zeros((1, 3), dtype=np.float32)])
        try:
            read_fragment(lg, (0, 0, 0), 5, dtype=np.float32, ndim=3)
            assert False, "Should raise"
        except ArrayError:
            pass

    def test_empty_group(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)

        g0 = np.array([[1, 2, 3]], dtype=np.float32)
        g_empty = np.zeros((0, 3), dtype=np.float32)
        g2 = np.array([[7, 8, 9]], dtype=np.float32)
        write_chunk_vertices(lg, (0, 0, 0), [g0, g_empty, g2])

        groups = read_chunk_vertices(lg, (0, 0, 0), dtype=np.float32, ndim=3)
        assert len(groups) == 3
        assert groups[1].shape == (0, 3)

    def test_count_fragments(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        write_chunk_vertices(lg, (0, 0, 0), [
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((5, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
        ])
        assert count_fragments(lg, (0, 0, 0)) == 3

    def test_2d_points(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg, dtype="float64")

        pts = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        write_chunk_vertices(lg, (0, 0), [pts], dtype=np.float64)

        groups = read_chunk_vertices(lg, (0, 0), dtype=np.float64, ndim=2)
        np.testing.assert_array_equal(groups[0], pts)


# ===================================================================
# Links write/read
# ===================================================================

class TestLinkArrays:

    def test_triangle_faces(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_links_array(lg, link_width=3)

        verts = [np.zeros((4, 3), dtype=np.float32)]
        write_chunk_vertices(lg, (0, 0, 0), verts)

        faces = [np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)]
        write_chunk_links(lg, (0, 0, 0), faces)

        read_faces = read_chunk_links(lg, (0, 0, 0), link_width=3)
        assert len(read_faces) == 1
        np.testing.assert_array_equal(read_faces[0], faces[0])

    def test_edge_list(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_links_array(lg, link_width=2)

        verts = [np.zeros((3, 3), dtype=np.float32)]
        write_chunk_vertices(lg, (0, 0, 0), verts)

        edges = [np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)]
        write_chunk_links(lg, (0, 0, 0), edges)

        read_edges = read_chunk_links(lg, (0, 0, 0), link_width=2)
        np.testing.assert_array_equal(read_edges[0], edges[0])

    def test_multiple_link_groups(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_links_array(lg, link_width=2)

        v0 = np.zeros((3, 3), dtype=np.float32)
        v1 = np.zeros((2, 3), dtype=np.float32)
        write_chunk_vertices(lg, (0, 0, 0), [v0, v1])

        e0 = np.array([[0, 1], [1, 2]], dtype=np.int64)
        e1 = np.array([[0, 1]], dtype=np.int64)
        write_chunk_links(lg, (0, 0, 0), [e0, e1])

        groups = read_chunk_links(lg, (0, 0, 0), link_width=2)
        assert len(groups) == 2
        np.testing.assert_array_equal(groups[0], e0)
        np.testing.assert_array_equal(groups[1], e1)


# ===================================================================
# Attribute write/read
# ===================================================================

class TestAttributeArrays:

    def test_scalar_attribute(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_attribute_array(lg, "intensity")

        verts = [np.zeros((3, 3), dtype=np.float32)]
        write_chunk_vertices(lg, (0, 0, 0), verts)

        intensity = [np.array([0.5, 0.8, 0.3], dtype=np.float32)]
        write_chunk_attributes(lg, "intensity", (0, 0, 0), intensity)

        read_back = read_chunk_attributes(lg, "intensity", (0, 0, 0),
                                          dtype=np.float32, ncols=1)
        assert len(read_back) == 1
        np.testing.assert_allclose(read_back[0], intensity[0])

    def test_multichannel_attribute(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_attribute_array(lg, "color", channel_names=["r", "g", "b"])

        verts = [np.zeros((2, 3), dtype=np.float32)]
        write_chunk_vertices(lg, (0, 0, 0), verts)

        color = [np.array([[255, 0, 0], [0, 255, 0]], dtype=np.float32)]
        write_chunk_attributes(lg, "color", (0, 0, 0), color, dtype=np.float32)

        read_back = read_chunk_attributes(lg, "color", (0, 0, 0),
                                          dtype=np.float32, ncols=3)
        np.testing.assert_array_equal(read_back[0], color[0])

    def test_multiple_attribute_groups(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_attribute_array(lg, "radius")

        v0 = np.zeros((2, 3), dtype=np.float32)
        v1 = np.zeros((3, 3), dtype=np.float32)
        write_chunk_vertices(lg, (0, 0, 0), [v0, v1])

        r0 = np.array([1.0, 2.0], dtype=np.float32)
        r1 = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        write_chunk_attributes(lg, "radius", (0, 0, 0), [r0, r1])

        read_back = read_chunk_attributes(lg, "radius", (0, 0, 0),
                                          dtype=np.float32, ncols=1)
        assert len(read_back) == 2
        np.testing.assert_array_equal(read_back[0], r0)
        np.testing.assert_array_equal(read_back[1], r1)


# ===================================================================
# Object index write/read
# ===================================================================

class TestObjectIndex:

    def test_basic_round_trip(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_object_index_array(lg)

        manifests = {
            0: [((0, 0, 0), 0), ((0, 0, 1), 2)],
            1: [((1, 1, 1), 0)],
            2: [((0, 0, 0), 1), ((0, 1, 0), 0)],
        }
        write_object_index(lg, manifests, sid_ndim=3)

        m0 = read_object_manifest(lg, 0)
        m1 = read_object_manifest(lg, 1)
        m2 = read_object_manifest(lg, 2)
        assert m0 == manifests[0]
        assert m1 == manifests[1]
        assert m2 == manifests[2]

    def test_read_all_manifests(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_object_index_array(lg)

        manifests = {
            0: [((0, 0, 0), 0)],
            1: [((1, 0, 0), 0), ((1, 0, 1), 0)],
        }
        write_object_index(lg, manifests, sid_ndim=3)

        all_m = read_all_object_manifests(lg)
        assert len(all_m) == 2
        assert all_m[0] == manifests[0]
        assert all_m[1] == manifests[1]

    def test_object_id_out_of_range(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_object_index_array(lg)
        write_object_index(lg, {0: [((0, 0, 0), 0)]}, sid_ndim=3)
        try:
            read_object_manifest(lg, 99)
            assert False
        except ArrayError:
            pass

    def test_read_object_vertices(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_object_index_array(lg)

        g0 = np.array([[1, 2, 3]], dtype=np.float32)
        g1 = np.array([[4, 5, 6]], dtype=np.float32)
        g2 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)
        write_chunk_vertices(lg, (0, 0, 0), [g0, g1])
        write_chunk_vertices(lg, (1, 0, 0), [g2])

        manifests = {
            0: [((0, 0, 0), 0), ((1, 0, 0), 0)],  # object spans 2 chunks
            1: [((0, 0, 0), 1)],                     # single chunk
        }
        write_object_index(lg, manifests, sid_ndim=3)

        verts_0 = read_object_vertices(lg, 0, dtype=np.float32, ndim=3)
        assert len(verts_0) == 2
        np.testing.assert_array_equal(verts_0[0], g0)
        np.testing.assert_array_equal(verts_0[1], g2)

        verts_1 = read_object_vertices(lg, 1, dtype=np.float32, ndim=3)
        assert len(verts_1) == 1
        np.testing.assert_array_equal(verts_1[0], g1)


# ===================================================================
# Object attributes write/read
# ===================================================================

class TestObjectAttributes:

    def test_1d_attribute(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        data = np.array([0.5, 0.8, 0.3], dtype=np.float32)
        write_object_attributes(lg, "score", data)
        read_back = read_object_attributes(lg, "score")
        np.testing.assert_array_equal(read_back, data)

    def test_2d_attribute(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
        write_object_attributes(lg, "termination", data)
        read_back = read_object_attributes(lg, "termination")
        np.testing.assert_array_equal(read_back, data)
        assert read_back.shape == (3, 2)


# ===================================================================
# Groupings write/read
# ===================================================================

class TestGroupings:

    def test_basic(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_groupings_array(lg)

        groups = {
            0: [0, 1, 5],
            1: [2, 3],
            2: [4],
        }
        write_groupings(lg, groups)

        g0 = read_group_object_ids(lg, 0)
        g1 = read_group_object_ids(lg, 1)
        g2 = read_group_object_ids(lg, 2)
        assert g0 == [0, 1, 5]
        assert g1 == [2, 3]
        assert g2 == [4]

    def test_read_all(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_groupings_array(lg)

        groups = {0: [10, 20], 1: [30]}
        write_groupings(lg, groups)

        all_g = read_all_groupings(lg)
        assert all_g == [[10, 20], [30]]

    def test_group_out_of_range(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_groupings_array(lg)
        write_groupings(lg, {0: [0]})
        try:
            read_group_object_ids(lg, 99)
            assert False
        except ArrayError:
            pass


# ===================================================================
# Groupings attributes
# ===================================================================

class TestGroupingsAttributes:

    def test_basic(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        data = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        write_groupings_attributes(lg, "mean_fa", data)
        read_back = read_groupings_attributes(lg, "mean_fa")
        np.testing.assert_array_equal(read_back, data)


# ===================================================================
# Cross-chunk links
# ===================================================================

class TestCrossChunkLinks:

    def test_basic_3d(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_cross_chunk_links_array(lg)

        links = [
            (((0, 0, 0), 4), ((0, 0, 1), 0)),
            (((0, 0, 0), 2), ((1, 0, 0), 1)),
        ]
        write_cross_chunk_links(lg, links, sid_ndim=3)

        read_back = read_cross_chunk_links(lg)
        assert len(read_back) == 2
        assert read_back[0] == links[0]
        assert read_back[1] == links[1]

    def test_2d(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_cross_chunk_links_array(lg)

        links = [
            (((0, 1), 3), ((1, 1), 0)),
        ]
        write_cross_chunk_links(lg, links, sid_ndim=2)

        read_back = read_cross_chunk_links(lg)
        assert read_back[0] == links[0]


# ===================================================================
# Link attributes
# ===================================================================

class TestLinkAttributes:

    def test_basic(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_links_array(lg, link_width=2)
        create_link_attributes_array(lg, "weight")

        verts = [np.zeros((3, 3), dtype=np.float32)]
        write_chunk_vertices(lg, (0, 0, 0), verts)

        edges = [np.array([[0, 1], [1, 2]], dtype=np.int64)]
        write_chunk_links(lg, (0, 0, 0), edges)

        weights = [np.array([0.5, 0.8], dtype=np.float32)]
        write_chunk_link_attributes(lg, "weight", (0, 0, 0), weights)

        # Read back via raw bytes (link_attributes use same encoding as fragments)
        # 0.4 multiscale layout: link_attributes/<name>/<delta>/<chunk_key>
        key = "0.0.0"
        raw = lg.read_bytes("link_attributes/weight/0", key)
        arr = np.frombuffer(raw, dtype=np.float32)
        np.testing.assert_allclose(arr, [0.5, 0.8])


# ===================================================================
# Integration: full object reconstruction
# ===================================================================

class TestObjectReconstruction:
    """Simulate a streamline spanning 2 chunks and verify full read path."""

    def test_streamline_across_chunks(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_object_index_array(lg)
        create_cross_chunk_links_array(lg)

        # Segment A in chunk (0,0,0): 3 points
        seg_a = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
        # Segment B in chunk (0,0,1): 2 points
        seg_b = np.array([[3, 3, 3], [4, 4, 4]], dtype=np.float32)

        write_chunk_vertices(lg, (0, 0, 0), [seg_a])
        write_chunk_vertices(lg, (0, 0, 1), [seg_b])

        # Object 0 = this streamline, spanning both chunks
        write_object_index(lg, {
            0: [((0, 0, 0), 0), ((0, 0, 1), 0)],
        }, sid_ndim=3)

        # Cross-chunk link: last vertex of seg_a → first vertex of seg_b
        write_cross_chunk_links(lg, [
            (((0, 0, 0), 2), ((0, 0, 1), 0)),
        ], sid_ndim=3)

        # Read back full object
        verts = read_object_vertices(lg, 0, dtype=np.float32, ndim=3)
        assert len(verts) == 2
        np.testing.assert_array_equal(verts[0], seg_a)
        np.testing.assert_array_equal(verts[1], seg_b)

        # Concatenate to get full streamline
        full = np.concatenate(verts, axis=0)
        assert full.shape == (5, 3)
        np.testing.assert_array_equal(full[0], [0, 0, 0])
        np.testing.assert_array_equal(full[4], [4, 4, 4])

        # Verify cross-chunk link
        ccl = read_cross_chunk_links(lg)
        assert len(ccl) == 1
        assert ccl[0] == (((0, 0, 0), 2), ((0, 0, 1), 0))

    def test_graph_multiple_objects_shared_chunks(self, tmp_path: Path) -> None:
        """Two objects (neurons) sharing the same spatial chunks."""
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_object_index_array(lg)

        # Chunk (0,0,0): fragment 0 = neuron A, group 1 = neuron B
        vg_a0 = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        vg_b0 = np.array([[5, 5, 5]], dtype=np.float32)
        write_chunk_vertices(lg, (0, 0, 0), [vg_a0, vg_b0])

        # Chunk (1,0,0): fragment 0 = neuron A, group 1 = neuron B
        vg_a1 = np.array([[10, 10, 10]], dtype=np.float32)
        vg_b1 = np.array([[15, 15, 15], [16, 16, 16]], dtype=np.float32)
        write_chunk_vertices(lg, (1, 0, 0), [vg_a1, vg_b1])

        write_object_index(lg, {
            0: [((0, 0, 0), 0), ((1, 0, 0), 0)],   # neuron A
            1: [((0, 0, 0), 1), ((1, 0, 0), 1)],   # neuron B
        }, sid_ndim=3)

        # Read neuron A
        verts_a = read_object_vertices(lg, 0, dtype=np.float32, ndim=3)
        full_a = np.concatenate(verts_a)
        assert full_a.shape == (3, 3)  # 2 + 1 vertices
        np.testing.assert_array_equal(full_a[0], [0, 0, 0])

        # Read neuron B
        verts_b = read_object_vertices(lg, 1, dtype=np.float32, ndim=3)
        full_b = np.concatenate(verts_b)
        assert full_b.shape == (3, 3)  # 1 + 2 vertices
        np.testing.assert_array_equal(full_b[0], [5, 5, 5])


# ===================================================================
# Explicit-mode fragments (CSR-backed, non-contiguous indices)
# ===================================================================

class TestExplicitFragments:
    """Readers must handle explicit fragments — not just contiguous ranges.

    The writers always emit range fragments today, so these tests lay down
    the raw chunk buffer via the regular writer and then overwrite the
    sibling fragment-index blob with an explicit-mode encoding.
    """

    def _write_vertex_explicit_index(
        self,
        lg: FsGroup,
        chunk_coords: tuple[int, ...],
        fragments: list,
    ) -> None:
        key = ".".join(str(c) for c in chunk_coords)
        lg.write_bytes(
            VERTEX_FRAGMENTS, key, encode_fragments(fragments),
        )

    def _write_link_explicit_index(
        self,
        lg: FsGroup,
        chunk_coords: tuple[int, ...],
        fragments: list,
    ) -> None:
        key = ".".join(str(c) for c in chunk_coords)
        lg.write_bytes(
            LINK_FRAGMENTS, key, encode_fragments(fragments),
        )

    # --- Vertices: single fragment via read_fragment -----------------

    def test_read_fragment_explicit(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)

        # Lay down 6 vertex rows via the writer (range fragments).
        buf = np.array(
            [[i, i, i] for i in range(6)], dtype=np.float32,
        )
        write_chunk_vertices(lg, (0, 0, 0), [buf])

        # Overwrite the fragment index with: range(0,2), explicit[3,5,4].
        self._write_vertex_explicit_index(
            lg, (0, 0, 0),
            [(0, 2), np.array([3, 5, 4], dtype=np.int64)],
        )

        # Range branch.
        frag0 = read_fragment(lg, (0, 0, 0), 0, dtype=np.float32, ndim=3)
        np.testing.assert_array_equal(frag0, buf[0:2])

        # Explicit branch.
        frag1 = read_fragment(lg, (0, 0, 0), 1, dtype=np.float32, ndim=3)
        np.testing.assert_array_equal(frag1, buf[[3, 5, 4]])

    # --- Vertices: whole chunk via read_chunk_vertices ---------------

    def test_read_chunk_vertices_mixed(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)

        buf = np.array(
            [[i, i, i] for i in range(6)], dtype=np.float32,
        )
        write_chunk_vertices(lg, (0, 0, 0), [buf])

        # Mixed: range, explicit, range.
        self._write_vertex_explicit_index(
            lg, (0, 0, 0),
            [
                (0, 2),
                np.array([3, 5, 4], dtype=np.int64),
                (5, 1),
            ],
        )

        groups = read_chunk_vertices(lg, (0, 0, 0), dtype=np.float32, ndim=3)
        assert len(groups) == 3
        np.testing.assert_array_equal(groups[0], buf[0:2])
        np.testing.assert_array_equal(groups[1], buf[[3, 5, 4]])
        np.testing.assert_array_equal(groups[2], buf[5:6])

    # --- Links: single fragment via read_chunk_link_fragment ---------

    def test_read_chunk_link_fragment_explicit(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_links_array(lg, link_width=2)

        # Need a single vertex fragment that covers all referenced rows so
        # the writer's vertex/link fragment-count check passes.
        write_chunk_vertices(
            lg, (0, 0, 0), [np.zeros((10, 3), dtype=np.float32)],
        )

        # Lay down 6 link rows in one fragment.
        links = np.array(
            [[i, i + 1] for i in range(6)], dtype=np.int64,
        )
        write_chunk_links(lg, (0, 0, 0), [links])

        # Overwrite link fragment index: range(0,2), explicit[3,5,4].
        self._write_link_explicit_index(
            lg, (0, 0, 0),
            [(0, 2), np.array([3, 5, 4], dtype=np.int64)],
        )

        # Range branch.
        f0 = read_chunk_link_fragment(lg, (0, 0, 0), 0, link_width=2)
        np.testing.assert_array_equal(f0, links[0:2])

        # Explicit branch.
        f1 = read_chunk_link_fragment(lg, (0, 0, 0), 1, link_width=2)
        np.testing.assert_array_equal(f1, links[[3, 5, 4]])

    def test_read_chunk_link_fragment_auto_link_width(
        self, tmp_path: Path,
    ) -> None:
        # link_width=None should auto-detect from array metadata.
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_links_array(lg, link_width=3)

        write_chunk_vertices(
            lg, (0, 0, 0), [np.zeros((5, 3), dtype=np.float32)],
        )
        links = np.array(
            [[i, i + 1, i + 2] for i in range(4)], dtype=np.int64,
        )
        write_chunk_links(lg, (0, 0, 0), [links])
        self._write_link_explicit_index(
            lg, (0, 0, 0),
            [np.array([2, 0, 3], dtype=np.int64)],
        )

        out = read_chunk_link_fragment(lg, (0, 0, 0), 0)
        np.testing.assert_array_equal(out, links[[2, 0, 3]])
        assert out.shape == (3, 3)

    # --- Links: whole chunk via read_chunk_links ---------------------

    def test_read_chunk_links_mixed(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_links_array(lg, link_width=2)

        write_chunk_vertices(
            lg, (0, 0, 0), [np.zeros((10, 3), dtype=np.float32)],
        )
        links = np.array(
            [[i, i + 1] for i in range(6)], dtype=np.int64,
        )
        write_chunk_links(lg, (0, 0, 0), [links])

        self._write_link_explicit_index(
            lg, (0, 0, 0),
            [
                (0, 2),
                np.array([3, 5, 4], dtype=np.int64),
                (5, 1),
            ],
        )

        groups = read_chunk_links(lg, (0, 0, 0), link_width=2)
        assert len(groups) == 3
        np.testing.assert_array_equal(groups[0], links[0:2])
        np.testing.assert_array_equal(groups[1], links[[3, 5, 4]])
        np.testing.assert_array_equal(groups[2], links[5:6])

    # --- Error parity ------------------------------------------------

    def test_read_chunk_link_fragment_out_of_range(
        self, tmp_path: Path,
    ) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_links_array(lg, link_width=2)
        write_chunk_vertices(
            lg, (0, 0, 0), [np.zeros((1, 3), dtype=np.float32)],
        )
        write_chunk_links(
            lg, (0, 0, 0), [np.array([[0, 0]], dtype=np.int64)],
        )
        try:
            read_chunk_link_fragment(lg, (0, 0, 0), 5, link_width=2)
            assert False, "Should raise"
        except ArrayError:
            pass


# ===================================================================
# R1: write_chunk_fragments — direct fragment-index writer + append
# ===================================================================

class TestWriteChunkFragments:
    """Covers the new public ``write_chunk_fragments`` (R1).

    Verifies both replace and append modes for vertex and link
    fragment-index blobs.
    """

    def test_replace_vertex(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        # Need a vertex buffer present so the resulting fragment-index is
        # interpretable, but write_chunk_fragments itself doesn't require it.
        write_chunk_vertices(
            lg, (0, 0, 0), [np.zeros((6, 3), dtype=np.float32)],
        )

        ret = write_chunk_fragments(
            lg, (0, 0, 0),
            [(0, 2), np.array([3, 5, 4], dtype=np.int64), (5, 1)],
            target="vertex",
            mode="replace",
        )
        assert ret == [0, 1, 2]

        fi = read_vertex_fragment_index(lg, (0, 0, 0))
        assert fi.num_fragments == 3
        assert fi.is_range(0)
        assert fi.range(0) == (0, 2)
        assert not fi.is_range(1)
        np.testing.assert_array_equal(fi.indices(1), [3, 5, 4])
        assert fi.is_range(2)
        assert fi.range(2) == (5, 1)

    def test_append_vertex(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)

        buf = np.array(
            [[i, i, i] for i in range(6)], dtype=np.float32,
        )
        # Range fragment 0 covers rows [0, 2).
        write_chunk_vertices(lg, (0, 0, 0), [buf[0:2]])

        # Rewrite the chunk vertex buffer to hold all 6 rows so the new
        # explicit indices reference real positions.
        lg.write_bytes(VERTICES, "0.0.0", buf.tobytes())

        ret = write_chunk_fragments(
            lg, (0, 0, 0),
            [np.array([1, 3, 2], dtype=np.int64)],
            target="vertex",
            mode="append",
        )
        assert ret == [1]

        fi = read_vertex_fragment_index(lg, (0, 0, 0))
        assert fi.num_fragments == 2
        assert fi.is_range(0)
        assert fi.range(0) == (0, 2)
        assert not fi.is_range(1)
        np.testing.assert_array_equal(fi.indices(1), [1, 3, 2])

    def test_append_link(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_links_array(lg, link_width=2)
        write_chunk_vertices(
            lg, (0, 0, 0), [np.zeros((5, 3), dtype=np.float32)],
        )
        write_chunk_links(
            lg, (0, 0, 0),
            [np.array([[0, 1], [1, 2]], dtype=np.int64)],
        )

        ret = write_chunk_fragments(
            lg, (0, 0, 0),
            [np.array([1, 0], dtype=np.int64)],
            target="link",
            mode="append",
        )
        assert ret == [1]

        fi = read_link_fragment_index(lg, (0, 0, 0))
        assert fi.num_fragments == 2
        assert fi.is_range(0)
        assert not fi.is_range(1)
        np.testing.assert_array_equal(fi.indices(1), [1, 0])

    def test_append_to_missing(self, tmp_path: Path) -> None:
        # Append to a chunk that has no fragment-index blob yet.
        lg = _make_level_group(tmp_path)
        # Note: we don't even call create_vertices_array; write_chunk_fragments
        # should just write the blob directly.
        ret = write_chunk_fragments(
            lg, (0, 0, 0),
            [(0, 3), np.array([2, 5], dtype=np.int64)],
            target="vertex",
            mode="append",
        )
        # First append to missing: existing=[], so new fragments are at 0,1.
        assert ret == [0, 1]

        fi = read_vertex_fragment_index(lg, (0, 0, 0))
        assert fi.num_fragments == 2
        assert fi.is_range(0)
        assert fi.range(0) == (0, 3)
        np.testing.assert_array_equal(fi.indices(1), [2, 5])

    def test_append_empty_noop(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        write_chunk_vertices(
            lg, (0, 0, 0), [np.zeros((2, 3), dtype=np.float32)],
        )
        before = lg.read_bytes(VERTEX_FRAGMENTS, "0.0.0")

        ret = write_chunk_fragments(
            lg, (0, 0, 0), [], target="vertex", mode="append",
        )
        assert ret == []

        after = lg.read_bytes(VERTEX_FRAGMENTS, "0.0.0")
        assert before == after

    def test_invalid_target(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        try:
            write_chunk_fragments(
                lg, (0, 0, 0), [(0, 1)], target="invalid",
            )
            assert False, "Should raise"
        except ArrayError:
            pass

    def test_invalid_mode(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        try:
            write_chunk_fragments(
                lg, (0, 0, 0), [(0, 1)], target="vertex", mode="invalid",
            )
            assert False, "Should raise"
        except ArrayError:
            pass


# ===================================================================
# R2: write_object_attributes append mode
# ===================================================================

class TestWriteObjectAttributesAppend:
    """Covers ``mode="append"`` on ``write_object_attributes`` (R2)."""

    def test_append_1d(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_object_attributes_array(lg, "labels", dtype="int32")
        write_object_attributes(
            lg, "labels",
            np.array([1, 2, 3], dtype=np.int32),
        )
        write_object_attributes(
            lg, "labels",
            np.array([4, 5], dtype=np.int32),
            mode="append",
        )
        back = read_object_attributes(lg, "labels")
        np.testing.assert_array_equal(back, [1, 2, 3, 4, 5])

    def test_append_2d(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_object_attributes_array(lg, "vecs", dtype="float32", num_channels=3)
        write_object_attributes(
            lg, "vecs",
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        )
        write_object_attributes(
            lg, "vecs",
            np.array([[7, 8, 9]], dtype=np.float32),
            mode="append",
        )
        back = read_object_attributes(lg, "vecs")
        assert back.shape == (3, 3)
        np.testing.assert_array_equal(
            back, [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        )

    def test_append_to_missing(self, tmp_path: Path) -> None:
        # First append acts like a replace when the attribute doesn't
        # exist yet.
        lg = _make_level_group(tmp_path)
        create_object_attributes_array(lg, "scores", dtype="int64")
        write_object_attributes(
            lg, "scores",
            np.array([10, 20], dtype=np.int64),
            mode="append",
        )
        back = read_object_attributes(lg, "scores")
        np.testing.assert_array_equal(back, [10, 20])

    def test_append_dtype_cast(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_object_attributes_array(lg, "vals", dtype="int32")
        write_object_attributes(
            lg, "vals",
            np.array([1, 2], dtype=np.int32),
        )
        # Append float32 values that round-trip safely to int32.
        write_object_attributes(
            lg, "vals",
            np.array([3.0, 4.0], dtype=np.float32),
            mode="append",
        )
        back = read_object_attributes(lg, "vals")
        assert back.dtype == np.int32
        np.testing.assert_array_equal(back, [1, 2, 3, 4])

    def test_append_shape_mismatch(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_object_attributes_array(lg, "vecs", dtype="float32", num_channels=2)
        write_object_attributes(
            lg, "vecs",
            np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        )
        # Tail-shape mismatch: existing (3, 2), append (2, 3).
        try:
            write_object_attributes(
                lg, "vecs",
                np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
                mode="append",
            )
            assert False, "Should raise"
        except ArrayError:
            pass

    def test_append_invalid_mode(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_object_attributes_array(lg, "x", dtype="int32")
        try:
            write_object_attributes(
                lg, "x",
                np.array([1], dtype=np.int32),
                mode="invalid",
            )
            assert False, "Should raise"
        except ArrayError:
            pass


# ===================================================================
# R3: write_cross_chunk_links / write_cross_chunk_link_attributes append
# ===================================================================

class TestWriteCrossChunkLinksAppend:
    """Covers ``mode="append"`` on cross-chunk-link writers (R3)."""

    def _records(self, n: int, base: int = 0) -> list:
        # Build n simple 2-endpoint records with distinct vertex indices.
        return [
            [((0, 0, 0), base + i), ((1, 0, 0), base + i + 100)]
            for i in range(n)
        ]

    def test_append_records(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_cross_chunk_links_array(lg, delta=0)

        ret0 = write_cross_chunk_links(
            lg, self._records(2), sid_ndim=3, delta=0,
        )
        assert ret0 == 0   # replace returns 0

        ret1 = write_cross_chunk_links(
            lg, self._records(1, base=10), sid_ndim=3,
            delta=0, mode="append",
        )
        assert ret1 == 2   # row index of first appended record

        records = read_cross_chunk_links(lg, delta=0)
        assert len(records) == 3
        assert records[0][0] == ((0, 0, 0), 0)
        assert records[2][0] == ((0, 0, 0), 10)

    def test_append_to_empty(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_cross_chunk_links_array(lg, delta=0)

        ret = write_cross_chunk_links(
            lg, self._records(2), sid_ndim=3, delta=0, mode="append",
        )
        assert ret == 0

        records = read_cross_chunk_links(lg, delta=0)
        assert len(records) == 2

    def test_append_attributes(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_cross_chunk_links_array(lg, delta=0)
        create_cross_chunk_link_attributes_array(
            lg, "weight", delta=0, dtype="float32",
        )

        write_cross_chunk_links(
            lg, self._records(2), sid_ndim=3, delta=0,
        )
        write_cross_chunk_link_attributes(
            lg, "weight", np.array([0.1, 0.2], dtype=np.float32),
            num_links=2, delta=0,
        )

        # Append one more record + its attribute.
        write_cross_chunk_links(
            lg, self._records(1, base=10), sid_ndim=3,
            delta=0, mode="append",
        )
        write_cross_chunk_link_attributes(
            lg, "weight", np.array([0.3], dtype=np.float32),
            num_links=3, delta=0, mode="append",
        )

        back = read_cross_chunk_link_attributes(lg, "weight", delta=0)
        assert back.shape == (3,)
        np.testing.assert_allclose(back, [0.1, 0.2, 0.3])

    def test_append_attributes_misaligned(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_cross_chunk_links_array(lg, delta=0)
        create_cross_chunk_link_attributes_array(
            lg, "weight", delta=0, dtype="float32",
        )
        write_cross_chunk_links(
            lg, self._records(2), sid_ndim=3, delta=0,
        )
        write_cross_chunk_link_attributes(
            lg, "weight", np.array([0.1, 0.2], dtype=np.float32),
            num_links=2, delta=0,
        )
        try:
            # Append one row, but pass num_links=5 (does not match
            # post-append length of 3).
            write_cross_chunk_link_attributes(
                lg, "weight", np.array([0.3], dtype=np.float32),
                num_links=5, delta=0, mode="append",
            )
            assert False, "Should raise"
        except ArrayError:
            pass

    def test_append_invalid_mode(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_cross_chunk_links_array(lg, delta=0)
        try:
            write_cross_chunk_links(
                lg, self._records(1), sid_ndim=3, delta=0, mode="invalid",
            )
            assert False, "Should raise"
        except ArrayError:
            pass


# ===================================================================
# R4: Idempotent create_*_array helpers
# ===================================================================

class TestCreateArraysIdempotent:
    """Covers ``exist_ok=True`` (default) and ``exist_ok=False`` on the
    ``create_*_array`` family (R4)."""

    def test_vertices_idempotent(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        # Second call must be a no-op (default exist_ok=True).
        create_vertices_array(lg)

    def test_vertices_strict(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        try:
            create_vertices_array(lg, exist_ok=False)
            assert False, "Should raise"
        except ArrayError:
            pass

    def test_links_idempotent(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_links_array(lg, link_width=2)
        create_links_array(lg, link_width=2)   # no-op

    def test_links_strict(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_links_array(lg, link_width=2)
        try:
            create_links_array(lg, link_width=2, exist_ok=False)
            assert False, "Should raise"
        except ArrayError:
            pass

    def test_object_attributes_idempotent(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_object_attributes_array(lg, "labels", dtype="int32")
        # Second call must not error.
        create_object_attributes_array(lg, "labels", dtype="int32")

    def test_object_attributes_strict(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_object_attributes_array(lg, "labels", dtype="int32")
        try:
            create_object_attributes_array(
                lg, "labels", dtype="int32", exist_ok=False,
            )
            assert False, "Should raise"
        except ArrayError:
            pass


# ===================================================================
# R5: default kwarg on read_fragment and read_chunk_link_fragment
# ===================================================================

class TestReadFragmentDefault:
    """Covers ``default=...`` soft-fail on single-fragment readers (R5)."""

    # --- read_fragment ----------------------------------------------

    def test_read_fragment_default_on_missing_chunk(
        self, tmp_path: Path,
    ) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        # Chunk doesn't exist — soft-fail returns the default.
        out = read_fragment(
            lg, (99, 99, 99), 0,
            dtype=np.float32, ndim=3,
            default=None,
        )
        assert out is None

    def test_read_fragment_default_on_out_of_range(
        self, tmp_path: Path,
    ) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        write_chunk_vertices(
            lg, (0, 0, 0), [np.zeros((1, 3), dtype=np.float32)],
        )
        sentinel = np.empty((0, 3), dtype=np.float32)
        out = read_fragment(
            lg, (0, 0, 0), 5,
            dtype=np.float32, ndim=3,
            default=sentinel,
        )
        assert out is sentinel

    def test_read_fragment_no_default_still_raises(
        self, tmp_path: Path,
    ) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        write_chunk_vertices(
            lg, (0, 0, 0), [np.zeros((1, 3), dtype=np.float32)],
        )
        try:
            read_fragment(lg, (0, 0, 0), 5, dtype=np.float32, ndim=3)
            assert False, "Should raise"
        except ArrayError:
            pass

    # --- read_chunk_link_fragment -----------------------------------

    def test_link_fragment_default_on_missing_chunk(
        self, tmp_path: Path,
    ) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_links_array(lg, link_width=2)
        out = read_chunk_link_fragment(
            lg, (99, 99, 99), 0, link_width=2, default=None,
        )
        assert out is None

    def test_link_fragment_default_on_out_of_range(
        self, tmp_path: Path,
    ) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_links_array(lg, link_width=2)
        write_chunk_vertices(
            lg, (0, 0, 0), [np.zeros((1, 3), dtype=np.float32)],
        )
        write_chunk_links(
            lg, (0, 0, 0), [np.array([[0, 0]], dtype=np.int64)],
        )
        sentinel = np.empty((0, 2), dtype=np.int64)
        out = read_chunk_link_fragment(
            lg, (0, 0, 0), 5, link_width=2, default=sentinel,
        )
        assert out is sentinel

    def test_link_fragment_no_default_still_raises(
        self, tmp_path: Path,
    ) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_links_array(lg, link_width=2)
        write_chunk_vertices(
            lg, (0, 0, 0), [np.zeros((1, 3), dtype=np.float32)],
        )
        write_chunk_links(
            lg, (0, 0, 0), [np.array([[0, 0]], dtype=np.int64)],
        )
        try:
            read_chunk_link_fragment(
                lg, (0, 0, 0), 5, link_width=2,
            )
            assert False, "Should raise"
        except ArrayError:
            pass
