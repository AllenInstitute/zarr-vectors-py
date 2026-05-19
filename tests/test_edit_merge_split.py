"""Iteration-3 tests: deterministic merge/split via ``update_objects``
plus the public ``split_fragment`` primitive.

The 12 verification items from the approved plan are organised into
four test classes below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.core.arrays import (
    read_chunk_links,
    read_chunk_vertices,
    read_object_manifest,
    read_object_vertices,
)
from zarr_vectors.core.store import open_store
from zarr_vectors.exceptions import EditError
from zarr_vectors.ops import (
    EditSession,
    FragmentRef,
    LinkRef,
    VertexRef,
    add_link,
    edit_link,
    remove_link,
    split_fragment,
)
from zarr_vectors.types.graphs import write_graph
from zarr_vectors.types.points import write_points


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def two_object_graph(tmp_path: Path) -> str:
    """Two 3-vertex chains stored as a single explicit graph.

    Object 0: vertices 0,1,2 connected 0-1-2.
    Object 1: vertices 3,4,5 connected 3-4-5.
    All in one chunk.  Using explicit convention so split tests work
    without a materialise step.
    """
    path = tmp_path / "store.zv"
    positions = np.array(
        [
            [10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0],
            [30.0, 30.0, 30.0],
            [60.0, 60.0, 60.0],
            [70.0, 70.0, 70.0],
            [80.0, 80.0, 80.0],
        ],
        dtype=np.float32,
    )
    edges = np.array(
        [[0, 1], [1, 2], [3, 4], [4, 5]], dtype=np.int64,
    )
    write_graph(
        str(path), positions, edges,
        chunk_shape=(200.0, 200.0, 200.0),
        bounds=([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]),
        kind="graph",  # explicit links convention
        object_ids=np.array([0, 0, 0, 1, 1, 1], dtype=np.int64),
    )
    return str(path)


@pytest.fixture
def fine_grained_chain(tmp_path: Path) -> str:
    """Single explicit object 0 with 6 vertices spread across chunks
    (one fragment per chunk).  Chain 0-1-2-3-4-5.

    All edges connect across chunks → land in cross_chunk_links/0/data.
    This is the inter-fragment split case: removing one such link should
    cleanly halve the manifest with no chunk rewrite.
    """
    path = tmp_path / "store.zv"
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
    edges = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.int64,
    )
    write_graph(
        str(path), positions, edges,
        chunk_shape=(50.0, 50.0, 50.0),
        bounds=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
        kind="graph",
        object_ids=np.zeros(6, dtype=np.int64),
    )
    return str(path)


@pytest.fixture
def single_fragment_chain(tmp_path: Path) -> str:
    """Chain 0-1-2-3-4-5 under object 0 in one chunk → one fragment
    containing all 6 rows.  Intra-fragment split case."""
    path = tmp_path / "store.zv"
    positions = np.array(
        [
            [10.0, 10.0, 10.0],
            [11.0, 11.0, 11.0],
            [12.0, 12.0, 12.0],
            [13.0, 13.0, 13.0],
            [14.0, 14.0, 14.0],
            [15.0, 15.0, 15.0],
        ],
        dtype=np.float32,
    )
    edges = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.int64,
    )
    write_graph(
        str(path), positions, edges,
        chunk_shape=(200.0, 200.0, 200.0),
        bounds=([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]),
        kind="graph",
        object_ids=np.zeros(6, dtype=np.int64),
    )
    return str(path)


@pytest.fixture
def skeleton_chain(tmp_path: Path) -> str:
    """3-vertex skeleton (implicit_sequential_with_branches)."""
    path = tmp_path / "store.zv"
    positions = np.array(
        [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0], [30.0, 30.0, 30.0]],
        dtype=np.float32,
    )
    edges = np.array([[1, 0], [2, 1]], dtype=np.int64)
    write_graph(
        str(path), positions, edges,
        chunk_shape=(200.0, 200.0, 200.0),
        bounds=([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]),
        kind="skeleton",
    )
    return str(path)


# =====================================================================
# Helpers
# =====================================================================

def _find_link_row(
    root,
    chunk,
    *,
    endpoints: set[int],
) -> LinkRef | None:
    """Return the LinkRef of the first row whose two endpoints (as a set)
    match ``endpoints``.  Searches every fragment in ``chunk``.  Returns
    ``None`` if not found (the edge may be a cross-chunk link)."""
    try:
        groups = read_chunk_links(root["0"], chunk, delta=0)
    except Exception:
        return None
    for fi, g in enumerate(groups):
        for ri in range(g.shape[0]):
            if {int(g[ri, 0]), int(g[ri, 1])} == endpoints:
                return LinkRef(
                    level=0, chunk=tuple(chunk), fragment=fi,
                    row=ri, delta=0,
                )
    return None


# =====================================================================
# Item 1 & 9: pass-through default + same-OID no-op
# =====================================================================

class TestPassThroughDefault:

    def test_add_link_default_leaves_oids(
        self, two_object_graph: str,
    ) -> None:
        """Item 1: add_link(update_objects=False) does not change manifests."""
        root = open_store(two_object_graph, mode="r+")
        m0_before = read_object_manifest(root["0"], 0)
        m1_before = read_object_manifest(root["0"], 1)
        # Bridge OID 0's vertex 2 and OID 1's vertex 3.  Need to write
        # into a link fragment slot — pick fragment 0.
        add_link(
            root, level=0, src=2, dst=3, chunk=(0, 0, 0),
            fragment=0,
        )
        assert read_object_manifest(root["0"], 0) == m0_before
        assert read_object_manifest(root["0"], 1) == m1_before

    def test_add_link_same_oid_kwarg_is_noop(
        self, two_object_graph: str,
    ) -> None:
        """Item 9: add_link(update_objects=True) inside one OID does
        not allocate a new OID."""
        root = open_store(two_object_graph, mode="r+")
        # OID 0 already has vertices 0-1-2; add edge 0-2 within the
        # same OID.  Place into fragment 0 (the link group corresponding
        # to OID 0's fragment).
        _, report = add_link(
            root, level=0, src=0, dst=2, chunk=(0, 0, 0), fragment=0,
            update_objects=True,
        )
        assert report.oid_remap == {}, (
            f"unexpected remap on same-OID add_link: {report.oid_remap}"
        )


# =====================================================================
# Items 2-3: merge via add_link(update_objects=True)
# =====================================================================

class TestMergeViaAddLink:

    def test_merge_atomic_appends_new_oid(
        self, two_object_graph: str,
    ) -> None:
        """Item 2: under atomic=True, merging via add_link allocates a
        new OID with concat(A, B) manifest; A and B keep their own."""
        root = open_store(two_object_graph, mode="r+")
        m0 = read_object_manifest(root["0"], 0)
        m1 = read_object_manifest(root["0"], 1)
        _, report = add_link(
            root, level=0, src=2, dst=3, chunk=(0, 0, 0),
            fragment=0,
            atomic=True, update_objects=True,
        )
        # Two source OIDs remap to one new OID.
        assert 0 in report.oid_remap
        assert 1 in report.oid_remap
        new_oid = report.oid_remap[0]
        assert report.oid_remap[1] == new_oid
        # Old manifests intact.
        assert read_object_manifest(root["0"], 0) == m0
        assert read_object_manifest(root["0"], 1) == m1
        # Merged manifest is the concatenation.
        merged = read_object_manifest(root["0"], new_oid)
        assert merged == m0 + m1

    def test_merge_nonatomic_overwrites_first(
        self, two_object_graph: str,
    ) -> None:
        """Item 3: under atomic=False, oids[0]'s manifest is the concat
        and oids[1]'s manifest is emptied in place."""
        root = open_store(two_object_graph, mode="r+")
        m0 = read_object_manifest(root["0"], 0)
        m1 = read_object_manifest(root["0"], 1)
        _, report = add_link(
            root, level=0, src=2, dst=3, chunk=(0, 0, 0),
            fragment=0,
            atomic=False, update_objects=True,
        )
        assert report.oid_remap == {}
        assert read_object_manifest(root["0"], 0) == m0 + m1
        assert read_object_manifest(root["0"], 1) == []


# =====================================================================
# Items 4-5-6: deterministic split via remove_link(update_objects=True)
# =====================================================================

class TestSplitInterFragment:

    def test_split_inter_fragment_atomic(
        self, single_fragment_chain: str,
    ) -> None:
        """Item 4 (inter-fragment via add_link sibling fragments): build
        a chain where the removed edge is the bridge between two
        distinct fragments.  We construct this manually: split the
        single fragment first, then add the bridging edge, then remove
        it.
        """
        root = open_store(single_fragment_chain, mode="r+")
        # Pre-split into two fragments {0,1,2} and {3,4,5}.
        new_refs, _ = split_fragment(
            root,
            FragmentRef(level=0, chunk=(0, 0, 0), fragment=0),
            row_partition=[np.array([0, 1, 2]), np.array([3, 4, 5])],
            atomic=False,
        )
        f_lo, f_hi = new_refs[0].fragment, new_refs[1].fragment

        # Add the bridging edge between row 2 (last of lo) and row 3
        # (first of hi).  chunk-local indices after split: rows are
        # appended after the (empty) original fragment, so the new
        # fragments' rows live at positions according to their order
        # in vertex_groups.
        verts = read_chunk_vertices(
            root["0"], (0, 0, 0), dtype=np.float32, ndim=3,
        )
        # Reconstruct chunk-local indices by walking vertex_groups.
        # f_lo has 3 rows starting at <some_offset>; we want the third.
        offsets: dict[int, int] = {}
        cur = 0
        for fi, g in enumerate(verts):
            offsets[fi] = cur
            cur += int(g.shape[0])

        last_lo_local = offsets[f_lo] + 2
        first_hi_local = offsets[f_hi] + 0
        link_ref, _ = add_link(
            root, level=0,
            src=last_lo_local, dst=first_hi_local,
            chunk=(0, 0, 0), fragment=f_lo,
        )
        # Refresh manifest after the pre-split.
        m_obj0 = read_object_manifest(root["0"], 0)

        # Now remove the bridging edge with update_objects=True.
        report = remove_link(
            root, link_ref,
            atomic=True, update_objects=True,
        )
        # Two new OIDs come out; original OID 0 untouched (atomic).
        assert 0 in report.oid_remap
        new_a = report.oid_remap[0]
        # Side A and Side B both exist; one of them is new_a.
        side_a = read_object_manifest(root["0"], new_a)
        # Verify Side A's manifest is the lo-fragment side of the cut.
        assert any(fi == f_lo for _cc, fi in side_a)
        # The other new OID has the hi-fragment side; locate it by
        # scanning for an OID whose manifest references f_hi only.
        from zarr_vectors.core.arrays import read_all_object_manifests
        all_m = read_all_object_manifests(root["0"])
        side_b_candidates = [
            i for i, m in enumerate(all_m)
            if m and any(fi == f_hi for _cc, fi in m)
            and not any(fi == f_lo for _cc, fi in m)
            and i != 0  # exclude the original OID 0
        ]
        assert len(side_b_candidates) >= 1


class TestSplitIntraFragment:

    def test_split_intra_fragment_atomic(
        self, single_fragment_chain: str,
    ) -> None:
        """Item 6: single fragment with 6 rows.  Removing the link
        between row 2 and row 3 slices the fragment.  Atomic mode
        keeps the original fragment intact + appends per-side
        fragments + new OIDs."""
        root = open_store(single_fragment_chain, mode="r+")
        link_ref = _find_link_row(
            root, (0, 0, 0), endpoints={2, 3},
        )
        assert link_ref is not None, (
            "couldn't locate the 2-3 link row in the test fixture"
        )

        verts_before = read_chunk_vertices(
            root["0"], (0, 0, 0), dtype=np.float32, ndim=3,
        )
        n_frags_before = len(verts_before)

        report = remove_link(
            root, link_ref, atomic=True, update_objects=True,
        )
        assert 0 in report.oid_remap

        verts_after = read_chunk_vertices(
            root["0"], (0, 0, 0), dtype=np.float32, ndim=3,
        )
        # Atomic: new fragments appended; original kept.
        assert len(verts_after) > n_frags_before
        # Original fragment 0 still has all 6 rows.
        np.testing.assert_allclose(
            verts_after[0], verts_before[0], atol=1e-5,
        )

    def test_split_intra_fragment_nonatomic(
        self, single_fragment_chain: str,
    ) -> None:
        """Item 7: same scenario as item 6 but atomic=False — the
        original fragment is tombstoned and the original OID's
        manifest is Side A."""
        root = open_store(single_fragment_chain, mode="r+")
        link_ref = _find_link_row(
            root, (0, 0, 0), endpoints={2, 3},
        )
        assert link_ref is not None
        report = remove_link(
            root, link_ref, atomic=False, update_objects=True,
        )
        assert report.n_edits >= 1
        # Original fragment 0 should have 0 rows (tombstoned).
        verts_after = read_chunk_vertices(
            root["0"], (0, 0, 0), dtype=np.float32, ndim=3,
        )
        assert verts_after[0].shape[0] == 0


# =====================================================================
# Items 8 & 10: edit_link decomposition + implicit-sequential refusal
# =====================================================================

class TestEditLinkAndConventions:

    def test_edit_link_update_objects_runs_split_then_merge(
        self, two_object_graph: str,
    ) -> None:
        """Item 8: edit_link with update_objects=True decomposes into a
        split (old endpoints leaving) + merge (new endpoints joining).
        Smoke check: completes without raising and records edits."""
        root = open_store(two_object_graph, mode="r+")
        groups = read_chunk_links(root["0"], (0, 0, 0), delta=0)
        target_frag = next(
            i for i, g in enumerate(groups) if g.shape[0] > 0
        )
        link_ref = LinkRef(
            level=0, chunk=(0, 0, 0), fragment=target_frag, row=0, delta=0,
        )
        report = edit_link(
            root, link_ref, new_endpoints=(2, 3),
            atomic=False, update_objects=True,
        )
        assert report.n_edits >= 1

    def test_remove_link_implicit_sequential_raises_when_fail_fast(
        self, skeleton_chain: str,
    ) -> None:
        """Item 10 (opt-in fail-fast): with
        ``auto_materialise_links=False`` on the session, a
        ``remove_link(update_objects=True)`` on an implicit skeleton
        raises EditError pointing at materialise_object_links_explicit.
        """
        from zarr_vectors.ops import EditSession
        root = open_store(skeleton_chain, mode="r+")
        ref, _ = add_link(
            root, level=0, src=2, dst=0, chunk=(0, 0, 0),
        )
        with pytest.raises(EditError) as exc:
            with EditSession(
                root, atomic=False, auto_materialise_links=False,
            ) as ed:
                ed.remove_link(ref, update_objects=True)
        assert "materialise_object_links_explicit" in str(exc.value)

    def test_remove_link_implicit_sequential_auto_materialises_by_default(
        self, skeleton_chain: str,
    ) -> None:
        """Item 10 (default behaviour): without opting out, the same
        edit auto-materialises every object's implicit chain to
        explicit branch-table rows, flips ``links_convention`` to
        ``"explicit"``, emits a UserWarning, and then succeeds.
        """
        import warnings as _w
        from zarr_vectors.core.metadata import RootMetadata

        root = open_store(skeleton_chain, mode="r+")
        ref, _ = add_link(
            root, level=0, src=2, dst=0, chunk=(0, 0, 0),
        )
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            report = remove_link(
                root, ref, atomic=False, update_objects=True,
            )
        # The conversion warning fires once, with the canonical text.
        materialise_warnings = [
            w for w in caught if "auto-materialised" in str(w.message)
        ]
        assert len(materialise_warnings) >= 1
        # Convention is now explicit.
        meta = RootMetadata.from_dict(root.attrs.to_dict())
        assert meta.links_convention == "explicit"
        # The edit completed.
        assert report.n_edits >= 1


# =====================================================================
# Items 11-12: split_fragment public primitive
# =====================================================================

class TestSplitFragment:

    def test_split_fragment_atomic(self, single_fragment_chain: str) -> None:
        """Item 11: 6-row fragment partitioned by [[0,1,2],[3,4,5]] →
        two new fragments materialised; original intact under atomic."""
        root = open_store(single_fragment_chain, mode="r+")
        ref = FragmentRef(level=0, chunk=(0, 0, 0), fragment=0)
        verts_before = read_chunk_vertices(
            root["0"], (0, 0, 0), dtype=np.float32, ndim=3,
        )
        n_frags_before = len(verts_before)
        new_refs, _ = split_fragment(
            root, ref,
            row_partition=[np.array([0, 1, 2]), np.array([3, 4, 5])],
            atomic=True,
        )
        assert len(new_refs) == 2
        verts_after = read_chunk_vertices(
            root["0"], (0, 0, 0), dtype=np.float32, ndim=3,
        )
        assert len(verts_after) == n_frags_before + 2
        assert verts_after[new_refs[0].fragment].shape == (3, 3)
        assert verts_after[new_refs[1].fragment].shape == (3, 3)
        np.testing.assert_allclose(
            verts_after[new_refs[0].fragment],
            verts_before[0][[0, 1, 2]],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            verts_after[new_refs[1].fragment],
            verts_before[0][[3, 4, 5]],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            verts_after[0], verts_before[0], atol=1e-5,
        )

    def test_split_fragment_propagate_subset(
        self, tmp_path: Path,
    ) -> None:
        """Item 12: same fragment referenced by two OIDs.

        ``split_fragment(..., propagate_to_objects=[1])`` rewrites only
        OID 1's manifest; OID 0 still references the original fragment
        unchanged.
        """
        path = tmp_path / "store.zv"
        positions = np.array(
            [
                [10.0, 10.0, 10.0],
                [11.0, 11.0, 11.0],
                [12.0, 12.0, 12.0],
                [13.0, 13.0, 13.0],
            ],
            dtype=np.float32,
        )
        write_points(
            str(path), positions,
            chunk_shape=(200.0, 200.0, 200.0),
            bounds=([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]),
            object_ids=np.zeros(4, dtype=np.int64),  # one OID, one fragment
        )
        root = open_store(str(path), mode="r+")

        # Add a second OID that shares OID 0's fragment.
        m0 = read_object_manifest(root["0"], 0)
        with EditSession(root, atomic=False) as ed:
            ref = ed.add_object(
                level=0,
                vertices=np.array([[50.0, 50.0, 50.0]], dtype=np.float32),
            )
            shared_oid = ref.object_id
            ed.edit_object(
                shared_oid, level=0,
                new_manifest=m0, atomic=False,
            )

        new_refs, _ = split_fragment(
            root,
            FragmentRef(level=0, chunk=(0, 0, 0), fragment=0),
            row_partition=[np.array([0, 1]), np.array([2, 3])],
            atomic=False,
            propagate_to_objects=[shared_oid],
        )
        # Propagated OID's manifest now references the two new fragments.
        m_new = read_object_manifest(root["0"], shared_oid)
        assert all(
            fi in {nr.fragment for nr in new_refs}
            for _cc, fi in m_new
        ), m_new
        # OID 0's manifest still references the original fragment.
        m0_after = read_object_manifest(root["0"], 0)
        assert m0_after == m0, (
            f"non-targeted OID 0 should be unchanged; got {m0_after} vs {m0}"
        )
