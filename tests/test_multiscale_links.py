"""End-to-end coverage for the multiscale links layout.

Covers:
- ``links/0/<offsets>/<chunk>`` regression (delta=0 writer/reader behavior).
- Manual round-trip of ``links/+1/`` and ``links/-2/``.
- The ``link_attributes`` writer/reader pair at a non-zero delta.
- ``build_pyramid`` cross-level emission across depth / storage modes.
- The hard schema-version cutoff.

There is no separate ``cross_chunk_links`` family: connectivity is one
family per delta, whose children are one array per relative-offset
segment.  Intra-chunk links are the all-zero offsets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.constants import (
    CAP_MULTISCALE_LINKS,
    FORMAT_VERSION,
    LINKS,
    XLEVEL_EXPLICIT,
    XLEVEL_IMPLICIT,
    XLEVEL_NONE,
)
from zarr_vectors.core.arrays import (
    create_link_attributes_array,
    create_links_array,
    create_vertices_array,
    list_link_attribute_deltas,
    list_link_deltas,
    list_link_offsets,
    read_chunk_links,
    read_link_attributes,
    read_links,
    write_chunk_link_attributes,
    write_chunk_links,
    write_chunk_vertices,
    write_link_attributes,
    write_links,
)
from zarr_vectors.core.metadata import RootMetadata
from zarr_vectors.core.paths import (
    format_delta,
    intra_offsets,
    link_attributes_group_path,
    link_attributes_path,
    links_group_path,
    links_path,
)
from zarr_vectors.core.store import (
    FsGroup,
    create_store,
    get_resolution_level,
)
from zarr_vectors.exceptions import ArrayError, MetadataError


def _make_level_group(tmp_path: Path, ndim: int = 3) -> FsGroup:
    """A resolution level backed by a real store.

    Every per-spatial-chunk array is ONE vlen array whose shape is the
    level's chunk grid, so a level group has to come from a store with
    ``bounds`` — there is no grid to allocate against otherwise.
    """
    root = create_store(
        str(tmp_path / "store.zarr"),
        bounds=([0.0] * ndim, [1000.0] * ndim),
        chunk_shape=(100.0,) * ndim,
        geometry_types=["graph"],
        ndim=ndim,
    )
    return get_resolution_level(root, 0)


# ===================================================================
# delta=0 regression (current behavior preserved under new path layout)
# ===================================================================

class TestDeltaZero:

    def test_intra_chunk_links_round_trip(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_links_array(lg, link_width=2, sid_ndim=3)
        write_chunk_vertices(lg, (0, 0, 0), [np.zeros((4, 3), dtype=np.float32)])
        edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
        write_chunk_links(lg, (0, 0, 0), [edges])

        groups = read_chunk_links(lg, (0, 0, 0), link_width=2, delta=0)
        assert len(groups) == 1
        np.testing.assert_array_equal(groups[0], edges)

    def test_cross_chunk_links_round_trip(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_links_array(lg, link_width=2, sid_ndim=3)
        links = [
            (((0, 0, 0), 4), ((0, 0, 1), 0)),
            (((0, 0, 0), 2), ((1, 0, 0), 1)),
        ]
        write_links(lg, links, sid_ndim=3)
        # The two records land in different offset arrays ("0.0.+1" and
        # "+1.0.0"), and reads enumerate by sorted segment — so the read
        # order is not the input order.  Compare as a set.
        assert set(read_links(lg)) == set(links)
        # ASCII sorts '+' (0x2b) before '0' (0x30), so the empty intra
        # array create_links_array pre-made sorts LAST.
        assert list_link_offsets(lg, 0) == ["+1.0.0", "0.0.+1", "0.0.0"]

    def test_paths_have_delta_segment(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_links_array(lg, link_width=2, delta=0, sid_ndim=3)
        # <delta> sits between the family prefix and the <offsets> array;
        # links/<delta> is a GROUP whose children are the offset arrays.
        assert lg.array_exists(f"{LINKS}/0")
        assert lg.array_exists(links_group_path(0))
        assert lg.array_exists(links_path(0, intra_offsets(3, 2)))
        assert lg[links_group_path(0)].children() == ["0.0.0"]


# ===================================================================
# delta != 0 — cross-pyramid-level links (manual writer/reader)
# ===================================================================

class TestCrossLevelManual:

    def test_links_plus_one(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        # Source vertices at this level; target side conceptually lives
        # at this_level + 1.  Per-chunk single edge group.
        create_links_array(lg, link_width=2, delta=1, sid_ndim=3)
        edges = np.array([[0, 0], [1, 0], [2, 1]], dtype=np.int64)
        write_chunk_links(lg, (0, 0, 0), [edges], delta=1)
        back = read_chunk_links(lg, (0, 0, 0), link_width=2, delta=1)
        assert len(back) == 1
        np.testing.assert_array_equal(back[0], edges)

    def test_cross_level_minus_two(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_links_array(lg, link_width=2, delta=-2, sid_ndim=3)
        links = [
            (((1, 0, 0), 0), ((0, 0, 0), 3)),
            (((1, 0, 0), 1), ((0, 0, 1), 7)),
        ]
        write_links(lg, links, sid_ndim=3, delta=-2)
        # Cross-level records are never canonical-sorted: the source is
        # always input endpoint 0, so input order survives.  Both records
        # source at (1,0,0) but under different offsets, so compare as a set.
        assert set(read_links(lg, delta=-2)) == set(links)
        # Both source at (1,0,0); the targets sit at -1 and (-1,0,+1).
        # '+' (0x2b) sorts before '0' (0x30), and the empty intra array
        # create_links_array pre-made sorts last.
        assert list_link_offsets(lg, -2) == ["-1.0.+1", "-1.0.0", "0.0.0"]

    def test_listing_helpers(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_links_array(lg, link_width=2, delta=0, sid_ndim=3)
        create_links_array(lg, link_width=2, delta=1, sid_ndim=3)
        create_links_array(lg, link_width=2, delta=-1, sid_ndim=3)
        # A non-zero-offset array under an existing delta adds no new delta.
        create_links_array(
            lg, link_width=2, delta=0, sid_ndim=3, offsets=((0, 0, 1),),
        )
        # ...but a brand-new delta does, even reached via a cross offset.
        create_links_array(
            lg, link_width=2, delta=2, sid_ndim=3, offsets=((0, 0, 1),),
        )
        # One family per delta now — cross-chunk records are offset arrays
        # inside it, not a parallel cross_chunk_links/<delta> tree.
        assert list_link_deltas(lg) == [-1, 0, 1, 2]
        assert list_link_offsets(lg, 0) == ["0.0.+1", "0.0.0"]
        assert list_link_offsets(lg, 2) == ["0.0.+1"]


# ===================================================================
# Link attributes at a non-zero delta
# ===================================================================

class TestLinkAttributes:

    def test_round_trip(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_links_array(lg, link_width=2, delta=1, sid_ndim=3)
        links = [
            (((0, 0, 0), 4), ((0, 0, 1), 0)),
            (((0, 0, 0), 2), ((1, 0, 0), 1)),
            (((1, 0, 0), 5), ((1, 0, 1), 0)),
        ]
        partition = write_links(lg, links, sid_ndim=3, delta=1)

        create_link_attributes_array(
            lg, "weight", dtype="float32", delta=1, sid_ndim=3,
        )
        weights = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        write_link_attributes(
            lg, "weight", weights, num_links=3, delta=1, partition=partition,
        )

        # Attribute rows align to link records only via the shared
        # (offsets segment, cell) enumeration — so pair them by that order
        # rather than assuming input order survives.
        back = read_link_attributes(lg, "weight", delta=1)
        assert back.shape == (3,)
        by_record = dict(zip(read_links(lg, delta=1), back))
        for rec, w in zip(links, weights):
            assert np.isclose(by_record[rec], w)

    def test_length_invariant_enforced(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        bad = np.array([1.0, 2.0], dtype=np.float32)
        with pytest.raises(ArrayError):
            write_link_attributes(
                lg, "weight", bad, num_links=3, delta=0,
            )

    def test_path_layout(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_link_attributes_array(
            lg, "weight", dtype="float32", delta=0, sid_ndim=3,
        )
        # link_attributes/<name>/<delta> is a GROUP mirroring
        # links/<delta>; its children are one array per offsets segment.
        assert lg.array_exists(link_attributes_group_path("weight", 0))
        assert lg.array_exists(
            link_attributes_path("weight", 0, intra_offsets(3, 2)),
        )
        assert lg[link_attributes_group_path("weight", 0)].children() == \
            ["0.0.0"]


# ===================================================================
# Path helpers wired against on-disk truth
# ===================================================================

def test_paths_module_matches_disk_layout(tmp_path: Path) -> None:
    lg = _make_level_group(tmp_path)
    create_links_array(lg, link_width=2, delta=2, sid_ndim=3)
    create_link_attributes_array(lg, "w", dtype="float32", delta=-1, sid_ndim=3)
    create_links_array(lg, link_width=2, delta=3, sid_ndim=3, offsets=((0, 0, 1),))
    create_link_attributes_array(
        lg, "w", delta=-2, sid_ndim=3, offsets=((0, 0, 1),),
    )

    intra = intra_offsets(3, 2)
    assert lg.array_exists(links_group_path(2))
    assert lg.array_exists(links_path(2, intra))
    assert lg.array_exists(link_attributes_group_path("w", -1))
    assert lg.array_exists(link_attributes_path("w", -1, intra))
    assert lg.array_exists(links_path(3, ((0, 0, 1),)))
    assert lg.array_exists(link_attributes_path("w", -2, ((0, 0, 1),)))

    # The deltas the path helpers composed are the deltas on disk.
    assert list_link_deltas(lg) == [2, 3]
    assert list_link_attribute_deltas(lg, "w") == [-2, -1]


# ===================================================================
# Schema version cutoff
# ===================================================================

def _minimal_root_md(**overrides):
    base = dict(
        spatial_index_dims=[
            {"name": "x", "type": "space", "unit": "um"},
            {"name": "y", "type": "space", "unit": "um"},
            {"name": "z", "type": "space", "unit": "um"},
        ],
        chunk_shape=(50.0, 50.0, 50.0),
        bounds=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
        geometry_types=["point_cloud"],
    )
    base.update(overrides)
    return RootMetadata(**base)


def test_default_zv_version_is_09():
    md = _minimal_root_md()
    md.validate()
    assert md.zv_version == FORMAT_VERSION
    assert FORMAT_VERSION.startswith("0.9")


def test_pre_05_zv_version_rejected():
    md = _minimal_root_md(zv_version="0.4.1")
    with pytest.raises(MetadataError, match="0.4.1"):
        md.validate()


def test_pre_05_zv_version_rejected_on_roundtrip():
    """A wire-format dict with zv_version='0.4.1' must round-trip into
    a RootMetadata that fails validate() — proving the cutoff catches
    both freshly-constructed and freshly-loaded stores."""
    md = _minimal_root_md(zv_version="0.4.1")
    md.zv_version = "0.4.1"
    d = md.to_dict()
    # 0.5.0 from_dict reads axes from the NGFF multiscales block; inject
    # a minimal one so the round trip can complete.
    d["multiscales"] = [{
        "version": "0.4",
        "axes": list(md.spatial_index_dims),
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [1.0] * md.sid_ndim},
        ]}],
        "metadata": {"format": "zarr_vectors"},
    }]
    reloaded = RootMetadata.from_dict(d)
    with pytest.raises(MetadataError):
        reloaded.validate()


def test_invalid_cross_level_storage_rejected():
    md = _minimal_root_md()
    md.cross_level_storage = "always"
    with pytest.raises(MetadataError, match="cross_level_storage"):
        md.validate()


def test_invalid_cross_level_depth_rejected():
    md = _minimal_root_md()
    md.cross_level_depth = -2
    with pytest.raises(MetadataError, match="cross_level_depth"):
        md.validate()


# ===================================================================
# build_pyramid integration (cross-level emission)
# ===================================================================

# These integration tests exercise the full graph write → pyramid build
# round-trip.  They depend on the zarr backend; xfail under environments
# where zarr can't import (e.g. a Python build without a numcodecs
# wheel) so the rest of the suite keeps running.
zarr = pytest.importorskip("zarr")

from zarr_vectors.core.store import (  # noqa: E402
    create_store,
    get_resolution_level,
    list_resolution_levels,
    open_store,
    read_root_metadata,
)
from zarr_vectors.multiresolution.coarsen import build_pyramid  # noqa: E402
from zarr_vectors.types.graphs import write_graph  # noqa: E402


def _seed_simple_graph(tmp_path: Path) -> Path:
    """Write a small 3D graph store usable by build_pyramid."""
    store_path = tmp_path / "graph.zarr"
    rng = np.random.default_rng(0)
    n = 64
    positions = rng.uniform(0.0, 100.0, size=(n, 3)).astype(np.float32)
    # Trivial spanning tree edges so the graph is connected enough to
    # exercise both intra- and cross-chunk links.
    edges = np.stack(
        [np.arange(n - 1, dtype=np.int64), np.arange(1, n, dtype=np.int64)],
        axis=1,
    )
    object_ids = np.zeros(n, dtype=np.int64)
    write_graph(
        store_path,
        positions=positions,
        edges=edges,
        object_ids=object_ids,
        chunk_shape=(40.0, 40.0, 40.0),
        bounds=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
    )
    return store_path


def _delta_dirs(root, level: int, prefix: str = LINKS) -> set[str]:
    """The ``<delta>`` segment names present under ``links/`` at a level.

    ``links/<delta>`` is a group (its children are the offset arrays), so
    plain iteration — which yields sub-groups only — sees them.
    """
    lg = get_resolution_level(root, level)
    if not lg.array_exists(prefix):
        return set()
    return {name for name in lg[prefix]}


def _assert_deltas_match_listing(root, level: int) -> set[str]:
    """Cross-check the directory view against ``list_link_deltas``.

    Guards the ``_delta_dirs`` helper itself: if it silently returned an
    empty set again, every ``all(...)`` assertion built on it would pass
    vacuously.
    """
    lg = get_resolution_level(root, level)
    dirs = _delta_dirs(root, level)
    assert dirs == {format_delta(d) for d in list_link_deltas(lg)}
    return dirs


def test_build_pyramid_depth_zero_emits_no_cross_level(tmp_path: Path) -> None:
    store_path = _seed_simple_graph(tmp_path)
    build_pyramid(
        store_path,
        factors=[(2.0, 1.0), (2.0, 1.0)],
        cross_level_depth=0,
        cross_level_storage=XLEVEL_NONE,
    )
    root = open_store(str(store_path))
    levels = list_resolution_levels(root)
    seen_any = False
    for lvl in levels:
        deltas = _assert_deltas_match_listing(root, lvl)
        # Only delta=0 (or nothing at all) should exist.
        assert deltas <= {"0"}
        seen_any = seen_any or bool(deltas)
    # Non-vacuity: the pyramid must actually have written a delta=0 family
    # somewhere, otherwise "<= {'0'}" holds for an empty set and proves
    # nothing about cross-level emission being suppressed.
    assert seen_any, "expected at least one level to carry links/0"


def test_build_pyramid_explicit_depth_one(tmp_path: Path) -> None:
    store_path = _seed_simple_graph(tmp_path)
    build_pyramid(
        store_path,
        factors=[(2.0, 1.0), (2.0, 1.0)],
        cross_level_depth=1,
        cross_level_storage=XLEVEL_EXPLICIT,
    )
    root = open_store(str(store_path))
    levels = sorted(list_resolution_levels(root))
    assert len(levels) >= 2, "pyramid should have produced at least one coarser level"

    # Root metadata reflects choices + capability is stamped.
    rmeta = read_root_metadata(root)
    assert rmeta.cross_level_depth == 1
    assert rmeta.cross_level_storage == XLEVEL_EXPLICIT
    assert CAP_MULTISCALE_LINKS in rmeta.format_capabilities

    # Fine levels carry +1, coarser levels carry -1.
    has_plus = False
    has_minus = False
    for lvl in levels[:-1]:
        if "+1" in _assert_deltas_match_listing(root, lvl):
            has_plus = True
    for lvl in levels[1:]:
        if "-1" in _assert_deltas_match_listing(root, lvl):
            has_minus = True
    assert has_plus, "explicit mode must materialize +1 at the finer level"
    assert has_minus, "explicit mode must materialize -1 at the coarser level"


def test_build_pyramid_implicit_only_plus(tmp_path: Path) -> None:
    store_path = _seed_simple_graph(tmp_path)
    build_pyramid(
        store_path,
        factors=[(2.0, 1.0), (2.0, 1.0)],
        cross_level_depth=1,
        cross_level_storage=XLEVEL_IMPLICIT,
    )
    root = open_store(str(store_path))
    levels = sorted(list_resolution_levels(root))
    # No -N deltas anywhere.
    all_deltas: set[str] = set()
    for lvl in levels:
        link_deltas = _assert_deltas_match_listing(root, lvl)
        all_deltas |= link_deltas
        assert all(not d.startswith("-") for d in link_deltas), \
            f"implicit mode should not emit negative links/<delta> at level {lvl}"
    # Non-vacuity: "no negative deltas" is trivially true of an empty set.
    # Implicit mode must still have emitted the +1 side it is defined by,
    # so the assertion above is ranging over a set that could have failed.
    assert "+1" in all_deltas, (
        f"implicit mode must still materialize +1 at the finer level; "
        f"saw deltas {sorted(all_deltas)}"
    )
