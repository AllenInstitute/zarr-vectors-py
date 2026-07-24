"""Tests for ``zarr_vectors.core.paths`` — the links layout path helpers."""

from __future__ import annotations

import pytest

from zarr_vectors.core.paths import (
    SELF_OFFSETS_SEGMENT,
    format_delta,
    format_offsets,
    intra_offsets,
    is_intra,
    link_attributes_group_path,
    link_attributes_path,
    links_group_path,
    links_path,
    parse_delta,
    parse_offsets,
)


# ===================================================================
# Level deltas
# ===================================================================

@pytest.mark.parametrize(
    "delta,expected",
    [(0, "0"), (1, "+1"), (-1, "-1"), (2, "+2"), (-3, "-3"), (10, "+10")],
)
def test_format_delta_round_trip(delta, expected):
    assert format_delta(delta) == expected
    assert parse_delta(expected) == delta


@pytest.mark.parametrize("bad", ["", " ", "abc", "01", "+", "-"])
def test_parse_delta_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_delta(bad)


# ===================================================================
# Offsets segments
# ===================================================================

@pytest.mark.parametrize(
    "offsets,expected",
    [
        # L=1: no other endpoint to locate.
        ((), "self"),
        # L=2 intra-chunk — all-zero offsets, no separate family.
        (((0, 0, 0),), "0.0.0"),
        # L=2 cross-chunk, positive / negative / mixed.
        (((0, 0, 1),), "0.0.+1"),
        (((0, 0, -1),), "0.0.-1"),
        (((-1, 0, 2),), "-1.0.+2"),
        # L=3 face spanning the source, +z and +y.
        (((0, 0, 1), (0, 1, 0)), "0.0.+1_0.+1.0"),
        # L=3 intra.
        (((0, 0, 0), (0, 0, 0)), "0.0.0_0.0.0"),
        # L=4 quad.
        (
            ((0, 0, 1), (0, 1, 0), (1, 0, 0)),
            "0.0.+1_0.+1.0_+1.0.0",
        ),
        # Non-3D stores.
        (((1,),), "+1"),
        (((0, -2),), "0.-2"),
    ],
)
def test_format_offsets(offsets, expected):
    assert format_offsets(offsets) == expected


@pytest.mark.parametrize(
    "offsets,sid_ndim",
    [
        ((), 3),
        (((0, 0, 0),), 3),
        (((0, 0, 1),), 3),
        (((0, 0, -1),), 3),
        (((-1, 2, -3),), 3),
        (((0, 0, 1), (0, 1, 0)), 3),
        (((0, 0, 0), (0, 0, 0)), 3),
        (((0, 0, 1), (0, 1, 0), (1, 0, 0)), 3),
        (((1,),), 1),
        (((0, -2),), 2),
        (((0, 0, 0, 0),), 4),
    ],
)
def test_format_parse_offsets_round_trip(offsets, sid_ndim):
    """format_offsets ↔ parse_offsets for L=1..4, negative, zero, self."""
    link_width = len(offsets) + 1
    seg = format_offsets(offsets)
    assert parse_offsets(seg, sid_ndim=sid_ndim, link_width=link_width) == \
        tuple(tuple(o) for o in offsets)


def test_empty_offsets_is_self_segment():
    assert format_offsets(()) == SELF_OFFSETS_SEGMENT
    assert parse_offsets(
        SELF_OFFSETS_SEGMENT, sid_ndim=3, link_width=1,
    ) == ()


class TestParseOffsetsGuards:
    """A malformed listing must fail fast, not decode to wrong geometry."""

    def test_self_requires_link_width_one(self):
        with pytest.raises(ValueError, match="link_width=1"):
            parse_offsets(SELF_OFFSETS_SEGMENT, sid_ndim=3, link_width=2)

    def test_link_width_one_requires_self(self):
        with pytest.raises(ValueError, match="self"):
            parse_offsets("0.0.0", sid_ndim=3, link_width=1)

    def test_offset_count_mismatch(self):
        # One offset present, but link_width=3 needs two.
        with pytest.raises(ValueError, match="expected 2"):
            parse_offsets("0.0.0", sid_ndim=3, link_width=3)
        # Two offsets present, but link_width=2 needs one.
        with pytest.raises(ValueError, match="expected 1"):
            parse_offsets("0.0.0_0.0.0", sid_ndim=3, link_width=2)

    def test_component_count_mismatch(self):
        with pytest.raises(ValueError, match="sid_ndim=3"):
            parse_offsets("0.0", sid_ndim=3, link_width=2)
        with pytest.raises(ValueError, match="sid_ndim=2"):
            parse_offsets("0.0.0", sid_ndim=2, link_width=2)

    def test_malformed_component_rejected(self):
        # "01" is not a valid signed delta component.
        with pytest.raises(ValueError):
            parse_offsets("0.0.01", sid_ndim=3, link_width=2)


class TestIntraOffsets:

    @pytest.mark.parametrize(
        "sid_ndim,link_width,expected",
        [
            (3, 1, ()),
            (3, 2, ((0, 0, 0),)),
            (3, 3, ((0, 0, 0), (0, 0, 0))),
            (2, 2, ((0, 0),)),
        ],
    )
    def test_intra_offsets(self, sid_ndim, link_width, expected):
        assert intra_offsets(sid_ndim, link_width) == expected

    def test_intra_offsets_format_to_zeros(self):
        assert format_offsets(intra_offsets(3, 2)) == "0.0.0"
        assert format_offsets(intra_offsets(3, 3)) == "0.0.0_0.0.0"
        assert format_offsets(intra_offsets(3, 1)) == SELF_OFFSETS_SEGMENT

    @pytest.mark.parametrize(
        "offsets,expected",
        [
            ((), True),                       # link_width=1 is intra
            (((0, 0, 0),), True),
            (((0, 0, 0), (0, 0, 0)), True),
            (((0, 0, 1),), False),
            (((0, 0, 0), (0, 0, -1)), False),
        ],
    )
    def test_is_intra(self, offsets, expected):
        assert is_intra(offsets) is expected


# ===================================================================
# Path composition
# ===================================================================

def test_links_group_path():
    assert links_group_path() == "links/0"
    assert links_group_path(0) == "links/0"
    assert links_group_path(1) == "links/+1"
    assert links_group_path(-2) == "links/-2"


def test_links_path():
    # Intra-chunk edges: the all-zero offsets array.
    assert links_path(0, ((0, 0, 0),)) == "links/0/0.0.0"
    # One chunk along +z.
    assert links_path(0, ((0, 0, 1),)) == "links/0/0.0.+1"
    # Cross-level.
    assert links_path(1, ((0, 0, 0),)) == "links/+1/0.0.0"
    assert links_path(-2, ((0, 0, -1),)) == "links/-2/0.0.-1"
    # link_width=1 parent refs.
    assert links_path(0, ()) == "links/0/self"
    # L=3 face.
    assert links_path(0, ((0, 0, 1), (0, 1, 0))) == "links/0/0.0.+1_0.+1.0"


def test_link_attributes_group_path():
    assert link_attributes_group_path("weight") == "link_attributes/weight/0"
    assert link_attributes_group_path("weight", 1) == \
        "link_attributes/weight/+1"
    assert link_attributes_group_path("kind", -2) == "link_attributes/kind/-2"


def test_link_attributes_path_mirrors_links_path():
    assert link_attributes_path("weight", 0, ((0, 0, 0),)) == \
        "link_attributes/weight/0/0.0.0"
    assert link_attributes_path("weight", 2, ((0, 0, 1),)) == \
        "link_attributes/weight/+2/0.0.+1"
    # The attribute family carries the same delta + offsets segment as the
    # link family it parallels — that is what keeps cells aligned 1:1.
    for delta, offsets in [
        (0, ((0, 0, 0),)),
        (1, ((0, 0, -1),)),
        (-2, ((0, 0, 1), (0, 1, 0))),
        (0, ()),
    ]:
        assert link_attributes_path("w", delta, offsets).split("/")[-2:] == \
            links_path(delta, offsets).split("/")[-2:]
