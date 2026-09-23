"""The RFC 8 ``ome`` root node (0.9.2).

What these pin is membership, not interpretation: that a store carries a
node an OME collection can resolve by path, that the node stays true as
levels come and go, and that adding it moved nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

import zarr_vectors as zv
from zarr_vectors.building import stamp_ome_node
from zarr_vectors.constants import FORMAT_VERSION
from zarr_vectors.core.metadata import axes_with_unit
from zarr_vectors.core.ome import (
    OME_ATTRS_KEY,
    OME_VERSION,
    WORLD,
    derive_store_name,
    is_owned_node,
)
from zarr_vectors.core.store import (
    create_store,
    open_store,
    remove_resolution_level,
)
from zarr_vectors.exceptions import MetadataError

MICRON_AXES = [
    {"name": "x", "type": "space", "unit": "micrometer"},
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "z", "type": "space", "unit": "micrometer"},
]


def _root_attrs(path):
    return open_store(path, mode="r").attrs.to_dict()


def _assert_rfc8_legal(node, *, is_root: bool) -> None:
    """The structural rules RFC 8 puts on every node.

    Deliberately a checker rather than a golden file: what has to hold is
    that the document is legal, and a literal comparison would pass for a
    node that is legal only by accident of how it was built.
    """
    assert isinstance(node, dict)
    assert node.get("type"), "every node carries a type"
    assert isinstance(node.get("name"), str) and node["name"], (
        "every node carries a non-empty name"
    )

    # ``version`` belongs to the root of a document and nowhere else.
    if is_root:
        assert node.get("version") == OME_VERSION
    else:
        assert "version" not in node

    # A collection carries children or a path, never both and never
    # neither.  Other node types are unconstrained, which is what lets a
    # prefixed leaf terminate the tree.
    if node["type"] == "collection":
        has_nodes, has_path = "nodes" in node, "path" in node
        assert has_nodes != has_path, "a collection has nodes XOR path"

    children = node.get("nodes") or []
    names = [c.get("name") for c in children]
    assert len(names) == len(set(names)), "names are unique within a collection"
    for child in children:
        _assert_rfc8_legal(child, is_root=False)

    # Anything that is not a core identifier must be prefixed.
    for key in (node.get("attributes") or {}):
        assert key in {"scene", "coordinateSystems", "coordinateTransformations",
                       "labels", "plate", "well"} or ":" in key, (
            f"unprefixed non-core attribute key: {key}"
        )


@pytest.fixture
def store(tmp_path):
    """A two-level point cloud at a path whose basename is a real name."""
    path = tmp_path / "skeleton.zarrvectors"
    ds = zv.create(
        str(path),
        schema=zv.Schema(
            bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
            kind="point_cloud",
            axes=[zv.Axis(a["name"], a["unit"]) for a in MICRON_AXES],
            layout=zv.Layout(cells=4),
        ),
    )
    rng = np.random.default_rng(0)
    ds.add_points(
        rng.uniform(0, 1000, size=(2000, 3)).astype("float32"),
        object_ids=np.repeat(np.arange(20), 100),
    )
    ds.build_pyramid(factors=[(2.0, 1.0)])
    return path


class TestNodeShape:
    def test_root_carries_a_legal_node(self, store):
        node = _root_attrs(store)[OME_ATTRS_KEY]
        _assert_rfc8_legal(node, is_root=True)
        assert node["type"] == "collection"

    def test_name_comes_from_the_store_path(self, store):
        assert _root_attrs(store)[OME_ATTRS_KEY]["name"] == "skeleton"

    def test_world_declares_the_stores_axes_and_units(self, store):
        scene = _root_attrs(store)[OME_ATTRS_KEY]["attributes"]["scene"]
        systems = scene["coordinateSystems"]
        assert [s["name"] for s in systems] == [WORLD]
        assert systems[0]["axes"] == MICRON_AXES
        # Vertices are stored in world coordinates, so there is no edge.
        assert scene["coordinateTransformations"] == []

    def test_world_is_identified_by_id_as_well_as_name(self, store):
        # RFC 8 binds a Reference to a coordinate system's id; RFC 5
        # readers look for its name.  Both, so neither kind is stranded.
        system = _root_attrs(store)[OME_ATTRS_KEY]["attributes"]["scene"][
            "coordinateSystems"
        ][0]
        assert system["id"] == WORLD and system["name"] == WORLD

    def test_an_axis_without_a_unit_carries_no_unit_key(self, tmp_path):
        # NGFF requires UDUNITS-2 names and rejects a placeholder, so an
        # undeclared unit must be an absent key rather than "".
        root = create_store(
            tmp_path / "unitless.zarrvectors",
            axes=[{"name": n, "type": "space"} for n in "xyz"],
        )
        axes = root.attrs.to_dict()[OME_ATTRS_KEY]["attributes"]["scene"][
            "coordinateSystems"
        ][0]["axes"]
        assert all("unit" not in a for a in axes)


class TestLevelList:
    def test_levels_track_the_pyramid(self, store):
        node = _root_attrs(store)[OME_ATTRS_KEY]
        assert [n["name"] for n in node["nodes"]] == ["0", "1"]
        assert {n["type"] for n in node["nodes"]} == {"zv:level"}

    def test_a_level_leaf_terminates_the_tree(self, store):
        # The point of a prefixed leaf: no level group and no array needs
        # a block of its own, so nothing cascades.
        for child in _root_attrs(store)[OME_ATTRS_KEY]["nodes"]:
            assert "path" not in child and "nodes" not in child
        level_attrs = open_store(store, mode="r")["0"].attrs.to_dict()
        assert OME_ATTRS_KEY not in level_attrs

    def test_removing_a_level_updates_the_list(self, store):
        remove_resolution_level(open_store(store, mode="r+"), 1)
        node = _root_attrs(store)[OME_ATTRS_KEY]
        assert [n["name"] for n in node["nodes"]] == ["0"]
        _assert_rfc8_legal(node, is_root=True)

    def test_a_finished_store_always_has_a_level(self, tmp_path):
        root = create_store(tmp_path / "warm.zarrvectors")
        assert root.attrs.to_dict()[OME_ATTRS_KEY]["nodes"] == [
            {"type": "zv:level", "name": "0"}
        ]


class TestAdditive:
    def test_the_other_blocks_are_untouched(self, store):
        attrs = _root_attrs(store)
        assert set(attrs) >= {"zarr_vectors", "multiscales", OME_ATTRS_KEY}
        # Still the source of truth, still where every reader looks.
        assert attrs["zarr_vectors"]["zv_version"] == FORMAT_VERSION
        assert attrs["multiscales"][0]["metadata"]["format"] == "zarr_vectors"
        assert attrs["multiscales"][0]["axes"] == MICRON_AXES

    def test_a_pre_0_9_2_store_still_opens(self, store):
        """The block is additive, so its absence is not an error."""
        root = open_store(store, mode="r+")
        attrs = root.attrs.to_dict()
        del attrs[OME_ATTRS_KEY]
        root._zarr.attrs.put(attrs)

        assert OME_ATTRS_KEY not in _root_attrs(store)
        ds = zv.open(str(store))
        assert ds.bounds[1].tolist() == [1000.0, 1000.0, 1000.0]
        assert ds.levels == (0, 1)

    def test_stamp_brings_such_a_store_up_to_date_in_place(self, store):
        root = open_store(store, mode="r+")
        attrs = root.attrs.to_dict()
        del attrs[OME_ATTRS_KEY]
        root._zarr.attrs.put(attrs)

        stamped = stamp_ome_node(str(store))
        _assert_rfc8_legal(stamped, is_root=True)
        assert stamped == _root_attrs(store)[OME_ATTRS_KEY]
        assert [n["name"] for n in stamped["nodes"]] == ["0", "1"]
        # Metadata-only: the data is still readable and unchanged.
        assert zv.open(str(store)).select(
            bbox=([0, 0, 0], [1000, 1000, 1000])
        ).read().positions.shape == (2000, 3)

    def test_stamping_twice_is_idempotent(self, store):
        assert stamp_ome_node(str(store)) == stamp_ome_node(str(store))


class TestName:
    @pytest.mark.parametrize(
        "url, expected",
        [
            ("file:///data/minnie65/skeleton.zarrvectors", "skeleton"),
            ("file:///data/tracts.zv", "tracts"),
            ("s3://bucket/prefix/synapses.zarrvectors/", "synapses"),
            ("/data/no_extension", "no_extension"),
            ("file:///data/with%20space.zarrvectors", "with space"),
            ("C:\\data\\windows.zarrvectors", "windows"),
            ("", "zarr_vectors"),
            ("<MemoryStore>", "<MemoryStore>"),
        ],
    )
    def test_derived_from_url(self, url, expected):
        assert derive_store_name(url) == expected

    def test_explicit_name_wins(self, tmp_path):
        root = create_store(tmp_path / "s.zarrvectors", name="Minnie65 cell 864")
        assert root.attrs.to_dict()[OME_ATTRS_KEY]["name"] == "Minnie65 cell 864"

    def test_explicit_name_survives_a_later_level(self, tmp_path):
        path = tmp_path / "s.zarrvectors"
        create_store(path, name="Minnie65 cell 864")
        ds = zv.open(str(path), mode="r+")
        ds.add_points(np.zeros((10, 3), dtype="float32"))
        ds.build_pyramid(factors=[(2.0, 1.0)])
        node = _root_attrs(path)[OME_ATTRS_KEY]
        assert node["name"] == "Minnie65 cell 864"
        assert [n["name"] for n in node["nodes"]] == ["0", "1"]

    def test_a_foreign_attribute_survives_a_refresh(self, store):
        """Only the keys this package owns are authoritative."""
        root = open_store(store, mode="r+")
        node = root.attrs.to_dict()[OME_ATTRS_KEY]
        node["attributes"]["zv:companionImage"] = {
            "id": "em", "path": {"type": "zarr", "path": "../em.ome.zarr"}
        }
        root.attrs.update({OME_ATTRS_KEY: node})

        refreshed = stamp_ome_node(str(store))
        assert refreshed["attributes"]["zv:companionImage"]["id"] == "em"
        assert refreshed["attributes"]["scene"]["coordinateSystems"][0]["name"] == WORLD


class TestForeignChildren:
    """``refresh_root_node`` owns the ``zv:level`` entries, nothing else."""

    BACKREF = {
        "type": "collection", "name": "bridge_subject",
        "path": {"type": "zarr", "path": "../"},
    }

    def _add_child(self, store, child):
        root = open_store(store, mode="r+")
        node = root.attrs.to_dict()[OME_ATTRS_KEY]
        node["nodes"] = [*node["nodes"], child]
        root.attrs.update({OME_ATTRS_KEY: node})

    def test_a_foreign_child_survives_a_refresh(self, store):
        self._add_child(store, self.BACKREF)
        refreshed = stamp_ome_node(str(store))
        assert self.BACKREF in refreshed["nodes"]
        assert [n["name"] for n in refreshed["nodes"]] == [
            "0", "1", "bridge_subject",
        ]
        _assert_rfc8_legal(refreshed, is_root=True)

    def test_a_foreign_child_survives_a_level_being_removed(self, store):
        self._add_child(store, self.BACKREF)
        remove_resolution_level(open_store(store, mode="r+"), 1)
        node = _root_attrs(store)[OME_ATTRS_KEY]
        assert [n["name"] for n in node["nodes"]] == ["0", "bridge_subject"]

    def test_a_foreign_child_named_like_a_level_is_dropped(self, store):
        # Names are unique within a collection, and the level is owned.
        self._add_child(store, {"type": "zv:thing", "name": "1"})
        refreshed = stamp_ome_node(str(store))
        assert [n["type"] for n in refreshed["nodes"]] == ["zv:level", "zv:level"]
        _assert_rfc8_legal(refreshed, is_root=True)
# ---------------------------------------------------------------------------
# ``world`` carries an RFC 8 ``id``
# ---------------------------------------------------------------------------

def _world(attrs):
    systems = attrs[OME_ATTRS_KEY]["attributes"]["scene"]["coordinateSystems"]
    assert len(systems) == 1
    return systems[0]


class TestWorldId:
    """RFC 8 identifies a coordinate system by ``id``; ``name`` is descriptive.

    Without the ``id`` a collection elsewhere has nothing to bind a
    ``Reference`` like ``{"path": {...}, "id": "world"}`` to, and has to
    re-declare the frame itself.
    """

    def test_world_is_identified_by_id_and_keeps_its_name(self, store):
        world = _world(_root_attrs(store))
        assert world["id"] == WORLD
        # The RFC 5 spelling stays, for a reader written against it.
        assert world["name"] == WORLD

    def test_a_store_stamped_before_the_id_gains_it_in_place(self, store):
        """An older block declares ``world`` by name only; a refresh adds
        the id and still finds the axes (units included) by that name."""
        root = open_store(store, mode="r+")
        node = root.attrs.to_dict()[OME_ATTRS_KEY]
        del node["attributes"]["scene"]["coordinateSystems"][0]["id"]
        root.attrs.update({OME_ATTRS_KEY: node})

        stamped = stamp_ome_node(str(store))
        world = stamped["attributes"]["scene"]["coordinateSystems"][0]
        assert world["id"] == WORLD
        assert world["axes"] == MICRON_AXES


# ---------------------------------------------------------------------------
# Foreign nodes survive a refresh
# ---------------------------------------------------------------------------

#: A back-reference to the collection the store sits in -- the case this
#: exists for.  A prefixed leaf, so a generic walker does not follow it
#: back up into the parent and round again.
BACKREF = {
    "type": "bridge:container",
    "name": "container",
    "path": {"type": "zarr", "path": "../"},
    "attributes": {"bridge:role": "graph"},
}


def _add_children(path, *children):
    root = open_store(path, mode="r+")
    node = root.attrs.to_dict()[OME_ATTRS_KEY]
    node["nodes"] = list(node["nodes"]) + [dict(c) for c in children]
    root.attrs.update({OME_ATTRS_KEY: node})


def _children(path):
    return _root_attrs(path)[OME_ATTRS_KEY]["nodes"]


class TestForeignNodes:
    """``refresh_root_node`` owns the ``zv:level`` entries and nothing else."""

    def test_removing_a_level_keeps_a_foreign_node(self, store):
        _add_children(store, BACKREF)
        remove_resolution_level(open_store(store, mode="r+"), 1)

        children = _children(store)
        assert children == [{"type": "zv:level", "name": "0"}, BACKREF]
        _assert_rfc8_legal(_root_attrs(store)[OME_ATTRS_KEY], is_root=True)

    def test_adding_a_level_keeps_a_foreign_node(self, tmp_path):
        path = tmp_path / "s.zarrvectors"
        create_store(path)
        _add_children(path, BACKREF)

        ds = zv.open(str(path), mode="r+")
        ds.add_points(np.zeros((10, 3), dtype="float32"))
        ds.build_pyramid(factors=[(2.0, 1.0)])

        assert _children(path) == [
            {"type": "zv:level", "name": "0"},
            {"type": "zv:level", "name": "1"},
            BACKREF,
        ]

    def test_stamping_keeps_foreign_nodes_in_order_and_is_idempotent(self, store):
        other = {"type": "multiscale", "name": "em",
                 "path": {"type": "zarr", "path": "../em.ome.zarr"},
                 "attributes": {"coordinateSystems": []}}
        _add_children(store, BACKREF, other)

        first = stamp_ome_node(str(store))
        assert first["nodes"][2:] == [BACKREF, other]
        assert stamp_ome_node(str(store)) == first

    def test_a_stale_level_entry_is_regenerated_away(self, store):
        """Level entries are owned: one with no level group behind it goes."""
        _add_children(store, {"type": "zv:level", "name": "7"})
        stamped = stamp_ome_node(str(store))
        assert [c["name"] for c in stamped["nodes"]] == ["0", "1"]

    def test_another_zv_type_is_foreign(self, store):
        """Ownership is the level type, not the whole ``zv:`` prefix --
        the same line the attributes merge draws around ``scene``."""
        companion = {"type": "zv:companionImage", "name": "em",
                     "path": {"type": "zarr", "path": "../em.ome.zarr"}}
        _add_children(store, companion)
        assert stamp_ome_node(str(store))["nodes"][-1] == companion

    def test_a_foreign_node_named_like_a_level_is_dropped_loudly(self, store):
        """Child names are unique within an RFC 8 collection; the level
        list is derived from disk, so it wins -- with a warning, not
        silently."""
        _add_children(store, {"type": "bridge:container", "name": "1"}, BACKREF)
        with pytest.warns(UserWarning, match="unique within a collection"):
            stamped = stamp_ome_node(str(store))
        assert stamped["nodes"] == [
            {"type": "zv:level", "name": "0"},
            {"type": "zv:level", "name": "1"},
            BACKREF,
        ]
        _assert_rfc8_legal(stamped, is_root=True)

    def test_a_child_that_is_not_an_object_is_dropped_loudly(self, store):
        _add_children(store, BACKREF)
        root = open_store(store, mode="r+")
        node = root.attrs.to_dict()[OME_ATTRS_KEY]
        node["nodes"].append("not a node")
        root.attrs.update({OME_ATTRS_KEY: node})

        with pytest.warns(UserWarning, match="JSON object"):
            stamped = stamp_ome_node(str(store))
        assert stamped["nodes"][-1] == BACKREF

    @pytest.mark.parametrize(
        "child, owned",
        [
            ({"type": "zv:level", "name": "0"}, True),
            ({"type": "zv:level", "name": "0", "attributes": {"x:y": 1}}, True),
            ({"type": "zv:companionImage", "name": "em"}, False),
            ({"type": "collection", "name": "c", "path": {"type": "zarr", "path": "../"}}, False),
            ({"type": "bridge:container", "name": "container"}, False),
            ({"name": "untyped"}, False),
            ("zv:level", False),
        ],
    )
    def test_the_ownership_rule(self, child, owned):
        assert is_owned_node(child) is owned


# ---------------------------------------------------------------------------
# A unit on the vertex frame
# ---------------------------------------------------------------------------

def _root_json_bytes(path) -> bytes:
    return (path / "zarr.json").read_bytes()


class TestWorldUnit:
    """``create_store(unit=)``: the frame's unit, when the writer knows it."""

    def test_unit_reaches_the_scene_and_the_canonical_axes(self, tmp_path):
        root = create_store(tmp_path / "u.zarrvectors", unit="micrometer")
        attrs = root.attrs.to_dict()
        expected = [{"name": n, "type": "space", "unit": "micrometer"} for n in "xyz"]
        assert _world(attrs)["axes"] == expected
        assert attrs["multiscales"][0]["axes"] == expected

    def test_unit_fills_axes_that_carry_none(self, tmp_path):
        root = create_store(
            tmp_path / "u.zarrvectors",
            axes=[{"name": n, "type": "space"} for n in "zyx"],
            unit="nanometer",
        )
        assert [a["unit"] for a in _world(root.attrs.to_dict())["axes"]] == ["nanometer"] * 3
        assert [a["name"] for a in _world(root.attrs.to_dict())["axes"]] == list("zyx")

    def test_a_matching_unit_on_axes_is_accepted(self, tmp_path):
        root = create_store(tmp_path / "u.zarrvectors", axes=MICRON_AXES, unit="micrometer")
        assert _world(root.attrs.to_dict())["axes"] == MICRON_AXES

    def test_a_conflicting_unit_is_refused(self, tmp_path):
        with pytest.raises(MetadataError, match="declare it once"):
            create_store(tmp_path / "u.zarrvectors", axes=MICRON_AXES, unit="millimeter")

    @pytest.mark.parametrize("bad", ["um", "micron", "pixel", ""])
    def test_a_non_ngff_spelling_is_refused(self, tmp_path, bad):
        with pytest.raises(MetadataError, match="NGFF space unit"):
            create_store(tmp_path / "u.zarrvectors", unit=bad)

    def test_only_space_axes_take_the_unit(self):
        axes = [{"name": "t", "type": "time", "unit": "second"},
                {"name": "c", "type": "channel"},
                {"name": "y"},
                {"name": "x", "type": "space"}]
        out = axes_with_unit(axes, "micrometer")
        assert out[0] == {"name": "t", "type": "time", "unit": "second"}
        assert out[1] == {"name": "c", "type": "channel"}
        assert out[2]["unit"] == out[3]["unit"] == "micrometer"
        assert "unit" not in axes[2], "the input is not modified"

    def test_no_unit_writes_exactly_what_it_always_did(self, tmp_path):
        """Absent unit is no claim: no ``unit`` key anywhere, and the
        root document is byte-identical to one created without the
        keyword."""
        a = tmp_path / "a" / "s.zarrvectors"
        b = tmp_path / "b" / "s.zarrvectors"
        create_store(a)
        create_store(b, unit=None)
        assert _root_json_bytes(a) == _root_json_bytes(b)
        assert '"unit"' not in _root_json_bytes(a).decode()

    def test_the_unit_survives_a_later_level(self, tmp_path):
        path = tmp_path / "u.zarrvectors"
        create_store(path, unit="micrometer")
        ds = zv.open(str(path), mode="r+")
        ds.add_points(np.zeros((10, 3), dtype="float32"))
        ds.build_pyramid(factors=[(2.0, 1.0)])
        world = _world(_root_attrs(path))
        assert {a["unit"] for a in world["axes"]} == {"micrometer"}
        assert world["id"] == WORLD
