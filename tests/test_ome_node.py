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
from zarr_vectors.core.ome import (
    OME_ATTRS_KEY,
    OME_VERSION,
    WORLD,
    derive_store_name,
)
from zarr_vectors.core.store import (
    create_store,
    open_store,
    remove_resolution_level,
)

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
