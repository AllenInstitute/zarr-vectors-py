"""Named object groups, and a sanctioned place for application metadata.

Both replace a downstream workaround.  Groups were addressed by bare row
index, so a store could not be understood without the writing
application's source next to it.  Application metadata went straight into
the group's ``attrs``, where it can collide with the format's own keys
and where two writers touching different keys still lose each other's.
"""

from __future__ import annotations

import numpy as np
import pytest

import zarr_vectors as zv
from zarr_vectors.core.user_metadata import USER_METADATA_KEY
from zarr_vectors.types.polylines import write_polylines


@pytest.fixture
def grouped(tmp_path):
    rng = np.random.default_rng(31)
    path = tmp_path / "grouped.zarrvectors"
    lines = [
        (rng.normal(0, 15, size=(6, 3)).cumsum(axis=0) + 300).astype(np.float32)
        for _ in range(30)
    ]
    write_polylines(
        path, lines, chunk_shape=(200.0, 200.0, 200.0), bin_shape=(50.0, 50.0, 50.0),
        groups={0: list(range(15)), 1: list(range(15, 30))},
    )
    return path


class TestNamedGroups:
    def test_an_unnamed_store_is_still_addressable(self, grouped):
        # Stores written before names existed must stay usable, and by
        # something better than a bare integer at the call site.
        assert zv.open(grouped).groups.names() == ("group_0", "group_1")

    def test_naming_is_additive(self, grouped):
        ds = zv.open(grouped, mode="r+")
        before = {n: len(ds.groups[n]) for n in ds.groups.names()}
        ds.groups.name_rows(["network", "fragments"])
        after = {n: len(zv.open(grouped).groups[n]) for n in ("network", "fragments")}
        assert list(before.values()) == list(after.values())

    def test_lookup_by_name(self, grouped):
        ds = zv.open(grouped, mode="r+")
        ds.groups.name_rows(["network", "fragments"])
        group = zv.open(grouped).groups["network"]
        assert group.id == 0
        assert sorted(group.members.tolist()) == list(range(15))

    def test_unknown_name_says_what_is_there(self, grouped):
        with pytest.raises(KeyError, match="group_0"):
            zv.open(grouped).groups["nonexistent"]

    def test_reading_a_group_returns_its_objects(self, grouped):
        ds = zv.open(grouped, mode="r+")
        ds.groups.name_rows(["network", "fragments"])
        assert zv.open(grouped).groups["network"].read().part_count == 15

    def test_by_id_still_works_for_unnamed_stores(self, grouped):
        assert zv.open(grouped).groups.by_id(1).id == 1

    def test_catalog_is_a_mapping(self, grouped):
        catalog = zv.open(grouped).groups
        assert len(catalog) == 2
        assert set(catalog) == {"group_0", "group_1"}
        assert "group_0" in catalog

    def test_a_store_with_no_groups_is_empty_not_broken(self, tmp_path):
        from zarr_vectors.types.points import write_points

        path = tmp_path / "nogroups.zarrvectors"
        write_points(
            path, np.random.default_rng(1).uniform(0, 100, (50, 3)).astype(np.float32),
            chunk_shape=(100.0, 100.0, 100.0),
        )
        assert zv.open(path).groups.names() == ()


class TestUserMetadata:
    def test_round_trips(self, grouped):
        ds = zv.open(grouped, mode="r+")
        ds.metadata.namespace("bridge")["run_id"] = "abc123"
        assert zv.open(grouped).metadata.namespace("bridge")["run_id"] == "abc123"

    def test_namespaces_do_not_collide(self, grouped):
        ds = zv.open(grouped, mode="r+")
        ds.metadata.namespace("a")["k"] = 1
        ds.metadata.namespace("b")["k"] = 2
        reopened = zv.open(grouped)
        assert reopened.metadata.namespace("a")["k"] == 1
        assert reopened.metadata.namespace("b")["k"] == 2

    def test_it_stays_out_of_the_formats_own_keys(self, grouped):
        # One reserved key, so an application can never shadow a format
        # field by choosing an unlucky name.
        ds = zv.open(grouped, mode="r+")
        ds.metadata.namespace("app")["bounds"] = "not the real bounds"
        assert USER_METADATA_KEY in ds.store.attrs.to_dict()
        assert zv.open(grouped).bounds[0].shape == (3,)

    def test_update_is_one_read_modify_write(self, grouped):
        ds = zv.open(grouped, mode="r+")
        ds.metadata.namespace("app").update({"a": 1, "b": 2})
        assert dict(zv.open(grouped).metadata.namespace("app")) == {"a": 1, "b": 2}

    def test_transact_writes_once(self, grouped):
        ds = zv.open(grouped, mode="r+")
        with ds.metadata.namespace("app").transact() as block:
            block["x"] = 1
            block["y"] = 2
        assert sorted(zv.open(grouped).metadata.namespace("app")) == ["x", "y"]

    def test_compare_and_set_guards_the_value(self, grouped):
        ds = zv.open(grouped, mode="r+")
        ns = ds.metadata.namespace("app")
        assert ns.compare_and_set("state", None, "claimed") is True
        assert ns.compare_and_set("state", None, "claimed again") is False
        assert ns["state"] == "claimed"

    def test_behaves_like_a_dict(self, grouped):
        ns = zv.open(grouped, mode="r+").metadata.namespace("app")
        ns["a"] = 1
        assert list(ns) == ["a"]
        assert len(ns) == 1
        del ns["a"]
        assert len(ns) == 0

    def test_missing_key_names_the_namespace(self, grouped):
        with pytest.raises(KeyError, match="app"):
            zv.open(grouped).metadata.namespace("app")["nope"]

    def test_namespace_names_are_validated(self, grouped):
        with pytest.raises(ValueError, match="must be non-empty"):
            zv.open(grouped).metadata.namespace("a/b")

    def test_levels_have_their_own(self, grouped):
        ds = zv.open(grouped, mode="r+")
        ds.metadata.namespace("app")["where"] = "root"
        ds.level(0).metadata.namespace("app")["where"] = "level"
        reopened = zv.open(grouped)
        assert reopened.metadata.namespace("app")["where"] == "root"
        assert reopened.level(0).metadata.namespace("app")["where"] == "level"

    def test_drop_removes_a_namespace(self, grouped):
        ds = zv.open(grouped, mode="r+")
        ds.metadata.namespace("app")["a"] = 1
        ds.metadata.drop("app")
        assert "app" not in zv.open(grouped).metadata.names()
