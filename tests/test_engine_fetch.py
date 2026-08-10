"""The Fetcher port.

The property under test is narrowness.  A one-method port is only worth
having if a fake over a plain dict is genuinely interchangeable with a
live store — otherwise the abstraction leaks and the layers above it are
quietly coupled to zarr after all.  So :class:`DictFetcher` and
:class:`GroupFetcher` are exercised against the same expectations.

The second property is that a missing capability produces a *different*
plan's worth of work, not an exception: a store that cannot list must
still answer everything else.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from zarr_vectors._engine.fetch import (
    Capabilities,
    DictFetcher,
    Fetcher,
    GroupFetcher,
    RecordingFetcher,
)
from zarr_vectors._engine.plan import ReadPlan, RowRequest
from zarr_vectors.core.group import _ABSENT, Group


def _memory_root() -> Group:
    """A Group at the store root, backed by memory."""
    return Group._from_zarr(zarr.open_group(store=MemoryStore(), mode="a"))


def _with_chunk_array(root: Group, name: str, keys: dict[str, bytes]) -> Group:
    """Allocate a per-chunk vlen array and fill the named cells."""
    root.create_sharded_chunk_array(name, grid_shape=(2, 2, 2), shard_shape=None)
    for key, payload in keys.items():
        root.write_bytes(name, key, payload)
    return root


class TestProtocol:
    def test_the_port_is_one_method(self):
        # If this grows, every fake grows with it -- which is exactly how
        # zarr's StorageTransformer became untestable.
        assert {
            m for m in Fetcher.__protocol_attrs__ if not m.startswith("_")
        } == {"capabilities", "fetch"}

    def test_dict_fetcher_satisfies_it(self):
        assert isinstance(DictFetcher(), Fetcher)


class TestDictFetcher:
    def test_serves_cells(self):
        f = DictFetcher(cells={("0/vertices", "0.0.0"): b"payload"})
        snap = f.fetch(ReadPlan.of(cells=[("0/vertices", "0.0.0")]))
        assert snap.chunks[("0/vertices", "0.0.0")] == b"payload"

    def test_unknown_node_resolves_to_absent_not_a_hole(self):
        # "Probed and missing" must be as answerable as "found", or a
        # legitimate array_exists->False looks like a prefetch gap.
        snap = DictFetcher().fetch(ReadPlan.of(nodes=["0/nope"]))
        assert snap.nodes["0/nope"] is _ABSENT

    def test_missing_cell_is_simply_absent_from_the_result(self):
        snap = DictFetcher().fetch(ReadPlan.of(cells=[("a", "0")]))
        assert ("a", "0") not in snap.chunks

    def test_serves_whole_arrays_and_listings(self):
        f = DictFetcher(
            arrays={"0/object_index": np.arange(4)},
            listings={"0/links/0": ["0.0.0", "0.0.+1"]},
        )
        snap = f.fetch(ReadPlan.of(arrays=["0/object_index"], listings=["0/links/0"]))
        assert list(snap.arrays["0/object_index"]) == [0, 1, 2, 3]
        assert snap.listings["0/links/0"] == ["0.0.0", "0.0.+1"]

    def test_serves_selected_rows(self):
        f = DictFetcher(arrays={"m": np.array([10, 11, 12, 13])})
        snap = f.fetch(ReadPlan.of(rows=[RowRequest.of("m", [1, 3])]))
        assert snap.rows["m"] == {1: 11, 3: 13}

    def test_cannot_list_yields_no_listings_rather_than_raising(self):
        # A browser fetch-backed Store has no listing operation. The read
        # must degrade, not fail.
        f = DictFetcher(
            listings={"0/links/0": ["0.0.0"]},
            capabilities=Capabilities(can_list=False),
        )
        snap = f.fetch(ReadPlan.of(listings=["0/links/0"]))
        assert snap.listings == {}

    def test_with_capabilities_keeps_the_data(self):
        f = DictFetcher(cells={("a", "0"): b"x"}).with_capabilities(can_list=False)
        assert f.capabilities.can_list is False
        assert f.fetch(ReadPlan.of(cells=[("a", "0")])).chunks

    def test_records_every_plan_it_was_given(self):
        f = DictFetcher()
        f.fetch(ReadPlan.of(nodes=["a"]))
        f.fetch(ReadPlan.of(nodes=["b"]))
        assert [p.nodes for p in f.calls] == [("a",), ("b",)]

    def test_empty_plan_is_a_no_op(self):
        f = DictFetcher()
        snap = f.fetch(ReadPlan())
        assert not snap.nodes and not snap.chunks


class TestGroupFetcher:
    def test_rejects_a_non_root_group(self):
        # Plans key paths root-relative; a sub-group handle would resolve
        # every one against the wrong base and silently fetch nothing.
        root = _memory_root()
        root.create_group("0")
        with pytest.raises(ValueError, match="root"):
            GroupFetcher(root["0"])

    def test_fetches_cells_from_a_live_store(self):
        root = _memory_root()
        level = root.create_group("0")
        _with_chunk_array(level, "vertices", {"0.0.0": b"aaa", "1.0.0": b"bbb"})

        snap = GroupFetcher(root).fetch(
            ReadPlan.for_cells("0/vertices", ["0.0.0", "1.0.0"])
        )
        assert snap.chunks[("0/vertices", "0.0.0")] == b"aaa"
        assert snap.chunks[("0/vertices", "1.0.0")] == b"bbb"

    def test_expand_implies_every_cell_the_array_holds(self):
        # An array's nonempty_chunks arrives with its metadata, so one
        # node resolution surfaces every cell it holds.
        root = _memory_root()
        level = root.create_group("0")
        _with_chunk_array(level, "vertices", {"0.0.0": b"aaa", "1.1.1": b"ccc"})

        snap = GroupFetcher(root).fetch(ReadPlan.for_array("0/vertices"))
        assert ("0/vertices", "0.0.0") in snap.chunks
        assert ("0/vertices", "1.1.1") in snap.chunks

    def test_naming_a_node_alone_does_not_fan_out(self):
        # The difference between "read this box" and "scan the level".
        root = _memory_root()
        level = root.create_group("0")
        _with_chunk_array(level, "vertices", {"0.0.0": b"aaa", "1.1.1": b"ccc"})

        snap = GroupFetcher(root).fetch(ReadPlan.of(nodes=["0/vertices"]))
        assert snap.nodes["0/vertices"] is not None
        assert snap.chunks == {}

    def test_missing_node_resolves_to_absent(self):
        snap = GroupFetcher(_memory_root()).fetch(ReadPlan.of(nodes=["0/nope"]))
        assert snap.nodes["0/nope"] is _ABSENT

    def test_fetches_listings(self):
        root = _memory_root()
        level = root.create_group("0")
        level.create_group("links")
        _with_chunk_array(level, "vertices", {"0.0.0": b"x"})

        snap = GroupFetcher(root).fetch(ReadPlan.of(listings=["0"]))
        assert set(snap.listings["0"]) >= {"links", "vertices"}

    def test_declares_capabilities(self):
        assert GroupFetcher(_memory_root()).capabilities.can_gather is True

    def test_serial_path_matches_the_gathered_one(self):
        # can_gather=False is the icechunk shape. Same answers, one
        # request at a time -- so a store that cannot be gathered against
        # is slower, never wrong.
        root = _memory_root()
        level = root.create_group("0")
        _with_chunk_array(level, "vertices", {"0.0.0": b"aaa", "1.0.0": b"bbb"})
        plan = ReadPlan.for_cells("0/vertices", ["0.0.0", "1.0.0"])

        gathered = GroupFetcher(root).fetch(plan)
        serial = GroupFetcher(
            root, capabilities=Capabilities(can_gather=False),
        ).fetch(plan)
        assert gathered.chunks == serial.chunks
        assert gathered.nodes.keys() == serial.nodes.keys()


class TestRecordingFetcher:
    def test_delegates_and_records(self):
        inner = DictFetcher(cells={("a", "0"): b"x"})
        rec = RecordingFetcher(inner)
        snap = rec.fetch(ReadPlan.of(cells=[("a", "0")]))
        assert snap.chunks[("a", "0")] == b"x"
        assert len(rec.plans) == 1

    def test_rounds_ignores_empty_plans(self):
        rec = RecordingFetcher(DictFetcher())
        rec.fetch(ReadPlan())
        rec.fetch(ReadPlan.of(nodes=["a"]))
        assert rec.rounds == 1

    def test_passes_capabilities_through(self):
        inner = DictFetcher(capabilities=Capabilities(can_list=False))
        assert RecordingFetcher(inner).capabilities.can_list is False
