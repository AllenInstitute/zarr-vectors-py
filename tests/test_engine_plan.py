"""ReadPlan: normalisation, algebra, and the miss vocabulary.

Pure value semantics — no store, no zarr, no fixtures.  That is the point
of having a plan type at all: what a read wants is inspectable and
comparable before any of it has been fetched.
"""

from __future__ import annotations

import pytest

from zarr_vectors._engine.plan import CellRequest, ReadPlan, RowRequest
from zarr_vectors._engine.snapshot import Snapshot
from zarr_vectors.core.group import _ABSENT


class TestNormalisation:
    def test_of_dedups_and_orders(self):
        a = ReadPlan.of(
            nodes=["b", "a", "b"],
            cells=[("z", "1"), ("a", "0"), ("z", "1")],
        )
        assert a.nodes == ("a", "b")
        assert a.cells == (CellRequest("a", "0"), CellRequest("z", "1"))

    def test_two_plans_wanting_the_same_io_compare_equal(self):
        # Canonical form is what lets a golden-plan test mean anything.
        a = ReadPlan.of(nodes=["x", "y"], cells=[("x", "0.0"), ("x", "1.0")])
        b = ReadPlan.of(nodes=["y", "x"], cells=[("x", "1.0"), ("x", "0.0")])
        assert a == b

    def test_row_requests_merge_per_array(self):
        plan = ReadPlan.of(
            rows=[RowRequest.of("m", [5, 1]), RowRequest.of("m", [1, 9])],
        )
        assert plan.rows == (RowRequest("m", (1, 5, 9)),)

    def test_row_request_of_dedups_and_sorts(self):
        assert RowRequest.of("m", [3, 1, 3, 2]).rows == (1, 2, 3)

    def test_for_cells_names_the_node_without_fanning_out(self):
        # The node is needed for the grid origin; fanning out from it
        # would turn a targeted read into a full scan.
        plan = ReadPlan.for_cells("0/vertices", ["0.0.0", "1.0.0"])
        assert plan.nodes == ("0/vertices",)
        assert plan.expand == ()
        assert len(plan.cells) == 2

    def test_for_array_asks_for_everything(self):
        plan = ReadPlan.for_array("0/vertices")
        assert plan.expand == ("0/vertices",)
        assert plan.cells == ()

    def test_len_and_bool(self):
        assert not ReadPlan()
        assert len(ReadPlan()) == 0
        assert ReadPlan.for_cells("a", ["0"])
        assert len(ReadPlan.for_cells("a", ["0"])) == 2  # node + cell


class TestMerge:
    def test_union(self):
        a = ReadPlan.of(nodes=["a"], cells=[("a", "0")])
        b = ReadPlan.of(nodes=["b"], cells=[("b", "0")])
        assert (a | b).nodes == ("a", "b")
        assert len((a | b).cells) == 2

    def test_merge_is_idempotent(self):
        a = ReadPlan.of(nodes=["a"], cells=[("a", "0")], arrays=["m"])
        assert a.merge(a) == a

    def test_merging_empty_returns_the_other_side(self):
        a = ReadPlan.of(nodes=["a"])
        assert a.merge(ReadPlan()) is a
        assert ReadPlan().merge(a) is a

    def test_without_listings(self):
        a = ReadPlan.of(nodes=["a"], listings=["links/0"])
        assert a.without_listings().listings == ()
        assert a.without_listings().nodes == ("a",)


class TestMinus:
    def test_drops_what_the_snapshot_holds(self):
        snap = Snapshot(nodes={"a": object()}, chunks={("a", "0"): b"x"})
        plan = ReadPlan.of(nodes=["a", "b"], cells=[("a", "0"), ("a", "1")])
        left = plan.minus(snap)
        assert left.nodes == ("b",)
        assert left.cells == (CellRequest("a", "1"),)

    def test_absent_node_counts_as_answered(self):
        # _ABSENT is how "probed and genuinely missing" is distinguished
        # from "never fetched". Re-requesting it would never terminate.
        snap = Snapshot(nodes={"gone": _ABSENT})
        assert not ReadPlan.of(nodes=["gone"]).minus(snap)

    def test_absent_cell_counts_as_answered(self):
        snap = Snapshot()
        snap.mark_absent(ReadPlan.of(cells=[("a", "0")]))
        assert not ReadPlan.of(cells=[("a", "0")]).minus(snap)

    def test_absent_array_and_listing_count_as_answered(self):
        snap = Snapshot()
        snap.mark_absent(ReadPlan.of(arrays=["m"], listings=["links/0"]))
        assert not ReadPlan.of(arrays=["m"], listings=["links/0"]).minus(snap)

    def test_rows_are_subtracted_individually(self):
        snap = Snapshot(rows={"m": {1: b"a", 2: b"b"}})
        left = ReadPlan.of(rows=[RowRequest.of("m", [1, 2, 3])]).minus(snap)
        assert left.rows == (RowRequest("m", (3,)),)

    def test_row_fully_covered_drops_the_request(self):
        snap = Snapshot(rows={"m": {1: b"a"}})
        assert not ReadPlan.of(rows=[RowRequest.of("m", [1])]).minus(snap)


class TestFromMisses:
    def test_decodes_all_four_kinds(self):
        plan = ReadPlan.from_misses({
            "0/vertices",                 # node
            ("0/vertices", "0.0.0"),      # cell
            ("array", "0/object_index"),  # whole array
            ("list", "0/links/0"),        # listing
        })
        assert plan.nodes == ("0/vertices",)
        assert plan.cells == (CellRequest("0/vertices", "0.0.0"),)
        assert plan.arrays == ("0/object_index",)
        assert plan.listings == ("0/links/0",)

    def test_round_trips_a_mixed_miss_set(self):
        misses = {"a", ("a", "0.0"), ("array", "m"), ("list", "g")}
        assert ReadPlan.from_misses(misses) == ReadPlan.of(
            nodes=["a"], cells=[("a", "0.0")], arrays=["m"], listings=["g"],
            expand=["a"],
        )

    def test_ignores_shapes_it_does_not_recognise(self):
        assert not ReadPlan.from_misses([("a", "b", "c"), 42, None])

    def test_a_cell_miss_fans_its_array_out(self):
        # Discovery is where fan-out belongs. Resolving the array yields
        # nonempty_chunks, from which the fetcher implies every cell it
        # holds -- so one round replaces one round per cell. Without it a
        # 64-chunk read_points took 66 rounds, 64 of them single cells of
        # vertex_fragments, an array no miss ever named directly.
        plan = ReadPlan.from_misses({("0/vertex_fragments", "0.0.0")})
        assert plan.expand == ("0/vertex_fragments",)
        assert plan.cells == (CellRequest("0/vertex_fragments", "0.0.0"),)

    def test_expand_is_not_duplicated_across_cells(self):
        plan = ReadPlan.from_misses({
            ("a", "0"), ("a", "1"), ("b", "0"), "a",
        })
        assert plan.expand == ("a", "b")
        assert plan.nodes == ("a",)  # only the explicit node miss


class TestByArray:
    def test_groups_cells_per_array(self):
        plan = ReadPlan.of(cells=[("a", "1"), ("b", "0"), ("a", "0")])
        assert plan.by_array() == [("a", ["0", "1"]), ("b", ["0"])]

    def test_prefix_rebases_onto_a_sub_group(self):
        # Group._prefetch_cache is keyed by the name the caller passes to
        # read_bytes, which is relative to that Group -- so a level
        # group's plan must say "vertices", not "0/vertices".
        plan = ReadPlan.of(cells=[("0/vertices", "0.0.0")])
        assert plan.by_array(prefix="0") == [("vertices", ["0.0.0"])]

    def test_prefix_drops_cells_outside_it(self):
        # A level-0 handle cannot address a level-1 array, so offering it
        # one would produce a cache entry nothing could ever hit.
        plan = ReadPlan.of(cells=[("0/vertices", "0.0.0"), ("1/vertices", "0.0.0")])
        assert plan.by_array(prefix="0") == [("vertices", ["0.0.0"])]

    def test_prefix_handles_the_group_itself(self):
        plan = ReadPlan.of(cells=[("0", "0.0.0")])
        assert plan.by_array(prefix="0") == [("", ["0.0.0"])]


class TestCost:
    def test_counts_every_kind(self):
        cost = ReadPlan.of(
            nodes=["a"],
            cells=[("a", "0"), ("a", "1")],
            rows=[RowRequest.of("m", [1, 2, 3])],
            arrays=["m"],
            listings=["g"],
        ).cost()
        assert (cost.nodes, cost.cells, cost.rows) == (1, 2, 1)
        assert cost.row_indices == 3
        assert (cost.arrays, cost.listings) == (1, 1)
        assert cost.total == 6

    def test_explain_names_the_empty_plan(self):
        assert "empty plan" in ReadPlan().explain()

    def test_explain_summarises(self):
        assert "2 cell(s)" in ReadPlan.of(cells=[("a", "0"), ("a", "1")]).explain()


def test_plan_is_immutable():
    plan = ReadPlan.of(nodes=["a"])
    with pytest.raises((AttributeError, TypeError)):
        plan.nodes = ("b",)  # type: ignore[misc]
