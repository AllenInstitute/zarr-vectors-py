"""Resolvers, and the golden plans that pin the I/O a read performs.

A plan is a value computed from metadata alone, so these tests need no
store: what a read *wants* is checkable before any of it is fetched.

The golden-plan tests are the decoupling guard.  An internal change that
alters the shape of a read shows up here as a one-line diff in a
comparable value, rather than downstream as a performance mystery — and
the fix is in a resolver rather than in anybody's code.
"""

from __future__ import annotations

import numpy as np
import pytest

import zarr_vectors as zv
from zarr_vectors._engine.plan import ReadPlan
from zarr_vectors._engine.resolve import LevelContext, resolve
from zarr_vectors.api.select import Selection
from zarr_vectors.types.points import write_points

# An 8x8x8 grid: big enough that fanning out when a box was asked for is
# obviously the wrong answer.
CTX = LevelContext(
    level=0,
    ndim=3,
    chunk_shape=(100.0, 100.0, 100.0),
    bounds=((0.0, 0.0, 0.0), (800.0, 800.0, 800.0)),
    attribute_names=("intensity", "label"),
    has_object_index=True,
)


class TestFullReads:
    def test_a_full_read_fans_out(self):
        plan = resolve(Selection(), CTX)
        assert plan.expand == (
            "0/vertex_attributes/intensity",
            "0/vertex_attributes/label",
            "0/vertex_fragments",
            "0/vertices",
        )
        assert plan.cells == ()

    def test_a_full_read_names_no_cells_it_cannot_know(self):
        # Which cells exist is not derivable from metadata, so a full
        # read must ask the arrays rather than guess.
        assert resolve(Selection(), CTX).cells == ()

    def test_the_object_index_is_read_whole(self):
        assert resolve(Selection(), CTX).arrays == ("0/object_index/manifests",)

    def test_no_object_index_means_no_whole_array_read(self):
        ctx = LevelContext(level=0, ndim=3, chunk_shape=CTX.chunk_shape)
        assert resolve(Selection(), ctx).arrays == ()


class TestTargetedReads:
    def test_a_bbox_names_cells_and_refuses_to_fan_out(self):
        # The whole point: 8 of 512 cells, not 512.
        plan = resolve(
            Selection(bbox=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0])), CTX,
        )
        assert plan.expand == ()
        keys = {c.key for c in plan.cells}
        assert keys == {"0.0.0", "0.0.1", "0.1.0", "0.1.1",
                        "1.0.0", "1.0.1", "1.1.0", "1.1.1"}

    def test_a_bbox_still_resolves_the_nodes(self):
        # The reader needs their metadata -- the grid origin among it --
        # so leaving them out costs a round without saving anything.
        plan = resolve(
            Selection(bbox=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0])), CTX,
        )
        assert "0/vertices" in plan.nodes
        assert "0/vertex_fragments" in plan.nodes

    def test_a_bbox_covers_every_array_the_read_touches(self):
        plan = resolve(
            Selection(bbox=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0])), CTX,
        )
        arrays = {c.array for c in plan.cells}
        assert arrays == {
            "0/vertices", "0/vertex_fragments",
            "0/vertex_attributes/intensity", "0/vertex_attributes/label",
        }

    def test_near_becomes_its_bounding_box(self):
        # The sphere itself is enforced after the read; the fetch is
        # driven by the box that contains it.
        sphere = resolve(Selection(near=([400.0, 400.0, 400.0], 50.0)), CTX)
        box = resolve(
            Selection(bbox=([350.0, 350.0, 350.0], [450.0, 450.0, 450.0])), CTX,
        )
        assert {c.key for c in sphere.cells} == {c.key for c in box.cells}

    def test_a_tiny_box_costs_one_cell_per_array(self):
        plan = resolve(
            Selection(bbox=([10.0, 10.0, 10.0], [20.0, 20.0, 20.0])), CTX,
        )
        assert {c.key for c in plan.cells} == {"0.0.0"}
        assert len(plan.cells) == 4  # one per array


class TestAttributeNarrowing:
    def test_all_includes_every_attribute_array(self):
        plan = resolve(Selection(), CTX)
        assert "0/vertex_attributes/intensity" in plan.expand
        assert "0/vertex_attributes/label" in plan.expand

    def test_naming_one_attribute_excludes_the_others(self):
        # Fetching an attribute the reader will not even decode is pure
        # waste on a latency-bound store.
        plan = resolve(Selection(attributes=["label"]), CTX)
        assert "0/vertex_attributes/label" in plan.expand
        assert "0/vertex_attributes/intensity" not in plan.expand

    def test_no_attributes_leaves_only_the_core_arrays(self):
        plan = resolve(Selection(attributes=[]), CTX)
        assert plan.expand == ("0/vertex_fragments", "0/vertices")


class TestDegradation:
    def test_no_declared_grid_falls_back_to_fanning_out(self):
        # Without a chunk shape there is no arithmetic to do, so the
        # honest answer is to ask for everything rather than to guess.
        ctx = LevelContext(level=0, ndim=3, chunk_shape=None)
        plan = resolve(Selection(bbox=([0.0, 0.0, 0.0], [10.0, 10.0, 10.0])), ctx)
        assert plan.expand
        assert plan.cells == ()

    def test_a_mismatched_bbox_falls_back_rather_than_producing_nonsense(self):
        plan = resolve(Selection(bbox=([0.0, 0.0], [10.0, 10.0])), CTX)
        assert plan.expand
        assert plan.cells == ()

    def test_a_box_larger_than_the_level_walks_the_level(self):
        # The box spans 1000 cells; the level holds two. Enumerating the
        # box to intersect it would cost 1000 tuples to find 1 -- the
        # scan has to run the other way round when the level is the
        # smaller of the two.
        ctx = LevelContext(
            level=0, ndim=3, chunk_shape=CTX.chunk_shape,
            known_cells=("0.0.0", "9.9.9"),
        )
        plan = resolve(
            Selection(bbox=([0.0, 0.0, 0.0], [500.0, 500.0, 500.0])), ctx,
        )
        assert {c.key for c in plan.cells} == {"0.0.0"}

    def test_a_binned_key_is_matched_on_its_spatial_tail(self):
        # chunk_by_attribute prefixes every key with a bin axis, so a
        # spatial box compared against the whole key matches nothing.
        ctx = LevelContext(
            level=0, ndim=3, chunk_shape=CTX.chunk_shape,
            known_cells=("0.0.0.0", "3.0.0.0", "3.9.9.9"),
        )
        plan = resolve(
            Selection(bbox=([0.0, 0.0, 0.0], [500.0, 500.0, 500.0])), ctx,
        )
        assert {c.key for c in plan.cells} == {"0.0.0.0", "3.0.0.0"}

    def test_named_cells_the_level_does_not_hold_are_dropped(self):
        # The caller's region is theirs and is not re-derived, but a cell
        # the level is known not to hold fetches nothing, so asking for
        # it only spends round-trips.
        from zarr_vectors.api.grid import CellRef

        ctx = LevelContext(
            level=0, ndim=3, chunk_shape=CTX.chunk_shape, known_cells=("0.0.0",),
        )
        plan = resolve(
            Selection(cells=[CellRef((0, 0, 0)), CellRef((7, 7, 7))]), ctx,
        )
        assert {c.key for c in plan.cells} == {"0.0.0"}

    def test_named_cells_survive_when_occupancy_is_unknown(self):
        from zarr_vectors.api.grid import CellRef

        ctx = LevelContext(level=0, ndim=3, chunk_shape=CTX.chunk_shape)
        plan = resolve(
            Selection(cells=[CellRef((0, 0, 0)), CellRef((7, 7, 7))]), ctx,
        )
        assert {c.key for c in plan.cells} == {"0.0.0", "7.7.7"}

    def test_known_cells_prune_the_plan(self):
        # When the caller already knows which cells exist, asking for
        # empty ones is wasted round-trip budget.
        ctx = LevelContext(
            level=0, ndim=3, chunk_shape=CTX.chunk_shape, known_cells=("0.0.0",),
        )
        plan = resolve(
            Selection(bbox=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0])), ctx,
        )
        assert {c.key for c in plan.cells} == {"0.0.0"}


class TestPlanIsAValue:
    def test_resolving_twice_gives_an_equal_plan(self):
        assert resolve(Selection(), CTX) == resolve(Selection(), CTX)

    def test_different_selections_give_different_plans(self):
        assert resolve(Selection(), CTX) != resolve(
            Selection(bbox=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0])), CTX,
        )

    def test_resolving_performs_no_io(self):
        # LevelContext is pure metadata, so this is structural: there is
        # no store to touch. Stated as a test because it is the property
        # that lets a plan be compared against a golden.
        assert isinstance(resolve(Selection(), CTX), ReadPlan)


# =====================================================================
# Golden plans -- against a real store, through the public API
# =====================================================================


@pytest.fixture
def store(tmp_path):
    rng = np.random.default_rng(4)
    path = tmp_path / "golden.zarrvectors"
    write_points(
        path,
        rng.uniform(0, 800, size=(4000, 3)).astype(np.float32),
        chunk_shape=(100.0, 100.0, 100.0),
        bin_shape=(50.0, 50.0, 50.0),
        vertex_attributes={"a": rng.random(4000).astype(np.float32)},
    )
    return path


# (selection kwargs, nodes, cells, arrays, expand) -- the I/O shape each
# query should imply. A diff here is a deliberate decision, not a
# surprise.
GOLDEN = [
    ("full level", {}, 1, 0, 1, 3),
    ("one cell", {"bbox": ([10.0, 10.0, 10.0], [20.0, 20.0, 20.0])}, 4, 3, 1, 0),
    ("eight cells", {"bbox": ([0.0, 0.0, 0.0], [100.0, 100.0, 100.0])}, 4, 24, 1, 0),
    ("sphere", {"near": ([400.0, 400.0, 400.0], 50.0)}, 4, 24, 1, 0),
    ("one attribute", {"attributes": ["a"]}, 1, 0, 1, 3),
    ("no attributes", {"attributes": []}, 1, 0, 1, 2),
]


@pytest.mark.parametrize("label,kwargs,nodes,cells,arrays,expand", GOLDEN)
def test_golden_plan(store, label, kwargs, nodes, cells, arrays, expand):
    cost = zv.open(store).select(**kwargs).plan().cost()
    assert (cost.nodes, cost.cells, cost.arrays, cost.expand) == (
        nodes, cells, arrays, expand,
    ), f"{label}: the I/O shape of this read changed"


def test_a_targeted_read_does_not_scan_the_level(store):
    # 512 cells exist. A read of one of them must not name them all --
    # this is the property the whole resolver exists for.
    plan = zv.open(store).select(
        bbox=([10.0, 10.0, 10.0], [20.0, 20.0, 20.0]),
    ).plan()
    assert plan.expand == ()
    assert len(plan.cells) < 10


def test_plan_costs_nothing_to_build(store):
    # Building a plan must not read data, or `explain()` becomes an
    # expensive debugging tool nobody uses.
    ds = zv.open(store)
    query = ds.select(bbox=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]))
    assert query.plan() == query.plan()


def test_explain_reports_the_plan(store):
    text = zv.open(store).select(bbox=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0])).explain()
    assert "read_points" in text


@pytest.fixture
def sparse_store(tmp_path):
    """1000 points in a grid that allocates 68,921 cells.

    The shape a real sparse dataset has: a bounding box covering the
    whole specimen, data in a thin part of it.
    """
    rng = np.random.default_rng(6)
    path = tmp_path / "sparse.zarrvectors"
    write_points(
        path,
        rng.uniform(0, 400, size=(1000, 3)).astype(np.float32),
        chunk_shape=(10.0, 10.0, 10.0),
        bounds=([0.0, 0.0, 0.0], [400.0, 400.0, 400.0]),
    )
    return path


def test_a_bbox_plan_scales_with_occupancy_not_allocation(sparse_store):
    # The grid allocates 41^3 = 68,921 cells and ~1000 hold data. Before
    # the level's presence manifest reached the resolver, a whole-domain
    # box planned one fetch per allocated cell per array -- 137,842 of
    # them -- and the fetcher performed every one.
    ds = zv.open(sparse_store)
    level = ds.level(0)
    occupied = len(level.store.list_chunks("vertices"))
    plan = ds.select(bbox=ds.bounds).plan()

    assert plan.expand == ()
    # Two arrays (vertices, vertex_fragments) per occupied cell.
    assert len(plan.cells) <= 2 * occupied
    assert len(plan.cells) < 5000


def test_a_sparse_whole_domain_read_returns_every_vertex(sparse_store):
    # Narrowing to occupancy must not lose data: the pruned cells are
    # empty ones, so the answer is unchanged.
    ds = zv.open(sparse_store)
    assert ds.select(bbox=ds.bounds).read().vertex_count == 1000


def test_naming_every_grid_cell_still_only_fetches_the_full_ones(sparse_store):
    ds = zv.open(sparse_store)
    level = ds.level(0)
    occupied = len(level.store.list_chunks("vertices"))
    plan = level.select(cells=level.grid.cells_in(ds.bounds)).plan()
    assert len(plan.cells) <= 2 * occupied
