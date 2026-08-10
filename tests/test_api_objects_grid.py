"""Selective object reads, and the grid as geometry.

Both exist because downstream had to reach past the API to get them: one
consumer imports a private manifest-block expander, a layout sentinel and
a raw coordinate selection to avoid decoding twenty-one million object
manifests; another re-derives the chunk allocator to predict whether its
chunk ids will fit.
"""

from __future__ import annotations

import numpy as np
import pytest

import zarr_vectors as zv
from zarr_vectors.core.arrays import (
    expand_manifest_blocks,
    object_count,
    read_all_object_manifests,
    read_object_manifests,
)
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.types.points import write_points
from zarr_vectors.types.polylines import write_polylines

N_OBJECTS = 300


@pytest.fixture
def tracts(tmp_path):
    rng = np.random.default_rng(21)
    path = tmp_path / "tracts.zarrvectors"
    lines = [
        (rng.normal(0, 15, size=(8, 3)).cumsum(axis=0) + 300).astype(np.float32)
        for _ in range(N_OBJECTS)
    ]
    write_polylines(path, lines, chunk_shape=(200.0, 200.0, 200.0),
                    bin_shape=(50.0, 50.0, 50.0))
    return path


@pytest.fixture
def cloud(tmp_path):
    rng = np.random.default_rng(22)
    path = tmp_path / "cloud.zarrvectors"
    write_points(
        path, rng.uniform(0, 800, size=(2000, 3)).astype(np.float32),
        chunk_shape=(100.0, 100.0, 100.0), bin_shape=(50.0, 50.0, 50.0),
    )
    return path


def _level_group(path):
    return get_resolution_level(open_store(str(path)), 0)


class TestSelectiveManifests:
    def test_a_subset_matches_the_full_decode(self, tracts):
        group = _level_group(tracts)
        everything = read_all_object_manifests(group)
        subset = read_object_manifests(group, ids=[3, 17, 200])
        assert sorted(subset) == [3, 17, 200]
        assert all(subset[i] == everything[i] for i in subset)

    def test_reading_all_matches_the_legacy_entry_point(self, tracts):
        group = _level_group(tracts)
        assert read_object_manifests(group) == dict(
            enumerate(read_all_object_manifests(group))
        )

    def test_ids_that_do_not_exist_are_dropped(self, tracts):
        # So a caller can pass a superset without pre-filtering -- which
        # is the normal case when ids come from somewhere else.
        assert read_object_manifests(_level_group(tracts), ids=[10**9]) == {}

    def test_empty_id_list_reads_nothing(self, tracts):
        assert read_object_manifests(_level_group(tracts), ids=[]) == {}

    def test_duplicate_ids_collapse(self, tracts):
        got = read_object_manifests(_level_group(tracts), ids=[5, 5, 5])
        assert sorted(got) == [5]

    def test_object_count_reads_no_manifests(self, tracts):
        assert object_count(_level_group(tracts)) == N_OBJECTS

    def test_object_count_on_a_store_without_objects(self, cloud):
        assert object_count(_level_group(cloud)) >= 0

    def test_the_private_block_expander_still_resolves(self):
        # Downstream imports the pre-promotion spelling. Renaming it
        # without an alias would break them for no gain.
        from zarr_vectors.core.arrays import _expand_blocks

        assert _expand_blocks is expand_manifest_blocks

    def test_plural_element_read_matches_the_singular_one(self, tracts):
        group = _level_group(tracts)
        path = "object_index/manifests"
        want = [1, 4, 9]
        assert group.read_vlen_elements(path, want) == [
            group.read_vlen_element(path, i) for i in want
        ]

    def test_plural_element_read_preserves_order(self, tracts):
        group = _level_group(tracts)
        path = "object_index/manifests"
        forwards = group.read_vlen_elements(path, [1, 4, 9])
        backwards = group.read_vlen_elements(path, [9, 4, 1])
        assert forwards == list(reversed(backwards))


class TestObjectCatalog:
    def test_length_comes_from_metadata(self, tracts):
        assert len(zv.open(tracts).level(0).objects) == N_OBJECTS

    def test_indexing_reads_one_object(self, tracts):
        result = zv.open(tracts).level(0).objects[7]
        assert result.part_count == 1
        assert int(result.part_objects[0]) == 7

    def test_indexing_a_list_reads_several(self, tracts):
        result = zv.open(tracts).level(0).objects[[7, 9, 11]]
        assert result.part_count == 3
        assert sorted(int(o) for o in result.part_objects) == [7, 9, 11]

    def test_slicing_works(self, tracts):
        assert zv.open(tracts).level(0).objects[0:4].part_count == 4

    def test_membership(self, tracts):
        catalog = zv.open(tracts).level(0).objects
        assert 0 in catalog
        assert N_OBJECTS not in catalog

    def test_manifests_are_reachable_without_private_imports(self, tracts):
        manifests = zv.open(tracts).level(0).objects.manifests(ids=[2, 3])
        assert sorted(manifests) == [2, 3]
        chunk_coords, fragment = manifests[2][0]
        assert isinstance(fragment, int)
        assert len(chunk_coords) == 3


class TestGrid:
    def test_plan_predicts_the_allocation_before_the_store_exists(self):
        grid = zv.Grid.plan(([0, 0, 0], [800, 800, 800]), cell_size=(100.0,) * 3)
        assert grid.shape == (8, 8, 8)
        assert grid.cells == 512

    def test_plan_from_a_target_cell_count(self):
        grid = zv.Grid.plan(([0, 0, 0], [800, 800, 800]), target_cells=4)
        assert grid.shape == (4, 4, 4)
        assert grid.cell_shape == (200.0, 200.0, 200.0)

    def test_a_levels_grid_matches_how_it_was_written(self, cloud):
        assert zv.open(cloud).level(0).grid.shape == (8, 8, 8)

    def test_cells_in_replaces_hand_built_chunk_coordinates(self, cloud):
        cells = zv.open(cloud).level(0).grid.cells_in(
            ([0.0, 0.0, 0.0], [150.0, 150.0, 150.0]),
        )
        assert cells.keys() == (
            "0.0.0", "0.0.1", "0.1.0", "0.1.1",
            "1.0.0", "1.0.1", "1.1.0", "1.1.1",
        )

    def test_cell_of_locates_a_point(self, cloud):
        assert zv.open(cloud).level(0).grid.cell_of([250.0, 50.0, 10.0]).key == "2.0.0"

    def test_holds_rejects_a_coordinate_outside_the_allocation(self, cloud):
        # The question one consumer re-derives the allocator to answer.
        grid = zv.open(cloud).level(0).grid
        assert grid.holds(grid.cell_of([10.0, 10.0, 10.0]))
        assert not grid.holds(zv.CellRef((99, 0, 0)))

    def test_capacity_flags_an_unwieldy_grid(self):
        tiny = zv.Grid.plan(([0, 0, 0], [800, 800, 800]), target_cells=1)
        verdict = tiny.capacity(n_vertices=2_000_000_000)
        assert not verdict.fits
        assert "use more cells" in (verdict.reason or "")

    def test_capacity_accepts_a_sensible_grid(self):
        grid = zv.Grid.plan(([0, 0, 0], [800, 800, 800]), target_cells=8)
        assert grid.capacity(n_vertices=20_000_000).fits

    def test_capacity_str_is_readable(self):
        grid = zv.Grid.plan(([0, 0, 0], [800, 800, 800]), target_cells=8)
        assert "512 cells" in str(grid.capacity(n_vertices=1000))


class TestCellSelection:
    def test_cells_narrows_a_read(self, cloud):
        level = zv.open(cloud).level(0)
        cells = level.grid.cells_in(([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]))
        got = level.select(cells=cells).read()
        assert got.vertex_count > 0
        # Everything returned must be in one of the requested cells.
        keys = set(cells.keys())
        assert all(level.grid.cell_of(p).key in keys for p in got.positions)

    def test_cells_and_bbox_agree(self, cloud):
        level = zv.open(cloud).level(0)
        box = ([0.0, 0.0, 0.0], [100.0, 100.0, 100.0])
        by_cells = level.select(cells=level.grid.cells_in(box)).read()
        by_bbox = level.select(bbox=box).read()
        # The cell form is coarser: it keeps whole cells, so it is a
        # superset of the box.
        assert by_cells.vertex_count >= by_bbox.vertex_count

    def test_the_plan_honours_explicit_cells(self, cloud):
        level = zv.open(cloud).level(0)
        cells = level.grid.cells_in(([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]))
        plan = level.select(cells=cells).plan()
        assert plan.expand == ()
        assert {c.key for c in plan.cells} == set(cells.keys())
