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


@pytest.fixture
def offset_cloud(tmp_path):
    """A store whose bounds are nowhere near zero.

    Every other grid fixture starts at ``[0, 0, 0]``, where a
    bounds-relative frame and an absolute one are indistinguishable.  That
    is exactly why the frame bug survived: nothing in the suite could see
    it.  Bounds land at about (1050, 2050) with a 200-wide cell, so the
    grid is anchored at cell (5, 5, 5).
    """
    rng = np.random.default_rng(23)
    path = tmp_path / "offset.zarrvectors"
    write_points(
        path, rng.uniform(1050, 2050, size=(2000, 3)).astype(np.float32),
        chunk_shape=(200.0, 200.0, 200.0), bin_shape=(50.0, 50.0, 50.0),
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
        # 9, not 8: bounds are inclusive, so a vertex at exactly 800 has
        # to be storable and it lands in cell 8.  The store allocates
        # floor(hi/c) - floor(lo/c) + 1 -- predicting ceil(extent/c) was
        # one plane short, which is the failure Grid.plan exists to catch.
        grid = zv.Grid.plan(([0, 0, 0], [800, 800, 800]), cell_size=(100.0,) * 3)
        assert grid.shape == (9, 9, 9)
        assert grid.cells == 729

    def test_plan_from_a_target_cell_count(self):
        # target_cells=n asks for n cell-WIDTHS; the allocation is n + 1
        # planes when the upper bound lands on a boundary.
        grid = zv.Grid.plan(([0, 0, 0], [800, 800, 800]), target_cells=4)
        assert grid.shape == (5, 5, 5)
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
        # target_cells=1 allocates 2x2x2 = 8 cells, not 1, so the vertex
        # count has to clear the 64 MB/cell target eight times over for
        # this to still be testing what it says.
        tiny = zv.Grid.plan(([0, 0, 0], [800, 800, 800]), target_cells=1)
        verdict = tiny.capacity(n_vertices=2_000_000_000)
        assert not verdict.fits
        assert "use more cells" in (verdict.reason or "")

    def test_capacity_accepts_a_sensible_grid(self):
        grid = zv.Grid.plan(([0, 0, 0], [800, 800, 800]), target_cells=8)
        assert grid.capacity(n_vertices=20_000_000).fits

    def test_capacity_str_is_readable(self):
        grid = zv.Grid.plan(([0, 0, 0], [800, 800, 800]), target_cells=8)
        assert "729 cells" in str(grid.capacity(n_vertices=1000))


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


class TestGridSpeaksTheStoresFrame:
    """Cells are addressed the way the store writes them, or not at all.

    ``cells_in`` was absolute and ``cell_of`` / ``holds`` / ``__iter__``
    were bounds-relative.  A reference is handed straight to the readers'
    ``chunks=`` term and to the engine's cell requests, both of which are
    absolute -- so the relative half was simply wrong, and invisible on
    any store whose bounds began at zero.
    """

    def test_cell_of_names_a_key_the_store_actually_has(self, offset_cloud):
        from zarr_vectors.building import list_chunk_keys

        level = zv.open(offset_cloud).level(0)
        real = set(list_chunk_keys(level.store))
        grid = level.grid
        for point in level.read().positions[:20]:
            assert grid.cell_of(point).coords in real

    def test_selecting_the_cell_a_point_is_in_finds_that_point(
        self, offset_cloud,
    ):
        """The reported failure: this read zero vertices."""
        level = zv.open(offset_cloud).level(0)
        point = level.read().positions[0]
        got = level.select(cells=[level.grid.cell_of(point)]).read()
        assert got.vertex_count > 0

    def test_cell_of_agrees_with_cells_in(self, offset_cloud):
        level = zv.open(offset_cloud).level(0)
        grid = level.grid
        point = level.read().positions[0]
        assert grid.cell_of(point) in grid.cells_in((point, point))

    def test_the_post_filter_uses_the_same_frame(self, offset_cloud):
        """``cells_split=False`` forces the in-memory path.

        It re-implemented ``cell_of`` vectorised, with the same origin
        subtraction -- so it disagreed with the references it was
        comparing against.
        """
        level = zv.open(offset_cloud).level(0)
        ref = level.grid.cell_of(level.read().positions[0])
        assert (
            level.select(cells=[ref], cells_split=False).read().vertex_count
            == level.select(cells=[ref]).read().vertex_count
            > 0
        )

    def test_holds_accepts_the_cells_the_grid_names(self, offset_cloud):
        level = zv.open(offset_cloud).level(0)
        grid = level.grid
        assert grid.holds(grid.cell_of(level.read().positions[0]))
        # ...and still rejects one outside the allocation.
        assert not grid.holds(zv.CellRef((999, 0, 0)))

    def test_anchor_is_the_cell_the_lower_corner_falls_in(self, offset_cloud):
        level = zv.open(offset_cloud).level(0)
        origin, _shape = level.store.chunk_grid_bounds("vertices")
        assert level.grid.anchor == tuple(origin or (0, 0, 0))

    def test_iterating_yields_absolute_references(self, offset_cloud):
        level = zv.open(offset_cloud).level(0)
        grid = level.grid
        anchor = grid.anchor
        seen = list(grid)
        assert len(seen) == grid.cells
        assert all(grid.holds(ref) for ref in seen)
        assert min(seen).coords == anchor

    def test_a_degenerate_grid_has_a_zero_anchor_and_no_cells(self):
        """``Level.grid`` builds this when the store declares no bounds.

        ``origin`` is ``()`` there, which the old post-filter's
        ``is not None`` guard did not catch -- it broadcast-crashed.
        """
        assert zv.Grid(shape=(), cell_shape=(100.0,) * 3).anchor == (0, 0, 0)
        assert list(zv.Grid(shape=(), cell_shape=())) == []


class TestGridPlanMatchesTheAllocator:
    """A prediction that disagrees with the allocation is worse than none.

    ``Grid.plan`` restated the allocator's arithmetic as
    ``ceil(extent/cell)`` instead of calling it, and the two differ
    whenever the upper bound lands on a cell boundary or the lower one
    does not.  Both are ordinary: ``Layout(cells=n)`` on round bounds
    produces the first, and any store not starting at zero the second.
    """

    @pytest.mark.parametrize(
        "bounds,cell",
        [
            (([0, 0, 0], [800, 800, 800]), (100.0,) * 3),
            (([1050, 1050, 1050], [2050, 2050, 2050]), (200.0,) * 3),
            (([150, 150, 150], [950, 950, 950]), (200.0,) * 3),
            (([0, 0, 0], [1000, 1000, 1000]), (1000.0,) * 3),
            (([-500, -500, -500], [500, 500, 500]), (200.0,) * 3),
        ],
    )
    def test_plan_agrees_with_the_allocator(self, bounds, cell):
        from zarr_vectors.core.arrays import level_grid_layout

        grid = zv.Grid.plan(bounds, cell_size=cell)
        origin, shape = level_grid_layout(bounds, cell)
        assert (grid.anchor, grid.shape) == (origin, shape)

    def test_a_point_on_the_upper_bound_is_inside_the_allocation(self):
        """The boundary case ``ceil`` could never express."""
        grid = zv.Grid.plan(([0, 0, 0], [800, 800, 800]), cell_size=(100.0,) * 3)
        assert grid.holds(grid.cell_of([800.0, 800.0, 800.0]))

    def test_a_levels_grid_matches_the_array_on_disk(self, offset_cloud):
        """The oracle: ask the store what it actually allocated."""
        level = zv.open(offset_cloud).level(0)
        origin, shape = level.store.chunk_grid_bounds("vertices")
        assert level.grid.anchor == tuple(origin or (0, 0, 0))
        assert level.grid.shape == shape

    def test_iterating_names_every_cell_the_bounds_cover(self, offset_cloud):
        ds = zv.open(offset_cloud)
        grid = ds.level(0).grid
        assert set(grid) == set(grid.cells_in(ds.bounds))
        assert all(grid.holds(ref) for ref in grid.cells_in(ds.bounds))

    def test_iterating_partitions_the_level_exactly(self, offset_cloud):
        """What hpc_pipelines.md promises when it shards work by cell."""
        level = zv.open(offset_cloud).level(0)
        total = sum(
            level.select(cells=[ref]).read().vertex_count
            for ref in level.grid
        )
        assert total == level.read().vertex_count


class TestGridOnAnAttributeChunkedLevel:
    """The grid is spatial; the level's keys lead with a bin.

    ``Grid`` hands out spatial cells, and they used to match no key of a
    level chunked by an attribute, so ``select(cells=grid.cells_in(box))``
    read nothing.  A spatial cell now selects that cell in every bin, as
    a box does; a ref from ``Level.cells`` names one bin's cell exactly.
    Offset bounds, as in ``offset_cloud``, so a frame error cannot hide.
    """

    @pytest.fixture
    def binned(self, tmp_path):
        rng = np.random.default_rng(29)
        pos = rng.uniform(1050, 2050, size=(1500, 3)).astype(np.float32)
        genes = np.array(["A", "B", "C"])[rng.integers(0, 3, 1500)]
        path = tmp_path / "binned.zarrvectors"
        write_points(
            path, pos, bounds=([1050.0] * 3, [2050.0] * 3),
            chunk_shape=(200.0, 200.0, 200.0),
            vertex_attributes={"gene": genes}, chunk_by_attribute="gene",
        )
        return path, pos, genes

    @staticmethod
    def _rows(a):
        return sorted(map(tuple, np.asarray(a).tolist()))

    @staticmethod
    def _in_cells(pos, coords):
        cells = np.floor(pos / 200.0).astype(np.int64)
        want = {tuple(c) for c in coords}
        return np.array([tuple(c) in want for c in cells.tolist()])

    def test_the_grid_is_the_spatial_tail_of_the_allocation(self, binned):
        path, _, _ = binned
        level = zv.open(path).level(0)
        origin, shape = level.store.chunk_grid_bounds("vertices")
        assert len(shape) == 4
        assert level.grid.shape == shape[1:]
        assert level.grid.anchor == tuple(origin[1:])

    @pytest.mark.parametrize("split", [True, False])
    def test_grid_cells_select_their_cell_in_every_bin(self, binned, split):
        path, pos, _ = binned
        level = zv.open(path).level(0)
        refs = level.grid.cells_in(([1100.0] * 3, [1500.0] * 3))

        got = level.read(cells=refs, cells_split=split)

        want = pos[self._in_cells(pos, [ref.coords for ref in refs])]
        assert len(want) > 0
        assert self._rows(got.positions) == self._rows(want)

    def test_iterating_the_grid_partitions_a_binned_level(self, binned):
        level = zv.open(binned[0]).level(0)
        total = sum(
            level.select(cells=[ref]).read().vertex_count for ref in level.grid
        )
        assert total == level.read().vertex_count

    def test_a_level_cells_ref_names_one_bins_cell(self, binned):
        path, pos, genes = binned
        level = zv.open(path).level(0)
        ref = next(iter(level.cells()))
        values = level.store.attrs.to_dict()["zarr_vectors_level"][
            "chunk_attribute_values"
        ]

        got = level.read(cells=[ref])

        mask = self._in_cells(pos, [ref.coords[1:]]) & (
            genes == values[ref.coords[0]]
        )
        assert mask.any()
        assert self._rows(got.positions) == self._rows(pos[mask])

    def test_a_level_cells_ref_post_filtered_keeps_its_cell_in_every_bin(
        self, binned,
    ):
        """A position has no bin, so the post-filter can only go spatial.

        It compared the whole ref, bin included, with a spatial cell and
        matched nothing.  Only a reader that takes the cells itself can
        tell bins apart; the post-filter keeps the cell in every bin.
        """
        path, pos, _ = binned
        level = zv.open(path).level(0)
        ref = next(iter(level.cells()))

        got = level.read(cells=[ref], cells_split=False)

        want = pos[self._in_cells(pos, [ref.coords[1:]])]
        assert self._rows(got.positions) == self._rows(want)
