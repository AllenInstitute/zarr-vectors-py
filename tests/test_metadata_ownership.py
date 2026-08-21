"""Who is allowed to write level metadata, and in which reference frame.

Three problems, one theme — the level block was hand-written at every call
site with no primitive to update it and no owner to reconcile it:

* ``arrays_present`` was maintained by hand, in two different conventions,
  and was wrong on the most ordinary write path there is;
* there was nothing at all between ``create_resolution_level`` and
  ``read_level_metadata``, so a consumer opened the store with raw zarr;
* four quantities that all read as "a factor" are measured against three
  different things, and nothing in a name or a type says which.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from zarr_vectors.building import (
    create_attribute_array,
    get_level_bin_shape,
    get_resolution_level,
    level_chunk_scale,
    level_factor,
    open_store,
    read_level_metadata,
    read_root_metadata,
    refresh_arrays_present,
    update_level_metadata,
    update_root_metadata,
)
from zarr_vectors.core.metadata import LevelMetadata, chunk_scale_from_root
from zarr_vectors.exceptions import MetadataError
from zarr_vectors.multiresolution.coarsen import build_pyramid
from zarr_vectors.types.points import write_points


def _store(**kw) -> str:
    path = os.path.join(tempfile.mkdtemp(), "s.zv")
    write_points(
        path,
        np.random.default_rng(0).uniform(0, 400, (600, 3)).astype(np.float32),
        chunk_shape=kw.pop("chunk_shape", (64.0, 64.0, 64.0)),
        object_ids=np.arange(600, dtype=np.int64),
        **kw,
    )
    return path


class TestArraysPresentIsDerived:
    """B3."""

    def test_it_repairs_the_list_an_ordinary_write_gets_wrong(self):
        path = _store(vertex_attributes={"x": np.zeros(600, np.float32)})
        root = open_store(path, mode="r+")
        before = read_level_metadata(root, 0).arrays_present
        # vertex_fragments is on disk from the very first write and has
        # never been declared.
        assert "vertex_fragments" not in before

        got = refresh_arrays_present(get_resolution_level(root, 0))
        assert "vertex_fragments" in got
        after = read_level_metadata(open_store(path, mode="r"), 0).arrays_present
        assert sorted(after) == sorted(got)

    def test_entries_are_families_never_family_slash_name(self):
        path = _store(vertex_attributes={"x": np.zeros(600, np.float32)})
        got = refresh_arrays_present(
            get_resolution_level(open_store(path, mode="r+"), 0)
        )
        assert "vertex_attributes" in got
        assert not [e for e in got if "/" in e]

    def test_allocation_does_not_advertise_unless_asked(self):
        # The default must not put a read-modify-write of the level's shared
        # attrs blob inside an allocator -- that is the race
        # record_presence=False exists to avoid.
        path = _store()
        root = open_store(path, mode="r+")
        lg = get_resolution_level(root, 0)
        create_attribute_array(lg, "y", dtype="float32")
        assert "vertex_attributes" not in read_level_metadata(
            open_store(path, mode="r"), 0
        ).arrays_present

    def test_register_family_repairs_an_already_existing_array(self):
        path = _store()
        root = open_store(path, mode="r+")
        lg = get_resolution_level(root, 0)
        create_attribute_array(lg, "y", dtype="float32")
        # Array exists but was never advertised; registering must happen
        # before the exist_ok short-circuit or this cannot be repaired.
        create_attribute_array(lg, "y", dtype="float32", register_family=True)
        assert "vertex_attributes" in read_level_metadata(
            open_store(path, mode="r"), 0
        ).arrays_present


class TestMetadataCanBeUpdated:
    """B4."""

    def test_a_field_can_be_corrected_after_creation(self):
        path = _store()
        lg = get_resolution_level(open_store(path, mode="r+"), 0)
        update_level_metadata(lg, vertex_count=99)
        assert read_level_metadata(open_store(path, mode="r"), 0).vertex_count == 99

    def test_capabilities_append_rather_than_replace(self):
        path = _store()
        root = open_store(path, mode="r+")
        update_root_metadata(root, add_capabilities=["cap_a"])
        update_root_metadata(root, add_capabilities=["cap_b"])
        caps = read_root_metadata(open_store(path, mode="r")).format_capabilities
        assert {"cap_a", "cap_b"} <= set(caps)

    def test_add_arrays_present_extends_and_dedupes(self):
        path = _store()
        lg = get_resolution_level(open_store(path, mode="r+"), 0)
        update_level_metadata(lg, add_arrays_present="links")
        update_level_metadata(lg, add_arrays_present="links")
        present = read_level_metadata(open_store(path, mode="r"), 0).arrays_present
        assert present.count("links") == 1

    @pytest.mark.parametrize(
        "field", ["bin_shape", "bin_ratio", "chunk_shape", "bounds"],
    )
    def test_derived_and_grid_defining_fields_are_refused(self, field):
        # bin_* live in the NGFF block and would be dead keys here.
        # chunk_shape/bounds feed level_grid_layout, so changing one after
        # allocation would mis-size the NEXT array -- a layout change
        # reached through a metadata-only call.
        path = _store()
        lg = get_resolution_level(open_store(path, mode="r+"), 0)
        with pytest.raises(MetadataError, match="cannot set"):
            update_level_metadata(lg, **{field: (1.0, 1.0, 1.0)})

    def test_an_unknown_field_is_refused(self):
        path = _store()
        lg = get_resolution_level(open_store(path, mode="r+"), 0)
        with pytest.raises(MetadataError, match="unknown field"):
            update_level_metadata(lg, not_a_field=1)


class TestTheThreeReferenceFrames:
    """B5 — the frames that produced two real bugs when mixed up."""

    def _pyramid(self):
        path = _store(bin_shape=(8.0, 8.0, 8.0))
        build_pyramid(
            path, factors=[(2.0, 1.0), (2.0, 1.0)],
            chunk_scale_factors=[2, 2], sparsity_seed=1,
        )
        return path

    def test_parent_relative_and_root_relative_differ_and_both_are_available(self):
        root = open_store(self._pyramid(), mode="r")
        rm = read_root_metadata(root)
        lms = {lv: read_level_metadata(root, lv) for lv in (1, 2)}

        # Absolute bins compound.
        assert get_level_bin_shape(rm, lms[1])[0] == pytest.approx(16.0)
        assert get_level_bin_shape(rm, lms[2])[0] == pytest.approx(32.0)
        # Parent-relative stays at the factor that was asked for...
        assert level_factor(rm, lms[2], lms[1])[0] == pytest.approx(2.0)
        assert level_chunk_scale(rm, lms[2], lms[1])[0] == 2
        # ...while root-relative doubles. Reading one as the other is the
        # bug that squared a pyramid refresh.
        assert chunk_scale_from_root(rm, lms[2])[0] == 4

    def test_a_level_with_no_bin_shape_falls_back_to_the_root(self):
        rm = read_root_metadata(open_store(_store(bin_shape=(8.0,) * 3), mode="r"))
        bare = LevelMetadata(level=1, vertex_count=0, arrays_present=[])
        assert get_level_bin_shape(rm, bare) == tuple(rm.effective_bin_shape)
        assert get_level_bin_shape(rm, None) == tuple(rm.effective_bin_shape)

    def test_non_uniform_per_axis_factors_raise_rather_than_pick_axis_zero(self):
        rm = read_root_metadata(open_store(_store(bin_shape=(8.0,) * 3), mode="r"))
        skew = LevelMetadata(
            level=1, vertex_count=0, arrays_present=[],
            bin_shape=(16.0, 8.0, 8.0),
        )
        with pytest.raises(MetadataError, match="not uniform"):
            level_factor(rm, skew, None)


class TestFromParent:
    """B6."""

    def _root(self):
        return read_root_metadata(
            open_store(_store(bin_shape=(8.0, 8.0, 8.0)), mode="r")
        )

    def test_bins_compound_and_the_ratio_stays_level_0_relative(self):
        rm = self._root()
        l1 = LevelMetadata.from_parent(
            rm, None, level=1, vertex_count=1, arrays_present=["vertices"],
            bin_scale_from_parent=2.0,
        )
        l2 = LevelMetadata.from_parent(
            rm, l1, level=2, vertex_count=1, arrays_present=["vertices"],
            bin_scale_from_parent=2.0,
        )
        assert l1.bin_shape[0] == pytest.approx(16.0)
        assert l2.bin_shape[0] == pytest.approx(32.0)
        assert l1.bin_ratio[0] == pytest.approx(2.0)
        assert l2.bin_ratio[0] == pytest.approx(4.0)

    def test_a_non_binning_coarsener_is_expressible(self):
        # The skeleton strategy prunes rather than bins and keeps the root
        # bin at every level.  Assuming monotone coarsening would rewrite it.
        rm = self._root()
        lm = LevelMetadata.from_parent(
            rm, None, level=1, vertex_count=1, arrays_present=["vertices"],
        )
        assert lm.bin_shape == tuple(rm.effective_bin_shape)
        assert lm.bin_ratio[0] == pytest.approx(1.0)

    def test_chunk_shape_is_omitted_when_it_matches_root(self):
        rm = self._root()
        same = LevelMetadata.from_parent(
            rm, None, level=1, vertex_count=1, arrays_present=["vertices"],
            bin_scale_from_parent=2.0,
        )
        bigger = LevelMetadata.from_parent(
            rm, None, level=1, vertex_count=1, arrays_present=["vertices"],
            bin_scale_from_parent=2.0, chunk_scale_from_parent=2,
        )
        assert same.chunk_shape is None
        assert bigger.chunk_shape == pytest.approx((128.0, 128.0, 128.0))

    def test_it_refuses_an_invalid_level_its_caller_would_have_written(self):
        """``validate()`` runs inside, so a bad block never reaches disk."""
        rm = self._root()
        with pytest.raises(MetadataError, match="inherited_num_objects"):
            LevelMetadata.from_parent(
                rm, None, level=1, vertex_count=1, arrays_present=[],
                preserves_object_ids=True,
            )

    def test_chunk_grids_must_nest(self):
        rm = self._root()
        with pytest.raises(MetadataError, match="not an integer"):
            LevelMetadata.from_parent(
                rm, None, level=1, vertex_count=1, arrays_present=[],
                chunk_scale_from_parent=1.5,
            )

    def test_it_validates_what_it_builds(self):
        # bin 8 * 3 = 24 does not tile a chunk of 64 * 2 = 128.
        # Via an override, because the cross-level validator is a
        # deliberate no-op for a level that inherits root's chunk_shape --
        # a coarse level's bin routinely exceeds the chunk it sits in.
        rm = self._root()
        with pytest.raises(MetadataError, match="does not divide"):
            LevelMetadata.from_parent(
                rm, None, level=1, vertex_count=1, arrays_present=[],
                bin_scale_from_parent=3.0, chunk_scale_from_parent=2,
            )

    def test_it_reproduces_a_hand_written_block(self):
        """The point of the classmethod: same answer, one place."""
        rm = self._root()
        root_bin = tuple(float(b) for b in rm.effective_bin_shape)
        coarsen_factor = 2.0
        hand = LevelMetadata(
            level=1, vertex_count=7, arrays_present=["vertices", "object_index"],
            bin_shape=tuple(b * coarsen_factor for b in root_bin),
            bin_ratio=tuple(
                (b * coarsen_factor) / r for b, r in zip(root_bin, root_bin)
            ),
            object_sparsity=0.5, coarsening_method="per_object",
            parent_level=0, preserves_object_ids=True,
            inherited_num_objects=600,
        )
        auto = LevelMetadata.from_parent(
            rm, None, level=1, vertex_count=7,
            arrays_present=["vertices", "object_index"],
            bin_scale_from_parent=coarsen_factor, object_sparsity=0.5,
            coarsening_method="per_object", preserves_object_ids=True,
            inherited_num_objects=600,
        )
        assert auto.bin_shape == pytest.approx(hand.bin_shape)
        assert auto.bin_ratio == pytest.approx(hand.bin_ratio)
        assert auto.chunk_shape == hand.chunk_shape
        assert auto.parent_level == hand.parent_level
        assert auto.object_sparsity == hand.object_sparsity
