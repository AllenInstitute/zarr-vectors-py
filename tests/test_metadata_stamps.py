"""The three facts a store now records instead of leaving to be inferred.

Each of these was recoverable only by guessing, and each guess was wrong in
a way that produced a plausible answer rather than an error:

* an attribute's column count was ``len(channel_names) if channel_names
  else 1``, so an unnamed 3-column attribute read back as 3N scalars;
* an object id's presence was ``0 <= id < num_objects``, so every object a
  sparsified level had dropped still reported as present;
* the NGFF scale came from a rounded ``bin_ratio`` while the translation in
  the same transform came from the unrounded ``bin_shape``, so a fractional
  coarsen factor made one transform contradict itself.

All three are metadata-only: no array family, path, grid, dtype, codec or
cell payload changes, and an older reader ignores the new keys.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from zarr_vectors.building import (
    attribute_layout,
    create_attribute_array,
    create_fragment_attribute_array,
    create_store,
    create_vertices_array,
    get_resolution_level,
    open_store,
    read_chunk_attributes,
    read_chunk_fragment_attributes,
    read_level_metadata,
    write_chunk_attributes,
    write_chunk_fragment_attributes,
    write_chunk_fragments,
    write_chunk_vertices,
)
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.multiresolution.coarsen import build_pyramid
from zarr_vectors.types.points import write_points

CS = (100.0, 100.0, 100.0)


def _level0(nvert: int = 6):
    path = os.path.join(tempfile.mkdtemp(), "s.zv")
    root = create_store(path, bounds=([0.0] * 3, [100.0] * 3), chunk_shape=CS)
    lg = get_resolution_level(root, 0)
    create_vertices_array(lg, dtype="float32")
    write_chunk_vertices(
        lg, (0, 0, 0), [np.zeros((nvert, 3), np.float32)], dtype=np.float32,
    )
    return path, lg


class TestAttributeWidthIsRecorded:
    """A6 — ``row_shape``, so width is not inferred from channel names."""

    def test_unnamed_multicolumn_attribute_round_trips(self):
        _, lg = _level0()
        # No channel_names: the case with no way to record its width before.
        create_attribute_array(lg, "normal", dtype="float32", ncols=3)
        write_chunk_attributes(
            lg, "normal", (0, 0, 0),
            [np.arange(18, dtype=np.float32).reshape(6, 3)], np.float32,
        )
        assert lg.read_array_meta("vertex_attributes/normal")["row_shape"] == [3]
        assert attribute_layout(lg, "normal") == (np.dtype("float32"), 3)
        (got,) = read_chunk_attributes(lg, "normal", (0, 0, 0))
        assert got.shape == (6, 3)
        np.testing.assert_array_equal(got, np.arange(18).reshape(6, 3))

    def test_channel_names_still_declare_the_width_alone(self):
        # The pre-existing spelling must keep working untouched: naming the
        # channels is a complete declaration and needs no ncols.
        _, lg = _level0()
        create_attribute_array(lg, "colour", channel_names=["r", "g", "b"])
        assert lg.read_array_meta("vertex_attributes/colour")["row_shape"] == [3]
        assert attribute_layout(lg, "colour")[1] == 3

    def test_channel_names_and_ncols_must_agree(self):
        _, lg = _level0()
        with pytest.raises(ArrayError, match="describe the same width"):
            create_attribute_array(lg, "bad", channel_names=["r", "g"], ncols=3)

    def test_a_contradicting_read_width_raises_instead_of_reshaping(self):
        _, lg = _level0()
        create_attribute_array(lg, "normal", dtype="float32", ncols=3)
        write_chunk_attributes(
            lg, "normal", (0, 0, 0),
            [np.zeros((6, 3), np.float32)], np.float32,
        )
        # Silently honouring this is the original bug: same bytes, wrong shape.
        with pytest.raises(ArrayError, match="3-column"):
            read_chunk_attributes(lg, "normal", (0, 0, 0), ncols=1)

    def test_a_contradicting_read_dtype_raises(self):
        _, lg = _level0()
        create_attribute_array(lg, "v", dtype="float64")
        write_chunk_attributes(
            lg, "v", (0, 0, 0), [np.zeros(6, np.float64)], np.float64,
        )
        with pytest.raises(ArrayError, match="stored as"):
            read_chunk_attributes(lg, "v", (0, 0, 0), dtype=np.float32)

    def test_fragment_attributes_carry_it_too(self):
        _, lg = _level0()
        write_chunk_fragments(
            lg, (0, 0, 0), [np.arange(0, 3, dtype=np.int64),
                            np.arange(3, 6, dtype=np.int64)],
            target="vertex", mode="replace",
        )
        create_fragment_attribute_array(lg, "bbox", dtype="float32", ncols=2)
        write_chunk_fragment_attributes(
            lg, "bbox", (0, 0, 0),
            np.arange(4, dtype=np.float32).reshape(2, 2), dtype=np.float32,
        )
        got = read_chunk_fragment_attributes(lg, "bbox", (0, 0, 0))
        assert got.shape == (2, 2)


class TestObjectPresenceIsRecorded:
    """A8 — ``num_present``, so a dropped id is distinguishable from an empty one."""

    def _sparsified(self):
        path = os.path.join(tempfile.mkdtemp(), "sp.zv")
        write_points(
            path,
            np.random.default_rng(1).uniform(0, 200, (40, 3)).astype(np.float32),
            chunk_shape=CS, object_ids=np.arange(40, dtype=np.int64),
        )
        build_pyramid(path, factors=[(1.0, 2.0)], sparsity_seed=7)
        return path

    def test_slots_and_count_differ_on_a_sparsified_level(self):
        import zarr_vectors as zv

        catalog = zv.open(self._sparsified()).level(1).objects
        assert catalog.slots == 40
        assert catalog.count == 20
        assert len(catalog.ids()) == 20
        assert len(catalog.ids(present=False)) == 40

    def test_a_dropped_id_is_not_in_the_catalog(self):
        import zarr_vectors as zv

        catalog = zv.open(self._sparsified()).level(1).objects
        mask = catalog.present_mask()
        dropped = int(np.flatnonzero(~mask)[0])
        # Was True for every id in range, so reading one gave vertex_count 0
        # -- indistinguishable from an object that exists but is elsewhere.
        assert dropped not in catalog
        assert int(np.flatnonzero(mask)[0]) in catalog

    def test_the_stamp_matches_a_full_decode(self):
        from zarr_vectors.core.arrays import (
            object_present_count,
            object_present_mask,
        )

        lg = get_resolution_level(open_store(self._sparsified(), mode="r"), 1)
        assert lg.read_array_meta("object_index")["num_present"] == 20
        assert object_present_count(lg) == int(object_present_mask(lg).sum())


class TestTheTransformAgreesWithItself:
    """A7 — the NGFF scale is derived from the same field as the translation."""

    def _pyramid(self, factor: float):
        path = os.path.join(tempfile.mkdtemp(), "ngff.zv")
        write_points(
            path,
            np.random.default_rng(0).uniform(0, 400, (600, 3)).astype(np.float32),
            chunk_shape=(64.0, 64.0, 64.0), bin_shape=(8.0, 8.0, 8.0),
            object_ids=np.arange(600, dtype=np.int64),
        )
        build_pyramid(path, factors=[(factor, 1.0)], sparsity_seed=1)
        return path

    def _transform(self, path):
        root = open_store(path, mode="r")
        for entry in root.attrs.to_dict()["multiscales"][0]["datasets"]:
            if entry["path"] == "1":
                return {
                    t["type"]: (t.get("scale") or t.get("translation"))
                    for t in entry["coordinateTransformations"]
                }
        raise AssertionError("level 1 missing from the multiscales block")

    @pytest.mark.parametrize("factor", [1.5, 2.0, 3.0])
    def test_scale_equals_bin_shape_over_root_bin(self, factor):
        path = self._pyramid(factor)
        tf = self._transform(path)
        lm = read_level_metadata(open_store(path, mode="r"), 1)
        # translation is bin_shape / 2 by construction, so this is the
        # self-consistency check the fractional case used to fail.
        assert tf["translation"] == pytest.approx([b / 2.0 for b in lm.bin_shape])
        assert tf["scale"] == pytest.approx([b / 8.0 for b in lm.bin_shape])
        assert tf["scale"] == pytest.approx([factor] * 3)

    def test_a_fractional_ratio_round_trips(self):
        lm = read_level_metadata(open_store(self._pyramid(1.5), mode="r"), 1)
        # int(round(...)) on the way back turned this into 2 even once the
        # writer stamped 1.5.
        assert lm.bin_ratio == pytest.approx((1.5, 1.5, 1.5))
