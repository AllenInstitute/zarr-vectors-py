"""A links family's stamped element type wins over the writing call's.

Appending to an int32-stamped segment with the default int64 call used to
decode the cell's old rows at the wrong width and write the new ones in
another: ``[(5, 6)]`` read back as ``[(5, 0), (6, 0)]``. Link attributes
had the twin bug: an append re-typed the whole segment to the new rows'
dtype, so every cell written before decoded at the wrong width.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.building import (
    create_link_attributes_array,
    create_links_array,
    create_store,
    finalize_links,
    get_resolution_level,
    read_link_arrays,
    read_link_attributes,
    write_link_attribute_cells,
    write_link_cells,
    write_links,
)
from zarr_vectors.core.paths import intra_offsets
from zarr_vectors.exceptions import ArrayError


def _level(tmp_path):
    root = create_store(
        tmp_path / "s.zarrvectors", bounds=([0.0] * 3, [100.0] * 3), chunk_shape=(50.0,) * 3,
    )
    return get_resolution_level(root, 0)


def _int32_family(lg):
    create_links_array(lg, 2, dtype="int32", delta=0, sid_ndim=3, offsets=intra_offsets(3, 2))


_FIRST = [[((0, 0, 0), 1), ((0, 0, 0), 2)], [((0, 0, 0), 3), ((0, 0, 0), 4)]]
_SECOND = [[((0, 0, 0), 5), ((0, 0, 0), 6)]]


def test_write_link_cells_appends_in_the_stamped_dtype(tmp_path):
    lg = _level(tmp_path)
    _int32_family(lg)
    write_link_cells(lg, _FIRST, 3, dtype=np.int32)
    write_link_cells(lg, _SECOND, 3)  # the default, int64
    finalize_links(lg, delta=0)
    _, vi = read_link_arrays(lg)
    assert vi.tolist() == [[1, 2], [3, 4], [5, 6]]


def test_write_links_appends_in_the_stamped_dtype(tmp_path):
    lg = _level(tmp_path)
    _int32_family(lg)
    write_links(lg, _FIRST, 3, dtype=np.int32)
    write_links(lg, _SECOND, 3, mode="append")
    _, vi = read_link_arrays(lg)
    assert vi.tolist() == [[1, 2], [3, 4], [5, 6]]


def test_a_value_the_stamped_dtype_cannot_hold_raises(tmp_path):
    lg = _level(tmp_path)
    _int32_family(lg)
    write_link_cells(lg, _FIRST, 3, dtype=np.int32)
    with pytest.raises(ArrayError, match="int32"):
        write_link_cells(lg, [[((0, 0, 0), 2**40), ((0, 0, 0), 1)]], 3)


def _seam(lg, value, dtype):
    """One cross-chunk record and its weight."""
    part = write_link_cells(lg, [[((0, 0, 0), 1), ((1, 0, 0), 2)]], 3)
    write_link_attribute_cells(
        lg, "w", np.array([value], dtype=dtype), partition=part, delta=0,
    )


def test_a_link_attribute_keeps_its_dtype_across_appends(tmp_path):
    lg = _level(tmp_path)
    _seam(lg, 0.25, np.float32)
    _seam(lg, 0.5, np.float64)  # a wider batch joins, it does not re-type
    finalize_links(lg, delta=0)
    seg = "link_attributes/w/0/" + lg["link_attributes"]["w"]["0"].children()[0]
    assert lg.read_array_meta(seg)["dtype"] == "float32"
    np.testing.assert_array_equal(read_link_attributes(lg, "w"), [0.25, 0.5])


def test_a_pre_created_segment_still_takes_the_datas_dtype(tmp_path):
    """A placeholder dtype on an empty segment is not a stored type."""
    lg = _level(tmp_path)
    part = write_link_cells(lg, [[((0, 0, 0), 1), ((1, 0, 0), 2)]], 3)
    ((seg, _cell),) = part.cell_indices
    from zarr_vectors.core.paths import parse_offsets

    create_link_attributes_array(
        lg, "w", dtype="float32", delta=0, sid_ndim=3, link_width=2,
        offsets=parse_offsets(seg, sid_ndim=3, link_width=2),
    )
    write_link_attribute_cells(lg, "w", np.array([7], dtype=np.int16), partition=part)
    assert lg.read_array_meta(f"link_attributes/w/0/{seg}")["dtype"] == "int16"


def test_finalize_rebuilds_attribute_presence_too(tmp_path):
    lg = _level(tmp_path)
    part = write_link_cells(lg, [[((0, 0, 0), 1), ((1, 0, 0), 2)]], 3)
    write_link_attribute_cells(
        lg, "w", np.array([3.0], np.float32), partition=part, record_presence=False,
    )
    ((seg, _cell),) = part.cell_indices
    assert lg.list_chunks(f"link_attributes/w/0/{seg}") == []
    finalize_links(lg, delta=0)
    assert lg.list_chunks(f"link_attributes/w/0/{seg}") == ["0.0.0"]
