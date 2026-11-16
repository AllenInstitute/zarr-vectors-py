"""A rechunk's bins are dense, labelled, and read back as what they hold.

Every test here reads data back. The older rechunk tests stop at the
level metadata, which is how a rechunk could hand a reader another bin's
data under the wrong label without any of them noticing: the values list
was compacted while the key prefixes kept the mapper's raw indices, and
a reader resolves a value to a bin by its position in that list.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.core.arrays import read_all_groupings
from zarr_vectors.core.store import (
    get_resolution_level,
    open_store,
    read_level_metadata,
)
from zarr_vectors.rechunk import RechunkSpec, rechunk, rechunk_by_attribute
from zarr_vectors.types.points import read_points, write_points
from zarr_vectors.types.polylines import read_polylines, write_polylines
from zarr_vectors.validate.structure import validate_structure

_CHUNK = (50.0, 50.0, 50.0)


def _rows(a):
    """Rows as a sorted list of tuples, for order-free comparison."""
    return sorted(map(tuple, np.asarray(a).tolist()))


def _level_meta(path):
    return read_level_metadata(open_store(str(path)), 0)


def _bin_extent(path):
    lg = get_resolution_level(open_store(str(path)), 0)
    return lg.chunk_grid_bounds("vertices")[1][0]


def _polylines():
    """Ten short polylines, several sharing a cell.

    Polyline 9 leaves cell (0, 0, 0) across x = 50 and comes back, so it
    has two runs in that cell.
    """
    rng = np.random.default_rng(3)
    polys = []
    for i in range(9):
        start = rng.uniform(5, 40, 3) if i < 5 else rng.uniform(55, 90, 3)
        steps = rng.uniform(-1.0, 1.0, (4, 3))
        polys.append(
            np.concatenate([[start], start + np.cumsum(steps, axis=0)])
            .astype("float32")
        )
    polys.append(np.array(
        [[10, 10, 10], [40, 10, 10], [60, 10, 10], [40, 12, 10], [10, 12, 10]],
        dtype="float32",
    ))
    return polys


def _whole(poly_result, i):
    """Polyline ``i`` of a read, its segments joined in order."""
    return np.concatenate(poly_result["polylines"][i], axis=0)


# --- bins cut by explicit edges -------------------------------------------


def test_explicit_edges_renumber_densely_and_label_by_lower_edge(tmp_path):
    """No object below 30, so the mapper's bin 0 is a hole."""
    pos = np.array(
        [[5, 5, 5], [15, 5, 5], [25, 5, 5], [60, 60, 60], [70, 60, 60],
         [80, 60, 60]], dtype="float32",
    )
    scores = np.array([45.0, 50.0, 60.0, 90.0, 120.0, 85.0])
    src = str(tmp_path / "src.zarrvectors")
    write_points(
        src, pos, chunk_shape=_CHUNK, object_ids=np.arange(6),
        object_attributes={"score": scores},
    )
    out = tmp_path / "by_score.zarrvectors"

    summary = rechunk(
        src, RechunkSpec(by="attribute:score", bins=[0, 30, 80, float("inf")]),
        output=str(out),
    )

    assert summary["bins_created"] == 2
    lm = _level_meta(out)
    assert lm.chunk_attribute_name == "score"
    assert lm.chunk_attribute_values == [30.0, 80.0]
    assert _bin_extent(out) == 2
    mid = read_points(str(out), attribute_filter={"score": 30.0})
    high = read_points(str(out), attribute_filter={"score": 80.0})
    assert _rows(mid["positions"]) == _rows(pos[:3])
    assert _rows(high["positions"]) == _rows(pos[3:])
    assert validate_structure(str(out)).ok


# --- group bins -----------------------------------------------------------


def test_group_bins_skip_empty_groups_and_put_the_ungrouped_last(tmp_path):
    polys = _polylines()
    src = str(tmp_path / "src.zarrvectors")
    write_polylines(
        src, polys, chunk_shape=_CHUNK,
        groups={0: [0, 1, 2], 1: [], 2: [3, 4, 5, 6]},
    )
    out = tmp_path / "grouped.zarrvectors"

    summary = rechunk(src, RechunkSpec(by="group"), output=str(out))

    assert summary["bins_created"] == 3
    lm = _level_meta(out)
    assert lm.chunk_attribute_name == "group"
    assert lm.chunk_attribute_values == [0, 2, -1]
    assert _bin_extent(out) == 3
    # Bin order is the output id order, and here that is the source's.
    for label, members in ((0, [0, 1, 2]), (2, [3, 4, 5, 6]), (-1, [7, 8, 9])):
        got = read_polylines(str(out), attribute_filter={"group": label})
        assert sorted(got["object_ids"]) == members
        for i, oid in enumerate(got["object_ids"]):
            np.testing.assert_array_equal(_whole(got, i), polys[oid])
    assert validate_structure(str(out)).ok


def test_group_rewrite_skips_objects_that_were_not_written(tmp_path, monkeypatch):
    """An object with no vertices gets no output id, so no group entry."""
    from zarr_vectors.rechunk import engine

    polys = _polylines()[:5]
    src = str(tmp_path / "src.zarrvectors")
    write_polylines(src, polys, chunk_shape=_CHUNK, groups={0: [0, 1, 2], 1: [3, 4]})
    real = engine.read_object_vertices

    def _object_1_is_empty(level_group, oid, **kw):
        return [] if oid == 1 else real(level_group, oid, **kw)

    monkeypatch.setattr(engine, "read_object_vertices", _object_1_is_empty)
    out = tmp_path / "grouped.zarrvectors"

    summary = rechunk(src, RechunkSpec(by="group"), output=str(out))

    assert summary["objects_rechunked"] == 4
    groupings = read_all_groupings(get_resolution_level(open_store(str(out)), 0))
    # Source 0, 2 -> output 0, 1; source 3, 4 -> output 2, 3.
    assert [sorted(g) for g in groupings] == [[0, 1], [2, 3]]


# --- one object reads as one object ---------------------------------------


def test_reading_one_object_returns_only_that_object(tmp_path):
    """Objects sharing a cell each keep their own fragment, in order."""
    polys = _polylines()
    src = str(tmp_path / "src.zarrvectors")
    write_polylines(src, polys, chunk_shape=_CHUNK, groups={0: list(range(10))})
    out = tmp_path / "grouped.zarrvectors"
    rechunk(src, RechunkSpec(by="group"), output=str(out))

    for oid, poly in enumerate(polys):
        got = read_polylines(str(out), object_ids=[oid])
        assert got["polyline_count"] == 1
        np.testing.assert_array_equal(_whole(got, 0), poly)
    assert validate_structure(str(out)).ok


# --- the declared dtype ---------------------------------------------------


def test_a_float64_source_round_trips(tmp_path):
    rng = np.random.default_rng(7)
    pos = rng.uniform(0, 100, (40, 3))  # float64, not representable in f32
    clusters = rng.integers(0, 3, 40)
    src = str(tmp_path / "src.zarrvectors")
    write_points(
        src, pos, chunk_shape=_CHUNK, dtype="float64",
        object_ids=np.arange(40), object_attributes={"cluster": clusters},
    )
    out = tmp_path / "by_cluster.zarrvectors"

    rechunk_by_attribute(src, "cluster", output=str(out))

    assert _level_meta(out).chunk_attribute_values == [0, 1, 2]
    for c in range(3):
        got = read_points(str(out), attribute_filter={"cluster": c})
        assert got["positions"].dtype == np.float64
        assert _rows(got["positions"]) == _rows(pos[clusters == c])


# --- the values list is in bin order ---------------------------------------


@pytest.mark.parametrize("categorical", [True, False])
def test_categorical_values_are_listed_in_bin_order(tmp_path, categorical):
    """Compared in order: a sorted-set comparison is blind to a shift."""
    pos = np.array([[5, 5, 5], [60, 5, 5], [5, 60, 5], [60, 60, 5]], "float32")
    labels = np.array([3, 1, 3, 2])
    src = str(tmp_path / "src.zarrvectors")
    write_points(
        src, pos, chunk_shape=_CHUNK, object_ids=np.arange(4),
        object_attributes={"cluster": labels},
    )
    out = tmp_path / "out.zarrvectors"

    rechunk(
        src, RechunkSpec(by="attribute:cluster", categorical=categorical),
        output=str(out),
    )

    assert _level_meta(out).chunk_attribute_values == [1, 2, 3]
    for value in (1, 2, 3):
        got = read_points(str(out), attribute_filter={"cluster": value})
        assert _rows(got["positions"]) == _rows(pos[labels == value])
