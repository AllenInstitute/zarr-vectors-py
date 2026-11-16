"""A box query returns every point inside it, on every kind of level.

Bin-level targeting reads fragment ``k`` of a cell as root bin ``k``.  It
used to be applied to levels where that does not hold -- coarsened levels
(one fragment per metavertex), levels on a scaled chunk grid, levels with
an object index (one fragment per object), and single-fragment cells --
and silently returned a fraction of the points.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.building import assign_chunks
from zarr_vectors.core.arrays import write_chunk_vertices
from zarr_vectors.core.store import create_store, get_resolution_level
from zarr_vectors.multiresolution.coarsen import build_pyramid
from zarr_vectors.types.points import read_points, write_points
from zarr_vectors.types.polylines import read_polylines, write_polylines

BOXES = [
    ([10.0, 10.0, 10.0], [60.0, 60.0, 60.0]),
    ([0.0, 0.0, 0.0], [30.0, 30.0, 30.0]),
    ([55.0, 5.0, 40.0], [95.0, 45.0, 99.0]),
]


def _inside(p, box):
    lo, hi = np.asarray(box[0]), np.asarray(box[1])
    return int(np.all((p >= lo) & (p <= hi), axis=1).sum())


def _check_points(store, levels):
    for level in levels:
        everything = read_points(store, level=level)["positions"]
        for box in BOXES:
            got = read_points(store, level=level, bbox=box)["positions"]
            assert len(got) == _inside(everything, box), (level, box)


@pytest.mark.parametrize("chunk_scale_factors", [None, [2, 2]])
@pytest.mark.parametrize("with_objects", [False, True])
def test_point_box_on_every_level(tmp_path, chunk_scale_factors, with_objects):
    pos = np.random.default_rng(1).uniform(0, 100, (300, 3)).astype(np.float32)
    store = str(tmp_path / "p.zarr")
    write_points(
        store, pos, chunk_shape=(20.0,) * 3, bin_shape=(5.0,) * 3,
        bounds=([0.0] * 3, [100.0] * 3),
        object_ids=(np.arange(300) % 11) if with_objects else None,
    )
    build_pyramid(
        store, factors=[(2, 1), (2, 1)],
        chunk_scale_factors=chunk_scale_factors,
    )
    _check_points(store, (0, 1, 2))


def test_point_box_level_zero_is_unchanged(tmp_path):
    pos = np.random.default_rng(2).uniform(0, 100, (500, 3)).astype(np.float32)
    store = str(tmp_path / "p.zarr")
    write_points(
        store, pos, chunk_shape=(20.0,) * 3, bin_shape=(5.0,) * 3,
        bounds=([0.0] * 3, [100.0] * 3),
    )
    for box in BOXES:
        got = read_points(store, bbox=box)["positions"]
        assert len(got) == _inside(pos, box)


def test_point_box_single_fragment_cell_with_bins(tmp_path):
    """``building.assign_chunks`` cells hold one fragment, not one per bin."""
    store = str(tmp_path / "c.zarr")
    root = create_store(
        store, chunk_shape=(100.0,) * 3, base_bin_shape=(10.0,) * 3,
        bounds=([0.0] * 3, [200.0] * 3),
    )
    lg = get_resolution_level(root, 0)
    pos = np.random.default_rng(3).uniform(0, 200, (5000, 3)).astype(np.float32)
    for cc, idx in assign_chunks(pos, (100.0,) * 3).items():
        write_chunk_vertices(lg, cc, [pos[idx]], dtype=np.float32)
    box = ([12.0, 30.0, 5.0], [96.0, 140.0, 77.0])
    got = read_points(store, bbox=box)["positions"]
    assert len(got) == _inside(pos, box) > 0


@pytest.mark.parametrize("chunk_scale_factors", [None, [2, 2]])
def test_polyline_box_on_every_level(tmp_path, chunk_scale_factors):
    rng = np.random.default_rng(4)
    lines = [
        np.clip(np.cumsum(rng.normal(0, 4, (30, 3)), 0) + 50, 1, 99)
        .astype(np.float32)
        for _ in range(15)
    ]
    store = str(tmp_path / "l.zarr")
    write_polylines(
        store, lines, chunk_shape=(20.0,) * 3, bin_shape=(5.0,) * 3,
        bounds=([0.0] * 3, [100.0] * 3),
    )
    build_pyramid(
        store, factors=[(2, 1), (2, 1)],
        chunk_scale_factors=chunk_scale_factors,
    )

    def flat(res):
        parts = [
            np.concatenate(p) if isinstance(p, list) else p
            for p in res["polylines"]
        ]
        return np.concatenate(parts) if parts else np.zeros((0, 3))

    for level in (0, 1, 2):
        everything = flat(read_polylines(store, level=level))
        for box in BOXES:
            got = flat(read_polylines(store, level=level, bbox=box))
            assert _inside(got, box) == _inside(everything, box), (level, box)
