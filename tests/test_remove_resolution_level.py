"""``remove_resolution_level`` keeps the store consistent.

It used to delete only the level's group: the NGFF ``multiscales``
entry stayed, ``links/+1`` on the level below kept pointing at a level
that no longer existed, and a middle level could be removed, leaving a
gap that validation level 5 rejects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.constants import LINKS, XLEVEL_EXPLICIT
from zarr_vectors.core.arrays import list_link_deltas
from zarr_vectors.core.store import (
    get_resolution_level,
    list_resolution_levels,
    open_store,
    remove_resolution_level,
)
from zarr_vectors.exceptions import StoreError
from zarr_vectors.multiresolution.coarsen import build_pyramid
from zarr_vectors.types.graphs import write_graph
from zarr_vectors.validate import validate


@pytest.fixture
def pyramid(tmp_path: Path) -> Path:
    path = tmp_path / "graph.zarr"
    rng = np.random.default_rng(0)
    n = 64
    write_graph(
        path,
        positions=rng.uniform(0.0, 100.0, size=(n, 3)).astype(np.float32),
        edges=np.stack([np.arange(n - 1), np.arange(1, n)], axis=1),
        object_ids=np.zeros(n, dtype=np.int64),
        chunk_shape=(40.0, 40.0, 40.0),
        bounds=([0.0] * 3, [100.0] * 3),
    )
    build_pyramid(
        path, factors=[(2.0, 1.0), (2.0, 1.0)],
        cross_level_depth=1, cross_level_storage=XLEVEL_EXPLICIT,
    )
    return path


def _ngff_paths(root) -> list[str]:
    return [d["path"] for d in root.attrs.to_dict()["multiscales"][0]["datasets"]]


def test_removing_the_top_level_clears_every_reference(pyramid):
    root = open_store(str(pyramid), mode="r+")
    assert 1 in list_link_deltas(get_resolution_level(root, 1))
    remove_resolution_level(root, 2)

    root = open_store(str(pyramid), mode="r")
    assert list_resolution_levels(root) == [0, 1]
    assert _ngff_paths(root) == ["0", "1"]
    # Level 1's links/+1 pointed at level 2; level 0's still has a target.
    assert 1 not in list_link_deltas(get_resolution_level(root, 1))
    assert 1 in list_link_deltas(get_resolution_level(root, 0))
    assert validate(str(pyramid), level=5).ok


def test_removing_down_to_level_zero_drops_links_plus_one(pyramid):
    root = open_store(str(pyramid), mode="r+")
    remove_resolution_level(root, 2)
    remove_resolution_level(root, 1)
    root = open_store(str(pyramid), mode="r")
    lg = get_resolution_level(root, 0)
    assert list_link_deltas(lg) == [0]
    assert _ngff_paths(root) == ["0"]
    assert LINKS in lg.attrs.to_dict()["zarr_vectors_level"]["arrays_present"]
    assert validate(str(pyramid), level=5).ok


def test_a_middle_level_is_refused_and_nothing_changes(pyramid):
    root = open_store(str(pyramid), mode="r+")
    with pytest.raises(StoreError, match="contiguous"):
        remove_resolution_level(root, 1)
    root = open_store(str(pyramid), mode="r")
    assert list_resolution_levels(root) == [0, 1, 2]
    assert _ngff_paths(root) == ["0", "1", "2"]
    assert 1 in list_link_deltas(get_resolution_level(root, 1))
    assert validate(str(pyramid), level=5).ok


def test_refresh_still_rebuilds_every_level(pyramid):
    from zarr_vectors.ops.refresh import rebuild_pyramid_from_level

    root = open_store(str(pyramid), mode="r+")
    summaries = rebuild_pyramid_from_level(root, 0)
    assert len(summaries) == 2
    root = open_store(str(pyramid), mode="r")
    assert list_resolution_levels(root) == [0, 1, 2]
    assert _ngff_paths(root) == ["0", "1", "2"]
    assert validate(str(pyramid), level=5).ok
