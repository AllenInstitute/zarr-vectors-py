"""A registered coarsen strategy over a level chunked by an attribute.

``zarr-vectors-tools`` registers its coarseners with
:func:`zarr_vectors.building.register_coarsen_strategy`, and
``build_pyramid(method=...)`` dispatches to them. These strategies are
stand-ins with the tools signature (positional store/levels, the core
keywords, the strategy's own options): one coarsens through the core
per-object writer, as tools' ``per_object`` does, the other writes a coarse
level without the bin axis and then links it, which must fail on the rank
mismatch and leave the store as it was.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import zarr_vectors.multiresolution.registry as reg
from zarr_vectors.building import register_coarsen_strategy
from zarr_vectors.constants import VERTICES, XLEVEL_EXPLICIT
from zarr_vectors.core.arrays import (
    create_vertices_array,
    read_link_arrays,
    write_chunk_vertices,
)
from zarr_vectors.core.metadata import LevelMetadata
from zarr_vectors.core.store import (
    create_resolution_level,
    get_resolution_level,
    list_resolution_levels,
    open_store,
    read_level_metadata,
)
from zarr_vectors.exceptions import CoarseningError
from zarr_vectors.multiresolution import coarsen
from zarr_vectors.multiresolution.coarsen import build_pyramid
from zarr_vectors.types.polylines import read_polylines, write_polylines
from zarr_vectors.validate import validate


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(reg, "_COARSEN_STRATEGIES", dict(reg._COARSEN_STRATEGIES))
    return reg


def _bundles(tmp_path: Path) -> tuple[str, list, list]:
    rng = np.random.default_rng(4)
    polys, labels = [], []
    for i in range(8):
        start = rng.uniform(10, 90, 3)
        polys.append((start + np.cumsum(rng.normal(0, 3, (20, 3)), 0))
                     .clip(1, 99).astype("f4"))
        labels.append(np.array(["A" if i % 2 else "B"] * 20))
    path = str(tmp_path / "bundles.zarrvectors")
    write_polylines(
        path, polys, chunk_shape=(50.0,) * 3, bounds=([0.0] * 3, [100.0] * 3),
        bin_shape=(5.0,) * 3, vertex_attributes={"bundle": labels},
        chunk_by_attribute="bundle",
    )
    return path, polys, labels


def _files(path: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for f in sorted(Path(path).rglob("*")):
        if f.is_file():
            rel = f.relative_to(path).as_posix()
            out[rel] = (json.loads(f.read_text()) if f.name == "zarr.json"
                        else f.read_bytes())
    return out


def test_a_registered_strategy_builds_a_binned_pyramid(tmp_path, registry):
    path, _polys, labels = _bundles(tmp_path)
    calls: list[dict[str, Any]] = []

    def per_object_like(store_path, source_level, target_level, *,
                        coarsen_factor, sparsity_factor, chunk_scale_factor,
                        sparsity_strategy, sparsity_seed, cross_level_storage,
                        coarsen_mode="rdp"):
        calls.append(dict(source_level=source_level, target_level=target_level,
                          coarsen_factor=coarsen_factor,
                          cross_level_storage=cross_level_storage,
                          coarsen_mode=coarsen_mode))
        return coarsen._per_object_coarsen(
            store_path=store_path, source_level=source_level,
            target_level=target_level, coarsen_factor=coarsen_factor,
            sparsity_factor=sparsity_factor,
            chunk_scale_factor=chunk_scale_factor,
            sparsity_strategy=sparsity_strategy, sparsity_seed=sparsity_seed,
            cross_level_storage=cross_level_storage,
        )

    register_coarsen_strategy("per_object_like", per_object_like)
    summary = build_pyramid(
        path, factors=[(2.0, 1.0), (2.0, 1.0)], method="per_object_like",
        options={"coarsen_mode": "stride"},
        cross_level_depth=2, cross_level_storage=XLEVEL_EXPLICIT,
    )

    assert summary["method"] == "per_object_like"
    assert [(c["source_level"], c["target_level"]) for c in calls] == [(0, 1), (1, 2)]
    assert all(c["coarsen_mode"] == "stride" for c in calls)
    assert all(c["cross_level_storage"] == XLEVEL_EXPLICIT for c in calls)

    root = open_store(path)
    assert list_resolution_levels(root) == [0, 1, 2]
    lm0 = read_level_metadata(root, 0)
    for level in (1, 2):
        lm = read_level_metadata(root, level)
        assert lm.chunk_dims == lm0.chunk_dims
        assert lm.chunk_attribute_name == "bundle"
        assert lm.chunk_attribute_values == ["A", "B"]
        keys = get_resolution_level(root, level).list_chunk_coords(VERTICES)
        assert keys and all(len(k) == 4 for k in keys)
        # Each bin reads back as only its own objects.
        for value, odd in (("A", 1), ("B", 0)):
            got = read_polylines(path, level=level,
                                 attribute_filter={"bundle": value})
            assert got["object_ids"]
            assert all(oid % 2 == odd for oid in got["object_ids"])
    # Cross-level links keep the bin axis, and never cross bins.
    for delta in (1, 2):
        chunks, _ = read_link_arrays(get_resolution_level(root, 0), delta=delta)
        assert chunks.shape[1:] == (2, 4)
        np.testing.assert_array_equal(chunks[:, 0, 0], chunks[:, 1, 0])
    assert validate(path, level=5).ok, validate(path, level=5).summary()


def test_a_strategy_that_drops_the_bin_axis_fails_cleanly(tmp_path, registry):
    path, _polys, _labels = _bundles(tmp_path)
    before = _files(path)

    def flat(store_path, source_level, target_level, *, cross_level_storage,
             **_kw):
        """Writes one rank-3 coarse vertex, then links the levels."""
        root = open_store(store_path, mode="r+")
        lg = create_resolution_level(root, target_level, LevelMetadata(
            level=target_level, vertex_count=1, arrays_present=[VERTICES],
            bin_shape=(10.0,) * 3, bin_ratio=(2, 2, 2), parent_level=source_level))
        create_vertices_array(lg)
        write_chunk_vertices(lg, (0, 0, 0), [np.zeros((1, 3), "f4")])
        fine = get_resolution_level(root, source_level)
        fine_assn, n_fine = coarsen._reconstruct_chunk_assignments(fine, 3)
        coarse_assn, n_coarse = coarsen._reconstruct_chunk_assignments(lg, 3)
        coarsen._write_cross_level_edges(
            root, fine_level=source_level, delta=1,
            fine_chunk_assignments=fine_assn,
            coarse_chunk_assignments=coarse_assn,
            n_fine=n_fine, n_coarse=n_coarse,
            parent=np.zeros(n_fine, np.int64), storage=cross_level_storage,
        )
        return {}

    register_coarsen_strategy("flat", flat)
    with pytest.raises(CoarseningError, match="level 0 has rank-4.*level 1 has rank-3"):
        build_pyramid(path, factors=[(2.0, 1.0)], method="flat",
                      cross_level_storage=XLEVEL_EXPLICIT)

    assert list_resolution_levels(open_store(path)) == [0]
    assert _files(path) == before
