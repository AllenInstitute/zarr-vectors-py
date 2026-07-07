"""Tests for the multiresolution strategy plug-in registry.

Core provides only the ``"random"`` object-selection strategy and the
``"per_object"`` coarsening method; everything else is dispatched to
implementations that ``zarr-vectors-tools`` registers via
:mod:`zarr_vectors.multiresolution.registry`.  These tests cover the
dispatch, the reserved built-in names, and the actionable error raised when
an advanced strategy is requested without tools installed.
"""

from __future__ import annotations

import numpy as np
import pytest

import zarr_vectors.multiresolution.registry as reg
from zarr_vectors.multiresolution.coarsen import coarsen_level
from zarr_vectors.multiresolution.object_selection import apply_sparsity


@pytest.fixture
def clean_registry():
    """Snapshot + restore the module-global registries around each test."""
    sel = dict(reg._SELECTION_STRATEGIES)
    coa = dict(reg._COARSEN_STRATEGIES)
    reg._SELECTION_STRATEGIES.clear()
    reg._COARSEN_STRATEGIES.clear()
    try:
        yield reg
    finally:
        reg._SELECTION_STRATEGIES.clear()
        reg._SELECTION_STRATEGIES.update(sel)
        reg._COARSEN_STRATEGIES.clear()
        reg._COARSEN_STRATEGIES.update(coa)


class TestBuiltinsWithoutTools:
    """Core built-ins work with nothing registered."""

    def test_random_selection(self, clean_registry) -> None:
        kept = apply_sparsity(10, 0.5, "random", seed=0)
        assert kept.shape == (5,)
        assert np.all(np.diff(kept) > 0)  # sorted, unique

    def test_sparsity_one_keeps_all(self, clean_registry) -> None:
        kept = apply_sparsity(6, 1.0, "spatial_coverage")  # strategy ignored
        assert kept.tolist() == list(range(6))


class TestSelectionDispatch:
    def test_registered_strategy_dispatched(self, clean_registry) -> None:
        seen = {}

        def strat(n_objects, target_count, **ctx):
            seen.update(n_objects=n_objects, target_count=target_count, ctx=ctx)
            return np.arange(target_count, dtype=np.int64)

        clean_registry.register_selection_strategy("length", strat)
        kept = apply_sparsity(
            10, 0.3, "length", lengths=np.arange(10), seed=7,
        )
        assert kept.tolist() == [0, 1, 2]
        assert seen["n_objects"] == 10 and seen["target_count"] == 3
        # Context (seed, lengths, ...) is forwarded verbatim.
        assert seen["ctx"]["seed"] == 7
        assert seen["ctx"]["lengths"].tolist() == list(range(10))

    def test_unregistered_strategy_raises_install_hint(self, clean_registry) -> None:
        with pytest.raises(ValueError) as exc:
            apply_sparsity(10, 0.5, "spatial_coverage")
        msg = str(exc.value)
        assert "spatial_coverage" in msg
        assert "zarr-vectors-tools" in msg

    def test_cannot_shadow_random(self, clean_registry) -> None:
        with pytest.raises(ValueError, match="built-in"):
            clean_registry.register_selection_strategy(
                "random", lambda n, t, **c: np.arange(t),
            )


class TestCoarsenDispatch:
    def test_unknown_method_raises_before_store_access(self, clean_registry) -> None:
        # require_coarsen_strategy fires before any store I/O, so a bogus
        # path is fine — we only assert the dispatch error.
        with pytest.raises(ValueError) as exc:
            coarsen_level(
                "/nonexistent/store.zv", 0, 1, method="cross_object_metanode",
            )
        msg = str(exc.value)
        assert "cross_object_metanode" in msg
        assert "zarr-vectors-tools" in msg

    def test_registered_method_dispatched(self, clean_registry) -> None:
        def fake(**kwargs):
            return {"method": "fake", "seen_kwargs": sorted(kwargs)}

        clean_registry.register_coarsen_strategy("fake", fake)
        out = coarsen_level("/store.zv", 2, 3, method="fake")
        assert out["method"] == "fake"
        assert "store_path" in out["seen_kwargs"]
        assert "cross_level_storage" in out["seen_kwargs"]

    def test_cannot_shadow_per_object(self, clean_registry) -> None:
        with pytest.raises(ValueError, match="built-in"):
            clean_registry.register_coarsen_strategy(
                "per_object", lambda **k: {},
            )


class TestIntrospection:
    def test_listing(self, clean_registry) -> None:
        clean_registry.register_selection_strategy("length", lambda n, t, **c: None)
        clean_registry.register_coarsen_strategy("grid", lambda **k: None)
        assert clean_registry.registered_selection_strategies() == ["length"]
        assert clean_registry.registered_coarsen_strategies() == ["grid"]
        assert clean_registry.get_selection_strategy("missing") is None
        assert clean_registry.get_coarsen_strategy("missing") is None
