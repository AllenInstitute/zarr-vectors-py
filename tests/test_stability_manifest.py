"""The manifest must cover every module, and cover each one once.

``zarr_vectors/_stability.py`` exists because three prose answers to
"which of this is supported?" had drifted apart.  Its own docstring says
*this* file "asserts every top-level module appears in exactly one tier,
so a new module cannot be born unclassified" — and until now this file did
not exist, so the manifest could drift exactly the way the prose it
replaced had.
"""

from __future__ import annotations

import pkgutil

import pytest

import zarr_vectors
from zarr_vectors._stability import (
    INTERNAL,
    SUPPORTED,
    UNDECIDED,
    is_internal,
    stability,
)


def _top_level_modules() -> list[str]:
    """Every module directly under ``zarr_vectors``, dotted."""
    return sorted(
        f"zarr_vectors.{m.name}"
        for m in pkgutil.iter_modules(zarr_vectors.__path__)
    )


def test_every_module_is_classified():
    """A module with no tier is a promise nobody made or refused."""
    unclassified = []
    for name in _top_level_modules():
        try:
            stability(name)
        except KeyError:
            unclassified.append(name)
    assert not unclassified, (
        f"not in the stability manifest: {unclassified}. Add each to "
        f"SUPPORTED, INTERNAL or UNDECIDED in zarr_vectors/_stability.py."
    )


def test_no_module_is_in_two_tiers():
    """Overlap would make ``stability()`` answer by list order."""
    names = [*SUPPORTED, *INTERNAL, *UNDECIDED]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    assert not duplicates, f"classified more than once: {duplicates}"


def test_every_classified_module_exists():
    """A manifest entry for a module that is gone is a stale promise."""
    known = set(_top_level_modules())
    missing = sorted(
        n for n in (*SUPPORTED, *INTERNAL, *UNDECIDED)
        if n.count(".") == 1 and n not in known
    )
    assert not missing, f"classified but absent from the package: {missing}"


def test_the_two_supported_surfaces_are_supported():
    """The whole point of the split, asserted directly."""
    assert stability("zarr_vectors.api") == "supported"
    assert stability("zarr_vectors.building") == "supported"
    assert not is_internal("zarr_vectors.api")
    assert not is_internal("zarr_vectors.building")


def test_a_submodule_inherits_its_package_tier():
    """``stability()`` answers for anything *under* a classified prefix."""
    assert stability("zarr_vectors.core.arrays") == "internal"
    assert stability("zarr_vectors.api.dataset") == "supported"
    assert stability("zarr_vectors.types.points") == "undecided"


def test_the_longest_matching_prefix_wins():
    """So a submodule could be promoted out of an internal package."""
    # ``_engine`` sits under no other prefix, but the resolution rule is
    # what would let ``zarr_vectors.core.something`` be promoted later
    # without reclassifying ``core`` itself.
    assert stability("zarr_vectors._engine.plan") == "internal"


def test_an_unknown_module_raises_rather_than_guessing():
    with pytest.raises(KeyError, match="not in the stability manifest"):
        stability("zarr_vectors.not_a_module")


def test_every_undecided_entry_carries_its_reason():
    """UNDECIDED is a dict precisely so the reason cannot be omitted."""
    for name, reason in UNDECIDED.items():
        assert reason.strip(), f"{name} is undecided with no reason given"
