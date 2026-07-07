"""Plug-in registry for advanced multiscale strategies (dependency-injection).

Core ships only the simplest, dependency-free coarsening: the
metavertex-binning pyramid in :mod:`zarr_vectors.multiresolution.coarsen`
(the ``"per_object"`` method) driven by random object selection (the
``"random"`` strategy in :mod:`zarr_vectors.multiresolution.object_selection`).

The *advanced* approaches — per-geometry coarsening (quadric mesh
decimation, Douglas–Peucker polyline simplification, graph/point metanodes)
and non-random object selection (spatial-coverage, length/attribute-ranked,
point-thinning) — live in ``zarr-vectors-tools``.  On import, that package
registers its implementations here, so core can dispatch to them by name
without taking a hard dependency on it.  When a name is requested but no
implementation is registered (e.g. core installed without tools), the
``require_*`` helpers raise a clear, actionable error.

This mirrors the dependency-injection slot pattern used elsewhere in the
SDK (see ``zarr_vectors/ops/refresh_hook.py`` on the tools-integrated
builds).

Registered signatures
----------------------
- Selection strategy: ``fn(n_objects: int, target_count: int, **ctx) ->
  ndarray[int64]`` returning the sorted kept object indices.  ``ctx`` may
  carry ``seed``, ``lengths``, ``attribute_values``, ``attribute_mode``,
  ``representative_points`` and ``bin_shape`` (see
  :func:`zarr_vectors.multiresolution.object_selection.apply_sparsity`).
- Coarsen method: ``fn(**kwargs) -> dict`` with the same keyword contract as
  :func:`zarr_vectors.multiresolution.coarsen._per_object_coarsen`,
  returning the per-level summary dict.
"""

from __future__ import annotations

from typing import Any, Callable

# Registered-callable signatures (documented above; kept loose so the tools
# package isn't forced to import core typing internals).
SelectionStrategy = Callable[..., Any]
CoarsenStrategy = Callable[..., Any]

_SELECTION_STRATEGIES: dict[str, SelectionStrategy] = {}
_COARSEN_STRATEGIES: dict[str, CoarsenStrategy] = {}

# Names core implements natively, reserved so tools can't shadow them.
_BUILTIN_SELECTION = frozenset({"random"})
_BUILTIN_COARSEN = frozenset({"per_object"})

_INSTALL_HINT = (
    "install it with `pip install zarr-vectors-tools` (it registers its "
    "strategies on import)"
)


def register_selection_strategy(name: str, fn: SelectionStrategy) -> None:
    """Register an object-selection strategy under ``name``.

    Called by ``zarr-vectors-tools`` on import.  ``name`` must not shadow a
    core built-in (``"random"``).
    """
    if name in _BUILTIN_SELECTION:
        raise ValueError(
            f"selection strategy {name!r} is a core built-in and cannot be "
            f"overridden"
        )
    _SELECTION_STRATEGIES[name] = fn


def get_selection_strategy(name: str) -> SelectionStrategy | None:
    """Return the registered selection strategy for ``name``, or ``None``."""
    return _SELECTION_STRATEGIES.get(name)


def require_selection_strategy(name: str) -> SelectionStrategy:
    """Return the selection strategy for ``name`` or raise a clear error."""
    fn = _SELECTION_STRATEGIES.get(name)
    if fn is None:
        raise ValueError(
            f"object-selection strategy {name!r} is not available in core "
            f"(core provides only {sorted(_BUILTIN_SELECTION)}). It is "
            f"provided by zarr-vectors-tools; {_INSTALL_HINT}. "
            f"Registered: {sorted(_SELECTION_STRATEGIES)}."
        )
    return fn


def register_coarsen_strategy(name: str, fn: CoarsenStrategy) -> None:
    """Register a whole-pyramid coarsening method under ``name``.

    Called by ``zarr-vectors-tools`` on import.  ``name`` must not shadow a
    core built-in (``"binning"``).
    """
    if name in _BUILTIN_COARSEN:
        raise ValueError(
            f"coarsen method {name!r} is a core built-in and cannot be "
            f"overridden"
        )
    _COARSEN_STRATEGIES[name] = fn


def get_coarsen_strategy(name: str) -> CoarsenStrategy | None:
    """Return the registered coarsen method for ``name``, or ``None``."""
    return _COARSEN_STRATEGIES.get(name)


def require_coarsen_strategy(name: str) -> CoarsenStrategy:
    """Return the coarsen method for ``name`` or raise a clear error."""
    fn = _COARSEN_STRATEGIES.get(name)
    if fn is None:
        raise ValueError(
            f"coarsen method {name!r} is not available in core (core "
            f"provides only {sorted(_BUILTIN_COARSEN)}). It is provided by "
            f"zarr-vectors-tools; {_INSTALL_HINT}. "
            f"Registered: {sorted(_COARSEN_STRATEGIES)}."
        )
    return fn


def registered_selection_strategies() -> list[str]:
    """Names of all registered (non-built-in) selection strategies."""
    return sorted(_SELECTION_STRATEGIES)


def registered_coarsen_strategies() -> list[str]:
    """Names of all registered (non-built-in) coarsen methods."""
    return sorted(_COARSEN_STRATEGIES)
