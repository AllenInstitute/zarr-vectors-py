"""Object selection for multi-resolution sparsity.

At a coarser pyramid level a subset of objects is retained.  Core ships
only the simplest, dependency-free strategy — **random** (uniform,
reproducible with a seed).

Advanced strategies (spatial coverage, length- or attribute-ranked,
point-thinning) live in ``zarr-vectors-tools``, which registers them via
:mod:`zarr_vectors.multiresolution.registry` on import.
:func:`apply_sparsity` dispatches to them by name and raises a clear
"install zarr-vectors-tools" error when one is requested but unregistered.

Selection functions return ``kept_indices`` — a sorted array of integer
indices into the original object list.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from zarr_vectors.multiresolution.registry import require_selection_strategy


def select_random(
    n_objects: int,
    target_count: int,
    *,
    seed: int | None = None,
) -> npt.NDArray[np.int64]:
    """Select objects uniformly at random.

    Reproducible with a fixed seed.

    Args:
        n_objects: Total number of objects.
        target_count: Number to keep.
        seed: Random seed for reproducibility.

    Returns:
        ``(target_count,)`` sorted array of kept object indices.
    """
    _validate_target(n_objects, target_count)

    if target_count >= n_objects:
        return np.arange(n_objects, dtype=np.int64)

    rng = np.random.default_rng(seed)
    chosen = rng.choice(n_objects, size=target_count, replace=False)
    return np.sort(chosen).astype(np.int64)


def apply_sparsity(
    n_objects: int,
    sparsity: float,
    strategy: str = "random",
    *,
    seed: int | None = None,
    lengths: npt.NDArray | None = None,
    attribute_values: npt.NDArray | None = None,
    attribute_mode: str = "max",
    representative_points: npt.NDArray | None = None,
    bin_shape: tuple[float, ...] | float | None = None,
) -> npt.NDArray[np.int64]:
    """Compute a target count from ``sparsity`` and select objects.

    ``strategy="random"`` (the default and only core built-in) uses
    :func:`select_random`.  Any other name is dispatched to an
    implementation registered by ``zarr-vectors-tools`` via
    :mod:`zarr_vectors.multiresolution.registry`; if none is registered the
    call raises a clear error naming the package to install.

    Args:
        n_objects: Total number of objects.
        sparsity: Fraction to keep, in (0, 1].  Some advanced strategies
            (e.g. point-thinning) derive the survivor count from
            ``bin_shape`` instead and may ignore this.
        strategy: ``"random"`` (core) or a name registered by tools.
        seed: Random seed.
        lengths / attribute_values / attribute_mode / representative_points /
            bin_shape: Optional per-object context forwarded verbatim to a
            registered strategy (unused by ``"random"``).

    Returns:
        Sorted array of kept object indices.

    Raises:
        ValueError: If a non-``"random"`` strategy is requested but not
            registered (i.e. ``zarr-vectors-tools`` is not installed).
    """
    if sparsity >= 1.0:
        return np.arange(n_objects, dtype=np.int64)

    target_count = max(1, round(n_objects * sparsity))

    if strategy == "random":
        return select_random(n_objects, target_count, seed=seed)

    # Delegate everything else to a tools-registered strategy.
    fn = require_selection_strategy(strategy)
    return fn(
        n_objects,
        target_count,
        seed=seed,
        lengths=lengths,
        attribute_values=attribute_values,
        attribute_mode=attribute_mode,
        representative_points=representative_points,
        bin_shape=bin_shape,
    )


def _validate_target(n_objects: int, target_count: int) -> None:
    """Validate target count."""
    if target_count < 1:
        raise ValueError(f"target_count must be >= 1, got {target_count}")
    if n_objects < 1:
        raise ValueError(f"n_objects must be >= 1, got {n_objects}")
