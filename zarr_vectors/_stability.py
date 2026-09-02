"""Which modules are supported, which are internal, and which are undecided.

One machine-readable answer, because there were three prose ones and they
disagreed.  The contract test listed eight internal prefixes; the two
surface docstrings listed five (omitting ``sharding``, ``multiresolution``
and ``_engine`` — from which ``building`` re-exports six names, so the
docstring reasonably read as "the rest is fair game"); and the published
docs listed ``types``, ``lazy``, ``spatial``, ``validate``, ``constants``
and ``typing`` while mentioning neither ``api`` nor ``building``.  A
downstream reader met whichever of those they happened to open.

The three tiers:

``SUPPORTED``
    Will not change without a deprecation path.  Import freely.

``INTERNAL``
    Changes without notice.  A name used from here is a name nobody knows
    is load-bearing; if you need one, ask for it to be promoted.

``UNDECIDED``
    Neither promised nor disowned — usually because the module predates
    the split and its replacement is not finished.  Treat as internal;
    each entry carries the reason it is not yet either.

``test_stability_manifest.py`` asserts every top-level module appears in
exactly one tier, so a new module cannot be born unclassified.
"""

from __future__ import annotations

from typing import Literal

Tier = Literal["supported", "internal", "undecided"]

#: The two contracts, plus the leaf modules both of them depend on.
SUPPORTED: tuple[str, ...] = (
    "zarr_vectors.api",
    "zarr_vectors.building",
    # Leaves with no internal imports of their own: `exceptions` is class
    # definitions, `typing` is TypeAliases, `constants` is strings. Both
    # surfaces raise, annotate and name things with these, so they cannot
    # be internal without making every supported signature internal too.
    "zarr_vectors.constants",
    "zarr_vectors.exceptions",
    "zarr_vectors.typing",
    "zarr_vectors.headers",
)

#: Change without notice.
INTERNAL: tuple[str, ...] = (
    "zarr_vectors.core",
    "zarr_vectors.encoding",
    "zarr_vectors.spatial",
    "zarr_vectors.lazy",
    "zarr_vectors.ops",
    "zarr_vectors.sharding",
    "zarr_vectors.multiresolution",
    "zarr_vectors._engine",
    "zarr_vectors.rechunk",
)

#: Not yet either, with the reason.
UNDECIDED: dict[str, str] = {
    "zarr_vectors.types": (
        "The five store-creating writers are promoted into `building` and "
        "are supported there. The readers are superseded by Level.read() / "
        "ReadResult, but cannot be deprecated until the api can carry "
        "per-vertex attributes for every geometry — pointing callers at a "
        "lossy replacement is worse than leaving them here."
    ),
    "zarr_vectors.validate": (
        "Stable in practice and widely used, but its result objects have "
        "never been given a compatibility promise."
    ),
    "zarr_vectors.composite": (
        "Multi-geometry stores round-trip now — add_geometry allocates its "
        "namespaced per-chunk arrays and its documented example passes — but "
        "the namespacing itself is unreviewed: a composite store's geometries "
        "live at vertices_<type>, which no other reader in the library knows "
        "how to find. Undecided until that layout is either specified or "
        "replaced."
    ),
}

_TIERS: tuple[tuple[str, Tier], ...] = (
    *((m, "supported") for m in SUPPORTED),
    *((m, "internal") for m in INTERNAL),
    *((m, "undecided") for m in UNDECIDED),
)


def stability(module: str) -> Tier:
    """Which tier ``module`` (or anything under it) is in.

    Args:
        module: A dotted module name, e.g. ``"zarr_vectors.core.arrays"``.

    Raises:
        KeyError: If nothing in the manifest covers it — which means a
            module was added without being classified, not that the
            caller asked wrongly.
    """
    best: tuple[int, Tier] | None = None
    for prefix, tier in _TIERS:
        if module == prefix or module.startswith(prefix + "."):
            if best is None or len(prefix) > best[0]:
                best = (len(prefix), tier)
    if best is None:
        raise KeyError(
            f"{module!r} is not in the stability manifest. Add it to "
            f"SUPPORTED, INTERNAL or UNDECIDED in zarr_vectors/_stability.py "
            f"— a module with no tier is a promise nobody made or refused."
        )
    return best[1]


def is_internal(module: str) -> bool:
    """Whether importing from ``module`` is reaching past a contract."""
    return stability(module) != "supported"
