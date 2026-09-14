"""What this build's CODE surface offers, and how to require it.

Three versions are in play and only one of them is expressible in a
dependency pin:

* the **package** version (``__version__``) — setuptools-scm, currently
  ``0.2.1.devN``;
* the **on-disk format** version (``FORMAT_VERSION``, 0.9.0) — already
  negotiable via :func:`zarr_vectors.api.require_format`;
* the **API surface** version, this module — which moved when ``api`` and
  ``building`` were introduced without the package version moving at all.

So a consumer that needs ``zarr_vectors.building`` cannot say so. Against
a build without it, it fails with a bare ``ImportError`` from whichever
module imports first, and the pin that was supposed to prevent that is
satisfied. ``require_format`` already exists for exactly this argument one
axis over; this is the same helper for the code.

``FEATURES`` is deliberately small and names things a caller *branches*
on, not every function that exists — a feature flag per symbol would just
be ``__all__`` with extra steps, and the contract test already pins that.
"""

from __future__ import annotations

#: Version of the Python API surface.  Bump the minor when a supported
#: name is added, the major when one is removed or changes meaning.
#: Independent of both ``__version__`` and the on-disk format.
__api_version__ = (1, 0)

#: Capabilities a caller may branch on, each True only when usable.
FEATURES: frozenset[str] = frozenset({
    # The two-surface split exists: zarr_vectors.api and .building.
    "surfaces",
    # building exports the level-wide presence rebuild and the walk it
    # needs, so a consumer need not fork _is_per_chunk_array.
    "presence-rebuild",
    # derive_nonempty_chunks refuses a sharded array instead of silently
    # emptying its manifest.
    "sharded-presence-guard",
    # Selection carries level=None for "unset", so level 0 is requestable.
    "selection-level-optional",
    # Vertex attributes come back from Level.read() for every geometry,
    # not only point clouds.
    "vertex-attributes-on-read",
    # Query.cells() enumerates the grid cells a query touches, for
    # sharding work across processes.
    "query-cells",
    # coarsen_level/build_pyramid forward options= to a registered
    # strategy, and api.coarsen_methods() lists what is installed.
    "coarsen-strategy-options",
})


def _parse(text: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in text.strip().split("."):
        digits = "".join(c for c in part if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def require_api(spec: str = "", *, features: object = ()) -> None:
    """Raise unless this build's API surface satisfies ``spec``.

    Args:
        spec: Comma-separated ``>=``/``>``/``<=``/``<``/``==`` clauses
            against :data:`__api_version__`, e.g. ``">=1.0"``.  Empty
            checks only ``features``.
        features: Feature names from :data:`FEATURES` that must all be
            present.  Prefer these to a version range: they say what the
            caller needs rather than when it happened to land.

    Raises:
        ImportError: If the requirement is not met.  ``ImportError``
            rather than a ZVError because the failure is "this build does
            not have what I import", and that is what a caller's
            dependency handling already catches.
    """
    if isinstance(features, str):
        features = (features,)
    missing = sorted(set(features) - FEATURES)
    if missing:
        raise ImportError(
            f"zarr-vectors {'.'.join(map(str, __api_version__))} does not "
            f"provide: {', '.join(missing)}. Known features: "
            f"{', '.join(sorted(FEATURES))}."
        )

    found = __api_version__
    for clause in (c.strip() for c in spec.split(",") if c.strip()):
        for op in (">=", "<=", "==", ">", "<"):
            if not clause.startswith(op):
                continue
            want = _parse(clause[len(op):])
            width = max(len(found), len(want))
            lhs = found + (0,) * (width - len(found))
            rhs = want + (0,) * (width - len(want))
            if not {
                ">=": lhs >= rhs, "<=": lhs <= rhs, "==": lhs == rhs,
                ">": lhs > rhs, "<": lhs < rhs,
            }[op]:
                raise ImportError(
                    f"zarr-vectors API surface is "
                    f"{'.'.join(map(str, found))}, which does not satisfy "
                    f"{spec!r}. Note this is NOT the package version — "
                    f"the two move independently, which is why a pin on "
                    f"the package cannot express this."
                )
            break
        else:
            raise ValueError(f"unparsable clause {clause!r} in spec {spec!r}")
