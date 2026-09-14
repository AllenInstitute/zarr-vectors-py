"""Pyramid refresh after edits.

:func:`rebuild_pyramid_from_level` re-runs the existing
:func:`zarr_vectors.multiresolution.coarsen.coarsen_level` engine for
every level *above* ``source_level``, reusing each level's existing
``bin_ratio`` / ``object_sparsity`` / ``chunk_shape`` settings so the
post-refresh pyramid is byte-for-byte equivalent to a from-scratch
``build_pyramid`` call.

Note: ``coarsen_level`` takes the handle it is given, so it sees
whatever that handle has written.  It used to re-open the store from a
URL and spawn its own backend session, which is why the caller had to
commit first -- and why the refresh was unreachable on any store whose
``Group.url`` is not an openable path.  :meth:`EditSession.flush` still
issues a pre-refresh commit on icechunk-backed stores; it is now a
checkpoint before the destructive level removal rather than a
correctness requirement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from zarr_vectors.exceptions import EditError

if TYPE_CHECKING:
    from zarr_vectors.core.group import Group


def rebuild_pyramid_from_level(
    root: Group,
    source_level: int,
) -> list[dict[str, Any]]:
    """Re-coarsen every level above ``source_level`` from scratch.

    Reads each existing target level's metadata (``bin_ratio``,
    ``object_sparsity``, ``chunk_shape``) and re-runs
    :func:`zarr_vectors.multiresolution.coarsen.coarsen_level` with the
    same parameters, replacing the old level data in place.

    Returns the list of per-level coarsening summaries.
    """
    from zarr_vectors.core.metadata import compute_bin_ratio
    from zarr_vectors.core.store import (
        list_resolution_levels,
        read_level_metadata,
        read_root_metadata,
        remove_resolution_level,
        session_for,
        commit,
    )
    from zarr_vectors.multiresolution.coarsen import coarsen_level

    levels = list_resolution_levels(root)
    above = [lv for lv in levels if lv > source_level]
    if not above:
        return []

    if source_level not in levels:
        raise EditError(
            f"source_level={source_level} not present in store "
            f"(have {levels})"
        )

    root_meta = read_root_metadata(root)
    base_bin = root_meta.effective_bin_shape
    base_chunk = tuple(root_meta.chunk_shape)
    if not base_bin:
        raise EditError(
            "Cannot refresh pyramid: root metadata is missing "
            "base_bin_shape (no level-0 bins to coarsen against)."
        )

    # Snapshot per-target settings *before* deleting; deleting wipes the
    # group's attrs along with everything else.
    plan: list[dict[str, Any]] = []
    for lv in sorted(above):
        try:
            lm = read_level_metadata(root, lv)
        except Exception as e:
            raise EditError(
                f"Cannot read level metadata for level {lv}: {e}"
            ) from None
        bin_ratio = lm.bin_ratio or compute_bin_ratio(
            base_bin, lm.bin_shape or base_bin,
        )
        coarsen_factor = float(bin_ratio[0])
        # If the level had a per-level chunk override, encode it as a
        # chunk_scale_factor relative to the root.
        if lm.chunk_shape is not None:
            try:
                scale = tuple(
                    int(round(s / b))
                    for s, b in zip(lm.chunk_shape, base_chunk)
                )
            except Exception:
                scale = (1,) * len(base_chunk)
        else:
            scale = (1,) * len(base_chunk)
        plan.append({
            "level": lv,
            "coarsen_factor": coarsen_factor,
            "sparsity_factor": (
                1.0 / lm.object_sparsity if lm.object_sparsity else 1.0
            ),
            "chunk_scale_factor": scale,
            "parent_level": lm.parent_level if lm.parent_level is not None else lv - 1,
        })

    # Commit pending writes before dropping and rebuilding levels.  This
    # is no longer needed for cross-handle visibility -- coarsen_level now
    # works through the handle passed to it rather than re-opening the
    # store from a URL -- but it remains a reasonable checkpoint before a
    # destructive level removal.  No-op on non-transactional backends.
    if session_for(root) is not None:
        commit(root, "pre-refresh commit")
    summaries: list[dict[str, Any]] = []
    for entry in plan:
        lv = entry["level"]
        remove_resolution_level(root, lv)
        if session_for(root) is not None:
            commit(root, f"drop level {lv} for refresh")
        summary = coarsen_level(
            root,
            source_level=entry["parent_level"],
            target_level=lv,
            coarsen_factor=entry["coarsen_factor"],
            sparsity_factor=entry["sparsity_factor"],
            chunk_scale_factor=entry["chunk_scale_factor"],
        )
        summaries.append(summary)
    return summaries
