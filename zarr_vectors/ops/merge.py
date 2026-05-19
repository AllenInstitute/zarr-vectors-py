"""Edit-report merging + OID-prefix allocator.

This module is the multi-writer concurrency surface called out in the
plan.  The two pieces it ships:

- :class:`~zarr_vectors.ops.change_set.OidPrefix` — re-exported from
  :mod:`change_set` for convenience.  Set on
  :class:`~zarr_vectors.ops.edit.EditSession` to confine atomic-OID
  allocations to a residue class so two cooperating writers cannot
  collide.
- :func:`merge_edit_reports` — replay-based composition of two or more
  :class:`EditReport` instances against a shared base.  Composes
  ``oid_remap``, set-unions ``touched_chunks`` and
  ``dirty_pyramid_levels``, validates that the prefixes (if any) are
  pairwise disjoint, and returns a single combined report.

Note: live icechunk three-way merge against snapshot history goes
through :func:`zarr_vectors.core.store.merge_branch`; this module
handles the *data* plumbing (oid remaps, touched-chunk unions) that
sits next to it.
"""

from __future__ import annotations

from itertools import combinations

from zarr_vectors.exceptions import EditError
from zarr_vectors.ops.change_set import EditReport, OidPrefix


def allocate_oid(
    prefix: OidPrefix | None,
    next_free: int,
) -> int:
    """Return the next OID consistent with ``prefix`` and ``>= next_free``.

    ``None`` prefix returns ``next_free`` unchanged.  Useful for code
    paths outside :class:`EditSession` that need the same allocation
    contract (e.g. ad-hoc tooling).
    """
    if prefix is None:
        return int(next_free)
    return prefix.next_after(int(next_free))


def merge_edit_reports(
    *reports: EditReport,
    base: EditReport | None = None,
) -> EditReport:
    """Compose multiple :class:`EditReport` instances into one.

    Args:
        *reports: The per-session reports to merge.  Order matters
            only for ``oid_remap`` composition (later reports' remaps
            are applied on top of earlier ones).
        base: Optional baseline report representing the snapshot the
            sessions forked from.  Today it's only used to seed the
            ``snapshot_id`` field; merge semantics don't depend on it.

    Returns:
        A new :class:`EditReport` whose ``oid_remap`` maps every
        original OID through the chain of session remaps, whose
        ``touched_chunks`` is the set-union of all inputs, and whose
        ``dirty_pyramid_levels`` is the set-union of dirty levels.

    Raises:
        EditError: If two input reports were emitted by sessions whose
            ``oid_prefix``es share a residue class (atomic OIDs from
            those sessions could collide).
    """
    if not reports:
        return EditReport()

    _validate_disjoint_prefixes(reports)

    combined = EditReport(
        oid_prefix=None,
        snapshot_id=(base.snapshot_id if base is not None else None),
    )

    # Set-union touched chunks (preserve a stable order via a list of
    # uniques).
    seen: set[tuple[int, tuple]] = set()
    for r in reports:
        for entry in r.touched_chunks:
            key = (entry[0], tuple(entry[1]))
            if key not in seen:
                seen.add(key)
                combined.touched_chunks.append(entry)

    # Set-union dirty pyramid levels.
    dirty = set()
    for r in reports:
        dirty.update(r.dirty_pyramid_levels)
    combined.dirty_pyramid_levels = sorted(dirty)

    # Compose oid_remap left-to-right.  If report[0] mapped 5 -> 8 and
    # report[1] mapped 8 -> 12, the merged map sends 5 -> 12.
    merged_remap: dict[int, int] = {}
    for r in reports:
        # 1. Apply this report's remap as a function on the values
        #    already in merged_remap (chained composition).
        for original, current in list(merged_remap.items()):
            if current in r.oid_remap:
                merged_remap[original] = r.oid_remap[current]
        # 2. Insert any new entries from this report that don't already
        #    appear in merged_remap.
        for k, v in r.oid_remap.items():
            if k not in merged_remap:
                merged_remap[k] = v
    combined.oid_remap = merged_remap

    combined.n_edits = sum(int(r.n_edits) for r in reports)
    return combined


def _validate_disjoint_prefixes(reports: tuple[EditReport, ...]) -> None:
    """Raise if any two reports share a residue class.

    Two prefixes ``OidPrefix(r1, k1)`` and ``OidPrefix(r2, k2)`` are
    *disjoint* iff for every integer ``n``, ``n % k1 != r1`` or
    ``n % k2 != r2``.  When ``k1 == k2`` this is just ``r1 != r2``.
    When ``k1 != k2`` the residue classes are dense and intersect for
    large enough ``n``, so we require ``k1 == k2`` (sessions that
    cooperate must agree on a single modulus up front).
    """
    prefixes = [
        (i, r.oid_prefix) for i, r in enumerate(reports)
        if r.oid_prefix is not None
    ]
    if len(prefixes) < 2:
        return
    for (i, p1), (j, p2) in combinations(prefixes, 2):
        if p1.modulus != p2.modulus:
            raise EditError(
                f"merge_edit_reports: reports[{i}].oid_prefix.modulus="
                f"{p1.modulus} != reports[{j}].oid_prefix.modulus="
                f"{p2.modulus}.  Cooperating sessions must agree on a "
                f"single modulus.",
            )
        if p1.residue == p2.residue:
            raise EditError(
                f"merge_edit_reports: reports[{i}] and reports[{j}] "
                f"share OID-prefix residue class "
                f"({p1.residue} mod {p1.modulus}); atomic OIDs from "
                f"these sessions can collide.  Re-run with disjoint "
                f"oid_prefix= values.",
            )


__all__ = [
    "OidPrefix",
    "allocate_oid",
    "merge_edit_reports",
]
