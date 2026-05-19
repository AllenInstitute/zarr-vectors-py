"""Post-edit cleanup passes.

Atomic edits accumulate dead weight:

- **Sparse OIDs**: every atomic edit appends a new OID; old ones are
  left as empty manifests scattered between live entries.
- **Empty fragments**: ``remove_fragment(atomic=True)`` empties a
  fragment's row list but keeps the index slot.
- **Parallel rows** after chunk-cross + keep-source: the same logical
  vertex exists in both the source and target chunk for different OID
  populations.

``vacuum(root, ...)`` reclaims this dead weight.  This iteration
implements **OID compaction only**; the other two passes are stubbed
out so the public API stays stable for when they land later.

OID compaction:

1. Read every manifest at every level (via ``read_all_object_manifests``).
2. Identify OIDs with non-empty manifests; these are the "live" OIDs.
3. Build a dense remap ``{old_oid: new_oid}`` so live OIDs occupy
   ``[0, n_live)`` in their original relative order.
4. Rewrite ``object_index/manifests`` at ``total_objects=n_live``.
5. For every present ``object_attributes/<name>/`` array, reorder rows
   by the remap (dropping orphan rows) and write back.

Returns a :class:`VacuumReport`.  External consumers that hold OIDs
must compose them through ``oid_remap`` to stay valid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from zarr_vectors.constants import OBJECT_ATTRIBUTES
from zarr_vectors.exceptions import EditError
from zarr_vectors.ops.change_set import VacuumReport

if TYPE_CHECKING:
    from zarr_vectors.core.group import Group


def vacuum(
    root: Group,
    *,
    compact_oids: bool = True,
    drop_empty_fragments: bool = False,
    dedup_parallel_rows: bool = False,
    dry_run: bool = False,
    message: str = "vacuum",
) -> VacuumReport:
    """Reclaim dead weight in a ZV store.

    See module docstring for the per-pass contract.  This iteration
    only implements ``compact_oids``; the other two raise
    :class:`NotImplementedError` so callers can pin the kwarg shape
    against the future API.
    """
    if drop_empty_fragments:
        raise NotImplementedError(
            "vacuum(drop_empty_fragments=True): tombstone-fragment GC is "
            "deferred to a future iteration; please run vacuum() with "
            "drop_empty_fragments=False (the default) for now."
        )
    if dedup_parallel_rows:
        raise NotImplementedError(
            "vacuum(dedup_parallel_rows=True): parallel-row dedup is "
            "deferred to a future iteration; please run vacuum() with "
            "dedup_parallel_rows=False (the default) for now."
        )

    report = VacuumReport()
    if not compact_oids:
        return report

    from zarr_vectors.core.arrays import (
        read_all_object_manifests,
        read_object_attribute_present_mask,
        read_object_attributes,
        write_object_attributes,
        write_object_index,
    )
    from zarr_vectors.core.metadata import RootMetadata
    from zarr_vectors.core.store import (
        commit,
        get_resolution_level,
        list_resolution_levels,
    )

    meta = RootMetadata.from_dict(root.attrs.to_dict())
    sid_ndim = meta.sid_ndim

    for level in list_resolution_levels(root):
        level_group = get_resolution_level(root, level)
        try:
            manifests = read_all_object_manifests(level_group)
        except Exception:
            continue
        n = len(manifests)
        if n == 0:
            continue

        live_oids = [oid for oid, m in enumerate(manifests) if m]
        if len(live_oids) == n:
            # Already dense — no remap needed at this level.  We still
            # carry the identity through so downstream consumers can
            # see "I checked this level".
            for oid in live_oids:
                report.oid_remap.setdefault(int(oid), int(oid))
            continue

        # Build the remap and the new dense manifest list.
        new_manifests: dict[int, list[tuple[tuple, int]]] = {}
        for new_oid, old_oid in enumerate(live_oids):
            report.oid_remap[int(old_oid)] = int(new_oid)
            new_manifests[new_oid] = manifests[old_oid]

        if dry_run:
            continue

        # Rewrite the object index at the new dense size.
        write_object_index(
            level_group, new_manifests, sid_ndim,
            total_objects=len(live_oids),
        )

        # Compact every per-object attribute array.
        if OBJECT_ATTRIBUTES in level_group:
            for name in level_group[OBJECT_ATTRIBUTES]:
                try:
                    arr = read_object_attributes(level_group, name)
                except Exception:
                    continue
                if arr.shape[0] != n:
                    # Mis-sized array (e.g. partially written): skip
                    # rather than corrupt.
                    continue
                new_arr = arr[live_oids]
                mask = read_object_attribute_present_mask(level_group, name)
                if mask is not None and mask.shape[0] == n:
                    new_mask = mask[live_oids]
                else:
                    new_mask = None
                write_object_attributes(
                    level_group, name, new_arr, present_mask=new_mask,
                )

    if not dry_run:
        commit(root, message)

    return report


# Silence unused-import warnings for type-only re-exports.
_ = EditError
__all__ = ["vacuum", "VacuumReport"]
