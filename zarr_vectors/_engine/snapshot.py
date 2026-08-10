"""The result of one fetch round: everything known so far.

:class:`Snapshot` extends
:class:`zarr_vectors.core.group._OfflineSession` rather than replacing
it, and that inheritance is the single highest-leverage reuse in this
package.  ``_OfflineSession`` already carries four hard-won properties
that a fresh class would have to re-earn:

* **Four request kinds, keyed root-relative**, so a lookup means the same
  thing from any Group in the tree.
* **The ``_ABSENT`` sentinel**, which distinguishes "probed, genuinely
  missing" from "never fetched".  Without it an ``array_exists`` probe
  for a legitimately absent array looks like a prefetch gap and the
  executor loops forever.
* **Miss recording that survives ``except Exception: pass``.**  Several
  readers wrap optional metadata reads in a bare except; a raised error
  would be swallowed, but a recorded miss is not.  This is what lets a
  decoder tell the executor what it needs without any per-reader
  knowledge.
* **The whole ``Group`` read path already knows how to be served from
  it** — ``read_bytes``, ``_lookup_node``, ``_offline_array`` and
  ``children`` all consult it today.

What is added here is ``rows`` (selectively-read rows of a standalone
array), plus the merge and rebase helpers the executor needs.

Note that ``rows`` is carried but not yet *served*: no reader asks for
rows today, because the API that would (``read_object_manifests(ids=)``)
does not exist yet.  The dimension is here so the plan algebra and the
fetchers are complete, and so adding that reader is a change to one
decoder rather than to the engine.
"""

from __future__ import annotations

from typing import Any

from zarr_vectors._engine.plan import ReadPlan
from zarr_vectors.core.group import Group, _OfflineSession


class Snapshot(_OfflineSession):
    """Positionally-complete result of one or more fetch rounds.

    Inherits ``nodes`` / ``chunks`` / ``arrays`` / ``listings`` /
    ``misses`` verbatim and adds ``rows`` and ``absent``.
    """

    __slots__ = ("rows", "absent", "expanded")

    def __init__(
        self,
        nodes: dict[str, Any] | None = None,
        chunks: dict[tuple[str, str], bytes] | None = None,
        arrays: dict[str, Any] | None = None,
        listings: dict[str, list[str]] | None = None,
        rows: dict[str, dict[int, Any]] | None = None,
    ) -> None:
        super().__init__(nodes=nodes, chunks=chunks, arrays=arrays, listings=listings)
        self.rows: dict[str, dict[int, Any]] = rows if rows is not None else {}
        # Requested and confirmed missing, in the same tagged encoding
        # ``misses`` uses.  ``nodes`` needs no equivalent: it records
        # absence in-band as ``_ABSENT``.
        self.absent: set[Any] = set()
        # Arrays already fanned out.  Fanning out twice is not wrong, but
        # it re-requests every cell of a large array on every round,
        # which is the difference between a loop that settles and one
        # that thrashes.
        self.expanded: set[str] = set()

    @classmethod
    def empty(cls) -> Snapshot:
        return cls()

    def absorb(self, other: Snapshot) -> None:
        """Fold another snapshot's contents in.

        Later rounds win on conflict, which matters for ``nodes``: a path
        first recorded as ``_ABSENT`` and later created should resolve.
        ``misses`` is deliberately *not* absorbed — it belongs to the
        decode pass that recorded it, and the executor clears it before
        each pass.
        """
        self.nodes.update(other.nodes)
        self.chunks.update(other.chunks)
        self.arrays.update(other.arrays)
        self.listings.update(other.listings)
        for array, rows in other.rows.items():
            self.rows.setdefault(array, {}).update(rows)
        self.absent |= other.absent
        self.expanded |= other.expanded

    def mark_absent(self, requested: ReadPlan) -> None:
        """Record which of ``requested`` the fetch did not produce.

        Without this, "asked for and genuinely missing" is
        indistinguishable from "not asked for yet", and the executor keeps
        re-requesting a cell that will never arrive — spinning to the
        round limit and then reporting non-convergence, when the real
        answer is that the object is not there.

        This is the "splice misses back as holes" half of the batch
        contract: a result that came back empty stays in the record,
        rather than being dropped as though it had never been sought.
        """
        for cell in requested.cells:
            if (cell.array, cell.key) not in self.chunks:
                self.absent.add((cell.array, cell.key))
        for path in requested.arrays:
            if path not in self.arrays:
                self.absent.add(("array", path))
        for path in requested.listings:
            if path not in self.listings:
                self.absent.add(("list", path))
        for req in requested.rows:
            known = self.rows.get(req.array, {})
            for row in req.rows:
                if row not in known:
                    self.absent.add(("row", req.array, row))
        # Anything that did arrive is no longer absent -- a later round
        # may fetch what an earlier one could not.
        self.absent -= {(a, k) for a, k in self.chunks}

    def as_prefetch_plan(self, group: Group) -> list[tuple[str, list[str]]]:
        """Cells this snapshot holds, in
        :meth:`~zarr_vectors.core.group.Group.batched_reads` form and
        rebased onto ``group``.

        The rebase is not cosmetic.  ``Group._prefetch_cache`` is keyed by
        the array name the caller passes to ``read_bytes`` — which is
        relative to the Group the context was opened on — whereas an
        offline snapshot keys everything root-relative.  A plan handed to
        ``batched_reads`` on a level group must therefore name
        ``"vertices"``, not ``"0/vertices"``.  Cells outside ``group``
        are dropped: that handle could not address them anyway.
        """
        base = group._zarr.path.strip("/")
        grouped: dict[str, list[str]] = {}
        for array_path, chunk_key in self.chunks:
            name = array_path
            if base:
                if name == base:
                    name = ""
                elif name.startswith(base + "/"):
                    name = name[len(base) + 1:]
                else:
                    continue
            grouped.setdefault(name, []).append(chunk_key)
        return [(name, keys) for name, keys in sorted(grouped.items())]

    def __repr__(self) -> str:
        return (
            f"Snapshot(nodes={len(self.nodes)}, cells={len(self.chunks)}, "
            f"rows={sum(len(r) for r in self.rows.values())}, "
            f"arrays={len(self.arrays)}, listings={len(self.listings)}, "
            f"misses={len(self.misses)})"
        )
