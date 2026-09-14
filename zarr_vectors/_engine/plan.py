"""What a read wants, before any of it has been fetched.

A :class:`ReadPlan` is a typed, mergeable, inspectable envelope around the
five kinds of request the ``read_*`` functions actually make.  It carries
no results and touches no store: a resolver produces one from metadata
alone, and a :class:`~zarr_vectors._engine.fetch.Fetcher` consumes it.

The five kinds are not a taxonomy invented here — they are the four
:class:`~zarr_vectors.core.group._OfflineSession` dimensions that the
async read path already proved sufficient, plus ``rows`` for the
selective reads that path never needed:

``nodes``
    Paths to resolve to a Zarr node.  Answers "does this array exist,
    and what are its attributes".
``cells``
    One vlen cell of a chunk-grid array — the bulk of every read.
``rows``
    Specific rows of a standalone array.  This is the difference between
    reading 200 of 21 million object manifests and reading all of them.
``arrays``
    A whole standalone array, read end-to-end.
``listings``
    A group's immediate child names.  The one kind a plain fetch-backed
    Store cannot serve, which is why it is tracked separately rather than
    folded into ``nodes``.
``expand``
    Arrays to resolve *and* fan out: fetch every cell they are known to
    hold.  This is the difference between "read the whole level" and
    "read this box", and it has to be stated rather than inferred --
    resolving an array's node yields its ``nonempty_chunks``, so a
    fetcher that always fanned out would turn a 24-cell bbox read into a
    512-cell scan the moment the reader touched the node.

Paths are **root-relative** throughout, matching ``_OfflineSession``, so
a plan means the same thing regardless of which Group in the tree built
it.  :meth:`ReadPlan.by_array` rebases onto a given group for the one
consumer that needs relative names.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zarr_vectors._engine.snapshot import Snapshot

# Miss tags used by ``_OfflineSession``.  A miss is either a bare ``str``
# (a node path), a 2-tuple tagged with one of these (a whole array or a
# listing), or an untagged 2-tuple of ``(array_path, chunk_key)`` (a
# cell).  Kept here so the encoding lives in one place rather than being
# re-derived at each inspection site.
_MISS_ARRAY = "array"
_MISS_LIST = "list"


@dataclass(frozen=True, slots=True, order=True)
class CellRequest:
    """One vlen cell of a chunk-grid array.

    Args:
        array: Root-relative array path, e.g. ``"0/vertices"``.
        key: Chunk key in the usual dotted form, e.g. ``"3.1.2"``.
    """

    array: str
    key: str


@dataclass(frozen=True, slots=True, order=True)
class RowRequest:
    """Specific rows of a standalone array.

    ``rows`` is a sorted tuple rather than an ndarray so the request is
    hashable and two plans can be merged without a numpy import.  Callers
    holding an index array pass ``RowRequest.of(path, idx)``.
    """

    array: str
    rows: tuple[int, ...]

    @classmethod
    def of(cls, array: str, rows: Iterable[int]) -> RowRequest:
        return cls(array, tuple(sorted({int(r) for r in rows})))


@dataclass(frozen=True, slots=True)
class PlanCost:
    """A plan's shape, for :meth:`ReadPlan.explain` and for asserting in
    tests that an internal change did not alter the I/O a read performs.
    """

    nodes: int = 0
    cells: int = 0
    rows: int = 0
    row_indices: int = 0
    arrays: int = 0
    listings: int = 0
    expand: int = 0

    @property
    def total(self) -> int:
        return (
            self.nodes + self.cells + self.rows + self.arrays
            + self.listings + self.expand
        )

    def __str__(self) -> str:
        parts = [
            f"{self.nodes} node(s)",
            f"{self.cells} cell(s)",
            f"{self.rows} row-request(s) covering {self.row_indices} row(s)",
            f"{self.arrays} whole array(s)",
            f"{self.listings} listing(s)",
            f"{self.expand} fanned-out array(s)",
        ]
        return ", ".join(parts)


def _norm(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _norm_rows(requests: Iterable[RowRequest]) -> tuple[RowRequest, ...]:
    """Collapse row requests so each array appears at most once."""
    merged: dict[str, set[int]] = {}
    for req in requests:
        merged.setdefault(req.array, set()).update(req.rows)
    return tuple(
        RowRequest(array, tuple(sorted(rows))) for array, rows in sorted(merged.items())
    )


@dataclass(frozen=True, slots=True)
class ReadPlan:
    """Everything a read wants, in one value.

    Immutable and mergeable, so resolvers compose by ``|`` without
    coordinating, and a plan can be compared against a golden in a test.
    """

    nodes: tuple[str, ...] = ()
    cells: tuple[CellRequest, ...] = ()
    rows: tuple[RowRequest, ...] = ()
    arrays: tuple[str, ...] = ()
    listings: tuple[str, ...] = ()
    expand: tuple[str, ...] = ()

    # ---------------- construction ----------------

    @classmethod
    def of(
        cls,
        *,
        nodes: Iterable[str] = (),
        cells: Iterable[CellRequest | tuple[str, str]] = (),
        rows: Iterable[RowRequest] = (),
        arrays: Iterable[str] = (),
        listings: Iterable[str] = (),
        expand: Iterable[str] = (),
    ) -> ReadPlan:
        """Build a normalised plan.  Duplicates collapse and order is
        canonical, so two plans wanting the same I/O compare equal.
        """
        return cls(
            expand=_norm(expand),
            nodes=_norm(nodes),
            cells=tuple(sorted({
                c if isinstance(c, CellRequest) else CellRequest(c[0], c[1])
                for c in cells
            })),
            rows=_norm_rows(rows),
            arrays=_norm(arrays),
            listings=_norm(listings),
        )

    @classmethod
    def for_cells(cls, array: str, keys: Iterable[str]) -> ReadPlan:
        """Exactly these cells of one array, and nothing implied."""
        return cls.of(nodes=[array], cells=[CellRequest(array, k) for k in keys])

    @classmethod
    def for_array(cls, array: str) -> ReadPlan:
        """Every cell of one array, however many that turns out to be."""
        return cls.of(expand=[array])

    @classmethod
    def from_misses(cls, misses: Iterable[Any]) -> ReadPlan:
        """Decode what a decoder recorded it could not serve.

        This is the inverse of the miss vocabulary that
        ``_OfflineSession`` writes and that
        :func:`zarr_vectors.core.aio.read_async` decoded inline.  Having
        it in one place is what lets the executor be generic over
        decoders.

        A cell miss also requests its array's **node**, and that is what
        makes discovery cheap rather than quadratic.  Resolving an array
        yields its ``nonempty_chunks`` attribute, from which the fetcher
        implies every cell the array holds — so one node resolution
        replaces one round per cell.  Without it, a reader that walks
        chunks in a plain loop reveals exactly one cell per round: a
        64-chunk ``read_points`` took 66 rounds, of which 64 were single
        cells of ``vertex_fragments``, an array no miss ever named
        directly.

        This applies to *discovered* work only.  A plan built by a
        resolver states its cells explicitly and omits the array from
        ``nodes`` when it wants precision — a bbox query over a large
        store must not fan out to every cell in the level.

        A cell miss is an *untagged* pair, so an array whose path is
        literally ``"array"`` or ``"list"`` would be misread.  That
        ambiguity predates this module and is preserved rather than
        silently changed; no ZV array is named either.
        """
        nodes: list[str] = []
        cells: list[CellRequest] = []
        arrays: list[str] = []
        listings: list[str] = []
        expand: list[str] = []
        for miss in misses:
            if isinstance(miss, str):
                nodes.append(miss)
            elif isinstance(miss, tuple) and len(miss) == 2:
                tag, value = miss
                if tag == _MISS_ARRAY:
                    arrays.append(value)
                elif tag == _MISS_LIST:
                    listings.append(value)
                else:
                    cells.append(CellRequest(tag, value))
                    expand.append(tag)
        return cls.of(
            nodes=nodes, cells=cells, arrays=arrays, listings=listings,
            expand=expand,
        )

    # ---------------- algebra ----------------

    def merge(self, other: ReadPlan) -> ReadPlan:
        """Union of two plans."""
        if not other:
            return self
        if not self:
            return other
        return ReadPlan.of(
            nodes=self.nodes + other.nodes,
            cells=self.cells + other.cells,
            rows=self.rows + other.rows,
            arrays=self.arrays + other.arrays,
            listings=self.listings + other.listings,
            expand=self.expand + other.expand,
        )

    __or__ = merge

    def minus(self, snapshot: Snapshot) -> ReadPlan:
        """What this plan still wants that ``snapshot`` cannot already
        answer.

        Refetching what the snapshot holds is not merely wasteful — it is
        the difference between a loop that converges and one that spins,
        because "we asked again and learned nothing" is precisely the
        stall condition the executor tests for.

        An item the snapshot has recorded as *absent* counts as answered.
        It was asked for and it is not there; asking again cannot change
        that, and treating it as still-wanted is what turns a missing
        object into a hundred wasted rounds.  ``nodes`` needs no such
        check because absence is recorded in-band as ``_ABSENT``.
        """
        have_cells = snapshot.chunks
        gone = snapshot.absent
        rows_left = []
        for req in self.rows:
            known = snapshot.rows.get(req.array, {})
            missing = tuple(
                r for r in req.rows
                if r not in known and ("row", req.array, r) not in gone
            )
            if missing:
                rows_left.append(RowRequest(req.array, missing))
        return ReadPlan(
            nodes=tuple(p for p in self.nodes if p not in snapshot.nodes),
            cells=tuple(
                c for c in self.cells
                if (c.array, c.key) not in have_cells
                and (c.array, c.key) not in gone
            ),
            rows=tuple(rows_left),
            arrays=tuple(
                p for p in self.arrays
                if p not in snapshot.arrays and ("array", p) not in gone
            ),
            listings=tuple(
                p for p in self.listings
                if p not in snapshot.listings and ("list", p) not in gone
            ),
            expand=tuple(p for p in self.expand if p not in snapshot.expanded),
        )

    def without_listings(self) -> ReadPlan:
        """This plan with listing requests dropped.

        For a store that cannot list (a browser ``fetch``-backed Store),
        asking is not a slow answer but no answer at all, so the resolver
        drops them and reaches the same data another way.
        """
        return replace(self, listings=())

    # ---------------- consumption ----------------

    def by_array(self, *, prefix: str = "") -> list[tuple[str, list[str]]]:
        """Cells grouped per array, in the shape the existing batch
        machinery already speaks.

        ``list[tuple[array_name, [chunk_key, ...]]]`` is exactly what
        :meth:`zarr_vectors.core.group.Group.batched_reads`,
        :func:`zarr_vectors.core._batch_reader.flush_prefetch` and
        ``_gather_plan`` consume.  This method is the whole reason those
        three need no rewriting.

        Args:
            prefix: Root-relative path of the group the names should be
                relative *to*.  ``Group._prefetch_cache`` is keyed by the
                name the caller passes to ``read_bytes``, which is
                relative to that Group — so serving a level group's reads
                requires rebasing.  Cells outside ``prefix`` are dropped,
                since that group could not address them anyway.
        """
        base = prefix.strip("/")
        grouped: dict[str, list[str]] = {}
        for cell in self.cells:
            name = cell.array
            if base:
                if name == base:
                    name = ""
                elif name.startswith(base + "/"):
                    name = name[len(base) + 1:]
                else:
                    continue
            grouped.setdefault(name, []).append(cell.key)
        return [(name, keys) for name, keys in sorted(grouped.items())]

    def cost(self) -> PlanCost:
        return PlanCost(
            nodes=len(self.nodes),
            cells=len(self.cells),
            rows=len(self.rows),
            row_indices=sum(len(r.rows) for r in self.rows),
            arrays=len(self.arrays),
            listings=len(self.listings),
            expand=len(self.expand),
        )

    def explain(self) -> str:
        """Human-readable summary of the I/O this plan implies."""
        if not self:
            return "empty plan (the reader will discover what it needs)"
        return str(self.cost())

    # ---------------- protocol ----------------

    def __len__(self) -> int:
        return (
            len(self.nodes)
            + len(self.cells)
            + len(self.rows)
            + len(self.arrays)
            + len(self.listings)
            + len(self.expand)
        )

    def __bool__(self) -> bool:
        return len(self) > 0


def cells_for(array: str, keys: Sequence[str]) -> tuple[CellRequest, ...]:
    """``[CellRequest(array, k) for k in keys]``, as a tuple."""
    return tuple(CellRequest(array, k) for k in keys)
