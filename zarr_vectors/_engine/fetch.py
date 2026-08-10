"""The port: one method, plural in and plural out.

A :class:`Fetcher` turns a whole :class:`~zarr_vectors._engine.plan.ReadPlan`
into a whole :class:`~zarr_vectors._engine.snapshot.Snapshot`.  That is the
entire contract — one method, deliberately.

Zarr's ``ByteGetter`` protocol has a single method and its sharding codec
satisfies it with an in-memory dict, which is why a sharded read and an
unsharded read run identical code.  The ``StorageTransformer`` that
approach replaced had roughly twenty methods, every one of which had to
sustain the illusion — listing, membership, prefix erasure — and the one
that mattered most for writes was never implemented at all.  A narrow
port is not a stylistic preference; it is what makes the fake credible
and therefore what makes the decoupling testable.

Three properties every implementation must hold, taken from
``ShardingStorageTransformer.get_partial_values``:

**One batch out, not N.**  Cells are grouped by array and issued through
a single gather.  A fetcher that loops is a fetcher that has given the
whole design away.

**A request-scoped index cache.**  Node resolutions and presence
manifests are resolved once per ``fetch`` call and dropped at the end.
Caching them longer would mean holding a handle across a write that
could invalidate it.

**Misses are recorded, never dropped or reordered.**  A cell that came
back absent must be distinguishable from one that was never asked for,
or the executor cannot tell "genuinely missing" from "not fetched yet"
and will not converge.

Implementations here:

=====================  ========================================
:class:`GroupFetcher`   a live store, driven through ``sync()``
:class:`AsyncFetcher`   the same code awaited — for Pyodide
:class:`DictFetcher`    a plain dict; no zarr, no filesystem
:class:`RecordingFetcher`  wraps another and records every plan
=====================  ========================================
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np
import zarr
from zarr.core.sync import sync

from zarr_vectors._engine.plan import ReadPlan, RowRequest
from zarr_vectors._engine.snapshot import Snapshot
from zarr_vectors.core.group import _ABSENT, Group


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What a backing store can actually do.

    Read *before* planning, so a missing capability produces a different
    plan rather than a failed one.  Declaring capabilities and degrading
    gracefully is the half of the storage-transformer design that
    survived: ``supports_efficient_get_partial_values`` guarded a fast
    path and fell back to a generic one, and the fallback was never
    optional.
    """

    can_list: bool = True
    """Store has a listing operation.  False for a browser ``fetch``-backed
    Store, where HTTP GET alone cannot enumerate a prefix."""

    can_gather: bool = True
    """``asyncio.gather`` over the store is safe.  False for icechunk,
    which tracks arrays as session-managed entities that the gather
    pattern bypasses."""

    can_select_rows: bool = True
    """Coordinate selection is available, so N rows of a large array cost
    one request rather than a whole-array read."""

    can_range_get: bool = True
    """Byte-range GETs are supported, which is what makes reading a single
    cell out of a shard cheap."""


@runtime_checkable
class Fetcher(Protocol):
    """Turn a plan into a snapshot.  One method."""

    capabilities: Capabilities

    def fetch(self, plan: ReadPlan) -> Snapshot:
        ...


# =====================================================================
# The shared async core.  GroupFetcher drives it through sync();
# AsyncFetcher awaits it.  One implementation, two drive modes -- the
# same reason there is no async mirror of each read_*.
# =====================================================================


async def _fetch_rows(
    async_root: Any,
    requests: tuple[RowRequest, ...],
    *,
    can_select_rows: bool,
) -> dict[str, dict[int, Any]]:
    """Read the requested rows of each standalone array.

    With coordinate selection this is one request per array regardless of
    how many rows are wanted — the difference between reading 200 object
    manifests and reading twenty-one million.  Without it, fall back to a
    whole-array read and index locally: slower, but the caller's code is
    identical either way, which is the point of putting the fallback here
    rather than at every call site.
    """
    async def one(req: RowRequest) -> tuple[str, dict[int, Any]]:
        try:
            node = await async_root.getitem(req.array)
        except KeyError:
            return req.array, {}
        if not isinstance(node, zarr.AsyncArray):
            return req.array, {}
        idx = np.asarray(req.rows, dtype=np.int64)
        if idx.size == 0:
            return req.array, {}
        if can_select_rows:
            selected = np.asarray(await node.get_coordinate_selection((idx,)))
        else:
            selected = np.asarray(await node.getitem(slice(None)))[idx]
        return req.array, dict(zip((int(r) for r in req.rows), selected))

    if not requests:
        return {}
    pairs = await asyncio.gather(*(one(r) for r in requests))
    return {array: rows for array, rows in pairs if rows}


async def _fetch_all(
    async_root: Any,
    plan: ReadPlan,
    caps: Capabilities,
) -> Snapshot:
    """Execute one whole plan against a live store.

    Node resolution happens first because it is what makes the rest
    cheap: an array's ``nonempty_chunks`` attribute arrives with its
    metadata, so resolving one node tells us every cell that array holds.
    That expansion is the semantic rewrite this layer exists to perform —
    the caller asked for an array, we issue reads for its cells.  Without
    it, a reader that walks chunks in a plain loop discovers exactly one
    per round and a store with more chunks than the round limit never
    finishes.

    The remaining four kinds are independent, so they go out together.
    """
    from zarr_vectors.core.aio import (
        _fetch_arrays,
        _fetch_chunks,
        _fetch_listings,
        _implied_chunks,
        _resolve_nodes,
    )

    nodes = await _resolve_nodes(async_root, set(plan.nodes) | set(plan.expand))

    want_cells = {(c.array, c.key) for c in plan.cells}
    # Fan out only where the plan asked for it. Implying an array's whole
    # contents from its node is right for "read the level" and ruinous
    # for "read this box" -- the same node resolution serves both, so the
    # difference has to come from the plan rather than from the fetcher.
    want_cells |= _implied_chunks(
        {p: n for p, n in nodes.items() if p in set(plan.expand)}
    )

    chunks, arrays, listings, rows = await asyncio.gather(
        _fetch_chunks(async_root, nodes, want_cells),
        _fetch_arrays(async_root, set(plan.arrays)),
        _fetch_listings(async_root, set(plan.listings)) if caps.can_list
        else _nothing({}),
        _fetch_rows(async_root, plan.rows, can_select_rows=caps.can_select_rows),
    )
    snapshot = Snapshot(
        nodes=nodes, chunks=chunks, arrays=arrays, listings=listings, rows=rows,
    )
    snapshot.expanded |= set(plan.expand)
    return snapshot


async def _nothing(value: Any) -> Any:
    """An already-satisfied coroutine, so the gather above stays one
    expression rather than growing a branch per capability."""
    return value


def _detect_capabilities(store: Any) -> Capabilities:
    from zarr_vectors.core._batch_reader import _is_icechunk_store

    if _is_icechunk_store(store):
        # Icechunk manages arrays as session entities; the async gather
        # pattern bypasses that contract, and flush_prefetch already
        # routes such stores to a serial fallback.
        return Capabilities(can_gather=False)
    return Capabilities(
        can_list=bool(getattr(store, "supports_listing", True)),
    )


# =====================================================================
# Implementations
# =====================================================================


class GroupFetcher:
    """Fetch from a live store, synchronously.

    The default for every non-browser read.  Concurrency comes from
    driving the same async core the Pyodide path uses through zarr's
    ``sync()`` bridge, so a plan of 500 cells costs one round-trip rather
    than 500 — and there is exactly one implementation of each fetch kind
    to keep correct.
    """

    def __init__(self, root: Group, *, capabilities: Capabilities | None = None) -> None:
        base = root._zarr.path.strip("/")
        if base:
            raise ValueError(
                f"GroupFetcher needs the Group at the store root, but got one "
                f"at {base!r}. Plans key their paths root-relative, so a "
                f"sub-group handle would resolve every path against the wrong "
                f"base and silently fetch nothing."
            )
        self._root = root
        self._async_root = root._zarr._async_group
        self.capabilities = (
            capabilities if capabilities is not None
            else _detect_capabilities(root._zarr.store)
        )

    def fetch(self, plan: ReadPlan) -> Snapshot:
        if not plan:
            return Snapshot.empty()
        if not self.capabilities.can_gather:
            return self._fetch_serial(plan)
        return sync(_fetch_all(self._async_root, plan, self.capabilities))

    def _fetch_serial(self, plan: ReadPlan) -> Snapshot:
        """Serial path for stores the gather pattern would misuse.

        Same result, one request at a time.  Reuses
        ``_batch_reader._sync_fallback`` for cells so the vlen cell layout
        and the ``chunk_grid_origin`` translation stay in one place.
        """
        from zarr_vectors.core._batch_reader import _sync_fallback
        from zarr_vectors.core.group import _NONEMPTY_CHUNKS_ATTR

        zg = self._root._zarr
        nodes: dict[str, Any] = {}
        for path in (*plan.nodes, *plan.expand):
            try:
                nodes[path] = zg[path]
            except KeyError:
                nodes[path] = _ABSENT

        by_array: dict[str, list[str]] = {}
        for cell in plan.cells:
            by_array.setdefault(cell.array, []).append(cell.key)
        for path in plan.expand:
            node = nodes.get(path)
            if not isinstance(node, zarr.Array):
                continue
            present = node.attrs.get(_NONEMPTY_CHUNKS_ATTR)
            if isinstance(present, (list, tuple)):
                by_array.setdefault(path, []).extend(str(k) for k in present)
        chunks = _sync_fallback(zg, [(k, sorted(set(v))) for k, v in by_array.items()])

        arrays: dict[str, Any] = {}
        for path in plan.arrays:
            try:
                node = zg[path]
            except KeyError:
                continue
            if isinstance(node, zarr.Array):
                arrays[path] = node[:]

        listings: dict[str, list[str]] = {}
        if self.capabilities.can_list:
            for path in plan.listings:
                try:
                    node = zg[path] if path else zg
                except KeyError:
                    listings[path] = []
                    continue
                if isinstance(node, zarr.Group):
                    listings[path] = sorted(
                        set(node.group_keys()) | set(node.array_keys())
                    )

        rows: dict[str, dict[int, Any]] = {}
        for req in plan.rows:
            try:
                node = zg[req.array]
            except KeyError:
                continue
            if not isinstance(node, zarr.Array):
                continue
            idx = np.asarray(req.rows, dtype=np.int64)
            if idx.size == 0:
                continue
            selected = np.asarray(
                node.get_coordinate_selection((idx,))
                if self.capabilities.can_select_rows
                else np.asarray(node[:])[idx]
            )
            rows[req.array] = dict(zip((int(r) for r in req.rows), selected))

        snapshot = Snapshot(
            nodes=nodes, chunks=chunks, arrays=arrays, listings=listings, rows=rows,
        )
        snapshot.expanded |= set(plan.expand)
        return snapshot


class AsyncFetcher:
    """Fetch from a live store without ever entering ``sync()``.

    For hosts where a synchronous zarr call cannot work — Pyodide, where
    it needs WebAssembly stack switching and deadlocks the JS event loop
    under concurrency.  Consumed by :func:`~zarr_vectors._engine.execute.aexecute`.

    ``fetch`` is the async one; the sync method raises rather than
    bridging, because bridging is the exact thing this class exists to
    avoid and a silent ``sync()`` here would be a deadlock in production
    and invisible in tests.
    """

    def __init__(
        self, root: Group, *, capabilities: Capabilities | None = None,
    ) -> None:
        self._root = root
        self._async_root = root._zarr._async_group
        self.capabilities = (
            capabilities if capabilities is not None
            else _detect_capabilities(root._zarr.store)
        )

    async def afetch(self, plan: ReadPlan) -> Snapshot:
        if not plan:
            return Snapshot.empty()
        return await _fetch_all(self._async_root, plan, self.capabilities)

    def fetch(self, plan: ReadPlan) -> Snapshot:  # pragma: no cover - guard
        raise TypeError(
            "AsyncFetcher.fetch() is not available: use aexecute(), which "
            "awaits afetch(). Bridging through sync() here would reintroduce "
            "the blocking call this fetcher exists to avoid."
        )


class DictFetcher:
    """A whole store as four plain dicts.

    No zarr, no filesystem, no network.  This is the fake that makes the
    decoupling checkable: if every facade read can be satisfied through a
    one-method port backed by a dict, then nothing above the port is
    reaching around it.  A future change that adds a stray
    ``zarr_group[...]`` to a decoder, or a ``Path`` operation to a
    resolver, fails here immediately and nowhere else.

    Also the honest way to test capability degradation.  Constructing a
    real store that cannot list is awkward; constructing this one with
    ``Capabilities(can_list=False)`` is a keyword argument.
    """

    def __init__(
        self,
        *,
        nodes: dict[str, Any] | None = None,
        cells: dict[tuple[str, str], bytes] | None = None,
        arrays: dict[str, Any] | None = None,
        listings: dict[str, list[str]] | None = None,
        capabilities: Capabilities | None = None,
    ) -> None:
        self._nodes = dict(nodes or {})
        self._cells = dict(cells or {})
        self._arrays = dict(arrays or {})
        self._listings = dict(listings or {})
        self.capabilities = capabilities or Capabilities()
        self.calls: list[ReadPlan] = []

    def fetch(self, plan: ReadPlan) -> Snapshot:
        self.calls.append(plan)
        # A path we know nothing about resolves to _ABSENT, not to a
        # hole: "probed and missing" has to be answerable as definitely
        # as "found", or a probe for a legitimately absent array looks
        # like a prefetch gap and the executor spins.
        nodes = {
            p: self._nodes.get(p, _ABSENT) for p in (*plan.nodes, *plan.expand)
        }
        cells = {
            (c.array, c.key): self._cells[(c.array, c.key)]
            for c in plan.cells
            if (c.array, c.key) in self._cells
        }
        arrays = {p: self._arrays[p] for p in plan.arrays if p in self._arrays}
        listings = {}
        if self.capabilities.can_list:
            listings = {
                p: self._listings[p] for p in plan.listings if p in self._listings
            }
        rows: dict[str, dict[int, Any]] = {}
        for req in plan.rows:
            whole = self._arrays.get(req.array)
            if whole is None:
                continue
            rows[req.array] = {int(r): whole[int(r)] for r in req.rows}
        if plan.expand:
            for (array, key), payload in self._cells.items():
                if array in set(plan.expand):
                    cells[(array, key)] = payload
        snapshot = Snapshot(
            nodes=nodes, chunks=cells, arrays=arrays, listings=listings, rows=rows,
        )
        snapshot.expanded |= set(plan.expand)
        return snapshot

    async def afetch(self, plan: ReadPlan) -> Snapshot:
        """The same fake, awaitable.

        A fake that only works in one drive mode would leave the async
        path testable solely against a live store — which is the path
        that most needs a fake, since its real consumer is a browser.
        """
        return self.fetch(plan)

    # -- test helpers -------------------------------------------------

    def with_capabilities(self, **kw: bool) -> DictFetcher:
        """Same data, different declared capabilities."""
        clone = DictFetcher(
            nodes=self._nodes, cells=self._cells, arrays=self._arrays,
            listings=self._listings,
            capabilities=replace(self.capabilities, **kw),
        )
        return clone


class RecordingFetcher:
    """Delegate to another fetcher, remembering every plan.

    The instrument behind "did this change alter the I/O a read
    performs?".  ``plans`` is what a golden-plan test compares.
    """

    def __init__(self, inner: Fetcher) -> None:
        self._inner = inner
        self.plans: list[ReadPlan] = []

    @property
    def capabilities(self) -> Capabilities:
        return self._inner.capabilities

    def fetch(self, plan: ReadPlan) -> Snapshot:
        self.plans.append(plan)
        return self._inner.fetch(plan)

    async def afetch(self, plan: ReadPlan) -> Snapshot:
        # ``afetch`` is not on the Fetcher protocol -- only the two
        # fetchers that can be awaited have it -- so this is a genuine
        # attribute probe rather than a type hole.
        inner_afetch = getattr(self._inner, "afetch", None)
        if inner_afetch is None:
            raise TypeError(
                f"{type(self._inner).__name__} cannot be awaited: it has no "
                f"afetch(). Wrap an AsyncFetcher or a DictFetcher instead."
            )
        result: Snapshot = await inner_afetch(plan)
        self.plans.append(plan)
        return result

    @property
    def rounds(self) -> int:
        """Non-empty fetch rounds — the round-trip count that matters."""
        return sum(1 for p in self.plans if p)
