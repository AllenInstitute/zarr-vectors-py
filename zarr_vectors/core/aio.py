"""Async read path.

Every public ``read_*`` in this package is synchronous, and under CPython
that is fine: zarr's own ``sync()`` bridge runs the coroutine on a
background event loop and blocks the caller.  Under Pyodide it is not.
There, a synchronous zarr call can only work via WebAssembly stack
switching, invoked from JS through a *promising* entrypoint, and under
concurrent requests that deadlocks the JS event loop outright.  So a
browser host needs a path that never blocks on ``sync()`` at all.

The obvious shape — an async mirror of each ``read_*`` — is the wrong
one.  Those functions are mostly *pure*: decoding fragment indices,
applying bbox and attribute filters, assembling the CSR-ish output.  Only
their I/O needs awaiting, and duplicating eight of them would fork the
format logic in two, which is exactly the failure mode this package
exists to avoid.

So instead of rewriting the readers, this module supplies their I/O up
front and then replays them offline:

1. Resolve nodes and fetch chunk cells with ``await`` (concurrently).
2. Run the ordinary sync ``read_*`` against a Group primed with that
   snapshot, via :meth:`~zarr_vectors.core.group.Group.offline_reads`.
   It performs no store access, so it cannot reach ``sync()``.
3. If it asked for something the snapshot lacked, the miss was recorded.
   Fetch that too, and go round again.

Step 3 is what makes this general.  Knowing up front exactly which paths
and chunks a given ``read_*`` will touch would mean re-deriving each
reader's logic here — the duplication again, one level down.  Letting the
reader *tell us* what it wants costs a few extra round-trips and works
for every reader, present and future, with no per-type knowledge.

Convergence is fast because each round fetches everything discovered so
far concurrently: a ``read_points`` settles in about three rounds
(root/level groups, then arrays, then cells) against the ~28 serial GETs
the sync path issues.

The loop is also why misses are *recorded* rather than merely raised:
several readers wrap optional metadata reads in ``except Exception``,
which would swallow the signal.  See
:class:`~zarr_vectors.core.group._OfflineSession`.

That loop now lives in :mod:`zarr_vectors._engine.execute`, generalised
over *where the bytes come from* and over whether a gap is fatal.  It
turned out that the same three steps describe an ordinary batched sync
read (fetch a plan, run the reader, fall through on a miss) as well as
this one, so keeping two copies would have re-forked exactly what this
module exists to keep single.  What remains here is the async half of
the port — the five fetch coroutines below, which
:class:`~zarr_vectors._engine.fetch.AsyncFetcher` drives — plus
:func:`read_async` as the public name for the composition.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

import zarr

from zarr_vectors._engine.execute import aexecute
from zarr_vectors._engine.fetch import AsyncFetcher
from zarr_vectors.core.group import _ABSENT, Group

T = TypeVar("T")

# Safety net for the discovery loop.  Each round must strictly grow the
# snapshot (see the invariant check in :func:`read_async`), so termination
# does not rest on this — it is here to turn a hypothetical
# non-converging reader into a diagnosable error instead of a hang.
#
# Raised from 12: read_polylines reassembles tracts across chunks by following
# cross-chunk links, and on a whole-brain store a long tract's chain of chunks
# is discovered a few per round, so a level-0-ish read legitimately needs many
# more passes than a simple read_points. The grow-invariant still bounds it.
_MAX_ROUNDS = 100


async def open_store_async(
    path: Any,
    *,
    mode: str = "r",
    **kwargs: Any,
) -> Group:
    """Open a ZV store without ever entering zarr's ``sync()`` bridge.

    Accepts the same ``path`` forms as
    :func:`~zarr_vectors.core.store.open_store`, including a pre-built
    ``zarr.abc.store.Store`` — which is the point for a browser host,
    where the only way in is a fetch-backed Store the caller constructs
    itself.

    Returns an ordinary :class:`~zarr_vectors.core.group.Group`.  It
    wraps the async group in zarr's sync facade, which is a pure
    dataclass wrapper and does no I/O of its own; reads through it are
    still synchronous, so drive them with :func:`read_async` rather than
    calling ``read_*`` on it directly.
    """
    from zarr_vectors.core.store import _make_zarr_store_with_session

    if isinstance(path, Group):
        return path

    store, _session = _make_zarr_store_with_session(path, mode=mode, **kwargs)
    async_group = await zarr.api.asynchronous.open_group(
        store=store, path="/", mode=mode,
    )
    return Group._from_zarr(zarr.Group(async_group))


async def _resolve_nodes(
    root: zarr.AsyncGroup,
    paths: set[str],
) -> dict[str, Any]:
    """Resolve ``paths`` (root-relative) concurrently.

    A path that does not exist maps to ``_ABSENT`` rather than being
    dropped: the snapshot has to be able to answer "no" as definitely as
    it answers "yes", or an ``array_exists`` probe for a legitimately
    missing array would look like a prefetch gap and loop forever.
    """
    async def one(path: str) -> tuple[str, Any]:
        try:
            node = await root.getitem(path)
        except KeyError:
            return path, _ABSENT
        # Wrap in the sync facade the readers expect.  Both wrappers are
        # plain dataclasses over the async object — no I/O here.
        if isinstance(node, zarr.AsyncArray):
            return path, zarr.Array(node)
        return path, zarr.Group(node)

    if not paths:
        return {}
    return dict(await asyncio.gather(*(one(p) for p in sorted(paths))))


async def _fetch_chunks(
    root: zarr.AsyncGroup,
    nodes: dict[str, Any],
    keys: set[tuple[str, str]],
) -> dict[tuple[str, str], bytes]:
    """Fetch ``(array_path, chunk_key)`` cells concurrently.

    Reuses :func:`~zarr_vectors.core._batch_reader._gather_plan`, which
    is already pure async and already knows the vlen-bytes cell layout
    and the ``chunk_grid_origin`` translation — the same code the sync
    batched-read path fans out with.
    """
    from zarr_vectors.core._batch_reader import _gather_plan

    if not keys:
        return {}

    by_array: dict[str, list[str]] = {}
    for array_path, chunk_key in sorted(keys):
        by_array.setdefault(array_path, []).append(chunk_key)

    # _gather_plan resolves array names against the group it is given, so
    # hand it the root and use root-relative paths as the "names".
    return await _gather_plan(root, list(by_array.items()))


def _implied_chunks(nodes: dict[str, Any]) -> set[tuple[str, str]]:
    """Every cell the just-resolved chunk-grid arrays are known to hold.

    Without this the loop still terminates, but slowly and in the worst
    way: a reader that walks its chunks in a plain loop (rather than
    declaring a bulk plan the way ``batched_reads`` callers do) aborts at
    the *first* cell it lacks, so each round discovers exactly one more
    and a store with more chunks than ``_MAX_ROUNDS`` never finishes.

    An array's ``nonempty_chunks`` attribute already lists its populated
    cells, and it arrives with the node metadata at no extra cost — so
    once the node is resolved, the whole array can be queued at once.
    That turns per-chunk discovery into one concurrent fetch and holds
    the round count at two or three regardless of store size.
    """
    from zarr_vectors.core.group import _NONEMPTY_CHUNKS_ATTR

    out: set[tuple[str, str]] = set()
    for path, node in nodes.items():
        if not isinstance(node, zarr.Array):
            continue
        present = node.attrs.get(_NONEMPTY_CHUNKS_ATTR)
        if not present:
            continue
        for chunk_key in present:
            out.add((path, chunk_key))
    return out


async def _fetch_arrays(
    root: zarr.AsyncGroup,
    paths: set[str],
) -> dict[str, Any]:
    """Read whole standalone arrays concurrently.

    These are the non-chunk-grid arrays — ``object_index/manifests``,
    object attributes — which readers consume end-to-end rather than a
    cell at a time.
    """
    async def one(path: str) -> tuple[str, Any]:
        try:
            node = await root.getitem(path)
        except KeyError:
            return path, None
        if not isinstance(node, zarr.AsyncArray):
            return path, None
        return path, await node.getitem(slice(None))

    if not paths:
        return {}
    pairs = await asyncio.gather(*(one(p) for p in sorted(paths)))
    return {p: v for p, v in pairs if v is not None}


async def _fetch_listings(
    root: zarr.AsyncGroup,
    paths: set[str],
) -> dict[str, list[str]]:
    """List each group's immediate children concurrently.

    The one dimension a plain fetch-backed Store cannot serve: HTTP GET
    alone has no listing operation.  A store that cannot list raises
    here, and the empty result propagates as a permanent miss so
    :func:`read_async` reports which read needed listing instead of
    spinning.
    """
    async def one(path: str) -> tuple[str, list[str]] | None:
        try:
            node = await root.getitem(path) if path else root
        except KeyError:
            return path, []
        if not isinstance(node, zarr.AsyncGroup):
            return path, []
        try:
            names = {k async for k in node.array_keys()}
            names |= {k async for k in node.group_keys()}
        except Exception:
            return None
        return path, sorted(names)

    if not paths:
        return {}
    pairs = await asyncio.gather(*(one(p) for p in sorted(paths)))
    return {p: v for pv in pairs if pv is not None for p, v in [pv]}


async def read_async(
    reader: Callable[..., T],
    source: Any,
    /,
    **kwargs: Any,
) -> T:
    """Run a synchronous ``read_*`` with its I/O performed asynchronously.

    Works with any reader in this package that takes the store as its
    first positional argument — :func:`read_points`,
    :func:`read_polylines`, :func:`read_mesh`, and the rest — because it
    learns what to fetch from the reader itself rather than modelling
    each one.

    Args:
        reader: The synchronous read function, e.g. ``read_points``.
        source: Anything :func:`open_store_async` accepts: a URL, a path,
            a pre-built Store, or an already-open Group.
        **kwargs: Forwarded to ``reader`` unchanged.

    Returns:
        Exactly what ``reader`` returns.

    Raises:
        StoreError: If the snapshot stops growing while the reader still
            wants data it cannot serve — a genuine missing object, or a
            read path this module cannot supply offline (one that needs
            to *list* the store, which a fetch-backed browser Store
            cannot do anyway).

    Example::

        root = await open_store_async(my_fetch_store)
        out = await read_async(read_points, root, bbox=bbox)
    """
    root = await open_store_async(source) if not isinstance(source, Group) else source

    def decode(group: Group) -> T:
        return reader(group, **kwargs)

    return await aexecute(
        root,
        decode,
        fetcher=AsyncFetcher(root),
        label=getattr(reader, "__name__", repr(reader)),
        what="Async read",
        max_rounds=_MAX_ROUNDS,
    )
