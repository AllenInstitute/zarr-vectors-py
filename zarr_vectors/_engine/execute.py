"""Run a synchronous decoder with its I/O supplied up front.

One loop serves three read modes that were three separate code paths:

===================================  ==========================================
``strict=True``  + :class:`GroupFetcher`   eager, offline-correct sync read
``strict=False`` + :class:`GroupFetcher`   today's ``batched_reads`` semantics
``strict=True``  + :class:`AsyncFetcher`   today's ``read_async``
===================================  ==========================================

The decoder is an ordinary synchronous function.  It is never rewritten,
mirrored, or made async — it is *replayed* against a snapshot, and when
it asks for something the snapshot lacks it records a miss and we go
round again.  Letting the decoder say what it wants, rather than
modelling each decoder here, is what makes this work for every reader
present and future without per-type knowledge.  That design is not new:
it is :func:`zarr_vectors.core.aio.read_async`, generalised over the
fetcher and over strictness.

**On strictness.**  ``strict`` selects the *context manager*, not merely
a policy, and the two differ in kind:

* ``offline_reads`` refuses to fall through to the store.  A gap is a
  loud error, which is what makes the loop able to discover the next
  round's work — and, under Pyodide, avoids a blocking ``sync()`` that
  would deadlock the JS event loop.
* ``batched_reads`` serves what it prefetched and quietly falls through
  to a synchronous read for anything else.  Under-specifying the plan
  costs round-trips but never correctness, so a single pass always
  completes.

Those two cannot be combined — ``offline_reads`` raises if a prefetch
cache is active — which is why this is a mode rather than a flag on one
path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from zarr_vectors._engine.fetch import Fetcher
from zarr_vectors._engine.plan import ReadPlan
from zarr_vectors._engine.snapshot import Snapshot
from zarr_vectors.core.group import Group
from zarr_vectors.exceptions import StoreError

T = TypeVar("T")

Decoder = Callable[[Group], T]

# Safety net for the discovery loop.  Each round must strictly grow the
# snapshot (see the stall check below), so termination does not rest on
# this -- it turns a hypothetical non-converging decoder into a
# diagnosable error instead of a hang.
#
# 100 rather than a dozen because read_polylines reassembles tracts
# across chunks by following links, and on a whole-brain store a long
# tract's chain of chunks is discovered a few per round.
MAX_ROUNDS = 100

_INCOMPLETE = object()


def execute(
    group: Group,
    decode: Decoder[T],
    *,
    fetcher: Fetcher,
    plan: ReadPlan | None = None,
    strict: bool = True,
    max_rounds: int = MAX_ROUNDS,
    label: str | None = None,
    what: str = "Read",
) -> T:
    """Run ``decode(group)`` against I/O fetched in batches.

    Args:
        group: The handle ``decode`` will read through.  For
            ``strict=False`` this must be the same handle, because a
            prefetch cache lives on one instance and is keyed relative to
            it.
        decode: An ordinary synchronous reader taking the group.
        fetcher: Where bytes come from.  Its
            :class:`~zarr_vectors._engine.fetch.Capabilities` are the
            caller's to consult when building ``plan``.
        plan: What we already know the read wants.  May be empty — the
            decoder will discover the rest — but a good plan is the
            difference between one round-trip and several.
        strict: See the module docstring.
        label: Name used in error messages; defaults to ``decode``'s.
        what: Leading word in error messages, e.g. ``"Async read"``.

    Returns:
        Whatever ``decode`` returns.

    Raises:
        StoreError: If the snapshot stops growing while the decoder still
            wants data — a genuinely missing object, or a read path that
            needs to *list* the store when the snapshot cannot supply it.
    """
    name = label or str(getattr(decode, "__name__", None) or repr(decode))
    snapshot = Snapshot.empty()
    want = plan or ReadPlan()

    for _round in range(max_rounds):
        outstanding = want.minus(snapshot)
        if outstanding:
            snapshot.absorb(fetcher.fetch(outstanding))
            snapshot.mark_absent(outstanding)

        if not strict:
            # One pass is always enough: anything the prefetch missed is
            # read synchronously on demand.
            with group.batched_reads(snapshot.as_prefetch_plan(group)):
                return decode(group)

        result, failure = _replay(group, decode, snapshot)
        if not snapshot.misses:
            return result  # type: ignore[return-value]

        want = _next_round(want, snapshot, name=name, what=what, failure=failure)

    raise StoreError(
        f"{what} of {name} did not converge within {max_rounds} rounds."
    )


async def aexecute(
    group: Group,
    decode: Decoder[T],
    *,
    fetcher: Any,
    plan: ReadPlan | None = None,
    max_rounds: int = MAX_ROUNDS,
    label: str | None = None,
    what: str = "Async read",
) -> T:
    """:func:`execute` for hosts that must never block.

    Always strict: the whole point is that no read touches the store
    during the decode, so there is nothing to fall through *to*.
    ``fetcher`` must expose ``afetch``.
    """
    name = label or str(getattr(decode, "__name__", None) or repr(decode))
    snapshot = Snapshot.empty()
    want = plan or ReadPlan()

    for _round in range(max_rounds):
        outstanding = want.minus(snapshot)
        if outstanding:
            snapshot.absorb(await fetcher.afetch(outstanding))
            snapshot.mark_absent(outstanding)

        result, failure = _replay(group, decode, snapshot)
        if not snapshot.misses:
            return result  # type: ignore[return-value]

        want = _next_round(want, snapshot, name=name, what=what, failure=failure)

    raise StoreError(
        f"{what} of {name} did not converge within {max_rounds} rounds."
    )


def _replay(
    group: Group, decode: Decoder[T], snapshot: Snapshot,
) -> tuple[T | Any, Exception | None]:
    """One decode pass against the snapshot, misses recorded.

    An exception here is usually not failure but information: the miss
    that caused it was recorded *before* the raise, so an incomplete
    snapshot is expected to blow up somewhere.  Only a raise with no
    recorded miss is genuine and propagates immediately.

    **Any** exception, not just ``StoreError``.  A reader is free to turn
    a missing node into something else on the way out, and one does:
    ``read_lines`` does ``meta["num_objects"]`` on the dict that
    ``read_array_meta`` returns empty for an unresolved node, so an
    incomplete snapshot surfaces there as a ``KeyError``.  Catching only
    ``StoreError`` let that escape the loop as a hard failure when one
    more fetch round would have fixed it.

    The exception is returned rather than swallowed so that it can be
    chained onto a later stall.  The stall message explains the
    *situation* and is the more useful headline; the raise that provoked
    it is the more useful detail, and chaining keeps both.
    """
    snapshot.misses.clear()
    try:
        with group.offline_reads(snapshot):
            return decode(group), None
    except Exception as exc:
        if not snapshot.misses:
            raise
        return _INCOMPLETE, exc


def _next_round(
    want: ReadPlan, snapshot: Snapshot, *, name: str, what: str,
    failure: Exception | None = None,
) -> ReadPlan:
    """Fold what the decoder asked for into the plan, or diagnose a stall.

    A miss that the snapshot *already* holds cannot be satisfied by
    fetching it again, so a round in which every miss is already covered
    taught us nothing and never will.  That is the termination condition;
    ``max_rounds`` is only a backstop.
    """
    discovered = ReadPlan.from_misses(snapshot.misses)
    if not discovered.minus(snapshot):

        # Sort by repr: misses mix bare strings with tuples, and ordering
        # those against each other is a TypeError.
        sample = sorted(snapshot.misses, key=repr)[:5]
        raise StoreError(
            f"{what} stalled: {name} still needs "
            f"{sample!r} but re-fetching yields "
            f"nothing new. Either the object is genuinely missing, or "
            f"this read path requires listing the store, which the "
            f"offline snapshot cannot supply."
        ) from failure
    return want.merge(discovered)
